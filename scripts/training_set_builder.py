"""
训练集构建器 - 符合技术文档规范
目标：构建用于训练模型的多交易日训练集
预测目标：预测当前 tick 之后第 11 个 tick 是否涨停
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple
import multiprocessing as mp  # 添加多进程支持

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.extract_7z import SevenZipExtractor
from scripts.build_dataset import save_dataset, get_feature_columns
from src.feature_engineering.limit_up_features import extract_limit_up_features
from src.feature_engineering.event_driven_labels import get_event_statistics
from src.feature_engineering.event_window_builder import EventWindowBuilder  # 使用V3.2事件窗口构建器
from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized as process_tick_file
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.data_processing.quality_check import run_quality_checks, print_quality_report
from src.data_processing.parallel_processor import (
    process_stocks_parallel,
    _optimize_dtypes,
    benchmark_processing_methods
)
from src.data_processing.sampling import (  # 使用标准采样模块
    undersample_train_set
)


class TrainingSetBuilder:
    """训练集构建器 - 按交易日独立处理、独立欠采样"""

    def __init__(self, archive_path: str, output_base_dir: str):
        """
        初始化训练集构建器

        Args:
            archive_path: 7z压缩文件路径
            output_base_dir: 输出基础目录
        """
        self.archive_path = archive_path
        self.output_base_dir = output_base_dir
        self.date_str = self._extract_date_from_filename()
        self.extractor = None
        self.extract_dir = None

        # 创建输出目录结构
        self.candidates_dir = os.path.join(output_base_dir, 'daily_train_candidates')
        self.train_shards_dir = os.path.join(output_base_dir, 'daily_train_undersampled')
        self.logs_dir = os.path.join(output_base_dir, 'logs')

        os.makedirs(self.candidates_dir, exist_ok=True)
        os.makedirs(self.train_shards_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # 使用V3.2首次触板事件窗口构建器（一板一样本 + 内存阻断 + 按需平铺 + 多线程）
        self.window_builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.05)  # 使用更合理的5%阈值

    def _extract_date_from_filename(self) -> str:
        """从文件名提取日期"""
        filename = os.path.basename(self.archive_path)
        return filename.replace('.7z', '')

    def step1_extract(self) -> str:
        """
        步骤1: 解压7z文件到临时目录

        Returns:
            解压目录路径
        """
        print(f"步骤1/3: 解压 {self.date_str}")
        self.extractor = SevenZipExtractor(self.archive_path)
        self.extract_dir = os.path.join(self.output_base_dir, 'temp_extract', self.date_str)
        os.makedirs(self.extract_dir, exist_ok=True)

        extract_result = self.extractor.extract_all(self.extract_dir)
        print(f"  解压完成: {extract_result}")
        return extract_result

    def step2_process_and_undersample(self, sampling_config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        步骤2: 处理数据并执行欠采样

        Args:
            sampling_config: 欠采样配置参数

        Returns:
            (候选训练集, 欠采样训练集, 统计日志)
        """
        print(f"步骤2/3: 处理数据并执行欠采样")

        try:
            # 处理解压后的数据
            candidate_df = self._process_extracted_data()

            if candidate_df.empty:
                print(f"  警告: {self.date_str} 没有生成有效数据")
                return pd.DataFrame(), pd.DataFrame(), {}

            # 保存候选训练集
            candidate_path = os.path.join(self.candidates_dir, f"{self.date_str}_candidate.csv")
            candidate_df.to_csv(candidate_path, index=False)
            print(f"  候选训练集已保存: {candidate_path}")

            # 执行两层欠采样（内存优化版）
            undersampled_df, sampling_stats = self._two_layer_undersampling_optimized(
                candidate_df, sampling_config
            )

            # 保存欠采样训练集
            train_path = os.path.join(self.train_shards_dir, f"{self.date_str}_train.csv")
            undersampled_df.to_csv(train_path, index=False)
            print(f"  欠采样训练集已保存: {train_path}")

            # 生成统计日志
            log_stats = self._generate_statistics_log(candidate_df, undersampled_df, sampling_stats)

            return candidate_df, undersampled_df, log_stats

        except Exception as e:
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame(), {}

    def _process_extracted_data(self, use_parallel: bool = True, n_workers: int = None) -> pd.DataFrame:
        """处理解压后的数据 - V3.2一板一样本协议 + 并行优化"""
        csv_files = list(Path(self.extract_dir).rglob("*.csv"))
        print(f"  找到 {len(csv_files)} 个股票文件")

        if len(csv_files) == 0:
            return pd.DataFrame()

        if use_parallel:
            # 使用并行处理（阶段一优化）
            print(f"  使用并行处理模式...")
            stock_data_list, processing_stats = process_stocks_parallel(
                [str(csv_file) for csv_file in csv_files],
                date_str=self.date_str,
                n_workers=n_workers,
                apply_features=True,  # 并行处理中直接应用特征提取
                tick_size=0.01,
                verbose=True
            )
            print(f"  并行处理完成: 成功 {processing_stats['success']}, 失败 {processing_stats['failed']}")

        else:
            # 串行处理模式（原有逻辑，保留作为备选）
            print(f"  使用串行处理模式...")
            all_stocks_data = []
            skipped_stocks = 0

            for csv_file in tqdm(csv_files, desc="  处理股票"):
                try:
                    stock_code = csv_file.stem
                    stock_type = determine_stock_type(stock_code)
                    limit_ratio = get_limit_ratio(stock_type)

                    # 获取基准价格
                    preclose = self._get_benchmark_price(csv_file)
                    if preclose is None or preclose <= 0:
                        skipped_stocks += 1
                        continue

                    # 处理tick文件
                    df = process_tick_file(str(csv_file), preclose, limit_ratio)
                    if df.empty:
                        skipped_stocks += 1
                        continue

                    # 添加股票代码和日期
                    df['code'] = stock_code
                    df['date'] = self.date_str

                    # 优化数据类型
                    df = _optimize_dtypes(df)

                    # 第一步：提取基础特征（15个核心特征）
                    df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=limit_ratio)

                    # 注意：V3.1版本不再提前生成标签，标签在事件窗口构建器中生成
                    # 这样可以在特征平铺前进行内存阻断

                    all_stocks_data.append(df)

                except Exception as e:
                    skipped_stocks += 1
                    continue

            print(f"  串行处理完成: 成功 {len(all_stocks_data)}, 跳过 {skipped_stocks}")
            stock_data_list = all_stocks_data

        print(f"  基础处理完成: 成功 {len(stock_data_list)}")

        if not stock_data_list:
            return pd.DataFrame()

        # 第二步：合并所有股票数据
        merged_df = pd.concat(stock_data_list, ignore_index=True)
        print(f"  原始Tick总数: {len(merged_df):,}")

        # 第三步：事件驱动协议 - 构建窗口样本（V3.2版本：一板一样本 + 内存阻断 + 按需平铺）
        print(f"  构建事件窗口样本（V3.2版本：一板一样本 + 内存阻断）...")

        # 显示内存使用（简化版）
        merged_memory_mb = merged_df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"  合并数据内存: {merged_memory_mb:.1f} MB")

        # 使用V3.2事件窗口构建器（集成首次触板标签 + 前置内存阻断 + 按需特征平铺 + 多线程）
        # 使用多线程加速窗口构建
        n_threads = min(24, mp.cpu_count())  # 最多使用24个线程
        window_df = self.window_builder.build_multi_stock_event_samples(
            merged_df,
            verbose=True,  # 启用详细日志
            n_workers=n_threads  # 指定线程数
        )

        if window_df.empty:
            print(f"  警告: 事件窗口样本构建后无数据")
            return pd.DataFrame()

        # ==========================================
        # 优化验尸代码：区分特征类型的智能填充
        # ==========================================
        print(f"\n执行智能数据质量验尸检查...")

        # 优先处理Inf值（避免影响NaN检测）
        for col in window_df.columns:
            if col in ['code', 'time', 'label']:
                continue

            # 只处理数值类型的列
            if not pd.api.types.is_numeric_dtype(window_df[col]):
                continue

            # 检测Inf值
            try:
                has_inf = np.isinf(window_df[col]).any()

                if has_inf:
                    # 获取非Inf的统计信息
                    finite_values = window_df[col][np.isfinite(window_df[col])]

                    if len(finite_values) > 0:
                        # 计算99分位数作为极大值基准
                        max_value = np.percentile(finite_values, 99)
                        extreme_value = max_value * 2.0  # 设定为2倍的99分位数

                        # 用极大值替换Inf
                        window_df[col] = window_df[col].replace([np.inf, -np.inf], extreme_value)

                        print(f"[WARNING] {col} 包含Inf值，已替换为极大值 {extreme_value:.4f}")
                    else:
                        # 没有有效值，用极大常数替换
                        window_df[col] = window_df[col].replace([np.inf, -np.inf], 99999.0)
            except (TypeError, ValueError):
                # 如果检测失败，跳过此列
                continue

        # 检查空值情况
        nan_counts = window_df.isnull().sum()
        bad_cols = nan_counts[nan_counts > 0]

        if not bad_cols.empty:
            print(f"[ALERT] 发现空值列 (当前总行数: {len(window_df)}):")
            bad_cols_sorted = bad_cols.sort_values(ascending=False)
            for col, count in bad_cols_sorted.items():
                pct = count / len(window_df) * 100
                print(f"   {col}: {count} 个NaN ({pct:.1f}%)")

            print(f"\n[SMART_FILL] 根据特征类型进行差异化填充...")

            # 定义价格类/指数类特征（绝对不能补0）
            price_like_patterns = ['price', 'limit_price', 'open', 'current', 'high', 'low',
                                 'spread', 'gap', 'ratio', 'slope', 'imbalance', 'depth', 'order']

            # 遍历每一列进行差异化填充
            for col in window_df.columns:
                if col in bad_cols.index and col not in ['code', 'time', 'label']:
                    col_lower = col.lower()
                    is_price_like = any(pattern in col_lower for pattern in price_like_patterns)

                    if is_price_like:
                        # 价格类特征：用列的非空均值填充（避免补0造成价格突变）
                        if window_df[col].notna().any():
                            fill_value = window_df[col].mean()
                            window_df[col].fillna(fill_value, inplace=True)
                        else:
                            # 如果整列都是空，用0填充（降级处理）
                            window_df[col].fillna(0, inplace=True)
                    else:
                        # 成交量/变化率特征：用 0 填充是正确的
                        window_df[col].fillna(0, inplace=True)

            print(f"   填充完成，数据行数: {len(window_df):,}")
        else:
            pass  # 未发现空值列，不输出调试信息
        # ==========================================

        # 显示窗口样本内存使用
        window_memory_mb = window_df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"  窗口样本内存: {window_memory_mb:.1f} MB")
        print(f"  内存节省: {(1 - window_memory_mb / merged_memory_mb) * 100:.1f}%")

        # V3.2版本：一板一样本模式
        print(f"  V3.2一板一样本模式：从{len(merged_df):,}Tick生成{len(window_df):,}事件样本")

        return window_df

    def _get_benchmark_price(self, csv_file: Path) -> float:
        """获取基准价格"""
        try:
            df_raw = pd.read_csv(csv_file, nrows=100)

            # 优先级：买一价 > 卖一价 > 开盘价 > 当前价
            for price_col in ['b1_p', 'a1_p', 'open', 'current']:
                if price_col in df_raw.columns:
                    valid_prices = df_raw[df_raw[price_col] > 0][price_col]
                    if len(valid_prices) > 0:
                        return valid_prices.iloc[0]
        except Exception:
            pass
        return None

    def _two_layer_undersampling(self, df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        执行两层欠采样策略

        Args:
            df: 候选训练集
            config: 欠采样配置

        Returns:
            (欠采样后数据集, 采样统计信息)
        """
        print(f"\n执行两层欠采样...")

        # 分离正负样本
        positive_df = df[df['label'] == 1].copy()
        negative_df = df[df['label'] == 0].copy()

        original_stats = {
            'positive_samples': len(positive_df),
            'negative_samples': len(negative_df),
            'total_samples': len(df),
            'positive_ratio': len(positive_df) / len(df) if len(df) > 0 else 0
        }

        print(f"  原始数据: 正样本 {len(positive_df):,}, 负样本 {len(negative_df):,}")

        # 第一层：分层筛选
        thresholds = config.get('layer1_thresholds', (0.01, 0.05))
        keep_ratios = config.get('layer1_keep_ratios', (1.0, 0.3, 0.05))

        # 使用窗口末端特征进行采样
        negative_filtered = self._layer1_stratified_sampling(negative_df, thresholds, keep_ratios)

        print(f"  第一层筛选: 负样本 {len(negative_df):,} → {len(negative_filtered):,}")

        # 第二层：比例控制
        target_ratio = config.get('target_ratio', 5.0)  # 1:5 正负比例
        negative_final = self._layer2_ratio_control(positive_df, negative_filtered, target_ratio)

        print(f"  第二层采样: 负样本 {len(negative_filtered):,} → {len(negative_final):,}")

        # 合并正负样本
        sampled_df = pd.concat([positive_df, negative_final], ignore_index=True)

        # 采样统计
        sampling_stats = {
            'original_positive': original_stats['positive_samples'],
            'original_negative': original_stats['negative_samples'],
            'layer1_filtered': len(negative_filtered),
            'layer2_final': len(negative_final),
            'final_positive': len(positive_df),
            'final_negative': len(negative_final),
            'final_ratio': len(negative_final) / len(positive_df) if len(positive_df) > 0 else float('inf')
        }

        print(f"  最终结果: 正样本 {len(positive_df):,}, 负样本 {len(negative_final):,}, 比例 1:{sampling_stats['final_ratio']:.2f}")

        return sampled_df, sampling_stats

    def _two_layer_undersampling_optimized(self, df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        执行两层欠采样策略（V3.2一板一样本版本）

        Args:
            df: 候选训练集（事件样本）
            config: 欠采样配置参数

        Returns:
            (欠采样后数据集, 采样统计信息)
        """
        print(f"\n执行两层欠采样（V3.2一板一样本版本）...")

        # 执行欠采样（使用标准版本）
        undersampled_df = undersample_train_set(
            df,
            dist_col='dist_to_limit',  # 使用基础特征
            label_col='label',
            thresholds=config.get('layer1_thresholds', (0.01, 0.05)),
            keep_ratios=config.get('layer1_keep_ratios', (1.0, 0.3, 0.05)),
            target_ratio=config.get('target_ratio', 5.0),
            random_seed=42,
            verbose=True
        )

        # 估算内存使用（简化版）
        sampled_memory_mb = len(undersampled_df) * 165 * 4 / 1024 / 1024  # 估算
        print(f"  采样后内存: {sampled_memory_mb:.1f} MB")

        # 计算采样统计（V3.2一板一样本版本）
        positive_df = df[df['label'] == 1]
        negative_df = df[df['label'] == 0]

        sampling_stats = {
            'original_positive': len(positive_df),
            'original_negative': len(negative_df),
            'layer1_filtered': len(undersampled_df[undersampled_df['label'] == 0]),
            'layer2_final': len(undersampled_df[undersampled_df['label'] == 0]),
            'final_positive': len(undersampled_df[undersampled_df['label'] == 1]),
            'final_negative': len(undersampled_df[undersampled_df['label'] == 0]),
            'final_ratio': len(undersampled_df[undersampled_df['label'] == 0]) / len(undersampled_df[undersampled_df['label'] == 1]) if len(undersampled_df[undersampled_df['label'] == 1]) > 0 else float('inf')
        }

        return undersampled_df, sampling_stats

    def _layer1_stratified_sampling(self, negative_df: pd.DataFrame,
                                    thresholds: Tuple[float, float],
                                    keep_ratios: Tuple[float, float, float]) -> pd.DataFrame:
        """
        第一层：基于窗口末端 dist_to_limit_last 的分层筛选

        Args:
            negative_df: 负样本数据（窗口样本）
            thresholds: 分层阈值 (t1, t2)
            keep_ratios: 各层保留率 (r1, r2, r3)

        Returns:
            筛选后的负样本
        """
        t1, t2 = thresholds
        r1, r2, r3 = keep_ratios

        # 分层（使用窗口末端特征 dist_to_limit_last）
        close_to_limit = negative_df[negative_df['dist_to_limit_last'] <= t1]
        medium_distance = negative_df[(negative_df['dist_to_limit_last'] > t1) & (negative_df['dist_to_limit_last'] <= t2)]
        far_from_limit = negative_df[negative_df['dist_to_limit_last'] > t2]

        # 采样
        sampled_close = close_to_limit.sample(frac=r1, random_state=42) if r1 < 1.0 and len(close_to_limit) > 0 else close_to_limit
        sampled_medium = medium_distance.sample(frac=r2, random_state=42) if r2 < 1.0 and len(medium_distance) > 0 else medium_distance
        sampled_far = far_from_limit.sample(frac=r3, random_state=42) if r3 < 1.0 and len(far_from_limit) > 0 else far_from_limit

        # 合并
        result = pd.concat([sampled_close, sampled_medium, sampled_far], ignore_index=True)

        print(f"    第一层分层: 接近涨停 {len(close_to_limit):,}→{len(sampled_close):,}, "
              f"中等距离 {len(medium_distance):,}→{len(sampled_medium):,}, "
              f"远离涨停 {len(far_from_limit):,}→{len(sampled_far):,}")

        return result

    def _layer2_ratio_control(self, positive_df: pd.DataFrame, negative_df: pd.DataFrame,
                            target_ratio: float) -> pd.DataFrame:
        """
        第二层：比例控制

        Args:
            positive_df: 正样本
            negative_df: 负样本（第一层筛选后）
            target_ratio: 目标正负比例（负/正）

        Returns:
            采样后的负样本
        """
        target_negative_count = int(len(positive_df) * target_ratio)

        if len(negative_df) > target_negative_count:
            # 负样本过多，随机抽样
            result = negative_df.sample(n=target_negative_count, random_state=42)
            print(f"    第二层采样: 从 {len(negative_df):,} 中抽取 {target_negative_count:,}")
        else:
            # 负样本不足，全部保留
            result = negative_df
            print(f"    第二层采样: 负样本不足，全部保留 {len(negative_df):,}")

        return result

    def _generate_statistics_log(self, candidate_df: pd.DataFrame,
                                undersampled_df: pd.DataFrame,
                                sampling_stats: Dict) -> Dict:
        """
        生成统计日志（V3.2一板一样本版本）

        Args:
            candidate_df: 候选训练集（事件样本）
            undersampled_df: 欠采样训练集（事件样本）
            sampling_stats: 采样统计信息

        Returns:
            日志字典
        """
        # 使用事件驱动统计信息
        event_stats = get_event_statistics(candidate_df)

        log_data = {
            'date': self.date_str,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': 'V3.2_一板一样本',
            'original_tick_count': event_stats['total_samples'],  # 原始tick数（估算）
            'limit_up_events': event_stats['limit_up_events'],  # 首次触板数量
            'limit_up_state': event_stats['limit_up_state'],    # 处于涨停状态的tick数
            'event_samples': len(candidate_df),  # 事件样本数
            'final_samples': len(undersampled_df),  # 最终事件样本数
            'positive_samples': sampling_stats['final_positive'],
            'negative_samples': sampling_stats['final_negative'],
            'final_ratio': sampling_stats['final_ratio'],
            'event_to_state_ratio': event_stats['event_to_state_ratio'],  # 事件/状态比例
            'reduction_rate': (1 - len(undersampled_df) / event_stats['total_samples']) * 100 if event_stats['total_samples'] > 0 else 0
        }

        # 保存日志文件
        log_path = os.path.join(self.logs_dir, f"{self.date_str}_summary.csv")
        log_df = pd.DataFrame([log_data])
        log_df.to_csv(log_path, index=False)

        print(f"\n  日志已保存: {log_path}")

        # 打印摘要（V3.2一板一样本格式）
        print(f"\n  📊 {self.date_str} 事件样本处理摘要 (V3.2):")
        print(f"    版本模式: 一板一样本（首次触板）")
        print(f"    触板事件数: {log_data['limit_up_events']:,} (真实突破瞬间)")
        print(f"    涨停状态数: {log_data['limit_up_state']:,} (处于涨停状态)")
        print(f"    事件/状态比例: {log_data['event_to_state_ratio']:.2%} (应该<10%)")
        print(f"    事件样本数: {log_data['event_samples']:,}")
        print(f"    最终样本数: {log_data['final_samples']:,}")
        print(f"    正样本: {log_data['positive_samples']:,}, 负样本: {log_data['negative_samples']:,}")
        print(f"    最终比例: 1:{log_data['final_ratio']:.2f}")
        print(f"    总体减少率: {log_data['reduction_rate']:.1f}%")

        return log_data

    def step3_cleanup(self):
        """步骤3: 清理临时目录"""
        if self.extract_dir and os.path.exists(self.extract_dir):
            shutil.rmtree(self.extract_dir, ignore_errors=True)
            print(f"步骤3/3: 临时目录已清理: {self.extract_dir}")

    def run_quality_checks(self, candidate_df: pd.DataFrame, undersampled_df: pd.DataFrame) -> bool:
        """
        执行数据质量检查（V3.1事件样本版本）

        Args:
            candidate_df: 候选训练集（事件样本）
            undersampled_df: 欠采样训练集（事件样本）

        Returns:
            是否通过质量检查
        """
        print(f"\n执行数据质量检查（V3.1事件样本）...")

        # V3.1事件样本的特征名称
        required_features = ['dist_to_limit_last', 'label', 'code', 'time']

        # 检查候选集
        print(f"\n  候选训练集质量检查:")
        if not all(col in candidate_df.columns for col in required_features):
            print(f"  ⚠️ 警告: 缺少必需特征列")
            print(f"  需要的特征: {required_features}")
            print(f"  实际的特征: {[col for col in required_features if col in candidate_df.columns]}")

        # 检查欠采样集
        print(f"\n  欠采样训练集质量检查:")
        if not all(col in undersampled_df.columns for col in required_features):
            print(f"  ⚠️ 警告: 缺少必需特征列")

        # 唯一性检查（基于事件样本标识）
        duplicates = candidate_df.duplicated(["date", "code", "time"]).sum()
        if duplicates > 0:
            print(f"  ⚠️ 警告: 发现 {duplicates} 个重复事件样本")
            return False

        # 空值检查
        required_cols = ['date', 'code', 'label', 'dist_to_limit_last']
        missing_values = candidate_df[required_cols].isnull().sum()
        if missing_values.sum() > 0:
            print(f"  ⚠️ 警告: 发现空值")
            print(missing_values)
            return False

        # 正样本检查（V3.1模式下正样本应该很少）
        positive_count = candidate_df['label'].sum()
        if positive_count == 0:
            print(f"  ⚠️ 警告: 没有正样本（触板事件）")
            return False
        elif positive_count > 100:
            print(f"  ⚠️ 警告: 正样本数量较多 ({positive_count})，建议检查标签逻辑")
        else:
            print(f"  ✅ 正样本数量合理: {positive_count}")

        # 事件状态比例检查
        if 'current' in candidate_df.columns and 'limit_price' in candidate_df.columns:
            limit_up_events = positive_count
            limit_up_state = (candidate_df['current'] >= candidate_df['limit_price']).sum()
            if limit_up_state > 0:
                event_to_state_ratio = limit_up_events / limit_up_state
                if event_to_state_ratio < 0.1:  # 应该小于10%
                    print(f"  ✅ 事件/状态比例合理: {event_to_state_ratio:.2%}")
                else:
                    print(f"  ⚠️ 警告: 事件/状态比例异常: {event_to_state_ratio:.2%}")

        print(f"  ✅ 质量检查通过")
        return True


def build_single_day_training_set(archive_path: str, output_dir: str,
                                 sampling_config: Dict = None) -> bool:
    """
    构建单日训练集

    Args:
        archive_path: 7z文件路径
        output_dir: 输出目录
        sampling_config: 欠采样配置

    Returns:
        是否成功
    """
    if sampling_config is None:
        sampling_config = {
            'layer1_thresholds': (0.01, 0.05),
            'layer1_keep_ratios': (1.0, 0.3, 0.05),
            'target_ratio': 5.0
        }

    builder = TrainingSetBuilder(archive_path, output_dir)

    try:
        # 步骤1: 解压
        builder.step1_extract()

        # 步骤2: 处理和欠采样
        candidate_df, undersampled_df, log_stats = builder.step2_process_and_undersample(sampling_config)

        if candidate_df.empty:
            print(f"❌ {builder.date_str} 处理失败: 没有生成有效数据")
            return False

        # 质量检查
        quality_passed = builder.run_quality_checks(candidate_df, undersampled_df)

        # 步骤3: 清理
        builder.step3_cleanup()

        print(f"\n✅ {builder.date_str} 处理完成")
        return quality_passed

    except Exception as e:
        print(f"❌ 处理 {archive_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_archive_files(input_dir: str, pattern: str = "*.7z") -> List[Path]:
    """
    查找目录中的压缩文件

    Args:
        input_dir: 输入目录
        pattern: 文件模式，默认 *.7z

    Returns:
        压缩文件路径列表
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"目录不存在: {input_dir}")

    archive_files = sorted(input_path.rglob(pattern))
    return archive_files


def build_batch_training_sets(input_dir: str, output_dir: str,
                            sampling_config: Dict = None,
                            max_files: int = None,
                            sample_interval: int = None) -> Dict:
    """
    批量构建多日训练集

    Args:
        input_dir: 输入目录（包含7z文件）
        output_dir: 输出目录
        sampling_config: 欠采样配置
        max_files: 最大处理文件数（用于测试）
        sample_interval: 采样间隔，如3表示每3个文件处理1个（用于测试）

    Returns:
        批量处理统计信息
    """
    if sampling_config is None:
        sampling_config = {
            'layer1_thresholds': (0.01, 0.05),
            'layer1_keep_ratios': (1.0, 0.3, 0.05),
            'target_ratio': 5.0
        }

    print("="*80)
    print("批量构建多日训练集")
    print("="*80)

    # 查找所有压缩文件
    archive_files = find_archive_files(input_dir)
    print(f"找到 {len(archive_files)} 个压缩文件")

    if len(archive_files) == 0:
        print("❌ 未找到压缩文件")
        return {'success': False, 'error': '未找到压缩文件'}

    # 采样或限制文件数
    if sample_interval:
        archive_files = archive_files[::sample_interval]
        print(f"采样模式: 每{sample_interval}个文件处理1个，剩余{len(archive_files)}个文件")

    if max_files and len(archive_files) > max_files:
        archive_files = archive_files[:max_files]
        print(f"限制模式: 最多处理{max_files}个文件")

    # 统计信息
    stats = {
        'total_files': len(archive_files),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'total_samples': 0,
        'total_positive': 0,
        'total_negative': 0,
        'failed_files': [],
        'successful_dates': []
    }

    # 批量处理
    for i, archive_file in enumerate(tqdm(archive_files, desc="处理交易日")):
        print(f"\n进度: {i+1}/{len(archive_files)}")
        print(f"文件: {archive_file.name}")

        try:
            success = build_single_day_training_set(
                str(archive_file),
                output_dir,
                sampling_config
            )

            if success:
                stats['successful'] += 1
                # 提取日期信息
                date_str = archive_file.stem.replace('.7z', '')
                stats['successful_dates'].append(date_str)
            else:
                stats['failed'] += 1
                stats['failed_files'].append(str(archive_file))

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            stats['failed'] += 1
            stats['failed_files'].append(str(archive_file))
            import traceback
            traceback.print_exc()
            continue

    # 汇总统计
    print("\n" + "="*80)
    print("批量处理完成")
    print("="*80)
    print(f"总文件数: {stats['total_files']}")
    print(f"成功: {stats['successful']}")
    print(f"失败: {stats['failed']}")
    print(f"跳过: {stats['skipped']}")

    if stats['failed_files']:
        print(f"\n失败的文件:")
        for failed_file in stats['failed_files']:
            print(f"  - {failed_file}")

    # 读取所有成功的训练集分片进行汇总
    if stats['successful'] > 0:
        print(f"\n汇总统计:")
        shards_dir = os.path.join(output_dir, 'daily_train_undersampled')
        shard_files = sorted(Path(shards_dir).glob("*_train.csv"))

        total_samples = 0
        total_positive = 0
        total_negative = 0

        for shard_file in shard_files:
            if shard_file.stem.replace('_train', '') in stats['successful_dates']:
                try:
                    shard_df = pd.read_csv(shard_file)
                    total_samples += len(shard_df)
                    total_positive += shard_df['label'].sum()
                    total_negative += len(shard_df) - shard_df['label'].sum()
                except Exception as e:
                    print(f"  警告: 无法读取 {shard_file.name}: {e}")

        stats['total_samples'] = total_samples
        stats['total_positive'] = total_positive
        stats['total_negative'] = total_negative

        print(f"总样本数: {total_samples:,}")
        print(f"正样本: {total_positive:,}")
        print(f"负样本: {total_negative:,}")
        if total_positive > 0:
            print(f"总体比例: 1:{total_negative/total_positive:.2f}")

    stats['success'] = stats['successful'] > 0

    return stats


def merge_training_shards(shards_dir: str, output_path: str) -> bool:
    """
    合并多个训练集分片

    Args:
        shards_dir: 训练集分片目录
        output_path: 合并后的输出路径

    Returns:
        是否成功
    """
    print("="*80)
    print("合并训练集分片")
    print("="*80)

    # 查找所有训练集分片
    shard_files = sorted(Path(shards_dir).glob("*_train.csv"))

    if not shard_files:
        print("❌ 未找到训练集分片")
        return False

    print(f"找到 {len(shard_files)} 个训练集分片")

    # 读取并合并
    all_data = []
    for shard_file in tqdm(shard_files, desc="读取训练集分片"):
        shard_df = pd.read_csv(shard_file)
        all_data.append(shard_df)
        print(f"  {shard_file.name}: {len(shard_df):,} 样本")

    # 合并
    print("\n合并训练集...")
    merged_df = pd.concat(all_data, ignore_index=True)

    # 全局打乱
    print(f"全局打乱数据...")
    merged_df = merged_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 保存
    merged_df.to_csv(output_path, index=False)

    print(f"合并完成: {output_path}")
    print(f"总样本数: {len(merged_df):,}")
    print(f"交易日数: {merged_df['date'].nunique()}")
    print(f"股票数: {merged_df['code'].nunique()}")

    # 标签统计
    stats = get_label_statistics(merged_df)
    print(f"正样本: {stats['positive_samples']:,} ({stats['positive_ratio']:.4%})")
    print(f"负样本: {stats['negative_samples']:,} ({stats['negative_ratio']:.4%})")
    print(f"正负比例: 1:{stats['imbalance_ratio']:.2f}")

    return True


def extract_window_features_from_candidates(candidate_path: str, output_dir: str,
                                           sampling_config: Dict = None) -> bool:
    """
    从候选训练集直接提取窗口特征（跳过基础数据处理）

    Args:
        candidate_path: 候选训练集文件路径
        output_dir: 输出目录
        sampling_config: 欠采样配置

    Returns:
        是否成功
    """
    if sampling_config is None:
        sampling_config = {
            'layer1_thresholds': (0.01, 0.05),
            'layer1_keep_ratios': (1.0, 0.3, 0.05),
            'target_ratio': 5.0
        }

    print("="*80)
    print("从候选训练集提取窗口特征")
    print("="*80)

    # 创建输出目录结构
    train_shards_dir = os.path.join(output_dir, 'daily_train_undersampled')
    logs_dir = os.path.join(output_dir, 'logs')

    os.makedirs(train_shards_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    try:
        # 读取候选训练集
        print(f"读取候选训练集: {candidate_path}")
        candidate_df = pd.read_csv(candidate_path)
        print(f"  候选训练集: {len(candidate_df)} 条记录")

        # 提取窗口特征
        print(f"\n提取窗口特征...")
        window_extractor = OptimizedWindowFeatureExtractor(window_size=10)

        # 按股票分组提取窗口特征
        all_window_samples = []
        for code, stock_df in tqdm(candidate_df.groupby('code'), desc="处理股票窗口特征"):
            try:
                window_features = window_extractor.extract_window_features(stock_df)
                if not window_features.empty:
                    # 添加股票代码和日期
                    window_features['code'] = code
                    if 'date' in stock_df.columns:
                        window_features['date'] = stock_df['date'].iloc[0]

                    all_window_samples.append(window_features)
            except Exception as e:
                print(f"股票 {code} 处理失败: {e}")
                continue

        if not all_window_samples:
            print("❌ 没有生成有效的窗口特征")
            return False

        # 合并所有窗口特征
        window_df = pd.concat(all_window_samples, ignore_index=True)
        print(f"  窗口特征提取完成: {len(window_df)} 个样本")

        # 执行欠采样
        print(f"\n执行欠采样...")
        undersampled_df, sampling_stats = _perform_two_layer_undersampling(
            window_df, sampling_config
        )

        # 保存结果
        from pathlib import Path
        date_str = Path(candidate_path).stem.replace('_candidate', '')
        train_path = os.path.join(train_shards_dir, f"{date_str}_train.csv")
        undersampled_df.to_csv(train_path, index=False)
        print(f"  欠采样训练集已保存: {train_path}")

        # 生成统计日志
        log_stats = _generate_statistics_log(window_df, undersampled_df, sampling_stats, date_str)
        log_path = os.path.join(logs_dir, f"{date_str}_summary.csv")
        pd.DataFrame([log_stats]).to_csv(log_path, index=False)
        print(f"  日志已保存: {log_path}")

        print(f"\n✅ 窗口特征提取完成")
        return True

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def _perform_two_layer_undersampling(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """执行两层欠采样（内部函数）"""
    # 分离正负样本
    positive_df = df[df['label'] == 1].copy()
    negative_df = df[df['label'] == 0].copy()

    original_stats = {
        'positive_samples': len(positive_df),
        'negative_samples': len(negative_df),
        'total_samples': len(df),
        'positive_ratio': len(positive_df) / len(df) if len(df) > 0 else 0
    }

    print(f"  原始数据: 正样本 {len(positive_df):,}, 负样本 {len(negative_df):,}")

    # 第一层：分层筛选
    thresholds = config.get('layer1_thresholds', (0.01, 0.05))
    keep_ratios = config.get('layer1_keep_ratios', (1.0, 0.3, 0.05))

    negative_filtered = _layer1_stratified_sampling(negative_df, thresholds, keep_ratios)
    print(f"  第一层筛选: 负样本 {len(negative_df):,} → {len(negative_filtered):,}")

    # 第二层：比例控制
    target_ratio = config.get('target_ratio', 5.0)
    negative_final = _layer2_ratio_control(positive_df, negative_filtered, target_ratio)
    print(f"  第二层采样: 负样本 {len(negative_filtered):,} → {len(negative_final):,}")

    # 合并正负样本
    sampled_df = pd.concat([positive_df, negative_final], ignore_index=True)

    # 采样统计
    sampling_stats = {
        'original_positive': original_stats['positive_samples'],
        'original_negative': original_stats['negative_samples'],
        'layer1_filtered': len(negative_filtered),
        'layer2_final': len(negative_final),
        'final_positive': len(positive_df),
        'final_negative': len(negative_final),
        'final_ratio': len(negative_final) / len(positive_df) if len(positive_df) > 0 else float('inf')
    }

    print(f"  最终结果: 正样本 {len(positive_df):,}, 负样本 {len(negative_final):,}, 比例 1:{sampling_stats['final_ratio']:.2f}")

    return sampled_df, sampling_stats


def _layer1_stratified_sampling(negative_df: pd.DataFrame,
                                thresholds: Tuple[float, float],
                                keep_ratios: Tuple[float, float, float]) -> pd.DataFrame:
    """第一层分层筛选（内部函数，窗口样本版本）"""
    t1, t2 = thresholds
    r1, r2, r3 = keep_ratios

    # 分层（使用窗口末端特征 dist_to_limit_last）
    close_to_limit = negative_df[negative_df['dist_to_limit_last'] <= t1]
    medium_distance = negative_df[(negative_df['dist_to_limit_last'] > t1) & (negative_df['dist_to_limit_last'] <= t2)]
    far_from_limit = negative_df[negative_df['dist_to_limit_last'] > t2]

    # 采样
    sampled_close = close_to_limit.sample(frac=r1, random_state=42) if r1 < 1.0 and len(close_to_limit) > 0 else close_to_limit
    sampled_medium = medium_distance.sample(frac=r2, random_state=42) if r2 < 1.0 and len(medium_distance) > 0 else medium_distance
    sampled_far = far_from_limit.sample(frac=r3, random_state=42) if r3 < 1.0 and len(far_from_limit) > 0 else far_from_limit

    # 合并
    result = pd.concat([sampled_close, sampled_medium, sampled_far], ignore_index=True)
    return result


def _layer2_ratio_control(positive_df: pd.DataFrame, negative_df: pd.DataFrame,
                        target_ratio: float) -> pd.DataFrame:
    """第二层比例控制（内部函数）"""
    target_negative_count = int(len(positive_df) * target_ratio)

    if len(negative_df) > target_negative_count:
        result = negative_df.sample(n=target_negative_count, random_state=42)
        print(f"    第二层采样: 从 {len(negative_df):,} 中抽取 {target_negative_count:,}")
    else:
        result = negative_df
        print(f"    第二层采样: 负样本不足，全部保留 {len(negative_df):,}")

    return result


def _generate_statistics_log(candidate_df: pd.DataFrame, undersampled_df: pd.DataFrame,
                            sampling_stats: Dict, date_str: str) -> Dict:
    """生成统计日志（内部函数）"""
    log_data = {
        'date': date_str,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_total_samples': len(candidate_df),
        'original_positive': sampling_stats['original_positive'],
        'original_negative': sampling_stats['original_negative'],
        'layer1_filtered': sampling_stats['layer1_filtered'],
        'layer2_final': sampling_stats['layer2_final'],
        'final_total_samples': len(undersampled_df),
        'final_positive': sampling_stats['final_positive'],
        'final_negative': sampling_stats['final_negative'],
        'final_ratio': sampling_stats['final_ratio'],
        'reduction_rate': (1 - len(undersampled_df) / len(candidate_df)) * 100 if len(candidate_df) > 0 else 0
    }

    # 打印摘要
    print(f"\n  📊 {date_str} 处理摘要:")
    print(f"    原始样本: {log_data['original_total_samples']:,} "
          f"(正样本 {log_data['original_positive']:,}, 负样本 {log_data['original_negative']:,})")
    print(f"    最终样本: {log_data['final_total_samples']:,} "
          f"(正样本 {log_data['final_positive']:,}, 负样本 {log_data['final_negative']:,})")
    print(f"    最终比例: 1:{log_data['final_ratio']:.2f}")
    print(f"    减少率: {log_data['reduction_rate']:.1f}%")

    return log_data


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='训练集构建器')
    parser.add_argument('--input', type=str,
                       help='7z文件路径 (单日处理)')
    parser.add_argument('--input-dir', type=str,
                       help='7z文件目录 (批量处理)')
    parser.add_argument('--output', type=str,
                       default="d:/Qoder-project/deep project/data",
                       help='输出目录 (默认: d:/Qoder-project/deep project/data)')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'merge', 'window'],
                       default='single', help='处理模式')

    # 窗口特征提取模式参数
    parser.add_argument('--candidate', type=str,
                       help='候选训练集文件路径 (window模式)')

    # 欠采样配置
    parser.add_argument('--layer1-thresholds', type=float, nargs=2, default=[0.01, 0.05],
                       help='第一层分层阈值')
    parser.add_argument('--layer1-keep-ratios', type=float, nargs=3, default=[1.0, 0.3, 0.05],
                       help='第一层保留率')
    parser.add_argument('--target-ratio', type=float, default=5.0,
                       help='目标正负比例 (默认: 5.0)')

    # 合并配置
    parser.add_argument('--shards-dir', type=str,
                       help='训练集分片目录 (合并模式)')
    parser.add_argument('--merged-output', type=str,
                       default="d:/Qoder-project/deep project/data/merged/multi_day_train.csv",
                       help='合并后输出路径')

    # 批量处理配置
    parser.add_argument('--max-files', type=int,
                       help='最大处理文件数（测试用）')
    parser.add_argument('--sample-interval', type=int,
                       help='采样间隔（测试用）')

    args = parser.parse_args()

    # 构建欠采样配置
    sampling_config = {
        'layer1_thresholds': tuple(args.layer1_thresholds),
        'layer1_keep_ratios': tuple(args.layer1_keep_ratios),
        'target_ratio': args.target_ratio
    }

    if args.mode == 'window':
        # 窗口特征提取模式（从候选训练集提取）
        if not args.candidate:
            print("❌ 窗口模式需要指定 --candidate 参数")
            sys.exit(1)

        success = extract_window_features_from_candidates(
            args.candidate,
            args.output,
            sampling_config
        )

        if success:
            print(f"\n✅ 窗口特征提取完成")
            sys.exit(0)
        else:
            print(f"\n❌ 窗口特征提取失败")
            sys.exit(1)

    elif args.mode == 'single':
        # 单日处理
        if not args.input:
            print("❌ 单日模式需要指定 --input 参数")
            sys.exit(1)

        success = build_single_day_training_set(args.input, args.output, sampling_config)

        if success:
            print(f"\n✅ 单日训练集构建完成")
            sys.exit(0)
        else:
            print(f"\n❌ 单日训练集构建失败")
            sys.exit(1)

    elif args.mode == 'batch':
        # 批量处理
        if not args.input_dir:
            print("❌ 批量模式需要指定 --input-dir 参数")
            sys.exit(1)

        stats = build_batch_training_sets(
            args.input_dir,
            args.output,
            sampling_config,
            max_files=args.max_files,
            sample_interval=args.sample_interval
        )

        if stats['success']:
            print(f"\n✅ 批量训练集构建完成")
            print(f"📁 输出目录: {args.output}")
            print(f"📊 训练集分片: {stats['successful']} 个")
            sys.exit(0)
        else:
            print(f"\n❌ 批量训练集构建失败")
            sys.exit(1)

    elif args.mode == 'merge':
        # 合并训练集
        if not args.shards_dir:
            print("❌ 合并模式需要指定 --shards-dir 参数")
            sys.exit(1)

        success = merge_training_shards(args.shards_dir, args.merged_output)

        if success:
            print(f"\n✅ 训练集合并完成")
            sys.exit(0)
        else:
            print(f"\n❌ 训练集合并失败")
            sys.exit(1)


if __name__ == "__main__":
    main()