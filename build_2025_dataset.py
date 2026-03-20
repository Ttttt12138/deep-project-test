"""
批量构建2025全年数据集
自动处理所有交易日数据，生成完整的数据集
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import Tuple
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.extract_7z import SevenZipExtractor
from scripts.build_dataset import (
    process_single_stock_csv, save_dataset, get_feature_columns,
    get_label_statistics
)
from src.feature_engineering.event_driven_labels import (
    generate_event_driven_label  # V3.2首次触板逻辑
)
from src.feature_engineering.limit_up_features import extract_limit_up_features
from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized as process_tick_file
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.data_processing.quality_check import run_quality_checks, print_quality_report
from src.data_processing.dataset_split import split_dataset_by_trading_day, save_split_datasets
from src.data_processing.sampling import undersample_train_set


def find_2025_data_files(max_months=3):
    """
    查找2025年数据文件

    Args:
        max_months: 最大月份数（默认3，表示前3个月）

    Returns:
        数据文件路径列表
    """
    print(f"查找2025年前{max_months}个月数据文件...")

    data_files = []

    # 查找2025目录下的前max_months个月份
    year_dir = '2025'
    if os.path.exists(year_dir):
        # 获取所有月份目录
        month_dirs = []
        for item in os.listdir(year_dir):
            item_path = os.path.join(year_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                month_num = int(item)
                if 1 <= month_num <= max_months:
                    month_dirs.append(item_path)

        # 按月份排序
        month_dirs.sort()

        print(f"找到 {len(month_dirs)} 个月份目录: {[os.path.basename(d) for d in month_dirs]}")

        # 在每个月份目录中查找7z文件
        for month_dir in month_dirs:
            # 递归查找该月份目录下的所有.7z文件
            for root, dirs, files in os.walk(month_dir):
                for file in files:
                    if file.endswith('.7z'):
                        file_path = os.path.join(root, file)
                        data_files.append(file_path)

    data_files.sort()  # 按日期排序
    print(f"找到 {len(data_files)} 个数据文件")
    return data_files


def process_single_day(archive_path, extract_base_dir, temp_dir_prefix="temp_extract_"):
    """
    处理单个交易日数据

    Args:
        archive_path: 7z文件路径
        extract_base_dir: 解压基础目录
        temp_dir_prefix: 临时目录前缀

    Returns:
        该交易日的DataFrame
    """
    try:
        # 从文件名中提取日期
        filename = os.path.basename(archive_path)
        date_str = filename.replace('.7z', '')

        print(f"\n处理日期: {date_str}")

        # 解压文件到指定目录
        temp_extract_dir = os.path.join(project_root, 'temp_extract', date_str)
        os.makedirs(temp_extract_dir, exist_ok=True)

        # 解压文件
        extractor = SevenZipExtractor(archive_path)
        extract_dir = extractor.extract_all(temp_extract_dir)

        # 查找所有CSV文件（递归查找，因为可能在子目录中）
        csv_files = list(Path(extract_dir).rglob("*.csv"))
        print(f"  找到 {len(csv_files)} 个股票文件")

        if len(csv_files) == 0:
            print(f"  警告: {date_str} 没有找到CSV文件")
            return pd.DataFrame()

        # 处理所有股票
        all_stocks_data = []

        for csv_file in tqdm(csv_files, desc=f"  处理股票"):
            try:
                stock_code = csv_file.stem

                # 根据股票代码动态确定涨停比例
                stock_type = determine_stock_type(stock_code)
                limit_ratio = get_limit_ratio(stock_type)

                # 获取基准价格（从盘口价格获取，因为没有昨收价）
                preclose = None
                try:
                    df_raw = pd.read_csv(csv_file, nrows=100)  # 读取更多行找到有效价格

                    # 尝试从盘口价格获取基准价
                    # 优先使用买一价格，其次用卖一价格
                    if 'b1_p' in df_raw.columns:
                        # 找到第一个非零的买一价格
                        valid_prices = df_raw[df_raw['b1_p'] > 0]['b1_p']
                        if len(valid_prices) > 0:
                            preclose = valid_prices.iloc[0]

                    if preclose is None and 'a1_p' in df_raw.columns:
                        # 找到第一个非零的卖一价格
                        valid_prices = df_raw[df_raw['a1_p'] > 0]['a1_p']
                        if len(valid_prices) > 0:
                            preclose = valid_prices.iloc[0]

                    # 如果盘口价格都没有，尝试用开盘价或当前价
                    if preclose is None:
                        if 'open' in df_raw.columns:
                            valid_prices = df_raw[df_raw['open'] > 0]['open']
                            if len(valid_prices) > 0:
                                preclose = valid_prices.iloc[0]

                    if preclose is None and 'current' in df_raw.columns:
                        valid_prices = df_raw[df_raw['current'] > 0]['current']
                        if len(valid_prices) > 0:
                            preclose = valid_prices.iloc[0]

                except Exception as e:
                    print(f"  读取 {stock_code} 数据出错: {e}")

                # 如果仍然没有有效价格，跳过这个股票
                if preclose is None or preclose <= 0:
                    print(f"  跳过: {stock_code} 缺少有效的价格数据")
                    continue

                print(f"  基准价格: {preclose:.2f} (从盘口价格获取)")

                # 处理tick文件
                df = process_tick_file(str(csv_file), preclose, limit_ratio)

                if df.empty:
                    continue

                # 添加股票代码和日期
                df['code'] = stock_code
                df['date'] = date_str

                # 提取特征（传入涨停比例）
                df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=limit_ratio)

                # 生成标签（V3.2首次触板逻辑）
                df = generate_event_driven_label(df)  # 直接使用V3.2

                all_stocks_data.append(df)

            except Exception as e:
                # 单个股票处理失败不影响整体
                continue

        # 清理临时目录
        import shutil
        shutil.rmtree(temp_extract_dir, ignore_errors=True)

        if all_stocks_data:
            # 合并所有股票数据
            day_df = pd.concat(all_stocks_data, ignore_index=True)
            print(f"  完成: {date_str}, 总计 {len(day_df)} 个样本")
            return day_df
        else:
            print(f"  警告: {date_str} 处理后没有有效数据")
            return pd.DataFrame()

    except Exception as e:
        print(f"  错误处理 {archive_path}: {e}")
        return pd.DataFrame()


def build_full_year_dataset(year="2025", output_path="data/processed/2025_full_dataset.csv",
                           max_files=None, sample_files=None,
                           split_dataset=False, train_ratio=0.70, val_ratio=0.15,
                           train_date_range=None, val_date_range=None, test_date_range=None,
                           max_months=3):
    """
    构建全年数据集

    Args:
        year: 年份
        output_path: 输出文件路径
        max_files: 最大处理文件数（用于测试）
        sample_files: 采样间隔，如3表示每3个文件处理1个（用于测试）
        split_dataset: 是否进行数据集划分
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        train_date_range: 训练集日期范围
        val_date_range: 验证集日期范围
        test_date_range: 测试集日期范围
        max_months: 最大月份数（默认3，表示前3个月）
    """
    print("="*80)
    print(f"构建{year}年全年数据集")
    print("="*80)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 查找所有数据文件（前max_months个月）
    data_files = find_2025_data_files(max_months=max_months)

    if len(data_files) == 0:
        print("错误: 未找到数据文件")
        return False

    # 如果设置了采样或最大文件数
    if sample_files:
        data_files = data_files[::sample_files]
        print(f"采样模式: 每{sample_files}个文件处理1个，剩余{len(data_files)}个文件")

    if max_files and len(data_files) > max_files:
        data_files = data_files[:max_files]
        print(f"限制模式: 最多处理{max_files}个文件")

    # 分批处理并保存
    all_data = []
    temp_output_dir = os.path.dirname(output_path)

    for i, archive_path in enumerate(tqdm(data_files, desc="处理交易日")):
        print(f"\n进度: {i+1}/{len(data_files)}")

        # 处理单个交易日
        day_df = process_single_day(archive_path, temp_output_dir)

        if not day_df.empty:
            all_data.append(day_df)

            # 定期保存中间结果（每10个交易日）
            if len(all_data) % 10 == 0:
                temp_df = pd.concat(all_data, ignore_index=True)
                temp_path = f"{output_path}.temp_{len(all_data)}_days"
                save_dataset(temp_df, temp_path)
                print(f"已保存中间结果: {temp_path}")

    # 合并所有数据
    if all_data:
        print("\n合并所有数据...")
        final_df = pd.concat(all_data, ignore_index=True)

        # 保存最终数据集
        print(f"保存最终数据集到: {output_path}")
        save_dataset(final_df, output_path)

        # 显示数据集统计
        print("\n数据集统计:")
        print(f"总样本数: {len(final_df)}")
        print(f"交易日数: {len(final_df['date'].unique())}")
        print(f"股票数: {len(final_df['code'].unique())}")
        print(f"特征数: {len(get_feature_columns(final_df))}")

        # 显示标签统计
        if 'label' in final_df.columns:
            stats = get_label_statistics(final_df)
            print(f"正样本数: {stats['positive_samples']}")
            print(f"负样本数: {stats['negative_samples']}")
            print(f"正样本比例: {stats['positive_ratio']:.6%}")

        # 运行质量检查
        REQUIRED_FEATURES = [
            'dist_to_limit', 'ticks_to_limit', 'ask1_to_limit', 'ask1_gap',
            'bid_depth', 'ask_depth', 'order_imbalance', 'b1_volume', 'a1_volume',
            'spread', 'ask_slope', 'bid_slope',
            'ret_1tick', 'vol_delta', 'money_delta'
        ]

        quality_results = run_quality_checks(final_df, REQUIRED_FEATURES)
        print_quality_report(quality_results)

        # 按交易日划分数据集
        if split_dataset:
            print("\n" + "="*80)
            print("数据集划分")
            print("="*80)

            # 检测使用哪种模式
            use_date_range_mode = (train_date_range is not None or
                                  val_date_range is not None or
                                  test_date_range is not None)

            if use_date_range_mode:
                print(f"使用日期范围模式")
                print(f"  训练集范围: {train_date_range}")
                print(f"  验证集范围: {val_date_range}")
                print(f"  测试集范围: {test_date_range}")

            train_df, val_df, test_df = split_dataset_by_trading_day(
                final_df,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=1.0 - train_ratio - val_ratio,
                train_date_range=train_date_range,
                val_date_range=val_date_range,
                test_date_range=test_date_range
            )

            # 【新增】对训练集进行负样本欠采样
            train_df = undersample_train_set(
                train_df,
                dist_col='dist_to_limit',
                label_col='label',
                thresholds=(0.01, 0.05),
                keep_ratios=(1.0, 0.3, 0.05),
                target_ratio=5.0,
                random_seed=42,
                verbose=True
            )

            # 保存划分后的数据集
            split_output_dir = os.path.join(os.path.dirname(output_path), 'split_datasets')
            save_split_datasets(train_df, val_df, test_df, split_output_dir, format='parquet')

            print("\n✅ 数据集划分完成！")
            print(f"训练集: {len(train_df):,} 样本 ({len(train_df)/len(final_df):.2%})")
            print(f"验证集: {len(val_df):,} 样本 ({len(val_df)/len(final_df):.2%})")
            print(f"测试集: {len(test_df):,} 样本 ({len(test_df)/len(final_df):.2%})")

        return True
    else:
        print("错误: 没有生成任何有效数据")
        return False


def build_incremental_dataset(year="2025", output_dir="data/processed/2025_incremental",
                             batch_size=10, max_months=3):
    """
    增量构建数据集，每个批次单独保存

    Args:
        year: 年份
        output_dir: 输出目录
        batch_size: 每个批次的交易日数
        max_months: 最大月份数（默认3）
    """
    print("="*80)
    print(f"增量构建{year}年数据集")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # 查找所有数据文件（前max_months个月）
    data_files = find_2025_data_files(max_months=max_months)

    if len(data_files) == 0:
        print("错误: 未找到数据文件")
        return False

    # 分批处理
    for batch_idx in range(0, len(data_files), batch_size):
        batch_files = data_files[batch_idx:batch_idx + batch_size]

        print(f"\n处理批次 {batch_idx // batch_size + 1}: 文件 {batch_idx+1}-{min(batch_idx+batch_size, len(data_files))}")

        batch_data = []
        processed_count = 0
        failed_count = 0

        for archive_path in batch_files:
            try:
                print(f"\n处理文件: {archive_path}")
                day_df = process_single_day(archive_path, output_dir)
                if not day_df.empty:
                    batch_data.append(day_df)
                    processed_count += 1
                    print(f"✓ 成功处理: {processed_count}/{len(batch_files)}")
                else:
                    failed_count += 1
                    print(f"✗ 无数据: {failed_count}/{len(batch_files)}")
            except Exception as e:
                failed_count += 1
                print(f"✗ 处理失败 {archive_path}: {e}")
                # 继续处理下一个文件，不中断整个批次
                continue

        # 保存批次结果（即使只有部分成功）
        if batch_data:
            try:
                batch_df = pd.concat(batch_data, ignore_index=True)
                batch_path = os.path.join(output_dir, f"batch_{batch_idx // batch_size + 1}.parquet")
                batch_df.to_parquet(batch_path, index=False)
                print(f"批次保存完成: {batch_path} (成功: {processed_count}, 失败: {failed_count})")
            except Exception as e:
                print(f"批次保存失败: {e}")
        else:
            print(f"批次 {batch_idx // batch_size + 1} 没有有效数据")

    print(f"\n增量构建完成！共保存 {len(os.listdir(output_dir))} 个批次文件")
    return True


def merge_incremental_datasets(inc_dir="data/processed/2025_incremental",
                              output_path="data/processed/2025_full_dataset.parquet"):
    """
    合并增量数据集

    Args:
        inc_dir: 增量数据目录
        output_path: 合并后的输出路径
    """
    print("="*80)
    print("合并增量数据集")
    print("="*80)

    # 查找所有批次文件
    batch_files = sorted(glob.glob(os.path.join(inc_dir, "batch_*.parquet")))

    if not batch_files:
        print("错误: 未找到批次文件")
        return False

    print(f"找到 {len(batch_files)} 个批次文件")

    # 读取并合并
    all_data = []
    for batch_file in tqdm(batch_files, desc="读取批次文件"):
        batch_df = pd.read_parquet(batch_file)
        all_data.append(batch_df)

    # 合并
    final_df = pd.concat(all_data, ignore_index=True)

    # 保存
    final_df.to_parquet(output_path, index=False)
    print(f"合并完成: {output_path}")
    print(f"总样本数: {len(final_df)}")

    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='构建2025全年数据集')
    parser.add_argument('--mode', type=str, choices=['full', 'incremental', 'merge'],
                       default='full', help='构建模式')
    parser.add_argument('--year', type=str, default='2025', help='年份')
    parser.add_argument('--output', type=str, default='data/processed/2025_full_dataset.csv',
                       help='输出文件路径')
    parser.add_argument('--max-files', type=int, help='最大处理文件数（测试用）')
    parser.add_argument('--sample', type=int, help='采样间隔（测试用）')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='增量模式的批次大小')
    parser.add_argument('--split-dataset', action='store_true', help='是否进行数据集划分')
    parser.add_argument('--train-ratio', type=float, default=0.70, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--max-months', type=int, default=3, help='最大月份数（默认3）')

    # 日期范围划分参数
    parser.add_argument('--train-date-start', type=str,
                       help='训练集开始日期 (YYYY-MM-DD)')
    parser.add_argument('--train-date-end', type=str,
                       help='训练集结束日期 (YYYY-MM-DD)')
    parser.add_argument('--val-date-start', type=str,
                       help='验证集开始日期 (YYYY-MM-DD)')
    parser.add_argument('--val-date-end', type=str,
                       help='验证集结束日期 (YYYY-MM-DD)')
    parser.add_argument('--test-date-start', type=str,
                       help='测试集开始日期 (YYYY-MM-DD)')
    parser.add_argument('--test-date-end', type=str,
                       help='测试集结束日期 (YYYY-MM-DD)')

    args = parser.parse_args()

    # 构建日期范围参数
    train_date_range = None
    val_date_range = None
    test_date_range = None

    if args.train_date_start and args.train_date_end:
        train_date_range = (args.train_date_start, args.train_date_end)

    if args.val_date_start and args.val_date_end:
        val_date_range = (args.val_date_start, args.val_date_end)

    if args.test_date_start and args.test_date_end:
        test_date_range = (args.test_date_start, args.test_date_end)

    if args.mode == 'full':
        # 构建完整数据集
        success = build_full_year_dataset(
            year=args.year,
            output_path=args.output,
            max_files=args.max_files,
            sample_files=args.sample,
            split_dataset=args.split_dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            train_date_range=train_date_range,
            val_date_range=val_date_range,
            test_date_range=test_date_range,
            max_months=args.max_months
        )
    elif args.mode == 'incremental':
        # 增量构建
        inc_dir = args.output.replace('.csv', '_incremental').replace('.parquet', '_incremental')
        success = build_incremental_dataset(
            year=args.year,
            output_dir=inc_dir,
            batch_size=args.batch_size
        )
    elif args.mode == 'merge':
        # 合并增量数据集
        inc_dir = args.output.replace('.csv', '_incremental').replace('.parquet', '_incremental')
        success = merge_incremental_datasets(
            inc_dir=inc_dir,
            output_path=args.output
        )

    if success:
        print("\n✅ 数据集构建成功！")
        sys.exit(0)
    else:
        print("\n❌ 数据集构建失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()