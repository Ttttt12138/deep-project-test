"""
调试诊断脚本 - 检查触板事件被过滤的具体原因
基于V3.2版本代码分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.feature_engineering.event_driven_labels import (
    generate_event_driven_label,
    filter_and_label_events,
    get_event_statistics
)
from src.feature_engineering.event_window_builder import EventWindowBuilder
from src.feature_engineering.limit_up_features import extract_limit_up_features
from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.data_processing.csv_utils import read_csv


def diagnose_single_stock(csv_file: str, verbose: bool = True) -> dict:
    """
    诊断单只股票的处理失败原因

    Args:
        csv_file: CSV文件路径
        verbose: 是否显示详细信息

    Returns:
        诊断结果字典
    """
    diagnosis = {
        'file': csv_file,
        'stock_code': Path(csv_file).stem,
        'success': False,
        'failure_stage': None,
        'failure_reason': None,
        'details': {}
    }

    try:
        # 诊断1：基准价格获取
        if verbose:
            print(f"\n[{diagnosis['stock_code']}] 诊断开始...")

        preclose = _get_benchmark_price(csv_file)
        if preclose is None or preclose <= 0:
            diagnosis['failure_stage'] = 'benchmark_price'
            diagnosis['failure_reason'] = f'无法获取有效基准价格 (preclose={preclose})'
            if verbose:
                print(f"  [FAIL] {diagnosis['failure_reason']}")
            return diagnosis

        diagnosis['details']['preclose'] = preclose

        # 诊断2：股票类型和涨停比例
        stock_type = determine_stock_type(diagnosis['stock_code'])
        limit_ratio = get_limit_ratio(stock_type)
        diagnosis['details']['stock_type'] = stock_type
        diagnosis['details']['limit_ratio'] = limit_ratio

        # 诊断3：数据加载和过滤
        df = process_tick_file_optimized(csv_file, preclose, limit_ratio)
        if df.empty:
            diagnosis['failure_stage'] = 'data_loading'
            diagnosis['failure_reason'] = '数据加载后为空'
            if verbose:
                print(f"  [FAIL] {diagnosis['failure_reason']}")
            return diagnosis

        diagnosis['details']['original_ticks'] = len(df)
        if verbose:
            print(f"  [OK] 数据加载成功: {len(df)} ticks")

        # 诊断4：特征提取（关键步骤！）
        try:
            df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=limit_ratio)
            if verbose:
                print(f"  [OK] 特征提取成功")
        except Exception as e:
            diagnosis['failure_stage'] = 'feature_extraction'
            diagnosis['failure_reason'] = f'特征提取失败: {str(e)[:50]}'
            if verbose:
                print(f"  [FAIL] {diagnosis['failure_reason']}")
            return diagnosis

        # 诊断5：特征完整性检查
        required_features = [
            'dist_to_limit', 'ticks_to_limit', 'ask1_to_limit', 'ask1_gap',
            'bid_depth', 'ask_depth', 'order_imbalance', 'b1_volume', 'a1_volume',
            'spread', 'ask_slope', 'bid_slope',
            'ret_1tick', 'vol_delta', 'money_delta'
        ]

        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            diagnosis['failure_stage'] = 'feature_extraction'
            diagnosis['failure_reason'] = f'缺少必需特征: {missing_features[:3]}...'
            diagnosis['details']['missing_features'] = missing_features
            if verbose:
                print(f"  [FAIL] {diagnosis['failure_reason']}")
            return diagnosis

        if verbose:
            print(f"  [OK] 特征提取完整")

        # 诊断6：触板事件检测
        try:
            # 尝试生成标签
            labeled_df = generate_event_driven_label(df)
            touch_events = int(labeled_df['label'].sum())

            diagnosis['details']['touch_events_detected'] = touch_events

            if verbose:
                print(f"  [OK] 触板事件检测: {touch_events} 个")

            if touch_events == 0:
                # 检查困难负样本
                if 'dist_to_limit' in labeled_df.columns:
                    hard_negatives = (labeled_df['dist_to_limit'] < 0.05).sum()
                    diagnosis['details']['hard_negatives'] = int(hard_negatives)

                    if hard_negatives == 0:
                        diagnosis['failure_stage'] = 'event_detection'
                        diagnosis['failure_reason'] = '无触板事件且无困难负样本'
                        if verbose:
                            print(f"  [WARN]  {diagnosis['failure_reason']}")
                        return diagnosis
                    else:
                        if verbose:
                            print(f"  [INFO]  困难负样本: {hard_negatives} 个")

        except Exception as e:
            diagnosis['failure_stage'] = 'label_generation'
            diagnosis['failure_reason'] = f'标签生成失败: {str(e)[:50]}'
            if verbose:
                print(f"  [FAIL] {diagnosis['failure_reason']}")
            return diagnosis

        # 诊断7：窗口构建
        # 添加必要列（窗口构建器需要）
        df['code'] = diagnosis['stock_code']
        df['date'] = diagnosis.get('date', 'unknown')

        builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.05)

        try:
            event_samples = builder.build_event_window_samples(
                df,
                verbose=False
            )

            if event_samples.empty:
                # 获取更详细的失败原因
                filtered_df, valid_indices = filter_and_label_events(df, 0.05)

                if len(valid_indices) == 0:
                    diagnosis['failure_stage'] = 'window_filtering'
                    diagnosis['failure_reason'] = '窗口过滤后无有效样本'

                    # 详细分析
                    if 'current' in df.columns and 'limit_price' in df.columns:
                        limit_up_state = (round(df['current'], 2) >= round(df['limit_price'], 2)).sum()
                        diagnosis['details']['limit_up_state'] = int(limit_up_state)

                        if limit_up_state > 0:
                            diagnosis['failure_reason'] = f'有{touch_events}个触板但被过滤（死板样本被排除）'
                        else:
                            diagnosis['failure_reason'] = f'触板检测{touch_events}个但窗口构建失败'

                    if verbose:
                        print(f"  [FAIL] {diagnosis['failure_reason']}")
                    return diagnosis
            else:
                diagnosis['success'] = True
                diagnosis['details']['event_samples'] = len(event_samples)
                diagnosis['details']['positive_samples'] = int(event_samples['label'].sum())

                if verbose:
                    print(f"  [OK] 窗口构建成功: {len(event_samples)} 样本 (正样本: {int(event_samples['label'].sum())})")

        except Exception as e:
            diagnosis['failure_stage'] = 'window_building'
            diagnosis['failure_reason'] = f'窗口构建异常: {str(e)[:50]}'
            if verbose:
                print(f"  [FAIL] {diagnosis['failure_reason']}")
            return diagnosis

        return diagnosis

    except Exception as e:
        diagnosis['failure_stage'] = 'general_exception'
        diagnosis['failure_reason'] = f'异常: {str(e)[:50]}'
        if verbose:
            print(f"  [FAIL] {diagnosis['failure_reason']}")
        return diagnosis


def _get_benchmark_price(csv_file: Path) -> float:
    """获取基准价格（复制自training_set_builder.py）"""
    try:
        df_raw = read_csv(csv_file, nrows=100, preserve_code=False)

        # 优先级：买一价 > 卖一价 > 开盘价 > 当前价
        for price_col in ['b1_p', 'a1_p', 'open', 'current']:
            if price_col in df_raw.columns:
                valid_prices = df_raw[df_raw[price_col] > 0][price_col]
                if len(valid_prices) > 0:
                    return valid_prices.iloc[0]
    except Exception:
        pass
    return None


def batch_diagnose(csv_files: list, sample_size: int = 10) -> dict:
    """
    批量诊断多个股票文件

    Args:
        csv_files: CSV文件列表
        sample_size: 采样数量（用于测试）

    Returns:
        批量诊断结果
    """
    print("="*80)
    print("批量诊断开始")
    print("="*80)

    if sample_size:
        csv_files = csv_files[:sample_size]
        print(f"采样模式: 诊断前 {sample_size} 个文件")

    results = {
        'total_files': len(csv_files),
        'success': 0,
        'failed': 0,
        'failures_by_stage': {},
        'failures_by_reason': {},
        'sample_results': []
    }

    for csv_file in csv_files:
        diagnosis = diagnose_single_stock(csv_file, verbose=True)
        results['sample_results'].append(diagnosis)

        if diagnosis['success']:
            results['success'] += 1
        else:
            results['failed'] += 1
            stage = diagnosis['failure_stage']
            reason = diagnosis['failure_reason']

            results['failures_by_stage'][stage] = results['failures_by_stage'].get(stage, 0) + 1
            results['failures_by_reason'][reason] = results['failures_by_reason'].get(reason, 0) + 1

    # 打印汇总
    print("\n" + "="*80)
    print("诊断汇总")
    print("="*80)
    print(f"总文件数: {results['total_files']}")
    print(f"成功: {results['success']}")
    print(f"失败: {results['failed']}")
    print(f"成功率: {results['success']/results['total_files']*100:.1f}%")

    if results['failures_by_stage']:
        print(f"\n失败阶段分布:")
        for stage, count in sorted(results['failures_by_stage'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {stage}: {count} ({count/results['failed']*100:.1f}%)")

    if results['failures_by_reason']:
        print(f"\n失败原因分布:")
        for reason, count in sorted(results['failures_by_reason'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {reason}: {count}")

    return results


def find_specific_failure_type(csv_files: list, failure_type: str) -> list:
    """
    查找特定失败类型的股票文件

    Args:
        csv_files: CSV文件列表
        failure_type: 失败类型（'benchmark_price', 'feature_extraction', 'event_detection'等）

    Returns:
        符合条件的文件列表
    """
    matching_files = []
    for csv_file in csv_files:
        diagnosis = diagnose_single_stock(csv_file, verbose=False)
        if diagnosis['failure_stage'] == failure_type:
            matching_files.append(diagnosis)

    print(f"\n找到 {len(matching_files)} 个 {failure_type} 失败的文件:")
    for diag in matching_files[:5]:
        print(f"  {diag['stock_code']}: {diag['failure_reason']}")

    return matching_files


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='调试诊断脚本')
    parser.add_argument('--csv-dir', type=str, default='data/extracted',
                       help='CSV文件目录')
    parser.add_argument('--single', type=str, help='诊断单个文件')
    parser.add_argument('--sample-size', type=int, default=10,
                       help='批量诊断采样数量')
    parser.add_argument('--failure-type', type=str,
                       help='查找特定失败类型')

    args = parser.parse_args()

    if args.single:
        # 诊断单个文件
        diagnosis = diagnose_single_stock(args.single, verbose=True)
        print(f"\n诊断结果: {'成功' if diagnosis['success'] else '失败'}")
        if not diagnosis['success']:
            print(f"失败阶段: {diagnosis['failure_stage']}")
            print(f"失败原因: {diagnosis['failure_reason']}")
    else:
        # 批量诊断
        csv_dir = Path(args.csv_dir)
        csv_files = list(csv_dir.rglob("*.csv"))

        if not csv_files:
            print(f"未找到CSV文件: {args.csv_dir}")
            return

        if args.failure_type:
            # 查找特定失败类型
            find_specific_failure_type(csv_files, args.failure_type)
        else:
            # 批量诊断
            batch_diagnose(csv_files, args.sample_size)


if __name__ == "__main__":
    main()
