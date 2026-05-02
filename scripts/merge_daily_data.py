"""
合并日频训练数据为滚动验证用的大数据集

将 data/daily_train_undersampled/{月份}/ 下的 CSV 文件合并
输出到 data/merged/multi_day_train.csv

使用方法:
    python scripts/merge_daily_data.py
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_processing.csv_utils import get_feature_columns, read_csv, write_csv

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
UNDERSAMPLED_DIR = os.path.join(DATA_DIR, 'daily_train_undersampled')
MERGED_DIR = os.path.join(DATA_DIR, 'merged')
OUTPUT_FILE = os.path.join(MERGED_DIR, 'multi_day_train.csv')


def get_month_from_path(csv_path: str) -> int:
    """从CSV路径中提取月份"""
    # 例如: data/daily_train_undersampled/01/2025-01-02_train.csv
    parts = csv_path.split(os.sep)
    for part in parts:
        if part.isdigit() and len(part) == 2:
            return int(part)
    return 0


def collect_csv_files() -> list:
    """收集所有待合并的CSV文件"""
    csv_files = []

    # 按月份遍历目录
    for month_dir in sorted(glob.glob(os.path.join(UNDERSAMPLED_DIR, '*'))):
        if not os.path.isdir(month_dir):
            continue

        month_str = os.path.basename(month_dir)
        if not month_str.isdigit() or int(month_str) > 12:
            continue

        # 收集该月份下的所有CSV
        pattern = os.path.join(month_dir, '*_train.csv')
        for csv_file in glob.glob(pattern):
            csv_files.append(csv_file)

    print(f"找到 {len(csv_files)} 个CSV文件")
    return csv_files


def merge_and_save():
    """合并所有CSV并保存为CSV"""
    start_time = datetime.now()

    # 确保输出目录存在
    os.makedirs(MERGED_DIR, exist_ok=True)

    # 收集CSV文件
    csv_files = collect_csv_files()

    if not csv_files:
        print("错误: 未找到任何CSV文件")
        return False

    # 按日期排序
    csv_files = sorted(csv_files)

    # 读取并合并
    print("\n开始读取和合并数据...")
    dfs = []

    for csv_file in tqdm(csv_files, desc="读取CSV"):
        try:
            df = read_csv(csv_file)

            # 添加月份列（如果不存在）
            if 'month' not in df.columns:
                month = get_month_from_path(csv_file)
                df['month'] = month

            # 确保日期列是字符串格式
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str)

            dfs.append(df)

        except Exception as e:
            print(f"\n警告: 读取 {csv_file} 失败: {e}")
            continue

    if not dfs:
        print("错误: 没有成功读取任何数据")
        return False

    # 合并所有数据
    print("\n合并数据...")
    merged_df = pd.concat(dfs, ignore_index=True)

    # 释放内存
    del dfs

    # 数据统计
    print("\n" + "="*60)
    print("合并完成!")
    print("="*60)
    print(f"总样本数: {len(merged_df):,}")
    print(f"总特征数: {len(get_feature_columns(merged_df))}")
    print(f"日期范围: {merged_df['date'].min()} ~ {merged_df['date'].max()}")
    print(f"月份分布:\n{merged_df['month'].value_counts().sort_index().to_string()}")

    # 标签分布
    if 'label' in merged_df.columns:
        pos_count = (merged_df['label'] == 1).sum()
        neg_count = (merged_df['label'] == 0).sum()
        print(f"\n标签分布:")
        print(f"  正样本: {pos_count:,} ({pos_count/len(merged_df)*100:.2f}%)")
        print(f"  负样本: {neg_count:,} ({neg_count/len(merged_df)*100:.2f}%)")

    # 保存为CSV
    print(f"\n保存到: {OUTPUT_FILE}")
    write_csv(merged_df, OUTPUT_FILE)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n耗时: {elapsed:.1f} 秒")
    print(f"文件大小: {os.path.getsize(OUTPUT_FILE) / 1024**2:.1f} MB")

    return True


def main():
    print("="*60)
    print("日频训练数据合并脚本")
    print("="*60)
    print(f"输入目录: {UNDERSAMPLED_DIR}")
    print(f"输出文件: {OUTPUT_FILE}")
    print("="*60)

    success = merge_and_save()

    if success:
        print("\n合并成功!")
        sys.exit(0)
    else:
        print("\n合并失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
