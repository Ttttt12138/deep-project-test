"""
测试放宽过滤条件后的效果
"""
import pandas as pd
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized as process_tick_file
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.feature_engineering.limit_up_features import extract_limit_up_features
from src.feature_engineering.event_driven_labels import filter_event_samples

# 测试单个股票文件
test_file = "data/temp_extract/2025-01-02/2025-01-02/000001.csv"

print("="*80)
print("测试放宽过滤条件 - 股票 000001")
print("="*80)

# 处理tick文件
print("\n1. 处理tick文件...")
preclose = 11.72  # 从之前的测试中获得
stock_type = determine_stock_type("000001")
limit_ratio = get_limit_ratio(stock_type)

df = process_tick_file(str(test_file), preclose, limit_ratio)
print(f"  处理后数据行数: {len(df)}")

if not df.empty:
    # 添加股票代码和日期
    df['code'] = '000001'
    df['date'] = '2025-01-02'

    print(f"\n2. 提取基础特征...")
    df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=limit_ratio)
    print(f"  特征提取后行数: {len(df)}")

    print(f"\n3. 应用事件驱动过滤...")
    filtered_df, valid_indices = filter_event_samples(df, limit_dist_threshold=0.05)
    print(f"  过滤前: {len(df)} 行")
    print(f"  过滤后保留: {len(valid_indices)} 行 ({len(valid_indices)/len(df)*100:.1f}%)")

    if len(valid_indices) > 0:
        print(f"\n4. 过滤后数据统计:")
        print(f"  正样本数量: {filtered_df.loc[valid_indices, 'label'].sum()}")
        print(f"  负样本数量: {len(valid_indices) - filtered_df.loc[valid_indices, 'label'].sum()}")

        # 距离涨停分布
        if 'dist_to_limit' in filtered_df.columns:
            distances = filtered_df.loc[valid_indices, 'dist_to_limit']
            print(f"  距离涨停统计:")
            print(f"    最小距离: {distances.min():.4f}")
            print(f"    最大距离: {distances.max():.4f}")
            print(f"    平均距离: {distances.mean():.4f}")
            print(f"    中位数距离: {distances.median():.4f}")

            # 分距离段统计
            print(f"  距离分布:")
            print(f"    <2%: {(distances < 0.02).sum()}")
            print(f"    2%-5%: {((distances >= 0.02) & (distances < 0.05)).sum()}")
            print(f"    5%-10%: {((distances >= 0.05) & (distances < 0.10)).sum()}")
            print(f"    10%-15%: {((distances >= 0.10) & (distances < 0.15)).sum()}")
            print(f"    >=15%: {(distances >= 0.15).sum()}")
    else:
        print(f"  [X] 过滤后无有效样本")
else:
    print(f"  [X] 处理后数据为空")

print("\n" + "="*80)
print("测试完成")
print("="*80)