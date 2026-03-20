"""
测试数据处理是否正常工作
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

# 测试单个股票文件
test_file = "data/temp_extract/2025-01-02/2025-01-02/000001.csv"

if not os.path.exists(test_file):
    print(f"文件不存在: {test_file}")
    sys.exit(1)

print("="*80)
print("测试数据处理 - 股票 000001")
print("="*80)

# 读取原始数据
print("\n1. 读取原始数据...")
raw_df = pd.read_csv(test_file, nrows=20)
print(f"原始数据前20行:")
print(raw_df[['time', 'open', 'current', 'high', 'low', 'b1_p', 'a1_p']])

# 获取基准价格
print(f"\n2. 获取基准价格...")
preclose = None
for price_col in ['b1_p', 'a1_p', 'open', 'current']:
    if price_col in raw_df.columns:
        valid_prices = raw_df[raw_df[price_col] > 0][price_col]
        if len(valid_prices) > 0:
            preclose = valid_prices.iloc[0]
            print(f"  使用 {price_col}: {preclose}")
            break

if preclose is None:
    print("  [ERROR] 无法获取基准价格")
    sys.exit(1)

# 股票类型和涨停比例
stock_type = determine_stock_type("000001")
limit_ratio = get_limit_ratio(stock_type)
print(f"  股票类型: {stock_type}")
print(f"  涨停比例: {limit_ratio}")

# 处理tick文件
print(f"\n3. 处理tick文件...")
try:
    df = process_tick_file(str(test_file), preclose, limit_ratio)
    print(f"  处理后数据行数: {len(df)}")

    if not df.empty:
        print(f"\n4. 处理后数据统计:")
        print(f"  时间范围: {df['time'].min()} - {df['time'].max()}")
        print(f"  价格范围: {df['current'].min()} - {df['current'].max()}")

        # 检查是否有有效的current价格
        zero_current_count = (df['current'] <= 0).sum()
        print(f"  current=0的数量: {zero_current_count}")

        if zero_current_count == 0:
            print(f"  [OK] 所有current价格都有效")

        # 检查盘口价
        if 'b1_p' in df.columns and 'a1_p' in df.columns:
            valid_b1_p = (df['b1_p'] > 0).sum()
            valid_a1_p = (df['a1_p'] > 0).sum()
            print(f"  有效的买一价数量: {valid_b1_p}")
            print(f"  有效的卖一价数量: {valid_a1_p}")

        # 检查涨停价
        if 'limit_price' in df.columns:
            print(f"  涨停价: {df['limit_price'].iloc[0]}")

            # 检查是否有触板事件
            limit_up_count = (df['current'] >= df['limit_price']).sum()
            print(f"  触板tick数量: {limit_up_count}")

            if limit_up_count > 0:
                print(f"  [OK] 有触板事件")
                print(f"  触板时间示例:")
                limit_up_samples = df[df['current'] >= df['limit_price']].head(5)
                print(limit_up_samples[['time', 'current', 'limit_price']])
            else:
                print(f"  [X] 无触板事件")

                # 检查接近涨停的情况
                if 'dist_to_limit' in df.columns:
                    near_limit = (df['dist_to_limit'] < 0.02).sum()
                    print(f"  接近涨停(2%以内)的tick数量: {near_limit}")

                    if near_limit > 0:
                        print(f"  [OK] 有困难负样本")
                    else:
                        print(f"  [X] 无困难负样本")
    else:
        print(f"  [X] 处理后数据为空")

except Exception as e:
    print(f"  [ERROR] 处理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试完成")
print("="*80)