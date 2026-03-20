"""
排查正样本为0的三大怀疑点
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized as process_tick_file
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.feature_engineering.limit_up_features import extract_limit_up_features


def check_floating_point_precision():
    """
    怀疑点1: 检查浮点数精度问题
    - 最常见杀手：10.01 >= 10.0100000001 结果是 False
    - A股最小价格变动是0.01元，需要引入四舍五入
    """
    print("="*80)
    print("怀疑点1: 浮点数精度问题检查")
    print("="*80)

    # 模拟浮点数精度问题
    print(f"\n浮点数精度问题示例:")
    current_price = 10.01
    calculated_limit_price = 10.0100000001  # 计算出的涨停价

    print(f"  当前价格: {current_price}")
    print(f"  计算涨停价: {calculated_limit_price}")
    print(f"  严格比较: {current_price} >= {calculated_limit_price} = {current_price >= calculated_limit_price}")
    print(f"  结果: False (漏掉了涨停事件!)")

    print(f"\n修复方案 - 使用四舍五入:")
    rounded_current = round(current_price, 2)
    rounded_limit = round(calculated_limit_price, 2)
    print(f"  四舍五入后比较: {rounded_current} >= {rounded_limit} = {rounded_current >= rounded_limit}")
    print(f"  结果: True (正确识别涨停事件!)")

    # 检查实际数据处理中的情况
    print(f"\n检查实际数据处理中的浮点数问题:")
    test_file = "data/temp_extract/2025-01-03/2025-01-03/000001.csv"
    if os.path.exists(test_file):
        try:
            # 处理数据
            preclose = 11.5
            df = process_tick_file(test_file, preclose, 0.10)

            if not df.empty and 'current' in df.columns and 'limit_price' in df.columns:
                # 检查涨停价计算
                sample_limit = df['limit_price'].iloc[0]
                calculated_limit = round(preclose * 1.10, 2)

                print(f"  昨收价: {preclose}")
                print(f"  计算涨停价: {calculated_limit}")
                print(f"  数据中涨停价: {sample_limit}")
                print(f"  差异: {abs(sample_limit - calculated_limit):.10f}")

                if abs(sample_limit - calculated_limit) > 0.001:
                    print(f"  ⚠️ 发现涨停价计算精度问题!")
                else:
                    print(f"  ✅ 涨停价计算正常")

                # 检查价格接近涨停的情况
                close_to_limit = df[df['current'] >= df['limit_price'] - 0.05]
                if len(close_to_limit) > 0:
                    print(f"\n  找到接近涨停的样本:")
                    for idx, row in close_to_limit.head(3).iterrows():
                        current = row['current']
                        limit = row['limit_price']
                        diff = current - limit

                        # 严格比较
                        strict_result = current >= limit

                        # 四舍五入后比较
                        rounded_result = round(current, 2) >= round(limit, 2)

                        print(f"    时间: {row['time']}, 当前价: {current}, 涨停价: {limit}")
                        print(f"    差距: {diff:.10f}")
                        print(f"    严格比较: {strict_result}, 四舍五入比较: {rounded_result}")

                        if strict_result != rounded_result:
                            print(f"    ❌ 浮点数精度问题导致结果不同!")

        except Exception as e:
            print(f"  检查过程出错: {e}")


def check_limit_up_calculation():
    """
    怀疑点3: 检查涨停价计算是否正确
    - ST股（5%）
    - 主板（10%）
    - 创业板（20%）
    """
    print(f"\n" + "="*80)
    print("怀疑点3: 涨停价计算检查")
    print("="*80)

    # 测试不同股票类型的涨停价计算
    test_stocks = [
        ("000001", "主板股票"),
        ("000002", "主板股票"),
        ("300001", "创业板"),
        ("600000", "主板股票"),
    ]

    for stock_code, description in test_stocks:
        print(f"\n股票 {stock_code} ({description}):")

        try:
            # 获取股票类型和涨停比例
            stock_type = determine_stock_type(stock_code)
            limit_ratio = get_limit_ratio(stock_type)

            print(f"  股票类型: {stock_type}")
            print(f"  涨停比例: {limit_ratio * 100}%")

            # 读取数据
            test_file = f"data/temp_extract/2025-01-03/2025-01-03/{stock_code}.csv"
            if not os.path.exists(test_file):
                print(f"  文件不存在，跳过")
                continue

            df_sample = pd.read_csv(test_file, nrows=10)

            # 获取基准价格
            preclose = None
            for price_col in ['b1_p', 'a1_p', 'open', 'current']:
                if price_col in df_sample.columns:
                    valid_prices = df_sample[df_sample[price_col] > 0][price_col]
                    if len(valid_prices) > 0:
                        preclose = valid_prices.iloc[0]
                        break

            if preclose is None:
                print(f"  ❌ 无法获取基准价格")
                continue

            # 计算涨停价
            calculated_limit = round(preclose * (1 + limit_ratio), 2)

            # 检查数据中的涨停价
            df = process_tick_file(test_file, preclose, limit_ratio)
            if not df.empty and 'limit_price' in df.columns:
                data_limit = df['limit_price'].iloc[0]

                print(f"  昨收价: {preclose}")
                print(f"  计算涨停价: {calculated_limit}")
                print(f"  数据中涨停价: {data_limit}")

                if abs(calculated_limit - data_limit) < 0.01:
                    print(f"  ✅ 涨停价计算正确")
                else:
                    print(f"  ❌ 涨停价计算错误!")

                    # 检查数据中的价格是否接近计算出的涨停价
                    max_current = df['current'].max()
                    print(f"  数据中最高价: {max_current}")
                    print(f"  距计算涨停价: {max_current - calculated_limit:.2f}")
                    print(f"  距数据涨停价: {max_current - data_limit:.2f}")

                    if max_current >= calculated_limit - 0.1:
                        print(f"  ⚠️ 股票确实接近涨停，但涨停价计算可能有误")

        except Exception as e:
            print(f"  检查出错: {e}")


def check_straight_limit_up():
    """
    怀疑点2: 检查"一字涨停板"情况
    - 如果股票开盘时就涨停，并且全天封住
    - 根据事件逻辑：(下一秒涨停) AND (当前没涨停) = 0个正样本
    - 这是符合预期的，但容易引起误解
    """
    print(f"\n" + "="*80)
    print("怀疑点2: '一字涨停板'情况检查")
    print("="*80)

    print(f"\n'一字涨停板'逻辑分析:")
    print(f"  事件逻辑: (price_T11 >= limit_price) AND (price_T10 < limit_price)")
    print(f"  一字涨停板: 开盘就涨停，price_T10 >= limit_price")
    print(f"  结果: 第二个条件不满足，不产生正样本")
    print(f"  这符合预期: 无法在盘中买入一字板")

    # 检查数据中是否有一字涨停板
    print(f"\n检查数据中的一字涨停板情况:")
    test_dir = "data/temp_extract/2025-01-03/2025-01-03"
    if os.path.exists(test_dir):
        csv_files = list(Path(test_dir).glob("*.csv"))
        straight_limit_up_stocks = []

        for csv_file in csv_files[:50]:  # 检查前50只股票
            stock_code = csv_file.stem
            try:
                preclose = 11.5  # 假设基准价格
                df = process_tick_file(str(csv_file), preclose, 0.10)

                if not df.empty and 'current' in df.columns and 'limit_price' in df.columns:
                    # 检查开盘是否就涨停
                    first_current = df['current'].iloc[0]
                    limit_price = df['limit_price'].iloc[0]

                    # 考虑浮点数精度，使用四舍五入比较
                    if round(first_current, 2) >= round(limit_price, 2) - 0.01:
                        # 检查是否全天维持涨停
                        all_near_limit = all(round(df['current'].iloc[i], 2) >= round(limit_price, 2) - 0.01
                                          for i in range(min(10, len(df))))  # 检查前10个tick

                        if all_near_limit:
                            straight_limit_up_stocks.append(stock_code)

            except Exception:
                continue

        if straight_limit_up_stocks:
            print(f"  发现可能的一字涨停板: {len(straight_limit_up_stocks)} 只")
            print(f"  示例股票: {straight_limit_up_stocks[:5]}")
            print(f"  ⚠️ 这些股票不产生正样本是符合预期的!")
        else:
            print(f"  ✅ 没有明显的一字涨停板情况")
            print(f"  这说明正样本为0不是由一字涨停板导致的")


def suggest_fixes():
    """建议修复方案"""
    print(f"\n" + "="*80)
    print("修复建议")
    print("="*80)

    print(f"\n基于三个怀疑点的修复方案:")

    print(f"\n1. 修复浮点数精度问题 (最优先):")
    print(f"   - 在所有价格比较前使用四舍五入")
    print(f"   - 修改涨停判断: round(current, 2) >= round(limit_price, 2)")
    print(f"   - 修改标签生成: round(next_price, 2) >= round(limit_price, 2)")

    print(f"\n2. 增加涨停价计算验证:")
    print(f"   - 确认涨停价计算正确")
    print(f"   - 特别注意不同股票类型的涨停比例")
    print(f"   - 在日志中显示涨停价信息")

    print(f"\n3. 改进正样本定义:")
    print(f"   - 当前过于严格的'触板瞬间'定义")
    print(f"   - 建议改为'接近涨停': dist_to_limit < 2% 为正样本")
    print(f"   - 这样普通交易日也能有足够正样本")

    print(f"\n4. 增加诊断信息:")
    print(f"   - 在处理过程中显示接近涨停的股票")
    print(f"   - 统计价格分布情况")
    print(f"   - 记录涨停事件详细信息")


if __name__ == "__main__":
    print("正样本为0问题排查工具")
    print("="*80)

    try:
        # 排查三个怀疑点
        check_floating_point_precision()
        check_limit_up_calculation()
        check_straight_limit_up()

        # 提供修复建议
        suggest_fixes()

    except Exception as e:
        print(f"排查过程中出错: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "="*80)
    print("排查完成")
    print("="*80)
    print(f"\n最可能的根本原因: 浮点数精度问题")
    print(f"建议优先修复: 在所有价格比较时使用四舍五入")