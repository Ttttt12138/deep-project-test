"""
检查正样本定义和实际数据情况
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized as process_tick_file
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.feature_engineering.limit_up_features import extract_limit_up_features
from src.feature_engineering.event_driven_labels import (
    generate_event_driven_label,
    filter_event_samples,
    get_event_statistics
)
from src.feature_engineering.event_window_builder import EventWindowBuilder


def check_positive_sample_logic():
    """检查正样本定义逻辑"""
    print("="*80)
    print("检查正样本定义逻辑")
    print("="*80)

    # 检查事件窗口构建器的配置
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.05)
    print(f"\n事件窗口构建器配置:")
    print(f"  窗口大小: {builder.window_size}")
    print(f"  困难负样本阈值: {builder.limit_dist_threshold} ({builder.limit_dist_threshold*100:.1f}%)")

    # 检查几个样本股票
    test_files = [
        "data/temp_extract/2025-01-03/2025-01-03/000001.csv",
        "data/temp_extract/2025-01-03/2025-01-03/000002.csv",
        "data/temp_extract/2025-01-03/2025-01-03/000004.csv",
        "data/temp_extract/2025-01-03/2025-01-03/000005.csv",
        "data/temp_extract/2025-01-03/2025-01-03/000006.csv"
    ]

    for test_file in test_files:
        if not os.path.exists(test_file):
            continue

        stock_code = Path(test_file).stem
        print(f"\n检查股票 {stock_code}:")

        try:
            # 处理数据
            preclose = 11.5  # 假设的基准价格
            df = process_tick_file(test_file, preclose, 0.10)

            if df.empty:
                print(f"  数据为空")
                continue

            # 提取特征
            df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=0.10)

            # 生成事件驱动标签
            labeled_df = generate_event_driven_label(df)

            # 统计信息
            stats = get_event_statistics(labeled_df)

            print(f"  原始tick数: {len(df)}")
            print(f"  标签后tick数: {len(labeled_df)}")
            print(f"  触板事件数: {stats['limit_up_events']}")
            print(f"  涨停状态数: {stats['limit_up_state']}")

            if stats['limit_up_events'] > 0:
                print(f"  ✅ 有触板事件!")
                # 显示触板事件的详细信息
                limit_up_ticks = labeled_df[labeled_df['label'] == 1]
                print(f"  触板事件样本:")
                for idx, row in limit_up_ticks.head(3).iterrows():
                    print(f"    时间: {row['time']}, 当前价: {row['current']:.2f}, 涨停价: {row['limit_price']:.2f}")
            else:
                print(f"  ❌ 无触板事件")

            # 检查价格分布
            print(f"  价格范围: {df['current'].min():.2f} - {df['current'].max():.2f}")
            print(f"  涨停价: {df['limit_price'].iloc[0]:.2f}")

            # 检查接近涨停的情况
            if 'dist_to_limit' in df.columns:
                near_limit = (df['dist_to_limit'] < 0.05).sum()
                very_near = (df['dist_to_limit'] < 0.02).sum()
                print(f"  接近涨停(5%内): {near_limit} 个tick")
                print(f"  非常接近(2%内): {very_near} 个tick")

                # 检查是否有突破涨停的情况
                breakthrough_found = False
                for i in range(1, len(df)):
                    if df['current'].iloc[i] >= df['limit_price'].iloc[i] and df['current'].iloc[i-1] < df['limit_price'].iloc[i-1]:
                        print(f"  ⚠️ 发现突破涨停的tick: 时间 {df['time'].iloc[i]}, 价格 {df['current'].iloc[i]:.2f}")
                        breakthrough_found = True
                        break

                if not breakthrough_found:
                    print(f"  ❌ 确实没有突破涨停的情况")

        except Exception as e:
            print(f"  处理异常: {e}")
            import traceback
            traceback.print_exc()


def check_limit_up_stock_day():
    """检查是否有涨停股票的交易日"""
    print(f"\n" + "="*80)
    print("检查涨停股票情况")
    print("="*80)

    # 查找可能的涨停股票
    test_dir = "data/temp_extract/2025-01-03/2025-01-03"
    if not os.path.exists(test_dir):
        print(f"目录不存在: {test_dir}")
        return

    csv_files = list(Path(test_dir).glob("*.csv"))
    print(f"找到 {len(csv_files)} 个股票文件")

    # 随机检查一些股票
    import numpy as np
    np.random.seed(42)
    sample_files = np.random.choice(csv_files, min(10, len(csv_files)), replace=False)

    limit_up_stocks = []

    for csv_file in sample_files:
        stock_code = csv_file.stem
        try:
            # 快速读取前100行检查价格
            df_sample = pd.read_csv(csv_file, nrows=100)

            # 获取涨停价
            preclose = None
            for price_col in ['b1_p', 'a1_p', 'open', 'current']:
                if price_col in df_sample.columns:
                    valid_prices = df_sample[df_sample[price_col] > 0][price_col]
                    if len(valid_prices) > 0:
                        preclose = valid_prices.iloc[0]
                        break

            if preclose is None:
                continue

            limit_price = round(preclose * 1.10, 2)

            # 检查是否有涨停
            if 'current' in df_sample.columns:
                max_current = df_sample['current'].max()
                if max_current >= limit_price:
                    limit_up_stocks.append({
                        'code': stock_code,
                        'limit_price': limit_price,
                        'max_current': max_current
                    })

        except Exception:
            continue

    print(f"\n随机检查中发现涨停股票: {len(limit_up_stocks)} 个")
    if limit_up_stocks:
        print(f"涨停股票示例:")
        for stock in limit_up_stocks[:3]:
            print(f"  {stock['code']}: 涨停价 {stock['limit_price']:.2f}, 最高价 {stock['max_current']:.2f}")
    else:
        print(f"❌ 检查的样本中没有涨停股票")
        print(f"这可能表明2025-01-03确实是一个普通交易日")


def check_filter_conditions():
    """检查过滤条件"""
    print(f"\n" + "="*80)
    print("检查过滤条件")
    print("="*80)

    print(f"\n当前过滤条件:")
    print(f"  1. 正样本(触板瞬间): price_T11 >= limit_price AND price_T10 < limit_price")
    print(f"  2. 困难负样本: dist_to_limit < 5%")
    print(f"  3. 自适应放宽: 困难负样本<10个时, 保留中等距离样本(<15%)")
    print(f"  4. 死板状态过滤: 剔除 current >= limit_price 的tick")

    print(f"\n边界过滤:")
    print(f"  - 前9个tick: 无法构建完整窗口")
    print(f"  - 最后1个tick: 无未来标签")

    print(f"\n问题分析:")
    print(f"  1. 如果没有涨停事件, 正样本数 = 0")
    print(f"  2. 如果大部分股票远离涨停, 困难负样本数 = 0")
    print(f"  3. 自适应放宽会保留中等距离样本, 但这些全是负样本")
    print(f"  4. 最终结果: 只有负样本, 没有正样本")


def suggest_improvements():
    """建议改进方案"""
    print(f"\n" + "="*80)
    print("建议改进方案")
    print("="*80)

    print(f"\n问题诊断:")
    print(f"  当前正样本定义过于严格: 要求'触板瞬间'(突破涨停)")
    print(f"  在普通交易日, 这种情况很少发生")

    print(f"\n可选方案:")

    print(f"\n方案1: 放宽正样本定义")
    print(f"  - 将正样本定义为'接近涨停'而非'触板瞬间'")
    print(f"  - 例如: dist_to_limit < 2% 为正样本")
    print(f"  - 优点: 增加正样本数量")
    print(f"  - 缺点: 可能降低预测准确性")

    print(f"\n方案2: 只选择有涨停事件的交易日")
    print(f"  - 预先筛选有涨停股票的交易日")
    print(f"  - 只处理这些交易日")
    print(f"  - 优点: 保证有正样本")
    print(f"  - 缺点: 训练数据量减少")

    print(f"\n方案3: 使用'价格突破'事件作为正样本")
    print(f"  - 将正样本定义为: 价格涨幅>8% 或 接近涨停(<2%)")
    print(f"  - 这样既能保持业务意义, 又有足够样本")
    print(f"  - 优点: 平衡样本数量和质量")
    print(f"  - 缺点: 需要修改标签定义逻辑")

    print(f"\n方案4: 混合训练策略")
    print(f"  - 对有涨停事件的交易日: 使用严格的'触板瞬间'标签")
    print(f"  - 对普通交易日: 使用放宽的'接近涨停'标签")
    print(f"  - 优点: 兼顾不同市场情况")
    print(f"  - 缺点: 实现复杂度较高")


if __name__ == "__main__":
    import pandas as pd

    print("正样本定义检查工具")
    print("="*80)

    try:
        # 检查正样本逻辑
        check_positive_sample_logic()

        # 检查涨停股票情况
        check_limit_up_stock_day()

        # 检查过滤条件
        check_filter_conditions()

        # 建议改进方案
        suggest_improvements()

    except Exception as e:
        print(f"检查过程中出错: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "="*80)
    print("检查完成")
    print("="*80)