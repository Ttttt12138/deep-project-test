"""
V3.2真实数据失败案例诊断
分析真实数据处理中的失败原因
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_engineering.event_driven_labels import (
    filter_and_label_events,
    get_event_statistics
)
from src.feature_engineering.event_window_builder import EventWindowBuilder


def simulate_failure_case_early_limit():
    """模拟涨停过早的情况"""
    print("="*60)
    print("案例1: 早期涨停 (第12个tick涨停)")
    print("="*60)

    n_ticks = 100
    limit_price = 12.0

    # 价格走势：很早就涨停，然后一直涨停
    prices = [10.0 + i * 0.1 for i in range(12)] + [12.0] * (n_ticks - 12)

    test_df = pd.DataFrame({
        'code': ['TEST001'] * n_ticks,
        'time': range(n_ticks),
        'current': prices,
        'limit_price': [limit_price] * n_ticks,
    })

    # 添加基础特征
    test_df['dist_to_limit'] = (test_df['limit_price'] - test_df['current']) / test_df['limit_price']
    test_df['ticks_to_limit'] = (test_df['limit_price'] - test_df['current']) / 0.01
    test_df['b1_volume'] = 100
    test_df['a1_volume'] = 90
    test_df['bid_depth'] = 100
    test_df['ask_depth'] = 90
    test_df['order_imbalance'] = 0.5
    test_df['spread'] = 0.02
    test_df['ask_slope'] = 10
    test_df['bid_slope'] = 8
    test_df['ret_1tick'] = 0.01
    test_df['vol_delta'] = 100
    test_df['money_delta'] = 1000
    test_df['ask1_to_limit'] = 0.02
    test_df['ask1_gap'] = 0.01

    # 步骤1：标签过滤
    filtered_df, valid_indices = filter_and_label_events(test_df, limit_dist_threshold=0.05, debug=False)

    print(f"原始tick数: {len(test_df)}")
    print(f"过滤后保留: {len(valid_indices)}")

    # 分析触板事件
    touch_events = filtered_df[filtered_df['label'] == 1]
    print(f"触板事件数: {len(touch_events)}")
    if len(touch_events) > 0:
        print(f"触板位置: {touch_events.index.tolist()}")
        for idx in touch_events.index:
            is_preserved = idx in valid_indices
            print(f"触板 {idx} 保留: {is_preserved}")
            if not is_preserved:
                print(f"  失败原因分析:")
                print(f"    dist_to_limit: {filtered_df.loc[idx, 'dist_to_limit']}")
                print(f"    是否<5%: {filtered_df.loc[idx, 'dist_to_limit'] < 0.05}")

    # 步骤2：窗口构建
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.05)
    result_df = builder.build_event_window_samples(filtered_df, verbose=False)

    print(f"窗口样本数: {len(result_df)}")

    if len(result_df) == 0:
        print("[FAILED] 早期涨停案例失败")
        return False
    else:
        print("[SUCCESS] 早期涨停案例成功")
        return True


def simulate_failure_case_no_difficult_negatives():
    """模拟没有困难负样本的情况"""
    print("\n" + "="*60)
    print("案例2: 没有困难负样本")
    print("="*60)

    n_ticks = 100
    limit_price = 12.0

    # 价格走势：大部分时间远离涨停（>5%），偶尔涨停
    prices = []
    for i in range(n_ticks):
        if i < 90:
            price = 10.0 + i * 0.01  # 缓慢上涨，远离涨停
        elif i < 95:
            price = 10.9 + (i - 90) * 0.02  # 接近涨停
        else:
            price = 12.0  # 涨停
        prices.append(price)

    test_df = pd.DataFrame({
        'code': ['TEST002'] * n_ticks,
        'time': range(n_ticks),
        'current': prices,
        'limit_price': [limit_price] * n_ticks,
    })

    # 添加基础特征
    test_df['dist_to_limit'] = (test_df['limit_price'] - test_df['current']) / test_df['limit_price']
    test_df['ticks_to_limit'] = (test_df['limit_price'] - test_df['current']) / 0.01
    test_df['b1_volume'] = 100
    test_df['a1_volume'] = 90
    test_df['bid_depth'] = 100
    test_df['ask_depth'] = 90
    test_df['order_imbalance'] = 0.5
    test_df['spread'] = 0.02
    test_df['ask_slope'] = 10
    test_df['bid_slope'] = 8
    test_df['ret_1tick'] = 0.01
    test_df['vol_delta'] = 100
    test_df['money_delta'] = 1000
    test_df['ask1_to_limit'] = 0.02
    test_df['ask1_gap'] = 0.01

    # 分析距离分布
    print(f"距离<5%的tick数: {(test_df['dist_to_limit'] < 0.05).sum()}")
    print(f"距离分布: min={test_df['dist_to_limit'].min():.6f}, max={test_df['dist_to_limit'].max():.6f}")

    # 标签过滤
    filtered_df, valid_indices = filter_and_label_events(test_df, limit_dist_threshold=0.05, debug=False)

    print(f"过滤后保留: {len(valid_indices)}")

    if len(valid_indices) == 0:
        print("[FAILED] 无困难负样本案例失败（符合预期）")
        return False
    else:
        print("[SUCCESS] 有困难负样本保留")
        return True


def simulate_failure_case_short_data():
    """模拟数据量不足的情况"""
    print("\n" + "="*60)
    print("案例3: 数据量不足")
    print("="*60)

    n_ticks = 8  # 只有8个tick
    limit_price = 12.0

    # 价格走势：快速涨停
    prices = [10.0 + i * 0.25 for i in range(n_ticks)]

    test_df = pd.DataFrame({
        'code': ['TEST003'] * n_ticks,
        'time': range(n_ticks),
        'current': prices,
        'limit_price': [limit_price] * n_ticks,
    })

    # 添加基础特征
    test_df['dist_to_limit'] = (test_df['limit_price'] - test_df['current']) / test_df['limit_price']
    test_df['ticks_to_limit'] = (test_df['limit_price'] - test_df['current']) / 0.01
    test_df['b1_volume'] = 100
    test_df['a1_volume'] = 90
    test_df['bid_depth'] = 100
    test_df['ask_depth'] = 90
    test_df['order_imbalance'] = 0.5
    test_df['spread'] = 0.02
    test_df['ask_slope'] = 10
    test_df['bid_slope'] = 8
    test_df['ret_1tick'] = 0.01
    test_df['vol_delta'] = 100
    test_df['money_delta'] = 1000
    test_df['ask1_to_limit'] = 0.02
    test_df['ask1_gap'] = 0.01

    print(f"原始tick数: {len(test_df)}")

    # 标签过滤
    filtered_df, valid_indices = filter_and_label_events(test_df, limit_dist_threshold=0.05, debug=False)

    print(f"过滤后保留: {len(valid_indices)}")

    # 窗口构建
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.05)
    result_df = builder.build_event_window_samples(filtered_df, verbose=False)

    print(f"窗口样本数: {len(result_df)}")

    if len(result_df) == 0:
        print("[FAILED] 数据不足案例失败（可能是因为窗口大小限制）")
        return False
    else:
        print("[SUCCESS] 数据不足案例成功")
        return True


def simulate_realistic_case():
    """模拟更真实的案例"""
    print("\n" + "="*60)
    print("案例4: 更真实的股市数据")
    print("="*60)

    n_ticks = 200
    limit_price = 12.0

    # 更真实的价格走势：震荡上涨，多次尝试涨停
    np.random.seed(42)
    prices = []
    for i in range(n_ticks):
        if i < 50:
            base_price = 10.0 + i * 0.02  # 缓慢上涨
        elif i < 100:
            base_price = 11.0 + (i - 50) * 0.01  # 继续上涨
        else:
            base_price = 11.5  # 基础价格

        # 添加随机波动
        noise = np.random.uniform(-0.1, 0.1)
        price = max(base_price + noise, 9.0)  # 确保不低于9.0
        prices.append(price)

    test_df = pd.DataFrame({
        'code': ['TEST004'] * n_ticks,
        'time': range(n_ticks),
        'current': prices,
        'limit_price': [limit_price] * n_ticks,
    })

    # 添加基础特征
    test_df['dist_to_limit'] = (test_df['limit_price'] - test_df['current']) / test_df['limit_price']
    test_df['ticks_to_limit'] = (test_df['limit_price'] - test_df['current']) / 0.01
    test_df['b1_volume'] = 100
    test_df['a1_volume'] = 90
    test_df['bid_depth'] = 100
    test_df['ask_depth'] = 90
    test_df['order_imbalance'] = 0.5
    test_df['spread'] = 0.02
    test_df['ask_slope'] = 10
    test_df['bid_slope'] = 8
    test_df['ret_1tick'] = 0.01
    test_df['vol_delta'] = 100
    test_df['money_delta'] = 1000
    test_df['ask1_to_limit'] = 0.02
    test_df['ask1_gap'] = 0.01

    print(f"距离<5%的tick数: {(test_df['dist_to_limit'] < 0.05).sum()}")

    # 标签过滤
    filtered_df, valid_indices = filter_and_label_events(test_df, limit_dist_threshold=0.05, debug=False)

    print(f"过滤后保留: {len(valid_indices)}")

    # 窗口构建
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.05)
    result_df = builder.build_event_window_samples(filtered_df, verbose=False)

    print(f"窗口样本数: {len(result_df)}")

    if len(result_df) == 0:
        print("[FAILED] 真实案例失败")
        return False
    else:
        print("[SUCCESS] 真实案例成功")
        return True


if __name__ == "__main__":
    results = []

    results.append(simulate_failure_case_early_limit())
    results.append(simulate_failure_case_no_difficult_negatives())
    results.append(simulate_failure_case_short_data())
    results.append(simulate_realistic_case())

    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    print(f"成功案例数: {sum(results)}/{len(results)}")

    if sum(results) < len(results):
        print("\n失败原因分析:")
        print("1. 如果早期涨停案例失败：窗口构建问题")
        print("2. 如果无困难负样本案例失败：阈值设置合理")
        print("3. 如果数据不足案例失败：窗口大小限制")
        print("4. 如果真实案例失败：数据质量问题")
    else:
        print("\n所有模拟案例都成功，问题可能在真实数据处理环节")