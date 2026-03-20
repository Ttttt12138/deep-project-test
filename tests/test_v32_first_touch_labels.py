"""
V3.2首次触板标签测试
测试一板一样本的核心逻辑
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_engineering.event_driven_labels import (
    generate_event_driven_label,
    filter_and_label_events,
    get_event_statistics
)


def test_first_touch_extraction():
    """测试首次触板提取逻辑"""
    # 创建测试数据：模拟一只股票多次触板过程
    n_ticks = 100
    limit_price = 12.0

    # 价格走势：缓慢上涨 -> 快速拉升 -> 首次触板 -> 烂板（多次打开回封） -> 死板
    prices = []
    for i in range(n_ticks):
        if i < 30:
            # 缓慢上涨阶段
            price = 10.0 + i * 0.05
        elif i < 35:
            # 快速拉升阶段：首次触板
            price = 11.5 + (i - 30) * 0.1
        elif i < 45:
            # 烂板阶段：反复打开又回封（模拟多次触板）
            price = 11.95 + np.random.uniform(-0.05, 0.05)
            if i % 2 == 0:
                price = 12.0
        else:
            # 死板阶段
            price = 12.0
        prices.append(price)

    test_df = pd.DataFrame({
        'code': ['000001'] * n_ticks,
        'time': range(n_ticks),
        'current': prices,
        'limit_price': [limit_price] * n_ticks,
    })

    # 生成V3.2首次触板标签
    labeled_df = generate_event_driven_label(test_df)

    # 验证：一只股票一天只有一个正样本
    positive_samples = int(labeled_df['label'].sum())
    assert positive_samples == 1, f"应该只有1个正样本，实际有{positive_samples}个"


def test_multiple_touches_scenario():
    """测试多次触板场景的处理"""
    # 创建测试数据：模拟一只股票烂板30次
    n_ticks = 200
    limit_price = 12.0

    # 价格走势：反复打开又回封
    prices = []
    for i in range(n_ticks):
        if i < 40:
            price = 10.0 + i * 0.05
        elif i < 50:
            price = 11.5 + (i - 40) * 0.05
        elif i < 110:
            # 烂板阶段：30次触板
            if i % 3 == 0:
                price = 12.0  # 触板
            else:
                price = 11.95 + np.random.uniform(-0.05, 0.05)
        else:
            price = 12.0

        prices.append(price)

    test_df = pd.DataFrame({
        'code': ['000001'] * n_ticks,
        'time': range(n_ticks),
        'current': prices,
        'limit_price': [limit_price] * n_ticks,
    })

    # 执行V3.2过滤
    filtered_df, valid_indices = filter_and_label_events(test_df, limit_dist_threshold=0.02)

    # 验证：一只股票烂板30次，但只有1个正样本
    positive_samples = int(filtered_df.loc[valid_indices, 'label'].sum())
    assert positive_samples == 1, f"烂板30次应该只产生1个正样本，实际有{positive_samples}个"


def test_no_touch_scenario():
    """测试无触板场景的处理"""
    # 创建测试数据：模拟一只股票从未触板
    n_ticks = 100
    limit_price = 12.0

    # 价格走势：一直在涨停价之下
    prices = [10.0 + i * 0.01 for i in range(n_ticks)]
    # 确保最高价也不会触碰涨停价
    prices = [min(p, 11.8) for p in prices]

    test_df = pd.DataFrame({
        'code': ['000001'] * n_ticks,
        'time': range(n_ticks),
        'current': prices,
        'limit_price': [limit_price] * n_ticks,
    })

    # 执行V3.2过滤
    filtered_df, valid_indices = filter_and_label_events(test_df, limit_dist_threshold=0.02)

    # 验证：没有正样本
    positive_samples = int(filtered_df.loc[valid_indices, 'label'].sum())
    assert positive_samples == 0, f"无触板场景应该没有正样本，实际有{positive_samples}个"


def test_single_stock_multiple_positives():
    """验证一只股票一天最多产生1个正样本"""
    # 创建测试数据：模拟一只股票有多次突破
    n_ticks = 300
    limit_price = 12.0

    # 价格走势：多次突破涨停价
    prices = []
    for i in range(n_ticks):
        if i < 40:
            price = 10.0 + i * 0.05
        elif i < 50:
            price = 11.5 + (i - 40) * 0.05
        elif i < 60:
            price = 12.0
        elif i < 80:
            price = 11.8 + np.random.uniform(-0.05, 0.05)
        elif i < 90:
            price = 11.9 + (i - 80) * 0.01
        elif i < 100:
            price = 12.0
        elif i < 120:
            price = 11.8 + np.random.uniform(-0.05, 0.05)
        elif i < 130:
            price = 11.9 + (i - 120) * 0.01
        else:
            price = 12.0

        prices.append(price)

    test_df = pd.DataFrame({
        'code': ['000001'] * n_ticks,
        'time': range(n_ticks),
        'current': prices,
        'limit_price': [limit_price] * n_ticks,
    })

    # 生成V3.2首次触板标签
    labeled_df = generate_event_driven_label(test_df)

    # 验证：即使有多次突破，也只保留首次突破
    positive_samples = int(labeled_df['label'].sum())
    assert positive_samples == 1, f"多次突破应该只保留首次突破，实际有{positive_samples}个正样本"


def test_first_touch_position():
    """测试首次触板的位置是否正确"""
    # 创建测试数据：首次触板在tick 35
    n_ticks = 100
    limit_price = 12.0

    prices = []
    for i in range(n_ticks):
        if i < 30:
            price = 10.0 + i * 0.05
        elif i < 35:
            price = 11.5 + (i - 30) * 0.1  # 第35个tick达到涨停
        else:
            price = 12.0
        prices.append(price)

    test_df = pd.DataFrame({
        'code': ['000001'] * n_ticks,
        'time': range(n_ticks),
        'current': prices,
        'limit_price': [limit_price] * n_ticks,
    })

    # 生成V3.2首次触板标签
    labeled_df = generate_event_driven_label(test_df)

    # 找到正样本的位置
    positive_indices = labeled_df[labeled_df['label'] == 1].index.tolist()

    # 验证：正样本应该在tick 34（因为shift(-1)检测下一秒）
    assert len(positive_indices) == 1, f"应该有1个正样本，实际有{len(positive_indices)}个"
    assert positive_indices[0] == 34, f"首次触板应该在tick 34，实际在{positive_indices[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])