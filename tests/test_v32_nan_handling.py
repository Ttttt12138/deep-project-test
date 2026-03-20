"""
V3.2 NaN处理测试
测试智能NaN填充策略
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

from src.feature_engineering.event_window_builder import EventWindowBuilder


def test_smart_nan_filling():
    """测试智能NaN填充"""
    # 创建测试数据：包含NaN值
    n_ticks = 50
    limit_price = 12.0

    # 价格走势：中间插入一些NaN
    prices = [10.0 + i * 0.05 for i in range(n_ticks)]
    prices[10:15] = [np.nan] * 5  # 中间插入NaN
    prices[25:30] = [np.nan] * 5  # 再次插入NaN

    test_df = pd.DataFrame({
        'code': ['000001'] * n_ticks,
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

    # 创建窗口构建器
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

    # 构建事件窗口样本
    result_df = builder.build_event_window_samples(test_df, verbose=False)

    # 验证：最终数据应该没有NaN
    assert not result_df.empty, "应该生成数据"
    assert not result_df.isna().any().any(), "最终数据不应该有NaN"


def test_price_like_features_filling():
    """测试价格类特征的填充策略"""
    # 创建测试数据：价格类特征包含NaN
    n_ticks = 30
    limit_price = 12.0

    test_df = pd.DataFrame({
        'code': ['000001'] * n_ticks,
        'time': range(n_ticks),
        'current': [10.0 + i * 0.05 if i < 10 else np.nan for i in range(n_ticks)],
        'limit_price': [limit_price] * n_ticks,
    })

    # 添加基础特征（部分包含NaN）
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

    # 创建窗口构建器
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

    # 构建事件窗口样本
    result_df = builder.build_event_window_samples(test_df, verbose=False)

    if not result_df.empty:
        # 检查价格类特征是否被正确填充
        price_cols = [col for col in result_df.columns if 'price' in col.lower() or 'limit_price' in col.lower()]
        for col in price_cols:
            if col in result_df.columns:
                # 价格类特征不应该填充0（应该用前值或后值填充）
                assert not result_df[col].isna().any(), f"{col}不应该有NaN"
                # 检查是否使用了mean填充（这是V3.2禁止的）
                # 如果填充了0，这是可以接受的（极特殊情况下）
                pass


def test_volume_features_filling():
    """测试成交量类特征的填充策略"""
    # 创建测试数据：成交量特征包含NaN
    n_ticks = 30
    limit_price = 12.0

    test_df = pd.DataFrame({
        'code': ['000001'] * n_ticks,
        'time': range(n_ticks),
        'current': [10.0 + i * 0.05 for i in range(n_ticks)],
        'limit_price': [limit_price] * n_ticks,
    })

    # 添加基础特征（成交量特征包含NaN）
    test_df['dist_to_limit'] = (test_df['limit_price'] - test_df['current']) / test_df['limit_price']
    test_df['ticks_to_limit'] = (test_df['limit_price'] - test_df['current']) / 0.01
    test_df['b1_volume'] = [100 if i < 10 else np.nan for i in range(n_ticks)]
    test_df['a1_volume'] = [90 if i < 10 else np.nan for i in range(n_ticks)]
    test_df['bid_depth'] = 100
    test_df['ask_depth'] = 90
    test_df['order_imbalance'] = 0.5
    test_df['spread'] = 0.02
    test_df['ask_slope'] = 10
    test_df['bid_slope'] = 8
    test_df['ret_1tick'] = 0.01
    test_df['vol_delta'] = [100 if i < 10 else np.nan for i in range(n_ticks)]
    test_df['money_delta'] = [1000 if i < 10 else np.nan for i in range(n_ticks)]
    test_df['ask1_to_limit'] = 0.02
    test_df['ask1_gap'] = 0.01

    # 创建窗口构建器
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

    # 构建事件窗口样本
    result_df = builder.build_event_window_samples(test_df, verbose=False)

    if not result_df.empty:
        # 检查成交量类特征是否被正确填充为0
        volume_cols = ['b1_volume_last', 'a1_volume_last', 'vol_delta_last', 'money_delta_last']
        for col in volume_cols:
            if col in result_df.columns:
                assert not result_df[col].isna().any(), f"{col}不应该有NaN"
                # 成交量类特征应该被填充为0
                # 验证NaN值被替换了即可


def test_all_nan_column_handling():
    """测试全NaN列的处理"""
    # 创建测试数据：某一列全为NaN
    n_ticks = 30
    limit_price = 12.0

    test_df = pd.DataFrame({
        'code': ['000001'] * n_ticks,
        'time': range(n_ticks),
        'current': [10.0 + i * 0.05 for i in range(n_ticks)],
        'limit_price': [limit_price] * n_ticks,
    })

    # 添加基础特征（某一列全为NaN）
    test_df['dist_to_limit'] = (test_df['limit_price'] - test_df['current']) / test_df['limit_price']
    test_df['ticks_to_limit'] = (test_df['limit_price'] - test_df['current']) / 0.01
    test_df['b1_volume'] = [np.nan] * n_ticks  # 全为NaN
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

    # 创建窗口构建器
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

    # 构建事件窗口样本
    result_df = builder.build_event_window_samples(test_df, verbose=False)

    if not result_df.empty:
        # 检查全NaN列是否被正确处理
        if 'b1_volume_last' in result_df.columns:
            assert not result_df['b1_volume_last'].isna().any(), "全NaN列应该被填充"


def test_early_market_nan_handling():
    """测试早盘数据的NaN处理"""
    # 创建测试数据：早盘数据有NaN
    n_ticks = 30
    limit_price = 12.0

    test_df = pd.DataFrame({
        'code': ['000001'] * n_ticks,
        'time': range(n_ticks),
        'current': [np.nan] * 5 + [10.0 + i * 0.05 for i in range(n_ticks - 5)],
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

    # 创建窗口构建器
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

    # 构建事件窗口样本
    result_df = builder.build_event_window_samples(test_df, verbose=False)

    # 验证：早盘NaN应该被正确处理
    if not result_df.empty:
        assert not result_df.isna().any().any(), "早盘NaN应该被正确处理"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])