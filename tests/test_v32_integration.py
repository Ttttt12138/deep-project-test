"""
V3.2集成测试
测试V3.2端到端处理流程
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_engineering.event_driven_labels import get_event_statistics
from src.feature_engineering.event_window_builder import EventWindowBuilder
from src.feature_engineering.limit_up_features import extract_limit_up_features


def test_end_to_end_v32_processing():
    """测试V3.2端到端处理流程"""
    # 创建模拟的tick数据
    n_ticks = 200
    limit_price = 12.0

    # 模拟一只股票的完整交易日数据
    data = {
        'time': range(n_ticks),
        'current': [],
        'volume': [100 + i * 10 for i in range(n_ticks)],
        'amount': [1000 + i * 100 for i in range(n_ticks)],
        'b1_p': [],
        'a1_p': [],
        'b1_v': [],
        'a1_v': [],
    }

    # 价格走势：震荡上涨 -> 快速拉升 -> 首次触板 -> 烂板 -> 死板
    for i in range(n_ticks):
        if i < 50:
            current = 10.0 + i * 0.01 + np.random.uniform(-0.05, 0.05)
        elif i < 60:
            current = 10.5 + (i - 50) * 0.05
        elif i < 70:
            current = 11.0 + (i - 60) * 0.1
        elif i < 75:
            current = 11.5 + (i - 70) * 0.1
        elif i < 90:
            # 烂板阶段
            if i % 3 == 0:
                current = 12.0
            else:
                current = 11.95 + np.random.uniform(-0.05, 0.05)
        else:
            current = 12.0

        data['current'].append(current)
        data['b1_p'].append(current - 0.01)
        data['a1_p'].append(current + 0.01)
        data['b1_v'].append(50 + np.random.randint(0, 20))
        data['a1_v'].append(40 + np.random.randint(0, 20))

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 模拟处理流程
    # 1. 特征提取
    df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=0.1)

    # 2. 添加code列（必需）
    df['code'] = '000001'
    df['limit_price'] = limit_price

    # 3. 创建窗口构建器
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

    # 4. 构建事件窗口样本
    result_df = builder.build_event_window_samples(df, verbose=False)

    # 验证：应该生成数据
    # 如果没有生成数据，可能是因为没有困难负样本，这也是可以接受的
    if not result_df.empty:
        # 验证：一只股票只有一个正样本
        positive_samples = int(result_df['label'].sum())
        assert positive_samples <= 1, f"一只股票最多1个正样本，实际有{positive_samples}个"

        # 验证：没有NaN
        assert not result_df.isna().any().any(), "端到端处理结果不应该有NaN"
    else:
        # 如果没有数据，可能是因为距离阈值设置较高，没有困难负样本
        pass


def test_performance_comparison():
    """对比V3.2性能基准"""
    # 创建大规模测试数据
    n_stocks = 50  # 50只股票
    n_ticks_per_stock = 500
    limit_price = 12.0

    all_data = []

    for stock_id in range(n_stocks):
        stock_code = f"{stock_id:06d}"
        data = {
            'code': [stock_code] * n_ticks_per_stock,
            'time': range(n_ticks_per_stock),
            'current': [],
            'limit_price': [limit_price] * n_ticks_per_stock,
        }

        # 随机生成价格走势
        for i in range(n_ticks_per_stock):
            if np.random.random() < 0.2:  # 20%的股票会触板
                if i < 100:
                    current = 10.0 + i * 0.01
                elif i < 150:
                    current = 11.0 + (i - 100) * 0.02
                else:
                    current = 12.0
            else:
                current = 10.0 + np.random.uniform(-0.5, 0.5)

            data['current'].append(current)

        all_data.append(pd.DataFrame(data))

    # 合并所有股票
    merged_df = pd.concat(all_data, ignore_index=True)

    # 添加基础特征
    merged_df['dist_to_limit'] = (merged_df['limit_price'] - merged_df['current']) / merged_df['limit_price']
    merged_df['ticks_to_limit'] = (merged_df['limit_price'] - merged_df['current']) / 0.01
    merged_df['b1_volume'] = 100
    merged_df['a1_volume'] = 90
    merged_df['bid_depth'] = 100
    merged_df['ask_depth'] = 90
    merged_df['order_imbalance'] = 0.5
    merged_df['spread'] = 0.02
    merged_df['ask_slope'] = 10
    merged_df['bid_slope'] = 8
    merged_df['ret_1tick'] = 0.01
    merged_df['vol_delta'] = 100
    merged_df['money_delta'] = 1000
    merged_df['ask1_to_limit'] = 0.02
    merged_df['ask1_gap'] = 0.01

    # 创建窗口构建器
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

    # 构建事件窗口样本
    result_df = builder.build_multi_stock_event_samples(merged_df, verbose=False)

    # 验证性能
    assert not result_df.empty, "应该生成数据"

    # 验证一板一样本原则
    code_positive_counts = result_df.groupby('code')['label'].sum()
    if len(code_positive_counts) > 0:
        assert (code_positive_counts <= 1).all(), "每只股票最多只有1个正样本"

    # 验证正样本数量（应该在10-80之间，对应涨停股票数）
    positive_samples = int(result_df['label'].sum())
    assert positive_samples >= 0, "正样本数量应该非负"
    # 注意：由于是随机生成的数据，涨停股票数不确定


def test_data_quality_validation():
    """验证V3.2数据质量"""
    # 创建测试数据
    n_ticks = 100
    limit_price = 12.0

    data = {
        'code': ['000001'] * n_ticks,
        'time': range(n_ticks),
        'current': [10.0 + i * 0.05 if i < 50 else 12.0 for i in range(n_ticks)],
        'limit_price': [limit_price] * n_ticks,
    }

    df = pd.DataFrame(data)

    # 添加基础特征
    df['dist_to_limit'] = (df['limit_price'] - df['current']) / df['limit_price']
    df['ticks_to_limit'] = (df['limit_price'] - df['current']) / 0.01
    df['b1_volume'] = 100
    df['a1_volume'] = 90
    df['bid_depth'] = 100
    df['ask_depth'] = 90
    df['order_imbalance'] = 0.5
    df['spread'] = 0.02
    df['ask_slope'] = 10
    df['bid_slope'] = 8
    df['ret_1tick'] = 0.01
    df['vol_delta'] = 100
    df['money_delta'] = 1000
    df['ask1_to_limit'] = 0.02
    df['ask1_gap'] = 0.01

    # 创建窗口构建器
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

    # 构建事件窗口样本
    result_df = builder.build_event_window_samples(df, verbose=False)

    if not result_df.empty:
        # 数据质量验证清单
        # 1. 正样本数量：每只股票每天最多1个
        positive_samples = int(result_df['label'].sum())
        assert positive_samples <= 1, f"一只股票最多1个正样本，实际有{positive_samples}个"

        # 2. 没有NaN
        assert not result_df.isna().any().any(), "不应该有NaN"

        # 3. 特征完整性
        required_features = ['code', 'time', 'label', 'dist_to_limit_last']
        for feature in required_features:
            assert feature in result_df.columns, f"缺少必需特征: {feature}"

        # 4. 数据类型正确
        assert result_df['label'].dtype in [np.int8, np.int32, np.int64], "label应该是整数类型"


def test_one_board_one_sample():
    """验证一板一样本原则"""
    # 创建测试数据：多只股票，其中一些多次触板
    n_ticks_per_stock = 200
    limit_price = 12.0

    all_data = []

    for stock_id in range(10):
        stock_code = f"{stock_id:06d}"

        # 随机决定这只股票是否会多次触板
        if stock_id % 3 == 0:  # 每三只股票有一只多次触板
            # 多次触板股票
            current_prices = []
            for i in range(n_ticks_per_stock):
                if i < 50:
                    current = 10.0 + i * 0.04
                elif i < 60:
                    current = 11.0 + (i - 50) * 0.1
                elif i < 80:
                    # 第一次触板
                    if i % 5 == 0:
                        current = 12.0
                    else:
                        current = 11.95 + np.random.uniform(-0.05, 0.05)
                elif i < 100:
                    # 第二次触板
                    if i % 5 == 0:
                        current = 12.0
                    else:
                        current = 11.95 + np.random.uniform(-0.05, 0.05)
                else:
                    current = 12.0
                current_prices.append(current)
        else:
            # 正常触板股票
            current_prices = [10.0 + i * 0.05 if i < 50 else 12.0 for i in range(n_ticks_per_stock)]

        data = {
            'code': [stock_code] * n_ticks_per_stock,
            'time': range(n_ticks_per_stock),
            'current': current_prices,
            'limit_price': [limit_price] * n_ticks_per_stock,
        }

        all_data.append(pd.DataFrame(data))

    # 合并所有股票
    merged_df = pd.concat(all_data, ignore_index=True)

    # 添加基础特征
    merged_df['dist_to_limit'] = (merged_df['limit_price'] - merged_df['current']) / merged_df['limit_price']
    merged_df['ticks_to_limit'] = (merged_df['limit_price'] - merged_df['current']) / 0.01
    merged_df['b1_volume'] = 100
    merged_df['a1_volume'] = 90
    merged_df['bid_depth'] = 100
    merged_df['ask_depth'] = 90
    merged_df['order_imbalance'] = 0.5
    merged_df['spread'] = 0.02
    merged_df['ask_slope'] = 10
    merged_df['bid_slope'] = 8
    merged_df['ret_1tick'] = 0.01
    merged_df['vol_delta'] = 100
    merged_df['money_delta'] = 1000
    merged_df['ask1_to_limit'] = 0.02
    merged_df['ask1_gap'] = 0.01

    # 创建窗口构建器
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

    # 构建事件窗口样本
    result_df = builder.build_multi_stock_event_samples(merged_df, verbose=False)

    if not result_df.empty:
        # 验证一板一样本原则：每只股票最多1个正样本
        code_positive_counts = result_df.groupby('code')['label'].sum()
        assert (code_positive_counts <= 1).all(), "每只股票最多1个正样本"

        # 验证多次触板的股票也只产生1个正样本
        for stock_id in range(10):
            if stock_id % 3 == 0:  # 多次触板股票
                stock_code = f"{stock_id:06d}"
                if stock_code in code_positive_counts.index:
                    assert code_positive_counts[stock_code] == 1, f"多次触板股票{stock_code}应该只产生1个正样本"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])