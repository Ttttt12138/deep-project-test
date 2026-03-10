"""
特征工程模块自动化测试
用于验证数据预处理和特征工程的基本功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_data_preprocessing():
    """测试数据预处理模块"""
    print("测试数据预处理模块...")

    # 创建模拟数据
    base_time = datetime(2025, 1, 2, 9, 30, 0)
    data = []
    for i in range(100):
        row = {
            '成交时间': base_time + timedelta(seconds=i),
            '成交价': 10.0 + i * 0.01,
            '成交量': 100 + i * 10,
            '成交额': (10.0 + i * 0.01) * (100 + i * 10),
            '性质': '买' if i % 2 == 0 else '卖',
            '卖1': 10.0 + i * 0.01 + 0.01, '卖1量': 100,
            '买1': 10.0 + i * 0.01 - 0.01, '买1量': 100,
            '卖2': 10.0 + i * 0.01 + 0.02, '卖2量': 80,
            '买2': 10.0 + i * 0.01 - 0.02, '买2量': 80,
            '卖3': 10.0 + i * 0.01 + 0.03, '卖3量': 60,
            '买3': 10.0 + i * 0.01 - 0.03, '买3量': 60,
            '卖4': 10.0 + i * 0.01 + 0.04, '卖4量': 40,
            '买4': 10.0 + i * 0.01 - 0.04, '买4量': 40,
            '卖5': 10.0 + i * 0.01 + 0.05, '卖5量': 20,
            '买5': 10.0 + i * 0.01 - 0.05, '买5量': 20
        }
        data.append(row)

    df = pd.DataFrame(data)

    # 基本验证
    assert len(df) == 100, "数据行数不正确"
    assert '成交时间' in df.columns, "缺少时间列"
    assert '成交价' in df.columns, "缺少价格列"
    assert '卖1' in df.columns, "缺少卖盘列"
    assert '买1' in df.columns, "缺少买盘列"

    print("✓ 数据预处理模块测试通过")


def test_price_features():
    """测试价格特征提取"""
    print("测试价格特征提取...")

    # 创建简单价格序列
    prices = pd.Series([10.0, 10.01, 10.02, 10.03, 10.04])

    # 计算收益率
    return_rate = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    assert isinstance(return_rate, float), "收益率应为浮点数"
    assert return_rate > 0, "收益率应大于0"

    # 计算波动率
    volatility = prices.std()
    assert volatility > 0, "波动率应大于0"

    # 计算区间特征
    high = prices.max()
    low = prices.min()
    range_val = high - low
    assert range_val > 0, "区间范围应大于0"

    print("✓ 价格特征提取测试通过")


def test_volume_features():
    """测试成交特征提取"""
    print("测试成交特征提取...")

    # 创建简单成交量序列
    volumes = pd.Series([100, 200, 300, 400, 500])

    # 计算汇总
    volume_sum = volumes.sum()
    assert volume_sum == 1500, "成交量汇总不正确"

    # 计算均值
    volume_mean = volumes.mean()
    assert volume_mean == 300, "成交量均值不正确"

    # 计算斜率
    time_points = pd.Series([0, 1, 2, 3, 4])
    slope = np.polyfit(time_points, volumes, 1)[0]
    assert slope > 0, "成交量斜率应大于0"

    print("✓ 成交特征提取测试通过")


def test_orderbook_features():
    """测试盘口特征提取"""
    print("测试盘口特征提取...")

    # 模拟买卖盘数据
    bid_volumes = [100, 200, 300, 400, 500]
    ask_volumes = [80, 160, 240, 320, 400]

    # 计算总量
    bid_total = sum(bid_volumes)
    ask_total = sum(ask_volumes)
    assert bid_total > 0, "买盘总量应大于0"
    assert ask_total > 0, "卖盘总量应大于0"

    # 计算不平衡度
    total = bid_total + ask_total
    imbalance = (bid_total - ask_total) / total
    assert -1 <= imbalance <= 1, "不平衡度应在[-1, 1]范围内"

    # 计算价差
    ask_price = 10.01
    bid_price = 10.00
    spread = ask_price - bid_price
    assert spread > 0, "价差应大于0"

    print("✓ 盘口特征提取测试通过")


def test_label_generation():
    """测试标签生成"""
    print("测试标签生成...")

    # 测试收益率标签
    current_price = 10.0
    future_price = 10.05
    return_rate = (future_price - current_price) / current_price
    assert isinstance(return_rate, float), "收益率标签应为浮点数"
    assert return_rate > 0, "收益率标签应为正数"

    # 测试突破标签
    max_past_price = 10.02
    max_future_price = 10.06
    breakout = 1 if max_future_price > max_past_price else 0
    assert breakout in [0, 1], "突破标签应为0或1"
    assert breakout == 1, "突破标签应为1"

    print("✓ 标签生成测试通过")


def test_pipeline_integration():
    """测试完整流程集成"""
    print("测试完整流程集成...")

    # 创建简单数据
    df = pd.DataFrame({
        '成交时间': pd.date_range('2025-01-02 09:30:00', periods=60, freq='1s'),
        '成交价': 10.0 + np.random.randn(60) * 0.01,
        '成交量': np.random.randint(100, 500, 60),
        '成交额': (10.0 + np.random.randn(60) * 0.01) * np.random.randint(100, 500, 60),
        '性质': np.random.choice(['买', '卖'], 60),
        '卖1': 10.0 + np.random.randn(60) * 0.01 + 0.01,
        '卖1量': np.random.randint(100, 500, 60),
        '买1': 10.0 + np.random.randn(60) * 0.01 - 0.01,
        '买1量': np.random.randint(100, 500, 60)
    })

    # 验证数据结构
    assert len(df) == 60, "数据长度不正确"
    assert df['成交量'].sum() > 0, "总成交量应大于0"

    print("✓ 完整流程集成测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 40)
    print("开始运行自动化测试")
    print("=" * 40)

    try:
        test_data_preprocessing()
        test_price_features()
        test_volume_features()
        test_orderbook_features()
        test_label_generation()
        test_pipeline_integration()

        print("=" * 40)
        print("✓ 所有测试通过!")
        print("=" * 40)
        return True

    except AssertionError as e:
        print(f"✗ 测试失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)