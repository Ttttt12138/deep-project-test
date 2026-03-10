"""
标签生成模块
"""

import pandas as pd
import numpy as np
from typing import Dict


def generate_return_labels(df: pd.DataFrame,
                          future_period: int = 30) -> Dict[str, float]:
    """
    生成收益率标签

    Args:
        df: 窗口数据DataFrame
        future_period: 未来周期(秒)

    Returns:
        收益率标签字典
    """
    labels = {}

    current_price = df['current'].iloc[-1]

    # 计算未来价格 (简单模拟，实际应用中需要真实未来数据)
    # 这里我们使用窗口最后价格加上一个小的随机变化来模拟
    np.random.seed(42)  # 固定随机种子以保证可重复性
    price_change = np.random.normal(0, 0.001)  # 0.1%的波动
    future_price = current_price * (1 + price_change)

    # 计算收益率
    return_rate = (future_price - current_price) / current_price

    if future_period == 30:
        labels['y_return_30s'] = return_rate
    elif future_period == 60:
        labels['y_return_60s'] = return_rate

    return labels


def generate_breakout_label(df: pd.DataFrame,
                           future_period: int = 60) -> Dict[str, int]:
    """
    生成突破标签

    Args:
        df: 窗口数据DataFrame
        future_period: 未来周期(秒)

    Returns:
        突破标签字典
    """
    labels = {}

    # 计算过去窗口的最高价
    max_past_price = df['current'].max()

    # 模拟未来价格 (简单模拟，实际应用中需要真实未来数据)
    current_price = df['current'].iloc[-1]
    np.random.seed(42)
    future_price_max = current_price * (1 + np.random.uniform(0, 0.01))  # 0-1%的上涨

    # 判断是否突破
    breakout = 1 if future_price_max > max_past_price else 0

    if future_period == 60:
        labels['y_breakout_60s'] = breakout

    return labels


def generate_all_labels(df: pd.DataFrame) -> Dict:
    """
    生成所有标签

    Args:
        df: 窗口数据DataFrame

    Returns:
        所有标签字典
    """
    labels = {}

    # 收益率标签
    return_labels_30s = generate_return_labels(df, future_period=30)
    return_labels_60s = generate_return_labels(df, future_period=60)
    labels.update(return_labels_30s)
    labels.update(return_labels_60s)

    # 突破标签
    breakout_label = generate_breakout_label(df, future_period=60)
    labels.update(breakout_label)

    return labels


def generate_labels_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    向量化的标签生成

    Args:
        df: 窗口数据DataFrame

    Returns:
        标签Series
    """
    labels = generate_all_labels(df)
    return pd.Series(labels)


def get_label_definitions() -> Dict[str, str]:
    """
    获取标签定义

    Returns:
        标签定义字典
    """
    return {
        'y_return_30s': '未来30秒收益率',
        'y_return_60s': '未来60秒收益率',
        'y_breakout_60s': '未来60秒突破标签(1/0)'
    }