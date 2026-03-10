"""
价格特征提取模块
"""

import pandas as pd
import numpy as np
from typing import Dict


def calculate_return(df: pd.DataFrame,
                     period_seconds: int,
                     base_price: float = None) -> float:
    """
    计算指定周期的收益率

    Args:
        df: 窗口数据DataFrame
        period_seconds: 周期(秒)
        base_price: 基准价格，默认为窗口起始价格

    Returns:
        收益率
    """
    if base_price is None:
        base_price = df['current'].iloc[0]

    # 获取周期结束时的价格
    end_price = df[df['relative_time'] <= period_seconds]['current'].iloc[-1]

    return (end_price - base_price) / base_price


def extract_price_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    提取价格特征

    Args:
        df: 窗口数据DataFrame

    Returns:
        价格特征字典
    """
    features = {}

    # 基础价格信息
    features['price_start'] = df['current'].iloc[0]
    features['price_end'] = df['current'].iloc[-1]
    features['price_mean'] = df['current'].mean()
    features['price_std'] = df['current'].std()

    # 收益率特征
    features['return_5s'] = calculate_return(df, 5)
    features['return_10s'] = calculate_return(df, 10)
    features['return_30s'] = calculate_return(df, 30)
    features['return_60s'] = calculate_return(df, 60)

    # 波动率特征
    features['volatility_30s'] = df['current'].std()
    features['volatility_60s'] = df['current'].std()

    # 区间特征
    features['high_60s'] = df['current'].max()
    features['low_60s'] = df['current'].min()
    features['range_60s'] = features['high_60s'] - features['low_60s']

    # 价格动量
    features['price_momentum'] = (df['current'].iloc[-1] - df['current'].iloc[0]) / df['current'].iloc[0]

    return features


def extract_price_features_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    向量化的价格特征提取

    Args:
        df: 窗口数据DataFrame

    Returns:
        价格特征Series
    """
    features = extract_price_features(df)
    return pd.Series(features)