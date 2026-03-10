"""
成交特征提取模块
"""

import pandas as pd
import numpy as np
from typing import Dict


def extract_volume_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    提取成交特征

    Args:
        df: 窗口数据DataFrame

    Returns:
        成交特征字典
    """
    features = {}

    # 成交量汇总特征
    features['volume_sum_10s'] = df[df['relative_time'] <= 10]['volume'].sum()
    features['volume_sum_30s'] = df['volume'].sum()
    features['volume_sum_60s'] = df['volume'].sum()

    # 成交额汇总特征
    features['money_sum_10s'] = df[df['relative_time'] <= 10]['money'].sum()
    features['money_sum_30s'] = df['money'].sum()
    features['money_sum_60s'] = df['money'].sum()

    # 平均成交量和成交额
    features['volume_mean'] = df['volume'].mean()
    features['money_mean'] = df['money'].mean()

    # 成交量变化特征
    volumes = df['volume'].values
    time_points = df['relative_time'].values

    if len(time_points) > 1:
        volume_slope = np.polyfit(time_points, volumes, 1)[0]
        features['volume_slope'] = volume_slope
    else:
        features['volume_slope'] = 0.0

    # 成交量标准差
    features['volume_std'] = df['volume'].std()

    # 成交量变化率
    if len(volumes) > 1:
        volume_change_rate = (volumes[-1] - volumes[0]) / volumes[0]
        features['volume_change_rate'] = volume_change_rate
    else:
        features['volume_change_rate'] = 0.0

    # 主动买卖成交特征
    if 'b_s' in df.columns:
        buy_volume = df[df['b_s'] == '买']['volume'].sum()
        sell_volume = df[df['b_s'] == '卖']['volume'].sum()

        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            features['buy_volume_ratio'] = buy_volume / total_volume
            features['sell_volume_ratio'] = sell_volume / total_volume
        else:
            features['buy_volume_ratio'] = 0.5
            features['sell_volume_ratio'] = 0.5
    else:
        features['buy_volume_ratio'] = 0.5
        features['sell_volume_ratio'] = 0.5

    return features


def extract_volume_features_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    向量化的成交特征提取

    Args:
        df: 窗口数据DataFrame

    Returns:
        成交特征Series
    """
    features = extract_volume_features(df)
    return pd.Series(features)