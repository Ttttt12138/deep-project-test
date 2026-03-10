"""
盘口特征提取模块
"""

import pandas as pd
import numpy as np
from typing import Dict


def extract_orderbook_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    提取盘口特征

    Args:
        df: 窗口数据DataFrame

    Returns:
        盘口特征字典
    """
    features = {}

    # 买卖盘总量
    bid_volume_total = 0
    ask_volume_total = 0

    for i in range(1, 6):
        bid_col = f'b{i}_v'
        ask_col = f'a{i}_v'

        if bid_col in df.columns and ask_col in df.columns:
            bid_volume_total += df[bid_col].sum()
            ask_volume_total += df[ask_col].sum()

    features['bid_volume_total'] = bid_volume_total
    features['ask_volume_total'] = ask_volume_total

    # 盘口不平衡特征
    total_volume = bid_volume_total + ask_volume_total
    if total_volume > 0:
        features['order_diff'] = bid_volume_total - ask_volume_total
        features['order_imbalance'] = features['order_diff'] / total_volume
    else:
        features['order_diff'] = 0.0
        features['order_imbalance'] = 0.0

    # 买卖价差特征
    if 'a1_p' in df.columns and 'b1_p' in df.columns:
        features['spread'] = (df['a1_p'] - df['b1_p']).mean()
        features['spread_relative'] = features['spread'] / df['current'].mean()

        # 价差波动
        features['spread_std'] = (df['a1_p'] - df['b1_p']).std()
    else:
        features['spread'] = 0.0
        features['spread_relative'] = 0.0
        features['spread_std'] = 0.0

    # 盘口深度斜率特征
    bid_depths = []
    ask_depths = []

    for i in range(1, 6):
        bid_col = f'b{i}_v'
        ask_col = f'a{i}_v'

        if bid_col in df.columns:
            bid_depths.append(df[bid_col].mean())
        if ask_col in df.columns:
            ask_depths.append(df[ask_col].mean())

    if len(bid_depths) > 1:
        bid_depth_slope = np.polyfit(range(len(bid_depths)), bid_depths, 1)[0]
        features['bid_depth_slope'] = bid_depth_slope
    else:
        features['bid_depth_slope'] = 0.0

    if len(ask_depths) > 1:
        ask_depth_slope = np.polyfit(range(len(ask_depths)), ask_depths, 1)[0]
        features['ask_depth_slope'] = ask_depth_slope
    else:
        features['ask_depth_slope'] = 0.0

    # 买卖盘平均深度
    features['bid_depth_mean'] = np.mean(bid_depths) if bid_depths else 0.0
    features['ask_depth_mean'] = np.mean(ask_depths) if ask_depths else 0.0

    # 盘口压力指标
    if total_volume > 0:
        features['bid_pressure'] = bid_volume_total / total_volume
        features['ask_pressure'] = ask_volume_total / total_volume
    else:
        features['bid_pressure'] = 0.5
        features['ask_pressure'] = 0.5

    # 第一档买卖盘比率
    if 'b1_v' in df.columns and 'a1_v' in df.columns:
        b1_mean = df['b1_v'].mean()
        a1_mean = df['a1_v'].mean()

        if (b1_mean + a1_mean) > 0:
            features['level1_ratio'] = b1_mean / (b1_mean + a1_mean)
        else:
            features['level1_ratio'] = 0.5
    else:
        features['level1_ratio'] = 0.5

    # 盘口集中度
    if 'b1_v' in df.columns:
        b1_total = df['b1_v'].sum()
        features['bid_concentration'] = b1_total / bid_volume_total if bid_volume_total > 0 else 0.0
    else:
        features['bid_concentration'] = 0.0

    if 'a1_v' in df.columns:
        a1_total = df['a1_v'].sum()
        features['ask_concentration'] = a1_total / ask_volume_total if ask_volume_total > 0 else 0.0
    else:
        features['ask_concentration'] = 0.0

    return features


def extract_orderbook_features_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    向量化的盘口特征提取

    Args:
        df: 窗口数据DataFrame

    Returns:
        盘口特征Series
    """
    features = extract_orderbook_features(df)
    return pd.Series(features)