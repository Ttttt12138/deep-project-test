"""
涨停特征工程模块
实现15个核心涨停预测特征
遵循单一职责原则，每个函数只提取一类特征
"""

import pandas as pd
import numpy as np
from typing import Dict


# ==================== 涨停接近度特征（4个）====================

def calculate_distance_to_limit_ratio(current: float, base_price: float, limit_ratio: float) -> float:
    """
    特征1：距离涨停比例

    Args:
        current: 当前价格
        base_price: 基准价格（昨收价或开盘价）
        limit_ratio: 涨停比例

    Returns:
        距离涨停比例 (0-1之间，越小越接近涨停)
    """
    limit_price = base_price * (1 + limit_ratio)
    return (limit_price - current) / limit_price if limit_price > 0 else 0.0


def calculate_ticks_to_limit(current: float, base_price: float, limit_ratio: float, tick_size: float) -> float:
    """
    特征2：距离涨停的最小价格跳数

    Args:
        current: 当前价格
        base_price: 基准价格（昨收价或开盘价）
        limit_ratio: 涨停比例
        tick_size: 最小价格跳动单位

    Returns:
        距离涨停的价格跳数
    """
    limit_price = base_price * (1 + limit_ratio)
    return (limit_price - current) / tick_size if limit_price > current and tick_size > 0 else 0.0


def calculate_ask1_to_limit_distance(ask1_price: float, base_price: float, limit_ratio: float) -> float:
    """
    特征3：卖一价格到涨停的距离

    Args:
        ask1_price: 卖一价格
        base_price: 基准价格（昨收价或开盘价）
        limit_ratio: 涨停比例

    Returns:
        卖一价到涨停的距离比例
    """
    limit_price = base_price * (1 + limit_ratio)
    return (limit_price - ask1_price) / limit_price if limit_price > 0 else 0.0


def calculate_ask1_current_spread(ask1_price: float, current: float) -> float:
    """
    特征4：卖一价与当前价的差距

    Args:
        ask1_price: 卖一价格
        current: 当前价格

    Returns:
        卖一价与当前价的差距比例
    """
    return (ask1_price - current) / current


# ==================== 盘口买卖强弱特征（5个）====================

def calculate_bid_volume_total(bid_volumes: Dict[int, float]) -> float:
    """
    特征5：买盘总量

    Args:
        bid_volumes: 买盘挂单量字典 {档位: 量}

    Returns:
        买盘总量
    """
    return sum(bid_volumes.values())


def calculate_ask_volume_total(ask_volumes: Dict[int, float]) -> float:
    """
    特征6：卖盘总量

    Args:
        ask_volumes: 卖盘挂单量字典 {档位: 量}

    Returns:
        卖盘总量
    """
    return sum(ask_volumes.values())


def calculate_order_imbalance(bid_volume: float, ask_volume: float) -> float:
    """
    特征7：盘口不平衡度

    Args:
        bid_volume: 买盘总量
        ask_volume: 卖盘总量

    Returns:
        盘口不平衡度 (-1到1之间，正数表示买盘强)
    """
    return (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-9)


def get_bid1_volume(bid1_volume: float) -> float:
    """
    特征8：买一挂单量

    Args:
        bid1_volume: 买一挂单量

    Returns:
        买一挂单量
    """
    return bid1_volume


def get_ask1_volume(ask1_volume: float) -> float:
    """
    特征9：卖一挂单量

    Args:
        ask1_volume: 卖一挂单量

    Returns:
        卖一挂单量
    """
    return ask1_volume


# ==================== 盘口价格结构特征（3个）====================

def calculate_spread(ask1_price: float, bid1_price: float) -> float:
    """
    特征10：买卖一价差

    Args:
        ask1_price: 卖一价格
        bid1_price: 买一价格

    Returns:
        买卖一价差
    """
    return ask1_price - bid1_price


def calculate_ask_depth_slope(ask_volumes: Dict[int, float]) -> float:
    """
    特征11：卖盘厚度斜率

    Args:
        ask_volumes: 卖盘挂单量字典 {档位: 量}

    Returns:
        卖盘厚度斜率 (负数表示越靠近涨停卖盘越厚)
    """
    volumes = [ask_volumes.get(i, 0) for i in range(1, 6)]
    if len(volumes) > 1:
        slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
        return slope
    return 0.0


def calculate_bid_depth_slope(bid_volumes: Dict[int, float]) -> float:
    """
    特征12：买盘厚度斜率

    Args:
        bid_volumes: 买盘挂单量字典 {档位: 量}

    Returns:
        买盘厚度斜率
    """
    volumes = [bid_volumes.get(i, 0) for i in range(1, 6)]
    if len(volumes) > 1:
        slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
        return slope
    return 0.0


# ==================== 成交与动量特征（3个）====================

def calculate_recent_return(current: float, previous_current: float) -> float:
    """
    特征13：最近一跳收益率

    Args:
        current: 当前价格
        previous_current: 前一个tick价格

    Returns:
        最近一跳收益率
    """
    return current / previous_current - 1


def calculate_recent_volume_change(current_volume: float, previous_volume: float) -> float:
    """
    特征14：最近成交量变化

    Args:
        current_volume: 当前tick成交量
        previous_volume: 前一个tick成交量

    Returns:
        最近成交量变化
    """
    return current_volume - previous_volume


def calculate_recent_money_change(current_money: float, previous_money: float) -> float:
    """
    特征15：最近成交额变化

    Args:
        current_money: 当前tick成交额
        previous_money: 前一个tick成交额

    Returns:
        最近成交额变化
    """
    return current_money - previous_money


# ==================== 批量特征提取函数 ====================

def extract_limit_up_features(df: pd.DataFrame, tick_size: float = 0.01, limit_ratio: float = 0.10) -> pd.DataFrame:
    """
    批量提取15个涨停特征

    Args:
        df: 包含tick数据的DataFrame（必须包含current列）
        tick_size: 最小价格跳动单位
        limit_ratio: 涨停比例（默认0.10）

    Returns:
        包含15个特征的DataFrame
    """
    features_df = df.copy()

    # 如果没有limit_price列，使用current作为基准价格
    if 'limit_price' not in features_df.columns:
        # 使用第一个有效价格作为基准价
        if 'preclose' in features_df.columns and features_df['preclose'].iloc[0] > 0:
            base_price = features_df['preclose'].iloc[0]
        else:
            # 使用第一个有效的current价格作为基准
            valid_currents = features_df[features_df['current'] > 0]['current']
            if len(valid_currents) > 0:
                base_price = valid_currents.iloc[0]
            else:
                base_price = 10.0  # 默认值

        # 计算涨停价
        features_df['limit_price'] = base_price * (1 + limit_ratio)
        features_df['base_price'] = base_price
    else:
        # 使用现有的limit_price作为基准
        features_df['base_price'] = features_df['limit_price'] / (1 + limit_ratio)

    # 涨停接近度特征（4个）
    features_df['dist_to_limit'] = features_df.apply(
        lambda row: calculate_distance_to_limit_ratio(row['current'], row['base_price'], limit_ratio),
        axis=1
    )
    features_df['ticks_to_limit'] = features_df.apply(
        lambda row: calculate_ticks_to_limit(row['current'], row['base_price'], limit_ratio, tick_size),
        axis=1
    )
    features_df['ask1_to_limit'] = features_df.apply(
        lambda row: calculate_ask1_to_limit_distance(row['a1_p'], row['base_price'], limit_ratio),
        axis=1
    )
    features_df['ask1_gap'] = features_df.apply(
        lambda row: calculate_ask1_current_spread(row['a1_p'], row['current']),
        axis=1
    )

    # 盘口买卖强弱特征（5个）
    features_df['bid_depth'] = features_df[['b1_v', 'b2_v', 'b3_v', 'b4_v', 'b5_v']].sum(axis=1)
    features_df['ask_depth'] = features_df[['a1_v', 'a2_v', 'a3_v', 'a4_v', 'a5_v']].sum(axis=1)
    features_df['order_imbalance'] = features_df.apply(
        lambda row: calculate_order_imbalance(row['bid_depth'], row['ask_depth']),
        axis=1
    )
    features_df['b1_volume'] = features_df['b1_v']
    features_df['a1_volume'] = features_df['a1_v']

    # 盘口价格结构特征（3个）
    features_df['spread'] = features_df.apply(
        lambda row: calculate_spread(row['a1_p'], row['b1_p']),
        axis=1
    )

    ask_volumes_list = [
        {'a1_v': row['a1_v'], 'a2_v': row['a2_v'], 'a3_v': row['a3_v'], 'a4_v': row['a4_v'], 'a5_v': row['a5_v']}
        for _, row in features_df.iterrows()
    ]
    features_df['ask_slope'] = [
        calculate_ask_depth_slope(ask_volumes) for ask_volumes in ask_volumes_list
    ]

    bid_volumes_list = [
        {'b1_v': row['b1_v'], 'b2_v': row['b2_v'], 'b3_v': row['b3_v'], 'b4_v': row['b4_v'], 'b5_v': row['b5_v']}
        for _, row in features_df.iterrows()
    ]
    features_df['bid_slope'] = [
        calculate_bid_depth_slope(bid_volumes) for bid_volumes in bid_volumes_list
    ]

    # 成交与动量特征（3个）
    features_df['ret_1tick'] = features_df['current'].pct_change()
    features_df['vol_delta'] = features_df['volume'].diff()
    features_df['money_delta'] = features_df['money'].diff()

    # 选择特征列
    feature_columns = [
        'dist_to_limit', 'ticks_to_limit', 'ask1_to_limit', 'ask1_gap',
        'bid_depth', 'ask_depth', 'order_imbalance', 'b1_volume', 'a1_volume',
        'spread', 'ask_slope', 'bid_slope',
        'ret_1tick', 'vol_delta', 'money_delta'
    ]

    # 保留原始列中的重要列（code, date, time, current, limit_price等）
    result_df = df[['time', 'current', 'limit_price'] + feature_columns].copy()

    # 如果原始df中有code和date列，也要保留
    if 'code' in df.columns:
        result_df['code'] = df['code'].values
    if 'date' in df.columns:
        result_df['date'] = df['date'].values

    return result_df