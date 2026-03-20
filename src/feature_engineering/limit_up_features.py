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
    try:
        # 确保数值类型正确
        current = float(current)
        base_price = float(base_price)
        limit_ratio = float(limit_ratio)

        limit_price = round(base_price * (1 + limit_ratio), 2)  # 先四舍五入涨停价
        current_rounded = round(current, 2)  # 四舍五入当前价格

        if limit_price > 0:
            return round((limit_price - current_rounded) / limit_price, 4)
        else:
            return 0.0
    except (TypeError, ValueError, AttributeError):
        return 0.0


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
    try:
        # 确保数值类型正确
        ask1_price = float(ask1_price)
        current = float(current)

        if current > 0:
            return (ask1_price - current) / current
        else:
            return 0.0
    except (TypeError, ValueError, ZeroDivisionError, AttributeError):
        return 0.0


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
    try:
        # 确保数值类型正确
        bid_volume = float(bid_volume)
        ask_volume = float(ask_volume)

        total_volume = bid_volume + ask_volume
        if total_volume > 1e-9:
            return (bid_volume - ask_volume) / total_volume
        else:
            return 0.0
    except (TypeError, ValueError, AttributeError):
        return 0.0


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
    try:
        volumes = [float(ask_volumes.get(i, 0)) for i in range(1, 6)]
        if len(volumes) > 1:
            slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
            return float(slope)
        return 0.0
    except (TypeError, ValueError, AttributeError, IndexError):
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
    提取涨停预测特征 - 安全版本，处理类型异常

    Args:
        df: 包含tick数据的DataFrame
        tick_size: 最小价格跳变单位
        limit_ratio: 涨停比例

    Returns:
        添加了特征列的DataFrame
    """
    df = df.copy()

    try:
        # 确保必需列存在
        required_cols = ['current', 'limit_price', 'b1_p', 'a1_p', 'b1_v', 'a1_v',
                        'b2_p', 'a2_p', 'b3_p', 'a3_p', 'b4_p', 'a4_p', 'b5_p', 'a5_p',
                        'volume', 'money']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0  # 如果缺少列，填入默认值

        # 确保数值列的类型正确
        numeric_cols = ['current', 'limit_price', 'b1_p', 'a1_p', 'b1_v', 'a1_v',
                      'b2_p', 'a2_p', 'b3_p', 'a3_p', 'b4_p', 'a4_p', 'b5_p', 'a5_p',
                      'volume', 'money']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 基准价格（昨收价）
        base_price = df['limit_price'].iloc[0] / (1 + limit_ratio) if 'limit_price' in df.columns else 10.0

        # 涨停接近度特征（4个）
        df['dist_to_limit'] = df.apply(lambda row: safe_calculate_distance_to_limit_ratio(
            row['current'], base_price, limit_ratio), axis=1)

        df['ticks_to_limit'] = df.apply(lambda row: safe_calculate_ticks_to_limit(
            row['current'], base_price, limit_ratio, tick_size), axis=1)

        df['ask1_to_limit'] = df.apply(lambda row: safe_calculate_ask1_to_limit_distance(
            row['a1_p'], base_price, limit_ratio), axis=1)

        df['ask1_gap'] = df.apply(lambda row: safe_calculate_ask1_current_spread(
            row['a1_p'], row['current']), axis=1)

        # 盘口买卖强弱特征（5个）
        df['bid_depth'] = df.apply(lambda row: safe_calculate_bid_volume_total({
            1: row['b1_v'], 2: row.get('b2_v', 0), 3: row.get('b3_v', 0),
            4: row.get('b4_v', 0), 5: row.get('b5_v', 0)
        }), axis=1)

        df['ask_depth'] = df.apply(lambda row: safe_calculate_ask_volume_total({
            1: row['a1_v'], 2: row.get('a2_v', 0), 3: row.get('a3_v', 0),
            4: row.get('a4_v', 0), 5: row.get('a5_v', 0)
        }), axis=1)

        df['order_imbalance'] = df.apply(lambda row: safe_calculate_order_imbalance(
            row['bid_depth'], row['ask_depth']), axis=1)

        df['b1_volume'] = df['b1_v'].values
        df['a1_volume'] = df['a1_v'].values

        # 盘口价格结构特征（3个）
        df['spread'] = df.apply(lambda row: safe_calculate_spread(
            row['a1_p'], row['b1_p']), axis=1)

        df['ask_slope'] = df.apply(lambda row: safe_calculate_ask_depth_slope({
            1: row['a1_v'], 2: row.get('a2_v', 0), 3: row.get('a3_v', 0),
            4: row.get('a4_v', 0), 5: row.get('a5_v', 0)
        }), axis=1)

        df['bid_slope'] = df.apply(lambda row: safe_calculate_bid_depth_slope({
            1: row['b1_v'], 2: row.get('b2_v', 0), 3: row.get('b3_v', 0),
            4: row.get('b4_v', 0), 5: row.get('b5_v', 0)
        }), axis=1)

        # 动量特征（3个）
        df['ret_1tick'] = df['current'].pct_change().fillna(0.0)
        df['vol_delta'] = df['volume'].diff().fillna(0.0)
        df['money_delta'] = df['money'].diff().fillna(0.0)

        # 确保所有特征都是数值类型
        feature_cols = ['dist_to_limit', 'ticks_to_limit', 'ask1_to_limit', 'ask1_gap',
                      'bid_depth', 'ask_depth', 'order_imbalance', 'b1_volume', 'a1_volume',
                      'spread', 'ask_slope', 'bid_slope',
                      'ret_1tick', 'vol_delta', 'money_delta']

        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        return df

    except Exception as e:
        print(f"特征提取出错: {e}")
        # 返回空DataFrame而不是抛出异常
        return pd.DataFrame()


# 安全包装函数
def safe_calculate_distance_to_limit_ratio(current, base_price, limit_ratio):
    """安全的距离涨停比例计算"""
    try:
        current = float(current)
        base_price = float(base_price)
        limit_ratio = float(limit_ratio)
        limit_price = round(base_price * (1 + limit_ratio), 2)
        current_rounded = round(current, 2)
        return round((limit_price - current_rounded) / limit_price, 4) if limit_price > 0 else 0.0
    except:
        return 0.0

def safe_calculate_ticks_to_limit(current, base_price, limit_ratio, tick_size):
    """安全的距离涨停跳数计算"""
    try:
        current = float(current)
        base_price = float(base_price)
        limit_ratio = float(limit_ratio)
        tick_size = float(tick_size)
        limit_price = base_price * (1 + limit_ratio)
        return (limit_price - current) / tick_size if limit_price > current and tick_size > 0 else 0.0
    except:
        return 0.0

def safe_calculate_ask1_to_limit_distance(ask1_price, base_price, limit_ratio):
    """安全的卖一到涨停距离计算"""
    try:
        ask1_price = float(ask1_price)
        base_price = float(base_price)
        limit_ratio = float(limit_ratio)
        limit_price = base_price * (1 + limit_ratio)
        return (limit_price - ask1_price) / limit_price if limit_price > 0 else 0.0
    except:
        return 0.0

def safe_calculate_ask1_current_spread(ask1_price, current):
    """安全的卖一价与当前价差距计算"""
    try:
        ask1_price = float(ask1_price)
        current = float(current)
        return (ask1_price - current) / current if current > 0 else 0.0
    except:
        return 0.0

def safe_calculate_bid_volume_total(bid_volumes):
    """安全的买盘总量计算"""
    try:
        return sum(float(v) for v in bid_volumes.values())
    except:
        return 0.0

def safe_calculate_ask_volume_total(ask_volumes):
    """安全的卖盘总量计算"""
    try:
        return sum(float(v) for v in ask_volumes.values())
    except:
        return 0.0

def safe_calculate_order_imbalance(bid_volume, ask_volume):
    """安全的盘口不平衡度计算"""
    try:
        bid_volume = float(bid_volume)
        ask_volume = float(ask_volume)
        total_volume = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total_volume if total_volume > 1e-9 else 0.0
    except:
        return 0.0

def safe_calculate_spread(ask1_price, bid1_price):
    """安全的买卖一价差计算"""
    try:
        ask1_price = float(ask1_price)
        bid1_price = float(bid1_price)
        return ask1_price - bid1_price
    except:
        return 0.0

def safe_calculate_ask_depth_slope(ask_volumes):
    """安全的卖盘厚度斜率计算"""
    try:
        volumes = [float(ask_volumes.get(i, 0)) for i in range(1, 6)]
        if len(volumes) > 1:
            slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
            return float(slope)
        return 0.0
    except:
        return 0.0

def safe_calculate_bid_depth_slope(bid_volumes):
    """安全的买盘厚度斜率计算"""
    try:
        volumes = [float(bid_volumes.get(i, 0)) for i in range(1, 6)]
        if len(volumes) > 1:
            slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
            return float(slope)
        return 0.0
    except:
        return 0.0
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
    result_df = features_df[['time', 'current', 'limit_price'] + feature_columns].copy()

    # 如果原始df中有code和date列，也要保留
    if 'code' in df.columns:
        result_df['code'] = df['code'].values
    if 'date' in df.columns:
        result_df['date'] = df['date'].values

    return result_df