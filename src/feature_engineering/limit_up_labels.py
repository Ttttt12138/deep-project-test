"""
涨停标签生成模块
实现next_tick_limit_up标签生成
"""

import pandas as pd
from typing import Tuple


def generate_next_tick_limit_up_label(df: pd.DataFrame, lookahead_ticks: int = 10) -> pd.DataFrame:
    """
    生成窗口后第N个tick是否涨停的标签
    方案A规范：前10个tick窗口 + 第11个tick标签

    Args:
        df: 包含tick数据的DataFrame，必须包含current和limit_price列
        lookahead_ticks: 向前看的tick数，默认为10（表示前10个tick窗口，第11个tick作为标签）

    Returns:
        添加了label列的DataFrame，label=1表示窗口后第N个tick（第11个tick）涨停
    """
    df = df.copy()

    # 生成标签：窗口后第N个tick是否涨停
    # 前lookahead_ticks个tick作为窗口，第lookahead_ticks+1个tick作为标签
    df['label'] = (df['current'].shift(-lookahead_ticks) >= df['limit_price']).astype(int)

    # 删除最后lookahead_ticks行（没有足够未来tick生成完整窗口和标签）
    df = df.iloc[:-lookahead_ticks].copy()

    return df


def generate_limit_up_label_with_probability(df: pd.DataFrame, lookahead_ticks: int = 5) -> pd.DataFrame:
    """
    生成未来N个tick内是否涨停的标签

    Args:
        df: 包含tick数据的DataFrame
        lookahead_ticks: 向前看多少个tick

    Returns:
        添加了label列的DataFrame
    """
    df = df.copy()

    # 检查未来lookahead_ticks个tick内是否有涨停
    future_prices = df['current'].rolling(window=lookahead_ticks + 1).apply(
        lambda x: 1 if any(x[1:] >= df.loc[x.index[0], 'limit_price']) else 0,
        raw=False
    )

    df['label'] = future_prices.shift(-lookahead_ticks).fillna(0).astype(int)

    # 删除最后lookahead_ticks行
    df = df.iloc[:-lookahead_ticks].copy()

    return df


def get_label_statistics(df: pd.DataFrame) -> dict:
    """
    获取标签统计信息

    Args:
        df: 包含label列的DataFrame

    Returns:
        标签统计信息字典
    """
    if 'label' not in df.columns:
        return {'error': 'DataFrame中没有label列'}

    total_samples = len(df)
    positive_samples = df['label'].sum()
    negative_samples = total_samples - positive_samples
    positive_ratio = positive_samples / total_samples if total_samples > 0 else 0

    return {
        'total_samples': total_samples,
        'positive_samples': int(positive_samples),
        'negative_samples': int(negative_samples),
        'positive_ratio': positive_ratio,
        'negative_ratio': 1 - positive_ratio,
        'imbalance_ratio': negative_samples / positive_samples if positive_samples > 0 else float('inf')
    }


def calculate_class_weights(df: pd.DataFrame) -> dict:
    """
    计算类别权重，用于处理类别不平衡

    Args:
        df: 包含label列的DataFrame

    Returns:
        类别权重字典
    """
    stats = get_label_statistics(df)

    neg_count = stats['negative_samples']
    pos_count = stats['positive_samples']

    # 计算scale_pos_weight
    scale_pos_weight = neg_count / max(pos_count, 1)

    return {
        'pos_weight': scale_pos_weight,
        'neg_weight': 1.0,
        'total_samples': stats['total_samples']
    }


def validate_labels(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    验证标签的有效性

    Args:
        df: 包含label列的DataFrame

    Returns:
        (是否有效, 错误信息)
    """
    if 'label' not in df.columns:
        return False, "DataFrame中没有label列"

    # 检查标签值是否只有0和1
    if not df['label'].isin([0, 1]).all():
        return False, "标签值必须为0或1"

    # 检查是否有缺失值
    if df['label'].isna().any():
        return False, "标签中存在缺失值"

    # 检查正样本数量
    stats = get_label_statistics(df)
    if stats['positive_samples'] == 0:
        return False, "数据集中没有正样本"

    return True, "标签验证通过"