"""
负样本欠采样模块
实现两层负样本欠采样策略，解决Tick级涨停预测中的极端数据不平衡问题
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


def stratified_negative_sample(
    df: pd.DataFrame,
    dist_col: str = 'dist_to_limit_last',  # 修改为使用窗口末端特征
    label_col: str = 'label',
    thresholds: Tuple[float, float] = (0.01, 0.05),
    keep_ratios: Tuple[float, float, float] = (1.0, 0.3, 0.05),
    random_seed: Optional[int] = 42,
    verbose: bool = False
) -> pd.DataFrame:
    """
    第一层采样：基于窗口末端 dist_to_limit 的分层筛选

    对负样本根据窗口末端距离涨停价的远近进行分层，每层保留不同比例：
    - 接近涨停 (dist_to_limit_last <= 1%): 保留 100%
    - 中等距离 (1% < dist_to_limit_last <= 5%): 保留 30%
    - 远离涨停 (dist_to_limit_last > 5%): 保留 5%

    Args:
        df: 输入数据集（窗口样本）
        dist_col: 距离涨停价的列名（默认使用窗口末端特征）
        label_col: 标签列名
        thresholds: 分层阈值，分别为 (接近阈值, 中等阈值)
        keep_ratios: 各层保留率，分别为 (接近保留率, 中等保留率, 远离保留率)
        random_seed: 随机种子
        verbose: 是否打印详细信息

    Returns:
        第一层采样后的数据集
    """
    df = df.copy()

    # 验证必需列存在
    if dist_col not in df.columns:
        raise ValueError(f"DataFrame中缺少必需列: {dist_col}")
    if label_col not in df.columns:
        raise ValueError(f"DataFrame中缺少必需列: {label_col}")

    # 分离正样本和负样本
    positive_samples = df[df[label_col] == 1].copy()
    negative_samples = df[df[label_col] == 0].copy()

    if verbose:
        print(f"\n第一层采样：基于 dist_to_limit 分层筛选")
        print(f"  原始负样本数: {len(negative_samples)}")

    # 对负样本进行分层筛选
    sampled_negative_parts = []
    thresholds_sorted = sorted(thresholds)

    # 第一层：接近涨停
    mask_close = negative_samples[dist_col] <= thresholds_sorted[0]
    df_close = negative_samples[mask_close]
    if len(df_close) > 0:
        keep_n = int(len(df_close) * keep_ratios[0])
        sampled_close = df_close.sample(n=keep_n, random_state=random_seed)
        sampled_negative_parts.append(sampled_close)
        if verbose:
            print(f"  接近涨停 (≤{thresholds_sorted[0]*100:.1f}%): {len(df_close)} → {len(sampled_close)} ({keep_ratios[0]*100:.0f}%)")

    # 第二层：中等距离
    mask_medium = (negative_samples[dist_col] > thresholds_sorted[0]) & \
                  (negative_samples[dist_col] <= thresholds_sorted[1])
    df_medium = negative_samples[mask_medium]
    if len(df_medium) > 0:
        keep_n = int(len(df_medium) * keep_ratios[1])
        sampled_medium = df_medium.sample(n=keep_n, random_state=random_seed)
        sampled_negative_parts.append(sampled_medium)
        if verbose:
            print(f"  中等距离 ({thresholds_sorted[0]*100:.1f}%-{thresholds_sorted[1]*100:.1f}%): {len(df_medium)} → {len(sampled_medium)} ({keep_ratios[1]*100:.0f}%)")

    # 第三层：远离涨停
    mask_far = negative_samples[dist_col] > thresholds_sorted[1]
    df_far = negative_samples[mask_far]
    if len(df_far) > 0:
        keep_n = int(len(df_far) * keep_ratios[2])
        sampled_far = df_far.sample(n=keep_n, random_state=random_seed)
        sampled_negative_parts.append(sampled_far)
        if verbose:
            print(f"  远离涨停 (>{thresholds_sorted[1]*100:.1f}%): {len(df_far)} → {len(sampled_far)} ({keep_ratios[2]*100:.0f}%)")

    # 合并采样的负样本
    if sampled_negative_parts:
        sampled_negative = pd.concat(sampled_negative_parts, ignore_index=True)
    else:
        sampled_negative = pd.DataFrame(columns=df.columns)

    if verbose:
        print(f"  第一层采样后负样本数: {len(sampled_negative)}")

    # 合并正样本（100%保留）和采样的负样本
    result_df = pd.concat([positive_samples, sampled_negative], ignore_index=True)

    return result_df


def balance_sampling(
    df: pd.DataFrame,
    target_ratio: float = 5.0,
    label_col: str = 'label',
    random_seed: Optional[int] = 42,
    min_positive_samples: int = 10
) -> pd.DataFrame:
    """
    第二层采样：随机采样控制正负样本比例

    Args:
        df: 输入数据集（通常是第一层采样后的结果）
        target_ratio: 目标正负样本比例（负样本/正样本）
        label_col: 标签列名
        random_seed: 随机种子
        min_positive_samples: 最小正样本数，低于此值时跳过采样

    Returns:
        第二层采样后的数据集
    """
    df = df.copy()

    # 统计正负样本数量
    positive_samples = df[df[label_col] == 1]
    negative_samples = df[df[label_col] == 0]

    n_positive = len(positive_samples)
    n_negative = len(negative_samples)

    # 如果正样本数量太少，跳过采样
    if n_positive < min_positive_samples:
        print(f"  警告: 正样本数量 ({n_positive}) 少于最小值 ({min_positive_samples})，跳过第二层采样")
        return df

    # 计算当前比例
    current_ratio = n_negative / n_positive if n_positive > 0 else float('inf')

    # 如果当前比例已经小于等于目标比例，不需要采样
    if current_ratio <= target_ratio:
        print(f"  当前比例 ({current_ratio:.2f}) 已小于等于目标比例 ({target_ratio:.2f})，跳过第二层采样")
        return df

    # 计算需要保留的负样本数
    target_negative_count = int(n_positive * target_ratio)

    # 随机采样负样本
    sampled_negative = negative_samples.sample(n=target_negative_count, random_state=random_seed)

    print(f"  第二层采样：调整正负比例")
    print(f"    当前比例: {n_negative}:{n_positive} = {current_ratio:.2f}")
    print(f"    目标比例: {target_negative_count}:{n_positive} = {target_ratio:.2f}")
    print(f"    负样本数: {n_negative} → {target_negative_count}")

    # 合并正样本和采样的负样本
    result_df = pd.concat([positive_samples, sampled_negative], ignore_index=True)

    return result_df


def two_layer_negative_sampling(
    df: pd.DataFrame,
    dist_col: str = 'dist_to_limit_last',  # 修改为使用窗口末端特征
    label_col: str = 'label',
    thresholds: Tuple[float, float] = (0.01, 0.05),
    keep_ratios: Tuple[float, float, float] = (1.0, 0.3, 0.05),
    target_ratio: float = 5.0,
    random_seed: Optional[int] = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    完整的两层负样本欠采样流程（窗口样本版本）

    1. 第一层：基于窗口末端 dist_to_limit_last 分层筛选
    2. 第二层：随机采样控制正负样本比例到目标值

    Args:
        df: 输入数据集（窗口样本）
        dist_col: 距离涨停价的列名（默认使用窗口末端特征）
        label_col: 标签列名
        thresholds: 分层阈值
        keep_ratios: 各层保留率
        target_ratio: 目标正负样本比例
        random_seed: 随机种子
        verbose: 是否打印详细信息

    Returns:
        欠采样后的数据集
    """
    if verbose:
        print("\n" + "="*80)
        print("负样本欠采样")
        print("="*80)

    # 获取原始统计信息
    original_stats = get_sampling_statistics(df, label_col=label_col)
    if verbose:
        print(f"原始数据集:")
        print(f"  总样本数: {original_stats['total_samples']}")
        print(f"  正样本数: {original_stats['positive_samples']}")
        print(f"  负样本数: {original_stats['negative_samples']}")
        print(f"  正负比例: 1:{original_stats['imbalance_ratio']:.2f}")

    # 第一层采样
    df_after_layer1 = stratified_negative_sample(
        df, dist_col=dist_col, label_col=label_col,
        thresholds=thresholds, keep_ratios=keep_ratios,
        random_seed=random_seed, verbose=verbose
    )

    # 第二层采样
    df_after_layer2 = balance_sampling(
        df_after_layer1, target_ratio=target_ratio,
        label_col=label_col, random_seed=random_seed,
        min_positive_samples=10
    )

    # 获取最终统计信息
    final_stats = get_sampling_statistics(df_after_layer2, label_col=label_col)
    if verbose:
        print(f"\n欠采样后数据集:")
        print(f"  总样本数: {final_stats['total_samples']}")
        print(f"  正样本数: {final_stats['positive_samples']}")
        print(f"  负样本数: {final_stats['negative_samples']}")
        print(f"  正负比例: 1:{final_stats['imbalance_ratio']:.2f}")
        print(f"  样本减少: {original_stats['total_samples'] - final_stats['total_samples']:,} " +
              f"({(1 - final_stats['total_samples']/original_stats['total_samples'])*100:.1f}%)")
        print("="*80)

    return df_after_layer2


def undersample_train_set(
    df: pd.DataFrame,
    dist_col: str = 'dist_to_limit_last',  # 修改为使用窗口末端特征
    label_col: str = 'label',
    thresholds: Tuple[float, float] = (0.01, 0.05),
    keep_ratios: Tuple[float, float, float] = (1.0, 0.3, 0.05),
    target_ratio: float = 5.0,
    random_seed: Optional[int] = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    对训练集进行负样本欠采样的封装函数（窗口样本版本）

    明确标识仅用于训练集，避免误用。现在适配窗口样本，使用窗口末端特征进行采样。

    Args:
        df: 训练集数据（窗口样本）
        dist_col: 距离涨停价的列名（默认使用窗口末端特征 dist_to_limit_last）
        label_col: 标签列名
        thresholds: 分层阈值
        keep_ratios: 各层保留率
        target_ratio: 目标正负样本比例
        random_seed: 随机种子
        verbose: 是否打印详细信息

    Returns:
        欠采样后的训练集
    """
    if verbose:
        print("\n" + "="*80)
        print("训练集负样本欠采样（严格仅用于训练集，窗口样本模式）")
        print("="*80)

    result = two_layer_negative_sampling(
        df=df,
        dist_col=dist_col,
        label_col=label_col,
        thresholds=thresholds,
        keep_ratios=keep_ratios,
        target_ratio=target_ratio,
        random_seed=random_seed,
        verbose=verbose
    )

    # 验证采样结果
    if verbose:
        validate_sampling_result(result, df, dist_col=dist_col, label_col=label_col)

    return result


def get_sampling_statistics(df: pd.DataFrame, label_col: str = 'label') -> Dict:
    """
    获取采样统计信息

    Args:
        df: 数据集
        label_col: 标签列名

    Returns:
        统计信息字典
    """
    if label_col not in df.columns:
        return {'error': f'DataFrame中缺少必需列: {label_col}'}

    total_samples = len(df)
    positive_samples = int(df[label_col].sum())
    negative_samples = total_samples - positive_samples
    positive_ratio = positive_samples / total_samples if total_samples > 0 else 0
    imbalance_ratio = negative_samples / positive_samples if positive_samples > 0 else float('inf')

    return {
        'total_samples': total_samples,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'positive_ratio': positive_ratio,
        'negative_ratio': 1 - positive_ratio,
        'imbalance_ratio': imbalance_ratio
    }


def validate_sampling_result(
    sampled_df: pd.DataFrame,
    original_df: pd.DataFrame,
    dist_col: str = 'dist_to_limit',
    label_col: str = 'label',
    target_ratio: float = 5.0
) -> None:
    """
    验证采样结果的正确性

    Args:
        sampled_df: 采样后的数据集
        original_df: 原始数据集
        dist_col: 距离涨停价的列名
        label_col: 标签列名
        target_ratio: 目标正负样本比例

    Returns:
        None，打印验证结果
    """
    print("\n采样结果验证:")

    # 1. 检查正样本是否100%保留
    original_positive = original_df[original_df[label_col] == 1]
    sampled_positive = sampled_df[sampled_df[label_col] == 1]

    if len(original_positive) == len(sampled_positive):
        print("  [PASS] 正样本100%保留")
    else:
        print(f"  [WARN] 警告: 正样本数量变化 {len(original_positive)} → {len(sampled_positive)}")

    # 2. 检查正负样本比例
    stats = get_sampling_statistics(sampled_df, label_col=label_col)
    ratio_error = abs(stats['imbalance_ratio'] - target_ratio) / target_ratio

    if ratio_error <= 0.1:  # 允许10%误差
        print(f"  [PASS] 正负比例接近目标 1:{target_ratio:.1f} (实际 1:{stats['imbalance_ratio']:.2f})")
    else:
        print(f"  [WARN] 警告: 正负比例偏离目标 1:{target_ratio:.1f} (实际 1:{stats['imbalance_ratio']:.2f})")

    # 3. 检查特征完整性
    missing_features = set(original_df.columns) - set(sampled_df.columns)
    if not missing_features:
        print(f"  [PASS] 特征完整性: 保留所有 {len(sampled_df.columns)} 个特征列")
    else:
        print(f"  [WARN] 警告: 缺失特征列: {missing_features}")

    # 4. 检查样本减少程度
    reduction_rate = (1 - len(sampled_df) / len(original_df)) * 100
    if reduction_rate >= 90:
        print(f"  [PASS] 样本显著减少: {reduction_rate:.1f}%")
    else:
        print(f"  [INFO] 样本减少: {reduction_rate:.1f}%")


def automatic_undersample_if_needed(
    df: pd.DataFrame,
    dist_col: str = 'dist_to_limit',
    label_col: str = 'label',
    auto_threshold: float = 8.0,
    **kwargs
) -> pd.DataFrame:
    """
    自动检测并执行欠采样

    当正负样本比例超过阈值时自动执行欠采样

    Args:
        df: 输入数据集
        dist_col: 距离涨停价的列名
        label_col: 标签列名
        auto_threshold: 自动采样阈值，当正负比例超过此值时自动采样
        **kwargs: 传递给 undersample_train_set 的其他参数

    Returns:
        采样后的数据集（如果需要）或原始数据集
    """
    stats = get_sampling_statistics(df, label_col=label_col)

    if stats['imbalance_ratio'] > auto_threshold:
        print(f"\n检测到数据不平衡严重 (比例 1:{stats['imbalance_ratio']:.2f} > 1:{auto_threshold:.0f})")
        print("自动执行负样本欠采样...")

        return undersample_train_set(
            df=df,
            dist_col=dist_col,
            label_col=label_col,
            **kwargs
        )
    else:
        print(f"\n数据比例合理 (1:{stats['imbalance_ratio']:.2f})，无需欠采样")
        return df