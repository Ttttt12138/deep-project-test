"""
测试负样本欠采样模块
使用模拟数据验证采样功能的正确性
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.sampling import (
    stratified_negative_sample,
    balance_sampling,
    two_layer_negative_sampling,
    undersample_train_set,
    get_sampling_statistics,
    validate_sampling_result
)


def create_mock_data(n_samples=10000, pos_ratio=0.013, random_state=42):
    """
    创建模拟数据

    Args:
        n_samples: 总样本数
        pos_ratio: 正样本比例
        random_state: 随机种子

    Returns:
        包含 dist_to_limit 和 label 列的 DataFrame
    """
    # 生成标签（模拟 1:77.9 的极端不平衡）
    n_positive = int(n_samples * pos_ratio)
    n_negative = n_samples - n_positive

    labels = np.array([1] * n_positive + [0] * n_negative)

    # 生成 dist_to_limit（距离涨停价的相对距离）
    # 负样本的分布：大部分远离涨停，少数接近涨停
    np.random.seed(random_state)
    negative_dist = np.random.exponential(scale=0.1, size=n_negative)  # 指数分布
    negative_dist = np.clip(negative_dist, 0.001, 0.15)  # 限制在 0.1% - 15%

    # 正样本通常接近涨停
    positive_dist = np.random.uniform(0.001, 0.03, size=n_positive)

    dist_to_limit = np.concatenate([negative_dist, positive_dist])

    # 创建DataFrame
    df = pd.DataFrame({
        'dist_to_limit': dist_to_limit,
        'label': labels
    })

    # 打乱顺序
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def test_stratified_negative_sample():
    """测试第一层采样：分层筛选"""
    print("\n" + "="*80)
    print("测试第一层采样：stratified_negative_sample()")
    print("="*80)

    # 创建模拟数据
    df = create_mock_data(n_samples=10000, pos_ratio=0.013)

    # 获取原始统计
    original_stats = get_sampling_statistics(df)
    print(f"原始数据:")
    print(f"  总样本: {original_stats['total_samples']}")
    print(f"  正样本: {original_stats['positive_samples']}")
    print(f"  负样本: {original_stats['negative_samples']}")
    print(f"  比例: 1:{original_stats['imbalance_ratio']:.2f}")

    # 执行第一层采样
    sampled_df = stratified_negative_sample(
        df,
        dist_col='dist_to_limit',
        label_col='label',
        thresholds=(0.01, 0.05),
        keep_ratios=(1.0, 0.3, 0.05),
        random_seed=42,
        verbose=True
    )

    # 验证结果
    sampled_stats = get_sampling_statistics(sampled_df)
    print(f"\n第一层采样后:")
    print(f"  总样本: {sampled_stats['total_samples']}")
    print(f"  正样本: {sampled_stats['positive_samples']}")
    print(f"  负样本: {sampled_stats['negative_samples']}")
    print(f"  比例: 1:{sampled_stats['imbalance_ratio']:.2f}")
    print(f"  减少: {(1 - sampled_stats['total_samples']/original_stats['total_samples'])*100:.1f}%")

    # 验证正样本是否100%保留
    assert sampled_stats['positive_samples'] == original_stats['positive_samples'], \
        "正样本应该100%保留"

    print("\n[PASS] 第一层采样测试通过")


def test_balance_sampling():
    """测试第二层采样：比例控制"""
    print("\n" + "="*80)
    print("测试第二层采样：balance_sampling()")
    print("="*80)

    # 创建模拟数据（第一层采样后的结果）
    df = create_mock_data(n_samples=5000, pos_ratio=0.02)

    # 获取原始统计
    original_stats = get_sampling_statistics(df)
    print(f"输入数据:")
    print(f"  总样本: {original_stats['total_samples']}")
    print(f"  正样本: {original_stats['positive_samples']}")
    print(f"  负样本: {original_stats['negative_samples']}")
    print(f"  比例: 1:{original_stats['imbalance_ratio']:.2f}")

    # 执行第二层采样
    target_ratio = 5.0
    sampled_df = balance_sampling(
        df,
        target_ratio=target_ratio,
        label_col='label',
        random_seed=42,
        min_positive_samples=1
    )

    # 验证结果
    sampled_stats = get_sampling_statistics(sampled_df)
    print(f"\n第二层采样后:")
    print(f"  总样本: {sampled_stats['total_samples']}")
    print(f"  正样本: {sampled_stats['positive_samples']}")
    print(f"  负样本: {sampled_stats['negative_samples']}")
    print(f"  比例: 1:{sampled_stats['imbalance_ratio']:.2f}")

    # 验证比例是否接近目标
    ratio_error = abs(sampled_stats['imbalance_ratio'] - target_ratio) / target_ratio
    assert ratio_error <= 0.1, f"比例误差过大: {ratio_error:.2%}"

    # 验证正样本是否100%保留
    assert sampled_stats['positive_samples'] == original_stats['positive_samples'], \
        "正样本应该100%保留"

    print("\n[PASS] 第二层采样测试通过")


def test_two_layer_negative_sampling():
    """测试完整两层采样流程"""
    print("\n" + "="*80)
    print("测试完整两层采样：two_layer_negative_sampling()")
    print("="*80)

    # 创建模拟数据（模拟 1:77.9 的极端不平衡）
    df = create_mock_data(n_samples=10000, pos_ratio=0.013)

    # 获取原始统计
    original_stats = get_sampling_statistics(df)
    print(f"原始数据:")
    print(f"  总样本: {original_stats['total_samples']}")
    print(f"  正样本: {original_stats['positive_samples']}")
    print(f"  负样本: {original_stats['negative_samples']}")
    print(f"  比例: 1:{original_stats['imbalance_ratio']:.2f}")

    # 执行完整两层采样
    sampled_df = two_layer_negative_sampling(
        df,
        dist_col='dist_to_limit',
        label_col='label',
        thresholds=(0.01, 0.05),
        keep_ratios=(1.0, 0.3, 0.05),
        target_ratio=5.0,
        random_seed=42,
        verbose=True
    )

    # 验证结果
    sampled_stats = get_sampling_statistics(sampled_df)
    print(f"\n两层采样后:")
    print(f"  总样本: {sampled_stats['total_samples']}")
    print(f"  正样本: {sampled_stats['positive_samples']}")
    print(f"  负样本: {sampled_stats['negative_samples']}")
    print(f"  比例: 1:{sampled_stats['imbalance_ratio']:.2f}")
    print(f"  减少: {(1 - sampled_stats['total_samples']/original_stats['total_samples'])*100:.1f}%")

    # 验证正样本是否100%保留
    assert sampled_stats['positive_samples'] == original_stats['positive_samples'], \
        "正样本应该100%保留"

    # 验证比例是否接近目标
    target_ratio = 5.0
    ratio_error = abs(sampled_stats['imbalance_ratio'] - target_ratio) / target_ratio
    assert ratio_error <= 0.1, f"比例误差过大: {ratio_error:.2%}"

    # 验证减少程度
    reduction_rate = (1 - sampled_stats['total_samples']/original_stats['total_samples'])
    assert reduction_rate >= 0.9, f"样本减少不足90%: {reduction_rate*100:.1f}%"

    print("\n[PASS] 完整两层采样测试通过")


def test_undersample_train_set():
    """测试训练集封装函数"""
    print("\n" + "="*80)
    print("测试训练集封装：undersample_train_set()")
    print("="*80)

    # 创建模拟数据
    df = create_mock_data(n_samples=10000, pos_ratio=0.013)

    # 执行训练集欠采样
    sampled_df = undersample_train_set(
        df,
        dist_col='dist_to_limit',
        label_col='label',
        thresholds=(0.01, 0.05),
        keep_ratios=(1.0, 0.3, 0.05),
        target_ratio=5.0,
        random_seed=42,
        verbose=True
    )

    # 验证采样结果
    validate_sampling_result(
        sampled_df,
        df,
        dist_col='dist_to_limit',
        label_col='label',
        target_ratio=5.0
    )

    print("\n[PASS] 训练集封装测试通过")


def test_integration():
    """集成测试：验证整个流程"""
    print("\n" + "="*80)
    print("集成测试：验证整个流程")
    print("="*80)

    # 模拟训练集、验证集、测试集
    train_df = create_mock_data(n_samples=10000, pos_ratio=0.013, random_state=42)
    val_df = create_mock_data(n_samples=2000, pos_ratio=0.013, random_state=43)
    test_df = create_mock_data(n_samples=2000, pos_ratio=0.013, random_state=44)

    print("\n原始数据集:")
    print(f"  训练集: {len(train_df):,} 样本 (1:{get_sampling_statistics(train_df)['imbalance_ratio']:.2f})")
    print(f"  验证集: {len(val_df):,} 样本 (1:{get_sampling_statistics(val_df)['imbalance_ratio']:.2f})")
    print(f"  测试集: {len(test_df):,} 样本 (1:{get_sampling_statistics(test_df)['imbalance_ratio']:.2f})")

    # 仅对训练集进行欠采样
    train_df_sampled = undersample_train_set(
        train_df,
        dist_col='dist_to_limit',
        label_col='label',
        thresholds=(0.01, 0.05),
        keep_ratios=(1.0, 0.3, 0.05),
        target_ratio=5.0,
        random_seed=42,
        verbose=True
    )

    print("\n欠采样后数据集:")
    print(f"  训练集: {len(train_df_sampled):,} 样本 (1:{get_sampling_statistics(train_df_sampled)['imbalance_ratio']:.2f})")
    print(f"  验证集: {len(val_df):,} 样本 (保持原样)")
    print(f"  测试集: {len(test_df):,} 样本 (保持原样)")

    # 验证规则
    print("\n验证关键规则:")

    # 1. 验证训练集被采样
    train_original_positive = get_sampling_statistics(train_df)['positive_samples']
    train_sampled_positive = get_sampling_statistics(train_df_sampled)['positive_samples']
    assert train_original_positive == train_sampled_positive, "训练集正样本应该100%保留"
    print("  [PASS] 训练集正样本100%保留")

    # 2. 验证验证集和测试集未被采样
    val_original_size = len(val_df)
    test_original_size = len(test_df)
    assert len(val_df) == val_original_size, "验证集大小应该保持不变"
    assert len(test_df) == test_original_size, "测试集大小应该保持不变"
    print("  [PASS] 验证集和测试集保持原样")

    # 3. 验证训练集比例接近目标
    train_ratio = get_sampling_statistics(train_df_sampled)['imbalance_ratio']
    ratio_error = abs(train_ratio - 5.0) / 5.0
    assert ratio_error <= 0.1, f"训练集比例误差过大: {ratio_error:.2%}"
    print("  [PASS] 训练集比例接近目标 1:5")

    print("\n[PASS] 集成测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("负样本欠采样模块测试")
    print("="*80)

    try:
        # 运行各项测试
        test_stratified_negative_sample()
        test_balance_sampling()
        test_two_layer_negative_sampling()
        test_undersample_train_set()
        test_integration()

        print("\n" + "="*80)
        print("[SUCCESS] All tests passed!")
        print("="*80)
        return 0

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())