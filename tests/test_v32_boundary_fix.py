"""
边界过滤修复验证测试
测试正样本在边界位置是否会被正确保护
"""

import pandas as pd
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_engineering.event_driven_labels import filter_and_label_events

def test_boundary_positive_sample_protection():
    """测试正样本在边界位置是否被保护"""
    print("测试边界位置正样本保护...")

    # 场景1：触板发生在倒数第二个位置
    n_ticks = 50
    test_data = {
        'time': range(n_ticks),
        'current': [10.0 + i * 0.04 for i in range(n_ticks-2)] + [11.92, 12.0],
        'limit_price': [12.0] * n_ticks
    }

    df = pd.DataFrame(test_data)
    df['code'] = 'TEST001'

    # 执行过滤
    filtered_df, valid_indices = filter_and_label_events(df, limit_dist_threshold=0.05, debug=False)

    # 检查正样本
    positive_samples = filtered_df[filtered_df['label'] == 1]
    print(f"检测到的正样本数量: {len(positive_samples)}")
    print(f"正样本位置: {positive_samples.index.tolist()}")

    if len(positive_samples) > 0:
        # 验证正样本是否被保留
        for idx in positive_samples.index:
            is_preserved = idx in valid_indices
            print(f"正样本 {idx} 是否被保留: {is_preserved}")
            assert is_preserved, f"正样本 {idx} 应该被保留，但被过滤了"

        print("[OK] 边界位置正样本保护测试通过")
    else:
        print("[FAIL] 没有检测到正样本")
        assert False, "应该检测到边界位置正样本"


def test_early_boundary_positive_sample():
    """测试早盘边界正样本"""
    print("\n测试早盘边界正样本...")

    # 场景2：触板发生在早期（第15个tick）
    n_ticks = 50
    test_data = {
        'time': range(n_ticks),
        'current': [10.0 + i * 0.15 for i in range(15)] + [12.0] * (n_ticks - 15),
        'limit_price': [12.0] * n_ticks
    }

    df = pd.DataFrame(test_data)
    df['code'] = 'TEST002'

    # 执行过滤
    filtered_df, valid_indices = filter_and_label_events(df, limit_dist_threshold=0.05, debug=False)

    # 检查正样本
    positive_samples = filtered_df[filtered_df['label'] == 1]
    print(f"检测到的正样本数量: {len(positive_samples)}")
    print(f"正样本位置: {positive_samples.index.tolist()}")

    if len(positive_samples) > 0:
        # 验证正样本是否被保留
        for idx in positive_samples.index:
            is_preserved = idx in valid_indices
            print(f"正样本 {idx} 是否被保留: {is_preserved}")
            assert is_preserved, f"正样本 {idx} 应该被保留，但被过滤了"

        print("[OK] 早盘边界正样本保护测试通过")
    else:
        print("[FAIL] 没有检测到正样本")
        assert False, "应该检测到早盘边界正样本"


def test_last_tick_protection():
    """测试最后一个tick的保护逻辑"""
    print("\n测试最后一个tick保护...")

    # 场景3：触板发生在最后一个有意义的tick前
    n_ticks = 30
    test_data = {
        'time': range(n_ticks),
        'current': [10.0 + i * 0.07 for i in range(n_ticks-2)] + [11.99, 12.0],
        'limit_price': [12.0] * n_ticks
    }

    df = pd.DataFrame(test_data)
    df['code'] = 'TEST003'

    # 执行过滤
    filtered_df, valid_indices = filter_and_label_events(df, limit_dist_threshold=0.05, debug=False)

    print(f"原始tick数: {len(df)}")
    print(f"过滤后保留数: {len(valid_indices)}")
    print(f"保留的索引: {sorted(valid_indices)}")

    # 检查正样本
    positive_samples = filtered_df[filtered_df['label'] == 1]
    if len(positive_samples) > 0:
        print(f"正样本位置: {positive_samples.index.tolist()}")
        for idx in positive_samples.index:
            is_preserved = idx in valid_indices
            print(f"正样本 {idx} 是否被保留: {is_preserved}")
            assert is_preserved, f"正样本 {idx} 应该被保留，但被过滤了"

        print("[OK] 最后一个tick保护测试通过")
    else:
        print("[FAIL] 没有检测到正样本")
        assert False, "应该检测到最后一个有意义tick前的正样本"


def test_negative_sample_boundary_filtering():
    """测试负样本在边界位置的正确过滤"""
    print("\n测试负样本边界过滤...")

    # 场景4：最后一个tick是普通负样本（应该被过滤）
    n_ticks = 20
    test_data = {
        'time': range(n_ticks),
        'current': [10.0 + i * 0.05 for i in range(n_ticks)],
        'limit_price': [12.0] * n_ticks
    }

    df = pd.DataFrame(test_data)
    df['code'] = 'TEST004'

    # 执行过滤
    filtered_df, valid_indices = filter_and_label_events(df, limit_dist_threshold=0.05, debug=False)

    print(f"原始tick数: {len(df)}")
    print(f"过滤后保留数: {len(valid_indices)}")

    # 检查最后一个tick是否被过滤
    last_index = df.index[-1]
    last_tick_preserved = last_index in valid_indices
    print(f"最后一个tick (idx={last_index}) 是否被保留: {last_tick_preserved}")

    # 最后一个tick应该被过滤（没有正样本，且不是困难负样本）
    assert not last_tick_preserved, "最后一个普通负样本应该被过滤"
    print("[OK] 负样本边界过滤测试通过")


if __name__ == "__main__":
    print("="*60)
    print("边界过滤修复验证测试")
    print("="*60)

    all_passed = True
    try:
        test_boundary_positive_sample_protection()
        test_early_boundary_positive_sample()
        test_last_tick_protection()
        test_negative_sample_boundary_filtering()
    except AssertionError:
        all_passed = False
        raise

    print("\n" + "="*60)
    if all_passed:
        print("[OK] 所有边界过滤修复测试通过")
    else:
        print("[FAIL] 部分测试失败")
    print("="*60)
