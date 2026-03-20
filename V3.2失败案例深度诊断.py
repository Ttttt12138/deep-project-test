"""
V3.2数据处理失败深度诊断
分析"有触板事件但被过滤"的根本原因
"""

"""
## 问题分析

从失败统计来看：
- **无触板事件且无困难负样本: 4836只股票** (95%失败率) - 主要问题
- **有触板事件(1个)但被过滤: 142只股票** (仍然存在)

## 可能的根本原因

### 1. 触板事件检测与过滤逻辑冲突

当前的过滤逻辑：
```python
condition_a = df['label'] == 1                            # 正样本（首次触板）
condition_b = df['dist_to_limit'] < limit_dist_threshold  # 困难负样本
is_dead_limit = current_is_limit                          # 死板状态

keep_mask = (condition_a | condition_b) & (~is_dead_limit)
```

**关键问题**：`is_dead_limit = current_is_limit`

触板瞬间检测：
- T10不涨停，T11涨停
- 此时current_is_limit[T10] = False，current_is_limit[T11] = True

但如果触板后立即进入死板状态，那么：
- 标记为正样本的那个tick（T10）的current_is_limit可能为False
- 但后续tick的current_is_limit为True
- 如果数据长度较短，可能导致问题

### 2. 多股票处理中的代码/日期问题

在`build_multi_stock_event_samples`中：
```python
# 按代码分组，取每个代码的第一个触板索引
first_touch_indices = df.loc[touch_indices].groupby('code').head(1).index
```

**问题**：如果传入的df已经包含了多只股票的数据，但在处理前没有正确的分组，可能导致：
- 某些股票的触板事件被误判
- 不同股票的数据混淆

### 3. 窗口构建失败

即使触板事件被正确检测和保留，但在窗口构建过程中可能失败：
- 缺少必需特征
- 窗口长度不足
- 特征平铺失败

## 诊断策略

### 策略1：增加详细日志
在filter_and_label_events中增加调试输出，显示每个触板事件的详细信息。

### 策略2：检查死板状态判断
检查is_dead_limit的计算是否正确，特别是在触板瞬间的tick上。

### 策略3：验证触板事件检测
确保触板事件检测逻辑与过滤逻辑一致。

## 最可能的根本原因

基于代码分析，**最可能的原因是：**

触板事件发生在股价已经非常接近涨停价的位置，导致：

1. **触板瞬间的dist_to_limit很小**（接近0）
2. **但触板后的tick处于死板状态**（current_is_limit = True）
3. **触板瞬间的tick本身可能被误判**为死板状态

或者：

触板事件被检测到，但：
- **窗口构建失败**（缺少必需特征）
- **数据质量问题**（价格计算错误）
- **特征提取失败**（dist_to_limit计算错误）

## 建议的修复方向

1. **简化过滤逻辑**：优先保留正样本
2. **增加数据验证**：确保特征提取正确
3. **优化触板检测**：提高检测准确性
4. **增加调试信息**：显示每个失败股票的详细信息
"""

def diagnose_failure_case():
    """诊断单个失败案例"""
    import pandas as pd
    import sys
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from src.feature_engineering.event_driven_labels import filter_and_label_events

    # 模拟一个"有触板事件但被过滤"的场景
    n_ticks = 100
    limit_price = 12.0

    # 价格走势：很早就达到涨停价，然后一直是涨停
    prices = []
    for i in range(n_ticks):
        if i < 10:
            price = 10.0 + i * 0.1  # 快速拉升
        elif i < 12:
            price = 11.0 + (i - 10) * 0.5  # 继续拉升
        else:
            price = 12.0  # 一直是涨停

        prices.append(price)

    test_df = pd.DataFrame({
        'code': ['TEST001'] * n_ticks,
        'time': range(n_ticks),
        'current': prices,
        'limit_price': [limit_price] * n_ticks,
    })

    # 执行过滤
    filtered_df, valid_indices = filter_and_label_events(test_df, limit_dist_threshold=0.05, debug=False)

    # 分析结果
    print("诊断案例分析：")
    print(f"原始tick数: {len(test_df)}")

    # 检查触板事件
    touch_events = filtered_df[filtered_df['label'] == 1]
    print(f"检测到的触板事件: {len(touch_events)}")
    if len(touch_events) > 0:
        print(f"触板事件位置: {touch_events.index.tolist()}")
        for idx in touch_events.index:
            is_preserved = idx in valid_indices
            print(f"触板事件 {idx} 是否被保留: {is_preserved}")
            print(f"  current: {filtered_df.loc[idx, 'current']}")
            print(f"  limit_price: {filtered_df.loc[idx, 'limit_price']}")
            print(f"  dist_to_limit: {filtered_df.loc[idx, 'dist_to_limit']}")

    # 检查死板状态
    dead_limit_ticks = test_df[test_df['current'] >= test_df['limit_price']]
    print(f"\n死板状态tick数: {len(dead_limit_ticks)}")
    if len(dead_limit_ticks) > 0:
        print(f"死板tick位置: {dead_limit_ticks.index.tolist()}")

    # 检查保留的样本
    preserved_samples = filtered_df.loc[valid_indices]
    print(f"\n保留的样本数: {len(preserved_samples)}")
    print(f"保留的样本类型: 正样本{preserved_samples['label'].sum()}, 负样本{len(preserved_samples) - preserved_samples['label'].sum()}")

    # 检查距离分布
    print(f"\n距离分布统计:")
    print(f"  最小距离: {filtered_df['dist_to_limit'].min():.6f}")
    print(f"  最大距离: {filtered_df['dist_to_limit'].max():.6f}")
    print(f"  平均距离: {filtered_df['dist_to_limit'].mean():.6f}")
    print(f"  距离<5%的tick数: {(filtered_df['dist_to_limit'] < 0.05).sum()}")

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*60)
    diagnose_failure_case()