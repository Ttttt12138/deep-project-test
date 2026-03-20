"""
V3.2中"有触板事件但被过滤"问题的根本原因分析
"""

"""
问题根因：边界过滤逻辑与正样本检测逻辑的冲突

## 核心逻辑分析

### 1. 正样本检测逻辑
```python
# 检测所有触板瞬间
all_touch_events = (next_is_limit & ~current_is_limit).astype(int)
```
- 检测：T10不涨停，T11涨停的瞬间
- 这个tick本身（T10）不是涨停状态

### 2. 过滤逻辑
```python
condition_a = df['label'] == 1                            # 正样本（首次触板）
condition_b = df['dist_to_limit'] < limit_dist_threshold  # 困难负样本
is_dead_limit = current_is_limit                          # 死板状态

keep_mask = (condition_a | condition_b) & (~is_dead_limit)
```
- 过滤条件：保留正样本 OR 困难负样本，但不能是死板状态
- 问题：`current_is_limit`判断当前tick是否涨停

### 3. 边界处理逻辑
```python
# 早盘数据免死金牌：只过滤最后一个tick
if len(df) > 0:
    keep_mask.iloc[-1] = False  # 过滤最后一个tick
```
- 问题：强制过滤最后一个tick，不管是否为正样本

## 问题的三种情况

### 情况1：边界过滤（最可能）
```
假设某股票只有50个tick的数据：

Tick 48: price = 11.9, limit = 12.0 (未涨停)
Tick 49: price = 12.0, limit = 12.0 (涨停)  <- 触板瞬间
Tick 50: price = 12.0, limit = 12.0 (涨停)

检测结果：
- Tick 48: label = 0, is_dead_limit = False
- Tick 49: label = 1, is_dead_limit = False  <- 正样本
- Tick 50: label = 0, is_dead_limit = True   <- 死板

过滤结果：
- Tick 48: condition_b可能为True（距离<2%），保留
- Tick 49: condition_a为True，is_dead_limit为False，应该保留
- Tick 50: 强制keep_mask.iloc[-1] = False，被过滤

问题：如果Tick 49恰好是最后一个tick，它会被边界过滤掉！
```

### 情况2：死板状态误判
```
由于浮点数精度问题，某些tick可能被错误判定为涨停状态：

price = 11.99999999  # 实际未涨停
limit = 12.0
current_is_limit = round(price, 2) >= round(limit, 2)
               = 11.99999999.round(2) >= 12.0
               = 12.0 >= 12.0
               = True  <- 错误判定为涨停！

结果：
- 这个tick的label可能是1（触板）
- 但is_dead_limit = True
- keep_mask = (True | ...) & ~True = False
- 被过滤掉
```

### 情况3：距离计算问题
```
如果正样本的dist_to_limit计算有误：

触板瞬间：
price_T10 = 11.99
limit = 12.0
dist_to_limit = (12.0 - 11.99) / 12.0 = 0.01 / 12.0 ≈ 0.0008 (0.08%)

按理说：
- condition_a = True（label=1）
- 无论dist_to_limit是多少，都应该保留

但如果代码逻辑有问题，导致：
- condition_a可能没有正确触发
- 或者is_dead_limit错误地设置为True
```

## 最可能的根本原因

基于代码逻辑分析，**最可能的原因是情况1：边界过滤**

具体来说：

1. **正样本检测**：检测到某股票在第49个tick触板（label=1）
2. **边界过滤**：由于这个tick是倒数第二个，而代码强制过滤最后一个tick
3. **实际结果**：如果触板发生在接近末尾的位置，会被边界过滤误伤

## 解决方案

### 方案1：保护正样本不被边界过滤
```python
# 修改边界过滤逻辑，保护正样本
if len(df) > 0:
    if isinstance(keep_mask, pd.Series):
        # 只过滤既不是正样本，也不是困难负样本的最后一个tick
        last_row_filter = ~((df['label'] == 1) | (df['dist_to_limit'] < limit_dist_threshold))
        keep_mask.iloc[-1] &= last_row_filter.iloc[-1]
```

### 方案2：优化边界条件
```python
# 不强制过滤最后一个tick，而是移除无法生成标签的tick
# 这已经在前面的逻辑中处理了（生成事件驱动标签时会移除最后一个tick）
```

### 方案3：增加调试信息
```python
# 在filter_and_label_events中增加详细日志
if debug and len(touch_indices) > 0:
    for idx in first_touch_indices:
        # 检查这个正样本是否被过滤
        if idx not in valid_indices:
            print(f"正样本被过滤: {idx}, 原因分析:")
            print(f"  label: {df.loc[idx, 'label']}")
            print(f"  is_dead_limit: {current_is_limit[idx]}")
            print(f"  dist_to_limit: {df.loc[idx, 'dist_to_limit']}")
            print(f"  是否为最后一个: {idx == df.index[-1]}")
```

## 验证方法

创建一个测试案例来重现这个问题：

```python
import pandas as pd

# 创建一个触板发生在倒数第二个位置的测试数据
test_data = {
    'time': range(50),
    'current': [10.0 + i * 0.04 for i in range(48)] + [11.92, 12.0],
    'limit_price': [12.0] * 50
}

df = pd.DataFrame(test_data)
df['code'] = 'TEST001'

# 检查第49个tick是否被正确处理
filtered_df, valid_indices = filter_and_label_events(df, limit_dist_threshold=0.05)

# 检查正样本是否被保留
positive_indices = filtered_df[filtered_df['label'] == 1].index
print(f"检测到的正样本: {positive_indices}")
print(f"保留的有效样本: {valid_indices}")
print(f"正样本是否被保留: {any(idx in valid_indices for idx in positive_indices)}")
```
"""

if __name__ == "__main__":
    print(__doc__)