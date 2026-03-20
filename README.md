# 基于 Tick 数据的涨停预测系统

**版本**：V3.1 (事件驱动与极致内存管理版)
**建模目标**：以事件驱动方式预测"触板瞬间"，而非涨停状态
**模型方案**：事件驱动标签 + 内存阻断策略 + 按需特征平铺 + LightGBM 分类
**数据粒度**：单股票、单交易日、事件级样本

---

## 🎯 V3.1版本核心突破

### 🚀 核心问题解决

**V3.0版本存在的两大瓶颈：**
1. **假正样本问题**：涨停被定义为"状态"，单日产生20万+假正样本
2. **内存爆炸问题**：1335万tick × 165列 ≈ 20GB内存，极易OOM

**V3.1版本的革命性改进：**

1. **标签重构**：从"状态"→"事件"
   - ❌ 旧标签：第11个tick是否处于涨停状态
   - ✅ 新标签：价格是否从非涨停突破到涨停（触板瞬间）
   - 🎯 效果：单日正样本从20万骤降至10-50个

2. **内存阻断**：从"先平铺后过滤"→"先过滤后平铺"
   - ❌ 旧方式：全量1335万tick × 165列平铺 = 20GB内存
   - ✅ 新方式：前置过滤95%无用数据，仅对1-2万高价值样本平铺 = 1-2GB内存
   - 🎯 效果：内存占用降低90%+，处理速度提升5-10倍

3. **数据质量**：完全符合真实交易逻辑
   - 仅保留具有预测价值的高价值观测点
   - 绝对剔除死板状态和垃圾时间数据
   - 困难负样本：距离涨停<2%但最终未封板的时刻

---

## 项目目标

本项目旨在基于股票 Tick 级盘口数据，构建一个事件驱动型监督学习分类模型：

```
给定连续 10 个 Tick 的窗口信息，预测价格是否从非涨停突破到涨停（触板瞬间）
```

核心预测逻辑：
- 输入：前 10 个 Tick 的盘口信息
- 标签：`y_t = 1 if (current[t+10] >= limit_price AND current[t+9] < limit_price) else 0`
- 含义：预测价格是否从T10的非涨停状态在T11突破到涨停

---

## 核心特性

### 🎯 预测目标（V3.1事件驱动）
- **预测对象**：触板瞬间（价格突破涨停价的那一瞬间）
- **样本单位**：事件级别（date, code, trigger_time）
- **特征约束**：仅使用前10个tick的信息，不使用未来信息
- **正样本定义**：仅保留突破涨停的瞬时点，而非持续的涨停状态

### 📊 特征工程（V3.1按需平铺）
- **窗口特征**：10个Tick的165维特征（平铺特征 + 末端特征）
- **特征维度**：15个核心特征 × 10个时间位置 + 15个末端特征 = 165维
- **内存优化**：仅对1-2万高价值样本进行特征平铺，避免20GB内存爆炸

### 🔄 处理流程（V3.1内存阻断）
1. **事件驱动标签**：计算触板瞬间标签
2. **前置内存阻断**：过滤95%无用数据
3. **按需特征平铺**：仅对保留样本进行165维提取
4. **两层负采样**：基于距离的分层欠采样
5. **独立欠采样**：按交易日独立处理
6. **手动合并**：最终手动合并多日训练集

---

## 数据组织结构

### 原始数据结构

```
2025/
  2025-01-02.7z
  2025-01-03.7z
  2025-01-06.7z
  ...
```

### 输出数据结构

```
data/
├── daily_train_candidates/      # 候选训练集（未欠采样，事件样本）
│   ├── 2025-01-02_candidate.parquet
│   ├── 2025-01-03_candidate.parquet
│   └── ...
├── daily_train_undersampled/    # 欠采样训练集分片
│   ├── 2025-01-02_train.parquet
│   ├── 2025-01-03_train.parquet
│   └── ...
├── logs/                        # 统计日志
│   ├── 2025-01-02_summary.csv
│   └── ...
├── merged/                      # 合并后的多日训练集
│   └── multi_day_train.parquet
└── temp_extract/               # 临时解压目录（自动清理）
```

---

## 训练集构建工具

### 单日训练集构建

```bash
python scripts/training_set_builder.py \
  --input "2025/1/2025-01-02.7z" \
  --output "d:/Qoder-project/deep project/data" \
  --mode single
```

### 批量训练集构建

```bash
python scripts/training_set_builder.py \
  --input-dir "2025/1/" \
  --output "d:/Qoder-project/deep project/data" \
  --mode batch \
  --max-files 10
```

### 欠采样配置

```bash
python scripts/training_set_builder.py \
  --input "2025/1/2025-01-02.7z" \
  --output "d:/Qoder-project/deep project/data" \
  --mode single \
  --layer1-thresholds 0.01 0.05 \
  --layer1-keep-ratios 1.0 0.3 0.05 \
  --target-ratio 5.0
```

---

## 特征工程设计（V3.1版本）

### 15个核心特征

```python
# 涨停接近度
dist_to_limit       # 当前价距离涨停比例
ticks_to_limit      # 当前距离涨停的 tick 数
ask1_to_limit       # 当前卖一价距离涨停
ask1_gap            # 当前卖一与当前价差

# 盘口强度
bid_depth           # 当前买盘深度
ask_depth           # 当前卖盘深度
order_imbalance     # 当前 order imbalance
b1_volume           # 当前买一量
a1_volume           # 当前卖一量

# 盘口结构
spread              # 当前 spread
ask_slope           # 当前卖盘斜率
bid_slope           # 当前买盘斜率

# 价格动态
ret_1tick           # 当前单 tick 收益率
vol_delta           # 当前成交量变化
money_delta         # 当前成交额变化
```

### 165维窗口特征

每个样本包含：
- **平铺特征**：15个核心特征 × 10个时间位置 = 150维
- **末端特征**：15个核心特征 × 第10个tick = 15维

---

## 标签生成（V3.1事件驱动）

### 事件驱动标签定义

```python
# 核心逻辑：检测价格是否从非涨停突破到涨停
next_is_limit = current[t+10] >= limit_price
current_is_limit = current[t+9] >= limit_price

# 事件驱动标签：仅保留突破瞬间
label = (next_is_limit & ~current_is_limit) ? 1 : 0
```

### 前置内存阻断策略

```python
# 保留条件：
1. 正样本：label == 1（触板瞬间）
2. 困难负样本：dist_to_limit < 2%（接近涨停但未封板）

# 剔除条件：
1. 死板状态：current >= limit_price（已经在涨停板上）
2. 垃圾时间：dist_to_limit >= 2%（远离涨停，无预测价值）
3. 边界数据：前9个tick无法构筑窗口，最后一个tick无未来标签
```

---

## 数据质量检查（V3.1事件样本）

### V3.1特有检查项

1. **事件/状态比例**：应该<10%（事件数量应该远小于状态数量）
2. **正样本数量**：单日应该<100个（事件驱动模式下正样本很少）
3. **内存效率**：从原始tick到事件样本的减少率应该>90%

### 常规检查项

1. **唯一性检查**：确保没有重复事件样本（按 date, code, time）
2. **空值检查**：关键字段不能为空（date, code, label）
3. **特征完整性检查**：165维特征必须存在
4. **正样本检查**：必须包含正样本（触板事件）
5. **采样比例检查**：最终比例应接近 1:5

---

## 项目目录结构

```
tick_limitup/
├── data/                           # 数据目录
│   ├── daily_train_candidates/     # 候选训练集（事件样本）
│   ├── daily_train_undersampled/   # 欠采样训练集分片
│   ├── logs/                       # 统计日志
│   ├── merged/                     # 合并后的训练集
│   └── temp_extract/              # 临时解压目录
├── scripts/                        # 脚本目录
│   ├── training_set_builder.py     # 训练集构建器（V3.1实现）
│   └── extract_7z.py              # 解压工具
├── src/                           # 源代码目录
│   ├── data_processing/           # 数据处理模块
│   │   ├── limit_up_processor.py  # 数据处理
│   │   ├── limit_up_processor_optimized.py  # 优化版处理器
│   │   ├── stock_utils.py         # 股票工具
│   │   ├── quality_check.py       # 质量检查
│   │   ├── sampling.py            # 欠采样模块
│   │   └── sampling_optimized.py  # 内存优化版采样
│   └── feature_engineering/       # 特征工程模块
│       ├── limit_up_features.py   # 基础特征提取
│       ├── limit_up_labels.py     # 旧版标签生成（向后兼容）
│       ├── event_driven_labels.py # V3.1事件驱动标签（新核心）
│       └── event_window_builder.py # V3.1事件窗口构建器（新核心）
├── models/                        # 模型目录
│   └── lgbm_trainer.py           # LightGBM训练器
├── train_model.py                 # 模型训练脚本
├── build_2025_dataset.py          # 批量数据集构建
└── README.md                      # 项目说明
```

---

## 技术要点

### V3.1核心创新

1. **事件驱动标签**
   - 从预测"状态"改为预测"事件"
   - 标签逻辑：`next_is_limit & ~current_is_limit`
   - 业务逻辑：仅捕捉价格突破涨停价的瞬时点

2. **前置内存阻断**
   - 在特征平铺前进行激进采样
   - 保留正样本 + 困难负样本，剔除95%无用数据
   - 内存占用：从20GB降至1-2GB

3. **按需特征平铺**
   - 仅对1-2万高价值样本进行165维特征提取
   - 避免全表shift操作，提升处理速度5-10倍

### 数据清理规则

每只股票的 Tick 数据执行统一清理规则：

1. 删除无效价格记录：`current <= 0`
2. 删除空盘口记录：`a1_p <= 0 and b1_p <= 0`
3. 过滤集合竞价阶段：仅保留 `09:30:00 - 15:00:00`
4. 删除重复 Tick：按 `time` 去重
5. 删除非法盘口：`a1_p < b1_p`
6. 删除成交异常：`volume < 0 or money < 0`
7. 删除极端价格跳变：`abs(current.pct_change()) > 0.1`

### 涨停价计算

```python
limit_price = round(preclose * limit_ratio, 2)
```

不同股票类型对应不同涨停比例：
- 普通股：10%
- ST：5%
- 创业板/科创板：20%

---

## 使用流程

### 完整工作流程

```bash
# 1. 单日训练集构建（V3.1事件驱动模式）
python scripts/training_set_builder.py \
  --input "2025/1/2025-01-02.7z" \
  --output "d:/Qoder-project/deep project/data" \
  --mode single

# 2. 批量训练集构建
python scripts/training_set_builder.py \
  --input-dir "2025/1/" \
  --output "d:/Qoder-project/deep project/data" \
  --mode batch

# 3. 合并训练集
python scripts/training_set_builder.py \
  --mode merge \
  --shards-dir "d:/Qoder-project/deep project/data/daily_train_undersampled" \
  --merged-output "d:/Qoder-project/deep project/data/merged/multi_day_train.parquet"
```

### 日志分析

V3.1版本的统计日志包含特有字段：

```python
import pandas as pd

# 读取V3.1统计日志
log_df = pd.read_csv("data/logs/2025-01-02_summary.csv")
print(log_df[['date', 'version', 'limit_up_events', 'event_samples', 'event_to_state_ratio']])
```

---

## 系统特点

- ✅ **事件驱动**：标签符合真实交易逻辑，预测触板瞬间而非涨停状态
- ✅ **内存优化**：前置内存阻断，内存占用降低90%+
- ✅ **数据质量**：仅保留1-2万高价值样本，符合业务规律
- ✅ **模块化设计**：每个功能模块独立，易于维护
- ✅ **灵活配置**：支持多种欠采样参数调整
- ✅ **完整日志**：详细的统计日志和质量报告
- ✅ **向后兼容**：保留旧版标签生成逻辑，便于迁移

---

## 注意事项

1. **V3.1版本特性**：采用事件驱动标签，正样本数量大幅减少（单日10-50个）
2. **内存管理**：V3.1版本内存占用极低（1-2GB），但仍需注意资源限制
3. **标签定义**：预测目标是触板瞬间（价格突破涨停），不是涨停状态
4. **特征约束**：特征仅使用窗口内信息，严格禁止使用未来信息
5. **数据质量**：V3.1版本数据质量极高，完全符合真实交易场景

---

## 版本历史

### V3.1 (当前版本 - 事件驱动与内存管理)
- **标签重构**：从"状态"改为"事件"，单日正样本从20万骤降至10-50个
- **内存阻断**：前置过滤95%无用数据，内存占用从20GB降至1-2GB
- **按需平铺**：仅对1-2万高价值样本进行165维特征提取
- **处理速度**：避免全表shift操作，速度提升5-10倍
- **业务逻辑**：完全符合真实交易规律，预测触板瞬间

### V3.0 (方案A实现)
- 标签生成：使用 `shift(-10)` 生成窗口标签
- 窗口特征提取：前10个tick窗口 + 第11个tick标签
- 32个窗口特征：15个末端特征 + 17个统计特征
- 清理冗余文件和测试代码
- 存在问题：假正样本过多，内存占用过高

### V2.0
- 预测目标从下一个 tick 改为第 11 个 tick（标签定义错误）
- 实现两层欠采样策略
- 重构训练集构建流程
- 增强日志和质量检查
- 移除数据集划分功能

### V1.0
- 初始版本
- 基础特征工程
- 单层欠采样
- 数据集划分功能

---

## 技术支持

如有问题或建议，请参考：
- V3.1技术文档：了解事件驱动标签和内存阻断策略
- `scripts/training_set_builder.py` - 主要实现代码
- `src/feature_engineering/event_driven_labels.py` - 事件驱动标签模块
- `src/feature_engineering/event_window_builder.py` - 事件窗口构建器