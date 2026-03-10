# 基于 Tick 数据的涨停预测系统

**版本**：V1.0  
**建模目标**：预测下一个 tick 是否涨停  
**模型方案**：统计特征 + LightGBM 分类  
**数据粒度**：单股票、单交易日、tick 级盘口数据

---

## 项目目标

本项目旨在基于股票 Tick 级盘口数据，构建一个监督学习分类模型，在当前 tick 时刻预测：

```
下一个 tick 是否达到涨停价
```

即在时刻 `t`，根据当前及历史盘口信息，预测标签：

```python
y_t = 1 if current[t+1] >= limit_price else 0
y_t = 0 otherwise
```

系统第一阶段采用轻量、可解释、易实现的方案：

- 手工构造 15 个核心特征
- 使用 LightGBM 训练二分类模型
- 输出每个 tick 的涨停概率

---

## 数据组织结构

当前数据组织形式如下：

```
2025/
  2025-01-02.7z
  2025-01-03.7z
  2025-01-06.7z
  ...
```

每个 `.7z` 文件表示**一个交易日**，解压后包含多只股票的 CSV：

```
2025-01-02/
  000001.csv
  000002.csv
  000004.csv
  ...
```

其中：

- **一个 7z = 一个交易日**
- **一个 CSV = 一只股票当天的 tick 数据**
- **CSV 中每一行 = 该股票当天的一个 tick / 盘口快照**

因此，系统的最小训练样本单位为：

```
某天 / 某只股票 / 某一个 tick
```

---

## 原始字段说明

CSV 字段格式如下：

```
time, open, current, high, low,
total_volume, total_money, volume, money,
a5_v, a5_p, a4_v, a4_p, a3_v, a3_p, a2_v, a2_p, a1_v, a1_p,
b1_v, b1_p, b2_v, b2_p, b3_v, b3_p, b4_v, b4_p, b5_v, b5_p,
b/s
```

主要字段含义如下：

| 字段 | 含义 |
| ---- | ---- |
| time | tick 时间戳 |
| current | 当前最新价 |
| open / high / low | 当前时点相关价格 |
| total_volume | 累计成交量 |
| total_money | 累计成交额 |
| volume | 当前 tick 成交量或增量成交量 |
| money | 当前 tick 成交额或增量成交额 |
| a1_p ~ a5_p | 卖一到卖五价格 |
| a1_v ~ a5_v | 卖一到卖五挂单量 |
| b1_p ~ b5_p | 买一到买五价格 |
| b1_v ~ b5_v | 买一到买五挂单量 |
| b/s | 买卖方向标记 |

---

## 预测目标定义

### 主目标

系统主目标定义为：

```
预测下一个 tick 是否涨停
```

标签公式：

```python
label = 1 if current.shift(-1) >= limit_price else 0
```

其中：

- `current.shift(-1)` 表示下一条 tick 记录的最新价
- `limit_price` 表示当日涨停价

该目标本质上是**next-event classification**，而不是固定秒数预测，因为 tick 时间间隔不固定。

### 涨停价定义

涨停价需要基于昨收价 `preclose` 计算：

```python
limit_price = round(preclose * limit_ratio, 2)
```

其中 `limit_ratio` 需按股票类别确定，例如：

- 普通 A 股：10%
- ST：5%
- 创业板 / 科创板：20%

注意：当前 CSV 字段中未明确包含 `preclose`，因此后续需要通过以下方式之一补充：

1. 原始数据中若存在昨收字段，直接读取
2. 从前一交易日收盘价推导
3. 从外部日线/基础行情表合并 `preclose` 和 `limit_ratio`

---

## 数据预处理方案

### 基本处理流程

每个 CSV 需要执行以下步骤：

1. 读取文件
2. 按 `time` 升序排序
3. 清洗无效数据
4. 构造特征
5. 生成标签
6. 输出标准化样本表

### 无效数据过滤

在开盘前后或集合竞价阶段，部分行存在如下情况：

- `current = 0`
- `open = 0`
- `high = 0`
- `low = 0`
- `volume = 0`
- `money = 0`

这类记录通常为初始化快照或无效行情，不应参与训练。建议过滤规则：

```python
df = df[df["current"] > 0].copy()
```

必要时增加：

```python
df = df[(df["a1_p"] > 0) | (df["b1_p"] > 0)]
```

### 时间字段处理

原始 `time` 类似：

```
20250102091500
20250102091509
20250102091518
```

需转换为标准时间格式，便于排序与后续计算：

```python
df["time"] = pd.to_datetime(df["time"].astype(str), format="%Y%m%d%H%M%S")
```

---

## 特征工程设计

系统第一版使用 **15 个核心特征**，重点覆盖以下四类信息：

1. 距离涨停还有多远
2. 盘口买卖力量对比
3. 盘口结构是否有利于封板
4. 最近一个 tick 是否在加速逼近涨停

### 涨停接近度特征（4 个）

#### 特征 1：距离涨停比例

```python
f1 = (limit_price - current) / limit_price
```

#### 特征 2：距离涨停的最小价格跳数

```python
f2 = (limit_price - current) / tick_size
```

#### 特征 3：卖一价格到涨停的距离

```python
f3 = (limit_price - a1_p) / limit_price
```

#### 特征 4：卖一价与当前价的差距

```python
f4 = (a1_p - current) / current
```

### 盘口买卖强弱特征（5 个）

#### 特征 5：买盘总量

```python
f5 = b1_v + b2_v + b3_v + b4_v + b5_v
```

#### 特征 6：卖盘总量

```python
f6 = a1_v + a2_v + a3_v + a4_v + a5_v
```

#### 特征 7：盘口不平衡度

```python
f7 = (f5 - f6) / (f5 + f6 + 1e-9)
```

#### 特征 8：买一挂单量

```python
f8 = b1_v
```

#### 特征 9：卖一挂单量

```python
f9 = a1_v
```

### 盘口价格结构特征（3 个）

#### 特征 10：买卖一价差

```python
f10 = a1_p - b1_p
```

#### 特征 11：卖盘厚度斜率

```python
f11 = a1_v - a5_v
```

#### 特征 12：买盘厚度斜率

```python
f12 = b1_v - b5_v
```

### 成交与动量特征（3 个）

#### 特征 13：最近一跳收益率

```python
f13 = current / current.shift(1) - 1
```

#### 特征 14：最近成交量变化

若 `volume` 为累计成交量，则：

```python
f14 = volume.diff()
```

若 `volume` 本身就是增量成交量，则直接使用。

#### 特征 15：最近成交额变化

若 `money` 为累计成交额，则：

```python
f15 = money.diff()
```

若 `money` 本身为增量成交额，则直接使用。

---

## 标签生成方案

### 主标签

标签定义如下：

```python
df["label"] = (df["current"].shift(-1) >= df["limit_price"]).astype(int)
```

含义为：

- 当前 tick 的输入特征，用来预测**下一条 tick 是否涨停**

### 边界处理

由于最后一条记录没有下一条 tick，因此不能生成标签，应删除：

```python
df = df.iloc[:-1].copy()
```

或在 `dropna` 阶段处理。

---

## 单股票样本构建流程

对单个股票 CSV 的处理流程如下：

```
读取 CSV
  ↓
按时间排序
  ↓
过滤无效行
  ↓
补充 preclose / limit_price
  ↓
生成 15 个核心特征
  ↓
生成 next_tick_limit_up 标签
  ↓
输出标准样本表
```

输出后的样本结构应为：

| date | code | time | f1 | f2 | ... | f15 | label |
| ---- | ---- | ---- | -: | -: | --: | --: | ----: |

其中：

- `date` 从压缩包名或时间戳中提取
- `code` 从文件名中提取

---

## 多日、多股票数据集构建

### 批量处理流程

完整数据集构建流程如下：

```
遍历每个交易日 7z
  ↓
解压当天全部股票 CSV
  ↓
逐个 CSV 读取与处理
  ↓
构造特征和标签
  ↓
拼接为统一训练表
  ↓
保存为 parquet
```

### 最终数据集格式

推荐最终输出为：

```
dataset.parquet
```

字段示例：

| date | code | time | f1 | f2 | ... | f15 | label |
| ---- | ---- | ---- | -: | -: | --: | --: | ----: |

使用 Parquet 的原因：

- 读写速度快
- 体积小
- 适合大规模批量训练
- 适合后续按日期切分训练集和验证集

---

## 训练集与验证集划分

### 划分原则

训练集、验证集、测试集必须按**交易日**划分，不能随机按 tick 打散。

原因：

- 同一天内相邻 tick 高度相关
- 若随机切分，同一天的相邻 tick 同时出现在训练集和验证集中，会造成严重数据泄漏

### 推荐划分方式

推荐使用按日期排序的时间切分：

```
训练集：前 70% 交易日
验证集：中间 15% 交易日
测试集：后 15% 交易日
```

例如：

```
训练：2025-01-02 ～ 2025-08-31
验证：2025-09-01 ～ 2025-10-15
测试：2025-10-16 ～ 2025-12-31
```

### 切分代码示意

```python
dates = sorted(df["date"].unique())

n = len(dates)
train_end = int(n * 0.7)
valid_end = int(n * 0.85)

train_dates = dates[:train_end]
valid_dates = dates[train_end:valid_end]
test_dates = dates[valid_end:]

train_df = df[df["date"].isin(train_dates)].copy()
valid_df = df[df["date"].isin(valid_dates)].copy()
test_df = df[df["date"].isin(test_dates)].copy()
```

---

## 模型方案

### 模型选择

第一版采用：

```
LightGBM Classifier
```

原因：

- 对表格特征效果好
- 训练速度快
- 能处理非线性关系
- 支持类别不平衡参数
- 易解释，便于做特征重要性分析

### 训练目标

模型输入：

```
15 个核心特征
```

模型输出：

```
P(下一个 tick 涨停)
```

### 类别不平衡处理

由于"下一个 tick 涨停"是极少数事件，训练时必须考虑类别不平衡。建议：

- 使用 `scale_pos_weight`
- 必要时对负样本做下采样
- 核心关注 AUC、PR-AUC、Recall@TopK，而不是单纯 accuracy

示例：

```python
scale_pos_weight = neg_count / max(pos_count, 1)
```

---

## 模型评估指标

由于标签极度稀疏，不推荐仅使用 Accuracy。建议使用以下指标：

### 核心指标

- **ROC-AUC**
- **PR-AUC**
- **Recall**
- **Precision**
- **F1-score**

### 交易视角指标

建议增加：

- **Top-K 命中率**
- **概率阈值下的精确率**
- **高概率样本覆盖率**

这些指标更接近真实交易使用场景。

---

## 项目目录结构

推荐目录结构如下：

```
tick_limitup/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── extract_7z.py
│   ├── build_dataset.py
│   └── split_dataset.py
├── features/
│   └── build_features.py
├── models/
│   ├── train_lgbm.py
│   └── evaluate.py
├── utils/
│   └── data_utils.py
└── main.py
```

---

## 第一阶段实现范围

第一阶段目标定义为：

1. 能解压单日 7z 文件
2. 能读取单个股票 CSV
3. 能清洗无效 tick 数据
4. 能生成 15 个核心特征
5. 能生成 `next_tick_limit_up` 标签
6. 能构建统一 parquet 数据集
7. 能基于 LightGBM 训练二分类模型
8. 能输出每个 tick 的涨停概率

---

## 风险与注意事项

### 涨停价来源问题

当前 CSV 未明确包含 `preclose`，需优先解决，否则无法准确定义标签。

### tick 时间不规则

该任务本质是"预测下一事件是否涨停"，不是"预测未来 1 秒是否涨停"。

### 开盘前无效行情较多

需清洗大量 `current=0`、`volume=0` 的行。

### 类别严重不平衡

需在训练和评估中采用不平衡处理策略。

### 涨停规则需按股票类别区分

后续必须支持普通股、ST、创业板、科创板等不同涨停幅度规则。

---

## 后续迭代方向

在第一版基础上，后续可逐步升级为：

### 标签升级

- 预测未来 5 个 tick 是否涨停
- 预测未来 N 秒是否触及涨停
- 预测触板后是否封住

### 特征升级

- 扩展到 50~200 个盘口微观结构特征
- 加入技术指标与多窗口统计特征
- 引入盘口变化速度、撤单强度等高级特征

### 模型升级

- LightGBM 多任务学习
- Temporal CNN / LSTM / Transformer
- 多标签分类与概率校准

---

## 技术路线总结

本系统第一版的技术路线：

```
数据：单日 7z + 多股票 CSV + tick 级盘口数据
目标：预测下一个 tick 是否涨停
特征：15 个核心盘口与价格特征
模型：LightGBM 分类器
切分：按交易日做时间切分
输出：每个 tick 的涨停概率
```

该方案具有以下特点：

- 贴近真实盘口交易决策
- 实现成本低
- 可解释性强
- 便于快速完成最小可用版本
- 可自然扩展到更复杂的涨停预测体系
