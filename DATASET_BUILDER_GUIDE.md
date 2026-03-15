# 数据集构建器使用指南

## 概述

新的数据集构建器实现了**两步处理策略 + 自动清理**，解决了临时文件占用大量磁盘空间的问题。

## 核心特性

- **两步分离处理**：解压 → 处理 → 清理
- **资源优化**：处理完单日后立即清理临时文件
- **简洁设计**：专注于单日处理，移除批量复杂度
- **错误处理**：处理失败时提供详细错误信息
- **灵活配置**：支持自定义输出路径和参数
- **命令行接口**：支持直接命令行调用

## 使用方法

### 1. 命令行使用

```bash
# 基本使用
python scripts/dataset_builder.py \
  --input "2025/1/2025-01-02.7z" \
  --output "d:/Qoder-project/deep project/data"

# 自定义输出路径
python scripts/dataset_builder.py \
  --input "2025/1/2025-01-02.7z" \
  --output "d:/Qoder-project/deep project/data/2025-01-02_dataset.csv"

# 划分数据集
python scripts/dataset_builder.py \
  --input "2025/1/2025-01-02.7z" \
  --output "d:/Qoder-project/deep project/data" \
  --split

# 自定义数据集划分比例
python scripts/dataset_builder.py \
  --input "2025/1/2025-01-02.7z" \
  --output "d:/Qoder-project/deep project/data" \
  --split \
  --train-ratio 0.80 \
  --val-ratio 0.10
```

### 2. Python代码调用

```python
from scripts.dataset_builder import process_single_day_two_step

# 基本使用
dataset_path = process_single_day_two_step(
    archive_path="2025/1/2025-01-02.7z",
    output_dir="d:/Qoder-project/deep project/data"
)

# 划分数据集
dataset_path = process_single_day_two_step(
    archive_path="2025/1/2025-01-02.7z",
    output_dir="d:/Qoder-project/deep project/data",
    split_dataset=True,
    train_ratio=0.80,
    val_ratio=0.10
)

# 检查结果
if dataset_path:
    print(f"数据集构建成功: {dataset_path}")
else:
    print("数据集构建失败")
```

### 3. 使用DatasetBuilder类

```python
from scripts.dataset_builder import DatasetBuilder

# 创建构建器实例
builder = DatasetBuilder(
    archive_path="2025/1/2025-01-02.7z",
    output_base_dir="d:/Qoder-project/deep project/data"
)

# 步骤1: 解压
extract_dir = builder.step1_extract()

# 步骤2: 处理并清理
output_path = "d:/Qoder-project/deep project/data/2025-01-02_dataset.csv"
success = builder.step2_process_and_cleanup(
    output_path=output_path,
    split_dataset=True
)
```

## 处理流程

### 两步处理策略

1. **步骤1: 解压**
   - 从7z文件中提取所有股票数据
   - 解压到临时目录: `temp_extract/{date}/`
   - 显示解压结果

2. **步骤2: 处理 + 清理**
   - 处理所有股票CSV文件
   - 提取特征和生成标签
   - 保存最终数据集
   - **自动清理临时目录**
   - 可选：划分训练/验证/测试集

### 资源管理

- ✅ **临时目录结构**: `temp_extract/{date}/`
- ✅ **自动清理**: 处理完成后立即删除
- ✅ **输出文件**: `{date}_dataset.csv`
- ✅ **划分数据集**: `split_datasets/{date}/` (可选)

## 命令行参数

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | ✓ | - | 7z文件路径 |
| `--output` | ✗ | `d:/Qoder-project/deep project/data` | 输出目录 |
| `--split` | ✗ | False | 是否划分数据集 |
| `--train-ratio` | ✗ | 0.70 | 训练集比例 |
| `--val-ratio` | ✗ | 0.15 | 验证集比例 |

## 输出文件

### 主要数据集
- **位置**: `{output_dir}/{date}_dataset.csv`
- **内容**: 处理后的股票数据，包含特征和标签
- **格式**: CSV (兼容pandas直接读取)

### 划分数据集 (可选)
- **位置**: `{output_base_dir}/split_datasets/{date}/`
- **文件**:
  - `train.parquet`: 训练集
  - `val.parquet`: 验证集
  - `test.parquet`: 测试集

## 验证和测试

### 运行验证测试
```bash
python test_dataset_builder.py
```

### 手动验证
```bash
# 1. 测试单日处理
python scripts/dataset_builder.py --input "2025/1/2025-01-02.7z"

# 2. 验证临时目录清理
ls temp_extract/  # 应该为空或只有当前处理的日期

# 3. 验证数据集完整性
python -c "import pandas as pd; df = pd.read_csv('d:/Qoder-project/deep project/data/2025-01-02_dataset.csv'); print(f'Total samples: {len(df)}')"
```

## 错误处理

### 常见问题

1. **ImportError**: 缺少依赖库
   ```bash
   pip install py7zr pandas tqdm
   ```

2. **FileNotFoundError**: 7z文件路径错误
   - 检查文件路径是否正确
   - 确保使用正确的路径分隔符

3. **Empty dataset**: 没有生成有效数据
   - 检查7z文件是否包含有效的CSV文件
   - 查看跳过股票的日志信息

### 错误信息

- ✅ **步骤1失败**: 显示解压错误详情
- ✅ **步骤2失败**: 显示处理错误详情
- ✅ **清理失败**: 显示清理错误详情 (不影响数据处理结果)

## 性能特点

### 资源优化
- **内存占用**: 单日处理，内存占用可控
- **磁盘空间**: 自动清理临时文件，节省空间
- **处理速度**: 支持进度条显示，便于监控

### 适用场景
- ✅ 单日数据处理
- ✅ 资源受限环境
- ✅ 需要频繁测试的场景
- ✅ 开发和调试阶段

## 与原有代码的兼容性

### 保持不变的模块
- `scripts/extract_7z.py` - 解压逻辑保持不变
- `scripts/build_dataset.py` - 数据处理逻辑保持不变
- `src/feature_engineering/` - 特征工程模块保持不变
- `src/data_processing/` - 数据处理模块保持不变

### 新增模块
- `scripts/dataset_builder.py` - 两步处理核心逻辑

### 保留的参考文件
- `process_extracted_day.py` - 原有处理逻辑，供参考

## 总结

新的数据集构建器提供了更高效、更节省资源的处理方式，特别适合单日数据处理和开发测试场景。两步分离策略确保了资源的及时释放，避免了大量临时文件的积累。