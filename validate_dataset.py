"""
数据集验证脚本
"""
import pandas as pd

print("=" * 60)
print("数据集验证")
print("=" * 60)

# 读取数据集
print("正在读取数据集...")
df = pd.read_csv('data/2025-01-02_dataset.csv')

print(f"样本数量: {len(df):,}")

# 计算特征数量
base_cols = {'time', 'current', 'limit_price', 'code', 'date', 'label'}
feature_cols = [col for col in df.columns if col not in base_cols]
print(f"特征数量: {len(feature_cols)}")

# 标签统计
if 'label' in df.columns:
    positive_count = df['label'].sum()
    negative_count = len(df) - positive_count
    positive_ratio = df['label'].mean()

    print(f"正样本数量: {positive_count:,}")
    print(f"负样本数量: {negative_count:,}")
    print(f"正样本比例: {positive_ratio:.4%}")

print("\n数据集基本信息:")
print(df.info())

print("\n前3行数据:")
print(df.head(3))

print("\n特征列:")
for i, col in enumerate(feature_cols, 1):
    print(f"{i}. {col}")

print("\n" + "=" * 60)
print("验证完成!")
print("=" * 60)