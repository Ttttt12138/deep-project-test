"""
训练LightGBM模型用于涨停预测
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.lgbm_trainer import (
    prepare_training_data, train_lgbm_classifier,
    evaluate_model, get_feature_importance, predict_limit_up_probability
)
from scripts.build_dataset import get_feature_columns
from src.data_processing.sampling import automatic_undersample_if_needed


def train_from_split_datasets(split_dir="data/processed/split_datasets",
                              output_dir="models",
                              model_name="lgbm_model_20250106"):
    """
    从划分好的数据集训练模型

    Args:
        split_dir: 划分后的数据集目录
        output_dir: 模型输出目录
        model_name: 模型名称
    """
    print("="*80)
    print("LightGBM模型训练")
    print("="*80)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取划分后的数据集
    print("\n加载数据集...")
    train_path = os.path.join(split_dir, "train.parquet")
    val_path = os.path.join(split_dir, "validation.parquet")
    test_path = os.path.join(split_dir, "test.parquet")

    if not os.path.exists(train_path):
        print(f"错误: 训练集文件不存在: {train_path}")
        return False

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path) if os.path.exists(val_path) else train_df.sample(frac=0.15, random_state=42)
    test_df = pd.read_parquet(test_path) if os.path.exists(test_path) else train_df.sample(frac=0.15, random_state=43)

    print(f"训练集: {len(train_df):,} 样本")
    print(f"验证集: {len(val_df):,} 样本")
    print(f"测试集: {len(test_df):,} 样本")

    # 【可选】自动检测并执行欠采样（当正负样本比例 > 1:8 时）
    train_df = automatic_undersample_if_needed(
        train_df,
        dist_col='dist_to_limit',
        label_col='label',
        auto_threshold=8.0,
        thresholds=(0.01, 0.05),
        keep_ratios=(1.0, 0.3, 0.05),
        target_ratio=5.0,
        random_seed=42,
        verbose=True
    )

    # 获取特征列
    feature_cols = get_feature_columns(train_df)
    print(f"\n找到 {len(feature_cols)} 个特征")

    if len(feature_cols) == 0:
        print("错误: 未找到特征列")
        return False

    # 准备训练数据
    print("\n准备训练数据...")
    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_valid, y_valid = prepare_training_data(val_df, feature_cols)
    X_test, y_test = prepare_training_data(test_df, feature_cols)

    print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
    print(f"验证数据形状: X={X_valid.shape}, y={y_valid.shape}")
    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")

    # 显示标签分布
    print(f"\n标签分布:")
    print(f"训练集: 正样本 {y_train.sum():,} ({y_train.mean():.4%}), 负样本 {(~y_train).sum():,}")
    print(f"验证集: 正样本 {y_valid.sum():,} ({y_valid.mean():.4%}), 负样本 {(~y_valid).sum():,}")
    print(f"测试集: 正样本 {y_test.sum():,} ({y_test.mean():.4%}), 负样本 {(~y_test).sum():,}")

    # 训练模型
    print("\n开始训练LightGBM模型...")
    model = train_lgbm_classifier(X_train, y_train, X_valid, y_valid)

    # 评估模型
    print("\n评估模型性能...")
    metrics = evaluate_model(model, X_test, y_test)

    print("\n模型评估结果:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    # 特征重要性
    print("\n特征重要性 (Top 15):")
    importance_df = get_feature_importance(model, feature_cols)
    print(importance_df.head(15).to_string(index=False))

    # 保存模型
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"\n模型已保存到: {model_path}")

    # 保存特征重要性
    importance_path = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"特征重要性已保存到: {importance_path}")

    # 保存评估结果
    results_path = os.path.join(output_dir, f"{model_name}_metrics.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"模型训练结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write("数据集统计:\n")
        f.write(f"  训练集: {len(train_df):,} 样本\n")
        f.write(f"  验证集: {len(val_df):,} 样本\n")
        f.write(f"  测试集: {len(test_df):,} 样本\n")
        f.write(f"  特征数: {len(feature_cols)}\n\n")
        f.write("标签分布:\n")
        f.write(f"  训练集: 正样本 {y_train.sum():,} ({y_train.mean():.4%})\n")
        f.write(f"  验证集: 正样本 {y_valid.sum():,} ({y_valid.mean():.4%})\n")
        f.write(f"  测试集: 正样本 {y_test.sum():,} ({y_test.mean():.4%})\n\n")
        f.write("模型评估结果:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"  {metric_name}: {metric_value:.4f}\n")
    print(f"评估结果已保存到: {results_path}")

    return True


def train_from_single_dataset(dataset_path="data/processed/2025-01-06_dataset.csv",
                               output_dir="models",
                               model_name="lgbm_model_20250106",
                               train_ratio=0.70, val_ratio=0.15):
    """
    从单个数据集文件训练模型（会自动划分）

    Args:
        dataset_path: 数据集路径
        output_dir: 模型输出目录
        model_name: 模型名称
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    print("="*80)
    print("LightGBM模型训练（单数据集模式）")
    print("="*80)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据集
    print(f"\n加载数据集: {dataset_path}")
    df = pd.read_csv(dataset_path)

    if df.empty:
        print("错误: 数据集为空")
        return False

    print(f"总样本数: {len(df):,}")

    # 获取特征列
    feature_cols = get_feature_columns(df)
    print(f"找到 {len(feature_cols)} 个特征")

    if len(feature_cols) == 0:
        print("错误: 未找到特征列")
        return False

    # 按交易日划分数据集（如果有多个交易日）
    if 'date' in df.columns and df['date'].nunique() > 1:
        from src.data_processing.dataset_split import split_dataset_by_trading_day
        print(f"\n按交易日划分数据集... (共{df['date'].nunique()}个交易日)")
        train_df, val_df, test_df = split_dataset_by_trading_day(
            df, train_ratio=train_ratio, val_ratio=val_ratio,
            test_ratio=1.0 - train_ratio - val_ratio
        )
    else:
        from sklearn.model_selection import train_test_split
        print("\n随机划分数据集...")
        train_df, temp_df = train_test_split(df, test_size=(1-train_ratio), random_state=42)
        valid_ratio_adjusted = val_ratio / (1-train_ratio)
        val_df, test_df = train_test_split(temp_df, test_size=(1-valid_ratio_adjusted), random_state=42)

    print(f"训练集: {len(train_df):,} 样本 ({len(train_df)/len(df):.2%})")
    print(f"验证集: {len(val_df):,} 样本 ({len(val_df)/len(df):.2%})")
    print(f"测试集: {len(test_df):,} 样本 ({len(test_df)/len(df):.2%})")

    # 准备训练数据
    print("\n准备训练数据...")
    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_valid, y_valid = prepare_training_data(val_df, feature_cols)
    X_test, y_test = prepare_training_data(test_df, feature_cols)

    print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
    print(f"验证数据形状: X={X_valid.shape}, y={y_valid.shape}")
    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")

    # 显示标签分布
    print(f"\n标签分布:")
    print(f"训练集: 正样本 {y_train.sum():,} ({y_train.mean():.4%}), 负样本 {(~y_train).sum():,}")
    print(f"验证集: 正样本 {y_valid.sum():,} ({y_valid.mean():.4%}), 负样本 {(~y_valid).sum():,}")
    print(f"测试集: 正样本 {y_test.sum():,} ({y_test.mean():.4%}), 负样本 {(~y_test).sum():,}")

    # 训练模型
    print("\n开始训练LightGBM模型...")
    model = train_lgbm_classifier(X_train, y_train, X_valid, y_valid)

    # 评估模型
    print("\n评估模型性能...")
    metrics = evaluate_model(model, X_test, y_test)

    print("\n模型评估结果:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    # 特征重要性
    print("\n特征重要性 (Top 15):")
    importance_df = get_feature_importance(model, feature_cols)
    print(importance_df.head(15).to_string(index=False))

    # 保存模型
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"\n模型已保存到: {model_path}")

    # 保存特征重要性
    importance_path = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"特征重要性已保存到: {importance_path}")

    # 保存评估结果
    results_path = os.path.join(output_dir, f"{model_name}_metrics.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"模型训练结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write("数据集统计:\n")
        f.write(f"  总样本数: {len(df):,}\n")
        f.write(f"  训练集: {len(train_df):,} 样本 ({len(train_df)/len(df):.2%})\n")
        f.write(f"  验证集: {len(val_df):,} 样本 ({len(val_df)/len(df):.2%})\n")
        f.write(f"  测试集: {len(test_df):,} 样本 ({len(test_df)/len(df):.2%})\n")
        f.write(f"  特征数: {len(feature_cols)}\n\n")
        f.write("标签分布:\n")
        f.write(f"  训练集: 正样本 {y_train.sum():,} ({y_train.mean():.4%})\n")
        f.write(f"  验证集: 正样本 {y_valid.sum():,} ({y_valid.mean():.4%})\n")
        f.write(f"  测试集: 正样本 {y_test.sum():,} ({y_test.mean():.4%})\n\n")
        f.write("模型评估结果:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"  {metric_name}: {metric_value:.4f}\n")
    print(f"评估结果已保存到: {results_path}")

    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='训练LightGBM模型')
    parser.add_argument('--mode', type=str, choices=['split', 'single'], default='split',
                       help='训练模式：split=使用划分后的数据集，single=使用单个数据集文件')
    parser.add_argument('--input', type=str, help='输入数据集路径（single模式）')
    parser.add_argument('--split-dir', type=str, default='data/processed/split_datasets',
                       help='划分后的数据集目录（split模式）')
    parser.add_argument('--output-dir', type=str, default='models', help='模型输出目录')
    parser.add_argument('--model-name', type=str, default='lgbm_model_20250106', help='模型名称')
    parser.add_argument('--train-ratio', type=float, default=0.70, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='验证集比例')

    args = parser.parse_args()

    if args.mode == 'split':
        success = train_from_split_datasets(
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            model_name=args.model_name
        )
    else:
        if not args.input:
            print("错误: single模式需要--input参数")
            sys.exit(1)

        success = train_from_single_dataset(
            dataset_path=args.input,
            output_dir=args.output_dir,
            model_name=args.model_name,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )

    if success:
        print("\n模型训练成功！")
        sys.exit(0)
    else:
        print("\n模型训练失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()