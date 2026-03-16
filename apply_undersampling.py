"""
对已有数据集执行欠采样
读取现有的数据集文件，划分数据集，并对训练集执行欠采样
"""

import pandas as pd
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.dataset_split import split_dataset_by_trading_day, save_split_datasets
from src.data_processing.sampling import undersample_train_set, get_sampling_statistics
from src.feature_engineering.limit_up_labels import get_label_statistics


def apply_undersampling_to_existing_dataset(
    input_path,
    output_dir,
    train_ratio=0.70,
    val_ratio=0.15,
    thresholds=(0.01, 0.05),
    keep_ratios=(1.0, 0.3, 0.05),
    target_ratio=5.0,
    random_seed=42,
    format='parquet'
):
    """
    对已有数据集执行欠采样

    Args:
        input_path: 输入数据集路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        thresholds: 欠采样分层阈值
        keep_ratios: 各层保留率
        target_ratio: 目标正负比例
        random_seed: 随机种子
        format: 输出格式 ('parquet' 或 'csv')
    """
    print("="*80)
    print("已有数据集欠采样处理")
    print("="*80)

    # 读取数据集
    print(f"\n[1] 读取数据集: {input_path}")
    try:
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        else:
            raise ValueError("不支持的文件格式，仅支持 .csv 或 .parquet")

        print(f"    成功读取 {len(df):,} 样本")
        print(f"    特征数: {len(df.columns)}")
        print(f"    股票数: {df['code'].nunique() if 'code' in df.columns else 'N/A'}")
        print(f"    交易日数: {df['date'].nunique() if 'date' in df.columns else 'N/A'}")

    except Exception as e:
        print(f"    读取失败: {e}")
        return False

    # 检查必需列
    print(f"\n[2] 检查数据质量")
    required_cols = ['dist_to_limit', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"    错误: 缺少必需列 {missing_cols}")
        return False

    print(f"    所有必需列存在")

    # 显示原始数据集统计
    print(f"\n[3] 原始数据集统计")
    if 'label' in df.columns:
        stats = get_label_statistics(df)
        print(f"    总样本数: {stats['total_samples']:,}")
        print(f"    正样本数: {stats['positive_samples']:,} ({stats['positive_ratio']:.4%})")
        print(f"    负样本数: {stats['negative_samples']:,} ({stats['negative_ratio']:.4%})")
        print(f"    正负比例: 1:{stats['imbalance_ratio']:.2f}")

    # 按交易日划分数据集
    print(f"\n[4] 划分数据集")

    # 检查是否只有一个交易日
    if 'date' in df.columns and df['date'].nunique() == 1:
        print(f"    检测到单日数据集，使用随机划分")
        from sklearn.model_selection import train_test_split

        # 首先分离出测试集
        train_val_df, test_df = train_test_split(
            df, test_size=(1.0 - train_ratio - val_ratio), random_state=42
        )

        # 然后从剩余数据中分离训练集和验证集
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio_adjusted, random_state=42
        )

    else:
        print(f"    训练集比例: {train_ratio:.1%}")
        print(f"    验证集比例: {val_ratio:.1%}")
        print(f"    测试集比例: {1.0 - train_ratio - val_ratio:.1%}")

        train_df, val_df, test_df = split_dataset_by_trading_day(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=1.0 - train_ratio - val_ratio
        )

    print(f"    训练集: {len(train_df):,} 样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    验证集: {len(val_df):,} 样本 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    测试集: {len(test_df):,} 样本 ({len(test_df)/len(df)*100:.1f}%)")

    # 显示训练集原始标签分布
    print(f"\n[5] 训练集原始标签分布")
    train_stats = get_label_statistics(train_df)
    print(f"    正样本: {train_stats['positive_samples']:,} ({train_stats['positive_ratio']:.4%})")
    print(f"    负样本: {train_stats['negative_samples']:,} ({train_stats['negative_ratio']:.4%})")
    print(f"    正负比例: 1:{train_stats['imbalance_ratio']:.2f}")

    # 对训练集执行欠采样
    print(f"\n[6] 执行负样本欠采样")
    print(f"    分层阈值: {thresholds}")
    print(f"    保留率: {keep_ratios}")
    print(f"    目标比例: 1:{target_ratio}")

    train_df_undersampled = undersample_train_set(
        train_df,
        dist_col='dist_to_limit',
        label_col='label',
        thresholds=thresholds,
        keep_ratios=keep_ratios,
        target_ratio=target_ratio,
        random_seed=random_seed,
        verbose=True
    )

    # 显示欠采样后统计
    print(f"\n[7] 欠采样效果汇总")
    train_under_stats = get_label_statistics(train_df_undersampled)

    print(f"    样本数量变化:")
    print(f"      训练集: {len(train_df):,} → {len(train_df_undersampled):,} " +
          f"(减少 {(1-len(train_df_undersampled)/len(train_df))*100:.1f}%)")
    print(f"      验证集: {len(val_df):,} (保持不变)")
    print(f"      测试集: {len(test_df):,} (保持不变)")

    print(f"    标签分布变化:")
    pos_change = train_under_stats['positive_samples'] - train_stats['positive_samples']
    neg_change = train_under_stats['negative_samples'] - train_stats['negative_samples']
    print(f"      训练集正样本: {train_stats['positive_samples']:,} → {train_under_stats['positive_samples']:,} " +
          f"({pos_change:+,})")
    print(f"      训练集负样本: {train_stats['negative_samples']:,} → {train_under_stats['negative_samples']:,} " +
          f"({neg_change:+,})")
    print(f"      训练集比例: 1:{train_stats['imbalance_ratio']:.2f} → 1:{train_under_stats['imbalance_ratio']:.2f}")

    # 验证集和测试集分布
    val_stats = get_label_statistics(val_df)
    test_stats = get_label_statistics(test_df)
    print(f"      验证集比例: 1:{val_stats['imbalance_ratio']:.2f} (保持不变)")
    print(f"      测试集比例: 1:{test_stats['imbalance_ratio']:.2f} (保持不变)")

    # 保存结果
    print(f"\n[8] 保存结果")
    os.makedirs(output_dir, exist_ok=True)
    save_split_datasets(train_df_undersampled, val_df, test_df, output_dir, format=format)
    print(f"    数据集已保存到: {output_dir}")

    # 验证保存的文件
    print(f"\n[9] 验证结果")
    saved_files = list(Path(output_dir).glob("*"))
    print(f"    生成了 {len(saved_files)} 个文件:")
    for file in saved_files:
        size_mb = file.stat().st_size / 1024 / 1024
        print(f"      - {file.name} ({size_mb:.2f} MB)")

    print("\n" + "="*80)
    print("✅ 欠采样处理完成！")
    print("="*80)

    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='对已有数据集执行欠采样')
    parser.add_argument('--input', type=str, required=True,
                       help='输入数据集路径 (.csv 或 .parquet)')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.70,
                       help='训练集比例 (默认: 0.70)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='验证集比例 (默认: 0.15)')
    parser.add_argument('--target-ratio', type=float, default=5.0,
                       help='欠采样目标比例 (默认: 5.0)')
    parser.add_argument('--format', type=str, choices=['parquet', 'csv'],
                       default='parquet', help='输出格式 (默认: parquet)')

    args = parser.parse_args()

    # 执行欠采样
    success = apply_undersampling_to_existing_dataset(
        input_path=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_ratio=args.target_ratio,
        format=args.format
    )

    if success:
        print(f"\n📁 输出目录: {args.output}")
        print(f"🚀 可以开始训练模型了！")
        sys.exit(0)
    else:
        print("\n❌ 欠采样处理失败")
        sys.exit(1)


if __name__ == "__main__":
    main()