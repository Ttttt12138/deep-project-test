"""
涨停预测系统主程序入口
"""

import argparse
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.build_dataset import (
    process_single_stock_csv, build_single_day_dataset, save_dataset, load_dataset, get_feature_columns
)
from scripts.extract_7z import (
    SevenZipExtractor, batch_process_7z_files, validate_7z_structure
)
from src.models.lgbm_trainer import (
    prepare_training_data, split_dataset_by_date, train_lgbm_classifier,
    evaluate_model, get_feature_importance, predict_limit_up_probability
)
from src.data_processing.stock_utils import (
    determine_stock_type, get_limit_ratio, get_stock_info
)
from src.data_processing.quality_check import (
    run_quality_checks, print_quality_report
)
from src.data_processing.dataset_split import (
    split_dataset_by_trading_day, save_split_datasets, load_split_datasets, validate_split
)
import build_2025_dataset  # 导入全年数据集构建模块


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='涨停预测系统')
    parser.add_argument('--mode', type=str, choices=['build', 'extract', 'train', 'predict', 'check', 'split', 'stock-info', 'full'],
                       default='build', help='运行模式')
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--output', type=str, default='data/processed/dataset.csv',
                       help='输出文件路径')
    parser.add_argument('--date', type=str, help='交易日期')
    parser.add_argument('--code', type=str, help='股票代码')
    parser.add_argument('--preclose', type=float, default=10.0, help='昨收价')
    parser.add_argument('--model-path', type=str, default='models/lgbm_model.pkl',
                       help='模型保存/加载路径')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--valid-ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='预测概率阈值')
    parser.add_argument('--split-dir', type=str, default='data/processed/split_datasets',
                       help='划分后数据集目录')
    parser.add_argument('--format', type=str, choices=['parquet', 'csv'],
                       default='parquet', help='数据集格式')
    parser.add_argument('--max-files', type=int, help='最大处理文件数（测试用）')
    parser.add_argument('--sample', type=int, help='采样间隔（测试用）')
    parser.add_argument('--max-months', type=int, default=3,
                       help='最大月份数（默认3）')

    args = parser.parse_args()

    # 处理full模式（构建全年数据集）
    if args.mode == 'full':
        print("模式: 构建2025全年数据集")

        # 构造参数传递给build_2025_dataset
        output_path = args.output or 'data/processed/2025_full_dataset.csv'

        # 调用build_2025_dataset的构建函数
        success = build_2025_dataset.build_full_year_dataset(
            year='2025',
            output_path=output_path,
            max_files=getattr(args, 'max_files', None),
            sample_files=getattr(args, 'sample', None),
            split_dataset=True,  # 自动进行数据集划分
            train_ratio=args.train_ratio,
            val_ratio=args.valid_ratio,
            max_months=getattr(args, 'max_months', 3)  # 前3个月
        )

        if success:
            print("\n数据集构建成功！")
            sys.exit(0)
        else:
            print("\n数据集构建失败！")
            sys.exit(1)

    if args.mode == 'build':
        print("模式: 构建数据集")

        if args.input and args.code and args.date:
            print(f"处理股票: {args.code}, 日期: {args.date}")
            df = process_single_stock_csv(
                args.input, args.code, args.date, args.preclose
            )

            if not df.empty:
                save_dataset(df, args.output)
            else:
                print("处理失败，数据为空")
        else:
            print("构建模式需要 --input, --code, --date 参数")

    elif args.mode == 'extract':
        print("模式: 解压7z文件")
        if args.input:
            extractor = SevenZipExtractor(args.input)
            extract_dir = extractor.extract_all(args.output)
            print(f"解压完成: {extract_dir}")
        else:
            print("解压模式需要 --input 参数")

    elif args.mode == 'train':
        print("模式: 训练模型")
        if args.input:
            # 加载数据集
            print(f"加载数据集: {args.input}")
            df = load_dataset(args.input)

            if df.empty:
                print("错误: 数据集为空")
                return

            # 获取特征列
            feature_cols = get_feature_columns(df)
            print(f"找到 {len(feature_cols)} 个特征")

            if len(feature_cols) == 0:
                print("错误: 未找到特征列")
                return

            # 检查是否有date列用于按日期划分
            if 'date' not in df.columns:
                print("警告: 数据集缺少date列，将使用随机划分")
                from sklearn.model_selection import train_test_split
                # 简单的随机划分
                train_df, temp_df = train_test_split(df, test_size=(1-args.train_ratio), random_state=42)
                valid_ratio_adjusted = args.valid_ratio / (1-args.train_ratio)
                valid_df, test_df = train_test_split(temp_df,
                                                   test_size=(1-valid_ratio_adjusted),
                                                   random_state=42)
            else:
                train_df, valid_df, test_df = split_dataset_by_date(df, args.train_ratio, args.valid_ratio)

            print(f"训练集: {len(train_df)} 样本")
            print(f"验证集: {len(valid_df)} 样本")
            print(f"测试集: {len(test_df)} 样本")

            # 准备训练数据
            print("准备训练数据...")
            X_train, y_train = prepare_training_data(train_df, feature_cols)
            X_valid, y_valid = prepare_training_data(valid_df, feature_cols)
            X_test, y_test = prepare_training_data(test_df, feature_cols)

            print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
            print(f"验证数据形状: X={X_valid.shape}, y={y_valid.shape}")
            print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")

            # 训练模型
            print("开始训练LightGBM模型...")
            model = train_lgbm_classifier(X_train, y_train, X_valid, y_valid)

            # 评估模型
            print("评估模型性能...")
            metrics = evaluate_model(model, X_test, y_test)

            print("\n模型评估结果:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

            # 特征重要性
            print("\n特征重要性 (Top 10):")
            importance_df = get_feature_importance(model, feature_cols)
            print(importance_df.head(10).to_string(index=False))

            # 保存模型
            import joblib
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            joblib.dump(model, args.model_path)
            print(f"\n模型已保存到: {args.model_path}")

        else:
            print("训练模式需要 --input 参数（数据集路径）")

    elif args.mode == 'predict':
        print("模式: 预测")
        if args.input and os.path.exists(args.model_path):
            # 加载模型
            import joblib
            print(f"加载模型: {args.model_path}")
            model = joblib.load(args.model_path)

            # 加载待预测数据
            print(f"加载数据: {args.input}")
            df = load_dataset(args.input)

            if df.empty:
                print("错误: 数据集为空")
                return

            # 获取特征列
            feature_cols = get_feature_columns(df)
            print(f"找到 {len(feature_cols)} 个特征")

            if len(feature_cols) == 0:
                print("错误: 未找到特征列")
                return

            # 准备预测数据
            X, _ = prepare_training_data(df, feature_cols)
            print(f"预测数据形状: {X.shape}")

            # 预测涨停概率
            print("开始预测...")
            probabilities = predict_limit_up_probability(model, X)

            # 添加预测结果到原始数据
            df['limit_up_probability'] = probabilities
            df['prediction'] = (probabilities >= args.threshold).astype(int)

            # 统计预测结果
            print(f"\n预测结果统计:")
            print(f"  平均涨停概率: {probabilities.mean():.4f}")
            print(f"  高概率样本数 (≥{args.threshold}): {(probabilities >= args.threshold).sum()}")
            print(f"  预测涨停数: {df['prediction'].sum()}")
            print(f"  总样本数: {len(df)}")

            # 保存预测结果
            output_path = args.output.replace('.csv', '_predictions.csv')
            df.to_csv(output_path, index=False)
            print(f"\n预测结果已保存到: {output_path}")

            # 显示一些高概率样本
            high_prob_samples = df[df['limit_up_probability'] >= args.threshold].head(10)
            if not high_prob_samples.empty:
                print("\n高概率涨停样本 (Top 10):")
                display_cols = ['code', 'date', 'limit_up_probability', 'prediction']
                available_cols = [col for col in display_cols if col in high_prob_samples.columns]
                print(high_prob_samples[available_cols].to_string(index=False))

        else:
            if not args.input:
                print("预测模式需要 --input 参数（待预测数据路径）")
            if not os.path.exists(args.model_path):
                print(f"错误: 模型文件不存在: {args.model_path}")

    elif args.mode == 'check':
        print("模式: 数据质量检查")
        if args.input:
            # 加载数据集
            print(f"加载数据集: {args.input}")
            df = load_dataset(args.input)

            if df.empty:
                print("错误: 数据集为空")
                return

            # 定义必需特征
            REQUIRED_FEATURES = [
                'dist_to_limit', 'ticks_to_limit', 'ask1_to_limit', 'ask1_gap',
                'bid_depth', 'ask_depth', 'order_imbalance', 'b1_volume', 'a1_volume',
                'spread', 'ask_slope', 'bid_slope',
                'ret_1tick', 'vol_delta', 'money_delta'
            ]

            # 运行质量检查
            print("运行质量检查...")
            quality_results = run_quality_checks(df, REQUIRED_FEATURES)

            # 打印质量报告
            print_quality_report(quality_results)

            # 检查是否全部通过
            all_passed = all(result['passed'] for result in quality_results)
            if all_passed:
                print("\n数据集质量良好，可用于训练！")
                sys.exit(0)
            else:
                print("\n数据集存在问题，请查看详细信息！")
                sys.exit(1)
        else:
            print("质量检查模式需要 --input 参数（数据集路径）")

    elif args.mode == 'split':
        print("模式: 数据集划分")
        if args.input:
            # 加载数据集
            print(f"加载数据集: {args.input}")
            df = load_dataset(args.input)

            if df.empty:
                print("错误: 数据集为空")
                return

            if 'date' not in df.columns:
                print("错误: 数据集缺少date列，无法按交易日划分")
                return

            # 按交易日划分数据集
            print(f"按交易日划分数据集（训练集 {args.train_ratio:.0%}, 验证集 {args.valid_ratio:.0%}, "
                  f"测试集 {1-args.train_ratio-args.valid_ratio:.0%}）...")

            train_df, val_df, test_df = split_dataset_by_trading_day(
                df,
                train_ratio=args.train_ratio,
                val_ratio=args.valid_ratio,
                test_ratio=1.0 - args.train_ratio - args.valid_ratio
            )

            # 验证划分
            validation_result = validate_split(train_df, val_df, test_df)
            if not validation_result['passed']:
                print("\n划分验证警告:")
                for issue in validation_result['issues']:
                    print(f"  - {issue}")

            # 保存划分后的数据集
            print(f"\n保存划分后的数据集到: {args.split_dir}")
            save_split_datasets(train_df, val_df, test_df, args.split_dir, format=args.format)

            print("\n数据集划分完成！")

        else:
            print("数据集划分模式需要 --input 参数（数据集路径）")

    elif args.mode == 'stock-info':
        print("模式: 股票信息查询")
        if args.code:
            # 查询股票信息
            stock_info = get_stock_info(args.code)

            print("\n股票信息:")
            print(f"  股票代码: {stock_info['stock_code']}")
            print(f"  股票类型: {stock_info['stock_type'].upper()}")
            print(f"  涨停比例: {stock_info['limit_ratio']:.0%}")
            print(f"  是否ST: {'是' if stock_info['is_st'] else '否'}")

            # 显示股票类型说明
            type_descriptions = {
                'st': 'ST股票（5%涨停）',
                'gem': '创业板（20%涨停）',
                'kcb': '科创板（20%涨停）',
                'bse': '北交所（30%涨停）',
                'normal': '普通股（10%涨停）'
            }
            print(f"  说明: {type_descriptions.get(stock_info['stock_type'], '未知')}")
        else:
            print("股票信息查询模式需要 --code 参数（股票代码）")


if __name__ == "__main__":
    main()