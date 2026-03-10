"""
涨停预测系统自动化运行脚本
一键运行完整的数据处理、训练和预测流程
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """执行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*60)

    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=False, text=True)
        print(f"✓ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} 失败: {e}")
        return False


def check_environment():
    """检查环境依赖"""
    print("\n" + "="*60)
    print("1. 检查环境依赖")
    print("="*60)

    # 检查必需的包
    required_packages = ['pandas', 'numpy', 'lightgbm', 'sklearn', 'joblib', 'py7zr']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (未安装)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n缺失的包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False

    # 检查项目结构
    required_dirs = ['data/raw', 'data/processed', 'data/extracted', 'models']
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ 目录 {dir_path} 准备就绪")

    return True


def extract_sample_data():
    """提取样本数据"""
    print("\n" + "="*60)
    print("2. 提取样本数据")
    print("="*60)

    # 查找第一个7z文件
    data_dirs = ['2025/01', '2026/01', '2025', '2026']
    sample_file = None

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = list(Path(data_dir).glob("*.7z"))
            if files:
                sample_file = str(files[0])
                break

    if not sample_file:
        print("✗ 未找到数据文件，请确保数据文件存在")
        return False

    print(f"找到样本文件: {sample_file}")

    # 解压文件
    extract_cmd = f'python main.py --mode extract --input "{sample_file}" --output data/extracted'
    return run_command(extract_cmd, "解压样本数据")


def build_sample_dataset():
    """构建样本数据集"""
    print("\n" + "="*60)
    print("3. 构建样本数据集")
    print("="*60)

    # 查找解压后的CSV文件
    extracted_dir = "data/extracted"
    if not os.path.exists(extracted_dir):
        print("✗ 解压目录不存在，请先提取数据")
        return False

    # 找到第一个CSV文件
    csv_files = list(Path(extracted_dir).glob("*.csv"))
    if not csv_files:
        print("✗ 未找到CSV文件")
        return False

    sample_csv = str(csv_files[0])
    stock_code = csv_files[0].stem  # 获取文件名作为股票代码
    date = "2025-01-02"  # 默认日期

    print(f"处理文件: {sample_csv}")
    print(f"股票代码: {stock_code}, 日期: {date}")

    build_cmd = f'python main.py --mode build --input "{sample_csv}" --code {stock_code} --date {date} --preclose 10.0 --output data/processed/dataset.csv'
    return run_command(build_cmd, "构建样本数据集")


def train_model():
    """训练模型"""
    print("\n" + "="*60)
    print("4. 训练模型")
    print("="*60)

    dataset_path = "data/processed/dataset.csv"
    if not os.path.exists(dataset_path):
        print("✗ 数据集不存在，请先构建数据集")
        return False

    train_cmd = f'python main.py --mode train --input {dataset_path} --model-path models/lgbm_model.pkl'
    return run_command(train_cmd, "训练LightGBM模型")


def make_predictions():
    """进行预测"""
    print("\n" + "="*60)
    print("5. 进行预测")
    print("="*60)

    dataset_path = "data/processed/dataset.csv"
    model_path = "models/lgbm_model.pkl"

    if not os.path.exists(dataset_path):
        print("✗ 数据集不存在，请先构建数据集")
        return False

    if not os.path.exists(model_path):
        print("✗ 模型不存在，请先训练模型")
        return False

    predict_cmd = f'python main.py --mode predict --input {dataset_path} --model-path {model_path} --output data/processed/predictions.csv'
    return run_command(predict_cmd, "进行涨停预测")


def quick_start():
    """快速开始：运行完整流程"""
    print("\n" + "="*60)
    print("涨停预测系统 - 快速开始")
    print("="*60)

    steps = [
        ("环境检查", check_environment),
        ("提取样本数据", extract_sample_data),
        ("构建数据集", build_sample_dataset),
        ("训练模型", train_model),
        ("进行预测", make_predictions)
    ]

    results = []
    for step_name, step_func in steps:
        success = step_func()
        results.append((step_name, success))
        if not success:
            print(f"\n⚠ 流程在 {step_name} 阶段失败")
            break

    # 显示总结
    print("\n" + "="*60)
    print("运行总结")
    print("="*60)

    for step_name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{step_name}: {status}")

    all_success = all(success for _, success in results)
    if all_success:
        print("\n🎉 所有步骤完成！系统运行正常")
        print("预测结果保存在: data/processed/dataset_predictions.csv")
    else:
        print("\n⚠ 部分步骤失败，请检查错误信息")


def custom_run(mode, **kwargs):
    """自定义运行模式"""
    print(f"\n运行模式: {mode}")

    cmd = f"python main.py --mode {mode}"
    for key, value in kwargs.items():
        if value is not None:
            cmd += f" --{key.replace('_', '-')} {value}"

    return run_command(cmd, f"{mode}模式")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='涨停预测系统自动化运行')
    parser.add_argument('--mode', type=str, choices=['quick', 'env', 'extract', 'build', 'train', 'predict'],
                       default='quick', help='运行模式')
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--code', type=str, help='股票代码')
    parser.add_argument('--date', type=str, help='交易日期')
    parser.add_argument('--preclose', type=float, default=10.0, help='昨收价')
    parser.add_argument('--model-path', type=str, default='models/lgbm_model.pkl', help='模型路径')

    args = parser.parse_args()

    if args.mode == 'quick':
        quick_start()
    elif args.mode == 'env':
        check_environment()
    elif args.mode == 'extract':
        custom_run('extract', input=args.input, output='data/extracted')
    elif args.mode == 'build':
        custom_run('build', input=args.input, code=args.code, date=args.date,
                  preclose=args.preclose, output='data/processed/dataset.csv')
    elif args.mode == 'train':
        custom_run('train', input=args.input, model_path=args.model_path)
    elif args.mode == 'predict':
        custom_run('predict', input=args.input, model_path=args.model_path,
                  output='data/processed/predictions.csv')


if __name__ == "__main__":
    main()