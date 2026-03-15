"""
两步数据集构建器 - 实现解压后自动清理的策略
"""
import os
import sys
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.extract_7z import SevenZipExtractor
from scripts.build_dataset import (
    save_dataset, get_feature_columns,
    generate_next_tick_limit_up_label, get_label_statistics
)
from src.feature_engineering.limit_up_features import extract_limit_up_features
from src.data_processing.limit_up_processor import process_tick_file
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.data_processing.quality_check import run_quality_checks, print_quality_report
from src.data_processing.dataset_split import split_dataset_by_trading_day, save_split_datasets


class DatasetBuilder:
    """数据集构建器 - 两步处理策略"""

    def __init__(self, archive_path, output_base_dir):
        self.archive_path = archive_path
        self.output_base_dir = output_base_dir
        self.date_str = self._extract_date_from_filename()
        self.extractor = None
        self.extract_dir = None

    def _extract_date_from_filename(self):
        """从文件名提取日期"""
        filename = os.path.basename(self.archive_path)
        return filename.replace('.7z', '')

    def step1_extract(self):
        """
        步骤1: 解压7z文件到临时目录

        Returns:
            解压目录路径
        """
        print(f"步骤1/2: 解压 {self.date_str}")
        self.extractor = SevenZipExtractor(self.archive_path)
        self.extract_dir = os.path.join(self.output_base_dir, 'temp_extract', self.date_str)
        os.makedirs(self.extract_dir, exist_ok=True)

        extract_result = self.extractor.extract_all(self.extract_dir)
        print(f"  解压完成: {extract_result}")
        return extract_result

    def step2_process_and_cleanup(self, output_path, split_dataset=False, **kwargs):
        """
        步骤2: 处理数据并清理临时目录

        Args:
            output_path: 数据集输出路径
            split_dataset: 是否划分数据集
            **kwargs: 数据集划分参数

        Returns:
            处理成功返回True，失败返回False
        """
        print(f"步骤2/2: 处理数据并清理临时文件")

        try:
            # 处理解压后的数据
            day_df = self._process_extracted_data()

            if day_df.empty:
                print(f"  警告: {self.date_str} 没有生成有效数据")
                return False

            # 保存数据集
            save_dataset(day_df, output_path)
            print(f"  数据集已保存: {output_path}")

            # 可选：划分数据集
            if split_dataset:
                self._split_dataset(day_df, **kwargs)

            return True

        except Exception as e:
            print(f"  处理失败: {e}")
            return False

        finally:
            # 清理临时目录
            self._cleanup()

    def _process_extracted_data(self):
        """处理解压后的数据"""
        csv_files = list(Path(self.extract_dir).rglob("*.csv"))
        print(f"  找到 {len(csv_files)} 个股票文件")

        if len(csv_files) == 0:
            return pd.DataFrame()

        all_stocks_data = []
        skipped_stocks = 0

        for csv_file in tqdm(csv_files, desc="  处理股票"):
            try:
                stock_code = csv_file.stem
                stock_type = determine_stock_type(stock_code)
                limit_ratio = get_limit_ratio(stock_type)

                # 获取基准价格
                preclose = self._get_benchmark_price(csv_file)
                if preclose is None or preclose <= 0:
                    skipped_stocks += 1
                    continue

                # 处理tick文件
                df = process_tick_file(str(csv_file), preclose, limit_ratio)
                if df.empty:
                    skipped_stocks += 1
                    continue

                # 添加股票代码和日期
                df['code'] = stock_code
                df['date'] = self.date_str

                # 提取特征和生成标签
                df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=limit_ratio)
                df = generate_next_tick_limit_up_label(df)

                all_stocks_data.append(df)

            except Exception as e:
                skipped_stocks += 1
                continue

        print(f"  处理完成: 成功 {len(all_stocks_data)}, 跳过 {skipped_stocks}")

        if all_stocks_data:
            return pd.concat(all_stocks_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def _get_benchmark_price(self, csv_file):
        """获取基准价格"""
        try:
            df_raw = pd.read_csv(csv_file, nrows=100)

            # 优先级：买一价 > 卖一价 > 开盘价 > 当前价
            for price_col in ['b1_p', 'a1_p', 'open', 'current']:
                if price_col in df_raw.columns:
                    valid_prices = df_raw[df_raw[price_col] > 0][price_col]
                    if len(valid_prices) > 0:
                        return valid_prices.iloc[0]
        except Exception:
            pass
        return None

    def _split_dataset(self, df, train_ratio=0.70, val_ratio=0.15):
        """划分数据集"""
        train_df, val_df, test_df = split_dataset_by_trading_day(
            df, train_ratio=train_ratio, val_ratio=val_ratio,
            test_ratio=1.0 - train_ratio - val_ratio
        )

        split_output_dir = os.path.join(self.output_base_dir, 'split_datasets', self.date_str)
        save_split_datasets(train_df, val_df, test_df, split_output_dir, format='parquet')
        print(f"  数据集已划分: {split_output_dir}")

    def _cleanup(self):
        """清理临时目录"""
        if self.extract_dir and os.path.exists(self.extract_dir):
            shutil.rmtree(self.extract_dir, ignore_errors=True)
            print(f"  临时目录已清理: {self.extract_dir}")


def process_single_day_two_step(archive_path, output_dir="d:/Qoder-project/deep project/data",
                               split_dataset=False, **kwargs):
    """
    两步法处理单个交易日数据

    Args:
        archive_path: 7z文件路径
        output_dir: 输出目录（默认：d:/Qoder-project/deep project/data）
        split_dataset: 是否划分数据集
        **kwargs: 数据集划分参数

    Returns:
        成功返回数据集路径，失败返回None
    """
    builder = DatasetBuilder(archive_path, output_dir)

    try:
        # 步骤1: 解压
        extract_dir = builder.step1_extract()

        # 步骤2: 处理并清理
        output_filename = f"{builder.date_str}_dataset.csv"
        output_path = os.path.join(output_dir, output_filename)

        success = builder.step2_process_and_cleanup(
            output_path,
            split_dataset=split_dataset,
            **kwargs
        )

        if success:
            print(f"OK {builder.date_str} 处理完成")
            return output_path
        else:
            print(f"FAIL {builder.date_str} 处理失败")
            return None

    except Exception as e:
        print(f"FAIL 处理 {archive_path} 时出错: {e}")
        return None


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='单日数据集构建器 - 两步处理策略')
    parser.add_argument('--input', type=str, required=True,
                       help='7z文件路径 (例如: "2025/1/2025-01-02.7z")')
    parser.add_argument('--output', type=str,
                       default="d:/Qoder-project/deep project/data",
                       help='输出目录 (默认: d:/Qoder-project/deep project/data)')
    parser.add_argument('--split', action='store_true',
                       help='是否划分数据集')
    parser.add_argument('--train-ratio', type=float, default=0.70,
                       help='训练集比例 (默认: 0.70)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='验证集比例 (默认: 0.15)')

    args = parser.parse_args()

    # 处理单个交易日
    result = process_single_day_two_step(
        args.input,
        output_dir=args.output,
        split_dataset=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    if result:
        print(f"\n数据集构建完成: {result}")
    else:
        print("\n数据集构建失败")
        exit(1)


if __name__ == "__main__":
    main()