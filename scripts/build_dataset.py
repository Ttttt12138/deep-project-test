"""
涨停预测数据集构建Pipeline
整合数据处理、特征工程、标签生成
"""

import pandas as pd
from typing import List, Tuple
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.limit_up_processor import (
    load_tick_csv, convert_time_column, sort_by_time,
    filter_invalid_ticks, process_tick_file
)
from src.feature_engineering.limit_up_features import extract_limit_up_features
from src.feature_engineering.limit_up_labels import (
    generate_next_tick_limit_up_label, get_label_statistics
)
from src.data_processing.csv_utils import (
    get_feature_columns as get_csv_feature_columns,
    read_csv,
    write_csv,
)


def process_single_stock_csv(file_path: str,
                             stock_code: str,
                             date: str,
                             preclose: float,
                             limit_ratio: float = 0.10,
                             tick_size: float = 0.01) -> pd.DataFrame:
    """
    处理单个股票CSV文件

    Args:
        file_path: CSV文件路径
        stock_code: 股票代码
        date: 交易日期
        preclose: 昨收价
        limit_ratio: 涨停比例
        tick_size: 最小价格跳动单位

    Returns:
        处理后的DataFrame，包含特征和标签
    """
    try:
        # 加载并处理tick数据
        df = process_tick_file(file_path, preclose, limit_ratio)

        # 添加股票代码和日期
        df['code'] = stock_code
        df['date'] = date

        # 提取涨停特征
        df = extract_limit_up_features(df, tick_size)

        # 生成标签
        df = generate_next_tick_limit_up_label(df)

        return df

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return pd.DataFrame()


def build_single_day_dataset(extracted_dir: str,
                             date: str,
                             preclose_data: dict = None) -> pd.DataFrame:
    """
    构建单个交易日的数据集

    Args:
        extracted_dir: 解压后的目录路径
        date: 交易日期
        preclose_data: 昨收价数据字典 {股票代码: 昨收价}

    Returns:
        单日数据集DataFrame
    """
    all_data = []

    # 遍历目录中的所有CSV文件
    for file_name in os.listdir(extracted_dir):
        if not file_name.endswith('.csv'):
            continue

        stock_code = file_name.replace('.csv', '')
        file_path = os.path.join(extracted_dir, file_name)

        # 获取昨收价（如果没有提供，使用默认值）
        preclose = preclose_data.get(stock_code, 10.0) if preclose_data else 10.0

        # 处理单个股票
        stock_df = process_single_stock_csv(
            file_path, stock_code, date, preclose
        )

        if not stock_df.empty:
            all_data.append(stock_df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def build_multi_day_dataset(daily_data_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    构建多日数据集

    Args:
        daily_data_list: 每日数据列表

    Returns:
        合并后的多日数据集
    """
    if not daily_data_list:
        return pd.DataFrame()

    return pd.concat(daily_data_list, ignore_index=True)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    获取特征列名

    Args:
        df: 数据集DataFrame

    Returns:
        特征列名列表
    """
    return get_csv_feature_columns(df)


def save_dataset(df: pd.DataFrame, output_path: str):
    """
    保存数据集

    Args:
        df: 数据集DataFrame
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not output_path.endswith('.csv'):
        output_path = f"{output_path}.csv"

    write_csv(df, output_path)

    print(f"数据集已保存到: {output_path}")
    print(f"样本数量: {len(df)}")
    print(f"特征数量: {len(get_feature_columns(df))}")

    # 显示标签统计
    if 'label' in df.columns:
        stats = get_label_statistics(df)
        print(f"正样本数量: {stats['positive_samples']}")
        print(f"负样本数量: {stats['negative_samples']}")
        print(f"正样本比例: {stats['positive_ratio']:.4%}")


def load_dataset(input_path: str) -> pd.DataFrame:
    """
    加载数据集

    Args:
        input_path: 输入文件路径

    Returns:
        数据集DataFrame
    """
    df = read_csv(input_path)

    print(f"数据集已加载: {len(df)} 个样本")
    return df


def full_pipeline(date: str,
                 csv_file_path: str,
                 stock_code: str,
                 preclose: float,
                 output_dir: str) -> pd.DataFrame:
    """
    完整的数据处理流水线

    Args:
        date: 交易日期
        csv_file_path: CSV文件路径
        stock_code: 股票代码
        preclose: 昨收价
        output_dir: 输出目录

    Returns:
        处理后的DataFrame
    """
    # 处理单个股票
    df = process_single_stock_csv(
        csv_file_path, stock_code, date, preclose
    )

    if df.empty:
        return df

    # 保存结果
    output_path = os.path.join(output_dir, f"{stock_code}_{date}.csv")
    os.makedirs(output_dir, exist_ok=True)
    write_csv(df, output_path)

    return df
