"""
数据集划分模块
按交易日划分数据集为训练集、验证集和测试集
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import os


def split_dataset_by_trading_day(df: pd.DataFrame,
                                 train_ratio: float = 0.70,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 date_col: str = 'date',
                                 random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按交易日划分数据集（70/15/15）

    关键原则：
    - 必须按交易日切分，禁止随机切分
    - 同一交易日的所有tick必须属于同一数据集
    - 避免未来数据泄漏

    Args:
        df: 待划分的DataFrame
        train_ratio: 训练集比例（默认0.70）
        val_ratio: 验证集比例（默认0.15）
        test_ratio: 测试集比例（默认0.15）
        date_col: 日期列名
        random_seed: 随机种子（用于交易日划分）

    Returns:
        (train_df, val_df, test_df) 元组

    Raises:
        ValueError: 比例之和不为1，或缺少日期列
    """
    # 验证比例
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"比例之和必须为1.0，当前为: {train_ratio + val_ratio + test_ratio}")

    # 检查日期列
    if date_col not in df.columns:
        raise ValueError(f"DataFrame中缺少日期列 '{date_col}'")

    # 获取所有唯一交易日
    unique_dates = df[date_col].unique()
    unique_dates = sorted(unique_dates)  # 确保按时间顺序排序

    n_dates = len(unique_dates)
    print(f"共有 {n_dates} 个交易日")

    # 计算划分点
    n_train = int(n_dates * train_ratio)
    n_val = int(n_dates * val_ratio)

    # 按顺序划分交易日（避免未来数据泄漏）
    train_dates = unique_dates[:n_train]
    val_dates = unique_dates[n_train:n_train + n_val]
    test_dates = unique_dates[n_train + n_val:]

    print(f"训练集: {len(train_dates)} 个交易日 ({train_ratio:.0%})")
    print(f"验证集: {len(val_dates)} 个交易日 ({val_ratio:.0%})")
    print(f"测试集: {len(test_dates)} 个交易日 ({test_ratio:.0%})")

    # 根据日期划分数据
    train_df = df[df[date_col].isin(train_dates)].copy()
    val_df = df[df[date_col].isin(val_dates)].copy()
    test_df = df[df[date_col].isin(test_dates)].copy()

    # 验证划分
    print(f"\n划分后样本数:")
    print(f"训练集: {len(train_df):,} ({len(train_df)/len(df):.2%})")
    print(f"验证集: {len(val_df):,} ({len(val_df)/len(df):.2%})")
    print(f"测试集: {len(test_df):,} ({len(test_df)/len(df):.2%})")

    # 验证日期范围
    print(f"\n日期范围:")
    print(f"训练集: {train_dates[0]} ~ {train_dates[-1]}")
    print(f"验证集: {val_dates[0]} ~ {val_dates[-1]}")
    print(f"测试集: {test_dates[0]} ~ {test_dates[-1]}")

    return train_df, val_df, test_df


def split_dataset_stratified(df: pd.DataFrame,
                             train_ratio: float = 0.70,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             label_col: str = 'label',
                             date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按交易日分层划分数据集（保持正样本比例）

    Args:
        df: 待划分的DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        label_col: 标签列名
        date_col: 日期列名

    Returns:
        (train_df, val_df, test_df) 元组
    """
    # 首先获取所有交易日
    unique_dates = df[date_col].unique()
    unique_dates = sorted(unique_dates)

    # 计算每个交易日的正样本比例
    date_info = []
    for date in unique_dates:
        date_df = df[df[date_col] == date]
        positive_ratio = (date_df[label_col] == 1).mean()
        date_info.append({
            'date': date,
            'positive_ratio': positive_ratio,
            'sample_count': len(date_df)
        })

    date_info_df = pd.DataFrame(date_info)

    # 按正样本比例排序，以便分层
    date_info_df = date_info_df.sort_values('positive_ratio')

    # 划分日期
    n_dates = len(unique_dates)
    n_train = int(n_dates * train_ratio)
    n_val = int(n_dates * val_ratio)

    train_dates = date_info_df.iloc[:n_train]['date'].values
    val_dates = date_info_df.iloc[n_train:n_train + n_val]['date'].values
    test_dates = date_info_df.iloc[n_train + n_val:]['date'].values

    # 划分数据
    train_df = df[df[date_col].isin(train_dates)].copy()
    val_df = df[df[date_col].isin(val_dates)].copy()
    test_df = df[df[date_col].isin(test_dates)].copy()

    print(f"分层划分完成:")
    print(f"训练集: {len(train_dates)} 个交易日, {len(train_df):,} 样本")
    print(f"验证集: {len(val_dates)} 个交易日, {len(val_df):,} 样本")
    print(f"测试集: {len(test_dates)} 个交易日, {len(test_df):,} 样本")

    return train_df, val_df, test_df


def save_split_datasets(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       output_dir: str,
                       format: str = 'parquet') -> None:
    """
    保存划分后的数据集

    Args:
        train_df: 训练集DataFrame
        val_df: 验证集DataFrame
        test_df: 测试集DataFrame
        output_dir: 输出目录
        format: 保存格式（'parquet' 或 'csv'）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 根据格式保存
    if format == 'parquet':
        train_path = os.path.join(output_dir, 'train.parquet')
        val_path = os.path.join(output_dir, 'val.parquet')
        test_path = os.path.join(output_dir, 'test.parquet')

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

    elif format == 'csv':
        train_path = os.path.join(output_dir, 'train.csv')
        val_path = os.path.join(output_dir, 'val.csv')
        test_path = os.path.join(output_dir, 'test.csv')

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

    else:
        raise ValueError(f"不支持的格式: {format}，仅支持 'parquet' 或 'csv'")

    print(f"\n数据集已保存到: {output_dir}")
    print(f"  - 训练集: {os.path.basename(train_path)} ({len(train_df):,} 样本)")
    print(f"  - 验证集: {os.path.basename(val_path)} ({len(val_df):,} 样本)")
    print(f"  - 测试集: {os.path.basename(test_path)} ({len(test_df):,} 样本)")


def load_split_datasets(input_dir: str, format: str = 'parquet') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载划分后的数据集

    Args:
        input_dir: 输入目录
        format: 文件格式（'parquet' 或 'csv'）

    Returns:
        (train_df, val_df, test_df) 元组
    """
    if format == 'parquet':
        train_path = os.path.join(input_dir, 'train.parquet')
        val_path = os.path.join(input_dir, 'val.parquet')
        test_path = os.path.join(input_dir, 'test.parquet')

        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

    elif format == 'csv':
        train_path = os.path.join(input_dir, 'train.csv')
        val_path = os.path.join(input_dir, 'val.csv')
        test_path = os.path.join(input_dir, 'test.csv')

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

    else:
        raise ValueError(f"不支持的格式: {format}，仅支持 'parquet' 或 'csv'")

    return train_df, val_df, test_df


def validate_split(train_df: pd.DataFrame,
                  val_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  date_col: str = 'date',
                  label_col: str = 'label') -> Dict[str, any]:
    """
    验证数据集划分的正确性

    Args:
        train_df: 训练集DataFrame
        val_df: 验证集DataFrame
        test_df: 测试集DataFrame
        date_col: 日期列名
        label_col: 标签列名

    Returns:
        包含验证结果的字典
    """
    validation_result = {
        'passed': True,
        'issues': []
    }

    # 1. 检查日期不重叠
    train_dates = set(train_df[date_col].unique())
    val_dates = set(val_df[date_col].unique())
    test_dates = set(test_df[date_col].unique())

    if train_dates & val_dates:
        validation_result['passed'] = False
        validation_result['issues'].append("训练集和验证集有重叠的交易日")

    if train_dates & test_dates:
        validation_result['passed'] = False
        validation_result['issues'].append("训练集和测试集有重叠的交易日")

    if val_dates & test_dates:
        validation_result['passed'] = False
        validation_result['issues'].append("验证集和测试集有重叠的交易日")

    # 2. 检查时间顺序
    if train_dates and val_dates and test_dates:
        max_train_date = max(train_dates)
        min_val_date = min(val_dates)
        max_val_date = max(val_dates)
        min_test_date = min(test_dates)

        if max_train_date > min_val_date:
            validation_result['passed'] = False
            validation_result['issues'].append("训练集日期晚于验证集日期（未来数据泄漏）")

        if max_val_date > min_test_date:
            validation_result['passed'] = False
            validation_result['issues'].append("验证集日期晚于测试集日期（未来数据泄漏）")

    # 3. 检查每个集合都有正样本
    if label_col in train_df.columns:
        if (train_df[label_col] == 1).sum() == 0:
            validation_result['passed'] = False
            validation_result['issues'].append("训练集中没有正样本")

    if label_col in val_df.columns:
        if (val_df[label_col] == 1).sum() == 0:
            validation_result['passed'] = False
            validation_result['issues'].append("验证集中没有正样本")

    if label_col in test_df.columns:
        if (test_df[label_col] == 1).sum() == 0:
            validation_result['passed'] = False
            validation_result['issues'].append("测试集中没有正样本")

    return validation_result