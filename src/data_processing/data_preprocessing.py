"""
数据预处理模块
包含数据加载、清洗和滑动窗口采样功能
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List
import os

from src.data_processing.csv_utils import read_csv


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    加载CSV盘口数据

    Args:
        file_path: CSV文件路径

    Returns:
        原始DataFrame
    """
    try:
        df = read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"加载数据失败: {e}")


def clean_orderbook_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗盘口数据

    Args:
        df: 原始DataFrame

    Returns:
        清洗后的DataFrame
    """
    # 转换时间列
    time_col = '成交时间'
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])

    # 移除空值
    df = df.dropna(subset=['成交价', '成交量'])

    # 过滤异常值
    df = df[df['成交量'] > 0]
    df = df[df['成交价'] > 0]

    # 按时间排序
    df = df.sort_values(time_col)

    # 重置索引
    df = df.reset_index(drop=True)

    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化列名

    Args:
        df: 原始DataFrame

    Returns:
        列名标准化后的DataFrame
    """
    # 列名映射
    column_mapping = {
        '成交时间': 'time',
        '成交价': 'current',
        '成交量': 'volume',
        '成交额': 'money',
        '性质': 'b_s'
    }

    # 添加买卖盘映射
    for i in range(1, 6):
        column_mapping[f'卖{i}'] = f'a{i}_p'
        column_mapping[f'卖{i}量'] = f'a{i}_v'
        column_mapping[f'买{i}'] = f'b{i}_p'
        column_mapping[f'买{i}量'] = f'b{i}_v'

    # 重命名列
    df = df.rename(columns=column_mapping)

    return df


def sliding_window_sampling(df: pd.DataFrame,
                           window_size: int = 60,
                           sample_interval: int = 5) -> List[pd.DataFrame]:
    """
    滑动窗口采样

    Args:
        df: 清洗后的DataFrame
        window_size: 窗口大小(秒)
        sample_interval: 采样间隔(秒)

    Returns:
        采样窗口列表
    """
    samples = []

    if len(df) == 0:
        return samples

    start_time = df['time'].iloc[0]
    end_time = df['time'].iloc[-1]

    current_time = start_time

    while True:
        # 计算窗口结束时间
        window_end = current_time + pd.Timedelta(seconds=window_size)

        # 检查是否超出数据范围
        if window_end > end_time:
            break

        # 获取窗口内数据
        window_data = df[
            (df['time'] >= current_time) &
            (df['time'] < window_end)
        ].copy()

        if len(window_data) > 0:
            # 设置窗口相对时间
            window_data['relative_time'] = (
                window_data['time'] - current_time
            ).dt.total_seconds()
            samples.append(window_data)

        # 移动到下一个采样点
        current_time += pd.Timedelta(seconds=sample_interval)

    return samples


def create_mock_data_for_testing(num_samples: int = 100) -> pd.DataFrame:
    """
    创建测试用模拟数据

    Args:
        num_samples: 样本数量

    Returns:
        模拟DataFrame
    """
    base_time = datetime(2025, 1, 2, 9, 30, 0)
    base_price = 10.0

    data = []
    for i in range(num_samples):
        time = base_time + pd.Timedelta(seconds=i)
        current = base_price + np.random.normal(0, 0.01)
        volume = int(np.random.randint(100, 1000))
        money = round(current * volume, 2)
        bs = np.random.choice(['买', '卖'])

        row = {
            '成交时间': time,
            '成交价': round(current, 2),
            '成交量': volume,
            '成交额': money,
            '性质': bs
        }

        # 添加买卖盘
        for j in range(1, 6):
            row[f'卖{j}'] = round(current + j * 0.01, 2)
            row[f'卖{j}量'] = int(np.random.randint(100, 500))
            row[f'买{j}'] = round(current - j * 0.01, 2)
            row[f'买{j}量'] = int(np.random.randint(100, 500))

        data.append(row)

    return pd.DataFrame(data)


def preprocess_pipeline(file_path: str = None,
                       df: pd.DataFrame = None,
                       window_size: int = 60,
                       sample_interval: int = 5) -> List[pd.DataFrame]:
    """
    完整的数据预处理流水线

    Args:
        file_path: CSV文件路径
        df: 已加载的DataFrame
        window_size: 窗口大小(秒)
        sample_interval: 采样间隔(秒)

    Returns:
        预处理后的样本窗口列表
    """
    # 加载数据
    if df is None:
        if file_path is None:
            raise ValueError("必须提供file_path或df参数")
        df = load_csv_data(file_path)

    # 清洗数据
    df = clean_orderbook_data(df)

    # 标准化列名
    df = standardize_column_names(df)

    # 滑动窗口采样
    samples = sliding_window_sampling(df, window_size, sample_interval)

    return samples
