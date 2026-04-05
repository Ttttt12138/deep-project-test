"""
涨停数据处理器
遵循单一职责原则，每个函数只做一件事
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import os
from decimal import Decimal, ROUND_HALF_UP


def load_tick_csv(file_path: str) -> pd.DataFrame:
    """
    加载tick数据CSV文件

    Args:
        file_path: CSV文件路径

    Returns:
        原始DataFrame

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件为空或格式错误
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("文件为空")

    return df


def filter_invalid_ticks(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    过滤无效的tick数据 - 实现8条清理规则

    Args:
        df: 原始DataFrame

    Returns:
        (过滤后的DataFrame, 过滤统计信息字典)

    清理规则：
        1. 无效价格：current ≤ 0
        2. 空盘口：a1_p ≤ 0 且 b1_p ≤ 0
        3. 集合竞价：time < 09:30:00
        4. 重复tick：time重复
        5. 非法盘口：a1_p < b1_p
        6. 成交异常：volume < 0 或 money < 0
        7. 异常跳价：abs(return) > 10%
        8. volume为0
    """
    initial_count = len(df)
    stats = {
        'initial_count': initial_count,
        'invalid_price': 0,
        'empty_order_book': 0,
        'call_auction': 0,
        'duplicate_ticks': 0,
        'illegal_order_book': 0,
        'abnormal_trade': 0,
        'abnormal_price_jump': 0,
        'zero_volume': 0
    }

    # 规则1: 无效价格 - current <= 0
    mask = df["current"] <= 0
    stats['invalid_price'] = mask.sum()
    df = df[~mask].copy()

    # 规则2: 空盘口 - a1_p <= 0 且 b1_p <= 0
    if 'a1_p' in df.columns and 'b1_p' in df.columns:
        mask = (df["a1_p"] <= 0) & (df["b1_p"] <= 0)
        stats['empty_order_book'] = mask.sum()
        df = df[~mask].copy()

    # 规则3: 集合竞价 - time < 09:30:00
    if 'time' in df.columns:
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], errors='coerce')

        # 提取时间部分，不考虑日期
        time_only = df['time'].dt.time
        auction_start = pd.to_datetime('09:30:00').time()

        mask = time_only < auction_start
        stats['call_auction'] = mask.sum()
        df = df[~mask].copy()

    # 规则4: 重复tick - time重复（同一股票）
    if 'time' in df.columns:
        # 保留每个时间点的第一条记录
        duplicate_mask = df.duplicated(subset=['time'], keep='first')
        stats['duplicate_ticks'] = duplicate_mask.sum()
        df = df[~duplicate_mask].copy()

    # 规则5: 非法盘口 - a1_p < b1_p（卖一价低于买一价）
    if 'a1_p' in df.columns and 'b1_p' in df.columns:
        mask = df["a1_p"] < df["b1_p"]
        stats['illegal_order_book'] = mask.sum()
        df = df[~mask].copy()

    # 规则6: 成交异常 - volume < 0 或 money < 0
    if 'volume' in df.columns:
        mask = df["volume"] < 0
        stats['abnormal_trade'] = mask.sum()
        df = df[~mask].copy()

    if 'money' in df.columns:
        mask = df["money"] < 0
        stats['abnormal_trade'] += mask.sum()
        df = df[~mask].copy()

    # 规则7: 异常跳价 - abs(return) > 10%
    if 'current' in df.columns and len(df) > 1:
        # 计算收益率
        df_temp = df.copy()
        df_temp['return'] = df_temp['current'].pct_change()

        # 标记异常跳价（第一条数据没有收益率，跳过）
        mask = (df_temp['return'].abs() > 0.10) & df_temp['return'].notna()
        stats['abnormal_price_jump'] = mask.sum()
        df = df[~mask].copy()

    # 规则8: volume为0（原有规则）
    if 'volume' in df.columns:
        mask = df["volume"] == 0
        stats['zero_volume'] = mask.sum()
        df = df[~mask].copy()

    stats['final_count'] = len(df)
    stats['filtered_count'] = initial_count - stats['final_count']
    stats['filter_ratio'] = stats['filtered_count'] / initial_count if initial_count > 0 else 0

    return df, stats


def convert_time_column(df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
    """
    转换时间列格式

    Args:
        df: DataFrame
        time_col: 时间列名

    Returns:
        转换后的DataFrame
    """
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col].astype(str), format="%Y%m%d%H%M%S")
    return df


def sort_by_time(df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
    """
    按时间升序排序

    Args:
        df: DataFrame
        time_col: 时间列名

    Returns:
        排序后的DataFrame
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def calculate_limit_price(preclose: float, limit_ratio: float) -> float:
    """
    计算涨停价（使用标准四舍五入取整到分位）

    Args:
        preclose: 昨收价
        limit_ratio: 涨停比例

    Returns:
        涨停价
    """
    d = Decimal(str(preclose * (1 + limit_ratio)))
    return float(d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def calculate_floor_price(preclose: float, limit_ratio: float) -> float:
    """
    计算跌停价（使用标准四舍五入取整到分位）

    Args:
        preclose: 昨收价
        limit_ratio: 跌停比例（与涨停比例相同）

    Returns:
        跌停价
    """
    d = Decimal(str(preclose * (1 - limit_ratio)))
    return float(d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def validate_required_columns(df: pd.DataFrame, required_cols: list) -> bool:
    """
    验证必需列是否存在

    Args:
        df: DataFrame
        required_cols: 必需列名列表

    Returns:
        是否包含所有必需列

    Raises:
        ValueError: 缺少必需列
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列: {missing_cols}")
    return True


def create_mock_tick_data(num_ticks: int = 100) -> pd.DataFrame:
    """
    创建模拟tick数据

    Args:
        num_ticks: tick数量

    Returns:
        模拟DataFrame
    """
    base_time = pd.to_datetime('2025-01-02 09:15:00')
    base_price = 10.0

    data = []
    for i in range(num_ticks):
        time = base_time + pd.Timedelta(seconds=i)
        current = base_price + np.random.normal(0, 0.01)

        row = {
            'time': time.strftime('%Y%m%d%H%M%S'),
            'open': round(current, 2),
            'current': round(current, 2),
            'high': round(current + 0.01, 2),
            'low': round(current - 0.01, 2),
            'total_volume': (i + 1) * 100,
            'total_money': round((i + 1) * 100 * current, 2),
            'volume': 100,
            'money': round(100 * current, 2),
            'b/s': np.random.choice(['B', 'S'])
        }

        # 添加买卖盘
        for j in range(1, 6):
            row[f'a{j}_v'] = np.random.randint(100, 500)
            row[f'a{j}_p'] = round(current + j * 0.01, 2)
            row[f'b{j}_v'] = np.random.randint(100, 500)
            row[f'b{j}_p'] = round(current - j * 0.01, 2)

        data.append(row)

    return pd.DataFrame(data)


def process_tick_file(file_path: str,
                      preclose: float,
                      limit_ratio: float = 0.10) -> pd.DataFrame:
    """
    处理单个tick文件的完整流程

    Args:
        file_path: CSV文件路径
        preclose: 昨收价
        limit_ratio: 涨停比例

    Returns:
        处理后的DataFrame
    """
    # 加载数据
    df = load_tick_csv(file_path)

    # 验证必需列
    required_cols = ['time', 'current', 'volume', 'open', 'high', 'low']
    validate_required_columns(df, required_cols)

    # 转换时间格式
    df = convert_time_column(df)

    # 按时间排序
    df = sort_by_time(df)

    # 过滤无效数据
    df, filter_stats = filter_invalid_ticks(df)

    # 打印过滤统计信息
    if filter_stats['filtered_count'] > 0:
        print(f"  数据过滤: {filter_stats['initial_count']} -> {filter_stats['final_count']} "
              f"({filter_stats['filtered_count']} 条, {filter_stats['filter_ratio']:.2%})")
    else:
        print(f"  数据过滤: 无需过滤")

    # 计算涨停价
    df['limit_price'] = calculate_limit_price(preclose, limit_ratio)

    return df