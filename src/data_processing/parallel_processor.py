"""
并行数据处理优化模块 - 阶段一优化
1. 多进程并行处理股票
2. CSV读取优化
3. 数据类型优化
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
import multiprocessing as mp
from functools import partial
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized as process_tick_file
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.feature_engineering.limit_up_features import extract_limit_up_features


# 优化的CSV读取配置
OPTIMIZED_CSV_DTYPE = {
    'time': 'str',
    'open': 'float32',
    'current': 'float32',
    'high': 'float32',
    'low': 'float32',
    'total_volume': 'float32',
    'total_money': 'float32',
    'volume': 'int32',
    'money': 'float32',
    # 盘口数据使用float32
    'a5_v': 'int32', 'a5_p': 'float32',
    'a4_v': 'int32', 'a4_p': 'float32',
    'a3_v': 'int32', 'a3_p': 'float32',
    'a2_v': 'int32', 'a2_p': 'float32',
    'a1_v': 'int32', 'a1_p': 'float32',
    'b1_v': 'int32', 'b1_p': 'float32',
    'b2_v': 'int32', 'b2_p': 'float32',
    'b3_v': 'int32', 'b3_p': 'float32',
    'b4_v': 'int32', 'b4_p': 'float32',
    'b5_v': 'int32', 'b5_p': 'float32',
    'b/s': 'str'
}

# 针对高性能硬件优化的配置
HARDWARE_OPTIMIZED_CONFIG = {
    # 进程数配置：根据CPU核心数自动优化
    'max_workers_ratio': 0.83,  # 使用83%的CPU核心（24核 → 20进程）
    'min_workers': 4,           # 最少4个进程
    'max_workers': 20,          # 最多20个进程

    # 内存优化配置
    'use_float32': True,         # 使用float32节省50%内存
    'use_int32': True,           # 使用int32节省50%内存
    'use_category': True,        # 字符串使用category类型

    # 进程池优化配置 - 极致性能优化
    'maxtasksperchild': 300,     # 每个进程处理300个任务后重启（进一步减少进程重启频率）
    'chunksize': 20,             # 每次分配20个任务给工作进程（进一步减少进程间通信次数）
    'prefetch_count': 2,         # 预取2个chunk，保持进程忙碌

    # CSV读取优化
    'csv_engine': 'c',           # 使用C引擎（更快）
    'chunksize_read': 10000,     # 分块读取（大文件优化）
}


def process_single_stock_optimized(csv_file_path: str,
                                  date_str: str,
                                  apply_features: bool = True,
                                  tick_size: float = 0.01) -> Tuple[str, pd.DataFrame, str]:
    """
    优化版单只股票处理函数（用于多进程并行）

    Args:
        csv_file_path: CSV文件路径
        date_str: 日期字符串
        apply_features: 是否应用特征提取
        tick_size: 最小价格跳变

    Returns:
        (股票代码, 处理后的DataFrame, 状态信息)
    """
    try:
        csv_file = Path(csv_file_path)
        stock_code = csv_file.stem

        # 获取股票类型和涨停比例（快速查询）
        stock_type = determine_stock_type(stock_code)
        limit_ratio = get_limit_ratio(stock_type)

        # 获取基准价格
        preclose = _get_benchmark_price_optimized(csv_file)
        if preclose is None or preclose <= 0:
            return stock_code, pd.DataFrame(), "无基准价格"

        # 处理tick文件（使用优化版处理器）
        df = process_tick_file(csv_file_path, preclose, limit_ratio)
        if df.empty:
            return stock_code, pd.DataFrame(), "数据处理后为空"

        # 添加股票代码和日期
        df['code'] = stock_code
        df['date'] = date_str

        # 优化数据类型
        df = _optimize_dtypes(df)

        # 提取基础特征
        if apply_features:
            df = extract_limit_up_features(df, tick_size=tick_size, limit_ratio=limit_ratio)

        return stock_code, df, "成功"

    except Exception as e:
        return stock_code if 'stock_code' in locals() else Path(csv_file_path).stem, pd.DataFrame(), f"异常: {str(e)[:50]}"


def _get_benchmark_price_optimized(csv_file: Path) -> float:
    """
    优化版基准价格获取 - 使用优化的CSV读取

    Args:
        csv_file: CSV文件路径

    Returns:
        基准价格
    """
    try:
        # 使用优化配置读取少量行
        try:
            df_raw = pd.read_csv(csv_file, nrows=100, dtype=OPTIMIZED_CSV_DTYPE, engine='c', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_raw = pd.read_csv(csv_file, nrows=100, dtype=OPTIMIZED_CSV_DTYPE, engine='c', encoding='gbk')
            except UnicodeDecodeError:
                df_raw = pd.read_csv(csv_file, nrows=100, dtype=OPTIMIZED_CSV_DTYPE, engine='python')

        # 优先级：买一价 > 卖一价 > 开盘价 > 当前价
        for price_col in ['b1_p', 'a1_p', 'open', 'current']:
            if price_col in df_raw.columns:
                valid_prices = df_raw[df_raw[price_col] > 0][price_col]
                if len(valid_prices) > 0:
                    return float(valid_prices.iloc[0])
    except Exception:
        pass
    return None


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    优化DataFrame的数据类型，减少内存占用

    Args:
        df: 输入DataFrame

    Returns:
        优化后的DataFrame
    """
    # 优化数值类型
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    # 优化object类型
    if 'code' in df.columns:
        df['code'] = df['code'].astype('category')

    if 'date' in df.columns:
        df['date'] = df['date'].astype('category')

    return df


def process_stocks_parallel(csv_files: List[str],
                           date_str: str,
                           n_workers: int = None,
                           apply_features: bool = True,
                           tick_size: float = 0.01,
                           verbose: bool = True) -> Tuple[List[pd.DataFrame], Dict[str, int]]:
    """
    多进程并行处理多只股票 - 针对高性能CPU优化

    Args:
        csv_files: CSV文件路径列表
        date_str: 日期字符串
        n_workers: 工作进程数，默认自动优化（24核心CPU使用20个进程）
        apply_features: 是否应用特征提取
        tick_size: 最小价格跳变
        verbose: 是否显示进度信息

    Returns:
        (处理成功的DataFrame列表, 统计信息字典)
    """
    cpu_count = mp.cpu_count()

    # 智能进程数配置 - 基于CPU核心数和内存
    if n_workers is None:
        cpu_count = mp.cpu_count()

        # 针对24核心的高性能处理器优化
        if cpu_count >= 24:
            n_workers = 20  # 高性能CPU激进配置（保留4个核心给系统）
        elif cpu_count >= 16:
            n_workers = 14  # 中高端CPU
        elif cpu_count >= 8:
            n_workers = 6   # 中等CPU
        else:
            n_workers = max(1, cpu_count - 1)  # 保守配置

        # 应用硬件优化配置
        n_workers = min(n_workers, HARDWARE_OPTIMIZED_CONFIG['max_workers'])
        n_workers = max(n_workers, HARDWARE_OPTIMIZED_CONFIG['min_workers'])
    else:
        cpu_count = mp.cpu_count()

    if verbose:
        efficiency = (n_workers / cpu_count) * 100
        print(f"  硬件优化: {cpu_count}核心CPU → 使用 {n_workers} 个并行进程 ({efficiency:.0f}%利用率)")
        print(f"  预期加速: {n_workers:.0f}x (理想情况)")

    # 创建处理函数的部分应用
    process_func = partial(
        process_single_stock_optimized,
        date_str=date_str,
        apply_features=apply_features,
        tick_size=tick_size
    )

    # 使用进程池并行处理（使用优化配置）
    results = []
    stats = {
        'total': len(csv_files),
        'success': 0,
        'failed': 0,
        'error_types': {},
        'n_workers': n_workers,
        'cpu_count': cpu_count
    }

    if verbose:
        print(f"  开始并行处理 {len(csv_files)} 个文件...")
        from tqdm import tqdm
        progress = tqdm(total=len(csv_files), desc="  处理进度", unit="文件")

    try:
        # 使用优化的进程池配置
        chunksize = HARDWARE_OPTIMIZED_CONFIG['chunksize']
        maxtasksperchild = HARDWARE_OPTIMIZED_CONFIG['maxtasksperchild']

        with mp.Pool(
            processes=n_workers,
            maxtasksperchild=maxtasksperchild
        ) as pool:
            # 使用imap_unordered获取结果（无序但更快）
            for stock_code, df, status in pool.imap_unordered(
                process_func, csv_files, chunksize=chunksize
            ):
                if not df.empty:
                    results.append(df)
                    stats['success'] += 1
                else:
                    stats['failed'] += 1

                    # 统计错误类型
                    error_type = status[:20]  # 截取前20个字符作为错误类型
                    stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1

                if verbose:
                    progress.update(1)

    except Exception as e:
        if verbose:
            print(f"  并行处理异常: {e}")
        import traceback
        traceback.print_exc()

    if verbose:
        progress.close()
        print(f"  并行处理完成: 成功 {stats['success']}, 失败 {stats['failed']}")
        if stats['error_types']:
            print(f"  失败类型统计:")
            for error_type, count in sorted(stats['error_types'].items(), key=lambda x: -x[1]):
                if count > 5:  # 只显示出现次数>5的错误
                    print(f"    {error_type}: {count}")

    return results, stats


def benchmark_processing_methods(csv_files: List[str],
                                 date_str: str,
                                 sample_size: int = 10) -> Dict:
    """
    对比串行和并行处理的性能

    Args:
        csv_files: CSV文件路径列表
        date_str: 日期字符串
        sample_size: 测试样本大小

    Returns:
        性能对比结果字典
    """
    import time

    print("="*80)
    print("性能基准测试 - 串行 vs 并行")
    print("="*80)

    # 随机采样部分文件进行测试
    np.random.seed(42)
    test_files = np.random.choice(csv_files, min(sample_size, len(csv_files)), replace=False).tolist()

    print(f"测试样本: {len(test_files)} 个文件")

    # 测试串行处理
    print("\n测试串行处理...")
    start_time = time.time()

    serial_results = []
    for csv_file in test_files:
        try:
            stock_code, df, status = process_single_stock_optimized(csv_file, date_str)
            if not df.empty:
                serial_results.append(df)
        except Exception as e:
            pass

    serial_time = time.time() - start_time
    print(f"  串行处理时间: {serial_time:.2f} 秒")
    print(f"  成功处理: {len(serial_results)} 个文件")

    # 测试并行处理
    print("\n测试并行处理...")
    start_time = time.time()

    n_workers = max(1, mp.cpu_count() - 1)
    parallel_results, _ = process_stocks_parallel(
        test_files, date_str,
        n_workers=n_workers,
        verbose=False
    )

    parallel_time = time.time() - start_time
    print(f"  并行处理时间: {parallel_time:.2f} 秒")
    print(f"  成功处理: {len(parallel_results)} 个文件")
    print(f"  使用进程数: {n_workers}")

    # 性能对比
    if parallel_time > 0:
        speedup = serial_time / parallel_time
        print(f"\n性能提升:")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  时间节省: {(1 - parallel_time/serial_time)*100:.1f}%")

    return {
        'sample_size': len(test_files),
        'serial_time': serial_time,
        'parallel_time': parallel_time,
        'serial_success': len(serial_results),
        'parallel_success': len(parallel_results),
        'speedup': serial_time / parallel_time if parallel_time > 0 else 0,
        'n_workers': n_workers
    }


if __name__ == "__main__":
    # 测试并行处理模块
    print("并行数据处理优化模块已加载")
    print("主要优化:")
    print("  1. 多进程并行处理股票")
    print("  2. 优化的CSV读取配置")
    print("  3. 数据类型优化（float32, int32）")
    print("  4. 进程池管理")

    # 可选：运行性能基准测试
    # test_dir = "data/temp_extract/2025-01-02/2025-01-02"
    # if os.path.exists(test_dir):
    #     csv_files = [str(f) for f in Path(test_dir).glob("*.csv")]
    #     if csv_files:
    #         benchmark_results = benchmark_processing_methods(csv_files, "2025-01-02", sample_size=5)