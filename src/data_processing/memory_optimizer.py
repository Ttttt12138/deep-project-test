"""
内存优化管理模块 - 减少GC压力，提升处理速度
"""
import gc
import sys
import psutil
from typing import Dict, Any
import pandas as pd
import numpy as np


class MemoryOptimizer:
    """内存优化管理器"""

    def __init__(self, aggressive_gc: bool = True):
        """
        初始化内存优化器

        Args:
            aggressive_gc: 是否启用激进的垃圾回收
        """
        self.aggressive_gc = aggressive_gc
        self.gc_threshold = 100  # GC触发阈值（MB）
        self.memory_profile = []  # 内存使用记录

    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取当前内存使用情况

        Returns:
            内存使用情况字典
        """
        process = psutil.Process()
        mem_info = process.memory_info()

        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # 物理内存（MB）
            'vms_mb': mem_info.vms / 1024 / 1024,  # 虚拟内存（MB）
            'percent': process.memory_percent(),    # 内存使用百分比
            'available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024  # 可用内存（GB）
        }

    def log_memory_usage(self, tag: str = ""):
        """记录内存使用情况"""
        mem_info = self.get_memory_usage()
        mem_info['tag'] = tag
        self.memory_profile.append(mem_info)
        return mem_info

    def force_gc(self, min_memory: float = None):
        """
        强制执行垃圾回收

        Args:
            min_memory: 最小内存阈值（MB），低于此值不执行GC
        """
        if not self.aggressive_gc:
            return False

        current_mem = self.get_memory_usage()['rss_mb']

        # 检查是否需要GC
        if min_memory and current_mem < min_memory:
            return False

        # 执行多代垃圾回收
        gc.collect()
        gc.collect(generation=1)
        gc.collect(generation=2)

        return True

    def optimize_dataframe(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        优化DataFrame的内存使用

        Args:
            df: 输入DataFrame
            inplace: 是否原地修改

        Returns:
            优化后的DataFrame
        """
        if not inplace:
            df = df.copy()

        # 优化数值类型
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        # 优化字符串类型
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df[col]) < 0.5:  # 唯一值比例小于50%
                df[col] = df[col].astype('category')

        # 删除不必要的索引
        if isinstance(df.index, pd.RangeIndex):
            pass  # RangeIndex已经是最优的
        else:
            df.reset_index(drop=True, inplace=True)

        return df

    def reduce_dataframe_memory(self, df: pd.DataFrame, target_reduction: float = 0.5) -> pd.DataFrame:
        """
        减少DataFrame内存占用

        Args:
            df: 输入DataFrame
            target_reduction: 目标减少比例（0-1）

        Returns:
            内存优化后的DataFrame
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB

        # 1. 优化数据类型
        df = self.optimize_dataframe(df, inplace=False)

        # 2. 删除全为NaN的列
        nan_cols = df.columns[df.isna().all()].tolist()
        if nan_cols:
            df.drop(columns=nan_cols, inplace=True)

        # 3. 删除重复列（如果有）
        df = df.loc[:, ~df.columns.duplicated()]

        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        reduction = (original_memory - optimized_memory) / original_memory

        return df

    def monitor_and_gc(self, threshold_mb: float = 500, tag: str = ""):
        """
        监控内存并在超过阈值时执行GC

        Args:
            threshold_mb: 内存阈值（MB）
            tag: 标签
        """
        mem_info = self.get_memory_usage()
        current_mb = mem_info['rss_mb']

        if current_mb > threshold_mb:
            if self.force_gc():
                print(f"[GC] 内存 {current_mb:.1f}MB > {threshold_mb}MB，执行垃圾回收")
                new_mem = self.get_memory_usage()['rss_mb']
                saved = current_mb - new_mem
                print(f"[GC] 释放内存: {saved:.1f}MB")
        else:
            if tag:
                self.log_memory_usage(tag)


def batch_process_with_memory_monitor(items: list, process_func, batch_size: int = 50,
                                      memory_threshold: float = 1000) -> list:
    """
    带内存监控的批量处理函数

    Args:
        items: 要处理的项目列表
        process_func: 处理函数
        batch_size: 批处理大小
        memory_threshold: 内存阈值（MB）

    Returns:
        处理结果列表
    """
    optimizer = MemoryOptimizer(aggressive_gc=True)
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        # 处理当前批次
        batch_results = [process_func(item) for item in batch]
        results.extend(batch_results)

        # 监控内存
        optimizer.monitor_and_gc(
            threshold_mb=memory_threshold,
            tag=f"batch_{i//batch_size}"
        )

        # 优化已处理结果的内存
        for j, result in enumerate(results):
            if isinstance(result, pd.DataFrame):
                results[j] = optimizer.optimize_dataframe(result, inplace=False)

    return results


def clear_large_objects(*objects):
    """
    清理大型对象并强制GC

    Args:
        objects: 要清理的对象
    """
    for obj in objects:
        if obj is not None:
            del obj

    # 强制垃圾回收
    gc.collect()
    gc.collect(generation=1)
    gc.collect(generation=2)


def optimize_for_batch_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    为批处理优化DataFrame

    Args:
        df: 输入DataFrame

    Returns:
        优化后的DataFrame
    """
    optimizer = MemoryOptimizer()

    # 执行多重优化
    df = optimizer.optimize_dataframe(df, inplace=False)
    df = optimizer.reduce_dataframe_memory(df, target_reduction=0.5)

    return df


if __name__ == "__main__":
    print("内存优化管理模块已加载")
    print("主要功能:")
    print("  1. 内存使用监控")
    print("  2. 智能垃圾回收")
    print("  3. DataFrame内存优化")
    print("  4. 批处理内存管理")

    # 示例：测试内存优化
    test_df = pd.DataFrame({
        'price': np.random.rand(10000) * 100,
        'volume': np.random.randint(0, 1000, 10000),
        'code': ['stock_' + str(i % 100) for i in range(10000)]
    })

    optimizer = MemoryOptimizer()
    print(f"\n原始内存: {test_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    optimized_df = optimizer.optimize_dataframe(test_df, inplace=False)
    print(f"优化后内存: {optimized_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")