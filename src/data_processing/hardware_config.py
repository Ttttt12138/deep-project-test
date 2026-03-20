"""
硬件优化配置文件 - 针对高性能处理器优化
"""

import multiprocessing as mp


def get_hardware_optimized_config():
    """
    获取针对当前硬件的优化配置

    Returns:
        优化配置字典
    """
    cpu_count = mp.cpu_count()
    total_memory_gb = 31.4  # 基于您的硬件配置

    # 智能配置
    config = {
        'cpu_count': cpu_count,
        'total_memory_gb': total_memory_gb,

        # 进程数配置
        'n_workers': _calculate_optimal_workers(cpu_count, total_memory_gb),

        # 进程池配置
        'maxtasksperchild': 50,  # 每个进程处理50个任务后重启
        'chunksize': 3,          # 每次分配3个任务给工作进程

        # 内存优化
        'use_float32': True,
        'use_int32': True,
        'use_category': True,

        # CSV读取优化
        'csv_engine': 'c',
        'chunksize_read': 10000,
    }

    return config


def _calculate_optimal_workers(cpu_count, total_memory_gb):
    """
    计算最优进程数

    Args:
        cpu_count: CPU核心数
        total_memory_gb: 总内存（GB）

    Returns:
        最优进程数
    """
    # 基于CPU核心数计算
    if cpu_count >= 24:
        # 高性能处理器：使用20个进程（保留4个核心给系统）
        n_workers = 20
    elif cpu_count >= 16:
        # 中高端处理器
        n_workers = 14
    elif cpu_count >= 8:
        # 中等处理器
        n_workers = 6
    else:
        # 低端处理器
        n_workers = max(1, cpu_count - 1)

    # 基于内存调整（每个进程预估需要1GB内存）
    max_workers_by_memory = int(total_memory_gb * 0.8)  # 保留20%内存给系统

    # 取两者中较小值
    n_workers = min(n_workers, max_workers_by_memory)

    # 确保至少有4个进程
    n_workers = max(n_workers, 4)

    return n_workers


def print_hardware_info():
    """打印硬件信息和优化配置"""
    cpu_count = mp.cpu_count()
    config = get_hardware_optimized_config()

    print("="*80)
    print("硬件优化配置")
    print("="*80)

    print(f"CPU信息:")
    print(f"  核心数: {cpu_count}")
    print(f"  优化进程数: {config['n_workers']}")
    print(f"  CPU利用率: {(config['n_workers']/cpu_count)*100:.0f}%")
    print(f"  预期加速: {config['n_workers']:.0f}x")

    print(f"\n内存优化:")
    print(f"  总内存: {config['total_memory_gb']:.1f} GB")
    print(f"  浮点数: float32 (节省50%内存)")
    print(f"  整数: int32 (节省50%内存)")
    print(f"  字符串: category类型")

    print(f"\n进程池配置:")
    print(f"  maxtasksperchild: {config['maxtasksperchild']}")
    print(f"  chunksize: {config['chunksize']}")

    print(f"\nCSV读取优化:")
    print(f"  引擎: {config['csv_engine']}")
    print(f"  分块大小: {config['chunksize_read']}")

    print("="*80)


if __name__ == "__main__":
    # 打印硬件优化配置
    print_hardware_info()