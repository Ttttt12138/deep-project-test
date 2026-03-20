"""
性能优化配置 - 针对高速数据处理
基于当前5094只股票，预计50+分钟处理时间的优化方案
"""

# ==========================================
# 硬件配置优化
# ==========================================
HARDWARE_CONFIG = {
    # CPU配置 - 针对24核心高性能处理器
    'cpu_cores': 24,              # CPU核心数
    'max_workers': 24,            # 最大化使用所有核心（激进模式）
    # 或者保守配置：'max_workers': 20,  # 保留4个核心给系统

    # 进程池优化配置
    'maxtasksperchild': 200,      # 提高到200（原50），减少进程重启频率
    'chunksize': 10,              # 提高到10（原3），减少进程间通信次数
    'prefetch_count': 2,          # 预取2个chunk，保持进程忙碌
}

# ==========================================
# 内存优化配置
# ==========================================
MEMORY_CONFIG = {
    # 数据类型优化
    'use_float32': True,          # 使用float32节省50%内存
    'use_int32': True,            # 使用int32节省50%内存
    'use_category': True,         # 字符串使用category类型

    # CSV读取优化
    'csv_engine': 'c',            # 使用C引擎
    'low_memory': False,          # 不使用低内存模式（更快）
    'dtype_optimization': True,   # 启用数据类型优化

    # 内存管理
    'early_cleanup': True,        # 及时清理中间数据
    'use_iterators': True,        # 使用迭代器减少内存占用
}

# ==========================================
# 并行处理优化配置
# ==========================================
PARALLEL_CONFIG = {
    # 激进并行配置（性能优先）
    'aggressive': {
        'max_workers': 24,         # 使用所有核心
        'maxtasksperchild': 300,   # 更高的任务数/进程
        'chunksize': 15,           # 更大的chunk
        'enable_prefetch': True,   # 启用预取
    },

    # 均衡配置（推荐）
    'balanced': {
        'max_workers': 20,         # 保留4核心给系统
        'maxtasksperchild': 200,   # 适中的任务数/进程
        'chunksize': 10,           # 适中的chunk大小
        'enable_prefetch': True,   # 启用预取
    },

    # 保守配置（稳定性优先）
    'conservative': {
        'max_workers': 16,         # 保守的核心使用
        'maxtasksperchild': 100,   # 适度的任务数/进程
        'chunksize': 5,            # 较小的chunk
        'enable_prefetch': False,  # 不启用预取
    },
}

# ==========================================
# 特征提取优化配置
# ==========================================
FEATURE_CONFIG = {
    # 特征提取策略
    'lazy_evaluation': True,      # 延迟计算，避免不必要特征提取
    'vectorized_operations': True, # 使用向量化操作
    'cache_frequent_features': True, # 缓存频繁使用的特征

    # 特征选择优化
    'essential_features_only': False,  # False = 提取所有特征，True = 只提取核心特征
    'batch_feature_extraction': True,  # 批量特征提取
}

# ==========================================
# 性能监控配置
# ==========================================
MONITORING_CONFIG = {
    'enable_profiling': True,     # 启用性能分析
    'log_slow_operations': True,  # 记录慢操作
    'slow_threshold': 2.0,        # 慢操作阈值（秒）
    'progress_interval': 10,      # 进度更新间隔（秒）
}

# ==========================================
# 预期性能提升
# ==========================================
PERFORMANCE_EXPECTATIONS = {
    'current_status': {
        'stocks_remaining': 5017,   # 剩余股票数（5094 - 77）
        'current_speed': '1.5-3.5s/stock',  # 当前速度
        'estimated_remaining': '50+ minutes',  # 预计剩余时间
    },

    'optimized_status': {
        'expected_speed': '0.8-1.5s/stock',   # 优化后预期速度
        'expected_remaining': '20-30 minutes', # 优化后预计时间
        'speedup_ratio': '2.0-3.0x',          # 预期加速比
    },

    'improvements': [
        '减少进程间通信次数 (chunksize: 3→10)',
        '减少进程重启频率 (maxtasksperchild: 50→200)',
        '最大化CPU利用率 (workers: 20→24)',
        '优化内存使用，减少GC压力',
        '使用向量化操作加速特征提取'
    ]
}

# ==========================================
# 应用配置的推荐模式
# ==========================================
def get_optimized_config(mode='balanced'):
    """
    获取优化配置

    Args:
        mode: 配置模式 ('aggressive', 'balanced', 'conservative')

    Returns:
        合并后的配置字典
    """
    config = {
        **HARDWARE_CONFIG,
        **MEMORY_CONFIG,
        **PARALLEL_CONFIG[mode],
        **FEATURE_CONFIG,
        **MONITORING_CONFIG,
        'mode': mode
    }

    return config


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("性能优化配置 - 使用指南")
    print("="*60)

    print("\n1. 激进模式（最快性能）:")
    aggressive = get_optimized_config('aggressive')
    print(f"   工作进程: {aggressive['max_workers']}")
    print(f"   每进程任务数: {aggressive['maxtasksperchild']}")
    print(f"   Chunk大小: {aggressive['chunksize']}")

    print("\n2. 均衡模式（推荐）:")
    balanced = get_optimized_config('balanced')
    print(f"   工作进程: {balanced['max_workers']}")
    print(f"   每进程任务数: {balanced['maxtasksperchild']}")
    print(f"   Chunk大小: {balanced['chunksize']}")

    print("\n3. 预期性能提升:")
    print(f"   当前速度: {PERFORMANCE_EXPECTATIONS['current_status']['current_speed']}")
    print(f"   优化后速度: {PERFORMANCE_EXPECTATIONS['optimized_status']['expected_speed']}")
    print(f"   预期加速: {PERFORMANCE_EXPECTATIONS['optimized_status']['speedup_ratio']}")
    print(f"   时间节省: {PERFORMANCE_EXPECTATIONS['current_status']['estimated_remaining']} → {PERFORMANCE_EXPECTATIONS['optimized_status']['expected_remaining']}")

    print("\n4. 应用配置:")
    print("   在代码中导入并使用:")
    print("   from src.data_processing.performance_config import get_optimized_config")
    print("   config = get_optimized_config('balanced')")
    print("   # 然后将config参数传递给并行处理函数")