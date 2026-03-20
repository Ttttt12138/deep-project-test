"""
硬件优化效果测试 - 针对Intel Ultra 9 275HX (24核心)
"""

import sys
import os
from pathlib import Path
import time
import psutil

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.hardware_config import get_hardware_optimized_config, print_hardware_info
from src.data_processing.parallel_processor import process_stocks_parallel


def monitor_resources():
    """监控硬件资源使用情况"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_total_gb': memory.total / (1024**3)
    }


def test_hardware_optimized_processing():
    """测试硬件优化后的处理效果"""
    print("="*80)
    print("硬件优化效果测试 - Intel Ultra 9 275HX (24核心)")
    print("="*80)

    # 显示硬件配置
    print_hardware_info()

    # 检查测试数据
    test_dir = "data/temp_extract/2025-01-02/2025-01-02"
    if not os.path.exists(test_dir):
        print(f"\n测试数据目录不存在: {test_dir}")
        print(f"请先运行数据处理命令生成测试数据")
        return

    csv_files = list(Path(test_dir).glob("*.csv"))
    print(f"\n测试数据:")
    print(f"  找到 {len(csv_files)} 个CSV文件")

    # 根据文件数量调整测试规模
    if len(csv_files) >= 100:
        test_count = 100
        print(f"  测试规模: 前 {test_count} 个文件")
    elif len(csv_files) >= 50:
        test_count = 50
        print(f"  测试规模: 前 {test_count} 个文件")
    else:
        test_count = len(csv_files)
        print(f"  测试规模: 全部 {test_count} 个文件")

    test_files = csv_files[:test_count]

    # 获取优化配置
    config = get_hardware_optimized_config()
    n_workers = config['n_workers']

    print(f"\n开始硬件优化处理...")
    print(f"  使用 {n_workers} 个并行进程")
    print(f"  预期加速: {n_workers}x")

    # 开始处理
    start_time = time.time()
    start_resources = monitor_resources()

    print(f"\n初始资源状态:")
    print(f"  CPU使用率: {start_resources['cpu_percent']:.1f}%")
    print(f"  内存使用: {start_resources['memory_used_gb']:.1f}GB / {start_resources['memory_total_gb']:.1f}GB ({start_resources['memory_percent']:.1f}%)")

    stock_data_list, processing_stats = process_stocks_parallel(
        [str(f) for f in test_files],
        date_str="2025-01-02",
        n_workers=n_workers,
        apply_features=True,
        tick_size=0.01,
        verbose=True
    )

    processing_time = time.time() - start_time
    end_resources = monitor_resources()

    # 结果统计
    print(f"\n" + "="*80)
    print("处理结果统计")
    print("="*80)

    print(f"硬件资源变化:")
    print(f"  CPU使用率: {start_resources['cpu_percent']:.1f}% → {end_resources['cpu_percent']:.1f}%")
    print(f"  内存使用: {start_resources['memory_used_gb']:.1f}GB → {end_resources['memory_used_gb']:.1f}GB")
    memory_delta = end_resources['memory_used_gb'] - start_resources['memory_used_gb']
    print(f"  内存增加: {memory_delta:+.1f}GB")

    print(f"\n处理统计:")
    print(f"  总文件数: {processing_stats['total']}")
    print(f"  成功处理: {processing_stats['success']}")
    print(f"  失败处理: {processing_stats['failed']}")
    success_rate = (processing_stats['success'] / processing_stats['total']) * 100
    print(f"  成功率: {success_rate:.1f}%")

    print(f"\n性能指标:")
    print(f"  处理时间: {processing_time:.2f} 秒")
    print(f"  处理速度: {processing_stats['success']/processing_time:.2f} 文件/秒")

    # 估算单文件处理时间
    avg_time_per_file = processing_time / processing_stats['success']
    print(f"  平均单文件处理时间: {avg_time_per_file:.3f} 秒")

    # 估算总处理时间（所有文件）
    if len(csv_files) > test_count:
        estimated_total_time = avg_time_per_file * len(csv_files)
        print(f"  估算总处理时间: {estimated_total_time/60:.1f} 分钟 ({len(csv_files)} 个文件)")

    # 硬件利用率分析
    print(f"\n硬件利用率分析:")
    cpu_utilization = end_resources['cpu_percent']
    print(f"  CPU利用率: {cpu_utilization:.1f}%")
    if cpu_utilization > 80:
        print(f"  评价: 优秀（充分利用多核性能）")
    elif cpu_utilization > 60:
        print(f"  评价: 良好（大部分核心在工作）")
    else:
        print(f"  评价: 一般（可能存在IO瓶颈）")

    memory_efficiency = (end_resources['memory_percent'] / 100) * (n_workers / 24)
    print(f"  内存效率: {memory_efficiency*100:.1f}%")

    # 并行加速效果评估
    print(f"\n并行加速效果:")
    theoretical_speedup = n_workers
    actual_speedup = (test_count * avg_time_per_file) / processing_time
    efficiency = (actual_speedup / theoretical_speedup) * 100
    print(f"  理论加速比: {theoretical_speedup}x")
    print(f"  实际加速比: {actual_speedup:.2f}x")
    print(f"  并行效率: {efficiency:.1f}%")

    if efficiency > 80:
        print(f"  评价: 优秀的并行性能")
    elif efficiency > 60:
        print(f"  评价: 良好的并行性能")
    else:
        print(f"  评价: 并行性能有待提升")

    print(f"\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == "__main__":
    try:
        test_hardware_optimized_processing()
    except KeyboardInterrupt:
        print(f"\n用户中断测试")
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()