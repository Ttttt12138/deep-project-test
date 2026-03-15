"""
数据集构建器使用示例
演示如何使用新的两步处理策略构建数据集
"""

from scripts.dataset_builder import process_single_day_two_step, DatasetBuilder
import os

def example_1_basic_usage():
    """示例1: 基本使用"""
    print("=" * 60)
    print("示例1: 基本使用 - 处理单个交易日")
    print("=" * 60)

    dataset_path = process_single_day_two_step(
        archive_path="2025/1/2025-01-02.7z",
        output_dir="d:/Qoder-project/deep project/data"
    )

    if dataset_path:
        print(f"\n成功! 数据集已保存到: {dataset_path}")
    else:
        print("\n失败! 无法构建数据集")

def example_2_with_split():
    """示例2: 划分数据集"""
    print("\n" + "=" * 60)
    print("示例2: 划分数据集 - 训练/验证/测试集")
    print("=" * 60)

    dataset_path = process_single_day_two_step(
        archive_path="2025/1/2025-01-02.7z",
        output_dir="d:/Qoder-project/deep project/data",
        split_dataset=True,
        train_ratio=0.70,
        val_ratio=0.15
    )

    if dataset_path:
        print(f"\n成功! 数据集已保存到: {dataset_path}")
        print("划分后的数据集位于: split_datasets/2025-01-02/")
    else:
        print("\n失败! 无法构建数据集")

def example_3_class_based():
    """示例3: 使用DatasetBuilder类"""
    print("\n" + "=" * 60)
    print("示例3: 使用DatasetBuilder类 - 更多控制")
    print("=" * 60)

    builder = DatasetBuilder(
        archive_path="2025/1/2025-01-02.7z",
        output_base_dir="d:/Qoder-project/deep project/data"
    )

    # 步骤1: 解压
    print("\n执行步骤1: 解压...")
    extract_dir = builder.step1_extract()

    if extract_dir:
        print(f"解压成功: {extract_dir}")

        # 步骤2: 处理并清理
        print("\n执行步骤2: 处理并清理...")
        output_path = "d:/Qoder-project/deep project/data/2025-01-02_manual.csv"
        success = builder.step2_process_and_cleanup(
            output_path=output_path,
            split_dataset=False
        )

        if success:
            print(f"\n成功! 数据集已保存到: {output_path}")
        else:
            print("\n失败! 无法处理数据集")
    else:
        print("\n失败! 无法解压文件")

def example_4_batch_processing():
    """示例4: 批量处理多个交易日"""
    print("\n" + "=" * 60)
    print("示例4: 批量处理多个交易日")
    print("=" * 60)

    # 要处理的交易日列表
    trading_days = [
        "2025/1/2025-01-02.7z",
        "2025/1/2025-01-03.7z",
        # 添加更多交易日...
    ]

    print(f"计划处理 {len(trading_days)} 个交易日")

    results = []
    for i, day_archive in enumerate(trading_days, 1):
        print(f"\n处理第 {i}/{len(trading_days)} 个交易日: {day_archive}")

        try:
            dataset_path = process_single_day_two_step(
                archive_path=day_archive,
                output_dir="d:/Qoder-project/deep project/data"
            )

            results.append({
                'archive': day_archive,
                'status': 'success',
                'path': dataset_path
            })

            print(f"  ✓ 处理成功")

        except Exception as e:
            results.append({
                'archive': day_archive,
                'status': 'failed',
                'error': str(e)
            })

            print(f"  ✗ 处理失败: {e}")

    # 显示批量处理结果
    print("\n" + "-" * 60)
    print("批量处理结果汇总:")
    print("-" * 60)

    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')

    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {failed_count}/{len(results)}")

    if failed_count > 0:
        print("\n失败的交易日:")
        for result in results:
            if result['status'] == 'failed':
                print(f"  - {result['archive']}: {result.get('error', 'Unknown error')}")

def main():
    """运行示例"""
    print("数据集构建器使用示例")
    print("请选择要运行的示例:")
    print("1. 基本使用")
    print("2. 划分数据集")
    print("3. 使用DatasetBuilder类")
    print("4. 批量处理多个交易日")
    print("5. 运行所有示例")

    choice = input("\n请输入选项 (1-5): ").strip()

    if choice == '1':
        example_1_basic_usage()
    elif choice == '2':
        example_2_with_split()
    elif choice == '3':
        example_3_class_based()
    elif choice == '4':
        example_4_batch_processing()
    elif choice == '5':
        example_1_basic_usage()
        example_2_with_split()
        example_3_class_based()
        # example_4_batch_processing()  # 取消注释以运行
    else:
        print("无效的选项!")

if __name__ == "__main__":
    main()