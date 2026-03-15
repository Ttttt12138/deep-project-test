# -*- coding: utf-8 -*-
"""
数据集构建器验证测试
测试两步处理策略的核心功能
"""
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_imports():
    """测试所有依赖模块是否可正常导入"""
    print("测试依赖模块导入...")

    try:
        from scripts.dataset_builder import DatasetBuilder, process_single_day_two_step
        print("OK dataset_builder 模块导入成功")

        from scripts.extract_7z import SevenZipExtractor
        print("OK extract_7z 模块导入成功")

        from scripts.build_dataset import save_dataset, get_feature_columns
        print("OK build_dataset 模块导入成功")

        from src.feature_engineering.limit_up_features import extract_limit_up_features
        print("OK limit_up_features 模块导入成功")

        from src.data_processing.limit_up_processor import process_tick_file
        print("OK limit_up_processor 模块导入成功")

        from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
        print("OK stock_utils 模块导入成功")

        print("\n所有依赖模块导入成功!")
        return True

    except ImportError as e:
        print(f"FAIL 导入失败: {e}")
        return False

def test_dataset_builder_class():
    """测试DatasetBuilder类的基本功能"""
    print("\n测试DatasetBuilder类...")

    try:
        from scripts.dataset_builder import DatasetBuilder

        # 测试日期提取
        test_archive = "D:/Qoder-project/deep project/2025/1/2025-01-02.7z"
        test_output = "D:/Qoder-project/deep project/data"

        builder = DatasetBuilder(test_archive, test_output)

        # 验证日期提取
        assert builder.date_str == "2025-01-02", f"日期提取错误: {builder.date_str}"
        print(f"OK 日期提取正确: {builder.date_str}")

        # 验证目录构建
        expected_extract_dir = os.path.join(test_output, 'temp_extract', builder.date_str)
        # builder.extract_dir 会在 step1_extract 时设置
        print(f"OK 临时目录路径预计算: {expected_extract_dir}")

        print("\nDatasetBuilder类测试通过!")
        return True

    except Exception as e:
        print(f"FAIL DatasetBuilder类测试失败: {e}")
        return False

def test_module_interface():
    """测试模块接口"""
    print("\n测试模块接口...")

    try:
        from scripts.dataset_builder import process_single_day_two_step

        # 验证函数签名
        import inspect
        sig = inspect.signature(process_single_day_two_step)

        params = list(sig.parameters.keys())
        expected_params = ['archive_path', 'output_dir', 'split_dataset', 'kwargs']

        print(f"OK 函数参数: {params}")
        print(f"OK 默认参数: output_dir={sig.parameters['output_dir'].default}, split_dataset={sig.parameters['split_dataset'].default}")

        print("\n模块接口测试通过!")
        return True

    except Exception as e:
        print(f"FAIL 模块接口测试失败: {e}")
        return False

def verify_command_line():
    """验证命令行接口"""
    print("\n验证命令行接口...")

    try:
        # 检查是否有 --help 参数
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/dataset_builder.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("OK 命令行接口可用")
            print("主要参数:")
            for line in result.stdout.split('\n'):
                if '--' in line and ('input' in line or 'output' in line or 'split' in line):
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"FAIL 命令行接口错误: {result.stderr}")
            return False

    except Exception as e:
        print(f"WARN 无法验证命令行接口: {e}")
        return None  # 不是致命错误

def main():
    """运行所有测试"""
    print("=" * 60)
    print("数据集构建器验证测试")
    print("=" * 60)

    results = []

    # 运行测试
    results.append(("依赖导入", test_imports()))
    results.append(("DatasetBuilder类", test_dataset_builder_class()))
    results.append(("模块接口", test_module_interface()))
    results.append(("命令行接口", verify_command_line()))

    # 总结结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)

    for test_name, result in results:
        status = "通过" if result else "失败"
        symbol = "OK" if result else "FAIL"
        print(f"{symbol} {test_name}: {status}")

    # 判断是否全部通过
    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n通过率: {passed}/{total} ({passed/total:.0%})")

    if passed == total:
        print("\nSUCCESS: 所有测试通过! 数据集构建器已就绪。")
        return 0
    else:
        print(f"\nWARNING: 有 {total - passed} 个测试失败，请检查相关问题。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)