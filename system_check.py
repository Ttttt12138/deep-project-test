"""
系统自检脚本
验证涨停预测系统各个模块是否正常工作
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_imports():
    """测试模块导入"""
    print("=" * 50)
    print("1. 测试模块导入")
    print("=" * 50)

    modules_to_test = [
        ("数据预处理", "src.data_processing"),
        ("涨停数据处理器", "src.data_processing.limit_up_processor"),
        ("涨停特征工程", "src.feature_engineering"),
        ("涨停标签生成", "src.feature_engineering.limit_up_labels"),
        ("LightGBM训练器", "src.models"),
        ("数据集构建", "scripts.build_dataset"),
        ("7z处理", "scripts.extract_7z")
    ]

    success_count = 0
    failed_modules = []

    for module_name, module_path in modules_to_test:
        try:
            __import__(module_path)
            print(f"[OK] {module_name}: {module_path}")
            success_count += 1
        except ImportError as e:
            print(f"[FAIL] {module_name}: {module_path}")
            print(f"  错误: {e}")
            failed_modules.append(module_name)

    print(f"\n导入结果: {success_count}/{len(modules_to_test)} 成功")

    if failed_modules:
        print(f"失败的模块: {', '.join(failed_modules)}")
        return False

    return True


def test_limit_up_processor():
    """测试涨停数据处理器"""
    print("\n" + "=" * 50)
    print("2. 测试涨停数据处理器")
    print("=" * 50)

    try:
        from src.data_processing.limit_up_processor import (
            create_mock_tick_data,
            process_tick_file
        )

        # 创建模拟数据
        df = create_mock_tick_data(num_ticks=10)
        print(f"[OK] 创建模拟数据: {len(df)} 条tick")

        # 保存临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # 处理tick文件
            processed_df = process_tick_file(temp_file, preclose=10.0, limit_ratio=0.10)
            print(f"[OK] 处理tick文件: {len(processed_df)} 条有效tick")
            print(f"[OK] 包含字段: {list(processed_df.columns)[:5]}...")

            return True

        finally:
            os.unlink(temp_file)

    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_limit_up_features():
    """测试涨停特征提取"""
    print("\n" + "=" * 50)
    print("3. 测试涨停特征提取")
    print("=" * 50)

    try:
        from src.data_processing.limit_up_processor import create_mock_tick_data
        from src.feature_engineering.limit_up_features import extract_limit_up_features

        # 创建并处理模拟数据
        df = create_mock_tick_data(num_ticks=10)
        df['limit_price'] = 11.0  # 设置涨停价

        # 提取特征
        feature_df = extract_limit_up_features(df, tick_size=0.01)
        feature_cols = [col for col in feature_df.columns if col.startswith('f')]

        print(f"[OK] 提取特征: {len(feature_cols)} 个")
        print(f"[OK] 特征列名: {', '.join(feature_cols[:5])}...")

        return True

    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_limit_up_labels():
    """测试涨停标签生成"""
    print("\n" + "=" * 50)
    print("4. 测试涨停标签生成")
    print("=" * 50)

    try:
        from src.data_processing.limit_up_processor import create_mock_tick_data
        from src.feature_engineering.limit_up_labels import (
            generate_next_tick_limit_up_label,
            get_label_statistics
        )

        # 创建模拟数据
        df = create_mock_tick_data(num_ticks=10)
        df['limit_price'] = 11.0

        # 生成标签
        labeled_df = generate_next_tick_limit_up_label(df)
        stats = get_label_statistics(labeled_df)

        print(f"[OK] 生成标签: {len(labeled_df)} 条样本")
        print(f"[OK] 正样本数: {stats['positive_samples']}")
        print(f"[OK] 负样本数: {stats['negative_samples']}")
        print(f"[OK] 正样本比例: {stats['positive_ratio']:.2%}")

        return True

    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_requirements():
    """测试数据依赖包"""
    print("\n" + "=" * 50)
    print("5. 测试数据依赖包")
    print("=" * 50)

    packages_to_test = [
        "pandas",
        "numpy",
        "lightgbm",
        "sklearn",
        "py7zr"
    ]

    success_count = 0
    failed_packages = []

    for package in packages_to_test:
        try:
            __import__(package)
            print(f"[OK] {package}")
            success_count += 1
        except ImportError:
            print(f"[FAIL] {package} (未安装)")
            failed_packages.append(package)

    print(f"\n依赖包检查: {success_count}/{len(packages_to_test)} 已安装")

    if failed_packages:
        print(f"缺失的包: {', '.join(failed_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False

    return True


def main():
    """主函数"""
    print("\n涨停预测系统自检\n")

    results = []

    # 运行所有测试
    results.append(("数据依赖包", test_data_requirements()))
    results.append(("模块导入", test_imports()))
    results.append(("涨停数据处理器", test_limit_up_processor()))
    results.append(("涨停特征提取", test_limit_up_features()))
    results.append(("涨停标签生成", test_limit_up_labels()))

    # 显示结果
    print("\n" + "=" * 50)
    print("自检结果总结")
    print("=" * 50)

    for test_name, result in results:
        status = "通过" if result else "失败"
        print(f"{test_name}: {status}")

    # 总体结果
    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n所有测试通过！系统运行正常。")
        return 0
    else:
        print("\n部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)