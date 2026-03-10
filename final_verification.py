"""
deep_project环境最终验证
验证包安装和系统功能
"""

import sys

def test_package_imports():
    """测试包导入"""
    print("=" * 50)
    print("deep_project环境 - 包导入测试")
    print("=" * 50)

    packages = [
        'pandas',
        'numpy',
        'lightgbm',
        'sklearn',
        'py7zr',
        'matplotlib',
        'seaborn',
        'pytest'
    ]

    success_count = 0
    for package in packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
            success_count += 1
        except ImportError as e:
            print(f"[FAIL] {package}: {e}")

    print(f"\n成功: {success_count}/{len(packages)}")
    return success_count == len(packages)


def test_project_modules():
    """测试项目模块导入"""
    print("\n" + "=" * 50)
    print("项目模块导入测试")
    print("=" * 50)

    modules = [
        'src.data_processing.limit_up_processor',
        'src.feature_engineering.limit_up_features',
        'src.feature_engineering.limit_up_labels',
        'src.models.lgbm_trainer'
    ]

    success_count = 0
    for module in modules:
        try:
            __import__(module)
            print(f"[OK] {module}")
            success_count += 1
        except ImportError as e:
            print(f"[FAIL] {module}: {e}")

    print(f"\n成功: {success_count}/{len(modules)}")
    return success_count == len(modules)


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 50)
    print("基本功能测试")
    print("=" * 50)

    try:
        from src.data_processing.limit_up_processor import create_mock_tick_data
        from src.feature_engineering.limit_up_features import extract_limit_up_features
        from src.feature_engineering.limit_up_labels import generate_next_tick_limit_up_label

        # 创建模拟数据
        df = create_mock_tick_data(num_ticks=10)
        print(f"[OK] 创建模拟数据: {len(df)} 条tick")

        # 设置涨停价
        df['limit_price'] = 11.0

        # 提取特征
        feature_df = extract_limit_up_features(df)
        feature_cols = [col for col in feature_df.columns if col.startswith('f')]
        print(f"[OK] 提取特征: {len(feature_cols)} 个")

        # 生成标签
        labeled_df = generate_next_tick_limit_up_label(feature_df)
        print(f"[OK] 生成标签: {len(labeled_df)} 条样本")

        return True

    except Exception as e:
        print(f"[FAIL] 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("deep_project环境验证\n")

    results = []
    results.append(("包导入测试", test_package_imports()))
    results.append(("项目模块测试", test_project_modules()))
    results.append(("基本功能测试", test_basic_functionality()))

    print("\n" + "=" * 50)
    print("验证结果")
    print("=" * 50)

    for name, success in results:
        status = "通过" if success else "失败"
        print(f"{name}: {status}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n所有测试通过！deep_project环境配置正确。")
        sys.exit(0)
    else:
        print("\n部分测试失败，请检查配置。")
        sys.exit(1)