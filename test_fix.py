"""
测试修复后的数据处理
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized as process_tick_file
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.feature_engineering.limit_up_features import extract_limit_up_features


def test_single_file():
    """测试单个文件处理"""
    print("="*80)
    print("测试单个文件处理")
    print("="*80)

    # 测试一个简单的文件 - 必须是CSV文件
    test_file = "2025/01/2025-01-03.7z"  # 这个会失败，因为不是CSV

    # 直接查找CSV文件
    test_dir = "data/temp_extract/2025-01-06/2025-01-06"
    if os.path.exists(test_dir):
        csv_files = list(Path(test_dir).glob("*.csv"))
        if csv_files:
            test_file = str(csv_files[0])
            print(f"使用测试CSV文件: {test_file}")
        else:
            print(f"测试目录中没有CSV文件: {test_dir}")
            return
    else:
        print(f"测试目录不存在: {test_dir}")
        return

    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return

    stock_code = Path(test_file).stem
    print(f"测试股票: {stock_code}")

    try:
        # 获取基准价格
        preclose = 11.5  # 假设
        stock_type = determine_stock_type(stock_code)
        limit_ratio = get_limit_ratio(stock_type)

        print(f"股票类型: {stock_type}, 涨停比例: {limit_ratio}")

        # 处理数据
        print("处理数据...")
        df = process_tick_file(test_file, preclose, limit_ratio)

        if df.empty:
            print("数据处理后为空")
            return

        print(f"处理成功，行数: {len(df)}")

        # 提取特征
        print("提取特征...")
        df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=limit_ratio)

        if df.empty:
            print("特征提取后为空")
            return

        print(f"特征提取成功，行数: {len(df)}")
        print("特征列:", df.columns.tolist())

        # 检查特征值
        if 'dist_to_limit' in df.columns:
            print(f"距离涨停范围: {df['dist_to_limit'].min():.4f} - {df['dist_to_limit'].max():.4f}")

        print("✅ 测试成功！")

    except Exception as e:
        print("❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single_file()