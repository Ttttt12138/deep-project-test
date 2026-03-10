"""
检查7z文件内容结构
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.extract_7z import SevenZipExtractor


def check_7z_structure(archive_path: str):
    """检查7z文件结构"""
    print(f"检查文件: {archive_path}")

    extractor = SevenZipExtractor(archive_path)

    # 获取文件列表
    file_list = extractor.get_file_list()
    print(f"文件总数: {len(file_list)}")

    # 显示前20个文件
    print("\n文件列表 (前20个):")
    for i, filename in enumerate(file_list[:20], 1):
        print(f"  {i:3d}. {filename}")

    # 统计CSV文件数量
    csv_files = [f for f in file_list if f.endswith('.csv')]
    print(f"\nCSV文件数量: {len(csv_files)}")

    if csv_files:
        print(f"CSV文件示例: {csv_files[:5]}")

        # 解压并检查第一个CSV文件
        import tempfile
        import pandas as pd

        extract_dir = extractor.extract_all()

        try:
            first_csv = csv_files[0]
            csv_path = os.path.join(extract_dir, first_csv)

            print(f"\n检查文件: {first_csv}")
            df = pd.read_csv(csv_path, nrows=10)
            print(f"列名: {list(df.columns)}")
            print(f"数据形状: {df.shape}")
            print(f"前5行数据:")
            print(df.head())

        finally:
            extractor.cleanup()
    else:
        print("没有找到CSV文件")


if __name__ == "__main__":
    # 检查第一个7z文件
    archive_file = "2025/01/2025-01-02.7z"
    check_7z_structure(archive_file)