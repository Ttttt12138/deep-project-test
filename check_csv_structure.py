"""
检查CSV文件结构
帮助了解数据文件中有哪些列
"""

import pandas as pd
from pathlib import Path
import os

# 解压一个文件查看结构
from scripts.extract_7z import SevenZipExtractor

def check_csv_structure(archive_path, extract_dir):
    """
    检查7z文件中的CSV结构
    """
    print(f"解压文件: {archive_path}")

    # 解压文件
    extractor = SevenZipExtractor(archive_path)
    extract_dir = extractor.extract_all(extract_dir)

    # 查找CSV文件
    csv_files = list(Path(extract_dir).rglob("*.csv"))
    print(f"找到 {len(csv_files)} 个CSV文件")

    if len(csv_files) == 0:
        print("没有找到CSV文件")
        return

    # 检查前几个文件的结构
    for i, csv_file in enumerate(csv_files[:3]):
        print(f"\n文件 {i+1}: {csv_file.name}")
        print("=" * 80)

        try:
            # 读取前几行
            df = pd.read_csv(csv_file, nrows=5)

            print(f"列名: {df.columns.tolist()}")
            print(f"\n前3行数据:")
            print(df.head(3).to_string())

            print(f"\n数据类型:")
            print(df.dtypes)

            print(f"\n列的统计信息:")
            print(df.describe())

        except Exception as e:
            print(f"读取文件出错: {e}")

    # 清理临时目录
    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)

if __name__ == "__main__":
    # 检查第一个数据文件
    archive_path = "2025/01/2025-01-02.7z"
    extract_dir = "temp_structure_check"

    if os.path.exists(archive_path):
        check_csv_structure(archive_path, extract_dir)
    else:
        print(f"文件不存在: {archive_path}")