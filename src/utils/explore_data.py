#!/usr/bin/env python3
"""
探索盘口数据文件结构
"""

import py7zr
import os
import pandas as pd

from src.data_processing.csv_utils import read_csv

def explore_sample_data(data_dir):
    """探索数据结构"""

    # 找到第一个压缩文件
    sample_file = None
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.7z'):
                sample_file = os.path.join(root, file)
                break
        if sample_file:
            break

    if not sample_file:
        print("未找到7z压缩文件")
        return

    print(f"探索文件: {sample_file}")

    # 解压到临时目录
    temp_dir = "temp_extract"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with py7zr.SevenZipFile(sample_file, mode='r') as z:
            z.extractall(temp_dir)
            print(f"解压内容: {z.getnames()}")

        # 查看CSV文件
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    print(f"\n查看CSV文件: {csv_path}")

                    # 读取前几行
                    df = read_csv(csv_path, nrows=10, preserve_code=False)
                    print(f"列名: {list(df.columns)}")
                    print(f"数据形状: {df.shape}")
                    print(f"前5行数据:")
                    print(df.head())

                    # 保存列名信息
                    columns_info = {
                        'filename': file,
                        'columns': list(df.columns),
                        'dtypes': df.dtypes.to_dict(),
                        'sample_data': df.head().to_dict()
                    }

                    import json
                    with open('data_structure.json', 'w', encoding='utf-8') as f:
                        json.dump(columns_info, f, ensure_ascii=False, indent=2, default=str)

                    print(f"\n列信息已保存到 data_structure.json")
                    break
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 清理临时文件
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    explore_sample_data("2025")
