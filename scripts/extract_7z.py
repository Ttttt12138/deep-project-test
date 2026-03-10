"""
7z文件处理脚本
用于解压和批量处理交易日数据
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


class SevenZipExtractor:
    """7z文件提取器"""

    def __init__(self, archive_path: str):
        """
        初始化提取器

        Args:
            archive_path: 7z文件路径
        """
        self.archive_path = archive_path
        self.temp_dir = None

    def extract_all(self, extract_to: str = None) -> str:
        """
        解压整个7z文件

        Args:
            extract_to: 解压目标目录，如果为None则使用临时目录

        Returns:
            解压目录路径
        """
        try:
            import py7zr

            # 创建目标目录
            if extract_to is None:
                self.temp_dir = tempfile.mkdtemp(prefix="tick_data_")
                extract_to = self.temp_dir
            else:
                os.makedirs(extract_to, exist_ok=True)

            # 解压文件
            with py7zr.SevenZipFile(self.archive_path, mode='r') as archive:
                archive.extractall(extract_to)

            print(f"已解压: {self.archive_path} -> {extract_to}")
            return extract_to

        except ImportError:
            print("错误: 未安装py7zr库，请运行: pip install py7zr")
            raise
        except Exception as e:
            print(f"解压失败: {e}")
            raise

    def get_file_list(self) -> list:
        """
        获取7z文件中的文件列表

        Returns:
            文件列表
        """
        try:
            import py7zr

            with py7zr.SevenZipFile(self.archive_path, mode='r') as archive:
                return archive.getnames()

        except Exception as e:
            print(f"获取文件列表失败: {e}")
            return []

    def cleanup(self):
        """清理临时目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"已清理临时目录: {self.temp_dir}")


def process_daily_7z_file(archive_path: str,
                         date: str,
                         output_dir: str,
                         cleanup: bool = True) -> str:
    """
    处理单个交易日的7z文件

    Args:
        archive_path: 7z文件路径
        date: 交易日期
        output_dir: 输出目录
        cleanup: 是否清理临时文件

    Returns:
        解压目录路径
    """
    extractor = SevenZipExtractor(archive_path)

    try:
        # 解压文件
        extract_dir = extractor.extract_all()

        # 这里可以添加对解压后文件的处理逻辑
        # 例如：提取特征、生成标签等

        print(f"处理完成: {date}")

        return extract_dir

    finally:
        if cleanup:
            extractor.cleanup()


def batch_process_7z_files(data_root: str,
                          output_dir: str,
                          max_files: int = None,
                          cleanup: bool = True) -> list:
    """
    批量处理7z文件

    Args:
        data_root: 数据根目录
        output_dir: 输出目录
        max_files: 最大处理文件数，None表示处理所有
        cleanup: 是否清理临时文件

    Returns:
        处理结果列表
    """
    results = []

    # 遍历数据根目录
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if not file.endswith('.7z'):
                continue

            # 检查是否达到最大处理数量
            if max_files and len(results) >= max_files:
                print(f"已达到最大处理数量: {max_files}")
                return results

            # 提取日期
            date_str = file.replace('.7z', '')
            file_path = os.path.join(root, file)

            print(f"处理文件: {file_path}")

            try:
                extract_dir = process_daily_7z_file(
                    file_path, date_str, output_dir, cleanup
                )
                results.append({
                    'date': date_str,
                    'archive_path': file_path,
                    'extract_dir': extract_dir,
                    'status': 'success'
                })

            except Exception as e:
                print(f"处理失败 {file}: {e}")
                results.append({
                    'date': date_str,
                    'archive_path': file_path,
                    'error': str(e),
                    'status': 'failed'
                })

    return results


def validate_7z_structure(archive_path: str) -> dict:
    """
    验证7z文件结构

    Args:
        archive_path: 7z文件路径

    Returns:
        验证结果字典
    """
    extractor = SevenZipExtractor(archive_path)

    try:
        # 获取文件列表
        file_list = extractor.get_file_list()

        # 统计信息
        csv_files = [f for f in file_list if f.endswith('.csv')]
        csv_count = len(csv_files)

        # 提取第一个CSV文件查看结构
        result = {
            'total_files': len(file_list),
            'csv_count': csv_count,
            'file_list': file_list[:10],  # 只显示前10个文件
            'status': 'valid'
        }

        if csv_count > 0:
            print(f"发现 {csv_count} 个CSV文件")
            # 这里可以添加对第一个CSV文件的详细检查
        else:
            result['status'] = 'no_csv_files'
            print("警告: 没有找到CSV文件")

        return result

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def main():
    """主函数示例"""
    import argparse

    parser = argparse.ArgumentParser(description='处理7z数据文件')
    parser.add_argument('--action', type=str, choices=['extract', 'validate', 'batch'],
                       default='extract', help='执行的操作')
    parser.add_argument('--file', type=str, help='7z文件路径')
    parser.add_argument('--dir', type=str, help='数据目录')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='输出目录')
    parser.add_argument('--max-files', type=int, help='最大处理文件数')

    args = parser.parse_args()

    if args.action == 'extract' and args.file:
        # 单文件解压
        extractor = SevenZipExtractor(args.file)
        extract_dir = extractor.extract_all(args.output)
        print(f"解压完成: {extract_dir}")

    elif args.action == 'validate' and args.file:
        # 验证文件结构
        result = validate_7z_structure(args.file)
        print(result)

    elif args.action == 'batch' and args.dir:
        # 批量处理
        results = batch_process_7z_files(
            args.dir, args.output, args.max_files
        )
        print(f"处理完成: {len(results)} 个文件")

    else:
        print("请提供正确的参数")


if __name__ == "__main__":
    main()