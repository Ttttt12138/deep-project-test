"""
涨停数据处理器测试用例
遵循TDD原则，先写测试，再写实现
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import tempfile
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestDataLoader:
    """测试数据加载器"""

    def test_load_csv_success(self):
        """测试成功加载CSV文件"""
        # Arrange
        test_data = self._create_mock_tick_data()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Act
            from src.data_processing.limit_up_processor import load_tick_csv
            df = load_tick_csv(temp_file)

            # Assert
            assert df is not None
            assert len(df) == len(test_data)
            assert list(df.columns) == list(test_data.columns)
        finally:
            os.unlink(temp_file)

    def test_load_csv_file_not_found(self):
        """测试文件不存在的情况"""
        # Arrange
        non_existent_file = "/tmp/non_existent_file.csv"

        # Act & Assert
        from src.data_processing.limit_up_processor import load_tick_csv
        with pytest.raises(FileNotFoundError):
            load_tick_csv(non_existent_file)

    def test_load_csv_empty_file(self):
        """测试空文件的情况"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name

        try:
            # Act & Assert
            from src.data_processing.limit_up_processor import load_tick_csv
            with pytest.raises(ValueError):
                load_tick_csv(temp_file)
        finally:
            os.unlink(temp_file)

    def _create_mock_tick_data(self):
        """创建模拟的tick数据"""
        data = {
            'time': ['20250102091500', '20250102091509', '20250102091518'],
            'open': [10.00, 10.01, 10.02],
            'current': [10.00, 10.01, 10.02],
            'high': [10.02, 10.03, 10.04],
            'low': [9.99, 10.00, 10.01],
            'total_volume': [1000, 2000, 3000],
            'total_money': [10000.0, 20020.0, 30060.0],
            'volume': [1000, 1000, 1000],
            'money': [10000.0, 10020.0, 10040.0],
            'a5_v': [100, 100, 100], 'a5_p': [10.06, 10.07, 10.08],
            'a4_v': [100, 100, 100], 'a4_p': [10.05, 10.06, 10.07],
            'a3_v': [100, 100, 100], 'a3_p': [10.04, 10.05, 10.06],
            'a2_v': [100, 100, 100], 'a2_p': [10.03, 10.04, 10.05],
            'a1_v': [100, 100, 100], 'a1_p': [10.02, 10.03, 10.04],
            'b1_v': [100, 100, 100], 'b1_p': [10.00, 10.01, 10.02],
            'b2_v': [100, 100, 100], 'b2_p': [9.99, 10.00, 10.01],
            'b3_v': [100, 100, 100], 'b3_p': [9.98, 9.99, 10.00],
            'b4_v': [100, 100, 100], 'b4_p': [9.97, 9.98, 9.99],
            'b5_v': [100, 100, 100], 'b5_p': [9.96, 9.97, 9.98],
            'b/s': ['B', 'S', 'B']
        }
        return pd.DataFrame(data)


class TestDataCleaner:
    """测试数据清洗器"""

    def test_filter_invalid_data(self):
        """测试过滤无效数据"""
        # Arrange
        data = {
            'time': ['20250102091500', '20250102091509', '20250102091518', '20250102091527'],
            'current': [10.00, 0.00, 10.02, 0.00],
            'open': [10.00, 0.00, 10.02, 0.00],
            'volume': [1000, 0, 1000, 0]
        }
        df = pd.DataFrame(data)

        # Act
        from src.data_processing.limit_up_processor import filter_invalid_ticks
        cleaned_df = filter_invalid_ticks(df)

        # Assert
        assert len(cleaned_df) == 2
        assert all(cleaned_df['current'] > 0)
        assert all(cleaned_df['volume'] > 0)

    def test_convert_time_format(self):
        """测试时间格式转换"""
        # Arrange
        data = {
            'time': ['20250102091500', '20250102091509', '20250102091518'],
            'current': [10.00, 10.01, 10.02]
        }
        df = pd.DataFrame(data)

        # Act
        from src.data_processing.limit_up_processor import convert_time_column
        converted_df = convert_time_column(df)

        # Assert
        assert pd.api.types.is_datetime64_any_dtype(converted_df['time'])
        assert converted_df['time'].iloc[0] == pd.to_datetime('2025-01-02 09:15:00')

    def test_sort_by_time(self):
        """测试按时间排序"""
        # Arrange
        data = {
            'time': ['20250102091518', '20250102091500', '20250102091509'],
            'current': [10.02, 10.00, 10.01]
        }
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S')

        # Act
        from src.data_processing.limit_up_processor import sort_by_time
        sorted_df = sort_by_time(df)

        # Assert
        assert list(sorted_df['time'].values) == sorted(sorted_df['time'].values)
        assert sorted_df['current'].iloc[0] == 10.00


class TestLimitPriceCalculator:
    """测试涨停价计算器"""

    def test_calculate_normal_stock_limit_price(self):
        """测试普通股票涨停价计算"""
        # Arrange
        preclose = 10.00
        limit_ratio = 0.10  # 10%

        # Act
        from src.data_processing.limit_up_processor import calculate_limit_price
        limit_price = calculate_limit_price(preclose, limit_ratio)

        # Assert
        assert limit_price == 11.00

    def test_calculate_st_stock_limit_price(self):
        """测试ST股票涨停价计算"""
        # Arrange
        preclose = 10.00
        limit_ratio = 0.05  # 5%

        # Act
        from src.data_processing.limit_up_processor import calculate_limit_price
        limit_price = calculate_limit_price(preclose, limit_ratio)

        # Assert
        assert limit_price == 10.50

    def test_calculate_gem_stock_limit_price(self):
        """测试创业板涨停价计算"""
        # Arrange
        preclose = 10.00
        limit_ratio = 0.20  # 20%

        # Act
        from src.data_processing.limit_up_processor import calculate_limit_price
        limit_price = calculate_limit_price(preclose, limit_ratio)

        # Assert
        assert limit_price == 12.00


class TestLimitUpProcessor:
    """测试涨停数据处理器整体流程"""

    def test_process_single_tick_file(self):
        """测试处理单个tick文件"""
        # Arrange
        test_data = TestDataLoader()._create_mock_tick_data()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Act
            from src.data_processing.limit_up_processor import process_tick_file
            result_df = process_tick_file(
                temp_file,
                preclose=10.00,
                limit_ratio=0.10
            )

            # Assert
            assert result_df is not None
            assert len(result_df) > 0
            assert 'time' in result_df.columns
            assert 'current' in result_df.columns
            assert 'limit_price' in result_df.columns
        finally:
            os.unlink(temp_file)

    def test_process_tick_file_with_mock_data(self):
        """测试用模拟数据处理tick文件"""
        # Arrange
        from src.data_processing.limit_up_processor import create_mock_tick_data

        # Act
        df = create_mock_tick_data(num_ticks=10)

        # Assert
        assert len(df) == 10
        assert all(col in df.columns for col in ['time', 'current', 'volume', 'a1_p', 'a1_v', 'b1_p', 'b1_v'])


class TestDataValidation:
    """测试数据验证器"""

    def test_validate_required_columns(self):
        """测试验证必需列"""
        # Arrange
        df = pd.DataFrame({
            'time': [1, 2, 3],
            'current': [10.0, 10.1, 10.2]
        })

        # Act
        from src.data_processing.limit_up_processor import validate_required_columns
        result = validate_required_columns(df, ['time', 'current'])

        # Assert
        assert result is True

    def test_validate_missing_columns(self):
        """测试验证缺失列"""
        # Arrange
        df = pd.DataFrame({
            'time': [1, 2, 3],
            'current': [10.0, 10.1, 10.2]
        })

        # Act & Assert
        from src.data_processing.limit_up_processor import validate_required_columns
        with pytest.raises(ValueError):
            validate_required_columns(df, ['time', 'current', 'missing_column'])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])