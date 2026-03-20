"""
测试单只股票的事件窗口构建
用于诊断为什么有触板事件的股票仍然失败
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 设置UTF-8编码避免控制台输出问题
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized as process_tick_file
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.feature_engineering.limit_up_features import extract_limit_up_features
from src.feature_engineering.event_window_builder import EventWindowBuilder

def test_single_stock(csv_file_path: str, stock_code: str):
    """测试单只股票的事件窗口构建"""
    print(f"\n{'='*80}")
    print(f"测试股票: {stock_code}")
    print(f"{'='*80}")

    # 获取股票类型和涨停比例
    stock_type = determine_stock_type(stock_code)
    limit_ratio = get_limit_ratio(stock_type)
    print(f"股票类型: {stock_type}, 涨停比例: {limit_ratio}")

    # 读取前100行获取基准价格
    try:
        df_sample = pd.read_csv(csv_file_path, nrows=100, encoding='utf-8')
        if 'b1_p' in df_sample.columns:
            preclose = df_sample['b1_p'][df_sample['b1_p'] > 0].iloc[0]
        elif 'current' in df_sample.columns:
            preclose = df_sample['current'][df_sample['current'] > 0].iloc[0]
        else:
            print("无法找到基准价格")
            return
        print(f"基准价格: {preclose}")
    except Exception as e:
        print(f"读取基准价格失败: {e}")
        return

    # 处理tick文件
    try:
        df = process_tick_file(csv_file_path, preclose, limit_ratio)
        print(f"处理后Tick数: {len(df)}")

        if df.empty:
            print("处理后的数据为空")
            return

        # 添加股票代码
        df['code'] = stock_code

        # 提取特征
        df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=limit_ratio)
        print(f"特征列数: {len(df.columns)}")

        # 检查必需列
        required_features = [
            'dist_to_limit', 'ticks_to_limit', 'ask1_to_limit', 'ask1_gap',
            'bid_depth', 'ask_depth', 'order_imbalance', 'b1_volume', 'a1_volume',
            'spread', 'ask_slope', 'bid_slope',
            'ret_1tick', 'vol_delta', 'money_delta'
        ]

        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"[WARNING] 缺少特征列: {missing_features}")
        else:
            print(f"[OK] 所有必需特征列都存在")

        # 使用事件窗口构建器
        window_builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

        print(f"\n开始构建事件窗口...")
        event_samples = window_builder.build_event_window_samples(df, verbose=True)

        if event_samples.empty:
            print(f"\n[FAILED] 事件窗口构建失败，结果为空")
        else:
            print(f"\n[SUCCESS] 事件窗口构建成功")
            print(f"样本数: {len(event_samples)}")
            print(f"正样本数: {int(event_samples['label'].sum())}")
            print(f"负样本数: {len(event_samples) - int(event_samples['label'].sum())}")
            print(f"特征列数: {len(event_samples.columns)}")
            print(f"\n样本列名（前10个）: {event_samples.columns.tolist()[:10]}")

            # 检查空值
            nan_counts = event_samples.isnull().sum()
            bad_cols = nan_counts[nan_counts > 0]
            if not bad_cols.empty:
                print(f"\n[WARNING] 发现空值列:")
                for col, count in bad_cols.items():
                    print(f"  {col}: {count} 个NaN")

    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 从之前成功的日志中选择一只股票进行测试
    # 选择600355，它之前有89个触板瞬间
    base_path = "D:/Qoder-project/deep project/data/temp_extract/2025-01-02/2025-01-02"

    # 尝试找几个CSV文件
    test_stocks = ["600355", "000029", "000530"]

    for stock_code in test_stocks:
        csv_file = f"{base_path}/{stock_code}.csv"
        if Path(csv_file).exists():
            test_single_stock(csv_file, stock_code)
            break
    else:
        print(f"未找到测试股票文件，请检查路径: {base_path}")

        # 如果找不到指定股票，显示可用的股票
        available_files = list(Path(base_path).glob("*.csv"))[:10]
        if available_files:
            print(f"可用的股票文件示例: {[f.stem for f in available_files]}")