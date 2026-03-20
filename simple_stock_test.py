"""
简单的单股票测试 - 避免编码问题
"""
import pandas as pd
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.feature_engineering.limit_up_features import extract_limit_up_features
from src.feature_engineering.event_window_builder import EventWindowBuilder

def test_stock_600355():
    """测试600355这只股票"""
    csv_path = "D:/Qoder-project/deep project/data/temp_extract/2025-01-02/2025-01-02/600355.csv"
    stock_code = "600355"

    print("="*60)
    print("Testing stock:", stock_code)
    print("="*60)

    # 读取基准价格
    df_sample = pd.read_csv(csv_path, nrows=100)
    if 'b1_p' in df_sample.columns:
        preclose = df_sample['b1_p'][df_sample['b1_p'] > 0].iloc[0]
    else:
        preclose = df_sample['current'][df_sample['current'] > 0].iloc[0]

    print("Preclose price:", preclose)

    # 处理数据
    stock_type = determine_stock_type(stock_code)
    limit_ratio = get_limit_ratio(stock_type)

    df = process_tick_file_optimized(csv_path, preclose, limit_ratio)
    print("Processed ticks:", len(df))

    if df.empty:
        print("Empty DataFrame after processing")
        return

    df['code'] = stock_code

    # 提取特征
    df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=limit_ratio)
    print("Features extracted:", len(df.columns))

    # 检查必需特征
    required = ['dist_to_limit', 'ticks_to_limit', 'current', 'limit_price']
    missing = [f for f in required if f not in df.columns]
    if missing:
        print("Missing features:", missing)
        return

    # 构建事件窗口
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)

    print("\nBuilding event windows...")
    try:
        event_df = builder.build_event_window_samples(df, verbose=True)
    except Exception as e:
        print(f"\nException during event window building: {e}")
        import traceback
        traceback.print_exc()
        return

    if event_df.empty:
        print("\n[FAILED] Event window building failed - empty result")
    else:
        print("\n[SUCCESS] Event window building completed")
        print("Sample count:", len(event_df))
        print("Positive samples:", int(event_df['label'].sum()))
        print("Negative samples:", len(event_df) - int(event_df['label'].sum()))
        print("Columns:", len(event_df.columns))

        # 显示前几个样本
        print("\nFirst 5 samples:")
        print(event_df[['code', 'label', 'dist_to_limit_last']].head())

if __name__ == "__main__":
    try:
        test_stock_600355()
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()