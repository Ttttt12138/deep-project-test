"""
真实数据处理流程诊断
检查数据处理过程中每个环节的输出
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.limit_up_processor_optimized import process_tick_file_optimized
from src.data_processing.stock_utils import determine_stock_type, get_limit_ratio
from src.feature_engineering.limit_up_features import extract_limit_up_features
from src.feature_engineering.event_driven_labels import filter_and_label_events
from src.feature_engineering.event_window_builder import EventWindowBuilder


def diagnose_single_csv(csv_path):
    """诊断单个CSV文件的处理流程"""
    print("="*60)
    print(f"诊断文件: {csv_path.name}")
    print("="*60)

    try:
        # 步骤1：获取基准价格
        print("\n步骤1: 获取基准价格...")
        preclose = None
        try:
            df_raw = pd.read_csv(csv_path, nrows=100)

            # 尝试从盘口价格获取基准价
            for price_col in ['b1_p', 'a1_p', 'open', 'current']:
                if price_col in df_raw.columns:
                    valid_prices = df_raw[df_raw[price_col] > 0][price_col]
                    if len(valid_prices) > 0:
                        preclose = valid_prices.iloc[0]
                        print(f"  基准价格: {preclose:.2f} (来源: {price_col})")
                        break

            if preclose is None or preclose <= 0:
                print(f"  [FAILED] 无法获取有效基准价格")
                return False

        except Exception as e:
            print(f"  [FAILED] 读取基准价格失败: {e}")
            return False

        # 步骤2：处理tick文件
        print("\n步骤2: 处理tick文件...")
        stock_code = csv_path.stem
        stock_type = determine_stock_type(stock_code)
        limit_ratio = get_limit_ratio(stock_type)

        try:
            df = process_tick_file_optimized(str(csv_file), preclose, limit_ratio)
            if df.empty:
                print(f"  [FAILED] 处理后数据为空")
                return False

            print(f"  原始tick数: {len(df)}")
            print(f"  价格范围: {df['current'].min():.2f} - {df['current'].max():.2f}")
            print(f"  涨停价范围: {df['limit_price'].min():.2f} - {df['limit_price'].max():.2f}")

            # 检查是否有股票达到涨停价
            has_limit_up = (df['current'] >= df['limit_price']).any()
            print(f"  是否有涨停tick: {has_limit_up}")

            if has_limit_up:
                limit_up_count = (df['current'] >= df['limit_price']).sum()
                print(f"  涨停tick数: {limit_up_count}")

        except Exception as e:
            print(f"  [FAILED] 处理tick文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 步骤3：添加标识
        print("\n步骤3: 添加标识...")
        df['code'] = stock_code
        df['date'] = '2025-01-03'

        # 步骤4：提取特征
        print("\n步骤4: 提取特征...")
        try:
            df = extract_limit_up_features(df, tick_size=0.01, limit_ratio=limit_ratio)
            print(f"  特征提取后tick数: {len(df)}")

            # 检查必需特征
            required_features = ['dist_to_limit', 'limit_price', 'current']
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                print(f"  [FAILED] 缺少必需特征: {missing_features}")
                return False

            print(f"  dist_to_limit范围: {df['dist_to_limit'].min():.6f} - {df['dist_to_limit'].max():.6f}")
            print(f"  距离<5%的tick数: {(df['dist_to_limit'] < 0.05).sum()}")
            print(f"  距离<2%的tick数: {(df['dist_to_limit'] < 0.02).sum()}")

        except Exception as e:
            print(f"  [FAILED] 特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 步骤5：V3.2首次触板标签 + 过滤
        print("\n步骤5: V3.2首次触板标签 + 过滤...")
        try:
            filtered_df, valid_indices = filter_and_label_events(df, limit_dist_threshold=0.05, debug=False)

            print(f"  过滤后保留: {len(valid_indices)}")

            # 检查触板事件
            touch_events = filtered_df[filtered_df['label'] == 1]
            print(f"  检测到的触板事件: {len(touch_events)}")

            if len(touch_events) > 0:
                print(f"  触板事件位置: {touch_events.index.tolist()}")
                for idx in touch_events.index:
                    is_preserved = idx in valid_indices
                    print(f"    触板 {idx} 保留: {is_preserved}")

            # 分析失败原因
            if len(valid_indices) == 0:
                print(f"  [ANALYSIS] 过滤后无有效样本，原因分析:")

                # 检查触板事件
                if len(touch_events) > 0:
                    print(f"    有触板事件({len(touch_events)}个)但都被过滤")
                    for idx in touch_events.index:
                        dist = filtered_df.loc[idx, 'dist_to_limit']
                        current = filtered_df.loc[idx, 'current']
                        limit = filtered_df.loc[idx, 'limit_price']
                        is_limit = current >= limit
                        print(f"      触板 {idx}: dist={dist:.6f}, is_limit={is_limit}, 保留={idx in valid_indices}")
                else:
                    print(f"    无触板事件")
                    print(f"    困难负样本数: {(filtered_df['dist_to_limit'] < 0.05).sum()}")
                    print(f"    死板状态tick数: {(filtered_df['current'] >= filtered_df['limit_price']).sum()}")

                return False

        except Exception as e:
            print(f"  [FAILED] V3.2标签过滤失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 步骤6：窗口构建
        print("\n步骤6: 窗口构建...")
        try:
            builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.05)
            result_df = builder.build_event_window_samples(filtered_df, verbose=False)

            print(f"  窗口样本数: {len(result_df)}")

            if len(result_df) > 0:
                positive_samples = result_df['label'].sum()
                print(f"  正样本数: {int(positive_samples)}")
                print(f"  负样本数: {len(result_df) - int(positive_samples)}")
                print(f"  [SUCCESS] 处理成功")
                return True
            else:
                print(f"  [FAILED] 窗口构建后无数据")
                return False

        except Exception as e:
            print(f"  [FAILED] 窗口构建失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"[ERROR] 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 查找一个实际的CSV文件进行诊断
    import glob

    # 查找已解压的数据文件
    extract_dirs = glob.glob("data/temp_extract/*/")

    if not extract_dirs:
        print("未找到已解压的数据目录，请先运行数据解压")
    else:
        # 从第一个目录中找CSV文件
        first_dir = extract_dirs[0]
        csv_files = list(Path(first_dir).rglob("*.csv"))[:5]  # 只检查前5个文件

        if not csv_files:
            print(f"未找到CSV文件在目录: {first_dir}")
        else:
            print(f"找到 {len(csv_files)} 个CSV文件，开始诊断...")

            success_count = 0
            for csv_file in csv_files:
                if diagnose_single_csv(csv_file):
                    success_count += 1
                print()

            print("\n" + "="*60)
            print(f"诊断总结: 成功 {success_count}/{len(csv_files)}")
            print("="*60)