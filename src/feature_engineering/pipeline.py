"""
完整的特征工程Pipeline
整合数据预处理、特征提取和标签生成
"""

import pandas as pd
from typing import List, Dict, Tuple
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing import preprocess_pipeline
from src.feature_engineering import (
    extract_price_features_vectorized,
    extract_volume_features_vectorized,
    extract_orderbook_features_vectorized,
    generate_labels_vectorized
)


def process_single_window(df: pd.DataFrame) -> pd.Series:
    """
    处理单个时间窗口，提取所有特征和标签

    Args:
        df: 窗口数据DataFrame

    Returns:
        特征和标签的Series
    """
    # 提取各类特征
    price_features = extract_price_features_vectorized(df)
    volume_features = extract_volume_features_vectorized(df)
    orderbook_features = extract_orderbook_features_vectorized(df)

    # 生成标签
    labels = generate_labels_vectorized(df)

    # 合并所有特征和标签
    all_features = pd.concat([price_features, volume_features, orderbook_features, labels])

    return all_features


def build_feature_dataset(samples: List[pd.DataFrame]) -> pd.DataFrame:
    """
    构建完整的特征数据集

    Args:
        samples: 预处理后的样本窗口列表

    Returns:
        特征数据集DataFrame
    """
    feature_list = []

    for i, sample in enumerate(samples):
        try:
            features = process_single_window(sample)
            feature_list.append(features)
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            continue

    # 构建DataFrame
    if len(feature_list) > 0:
        feature_df = pd.DataFrame(feature_list)
        return feature_df
    else:
        return pd.DataFrame()


def full_pipeline(file_path: str = None,
                 df: pd.DataFrame = None,
                 window_size: int = 60,
                 sample_interval: int = 5) -> pd.DataFrame:
    """
    完整的数据处理流水线

    Args:
        file_path: CSV文件路径
        df: 已加载的DataFrame
        window_size: 窗口大小(秒)
        sample_interval: 采样间隔(秒)

    Returns:
        特征数据集DataFrame
    """
    # 数据预处理
    samples = preprocess_pipeline(
        file_path=file_path,
        df=df,
        window_size=window_size,
        sample_interval=sample_interval
    )

    print(f"预处理完成，生成 {len(samples)} 个样本窗口")

    # 特征工程
    feature_df = build_feature_dataset(samples)

    if len(feature_df) > 0:
        print(f"特征工程完成，生成 {len(feature_df)} 个样本，{len(feature_df.columns)} 个特征")
    else:
        print("特征工程完成，但未生成任何样本")

    return feature_df


def split_features_labels(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    分离特征和标签

    Args:
        feature_df: 包含特征和标签的DataFrame

    Returns:
        (特征DataFrame, 标签DataFrame)
    """
    # 识别标签列
    label_columns = [col for col in feature_df.columns if col.startswith('y_')]

    # 分离特征和标签
    features_df = feature_df.drop(columns=label_columns)
    labels_df = feature_df[label_columns]

    return features_df, labels_df


def get_feature_info(feature_df: pd.DataFrame) -> Dict:
    """
    获取特征信息

    Args:
        feature_df: 特征DataFrame

    Returns:
        特征信息字典
    """
    info = {
        'num_samples': len(feature_df),
        'num_features': len(feature_df.columns),
        'feature_names': list(feature_df.columns),
        'data_types': feature_df.dtypes.to_dict(),
        'missing_values': feature_df.isnull().sum().to_dict(),
        'statistics': feature_df.describe().to_dict()
    }

    # 分离特征和标签
    features_df, labels_df = split_features_labels(feature_df)
    info['num_feature_columns'] = len(features_df.columns)
    info['num_label_columns'] = len(labels_df.columns)
    info['label_names'] = list(labels_df.columns)

    return info


def save_feature_dataset(feature_df: pd.DataFrame, output_path: str):
    """
    保存特征数据集

    Args:
        feature_df: 特征DataFrame
        output_path: 输出文件路径
    """
    feature_df.to_csv(output_path, index=False)
    print(f"特征数据集已保存到: {output_path}")


def load_feature_dataset(input_path: str) -> pd.DataFrame:
    """
    加载特征数据集

    Args:
        input_path: 输入文件路径

    Returns:
        特征DataFrame
    """
    feature_df = pd.read_csv(input_path)
    print(f"特征数据集已加载: {len(feature_df)} 个样本，{len(feature_df.columns)} 个特征")
    return feature_df