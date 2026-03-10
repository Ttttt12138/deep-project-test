"""
特征工程包
"""

# 旧版特征工程（已废弃）
from .price_features import (
    extract_price_features,
    extract_price_features_vectorized
)

from .volume_features import (
    extract_volume_features,
    extract_volume_features_vectorized
)

from .orderbook_features import (
    extract_orderbook_features,
    extract_orderbook_features_vectorized
)

from .label_generation import (
    generate_return_labels,
    generate_breakout_label,
    generate_all_labels,
    generate_labels_vectorized,
    get_label_definitions
)

# 新版涨停特征工程
from .limit_up_features import (
    calculate_distance_to_limit_ratio,
    calculate_ticks_to_limit,
    calculate_ask1_to_limit_distance,
    calculate_ask1_current_spread,
    calculate_bid_volume_total,
    calculate_ask_volume_total,
    calculate_order_imbalance,
    get_bid1_volume,
    get_ask1_volume,
    calculate_spread,
    calculate_ask_depth_slope,
    calculate_bid_depth_slope,
    calculate_recent_return,
    calculate_recent_volume_change,
    calculate_recent_money_change,
    extract_limit_up_features
)

from .limit_up_labels import (
    generate_next_tick_limit_up_label,
    generate_limit_up_label_with_probability,
    get_label_statistics,
    calculate_class_weights,
    validate_labels
)

# 旧版Pipeline（已废弃）
from .pipeline import (
    process_single_window,
    build_feature_dataset,
    full_pipeline,
    split_features_labels,
    get_feature_info,
    save_feature_dataset,
    load_feature_dataset
)

__all__ = [
    # 旧版特征工程（已废弃）
    'extract_price_features',
    'extract_price_features_vectorized',
    'extract_volume_features',
    'extract_volume_features_vectorized',
    'extract_orderbook_features',
    'extract_orderbook_features_vectorized',
    'generate_return_labels',
    'generate_breakout_label',
    'generate_all_labels',
    'generate_labels_vectorized',
    'get_label_definitions',

    # 新版涨停特征
    'calculate_distance_to_limit_ratio',
    'calculate_ticks_to_limit',
    'calculate_ask1_to_limit_distance',
    'calculate_ask1_current_spread',
    'calculate_bid_volume_total',
    'calculate_ask_volume_total',
    'calculate_order_imbalance',
    'get_bid1_volume',
    'get_ask1_volume',
    'calculate_spread',
    'calculate_ask_depth_slope',
    'calculate_bid_depth_slope',
    'calculate_recent_return',
    'calculate_recent_volume_change',
    'calculate_recent_money_change',
    'extract_limit_up_features',

    # 新版涨停标签
    'generate_next_tick_limit_up_label',
    'generate_limit_up_label_with_probability',
    'get_label_statistics',
    'calculate_class_weights',
    'validate_labels',

    # 旧版Pipeline（已废弃）
    'process_single_window',
    'build_feature_dataset',
    'full_pipeline',
    'split_features_labels',
    'get_feature_info',
    'save_feature_dataset',
    'load_feature_dataset'
]