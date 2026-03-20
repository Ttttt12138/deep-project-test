"""
特征工程包 - V3.2一板一样本版本
"""

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

# V3.2首次触板标签（新核心模块 - 一板一样本）
from .event_driven_labels import (
    generate_event_driven_label,  # V3.2首次触板逻辑
    filter_and_label_events,       # V3.2首次触板过滤
    get_event_statistics,
    validate_event_driven_labels
)

# V3.2事件窗口构建器（新核心模块 - 一板一样本）
from .event_window_builder import (
    EventWindowBuilder  # V3.2一板一样本逻辑
)

# 旧版标签生成（保留向后兼容）
from .limit_up_labels import (
    generate_next_tick_limit_up_label,
    generate_limit_up_label_with_probability,
    get_label_statistics,
    calculate_class_weights,
    validate_labels
)

__all__ = [
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

    # V3.2首次触板标签（新核心 - 一板一样本）
    'generate_event_driven_label',
    'filter_and_label_events',
    'get_event_statistics',
    'validate_event_driven_labels',

    # V3.2事件窗口构建器（新核心 - 一板一样本）
    'EventWindowBuilder',

    # 旧版标签生成（向后兼容）
    'generate_next_tick_limit_up_label',
    'generate_limit_up_label_with_probability',
    'get_label_statistics',
    'calculate_class_weights',
    'validate_labels'
]