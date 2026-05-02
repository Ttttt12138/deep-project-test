"""Limit-up feature engineering.

This module keeps the original 15 feature names and adds higher-value
microstructure, momentum, and session-time features. All rolling features are
computed from the current tick and historical ticks only.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


BID_VOLUME_COLS = [f"b{i}_v" for i in range(1, 6)]
ASK_VOLUME_COLS = [f"a{i}_v" for i in range(1, 6)]
BID_PRICE_COLS = [f"b{i}_p" for i in range(1, 6)]
ASK_PRICE_COLS = [f"a{i}_p" for i in range(1, 6)]
MOMENTUM_WINDOWS = (3, 5, 10, 20)


def _safe_divide(numerator, denominator, default: float = 0.0):
    result = np.divide(
        numerator,
        denominator,
        out=np.full_like(np.asarray(numerator, dtype="float64"), default, dtype="float64"),
        where=np.asarray(denominator, dtype="float64") != 0,
    )
    return result


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _group_keys(df: pd.DataFrame) -> list[str]:
    keys = [col for col in ("code", "date") if col in df.columns]
    return keys


def _grouped_series(df: pd.DataFrame, col: str):
    keys = _group_keys(df)
    if keys:
        return df.groupby(keys, sort=False)[col]
    return None


def _pct_change(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    grouped = _grouped_series(df, col)
    if grouped is not None:
        return grouped.pct_change(periods=periods)
    return df[col].pct_change(periods=periods)


def _diff(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    grouped = _grouped_series(df, col)
    if grouped is not None:
        return grouped.diff(periods=periods)
    return df[col].diff(periods=periods)


def _shift(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    grouped = _grouped_series(df, col)
    if grouped is not None:
        return grouped.shift(periods)
    return df[col].shift(periods)


def _rolling_max(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    keys = _group_keys(df)
    if keys:
        return (
            df.groupby(keys, sort=False)[col]
            .rolling(window=window, min_periods=1)
            .max()
            .reset_index(level=keys, drop=True)
        )
    return df[col].rolling(window=window, min_periods=1).max()


def _rolling_sum_bool(series: pd.Series, df: pd.DataFrame, window: int) -> pd.Series:
    tmp = series.astype(float)
    keys = _group_keys(df)
    if keys:
        work = df[keys].copy()
        work["_value"] = tmp.to_numpy()
        return (
            work.groupby(keys, sort=False)["_value"]
            .rolling(window=window, min_periods=window)
            .sum()
            .reset_index(level=keys, drop=True)
        )
    return tmp.rolling(window=window, min_periods=window).sum()


def _volume_slope(values: np.ndarray) -> np.ndarray:
    levels = np.arange(5, dtype="float64")
    x_centered = levels - levels.mean()
    denominator = np.sum(x_centered**2)
    centered = values - values.mean(axis=1, keepdims=True)
    return centered.dot(x_centered) / denominator


def _parse_time_to_seconds(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, pd.Timestamp):
        return float(value.hour * 3600 + value.minute * 60 + value.second)

    text = str(value).strip()
    if not text:
        return np.nan
    if " " in text or "-" in text or ":" in text:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return float(parsed.hour * 3600 + parsed.minute * 60 + parsed.second)

    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 14:
        digits = digits[-6:]
    elif len(digits) >= 6:
        digits = digits[-6:]
    elif len(digits) in (3, 4):
        digits = digits.zfill(4) + "00"
    else:
        return np.nan

    hour = int(digits[:2])
    minute = int(digits[2:4])
    second = int(digits[4:6])
    if hour > 23 or minute > 59 or second > 59:
        return np.nan
    return float(hour * 3600 + minute * 60 + second)


def _trading_minutes_from_open(seconds: pd.Series) -> pd.Series:
    open_s = 9 * 3600 + 30 * 60
    morning_end = 11 * 3600 + 30 * 60
    afternoon_start = 13 * 3600
    close_s = 15 * 3600

    minutes = np.zeros(len(seconds), dtype="float64")
    sec = seconds.to_numpy(dtype="float64")

    morning = (sec >= open_s) & (sec <= morning_end)
    lunch = (sec > morning_end) & (sec < afternoon_start)
    afternoon = (sec >= afternoon_start) & (sec <= close_s)
    after_close = sec > close_s

    minutes[morning] = (sec[morning] - open_s) / 60.0
    minutes[lunch] = 120.0
    minutes[afternoon] = 120.0 + (sec[afternoon] - afternoon_start) / 60.0
    minutes[after_close] = 240.0
    return pd.Series(minutes, index=seconds.index).fillna(0.0)


def _add_time_features(df: pd.DataFrame) -> None:
    if "time" not in df.columns:
        df["minutes_from_open"] = 0.0
        df["minutes_to_close"] = 240.0
        df["is_morning"] = 0
        df["is_afternoon"] = 0
        df["is_open_30min"] = 0
        df["is_close_30min"] = 0
        df["tick_interval_seconds"] = 0.0
        return

    seconds = df["time"].map(_parse_time_to_seconds)
    minutes_from_open = _trading_minutes_from_open(seconds)
    df["minutes_from_open"] = minutes_from_open
    df["minutes_to_close"] = (240.0 - minutes_from_open).clip(lower=0.0)
    df["is_morning"] = ((seconds >= 9 * 3600 + 30 * 60) & (seconds <= 11 * 3600 + 30 * 60)).astype(int)
    df["is_afternoon"] = ((seconds >= 13 * 3600) & (seconds <= 15 * 3600)).astype(int)
    df["is_open_30min"] = ((minutes_from_open >= 0.0) & (minutes_from_open <= 30.0)).astype(int)
    df["is_close_30min"] = ((minutes_from_open >= 210.0) & (minutes_from_open <= 240.0)).astype(int)

    df["_time_seconds_for_diff"] = seconds
    interval = _diff(df, "_time_seconds_for_diff", 1)
    interval = interval.where(interval >= 0, 0.0)
    df["tick_interval_seconds"] = interval.fillna(0.0)
    df.drop(columns=["_time_seconds_for_diff"], inplace=True)


def extract_limit_up_features(
    df: pd.DataFrame,
    tick_size: float = 0.01,
    limit_ratio: float = 0.10,
) -> pd.DataFrame:
    """Add limit-up prediction features to a tick DataFrame."""
    if df.empty:
        return df.copy()

    features_df = df.copy()
    numeric_cols = (
        ["current", "limit_price", "preclose", "volume", "money"]
        + BID_VOLUME_COLS
        + ASK_VOLUME_COLS
        + BID_PRICE_COLS
        + ASK_PRICE_COLS
    )
    _ensure_numeric(features_df, numeric_cols)

    if (features_df["limit_price"] <= 0).all():
        if "preclose" in features_df.columns and (features_df["preclose"] > 0).any():
            base_price = features_df["preclose"].where(features_df["preclose"] > 0).ffill().bfill()
        else:
            first_price = features_df.loc[features_df["current"] > 0, "current"]
            base_price = float(first_price.iloc[0]) if not first_price.empty else 10.0
        features_df["limit_price"] = pd.Series(base_price, index=features_df.index) * (1 + limit_ratio)

    features_df["limit_price"] = features_df["limit_price"].where(
        features_df["limit_price"] > 0,
        features_df["current"] * (1 + limit_ratio),
    )
    features_df["base_price"] = features_df["limit_price"] / (1 + limit_ratio)

    current = features_df["current"].replace(0, np.nan)
    limit_price = features_df["limit_price"].replace(0, np.nan)
    tick_size = float(tick_size) if tick_size else 0.01

    features_df["dist_to_limit"] = ((features_df["limit_price"] - features_df["current"]) / limit_price).fillna(0.0)
    features_df["ticks_to_limit"] = ((features_df["limit_price"] - features_df["current"]) / tick_size).clip(lower=0.0).fillna(0.0)
    features_df["ticks_to_limit_current"] = features_df["ticks_to_limit"]
    features_df["ask1_to_limit"] = ((features_df["limit_price"] - features_df["a1_p"]) / limit_price).fillna(0.0)
    features_df["ask1_gap"] = ((features_df["a1_p"] - features_df["current"]) / current).fillna(0.0)

    bid_values = features_df[BID_VOLUME_COLS].to_numpy(dtype="float64")
    ask_values = features_df[ASK_VOLUME_COLS].to_numpy(dtype="float64")
    bid_depth = bid_values.sum(axis=1)
    ask_depth = ask_values.sum(axis=1)
    total_depth = bid_depth + ask_depth

    features_df["bid_depth"] = bid_depth
    features_df["ask_depth"] = ask_depth
    features_df["order_imbalance"] = _safe_divide(bid_depth - ask_depth, total_depth)
    features_df["b1_volume"] = features_df["b1_v"]
    features_df["a1_volume"] = features_df["a1_v"]
    features_df["bid_ask_depth_ratio"] = _safe_divide(bid_depth, ask_depth)

    weights = np.array([5, 4, 3, 2, 1], dtype="float64")
    weighted_bid = bid_values.dot(weights)
    weighted_ask = ask_values.dot(weights)
    weighted_total = weighted_bid + weighted_ask
    features_df["weighted_bid_pressure"] = _safe_divide(weighted_bid, weighted_total, default=0.5)
    features_df["weighted_ask_pressure"] = _safe_divide(weighted_ask, weighted_total, default=0.5)

    features_df["b1_bid_share"] = _safe_divide(features_df["b1_v"].to_numpy(dtype="float64"), bid_depth)
    features_df["a1_ask_share"] = _safe_divide(features_df["a1_v"].to_numpy(dtype="float64"), ask_depth)

    features_df["spread"] = (features_df["a1_p"] - features_df["b1_p"]).fillna(0.0)
    features_df["spread_pct"] = (features_df["spread"] / current).fillna(0.0)
    features_df["ask_slope"] = _volume_slope(ask_values)
    features_df["bid_slope"] = _volume_slope(bid_values)
    features_df["ask_slope_norm"] = _safe_divide(features_df["ask_slope"].to_numpy(dtype="float64"), ask_depth / 5.0)
    features_df["bid_slope_norm"] = _safe_divide(features_df["bid_slope"].to_numpy(dtype="float64"), bid_depth / 5.0)

    ask_prices = features_df[ASK_PRICE_COLS].to_numpy(dtype="float64")
    near_limit = ask_prices >= (features_df["limit_price"].to_numpy(dtype="float64")[:, None] - 2 * tick_size)
    near_limit &= ask_prices <= (features_df["limit_price"].to_numpy(dtype="float64")[:, None] + tick_size)
    near_limit_ask_volume = (ask_values * near_limit).sum(axis=1)
    features_df["near_limit_ask_volume"] = near_limit_ask_volume
    features_df["near_limit_big_ask_ratio"] = _safe_divide(near_limit_ask_volume, ask_depth)
    features_df["near_limit_big_ask"] = ((near_limit_ask_volume > 0) & (near_limit_ask_volume >= ask_depth * 0.4)).astype(int)

    features_df["ret_1tick"] = _pct_change(features_df, "current", 1).fillna(0.0)
    features_df["vol_delta"] = _diff(features_df, "volume", 1).fillna(0.0)
    features_df["money_delta"] = _diff(features_df, "money", 1).fillna(0.0)

    up_tick = _diff(features_df, "current", 1).fillna(0.0) > 0
    for window in MOMENTUM_WINDOWS:
        features_df[f"ret_{window}tick"] = _pct_change(features_df, "current", window).fillna(0.0)
        features_df[f"vol_change_{window}tick"] = _diff(features_df, "volume", window).fillna(0.0)
        features_df[f"money_change_{window}tick"] = _diff(features_df, "money", window).fillna(0.0)
        rolling_high = _rolling_max(features_df, "current", window)
        features_df[f"high_to_limit_{window}tick"] = ((features_df["limit_price"] - rolling_high) / limit_price).fillna(0.0)
        features_df[f"continuous_up_{window}tick"] = (_rolling_sum_bool(up_tick, features_df, window) >= window).astype(int)

    features_df["price_accel"] = features_df["ret_3tick"] - features_df["ret_10tick"]
    features_df["money_accel"] = features_df["money_change_3tick"] - features_df["money_change_10tick"]
    features_df["order_imbalance_change_rate"] = _diff(features_df, "order_imbalance", 3).fillna(0.0)

    _add_time_features(features_df)

    feature_cols = [
        "dist_to_limit", "ticks_to_limit", "ask1_to_limit", "ask1_gap",
        "bid_depth", "ask_depth", "order_imbalance", "b1_volume", "a1_volume",
        "spread", "ask_slope", "bid_slope", "ret_1tick", "vol_delta", "money_delta",
        "weighted_bid_pressure", "weighted_ask_pressure", "bid_ask_depth_ratio",
        "ask_slope_norm", "bid_slope_norm", "spread_pct", "ticks_to_limit_current",
        "near_limit_ask_volume", "near_limit_big_ask_ratio", "near_limit_big_ask",
        "b1_bid_share", "a1_ask_share", "price_accel", "money_accel",
        "order_imbalance_change_rate", "minutes_from_open", "minutes_to_close",
        "is_morning", "is_afternoon", "is_open_30min", "is_close_30min",
        "tick_interval_seconds",
    ]
    for window in MOMENTUM_WINDOWS:
        feature_cols.extend([
            f"ret_{window}tick",
            f"vol_change_{window}tick",
            f"money_change_{window}tick",
            f"high_to_limit_{window}tick",
            f"continuous_up_{window}tick",
        ])

    for col in feature_cols:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    features_df.replace([np.inf, -np.inf], 0.0, inplace=True)
    features_df[feature_cols] = features_df[feature_cols].fillna(0.0)
    return features_df


# Backward-compatible helper names used by older tests/imports.
def calculate_distance_to_limit_ratio(current: float, base_price: float, limit_ratio: float) -> float:
    limit_price = round(float(base_price) * (1 + float(limit_ratio)), 2)
    return (limit_price - round(float(current), 2)) / limit_price if limit_price > 0 else 0.0


def calculate_ticks_to_limit(current: float, base_price: float, limit_ratio: float, tick_size: float) -> float:
    limit_price = float(base_price) * (1 + float(limit_ratio))
    return (limit_price - float(current)) / float(tick_size) if limit_price > current and tick_size > 0 else 0.0


def calculate_ask1_to_limit_distance(ask1_price: float, base_price: float, limit_ratio: float) -> float:
    limit_price = float(base_price) * (1 + float(limit_ratio))
    return (limit_price - float(ask1_price)) / limit_price if limit_price > 0 else 0.0


def calculate_ask1_current_spread(ask1_price: float, current: float) -> float:
    return (float(ask1_price) - float(current)) / float(current) if float(current) > 0 else 0.0


def calculate_order_imbalance(bid_volume: float, ask_volume: float) -> float:
    total = float(bid_volume) + float(ask_volume)
    return (float(bid_volume) - float(ask_volume)) / total if total > 1e-9 else 0.0


def calculate_spread(ask1_price: float, bid1_price: float) -> float:
    return float(ask1_price) - float(bid1_price)


def calculate_bid_volume_total(bid_volumes: dict[int, float]) -> float:
    return float(sum(float(v) for v in bid_volumes.values()))


def calculate_ask_volume_total(ask_volumes: dict[int, float]) -> float:
    return float(sum(float(v) for v in ask_volumes.values()))


def get_bid1_volume(bid1_volume: float) -> float:
    return float(bid1_volume)


def get_ask1_volume(ask1_volume: float) -> float:
    return float(ask1_volume)


def calculate_ask_depth_slope(ask_volumes: dict[int, float]) -> float:
    values = np.asarray([float(ask_volumes.get(i, 0.0)) for i in range(1, 6)], dtype="float64")
    return float(_volume_slope(values.reshape(1, -1))[0])


def calculate_bid_depth_slope(bid_volumes: dict[int, float]) -> float:
    values = np.asarray([float(bid_volumes.get(i, 0.0)) for i in range(1, 6)], dtype="float64")
    return float(_volume_slope(values.reshape(1, -1))[0])


def calculate_recent_return(current: float, previous_current: float) -> float:
    previous_current = float(previous_current)
    return float(current) / previous_current - 1.0 if previous_current != 0 else 0.0


def calculate_recent_volume_change(current_volume: float, previous_volume: float) -> float:
    return float(current_volume) - float(previous_volume)


def calculate_recent_money_change(current_money: float, previous_money: float) -> float:
    return float(current_money) - float(previous_money)
