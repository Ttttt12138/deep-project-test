"""Event-window sample builder for one-board-one-sample training data."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_engineering.event_driven_labels import (
    filter_and_label_events,
    generate_event_driven_label,
)


class EventWindowBuilder:
    """Build compact event-window samples.

    Five core features are expanded over the historical window. Newer
    microstructure, momentum, and time features are included as current-tick
    snapshots so the final dataset remains near 80-100 feature columns.
    """

    BASE_FEATURES = [
        "dist_to_limit",
        "order_imbalance",
        "b1_volume",
        "a1_volume",
        "money_delta",
    ]
    SNAPSHOT_FEATURES = [
        "weighted_bid_pressure",
        "weighted_ask_pressure",
        "bid_ask_depth_ratio",
        "bid_slope_norm",
        "ask_slope_norm",
        "spread_pct",
        "ticks_to_limit_current",
        "near_limit_big_ask",
        "near_limit_big_ask_ratio",
        "b1_bid_share",
        "a1_ask_share",
        "ret_3tick",
        "ret_5tick",
        "ret_10tick",
        "ret_20tick",
        "vol_change_3tick",
        "vol_change_5tick",
        "money_change_3tick",
        "money_change_5tick",
        "high_to_limit_10tick",
        "high_to_limit_20tick",
        "continuous_up_3tick",
        "continuous_up_5tick",
        "price_accel",
        "money_accel",
        "order_imbalance_change_rate",
        "minutes_from_open",
        "minutes_to_close",
        "is_morning",
        "is_afternoon",
        "is_open_30min",
        "is_close_30min",
        "tick_interval_seconds",
    ]
    TREND_FEATURES = [
        "dist_to_limit_slope",
        "a1_volume_trend",
        "b1_volume_trend",
        "order_imbalance_pct",
        "money_delta_accel",
        "price_accel_trend",
        "weighted_bid_pressure_trend",
        "spread_pct_trend",
        "order_imbalance_change_trend",
    ]

    def __init__(self, window_size: int = 10, limit_dist_threshold: float = 0.05):
        self.window_size = window_size
        self.limit_dist_threshold = limit_dist_threshold

    def build_event_window_samples(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if len(df) < 1:
            if verbose:
                print(f"warning: no ticks ({len(df)})")
            return pd.DataFrame()

        missing_features = [f for f in self.BASE_FEATURES if f not in df.columns]
        if missing_features:
            if verbose:
                print(f"warning: missing required features: {missing_features}")
            return pd.DataFrame()

        try:
            filtered_df, valid_indices = filter_and_label_events(
                df, self.limit_dist_threshold
            )
            if len(valid_indices) == 0:
                if verbose:
                    print("no valid event-window samples after filtering")
                return pd.DataFrame()

            result_df = self._build_window_features_on_demand(filtered_df, valid_indices)
            result_df = self._smart_fill_nan(result_df)
            if verbose:
                print(f"  original ticks: {len(df):,}")
                print(f"  kept samples: {len(valid_indices):,}")
                print(f"  feature rows: {len(result_df):,}")
            return result_df
        except Exception as exc:
            if verbose:
                print(f"event-window build failed: {exc}")
                import traceback

                traceback.print_exc()
            return pd.DataFrame()

    def _values_at_lag(
        self,
        df: pd.DataFrame,
        loc_indices: np.ndarray,
        col: str,
        lag: int,
        current_values: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.zeros(len(loc_indices), dtype=np.float32)
        lag_locs = loc_indices - lag
        valid_mask = (lag_locs >= 0) & (lag_locs < len(df))
        if valid_mask.any() and col in df.columns:
            values[valid_mask] = (
                pd.to_numeric(df.iloc[lag_locs[valid_mask]][col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        if current_values is not None:
            values[~valid_mask] = current_values[~valid_mask]
        return values

    def _build_window_features_on_demand(
        self,
        df: pd.DataFrame,
        valid_indices: np.ndarray,
    ) -> pd.DataFrame:
        snapshot_cols = [col for col in self.SNAPSHOT_FEATURES if col in df.columns]
        core_cols = ["code", "time", "label"] + self.BASE_FEATURES + snapshot_cols
        core_df = df.loc[valid_indices, core_cols].copy()
        n_samples = len(core_df)
        if n_samples == 0:
            return pd.DataFrame()

        loc_indices = df.index.get_indexer(valid_indices)
        features_dict: Dict[str, np.ndarray] = {}

        for col in self.BASE_FEATURES:
            features_dict[f"{col}_last"] = (
                pd.to_numeric(core_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            )

        for col in snapshot_cols:
            features_dict[f"{col}_last"] = (
                pd.to_numeric(core_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            )

        for lag in range(1, self.window_size):
            tick_name = f"T{self.window_size - lag}"
            for col in self.BASE_FEATURES:
                current_values = features_dict[f"{col}_last"]
                features_dict[f"{col}_{tick_name}"] = self._values_at_lag(
                    df, loc_indices, col, lag, current_values=current_values
                )

        if "time" in df.columns:
            start_locs = np.maximum(loc_indices - self.window_size + 1, 0)
            start_times = pd.to_numeric(df.iloc[start_locs]["time"], errors="coerce").fillna(0)
            end_times = pd.to_numeric(df.iloc[loc_indices]["time"], errors="coerce").fillna(0)
            features_dict["window_start_time"] = start_times.to_numpy(dtype=np.int64)
            features_dict["window_end_time"] = end_times.to_numpy(dtype=np.int64)

        features_dict["code"] = core_df["code"].values
        features_dict["time"] = core_df["time"].values
        features_dict["label"] = core_df["label"].astype(np.int8).values
        if "date" in df.columns:
            features_dict["date"] = df.loc[valid_indices, "date"].values

        features_dict.update(self._compute_trend_features_vectorized(loc_indices, n_samples, df))
        return self._smart_fill_nan(pd.DataFrame(features_dict))

    def _smart_fill_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.columns:
            if col in {"code", "time", "label", "date"}:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
            col_lower = col.lower()
            price_like = any(
                pattern in col_lower
                for pattern in ["price", "limit", "spread", "gap", "ratio", "slope", "imbalance", "depth"]
            )
            if price_like:
                df[col] = df[col].ffill().bfill().fillna(0.0)
            else:
                df[col] = df[col].fillna(0.0)
        return df

    def _history_matrix(
        self,
        df: pd.DataFrame,
        loc_indices: np.ndarray,
        col: str,
        n_samples: int,
    ) -> np.ndarray:
        n_ticks = max(self.window_size - 1, 1)
        matrix = np.zeros((n_samples, n_ticks), dtype=np.float32)
        if col not in df.columns:
            return matrix
        for j, lag in enumerate(range(1, self.window_size)):
            matrix[:, j] = self._values_at_lag(df, loc_indices, col, lag)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def _slope(self, values: np.ndarray) -> np.ndarray:
        n_ticks = values.shape[1]
        if n_ticks <= 1:
            return np.zeros(values.shape[0], dtype=np.float32)
        x = np.arange(n_ticks, dtype=np.float32)
        x_centered = x - x.mean()
        denominator = float(np.sum(x_centered**2)) + 1e-10
        return ((values - values.mean(axis=1, keepdims=True)).dot(x_centered) / denominator).astype(np.float32)

    def _compute_trend_features_vectorized(
        self,
        loc_indices: np.ndarray,
        n_samples: int,
        df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        dist_vals = self._history_matrix(df, loc_indices, "dist_to_limit", n_samples)
        a1_vals = self._history_matrix(df, loc_indices, "a1_volume", n_samples)
        b1_vals = self._history_matrix(df, loc_indices, "b1_volume", n_samples)
        oi_vals = self._history_matrix(df, loc_indices, "order_imbalance", n_samples)
        money_vals = self._history_matrix(df, loc_indices, "money_delta", n_samples)
        price_accel_vals = self._history_matrix(df, loc_indices, "price_accel", n_samples)
        bid_pressure_vals = self._history_matrix(df, loc_indices, "weighted_bid_pressure", n_samples)
        spread_vals = self._history_matrix(df, loc_indices, "spread_pct", n_samples)

        money_accel = np.zeros(n_samples, dtype=np.float32)
        if money_vals.shape[1] >= 3:
            money_accel = np.diff(np.diff(money_vals, axis=1), axis=1).mean(axis=1).astype(np.float32)

        return {
            "dist_to_limit_slope": self._slope(dist_vals),
            "a1_volume_trend": (a1_vals[:, -1] - a1_vals[:, 0]).astype(np.float32),
            "b1_volume_trend": (b1_vals[:, -1] - b1_vals[:, 0]).astype(np.float32),
            "order_imbalance_pct": (oi_vals > 0).mean(axis=1).astype(np.float32),
            "money_delta_accel": np.nan_to_num(money_accel, nan=0.0).astype(np.float32),
            "price_accel_trend": self._slope(price_accel_vals),
            "weighted_bid_pressure_trend": self._slope(bid_pressure_vals),
            "spread_pct_trend": self._slope(spread_vals),
            "order_imbalance_change_trend": (oi_vals[:, -1] - oi_vals[:, 0]).astype(np.float32),
        }

    def _process_single_stock(self, code: str, stock_df: pd.DataFrame) -> Dict:
        try:
            event_samples = self.build_event_window_samples(stock_df, verbose=False)
            if not event_samples.empty:
                event_samples["code"] = code
                if "date" in stock_df.columns and "date" not in event_samples.columns:
                    event_samples["date"] = stock_df["date"].iloc[0]
                return {
                    "success": True,
                    "code": code,
                    "data": event_samples,
                    "stats": {
                        "code": code,
                        "original_ticks": len(stock_df),
                        "event_samples": len(event_samples),
                        "positive_events": int(event_samples["label"].sum()),
                    },
                }
            return {"success": False, "code": code, "reason": self._check_failure_reason(stock_df)}
        except Exception as exc:
            return {"success": False, "code": code, "reason": f"exception: {exc}"}

    def build_multi_stock_event_samples(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
        n_workers: int | None = None,
    ) -> pd.DataFrame:
        if "code" not in df.columns:
            raise ValueError("data must contain a code column")
        if n_workers is None:
            n_workers = min(mp.cpu_count(), max(1, df["code"].nunique()))

        if verbose:
            print(f"building event windows for {df['code'].nunique()} stocks with {n_workers} workers")

        all_event_samples = []
        stock_stats = []
        failed_stocks = []
        tasks = list(df.groupby("code", sort=False))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_code = {
                executor.submit(self._process_single_stock, code, stock_df): code
                for code, stock_df in tasks
            }
            futures = tqdm(as_completed(future_to_code), total=len(tasks), desc="stocks") if verbose else as_completed(future_to_code)
            for future in futures:
                result = future.result()
                if result["success"] and not result["data"].empty:
                    all_event_samples.append(result["data"])
                    stock_stats.append(result["stats"])
                else:
                    failed_stocks.append({"code": result["code"], "reason": result["reason"]})

        if not all_event_samples:
            if verbose:
                print(f"no event-window samples generated; failed stocks: {len(failed_stocks)}")
            return pd.DataFrame()

        result_df = pd.concat(all_event_samples, ignore_index=True)
        result_df = self._smart_fill_nan(result_df)
        if verbose:
            self._print_building_stats(stock_stats, result_df)
        return result_df

    def _check_failure_reason(self, df: pd.DataFrame) -> str:
        if len(df) < 1:
            return "no ticks"
        missing_features = [f for f in self.BASE_FEATURES if f not in df.columns]
        if missing_features:
            return f"missing features: {missing_features[:3]}"
        if "limit_price" not in df.columns:
            return "missing limit_price"
        try:
            labeled_df = generate_event_driven_label(df)
            if labeled_df["label"].sum() == 0:
                hard_negatives = (labeled_df.get("dist_to_limit", pd.Series(dtype=float)) < self.limit_dist_threshold).sum()
                return f"no touch event; hard negatives={hard_negatives}"
        except Exception as exc:
            return f"label generation failed: {str(exc)[:40]}"
        return "filtered out"

    def _print_building_stats(self, stock_stats: List[Dict], result_df: pd.DataFrame) -> None:
        if not stock_stats:
            return
        total_ticks = sum(s["original_ticks"] for s in stock_stats)
        total_samples = sum(s["event_samples"] for s in stock_stats)
        total_positive = sum(s["positive_events"] for s in stock_stats)
        memory_mb = result_df.memory_usage(deep=True).sum() / 1024 / 1024
        print("\nevent-window sample stats:")
        print(f"  stocks: {len(stock_stats)}")
        print(f"  original ticks: {total_ticks:,}")
        print(f"  samples: {total_samples:,}")
        print(f"  positives: {total_positive:,}")
        print(f"  negatives: {total_samples - total_positive:,}")
        print(f"  memory: {memory_mb:.1f} MB")

    def get_all_feature_names(self) -> List[str]:
        features: List[str] = []
        for base_feature in self.BASE_FEATURES:
            features.append(f"{base_feature}_last")
        for snapshot_feature in self.SNAPSHOT_FEATURES:
            features.append(f"{snapshot_feature}_last")
        for t in range(1, self.window_size):
            tick_name = f"T{t}"
            for base_feature in self.BASE_FEATURES:
                features.append(f"{base_feature}_{tick_name}")
        features.extend(self.TREND_FEATURES)
        return features


def test_event_window_builder():
    n_ticks = 500
    limit_price = 12.0
    rng = np.random.default_rng(42)
    prices = 10.0 + rng.uniform(-0.5, 0.5, size=n_ticks)
    prices[100:150] = 11.8 + rng.uniform(0, 0.15, size=50)
    prices[300:305] = [11.90, 11.92, 11.94, 11.96, 11.98]
    test_df = pd.DataFrame({
        "code": ["000001"] * n_ticks,
        "time": np.arange(n_ticks),
        "current": prices,
        "limit_price": [limit_price] * n_ticks,
        "dist_to_limit": (limit_price - prices) / limit_price,
        "order_imbalance": rng.uniform(-0.5, 0.5, size=n_ticks),
        "b1_volume": 50 + rng.integers(-10, 20, size=n_ticks),
        "a1_volume": 45 + rng.integers(-10, 20, size=n_ticks),
        "money_delta": [1000] * n_ticks,
    })
    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.02)
    event_samples = builder.build_event_window_samples(test_df)
    print(f"samples={len(event_samples)}, features={len(builder.get_all_feature_names())}")
    return not event_samples.empty


if __name__ == "__main__":
    test_event_window_builder()
