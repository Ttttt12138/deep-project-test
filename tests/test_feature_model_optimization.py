import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.event_window_builder import EventWindowBuilder
from src.feature_engineering.limit_up_features import extract_limit_up_features


def make_tick_df(n=25):
    prices = np.linspace(10.0, 11.95, n)
    data = {
        "code": ["000001"] * n,
        "date": ["2025-01-02"] * n,
        "time": [20250102093000 + i * 100 for i in range(n)],
        "current": prices,
        "limit_price": [12.0] * n,
        "volume": np.arange(n) * 100 + 1000,
        "money": np.arange(n) * 1000 + 10000,
    }
    for level in range(1, 6):
        data[f"b{level}_v"] = 100 + level * 10 + np.arange(n)
        data[f"a{level}_v"] = 90 + level * 12 + np.arange(n)
        data[f"b{level}_p"] = prices - level * 0.01
        data[f"a{level}_p"] = prices + level * 0.01
    return pd.DataFrame(data)


def test_expanded_features_are_present_finite_and_historical_only():
    df = make_tick_df(25)
    result = extract_limit_up_features(df)
    expected = [
        "weighted_bid_pressure",
        "weighted_ask_pressure",
        "bid_ask_depth_ratio",
        "ask_slope_norm",
        "bid_slope_norm",
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
        "vol_change_20tick",
        "money_change_20tick",
        "high_to_limit_20tick",
        "continuous_up_20tick",
        "price_accel",
        "money_accel",
        "order_imbalance_change_rate",
        "minutes_from_open",
        "minutes_to_close",
        "is_morning",
        "is_open_30min",
        "tick_interval_seconds",
    ]
    for col in expected:
        assert col in result.columns
    assert np.isfinite(result[expected].to_numpy(dtype=float)).all()

    changed_future = df.copy()
    changed_future.loc[15:, "current"] = 1.0
    changed_future_result = extract_limit_up_features(changed_future)
    historical_cols = ["ret_5tick", "high_to_limit_10tick", "price_accel", "order_imbalance_change_rate"]
    pd.testing.assert_series_equal(
        result.loc[:10, historical_cols].stack(),
        changed_future_result.loc[:10, historical_cols].stack(),
        check_names=False,
    )


def test_event_window_keeps_core_history_and_adds_snapshot_features():
    df = make_tick_df(40)
    df.loc[30, "current"] = 11.98
    df.loc[31:, "current"] = 12.0
    featured = extract_limit_up_features(df)

    builder = EventWindowBuilder(window_size=10, limit_dist_threshold=0.05)
    samples = builder.build_event_window_samples(featured, verbose=False)

    assert not samples.empty
    assert "dist_to_limit_last" in samples.columns
    assert "dist_to_limit_T1" in samples.columns
    assert "weighted_bid_pressure_last" in samples.columns
    assert "price_accel_trend" in samples.columns
    assert 80 <= len(builder.get_all_feature_names()) <= 110
    assert np.isfinite(samples.drop(columns=["code", "date"], errors="ignore").select_dtypes("number")).all().all()


def make_model_df(n=96):
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    dates = np.repeat([f"2025-01-{day:02d}" for day in range(2, 8)], n // 6)
    label = ((x1 + x2) > 1.0).astype(int)
    return pd.DataFrame({
        "date": dates,
        "code": [f"{i % 6:06d}" for i in range(n)],
        "time": np.arange(n),
        "feature_a": x1,
        "feature_b": x2,
        "dist_to_limit": np.clip(0.02 - x1 * 0.005, 0, 0.05),
        "order_imbalance": x2,
        "label": label,
    })


def test_evaluation_and_baselines_return_topk_metrics():
    pytest.importorskip("sklearn")
    pytest.importorskip("lightgbm")
    from src.models.lgbm_trainer import evaluate_predictions, run_baseline_comparison

    df = make_model_df()
    train_df = df[df["date"] < "2025-01-05"]
    valid_df = df[df["date"] == "2025-01-05"]
    test_df = df[df["date"] > "2025-01-05"]
    feature_cols = ["feature_a", "feature_b", "dist_to_limit", "order_imbalance"]

    metrics = evaluate_predictions(test_df["label"], np.linspace(0, 1, len(test_df)), eval_df=test_df)
    assert "pr_auc" in metrics
    assert "recall_at_10" in metrics
    assert "daily_precision_at_10" in metrics

    baseline_df = run_baseline_comparison(train_df, valid_df, test_df, feature_cols)
    assert {"rule", "logistic_regression", "lightgbm"}.issubset(set(baseline_df["model"]))
    assert "daily_recall_at_50" in baseline_df.columns


def test_optuna_tuning_smoke():
    pytest.importorskip("sklearn")
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")
    from src.models.lgbm_trainer import optimize_lgbm_params

    df = make_model_df()
    train_df = df[df["date"] < "2025-01-05"]
    valid_df = df[df["date"] == "2025-01-05"]
    feature_cols = ["feature_a", "feature_b", "dist_to_limit", "order_imbalance"]

    params = optimize_lgbm_params(train_df, valid_df, feature_cols, n_trials=2)
    assert "num_leaves" in params
    assert "learning_rate" in params
