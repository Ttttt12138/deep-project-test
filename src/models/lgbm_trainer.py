"""Model training, evaluation, calibration, and baseline utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - exercised in environments without ML deps
    lgb = None


DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


@dataclass
class TrainedModel:
    estimator: object
    base_estimator: object
    calibration: str = "none"

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def prepare_training_data(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["label"])
    X = work[feature_cols]
    y = work["label"].astype(int)
    return X, y


def split_dataset_by_date(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].astype(str).unique())
    n_dates = len(dates)
    train_end = int(n_dates * train_ratio)
    valid_end = int(n_dates * (train_ratio + valid_ratio))
    train_dates = dates[:train_end]
    valid_dates = dates[train_end:valid_end]
    test_dates = dates[valid_end:]
    return (
        df[df["date"].astype(str).isin(train_dates)].copy(),
        df[df["date"].astype(str).isin(valid_dates)].copy(),
        df[df["date"].astype(str).isin(test_dates)].copy(),
    )


def _scale_pos_weight(y: Iterable[int]) -> float:
    y_arr = np.asarray(y)
    pos_count = max(int((y_arr == 1).sum()), 1)
    neg_count = int((y_arr == 0).sum())
    return neg_count / pos_count


def _require_lightgbm():
    if lgb is None:
        raise ImportError("LightGBM is required for model training. Install lightgbm>=4.0.0.")
    return lgb


def _build_lgbm(params: Optional[Dict] = None, scale_pos_weight: Optional[float] = None):
    lightgbm = _require_lightgbm()
    merged = dict(DEFAULT_LGBM_PARAMS)
    if params:
        translated = dict(params)
        if "feature_fraction" in translated:
            translated["colsample_bytree"] = translated.pop("feature_fraction")
        if "bagging_fraction" in translated:
            translated["subsample"] = translated.pop("bagging_fraction")
        if "lambda_l1" in translated:
            translated["reg_alpha"] = translated.pop("lambda_l1")
        if "lambda_l2" in translated:
            translated["reg_lambda"] = translated.pop("lambda_l2")
        merged.update(translated)
    if scale_pos_weight is not None:
        merged["scale_pos_weight"] = scale_pos_weight
    return lightgbm.LGBMClassifier(**merged)


def _can_calibrate(y_valid: pd.Series, calibration: str) -> bool:
    return calibration in {"sigmoid", "isotonic"} and len(np.unique(y_valid)) == 2 and len(y_valid) >= 10


def train_lgbm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    class_weight: float = None,
    params: Optional[Dict] = None,
    calibration: str = "none",
) -> object:
    if class_weight is None:
        class_weight = _scale_pos_weight(y_train)
    lightgbm = _require_lightgbm()
    model = _build_lgbm(params=params, scale_pos_weight=class_weight)

    fit_kwargs = {}
    if len(X_valid) > 0 and len(np.unique(y_valid)) >= 1:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["callbacks"] = [
            lightgbm.early_stopping(stopping_rounds=50, verbose=False),
            lightgbm.log_evaluation(period=100),
        ]
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except ValueError:
        model.fit(X_train, y_train)

    if _can_calibrate(y_valid, calibration):
        try:
            try:
                from sklearn.frozen import FrozenEstimator

                calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=calibration)
            except ImportError:
                calibrated = CalibratedClassifierCV(model, method=calibration, cv="prefit")
            calibrated.fit(X_valid, y_valid)
            return TrainedModel(estimator=calibrated, base_estimator=model, calibration=calibration)
        except Exception as exc:
            print(f"calibration skipped: {exc}")

    return model


def _positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return np.zeros(len(X), dtype=float)
    return proba[:, 1]


def _safe_metric(metric_fn, y_true, y_score, default=np.nan):
    try:
        if len(np.unique(y_true)) < 2:
            return default
        return float(metric_fn(y_true, y_score))
    except Exception:
        return default


def calculate_top_k_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred_proba: np.ndarray,
    k_values: List[int],
) -> Dict[str, float]:
    y_arr = np.asarray(y_true).astype(int)
    if len(y_arr) == 0:
        return {}
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    total_pos = y_arr.sum()
    metrics: Dict[str, float] = {}
    for k in k_values:
        actual_k = min(k, len(y_arr))
        if actual_k <= 0:
            precision = 0.0
            recall = 0.0
            hits = 0
        else:
            top_indices = sorted_indices[:actual_k]
            hits = int(y_arr[top_indices].sum())
            precision = hits / actual_k
            recall = hits / total_pos if total_pos > 0 else 0.0
        metrics[f"top_{k}_hit_rate"] = precision
        metrics[f"precision_at_{k}"] = precision
        metrics[f"recall_at_{k}"] = recall
        metrics[f"hits_at_{k}"] = hits
    return metrics


def calculate_daily_top_k_metrics(
    df: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_pred_proba: np.ndarray,
    k_values: List[int],
) -> Dict[str, float]:
    if "date" not in df.columns or len(df) == 0:
        return {}
    work = pd.DataFrame({
        "date": df["date"].astype(str).to_numpy(),
        "label": np.asarray(y_true).astype(int),
        "proba": y_pred_proba,
    })
    rows = []
    for _, group in work.groupby("date", sort=True):
        daily = calculate_top_k_metrics(group["label"].to_numpy(), group["proba"].to_numpy(), k_values)
        daily["daily_pos"] = int(group["label"].sum())
        rows.append(daily)
    if not rows:
        return {}
    daily_df = pd.DataFrame(rows)
    metrics: Dict[str, float] = {}
    for k in k_values:
        for source, target in [
            (f"precision_at_{k}", f"daily_precision_at_{k}"),
            (f"recall_at_{k}", f"daily_recall_at_{k}"),
            (f"hits_at_{k}", f"daily_hits_at_{k}"),
        ]:
            if source in daily_df.columns:
                metrics[target] = float(daily_df[source].mean())
    return metrics


def calculate_threshold_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_arr = np.asarray(y_true).astype(int)
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        f"precision@threshold_{threshold}": precision_score(y_arr, y_pred, zero_division=0),
        f"recall@threshold_{threshold}": recall_score(y_arr, y_pred, zero_division=0),
        "high_prob_coverage": float((y_pred_proba >= threshold).sum() / max(len(y_pred_proba), 1)),
    }


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred_proba: np.ndarray,
    eval_df: Optional[pd.DataFrame] = None,
    threshold: float = 0.7,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    if k_values is None:
        k_values = [1, 5, 10, 50]
    y_arr = np.asarray(y_true).astype(int)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": _safe_metric(roc_auc_score, y_arr, y_pred_proba),
        "pr_auc": _safe_metric(average_precision_score, y_arr, y_pred_proba),
        "accuracy": accuracy_score(y_arr, y_pred),
        "precision": precision_score(y_arr, y_pred, zero_division=0),
        "recall": recall_score(y_arr, y_pred, zero_division=0),
        "f1_score": f1_score(y_arr, y_pred, zero_division=0),
        "total_pos": int(y_arr.sum()),
        "total_samples": int(len(y_arr)),
    }
    metrics.update(calculate_top_k_metrics(y_arr, y_pred_proba, k_values=k_values))
    metrics.update(calculate_threshold_metrics(y_arr, y_pred_proba, threshold=threshold))
    if eval_df is not None:
        metrics.update(calculate_daily_top_k_metrics(eval_df, y_arr, y_pred_proba, k_values=[10, 50]))
    return metrics


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    eval_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    return evaluate_predictions(y_test, _positive_proba(model, X_test), eval_df=eval_df)


def _rule_baseline_scores(df: pd.DataFrame) -> np.ndarray:
    dist_col = "dist_to_limit_last" if "dist_to_limit_last" in df.columns else "dist_to_limit"
    oi_col = "order_imbalance_last" if "order_imbalance_last" in df.columns else "order_imbalance"
    dist = pd.to_numeric(df.get(dist_col, 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    imbalance = pd.to_numeric(df.get(oi_col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rule_hit = (dist <= 0.01) & (imbalance >= 0.2)
    raw = (-dist) + imbalance + rule_hit.astype(float)
    if raw.max() == raw.min():
        return rule_hit.astype(float)
    return (raw - raw.min()) / (raw.max() - raw.min())


def run_baseline_comparison(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    lgbm_params: Optional[Dict] = None,
) -> pd.DataFrame:
    rows = []
    y_test = test_df["label"].astype(int)

    rule_scores = _rule_baseline_scores(test_df)
    rows.append({"model": "rule", **evaluate_predictions(y_test, rule_scores, eval_df=test_df)})

    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_valid, y_valid = prepare_training_data(valid_df, feature_cols)
    X_test, y_test_prepared = prepare_training_data(test_df, feature_cols)
    eval_df = test_df.loc[X_test.index]

    if len(np.unique(y_train)) == 2:
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None),
        )
        logistic.fit(X_train, y_train)
        rows.append({
            "model": "logistic_regression",
            **evaluate_predictions(y_test_prepared, _positive_proba(logistic, X_test), eval_df=eval_df),
        })

    lgbm_model = train_lgbm_classifier(
        X_train,
        y_train,
        X_valid,
        y_valid,
        params=lgbm_params,
        calibration="none",
    )
    rows.append({
        "model": "lightgbm",
        **evaluate_model(lgbm_model, X_test, y_test_prepared, eval_df=eval_df),
    })
    return pd.DataFrame(rows)


def optimize_lgbm_params(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    primary_metric: str = "pr_auc",
    n_trials: int = 30,
    random_state: int = 42,
) -> Dict:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for --tune. Install optuna>=3.0.0.") from exc

    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_valid, y_valid = prepare_training_data(valid_df, feature_cols)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                max(_scale_pos_weight(y_train) * 0.5, 0.1),
                max(_scale_pos_weight(y_train) * 2.0, 0.2),
            ),
            "n_estimators": 250,
            "random_state": random_state,
        }
        class_weight = params.pop("scale_pos_weight")
        model = train_lgbm_classifier(
            X_train,
            y_train,
            X_valid,
            y_valid,
            class_weight=class_weight,
            params=params,
            calibration="none",
        )
        metrics = evaluate_model(model, X_valid, y_valid, eval_df=valid_df.loc[X_valid.index])
        value = metrics.get(primary_metric, np.nan)
        if pd.isna(value):
            return -1.0
        return float(value)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = dict(study.best_params)
    return best_params


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_train_periods: int = 3,
    primary_metric: str = "pr_auc",
    calibration: str = "sigmoid",
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("walk-forward validation requires a date column")

    dates = sorted(df["date"].astype(str).unique())
    if len(dates) <= min_train_periods:
        raise ValueError(f"need more than {min_train_periods} dates for walk-forward validation")

    rows = []
    for fold_idx, val_date in enumerate(dates[min_train_periods:], start=1):
        train_dates = dates[: dates.index(val_date)]
        train_df = df[df["date"].astype(str).isin(train_dates)].copy()
        val_df = df[df["date"].astype(str) == val_date].copy()
        if train_df.empty or val_df.empty or train_df["label"].sum() == 0:
            continue

        es_date = train_dates[-1]
        pure_train_df = train_df[train_df["date"].astype(str) != es_date]
        es_df = train_df[train_df["date"].astype(str) == es_date]
        if pure_train_df.empty:
            pure_train_df = train_df
            es_df = train_df

        X_train, y_train = prepare_training_data(pure_train_df, feature_cols)
        X_es, y_es = prepare_training_data(es_df, feature_cols)
        X_val, y_val = prepare_training_data(val_df, feature_cols)
        if X_train.empty or X_val.empty or y_train.sum() == 0:
            continue
        model = train_lgbm_classifier(X_train, y_train, X_es, y_es, params=params, calibration=calibration)
        metrics = evaluate_model(model, X_val, y_val, eval_df=val_df.loc[X_val.index])
        rows.append({
            "fold": fold_idx,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "val_date": val_date,
            "primary_metric": primary_metric,
            "primary_value": metrics.get(primary_metric, np.nan),
            **metrics,
        })
    return pd.DataFrame(rows)


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    estimator = model.base_estimator if isinstance(model, TrainedModel) else model
    importance = getattr(estimator, "feature_importances_", np.zeros(len(feature_names)))
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )


def predict_limit_up_probability(model, X: pd.DataFrame) -> np.ndarray:
    return _positive_proba(model, X)
