"""Canonical CLI entrypoint for the limit-up prediction project."""

from __future__ import annotations

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def split_dataset_by_trading_day(df, train_ratio=0.70, val_ratio=0.15):
    dates = sorted(df["date"].astype(str).unique())
    train_end = int(len(dates) * train_ratio)
    val_end = int(len(dates) * (train_ratio + val_ratio))
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    return (
        df[df["date"].astype(str).isin(train_dates)].copy(),
        df[df["date"].astype(str).isin(val_dates)].copy(),
        df[df["date"].astype(str).isin(test_dates)].copy(),
    )


def save_split_datasets(train_df, val_df, test_df, split_dir):
    from src.data_processing.csv_utils import write_csv

    os.makedirs(split_dir, exist_ok=True)
    write_csv(train_df, os.path.join(split_dir, "train.csv"))
    write_csv(val_df, os.path.join(split_dir, "validation.csv"))
    write_csv(test_df, os.path.join(split_dir, "test.csv"))


def validate_split(train_df, val_df, test_df):
    issues = []
    if train_df.empty:
        issues.append("training set is empty")
    if val_df.empty:
        issues.append("validation set is empty")
    if test_df.empty:
        issues.append("test set is empty")
    return {"passed": not issues, "issues": issues}


def run_build(args):
    from scripts.build_dataset import process_single_stock_csv, save_dataset

    if not (args.input and args.code and args.date):
        print("build mode requires --input, --code, and --date")
        return 1
    df = process_single_stock_csv(args.input, args.code, args.date, args.preclose)
    if df.empty:
        print("build failed: empty dataset")
        return 1
    save_dataset(df, args.output)
    return 0


def run_extract(args):
    from scripts.extract_7z import SevenZipExtractor

    if not args.input:
        print("extract mode requires --input")
        return 1
    extractor = SevenZipExtractor(args.input)
    extract_dir = extractor.extract_all(args.output)
    print(f"extracted to: {extract_dir}")
    return 0


def run_split(args):
    from scripts.build_dataset import load_dataset

    if not args.input:
        print("split mode requires --input")
        return 1
    df = load_dataset(args.input)
    if df.empty:
        print("dataset is empty")
        return 1
    if "date" not in df.columns:
        print("dataset must contain a date column")
        return 1
    train_df, val_df, test_df = split_dataset_by_trading_day(df, args.train_ratio, args.valid_ratio)
    validation_result = validate_split(train_df, val_df, test_df)
    if not validation_result["passed"]:
        print("split validation warnings:")
        for issue in validation_result["issues"]:
            print(f"  - {issue}")
    save_split_datasets(train_df, val_df, test_df, args.split_dir)
    print(f"split datasets saved to: {args.split_dir}")
    return 0


def _load_train_valid_test(args):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from scripts.build_dataset import load_dataset
    from src.models.lgbm_trainer import split_dataset_by_date

    if args.input:
        print(f"loading dataset: {args.input}")
        full_df = load_dataset(args.input)
        if full_df.empty:
            raise ValueError("dataset is empty")
        if "date" in full_df.columns and full_df["date"].nunique() > 1:
            train_df, valid_df, test_df = split_dataset_by_date(full_df, args.train_ratio, args.valid_ratio)
        else:
            train_df, temp_df = train_test_split(full_df, test_size=(1 - args.train_ratio), random_state=42)
            valid_ratio_adjusted = args.valid_ratio / (1 - args.train_ratio)
            valid_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - valid_ratio_adjusted),
                random_state=42,
            )
        return full_df, train_df, valid_df, test_df

    train_path = os.path.join(args.split_dir, "train.csv")
    valid_path = os.path.join(args.split_dir, "validation.csv")
    test_path = os.path.join(args.split_dir, "test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"training file not found: {train_path}")

    train_df = load_dataset(train_path)
    valid_df = load_dataset(valid_path) if os.path.exists(valid_path) else train_df.sample(frac=0.15, random_state=42)
    test_df = load_dataset(test_path) if os.path.exists(test_path) else train_df.sample(frac=0.15, random_state=43)
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    return full_df, train_df, valid_df, test_df


def run_train(args):
    import joblib

    from src.data_processing.csv_utils import get_feature_columns, write_csv
    from src.models.lgbm_trainer import (
        evaluate_model,
        get_feature_importance,
        optimize_lgbm_params,
        prepare_training_data,
        run_baseline_comparison,
        train_lgbm_classifier,
        walk_forward_validation,
    )

    try:
        full_df, train_df, valid_df, test_df = _load_train_valid_test(args)
    except Exception as exc:
        print(f"failed to load training data: {exc}")
        return 1

    feature_cols = get_feature_columns(train_df)
    if not feature_cols:
        print("no feature columns found")
        return 1

    tuned_params = None
    if args.tune:
        print(f"Optuna tuning: n_trials={args.n_trials}, primary_metric={args.primary_metric}")
        tuned_params = optimize_lgbm_params(
            train_df,
            valid_df,
            feature_cols,
            primary_metric=args.primary_metric,
            n_trials=args.n_trials,
        )
        print(f"best params: {tuned_params}")

    if args.validation_mode in {"walk-forward", "rolling"}:
        if "date" not in full_df.columns:
            print("walk-forward validation requires a date column")
            return 1
        wf_results = walk_forward_validation(
            full_df,
            feature_cols,
            min_train_periods=args.min_train_periods,
            primary_metric=args.primary_metric,
            calibration=args.calibration,
            params=tuned_params,
        )
        if wf_results.empty:
            print("walk-forward validation produced no valid folds")
            return 1
        output_dir = os.path.dirname(args.model_path) or "models"
        os.makedirs(output_dir, exist_ok=True)
        wf_path = os.path.join(output_dir, "walk_forward_results.csv")
        write_csv(wf_results, wf_path)
        print(f"walk-forward results saved to: {wf_path}")
        display_cols = [
            col for col in ["fold", "val_date", "pr_auc", "daily_precision_at_10", "daily_recall_at_10"]
            if col in wf_results.columns
        ]
        print(wf_results[display_cols].to_string(index=False))
        return 0

    print(f"train samples: {len(train_df):,}")
    print(f"valid samples: {len(valid_df):,}")
    print(f"test samples: {len(test_df):,}")
    print(f"features: {len(feature_cols)}")

    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_valid, y_valid = prepare_training_data(valid_df, feature_cols)
    X_test, y_test = prepare_training_data(test_df, feature_cols)

    model = train_lgbm_classifier(
        X_train,
        y_train,
        X_valid,
        y_valid,
        params=tuned_params,
        calibration=args.calibration,
    )
    metrics = evaluate_model(model, X_test, y_test, eval_df=test_df.loc[X_test.index])

    if args.run_baselines:
        baseline_df = run_baseline_comparison(train_df, valid_df, test_df, feature_cols, lgbm_params=tuned_params)
        baseline_path = args.model_path.replace(".pkl", "_baselines.csv")
        write_csv(baseline_df, baseline_path)
        print(f"baseline comparison saved to: {baseline_path}")
        display_cols = [
            col for col in [
                "model", "pr_auc", "precision_at_10", "recall_at_10",
                "daily_precision_at_10", "daily_recall_at_10",
            ]
            if col in baseline_df.columns
        ]
        print(baseline_df[display_cols].to_string(index=False))

    print("\nmodel metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: {metric_value}")

    model_path = args.model_path
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"model saved to: {model_path}")

    importance_df = get_feature_importance(model, feature_cols)
    importance_path = model_path.replace(".pkl", "_feature_importance.csv")
    write_csv(importance_df, importance_path)
    print(f"feature importance saved to: {importance_path}")
    return 0


def run_predict(args):
    import joblib
    import numpy as np

    from scripts.build_dataset import load_dataset
    from src.data_processing.csv_utils import get_feature_columns, write_csv

    if not args.input:
        print("predict mode requires --input")
        return 1
    if not os.path.exists(args.model_path):
        print(f"model file does not exist: {args.model_path}")
        return 1
    model = joblib.load(args.model_path)
    df = load_dataset(args.input)
    if df.empty:
        print("dataset is empty")
        return 1
    feature_cols = get_feature_columns(df)
    if not feature_cols:
        print("no feature columns found")
        return 1
    X = df[feature_cols]
    valid_mask = X.notna().all(axis=1)
    probabilities = np.full(len(df), np.nan, dtype=float)
    if valid_mask.any():
        probabilities[valid_mask.to_numpy()] = model.predict_proba(X.loc[valid_mask])[:, 1]
    df["limit_up_probability"] = probabilities
    df["prediction"] = (df["limit_up_probability"] >= args.threshold).astype(int)
    output_path = args.output
    if output_path == "data/processed/dataset.csv":
        output_path = args.input.replace(".csv", "_predictions.csv")
    write_csv(df, output_path)
    print(f"predictions saved to: {output_path}")
    return 0


def run_check(args):
    from scripts.build_dataset import load_dataset
    from src.data_processing.quality_check import print_quality_report, run_quality_checks

    if not args.input:
        print("check mode requires --input")
        return 1
    df = load_dataset(args.input)
    if df.empty:
        print("dataset is empty")
        return 1
    tick_required_features = [
        "dist_to_limit", "ticks_to_limit", "ask1_to_limit", "ask1_gap",
        "bid_depth", "ask_depth", "order_imbalance", "b1_volume", "a1_volume",
        "spread", "ask_slope", "bid_slope", "ret_1tick", "vol_delta", "money_delta",
        "weighted_bid_pressure", "weighted_ask_pressure", "bid_ask_depth_ratio",
        "spread_pct", "ticks_to_limit_current",
    ]
    window_required_features = [
        "dist_to_limit_last", "order_imbalance_last", "b1_volume_last",
        "a1_volume_last", "money_delta_last", "weighted_bid_pressure_last",
        "weighted_ask_pressure_last", "bid_ask_depth_ratio_last",
        "spread_pct_last", "ticks_to_limit_current_last",
        "dist_to_limit_T1", "order_imbalance_T1",
        "dist_to_limit_slope", "money_delta_accel",
        "weighted_bid_pressure_trend", "spread_pct_trend",
    ]
    required_features = window_required_features if "dist_to_limit_last" in df.columns else tick_required_features
    quality_results = run_quality_checks(df, required_features)
    if "dist_to_limit_last" in df.columns and "code" in df.columns and "time" in df.columns:
        for result in quality_results:
            if result["check_name"] == "time_monotonic" and not result["passed"]:
                grouped_monotonic = df.groupby("code", sort=False)["time"].apply(lambda s: s.is_monotonic_increasing)
                if bool(grouped_monotonic.all()):
                    result["passed"] = True
                    result["message"] = "time is monotonic within each code"
                    result["details"] = {
                        "is_monotonic": True,
                        "grouped_by": "code",
                        "groups": int(grouped_monotonic.size),
                    }
    print_quality_report(quality_results)
    return 0 if all(result["passed"] for result in quality_results) else 1


def run_stock_info(args):
    from src.data_processing.stock_utils import get_stock_info

    if not args.code:
        print("stock-info mode requires --code")
        return 1
    stock_info = get_stock_info(args.code)
    print(f"stock_code: {stock_info['stock_code']}")
    print(f"stock_type: {stock_info['stock_type'].upper()}")
    print(f"limit_ratio: {stock_info['limit_ratio']:.0%}")
    print(f"is_st: {stock_info['is_st']}")
    return 0


def run_full(args):
    import build_2025_dataset

    output_path = args.output
    if output_path == "data/processed/dataset.csv":
        output_path = "data/processed/2025_full_dataset.csv"
    success = build_2025_dataset.build_full_year_dataset(
        year="2025",
        output_path=output_path,
        max_files=args.max_files,
        sample_files=args.sample,
        split_dataset=True,
        train_ratio=args.train_ratio,
        val_ratio=args.valid_ratio,
        max_months=args.max_months,
    )
    return 0 if success else 1


def run_merge(args):
    from scripts.training_set_builder import merge_training_shards

    success = merge_training_shards(args.shards_dir, args.merged_output)
    return 0 if success else 1


def run_rolling_cv(args):
    from scripts.rolling_cv import run_rolling_cv

    result = run_rolling_cv()
    return 0 if result is not None else 1


def build_parser():
    parser = argparse.ArgumentParser(description="Limit-up prediction system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "build", "extract", "train", "predict", "check", "split",
            "stock-info", "full", "merge", "rolling-cv",
        ],
        default="build",
        help="Run mode",
    )
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--input-dir", type=str, help="Input directory")
    parser.add_argument("--output", type=str, default="data/processed/dataset.csv", help="Output CSV path")
    parser.add_argument("--date", type=str, help="Trading date")
    parser.add_argument("--code", type=str, help="Stock code")
    parser.add_argument("--preclose", type=float, default=10.0, help="Previous close")
    parser.add_argument("--model-path", type=str, default="models/lgbm_model.pkl", help="Model save/load path")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--threshold", type=float, default=0.70, help="Prediction threshold")
    parser.add_argument("--split-dir", type=str, default="data/processed/split_datasets", help="Split dataset directory")
    parser.add_argument("--format", type=str, choices=["csv"], default="csv", help="Dataset format")
    parser.add_argument("--max-files", type=int, help="Maximum files to process")
    parser.add_argument("--sample", type=int, help="Sampling interval")
    parser.add_argument("--max-months", type=int, default=3, help="Maximum months")
    parser.add_argument("--shards-dir", type=str, default="data/daily_train_undersampled", help="Training shard directory")
    parser.add_argument("--merged-output", type=str, default="data/merged/multi_day_train.csv", help="Merged CSV output")
    parser.add_argument("--validation-mode", choices=["single", "walk-forward", "rolling"], default="single")
    parser.add_argument("--min-train-periods", type=int, default=3, help="Minimum dates before rolling validation starts")
    parser.add_argument(
        "--primary-metric",
        choices=["pr_auc", "recall_at_10", "recall_at_50", "daily_recall_at_10", "daily_recall_at_50"],
        default="pr_auc",
    )
    parser.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trial count")
    parser.add_argument("--run-baselines", action="store_true", help="Evaluate rule, logistic, and LightGBM baselines")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "build": run_build,
        "extract": run_extract,
        "train": run_train,
        "predict": run_predict,
        "check": run_check,
        "split": run_split,
        "stock-info": run_stock_info,
        "full": run_full,
        "merge": run_merge,
        "rolling-cv": run_rolling_cv,
    }
    return handlers[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
