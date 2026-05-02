"""Evaluate a saved model on the train, validation, and test splits."""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from main import _load_train_valid_test
from src.data_processing.csv_utils import get_feature_columns, write_csv
from src.models.lgbm_trainer import evaluate_model, prepare_training_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved model on train/valid/test splits")
    parser.add_argument("--input", required=True, help="Full dataset CSV path")
    parser.add_argument("--model-path", required=True, help="Saved model .pkl path")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--split-dir", default="data/processed/split_datasets")
    return parser.parse_args()


def compact_metrics(split: str, df: pd.DataFrame, feature_cols: list[str], model) -> dict:
    X, y = prepare_training_data(df, feature_cols)
    metrics = evaluate_model(model, X, y, eval_df=df.loc[X.index])
    return {
        "split": split,
        "samples": len(X),
        "positives": int(y.sum()),
        "pos_rate": float(y.mean()),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "precision_at_10": metrics.get("precision_at_10"),
        "recall_at_10": metrics.get("recall_at_10"),
        "precision_at_50": metrics.get("precision_at_50"),
        "recall_at_50": metrics.get("recall_at_50"),
        "daily_precision_at_10": metrics.get("daily_precision_at_10"),
        "daily_recall_at_10": metrics.get("daily_recall_at_10"),
        "daily_precision_at_50": metrics.get("daily_precision_at_50"),
        "daily_recall_at_50": metrics.get("daily_recall_at_50"),
        "precision@threshold_0.7": metrics.get("precision@threshold_0.7"),
        "recall@threshold_0.7": metrics.get("recall@threshold_0.7"),
        "high_prob_coverage": metrics.get("high_prob_coverage"),
    }


def main() -> int:
    args = parse_args()
    model = joblib.load(args.model_path)
    _, train_df, valid_df, test_df = _load_train_valid_test(args)
    feature_cols = get_feature_columns(train_df)

    rows = [
        compact_metrics("train", train_df, feature_cols, model),
        compact_metrics("valid", valid_df, feature_cols, model),
        compact_metrics("test", test_df, feature_cols, model),
    ]
    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    if args.output:
        write_csv(result, args.output)
        print(f"\nsaved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
