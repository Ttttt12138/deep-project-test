"""
滚动验证 - Expanding Window
训练集：欠采样数据（undersampled，1:5）
验证集：原始真实分布（candidates，0.03%）
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_processing.csv_utils import get_feature_columns, read_csv, write_csv
from src.models.lgbm_trainer import evaluate_predictions


# ================================================================
# 配置
# ================================================================
CANDIDATES_DIR = os.path.join(PROJECT_ROOT, "data/daily_train_candidates")
UNDERSAMPLED_DIR = os.path.join(PROJECT_ROOT, "data/daily_train_undersampled")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/rolling_cv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models/rolling_cv")
MIN_TRAIN_MONTHS = 3
DECAY_RATE = 0.85   # 每早一个月权重降低15%，可调范围 0.7~0.95

LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 500,
    "verbose": -1,
}

# ================================================================
# 数据加载
# ================================================================
def load_undersampled_by_month(directory: str) -> dict:
    """
    加载欠采样数据（量小，可以整月加载）
    返回：{"2025-01": df, "2025-02": df, ...}
    """
    monthly = {}
    for month_dir in sorted(glob.glob(os.path.join(directory, "*"))):
        if not os.path.isdir(month_dir):
            continue

        month_str = os.path.basename(month_dir)
        year_month = f"2025-{month_str}"

        files = sorted(glob.glob(os.path.join(month_dir, "*_train.csv")))
        dfs = []
        for f in files:
            df = read_csv(f)
            stem = Path(f).stem
            date_part = stem[:10]
            df["date"] = date_part
            dfs.append(df)

        if dfs:
            monthly[year_month] = pd.concat(dfs, ignore_index=True)

    return monthly


def get_feature_cols(df: pd.DataFrame) -> list:
    return get_feature_columns(df)


def load_candidates_for_month(month: str) -> pd.DataFrame:
    """
    加载某月的 candidates 数据（流式处理，避免内存爆炸）
    """
    month_dir = os.path.join(CANDIDATES_DIR, month[5:])  # "2025-01" -> "01"
    if not os.path.isdir(month_dir):
        return pd.DataFrame()

    files = sorted(glob.glob(os.path.join(month_dir, "*_candidate.csv")))

    dfs = []
    for f in files:
        df = read_csv(f)
        stem = Path(f).stem
        date_part = stem[:10]
        df["date"] = date_part
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# ================================================================
# 指标计算
# ================================================================
def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, eval_df: pd.DataFrame = None) -> dict:
    """
    在真实分布下计算指标。
    """
    if y_true.sum() == 0:
        return {k: None for k in
                ["roc_auc", "pr_auc", "p_at_10", "p_at_50", "p_at_100", "p_at_500", "recall_at_10", "recall_at_50", "recall_at_100"]}

    metrics = evaluate_predictions(y_true, y_proba, eval_df=eval_df, k_values=[10, 50, 100, 500])
    metrics["p_at_10"] = metrics.get("precision_at_10")
    metrics["p_at_50"] = metrics.get("precision_at_50")
    metrics["p_at_100"] = metrics.get("precision_at_100")
    metrics["p_at_500"] = metrics.get("precision_at_500")
    metrics["recall_at_10"] = metrics.get("recall_at_10")
    metrics["recall_at_50"] = metrics.get("recall_at_50")
    metrics["recall_at_100"] = metrics.get("recall_at_100")
    return metrics


# ================================================================
# 主流程
# ================================================================
def run_rolling_cv():
    import joblib
    import lightgbm as lgb

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # ---------- 1. 加载训练数据（欠采样，量小） ----------
    print("加载训练数据（欠采样）...")
    train_monthly = load_undersampled_by_month(UNDERSAMPLED_DIR)
    print(f"  训练数据月份: {list(train_monthly.keys())}")

    # 确定可用月份
    common_months = sorted(train_monthly.keys())
    if not common_months:
        raise ValueError("未找到训练月份，请检查 data/daily_train_undersampled/01..12")

    print(f"可用月份: {common_months[0]} ~ {common_months[-1]} ({len(common_months)} 个月)")

    if len(common_months) <= MIN_TRAIN_MONTHS:
        raise ValueError(f"月份数不足，需要 > {MIN_TRAIN_MONTHS} 个月")

    # 特征列
    sample_df = next(iter(train_monthly.values()))
    feature_cols = get_feature_cols(sample_df)
    print(f"特征数: {len(feature_cols)}")

    # 打印真实分布信息
    first_train = next(iter(train_monthly.values()))
    train_pos_rate = first_train["label"].mean()
    print(f"训练集正样本占比（欠采样后）: {train_pos_rate:.1%}\n")

    # ---------- 2. 生成 fold ----------
    folds = [
        (common_months[:MIN_TRAIN_MONTHS + i], common_months[MIN_TRAIN_MONTHS + i])
        for i in range(len(common_months) - MIN_TRAIN_MONTHS)
    ]
    print(f"共 {len(folds)} 个 fold\n")

    # ---------- 3. 滚动训练 ----------
    all_results = []

    for fold_idx, (train_months, val_month) in enumerate(folds):
        print("=" * 65)
        print(f"Fold {fold_idx+1}/{len(folds)}  "
              f"训练: {train_months[0]}~{train_months[-1]}  "
              f"验证: {val_month}")

        # 合并训练月份（欠采样数据）
        train_df = pd.concat(
            [train_monthly[m] for m in train_months],
            ignore_index=True
        )

        # 用训练集最后一个月做 early stopping
        es_month = train_months[-1]
        pure_train_months = train_months[:-1]

        if len(pure_train_months) == 0:
            # 训练月只有1个时降级处理，直接用训练集
            pure_train_df = train_df
            es_df = train_df
            print(f"  训练集: {len(train_df):,} 样本  正样本: {int(train_df['label'].sum())} ({train_df['label'].mean():.1%})")
            print(f"  (训练月仅1个，使用训练集自身做early stopping)")
        else:
            pure_train_df = pd.concat(
                [train_monthly[m] for m in pure_train_months], ignore_index=True
            )
            es_df = train_monthly[es_month]
            print(f"  纯训练集: {len(pure_train_df):,} 样本  正样本: {int(pure_train_df['label'].sum())} ({pure_train_df['label'].mean():.1%})")
            print(f"  EarlyStopping集: {es_month} ({len(es_df):,} 样本, 正样本: {int(es_df['label'].sum())})")

        y_train = pure_train_df["label"].values
        X_train = pure_train_df[feature_cols].values

        # 时间衰减加权
        def make_sample_weight(dates: pd.Series, decay: float = DECAY_RATE) -> np.ndarray:
            months = dates.str[:7]
            unique_months = sorted(months.unique())
            month_rank = {m: i for i, m in enumerate(unique_months)}
            max_rank = len(unique_months) - 1
            weights = months.map(lambda m: decay ** (max_rank - month_rank[m]))
            return weights.values

        pure_weight = make_sample_weight(pure_train_df["date"])
        print(f"  样本权重范围: {pure_weight.min():.3f} ~ {pure_weight.max():.3f}")

        if y_train.sum() == 0:
            print("  跳过：训练集正样本为0")
            continue

        # 动态正负样本权重
        scale_pos_weight = round((y_train == 0).sum() / y_train.sum(), 2)
        params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}
        print(f"  scale_pos_weight: {scale_pos_weight}")

        # 训练
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=pure_weight,
            eval_set=[(es_df[feature_cols].values, es_df["label"].values)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        # 加载验证集（candidates，真实分布）并评估
        print(f"  加载验证集（真实分布）...")
        val_df = load_candidates_for_month(val_month)

        if val_df.empty or val_df["label"].sum() == 0:
            print("  跳过：验证集为空或无正样本")
            continue

        y_val = val_df["label"].values
        X_val = val_df[feature_cols].values

        print(f"  验证集: {len(val_df):,} 样本  正样本: {int(y_val.sum())} ({y_val.mean():.4%})")

        # 评估
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_proba, eval_df=val_df)

        print(f"\n  ── 真实分布下的指标 ──")
        print(f"  ROC-AUC:       {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:        {metrics['pr_auc']:.4f}  ← 重点关注")
        print(f"  Precision@100: {metrics['p_at_100']:.4f}  (Top100预测中命中率)")
        print(f"  Precision@500: {metrics['p_at_500']:.4f}  (Top500预测中命中率)")
        print(f"  Recall@100:    {metrics['recall_at_100']:.4f}  (Top100能召回多少正样本)")

        # 保存模型
        model_path = f"{MODEL_DIR}/fold{fold_idx+1}_{val_month}.pkl"
        joblib.dump(model, model_path)

        all_results.append({
            "fold": fold_idx + 1,
            "train_start": train_months[0],
            "train_end": train_months[-1],
            "train_months": len(train_months),
            "val_month": val_month,
            "train_samples": len(train_df),
            "train_pos": int(y_train.sum()),
            "val_samples": len(val_df),
            "val_pos": int(y_val.sum()),
            "model_path": model_path,
            **metrics,
        })

    # ---------- 4. 汇总 ----------
    if not all_results:
        print("\n所有fold均跳过，请检查数据")
        return None

    results_df = pd.DataFrame(all_results)
    results_path = f"{OUTPUT_DIR}/results.csv"
    write_csv(results_df, results_path)

    print("\n" + "=" * 65)
    print("滚动验证完成 — 真实分布下的指标汇总")
    print("=" * 65)
    cols = ["val_month", "train_months", "train_pos", "val_pos",
            "roc_auc", "pr_auc", "p_at_100", "p_at_500"]
    print(results_df[cols].to_string(index=False))

    valid = results_df.dropna(subset=["roc_auc"])
    print(f"\nROC-AUC  均值: {valid['roc_auc'].mean():.4f} ± {valid['roc_auc'].std():.4f}")
    print(f"PR-AUC   均值: {valid['pr_auc'].mean():.4f} ± {valid['pr_auc'].std():.4f}")
    print(f"P@100    均值: {valid['p_at_100'].mean():.4f} ± {valid['p_at_100'].std():.4f}")
    print(f"\n结果已保存至: {results_path}")

    return results_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="滚动验证 - CSV workflow")
    parser.add_argument("--run", action="store_true", help="保留兼容参数；脚本默认执行滚动验证")
    parser.parse_args()
    run_rolling_cv()


if __name__ == "__main__":
    main()
