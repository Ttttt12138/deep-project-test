"""
LightGBM模型训练模块
实现涨停预测的二分类模型训练
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)


def prepare_training_data(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备训练数据

    Args:
        df: 包含特征和标签的DataFrame
        feature_cols: 特征列名列表

    Returns:
        (特征DataFrame, 标签Series)
    """
    # 删除包含缺失值的行
    df = df.dropna(subset=feature_cols + ['label'])

    # 分离特征和标签
    X = df[feature_cols]
    y = df['label']

    return X, y


def split_dataset_by_date(df: pd.DataFrame,
                         train_ratio: float = 0.7,
                         valid_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按交易日划分数据集

    Args:
        df: 包含date列的DataFrame
        train_ratio: 训练集比例
        valid_ratio: 验证集比例

    Returns:
        (训练集, 验证集, 测试集)
    """
    # 获取所有日期并排序
    dates = sorted(df['date'].unique())
    n = len(dates)

    # 计算划分点
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    # 按日期划分
    train_dates = dates[:train_end]
    valid_dates = dates[train_end:valid_end]
    test_dates = dates[valid_end:]

    train_df = df[df['date'].isin(train_dates)].copy()
    valid_df = df[df['date'].isin(valid_dates)].copy()
    test_df = df[df['date'].isin(test_dates)].copy()

    return train_df, valid_df, test_df


def train_lgbm_classifier(X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_valid: pd.DataFrame,
                          y_valid: pd.Series,
                          class_weight: float = None) -> lgb.LGBMClassifier:
    """
    训练LightGBM分类器

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_valid: 验证特征
        y_valid: 验证标签
        class_weight: 类别权重（scale_pos_weight）

    Returns:
        训练好的模型
    """
    # 计算类别权重
    if class_weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        class_weight = neg_count / max(pos_count, 1)

    # 创建模型
    model = lgb.LGBMClassifier(
        objective='binary',
        metric=['auc', 'binary_logloss'],
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        n_estimators=1000,
        scale_pos_weight=class_weight,  # 使用scale_pos_weight而不是class_weight
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # 训练模型
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
    except ValueError as e:
        # 处理训练集或验证集缺少某一类别的情况
        if "previously unseen labels" in str(e):
            print(f"警告: {e}")
            print("使用默认参数重新训练...")
            model.fit(
                X_train, y_train,
                callbacks=[
                    lgb.log_evaluation(period=100)
                ]
            )
        else:
            raise

    return model


def evaluate_model(model: lgb.LGBMClassifier,
                  X_test: pd.DataFrame,
                  y_test: pd.Series) -> Dict:
    """
    评估模型性能

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        评估指标字典
    """
    # 预测概率
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 预测类别
    y_pred = model.predict(X_test)

    # 计算基础指标
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }

    # 计算Top-K命中率
    top_k_metrics = calculate_top_k_metrics(y_test, y_pred_proba, k_values=[1, 5, 10, 50])
    metrics.update(top_k_metrics)

    # 计算概率阈值指标
    threshold_metrics = calculate_threshold_metrics(y_test, y_pred_proba, threshold=0.7)
    metrics.update(threshold_metrics)

    return metrics


def calculate_top_k_metrics(y_true: pd.Series,
                            y_pred_proba: np.ndarray,
                            k_values: List[int]) -> Dict:
    """
    计算Top-K命中率

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        k_values: K值列表

    Returns:
        Top-K指标字典
    """
    metrics = {}

    # 按预测概率降序排序
    sorted_indices = np.argsort(y_pred_proba)[::-1]

    for k in k_values:
        if k <= len(y_true):
            top_k_indices = sorted_indices[:k]
            top_k_hits = y_true.iloc[top_k_indices].sum()
            metrics[f'top_{k}_hit_rate'] = top_k_hits / k
        else:
            metrics[f'top_{k}_hit_rate'] = 0.0

    return metrics


def calculate_threshold_metrics(y_true: pd.Series,
                               y_pred_proba: np.ndarray,
                               threshold: float) -> Dict:
    """
    计算指定阈值下的指标

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        threshold: 概率阈值

    Returns:
        阈值指标字典
    """
    # 根据阈值预测
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        f'precision@threshold_{threshold}': precision_score(y_true, y_pred, zero_division=0),
        f'recall@threshold_{threshold}': recall_score(y_true, y_pred, zero_division=0),
        f'high_prob_coverage': (y_pred_proba >= threshold).sum() / len(y_pred_proba)
    }

    return metrics


def get_feature_importance(model: lgb.LGBMClassifier, feature_names: List[str]) -> pd.DataFrame:
    """
    获取特征重要性

    Args:
        model: 训练好的模型
        feature_names: 特征名称列表

    Returns:
        特征重要性DataFrame
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df


def predict_limit_up_probability(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    预测涨停概率

    Args:
        model: 训练好的模型
        X: 特征数据

    Returns:
        涨停概率数组
    """
    return model.predict_proba(X)[:, 1]