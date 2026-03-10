"""
数据质量检查模块
提供数据集质量检查功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def check_uniqueness(df: pd.DataFrame, unique_keys: List[str] = None) -> Dict[str, any]:
    """
    检查数据唯一性

    Args:
        df: 待检查的DataFrame
        unique_keys: 需要检查唯一性的列名列表，默认使用所有列

    Returns:
        包含唯一性检查结果的字典
    """
    result = {
        'check_name': 'uniqueness',
        'passed': True,
        'details': {}
    }

    if unique_keys is None:
        unique_keys = df.columns.tolist()

    # 检查是否有重复行
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates(subset=unique_keys))
    duplicate_count = total_rows - unique_rows

    result['details'] = {
        'total_rows': total_rows,
        'unique_rows': unique_rows,
        'duplicate_count': duplicate_count,
        'duplicate_ratio': duplicate_count / total_rows if total_rows > 0 else 0
    }

    # 判断是否通过（允许少量重复，不超过1%）
    if duplicate_count > 0 and result['details']['duplicate_ratio'] > 0.01:
        result['passed'] = False
        result['message'] = f"发现 {duplicate_count} 条重复样本（{result['details']['duplicate_ratio']:.2%}）"
    elif duplicate_count > 0:
        result['message'] = f"发现 {duplicate_count} 条重复样本（可接受）"
    else:
        result['message'] = "无重复样本"

    return result


def check_null_values(df: pd.DataFrame, check_columns: List[str] = None) -> Dict[str, any]:
    """
    检查空值

    Args:
        df: 待检查的DataFrame
        check_columns: 需要检查的列名列表，默认检查所有列

    Returns:
        包含空值检查结果的字典
    """
    result = {
        'check_name': 'null_values',
        'passed': True,
        'details': {}
    }

    if check_columns is None:
        check_columns = df.columns.tolist()

    # 检查每列的空值
    null_counts = df[check_columns].isnull().sum()
    total_rows = len(df)

    null_info = {}
    columns_with_nulls = []

    for col in check_columns:
        null_count = null_counts[col]
        null_ratio = null_count / total_rows if total_rows > 0 else 0

        null_info[col] = {
            'null_count': null_count,
            'null_ratio': null_ratio
        }

        if null_count > 0:
            columns_with_nulls.append(col)

            # 空值比例超过5%则不通过
            if null_ratio > 0.05:
                result['passed'] = False

    result['details'] = {
        'total_rows': total_rows,
        'columns_with_nulls': columns_with_nulls,
        'null_info': null_info
    }

    if columns_with_nulls:
        result['message'] = f"发现 {len(columns_with_nulls)} 列包含空值"
    else:
        result['message'] = "无空值"

    return result


def check_positive_samples(df: pd.DataFrame, label_col: str = 'label') -> Dict[str, any]:
    """
    检查正样本是否存在

    Args:
        df: 待检查的DataFrame
        label_col: 标签列名

    Returns:
        包含正样本检查结果的字典
    """
    result = {
        'check_name': 'positive_samples',
        'passed': True,
        'details': {}
    }

    if label_col not in df.columns:
        result['passed'] = False
        result['message'] = f"标签列 '{label_col}' 不存在"
        result['details'] = {'label_exists': False}
        return result

    total_samples = len(df)
    positive_count = (df[label_col] == 1).sum()
    negative_count = (df[label_col] == 0).sum()
    positive_ratio = positive_count / total_samples if total_samples > 0 else 0

    result['details'] = {
        'total_samples': total_samples,
        'positive_count': int(positive_count),
        'negative_count': int(negative_count),
        'positive_ratio': positive_ratio
    }

    # 判断是否通过（正样本比例必须大于0）
    if positive_count == 0:
        result['passed'] = False
        result['message'] = "未发现正样本"
    else:
        result['message'] = f"正样本: {positive_count:,} ({positive_ratio:.4%})"

    return result


def check_time_monotonic(df: pd.DataFrame, time_col: str = 'time') -> Dict[str, any]:
    """
    检查时间单调性

    Args:
        df: 待检查的DataFrame
        time_col: 时间列名

    Returns:
        包含时间单调性检查结果的字典
    """
    result = {
        'check_name': 'time_monotonic',
        'passed': True,
        'details': {}
    }

    if time_col not in df.columns:
        result['passed'] = False
        result['message'] = f"时间列 '{time_col}' 不存在"
        result['details'] = {'time_exists': False}
        return result

    # 检查时间列是否单调递增
    is_monotonic = df[time_col].is_monotonic_increasing

    result['details'] = {
        'is_monotonic': is_monotonic,
        'total_rows': len(df)
    }

    if not is_monotonic:
        result['passed'] = False
        result['message'] = "时间列不单调递增"
    else:
        result['message'] = "时间列单调递增"

    return result


def check_feature_completeness(df: pd.DataFrame, required_features: List[str]) -> Dict[str, any]:
    """
    检查特征完整性

    Args:
        df: 待检查的DataFrame
        required_features: 必需的特征列表

    Returns:
        包含特征完整性检查结果的字典
    """
    result = {
        'check_name': 'feature_completeness',
        'passed': True,
        'details': {}
    }

    existing_features = set(df.columns.tolist())
    missing_features = [f for f in required_features if f not in existing_features]
    extra_features = [f for f in existing_features if f not in required_features]

    result['details'] = {
        'required_count': len(required_features),
        'existing_count': len(existing_features),
        'missing_features': missing_features,
        'extra_features': extra_features[:10]  # 只显示前10个
    }

    if missing_features:
        result['passed'] = False
        result['message'] = f"缺少 {len(missing_features)} 个必需特征"
    else:
        result['message'] = f"所有 {len(required_features)} 个特征都存在"

    return result


def check_label_validity(df: pd.DataFrame, label_col: str = 'label') -> Dict[str, any]:
    """
    检查标签有效性

    Args:
        df: 待检查的DataFrame
        label_col: 标签列名

    Returns:
        包含标签有效性检查结果的字典
    """
    result = {
        'check_name': 'label_validity',
        'passed': True,
        'details': {}
    }

    if label_col not in df.columns:
        result['passed'] = False
        result['message'] = f"标签列 '{label_col}' 不存在"
        result['details'] = {'label_exists': False}
        return result

    # 检查标签值是否只有0和1
    unique_labels = df[label_col].unique()
    invalid_labels = [label for label in unique_labels if label not in [0, 1]]

    result['details'] = {
        'unique_labels': unique_labels.tolist(),
        'invalid_labels': invalid_labels,
        'label_count': len(df[label_col])
    }

    if invalid_labels:
        result['passed'] = False
        result['message'] = f"发现无效标签值: {invalid_labels}"
    else:
        result['message'] = "标签值有效（只有0和1）"

    return result


def check_date_distribution(df: pd.DataFrame, date_col: str = 'date') -> Dict[str, any]:
    """
    检查日期分布

    Args:
        df: 待检查的DataFrame
        date_col: 日期列名

    Returns:
        包含日期分布检查结果的字典
    """
    result = {
        'check_name': 'date_distribution',
        'passed': True,
        'details': {}
    }

    if date_col not in df.columns:
        result['passed'] = False
        result['message'] = f"日期列 '{date_col}' 不存在"
        result['details'] = {'date_exists': False}
        return result

    unique_dates = df[date_col].nunique()
    date_range = (df[date_col].min(), df[date_col].max())

    result['details'] = {
        'unique_dates': unique_dates,
        'date_range': date_range,
        'samples_per_date': len(df) / unique_dates if unique_dates > 0 else 0
    }

    result['message'] = f"数据覆盖 {unique_dates} 个交易日"

    return result


def run_quality_checks(df: pd.DataFrame,
                      required_features: List[str],
                      label_col: str = 'label',
                      time_col: str = 'time',
                      date_col: str = 'date',
                      unique_keys: List[str] = None) -> List[Dict[str, any]]:
    """
    运行所有质量检查

    Args:
        df: 待检查的DataFrame
        required_features: 必需的特征列表
        label_col: 标签列名
        time_col: 时间列名
        date_col: 日期列名
        unique_keys: 需要检查唯一性的列名列表

    Returns:
        包含所有检查结果的列表
    """
    results = []

    # 1. 检查唯一性
    results.append(check_uniqueness(df, unique_keys))

    # 2. 检查空值
    results.append(check_null_values(df, required_features))

    # 3. 检查正样本
    results.append(check_positive_samples(df, label_col))

    # 4. 检查时间单调性
    results.append(check_time_monotonic(df, time_col))

    # 5. 检查特征完整性
    results.append(check_feature_completeness(df, required_features))

    # 6. 检查标签有效性
    results.append(check_label_validity(df, label_col))

    # 7. 检查日期分布
    results.append(check_date_distribution(df, date_col))

    return results


def print_quality_report(quality_results: List[Dict[str, any]]):
    """
    打印质量检查报告

    Args:
        quality_results: 质量检查结果列表
    """
    print("\n" + "="*80)
    print("数据质量检查报告")
    print("="*80)

    passed_count = sum(1 for result in quality_results if result['passed'])
    total_count = len(quality_results)

    for i, result in enumerate(quality_results, 1):
        status = "✅ 通过" if result['passed'] else "❌ 失败"
        print(f"\n{i}. {result['check_name'].upper().replace('_', ' ')}: {status}")
        print(f"   {result['message']}")

        # 打印详细信息
        if result['check_name'] == 'positive_samples' and 'details' in result:
            details = result['details']
            print(f"   总样本: {details['total_samples']:,}")
            print(f"   正样本: {details['positive_count']:,}")
            print(f"   负样本: {details['negative_count']:,}")
            print(f"   正样本比例: {details['positive_ratio']:.4%}")

        elif result['check_name'] == 'feature_completeness' and 'details' in result:
            details = result['details']
            print(f"   需要特征: {details['required_count']}")
            print(f"   实际特征: {details['existing_count']}")
            if details['missing_features']:
                print(f"   缺失特征: {', '.join(details['missing_features'][:10])}")

        elif result['check_name'] == 'date_distribution' and 'details' in result:
            details = result['details']
            print(f"   交易日数: {details['unique_dates']}")
            print(f"   平均每交易日样本: {details['samples_per_date']:.0f}")

    print("\n" + "="*80)
    print(f"总体结果: {passed_count}/{total_count} 项检查通过")
    print("="*80)

    if passed_count == total_count:
        print("✅ 所有质量检查通过！")
    else:
        print(f"❌ 有 {total_count - passed_count} 项检查失败，请查看详细信息")