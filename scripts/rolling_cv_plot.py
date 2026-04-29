"""
滚动验证结果可视化脚本

生成:
- data/rolling_cv/rolling_cv_plot.png: AUC / Precision / Recall 趋势图

使用方法:
    python scripts/rolling_cv_plot.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'rolling_cv', 'results.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'rolling_cv', 'rolling_cv_plot.png')


def plot_rolling_cv_results():
    """绘制滚动验证结果"""
    if not os.path.exists(RESULTS_FILE):
        print(f"错误: 结果文件不存在: {RESULTS_FILE}")
        print("请先运行: python scripts/rolling_cv.py")
        sys.exit(1)

    # 读取结果
    df = pd.read_csv(RESULTS_FILE)
    print(f"加载结果: {RESULTS_FILE}")
    print(f"Folds: {len(df)}")

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rolling Cross-Validation Results', fontsize=14, fontweight='bold')

    # 月份标签
    months = [f"{int(m):02d}" for m in df['val_month']]
    x = range(len(months))

    # 1. ROC-AUC 趋势
    ax1 = axes[0, 0]
    ax1.plot(x, df['roc_auc'], 'b-o', linewidth=2, markersize=8, label='ROC-AUC')
    ax1.fill_between(x,
                     df['roc_auc'] - df['roc_auc'].std(),
                     df['roc_auc'] + df['roc_auc'].std(),
                     alpha=0.2, color='blue')
    ax1.axhline(y=df['roc_auc'].mean(), color='b', linestyle='--', alpha=0.5, label=f'Mean: {df["roc_auc"].mean():.4f}')
    ax1.set_xlabel('Validation Month')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('ROC-AUC Trend')
    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.985, 1.0])

    # 2. PR-AUC 趋势
    ax2 = axes[0, 1]
    ax2.plot(x, df['pr_auc'], 'r-o', linewidth=2, markersize=8, label='PR-AUC')
    ax2.fill_between(x,
                     df['pr_auc'] - df['pr_auc'].std(),
                     df['pr_auc'] + df['pr_auc'].std(),
                     alpha=0.2, color='red')
    ax2.axhline(y=df['pr_auc'].mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {df["pr_auc"].mean():.4f}')
    ax2.set_xlabel('Validation Month')
    ax2.set_ylabel('PR-AUC')
    ax2.set_title('PR-AUC Trend')
    ax2.set_xticks(x)
    ax2.set_xticklabels(months)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.94, 0.98])

    # 3. Precision & Recall
    ax3 = axes[1, 0]
    ax3.plot(x, df['precision'], 'g-o', linewidth=2, markersize=8, label='Precision')
    ax3.plot(x, df['recall'], 'm-s', linewidth=2, markersize=8, label='Recall')
    ax3.plot(x, df['f1_score'], 'c-^', linewidth=2, markersize=8, label='F1-Score')
    ax3.set_xlabel('Validation Month')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision / Recall / F1-Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(months)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.8, 1.0])

    # 4. 训练集规模 vs 性能
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    # 柱状图：训练样本数
    bars = ax4.bar(x, df['train_samples'] / 1000, alpha=0.3, color='gray', label='Train Samples (K)')
    ax4.set_xlabel('Validation Month')
    ax4.set_ylabel('Train Samples (K)', color='gray')
    ax4.tick_params(axis='y', labelcolor='gray')

    # 折线图：PR-AUC
    ax4_twin.plot(x, df['pr_auc'], 'r-o', linewidth=2, markersize=8, label='PR-AUC')
    ax4_twin.set_ylabel('PR-AUC', color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')

    ax4.set_title('Training Set Size vs Performance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(months)

    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {OUTPUT_FILE}")

    # 打印统计摘要
    print("\n" + "="*60)
    print("统计摘要")
    print("="*60)
    print(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*60)
    print(f"{'ROC-AUC':<15} {df['roc_auc'].mean():>10.4f} {df['roc_auc'].std():>10.4f} {df['roc_auc'].min():>10.4f} {df['roc_auc'].max():>10.4f}")
    print(f"{'PR-AUC':<15} {df['pr_auc'].mean():>10.4f} {df['pr_auc'].std():>10.4f} {df['pr_auc'].min():>10.4f} {df['pr_auc'].max():>10.4f}")
    print(f"{'Precision':<15} {df['precision'].mean():>10.4f} {df['precision'].std():>10.4f} {df['precision'].min():>10.4f} {df['precision'].max():>10.4f}")
    print(f"{'Recall':<15} {df['recall'].mean():>10.4f} {df['recall'].std():>10.4f} {df['recall'].min():>10.4f} {df['recall'].max():>10.4f}")
    print(f"{'F1-Score':<15} {df['f1_score'].mean():>10.4f} {df['f1_score'].std():>10.4f} {df['f1_score'].min():>10.4f} {df['f1_score'].max():>10.4f}")


if __name__ == "__main__":
    print("="*60)
    print("滚动验证结果可视化")
    print("="*60)
    plot_rolling_cv_results()
    print("\n完成!")