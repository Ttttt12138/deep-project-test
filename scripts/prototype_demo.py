import argparse
import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def fig_to_base64(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_metric_chart(df: pd.DataFrame, metric: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 3))
    x = df["fold"].astype(str)
    y = df[metric]
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    return fig_to_base64(fig)


def build_html(
    rolling_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    chart_auc: str,
    chart_p100: str,
    chart_pos: str,
) -> str:
    best_fold = rolling_df.loc[rolling_df["pr_auc"].idxmax()]
    latest_day = daily_df.sort_values("date").iloc[-1]

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deep Project 原型演示</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #182230;
      --muted: #576579;
      --line: #d9e1ea;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(140deg, #f4f7fb 0%, #e8f1ff 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1080px;
      margin: 28px auto;
      padding: 0 16px 32px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px;
      margin-bottom: 14px;
      box-shadow: 0 6px 20px rgba(9, 30, 66, 0.06);
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    .kpi {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
    }}
    .kpi-item {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: #fbfdff;
    }}
    .k {{ font-size: 12px; color: var(--muted); }}
    .v {{ font-size: 22px; font-weight: 700; margin-top: 4px; color: var(--accent); }}
    img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 8px;
    }}
    th {{ background: #f7fbff; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Deep Project 原型演示</h1>
      <p>基于 rolling CV 结果和日级日志的快速可视化样例，用于对外演示与内部复盘。</p>
      <div class="kpi">
        <div class="kpi-item"><div class="k">最佳 Fold</div><div class="v">{int(best_fold["fold"])}</div></div>
        <div class="kpi-item"><div class="k">最佳 PR AUC</div><div class="v">{best_fold["pr_auc"]:.4f}</div></div>
        <div class="kpi-item"><div class="k">平均 ROC AUC</div><div class="v">{rolling_df["roc_auc"].mean():.4f}</div></div>
        <div class="kpi-item"><div class="k">最新日期</div><div class="v">{latest_day["date"]}</div></div>
        <div class="kpi-item"><div class="k">最新正样本</div><div class="v">{int(latest_day["positive_samples"])}</div></div>
        <div class="kpi-item"><div class="k">最新降采样比</div><div class="v">{latest_day["final_ratio"]:.2f}</div></div>
      </div>
    </div>
    <div class="card">
      <h2>Rolling CV 指标趋势</h2>
      <img alt="ROC AUC" src="data:image/png;base64,{chart_auc}" />
      <div style="height:10px"></div>
      <img alt="P@100" src="data:image/png;base64,{chart_p100}" />
    </div>
    <div class="card">
      <h2>日志侧正样本趋势</h2>
      <img alt="Positive Samples" src="data:image/png;base64,{chart_pos}" />
    </div>
    <div class="card">
      <h2>Rolling CV 明细（前 10 行）</h2>
      {rolling_df.head(10).to_html(index=False, classes="table", border=0)}
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prototype demo HTML.")
    parser.add_argument(
        "--rolling-csv",
        default="data/rolling_cv/results_fixed.csv",
        help="Path to rolling cv result csv",
    )
    parser.add_argument(
        "--logs-dir",
        default="data/logs/01",
        help="Directory containing daily summary csv files",
    )
    parser.add_argument(
        "--output",
        default="docs/prototype_demo.html",
        help="Output html path",
    )
    args = parser.parse_args()

    rolling_path = Path(args.rolling_csv)
    logs_dir = Path(args.logs_dir)
    output_path = Path(args.output)

    if not rolling_path.exists():
        raise FileNotFoundError(f"Missing rolling csv: {rolling_path}")
    if not logs_dir.exists():
        raise FileNotFoundError(f"Missing logs directory: {logs_dir}")

    rolling_df = pd.read_csv(rolling_path)
    log_files = sorted(logs_dir.glob("*_summary.csv"))
    if not log_files:
        raise FileNotFoundError(f"No summary files found in {logs_dir}")

    daily_df = pd.concat([pd.read_csv(f) for f in log_files], ignore_index=True)
    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    daily_df = daily_df.dropna(subset=["date"]).sort_values("date")
    daily_df["date"] = daily_df["date"].dt.strftime("%Y-%m-%d")

    chart_auc = render_metric_chart(rolling_df, "roc_auc", "ROC AUC by Fold")
    chart_p100 = render_metric_chart(rolling_df, "p_at_100", "P@100 by Fold")
    chart_pos = render_metric_chart(
        daily_df.reset_index(drop=True).assign(fold=lambda x: x.index + 1),
        "positive_samples",
        "Daily Positive Samples",
    )

    html = build_html(rolling_df, daily_df, chart_auc, chart_p100, chart_pos)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Prototype generated: {output_path.resolve()}")


if __name__ == "__main__":
    main()
