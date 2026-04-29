# 原型演示使用说明

该原型会读取以下数据并生成一个可打开的 HTML 页面：

- `data/rolling_cv/results_fixed.csv`
- `data/logs/01/*_summary.csv`

## 运行方式

```bash
python scripts/prototype_demo.py
```

默认输出：

- `docs/prototype_demo.html`

## 可选参数

```bash
python scripts/prototype_demo.py \
  --rolling-csv data/rolling_cv/results_fixed.csv \
  --logs-dir data/logs/01 \
  --output docs/prototype_demo.html
```

## 页面内容

- 关键 KPI（最佳 Fold、PR AUC、平均 ROC AUC、最新日期正样本等）
- Rolling CV 趋势图（ROC AUC、P@100）
- 日志正样本趋势图
- Rolling CV 明细表（前 10 行）
