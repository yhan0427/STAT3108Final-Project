"""Portable, offline historical monitoring demo and reproducible research figures."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, write_json


def report(root, run_name):
    run = root / "results/runs" / run_name
    stats = read_json(run / "test_metrics.json")
    frozen = read_json(run / "frozen_run.json")
    predictions = pd.read_csv(run / "test_predictions.csv", parse_dates=["date", "target_end"])
    panel = pd.read_pickle(root / "data/processed/main/panel.pkl")
    output = root / "results/demo" / run_name
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = frozen["models"]
    table = pd.DataFrame(stats["metrics"]).T
    table.to_csv(output / "model_comparison.csv", index_label="model")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#9da8b4", "#7794ad", "#367b83", "#43a39d", "#e6a449", "#c96e48"][:len(models)]
    labels = [m.replace("_", "\n") for m in models]
    for ax, metric, title in zip(axs, ["mae", "rmse", "qlike"], ["MAE · annualized volatility", "RMSE · annualized volatility", "QLIKE · variance loss"]):
        ax.bar(labels, table.loc[models, metric].astype(float), color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Held-out 2024–2025 forecasts · lower is better", y=1.01)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    values = [stats["ranking"][m]["top_k_overlap"] for m in models]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean overlap with the actual top five")
    ax.set_title("Fixed attention budget: inspect five of thirty companies")
    ax.axhline(5 / predictions.ticker.nunique(), color="#64748b", linestyle="--", label="Random selection expectation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "risk_screening.png", dpi=180)
    plt.close(fig)

    first_date = predictions.date.min()
    # Include only trailing realized volatility in the history; future data is never
    # used by the live selection. Realized outcomes remain a separate reveal field.
    history = {}
    for ticker, group in panel.groupby("ticker"):
        keep = group.loc[(group.date >= first_date - pd.Timedelta(days=130)) & (group.date <= predictions.date.max())]
        history[ticker] = [[d.strftime("%Y-%m-%d"), round(float(v), 6)] for d, v in zip(keep.date, keep.rv20) if np.isfinite(v)]
    rows = []
    for row in predictions.to_dict("records"):
        item = {"date": row["date"].strftime("%Y-%m-%d"), "end": row["target_end"].strftime("%Y-%m-%d"), "ticker": row["ticker"], "sector": row["sector"], "actual": round(row["target"], 6), "trailing": round(row["rv20"], 6)}
        item["predictions"] = {m: round(row[m], 6) for m in models}
        rows.append(item)
    data = {"rows": rows, "history": history, "models": models, "metrics": stats["metrics"], "ranking": stats["ranking"], "run": run_name, "universe_note": frozen["dataset_manifest"]["universe_description"]}
    html = (root / "src/volrisk/dashboard.html").read_text()
    html = html.replace("__REPORT_DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c"))
    (output / "index.html").write_text(html)

    lines = ["# Local experiment results", "", "Historical, exploratory v1 study; fixed convenience sample, not historical S&P 500 membership.", "", f"Forecast origins: {predictions.date.min().date()} to {predictions.date.max().date()}; {len(predictions):,} firm-days, {predictions.ticker.nunique()} companies.", "", "| Model | MAE | RMSE | QLIKE | R² vs persistence | Top-5 overlap |", "|---|---:|---:|---:|---:|---:|"]
    for name in models:
        m = stats["metrics"][name]
        lines.append(f"| {name} | {m['mae']:.5f} | {m['rmse']:.5f} | {m['qlike']:.5f} | {m['r2_vs_persistence']:.3f} | {stats['ranking'][name]['top_k_overlap']:.3f} |")
    lines += ["", "MAE/RMSE are annualized volatility fractions (0.01 = one percentage point). QLIKE uses variance. R² is relative to persistence, not to a constant mean. LSTM rows average predictions from three seeds with validation-selected stopping epochs.", "", "## Paired loss comparisons", "", "Negative differences favor the first model. Intervals resample entire dates and preserve the cross-section.", ""]
    for pair, blocks in stats["paired_qlike_intervals"].items():
        for block, ci in blocks.items():
            if ci:
                lines.append(f"- {pair}, {block}-day blocks: mean {ci['mean_difference']:.4f}, 95% interval [{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}].")
    lines += ["", "## Limits", "", "Survivorship and fixed-universe selection; non-financial large caps only; daily-return volatility proxy; historical vendor data may be revised; SEC standard-tag coverage varies; missing financial features are imputed from training data with missing flags; no transaction costs, returns, live deployment or human-efficiency claims. Broad sector groups are manually assigned. Interpret market regimes using contemporaneously observed SPY volatility and a training-only threshold. Demo dates are user-selected examples; assess the full test aggregate rather than a selected success.", "", "## Artifacts", "", "Open index.html for offline historical replay. Full per-company/date predictions, seed histories, coverage, paired block-bootstrap statistics and run/source hashes are retained in the local run and processed-data folders."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (root / "results").mkdir(exist_ok=True)
    (root / "results/RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(output / "summary.json", {"run": run_name, "test_rows": len(predictions), "models": models, "metrics": stats["metrics"], "ranking": stats["ranking"]})
    print(f"Offline demo: {output / 'index.html'}", flush=True)
