import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def qlike(actual, prediction):
    ratio = np.maximum(np.asarray(actual, dtype=float), 1e-8) ** 2 / np.maximum(np.asarray(prediction, dtype=float), 1e-8) ** 2
    return ratio - np.log(ratio) - 1.0


def forecast_metrics(actual, prediction, persistence):
    y, p, b = map(lambda x: np.asarray(x, dtype=float), (actual, prediction, persistence))
    if not (np.isfinite(y).all() and np.isfinite(p).all() and (p > 0).all()):
        raise ValueError("Forecast metrics require finite positive predictions")
    mse, denom = np.mean((y - p) ** 2), np.mean((y - b) ** 2)
    return {"n": len(y), "mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(mse)), "qlike": float(np.mean(qlike(y, p))), "r2_vs_persistence": float(1 - mse / denom) if denom > 0 else None}


def ranking_by_date(predictions, model, k=5):
    records, previous = [], None
    for date, group in predictions.groupby("date", sort=True):
        if len(group) <= k:
            continue  # Top-five overlap in a five-stock universe is meaningless.
        predicted = set(group.sort_values([model, "ticker"], ascending=[False, True]).head(k).ticker)
        actual = set(group.sort_values(["target", "ticker"], ascending=[False, True]).head(k).ticker)
        rho = spearmanr(group[model], group.target).statistic
        records.append({"date": date, "top_k_overlap": len(predicted & actual) / k, "spearman": float(rho) if np.isfinite(rho) else np.nan, "turnover": np.nan if previous is None else 1 - len(predicted & previous) / k})
        previous = predicted
    return pd.DataFrame(records)


def block_bootstrap(values, block=20, repetitions=1000, seed=2026):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < block:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for i in range(repetitions):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
        idx = ((starts[:, None] + np.arange(block)) % len(values)).ravel()[:len(values)]
        means[i] = values[idx].mean()
    return {"mean_difference": float(values.mean()), "ci95_low": float(np.quantile(means, 0.025)), "ci95_high": float(np.quantile(means, 0.975)), "block_days": block, "replicates": repetitions}


def evaluate_predictions(frame, models, k=5, bootstrap_replicates=1000):
    summary, ranks, intervals, sectors, regimes = {}, {}, {}, {}, {}
    for model in models:
        summary[model] = forecast_metrics(frame.target, frame[model], frame.Persistence)
        ranking = ranking_by_date(frame, model, k)
        ranks[model] = {col: float(ranking[col].mean()) for col in ["top_k_overlap", "spearman", "turnover"]} if not ranking.empty else {"note": "Not scored: stock universe must be larger than k"}
        for field, dest in [("sector", sectors), ("regime", regimes)]:
            dest[model] = {name: forecast_metrics(g.target, g[model], g.Persistence) for name, g in frame.groupby(field)}
    pairs = [(m, "Persistence") for m in models if m != "Persistence"]
    if "LSTM_fundamentals" in models and "LSTM_market" in models:
        pairs.append(("LSTM_fundamentals", "LSTM_market"))
    if "Boosting_fundamentals" in models:
        pairs.append(("Boosting_fundamentals", "Boosting_market"))
    for model, baseline in pairs:
        differences = pd.DataFrame({"date": frame.date, "loss_difference": qlike(frame.target, frame[model]) - qlike(frame.target, frame[baseline])})
        daily = differences.groupby("date").loss_difference.mean().to_numpy()
        intervals[f"{model} minus {baseline}"] = {str(block): block_bootstrap(daily, block, bootstrap_replicates) for block in [20, 60]}
    return {"metrics": summary, "ranking": ranks, "paired_qlike_intervals": intervals, "by_sector": sectors, "by_regime": regimes, "interpretation": "Negative QLIKE differences favor the first model. CIs are paired date-block bootstrap intervals, not independent firm-day tests. Forecast volatility is not direction, default probability or an investment return."}
