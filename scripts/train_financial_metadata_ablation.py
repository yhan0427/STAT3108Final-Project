"""Post-result age/missingness diagnostic with unchanged sample and neural budgets."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.aligned import variance_calibration
from volrisk.dataset import MARKET_FEATURES
from volrisk.fundamentals_v3 import V3_FEATURES
from volrisk.io import read_json, sha256, utc_now, write_json
from volrisk.metrics import block_bootstrap, forecast_metrics, qlike
from volrisk.models import Sequences, train_network, predict_network
from volrisk.stream_ridge import StreamRidge, mask_statistics


def main():
    cfg_path = ROOT / "config/financial-metadata-ablation-v1.json"
    cfg = read_json(cfg_path)
    path = ROOT / "data/processed/financial-full-v1/panel.pkl"
    if sha256(path) != cfg["dataset_sha256"]:
        raise ValueError("Diagnostic sample changed")
    dest = ROOT / "results/financial/metadata-ablation-v1"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "COMPLETED.json").exists():
        raise FileExistsError("Completed mechanism diagnostic is immutable")
    plan = {"config_sha256": sha256(cfg_path), "source_sha256": sha256(Path(__file__)),
            "model_source_sha256": sha256(ROOT / "src/volrisk/models.py"),
            "ridge_source_sha256": sha256(ROOT / "src/volrisk/stream_ridge.py")}
    plan_path = dest / "registered_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Diagnostic implementation changed")
    write_json(plan_path, plan)
    panel = pd.read_pickle(path)
    if panel.date.max() >= pd.Timestamp("2026-01-01"):
        raise ValueError("Sealed data found")
    held = panel.role.eq("held_company")
    tr = panel.index[panel.split.eq("train") & ~held].to_numpy()
    va = panel.index[panel.split.eq("validation") & ~held & panel.context_ready.eq(True)].to_numpy()
    te = panel.index[panel.split.eq("test") & held & panel.context_ready.eq(True)].to_numpy()
    ids = np.r_[va, te]
    frozen = joblib.load(ROOT / "results/financial/full-v1/preprocessing.joblib")
    market = frozen["market"].transform(panel[MARKET_FEATURES])
    static = frozen["financial"].transform(panel[V3_FEATURES], missing=True)
    y = np.log(panel.target.clip(lower=1e-8).fillna(1)).to_numpy()
    ys = ((y - frozen["y_mean"]) / frozen["y_std"]).astype(np.float32)
    base = panel.loc[ids, ["date", "ticker", "sector", "role", "split", "target", "target_end", "rv20"]].copy()
    anchors = pd.read_csv(ROOT / "results/financial/full-v1/predictions.csv", parse_dates=["date"])
    strong = pd.read_csv(ROOT / "results/financial/ridge-global-audit-v1/non_financial_predictions.csv", parse_dates=["date"])
    for source in [anchors, strong]:
        if not base[["date", "ticker"]].reset_index(drop=True).equals(source[["date", "ticker"]]):
            raise ValueError("Anchor forecast origins differ")
        np.testing.assert_allclose(base.target, source.target, rtol=1e-12, atol=1e-12)
    for name in [n for n in anchors if n.startswith("LSTM_")]:
        base[name] = anchors[name].to_numpy()
    for name in [n for n in strong if n.startswith("Ridge_")]:
        base[name] = strong[name].to_numpy()
    stats_path = ROOT / "results/financial/ridge-global-audit-v1/non_financial_statistics.npz"
    stored = np.load(stats_path)
    statistics = stored["xx"], stored["xy"], stored["mean"], float(stored["ym"])
    choices, histories, scales = {}, {}, {}
    with threadpool_limits(limits=2):
        for group, values in cfg["masks"].items():
            mask = np.asarray(values, dtype=np.float32)
            controlled = static * mask
            stats = mask_statistics(statistics, np.r_[np.ones(600), mask])
            grid, best = [], None
            for alpha in cfg["ridge_alpha_grid"]:
                model = StreamRidge(alpha).fit_statistics(stats)
                error = float(np.mean((model.predict(market, va, static=controlled) - y[va]) ** 2))
                grid.append({"alpha": alpha, "validation_log_mse": error})
                if best is None or error < best[0]:
                    best = error, alpha, model
            name = f"Ridge_{group}"
            base[name] = np.exp(np.clip(best[2].predict(market, ids, static=controlled), -18, 5))
            joblib.dump(best[2], dest / f"{name}.joblib")
            choices[group] = {"grid": grid, "selected_alpha": best[1]}
            print("RIDGE", group, best[1], flush=True)
    for group, values in cfg["masks"].items():
        controlled = static * np.asarray(values, dtype=np.float32)
        predictions = []
        for seed in cfg["lstm_config"]["seeds"]:
            path_pred = dest / f"LSTM_{group}_{seed}.npz"
            path_meta = dest / f"LSTM_{group}_{seed}.json"
            if path_pred.exists():
                meta = read_json(path_meta)
                if sha256(path_pred) != meta["predictions_sha256"]:
                    raise ValueError("Seed forecast changed")
                pred = np.load(path_pred)["predictions"]
                history = meta["history"]
            else:
                print("LSTM START", group, seed, flush=True)
                model, history = train_network(market, controlled, ys, tr, va, cfg["lstm_config"], seed, True)
                pred = np.exp(np.clip(predict_network(model, Sequences(market, controlled, ys, ids, 60)) * frozen["y_std"] + frozen["y_mean"], -18, 5))
                weight_path = dest / f"LSTM_{group}_{seed}.pt"
                torch.save(model.state_dict(), weight_path)
                np.savez(path_pred, predictions=pred)
                write_json(path_meta, {"history": history, "weights_sha256": sha256(weight_path),
                                      "predictions_sha256": sha256(path_pred)})
            predictions.append(pred)
            histories[f"{group}/{seed}"] = history
        base[f"LSTM_{group}"] = np.mean(predictions, axis=0)
    for backbone in ["Ridge", "LSTM"]:
        for group in cfg["masks"]:
            name = f"{backbone}_{group}"
            scales[name] = variance_calibration(base.loc[va, "target"], base.loc[va, name])
            base[name + "_calibrated"] = base[name] * scales[name]
    base.to_csv(dest / "predictions.csv", index=False)
    test = base.loc[te]
    models = [n for n in base if n.startswith(("Ridge_", "LSTM_"))]
    rows, years, intervals = [], [], {}
    for name in models:
        rows.append({"model": name, "companies": test.ticker.nunique(),
                     **forecast_metrics(test.target, test[name], test.rv20)})
        for year, part in test.groupby(test.date.dt.year):
            years.append({"model": name, "year": int(year), **forecast_metrics(part.target, part[name], part.rv20)})
    for backbone in ["Ridge", "LSTM"]:
        for suffix in ["", "_calibrated"]:
            for pair in cfg["comparisons"]:
                first, second = pair.split("-minus-")
                a, b = f"{backbone}_{first}{suffix}", f"{backbone}_{second}{suffix}"
                daily = pd.DataFrame({"date": test.date, "d": qlike(test.target, test[a]) - qlike(test.target, test[b])}).groupby("date").d.mean()
                intervals[f"{a}-minus-{b}"] = {str(n): block_bootstrap(daily, n, 1000) for n in [20, 60]}
    pd.DataFrame(rows).to_csv(ROOT / "results/FINANCIAL_METADATA_ABLATION_V1.csv", index=False)
    pd.DataFrame(years).to_csv(ROOT / "results/FINANCIAL_METADATA_ABLATION_PERIODS_V1.csv", index=False)
    write_json(ROOT / "results/FINANCIAL_METADATA_ABLATION_V1.json", {"completed_utc": utc_now(),
        "registered_config": cfg, "ridge_selection": choices, "training_history": histories,
        "calibration": scales, "paired_intervals": intervals,
        "statistics_sha256": sha256(stats_path), "predictions_sha256": sha256(dest / "predictions.csv"),
        "market_outcomes_2026_accessed": False})
    write_json(dest / "COMPLETED.json", {"completed_utc": utc_now()})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
