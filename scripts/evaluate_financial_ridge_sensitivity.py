"""Expanded Ridge regularization grid, separately registered after validation-boundary selection."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.aligned import variance_calibration
from volrisk.dataset import MARKET_FEATURES
from volrisk.fundamentals_v3 import V3_FEATURES
from volrisk.io import read_json, sha256, utc_now, write_json
from volrisk.metrics import block_bootstrap, forecast_metrics, qlike
from volrisk.stream_ridge import StreamRidge, mask_statistics


def main():
    config_path = ROOT / "config/financial-ridge-sensitivity-v1.json"
    cfg = read_json(config_path)
    primary_path = ROOT / "results/financial/full-v1/registered_plan.json"
    if sha256(primary_path) != cfg["main_plan_sha256"]:
        raise ValueError("Primary plan changed")
    primary = read_json(primary_path)
    path = ROOT / "data/processed/financial-full-v1/panel.pkl"
    if sha256(path) != primary["panel_sha256"]:
        raise ValueError("Dataset changed")
    dest = ROOT / "results/financial/ridge-sensitivity-v1"
    if dest.exists():
        raise FileExistsError("Preserve sensitivity run")
    dest.mkdir(parents=True)
    write_json(dest / "registered_plan.json", {"started_utc": utc_now(), "config": cfg,
        "source_sha256": sha256(Path(__file__)), "stream_ridge_sha256": sha256(ROOT / "src/volrisk/stream_ridge.py")})
    panel = pd.read_pickle(path)
    if panel.date.max() >= pd.Timestamp("2026-01-01"):
        raise ValueError("Sealed market data found")
    held = panel.role.eq("held_company")
    tr = panel.index[panel.split.eq("train") & ~held].to_numpy()
    va = panel.index[panel.split.eq("validation") & ~held & panel.context_ready.eq(True)].to_numpy()
    te = panel.index[panel.split.eq("test") & held & panel.context_ready.eq(True)].to_numpy()
    ids = np.r_[va, te]
    frozen = joblib.load(ROOT / "results/financial/full-v1/preprocessing.joblib")
    market = frozen["market"].transform(panel[MARKET_FEATURES])
    static = frozen["financial"].transform(panel[V3_FEATURES], missing=True)
    y = np.log(panel.target.clip(lower=1e-8).fillna(1)).to_numpy()
    base = panel.loc[ids, ["date", "ticker", "sector", "role", "split", "target", "target_end", "rv20"]].copy()
    rows, selections, calibration, intervals = [], {}, {}, {}
    with threadpool_limits(limits=2):
        statistics = StreamRidge.statistics(market, tr, y, 60, static=static)
        for group, mask_values in primary["masks"].items():
            mask = np.asarray(mask_values)
            controlled = static * mask.astype(np.float32)
            statistics_masked = mask_statistics(statistics, np.r_[np.ones(600), mask])
            best, choices = None, []
            for alpha in cfg["alpha_grid"]:
                model = StreamRidge(alpha).fit_statistics(statistics_masked)
                loss = float(np.mean((model.predict(market, va, static=controlled) - y[va]) ** 2))
                choices.append({"alpha": alpha, "validation_log_mse": loss})
                if best is None or loss < best[0]:
                    best = loss, alpha, model
            name = f"Ridge_{group}"
            base[name] = np.exp(np.clip(best[2].predict(market, ids, static=controlled), -18, 5))
            calibration[name] = variance_calibration(base.loc[va, "target"], base.loc[va, name])
            base[name + "_calibrated"] = base[name] * calibration[name]
            selections[group] = {"grid": choices, "selected_alpha": best[1]}
            joblib.dump(best[2], dest / f"{name}.joblib")
            print(group, selections[group], flush=True)
    for group in primary["masks"]:
        for suffix in ["", "_calibrated"]:
            name = f"Ridge_{group}{suffix}"
            test = base.loc[te]
            rows.append({"model": name, "companies": test.ticker.nunique(),
                         **forecast_metrics(test.target, test[name], test.rv20)})
    for suffix in ["", "_calibrated"]:
        for first, second in [("financial", "availability"), ("financial", "market"), ("availability", "market")]:
            a, b = f"Ridge_{first}{suffix}", f"Ridge_{second}{suffix}"
            test = base.loc[te]
            diff = pd.DataFrame({"date": test.date, "d": qlike(test.target, test[a]) - qlike(test.target, test[b])}).groupby("date").d.mean()
            intervals[f"{a}-minus-{b}"] = {str(block): block_bootstrap(diff, block, 1000) for block in [20, 60]}
    base.to_csv(dest / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "results/FINANCIAL_RIDGE_SENSITIVITY_V1.csv", index=False)
    write_json(ROOT / "results/FINANCIAL_RIDGE_SENSITIVITY_V1.json", {
        "completed_utc": utc_now(), "registered_config": cfg, "selections": selections,
        "calibration": calibration, "paired_intervals": intervals,
        "predictions_sha256": sha256(dest / "predictions.csv"), "market_outcomes_2026_accessed": False})
    write_json(dest / "COMPLETED.json", {"completed_utc": utc_now()})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
