"""One broad fixed-grid audit for the earlier market and current financial populations."""
from pathlib import Path
import gc
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
from volrisk.metrics import forecast_metrics, qlike, block_bootstrap
from volrisk.stream_ridge import StreamRidge, mask_statistics


def main():
    config_path = ROOT / "config/ridge-regularization-audit-v1.json"
    cfg = read_json(config_path)
    dest = ROOT / "results/financial/ridge-global-audit-v1"
    if dest.exists():
        raise FileExistsError("Preserve the broad-grid audit")
    dest.mkdir(parents=True)
    write_json(dest / "registered_plan.json", {"started_utc": utc_now(), "config": cfg,
        "source_sha256": sha256(Path(__file__)), "ridge_source_sha256": sha256(ROOT / "src/volrisk/stream_ridge.py")})
    results, audits = [], {}
    for population in cfg["populations"]:
        financial = population == "non_financial"
        dataset = "financial-full-v1" if financial else "full-cohort-v2"
        path = ROOT / f"data/processed/{dataset}/panel.pkl"
        meta = read_json(path.parent / "manifest.json")
        if sha256(path) != meta["panel_sha256"]:
            raise ValueError("Panel changed")
        panel = pd.read_pickle(path)
        if panel.date.max() >= pd.Timestamp("2026-01-01"):
            raise ValueError("Sealed market data found")
        held = panel.role.eq("held_company")
        tr = panel.index[panel.split.eq("train") & ~held].to_numpy()
        va = panel.index[panel.split.eq("validation") & ~held & panel.context_ready.eq(True)].to_numpy()
        te = panel.index[panel.split.eq("test") & held & panel.context_ready.eq(True)].to_numpy()
        ids = np.r_[va, te]
        if financial:
            frozen = joblib.load(ROOT / "results/financial/full-v1/preprocessing.joblib")
            market = frozen["market"].transform(panel[MARKET_FEATURES])
            static = frozen["financial"].transform(panel[V3_FEATURES], missing=True)
            primary = read_json(ROOT / "results/financial/full-v1/registered_plan.json")
            masks = {k: np.asarray(v) for k, v in primary["masks"].items()}
        else:
            frozen = joblib.load(ROOT / "results/aligned/full-cohort-v1/fitted.joblib")
            market = frozen["scaler"].transform(panel[MARKET_FEATURES])
            static = None
            masks = {"market": None}
        y = np.log(panel.target.clip(lower=1e-8).fillna(1)).to_numpy()
        base = panel.loc[ids, ["date", "ticker", "sector", "role", "split", "target", "target_end", "rv20"]].copy()
        selection = {}
        with threadpool_limits(limits=2):
            statistics = StreamRidge.statistics(market, tr, y, 60, static=static)
            xx, xy, mean, ym = statistics
            np.savez(dest / f"{population}_statistics.npz", xx=xx, xy=xy, mean=mean, ym=ym)
            for group, mask in masks.items():
                control = static * mask.astype(np.float32) if mask is not None else None
                stats = mask_statistics(statistics, np.r_[np.ones(600), mask]) if mask is not None else statistics
                choices, best = [], None
                for alpha in cfg["alpha_grid"]:
                    model = StreamRidge(alpha).fit_statistics(stats)
                    score = float(np.mean((model.predict(market, va, static=control) - y[va]) ** 2))
                    choices.append({"alpha": alpha, "validation_log_mse": score})
                    if best is None or score < best[0]:
                        best = score, alpha, model
                name = f"Ridge_{group}"
                base[name] = np.exp(np.clip(best[2].predict(market, ids, static=control), -18, 5))
                scale = variance_calibration(base.loc[va, "target"], base.loc[va, name])
                base[name + "_calibrated"] = base[name] * scale
                selection[group] = {"grid": choices, "selected_alpha": best[1], "calibration": scale,
                                    "selected_at_boundary": best[1] in [cfg["alpha_grid"][0], cfg["alpha_grid"][-1]]}
                joblib.dump(best[2], dest / f"{population}_{name}.joblib")
                for suffix in ["", "_calibrated"]:
                    test = base.loc[te]
                    results.append({"population": population, "model": name + suffix,
                                    "companies": test.ticker.nunique(), "selected_alpha": best[1],
                                    **forecast_metrics(test.target, test[name + suffix], test.rv20)})
                print(population, group, "selected alpha", best[1], "boundary", selection[group]["selected_at_boundary"], flush=True)
        intervals = {}
        if financial:
            for suffix in ["", "_calibrated"]:
                for first, second in [("financial", "availability"), ("financial", "market"), ("availability", "market")]:
                    a, b = f"Ridge_{first}{suffix}", f"Ridge_{second}{suffix}"
                    test = base.loc[te]
                    diff = pd.DataFrame({"date": test.date, "d": qlike(test.target, test[a]) - qlike(test.target, test[b])}).groupby("date").d.mean()
                    intervals[f"{a}-minus-{b}"] = {str(n): block_bootstrap(diff, n, 1000) for n in [20, 60]}
        base.to_csv(dest / f"{population}_predictions.csv", index=False)
        audits[population] = {"panel_sha256": sha256(path), "train_rows": len(tr), "validation_rows": len(va),
            "evaluation_rows": len(te), "selection": selection, "paired_intervals": intervals,
            "predictions_sha256": sha256(dest / f"{population}_predictions.csv")}
        del panel, market, static, base, statistics, xx, xy
        gc.collect()
    pd.DataFrame(results).to_csv(ROOT / "results/RIDGE_REGULARIZATION_AUDIT_V1.csv", index=False)
    write_json(ROOT / "results/RIDGE_REGULARIZATION_AUDIT_V1.json", {"completed_utc": utc_now(),
        "registered_config": cfg, "populations": audits, "market_outcomes_2026_accessed": False})
    write_json(dest / "COMPLETED.json", {"completed_utc": utc_now()})
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
