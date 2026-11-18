"""Matched market / availability / financial comparisons, without opening 2026."""
from pathlib import Path
import argparse
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
from volrisk.models import TrainOnlyScaler, Sequences, train_network, predict_network
from volrisk.stream_ridge import StreamRidge, mask_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity-guard", action="store_true")
    args = parser.parse_args()
    run = "entity-v1" if args.entity_guard else "full-v1"
    dataset = "financial-entity-v1" if args.entity_guard else "financial-full-v1"
    result_stem = "FINANCIAL_ENTITY" if args.entity_guard else "FINANCIAL_FULL"
    dest = ROOT / f"results/financial/{run}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "COMPLETED.json").exists():
        raise FileExistsError("Completed financial experiment is immutable")
    path = ROOT / f"data/processed/{dataset}/panel.pkl"
    manifest = read_json(path.parent / "manifest.json")
    if sha256(path) != manifest["panel_sha256"]:
        raise ValueError("Financial dataset changed")
    panel = pd.read_pickle(path)
    if panel.date.max() >= pd.Timestamp("2026-01-01") or panel.sector.eq("Financials").any():
        raise ValueError("Unexpected scope or sealed data")
    cfg = read_json(ROOT / "config/study.json")
    held = panel.role.eq("held_company")
    train = panel.index[panel.split.eq("train") & ~held].to_numpy()
    validation = panel.index[panel.split.eq("validation") & ~held & panel.context_ready.eq(True)].to_numpy()
    evaluation = panel.index[panel.split.eq("test") & held & panel.context_ready.eq(True)].to_numpy()
    indices = np.r_[validation, evaluation]
    for ids in [train, validation, evaluation]:
        if not np.array_equal(panel.loc[ids, "ticker"], panel.loc[ids - 59, "ticker"]):
            raise ValueError("Cross-company market history")
    if not set(panel.loc[train, "ticker"]).isdisjoint(panel.loc[evaluation, "ticker"]):
        raise ValueError("Held-company fitting contamination")
    if panel.loc[train, "target_end"].max() >= pd.Timestamp("2022-01-01") or panel.loc[validation, "target_end"].max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("Target boundary purge failed")
    masks = {"market": np.zeros(16), "availability": np.r_[np.zeros(7), np.ones(9)],
             "financial": np.ones(16)}
    if args.entity_guard:
        masks = {"market": np.zeros(16), "filing_age": np.r_[np.zeros(7), 1., np.zeros(7), 1.],
                 "availability": masks["availability"], "financial": masks["financial"]}
    alpha_grid = read_json(ROOT / "config/ridge-regularization-audit-v1.json")["alpha_grid"] if args.entity_guard else [1., 100., 10000.]
    if len(V3_FEATURES) != 8 or V3_FEATURES[-1] != "filing_age_days":
        raise ValueError("Frozen financial coordinate layout changed")
    plan = {"panel_sha256": sha256(path), "config": cfg, "financial_features": V3_FEATURES,
            "masks": {k: v.tolist() for k, v in masks.items()},
            "source_sha256": sha256(Path(__file__)),
            "implementation_hashes": {name: sha256(ROOT / f"src/volrisk/{name}.py")
                                      for name in ["models", "stream_ridge", "metrics", "aligned"]},
            "train_companies": sorted(panel.loc[train, "ticker"].unique()),
            "validation_companies": sorted(panel.loc[validation, "ticker"].unique()),
            "evaluation_companies": sorted(panel.loc[evaluation, "ticker"].unique()),
            "row_counts": {"train": len(train), "validation": len(validation), "evaluation": len(evaluation)},
            "ridge_alpha_grid": alpha_grid,
            "entity_guard": args.entity_guard,
            "entity_policy_sha256": manifest.get("policy_sha256"),
            "controls": "All LSTMs have the same 16-coordinate branch and parameters; market zeros all static inputs, availability zeros seven economic values but retains filing age/eight missing flags. Ridge uses the same sixty-day market lag plus current static coordinates.",
            "scope": "Retained non-financial reference cohort with unchanged company roles and no complete-case deletion. Entity variant conservatively excludes unverified predecessor windows/disclosures for documented transitions; data and evaluation counts can differ. Reuses inspected development years/company cohorts; not a new confirmatory holdout. No text or Timestral refitting in this stage.",
            "preprocessing": "Training-company training-period imputation/clipping/scaling only; validation-only alpha, epochs and calibration. Same seeds and budgets across groups.",
            "primary_comparisons": ["financial-minus-availability", "financial-minus-market", "availability-minus-market"],
            "uncertainty": "Paired whole-date circular blocks 20/60 sessions; 1000 replicates; conditional ensemble diagnostics, no multiplicity adjustment or tuning on held labels",
            "market_outcomes_2026_accessed": False}
    plan_path = dest / "registered_plan.json"
    if plan_path.exists():
        old = read_json(plan_path)
        if {k: v for k, v in old.items() if k != "registered_utc"} != plan:
            raise ValueError("Experiment plan/code changed; version this run")
        plan = old
    else:
        plan["registered_utc"] = utc_now()
        write_json(plan_path, plan)
    market_scaler = TrainOnlyScaler().fit(panel.loc[train, MARKET_FEATURES])
    financial_scaler = TrainOnlyScaler().fit(panel.loc[train, V3_FEATURES])
    market = market_scaler.transform(panel[MARKET_FEATURES])
    static = financial_scaler.transform(panel[V3_FEATURES], missing=True)
    y = np.log(panel.target.clip(lower=1e-8).fillna(1)).to_numpy()
    mean, std = y[train].mean(), y[train].std()
    ys = ((y - mean) / std).astype(np.float32)
    base = panel.loc[indices, ["date", "ticker", "sector", "role", "split", "target", "target_end", "rv20"]].copy()
    base["Persistence"] = base.rv20
    joblib.dump({"market": market_scaler, "financial": financial_scaler, "y_mean": mean, "y_std": std}, dest / "preprocessing.joblib")
    print("FROZEN COUNTS", plan["row_counts"], flush=True)
    ridge_choices = {}
    with threadpool_limits(limits=4):
        statistics = StreamRidge.statistics(market, train, y, 60, static=static)
        for group, mask in masks.items():
            controlled = static * mask.astype(np.float32)
            stats = mask_statistics(statistics, np.r_[np.ones(600), mask])
            choices, best = [], None
            for alpha in plan["ridge_alpha_grid"]:
                model = StreamRidge(alpha).fit_statistics(stats)
                error = float(np.mean((model.predict(market, validation, static=controlled) - y[validation]) ** 2))
                choices.append({"alpha": alpha, "validation_log_mse": error})
                if best is None or error < best[0]:
                    best = error, model
            base[f"Ridge_{group}"] = np.exp(np.clip(best[1].predict(market, indices, static=controlled), -18, 5))
            joblib.dump(best[1], dest / f"Ridge_{group}.joblib")
            ridge_choices[group] = choices
            print("RIDGE", group, choices, flush=True)
    del statistics
    histories = {}
    for group, mask in masks.items():
        controlled = static * mask.astype(np.float32)
        predictions = []
        for seed in cfg["seeds"]:
            cache = dest / f"LSTM_{group}_{seed}.npz"
            history_path = dest / f"LSTM_{group}_{seed}.json"
            if cache.exists():
                meta = read_json(history_path)
                if sha256(cache) != meta["predictions_sha256"] or sha256(dest / meta["weights_file"]) != meta["weights_sha256"]:
                    raise ValueError("Completed seed checkpoint changed")
                pred = np.load(cache)["predictions"]
                history = meta["history"]
            else:
                print("LSTM START", group, seed, flush=True)
                model, history = train_network(market, controlled, ys, train, validation, cfg, seed, True)
                pred = np.exp(np.clip(predict_network(model, Sequences(market, controlled, ys, indices, 60)) * std + mean, -18, 5))
                weights = dest / f"LSTM_{group}_{seed}.pt"
                torch.save(model.state_dict(), weights)
                np.savez(cache, predictions=pred)
                write_json(history_path, {"group": group, "seed": seed, "history": history,
                                          "predictions_sha256": sha256(cache), "weights_file": weights.name,
                                          "weights_sha256": sha256(weights)})
            histories[f"{group}/{seed}"] = history
            predictions.append(pred)
        base[f"LSTM_{group}"] = np.mean(predictions, axis=0)
    models = ["Persistence", *[f"{backbone}_{group}" for backbone in ["Ridge", "LSTM"] for group in masks]]
    scales = {}
    for name in models:
        scales[name] = variance_calibration(base.loc[validation, "target"], base.loc[validation, name])
        base[name + "_calibrated"] = base[name] * scales[name]
    base.to_csv(dest / "predictions.csv", index=False)
    results, companies, periods, intervals = [], [], [], {}
    held_eval = base.loc[evaluation]
    for name in [*models, *[n + "_calibrated" for n in models]]:
        results.append({"model": name, "companies": held_eval.ticker.nunique(),
                        **forecast_metrics(held_eval.target, held_eval[name], held_eval.Persistence)})
        for ticker, part in held_eval.groupby("ticker"):
            companies.append({"model": name, "ticker": ticker, "sector": part.sector.iloc[0],
                              **forecast_metrics(part.target, part[name], part.Persistence)})
        for year, part in held_eval.groupby(held_eval.date.dt.year):
            periods.append({"model": name, "year": int(year),
                            **forecast_metrics(part.target, part[name], part.Persistence)})
    for backbone in ["Ridge", "LSTM"]:
        for calibrated in [False, True]:
            suffix = "_calibrated" if calibrated else ""
            for first, second in [("financial", "availability"), ("financial", "market"), ("availability", "market")]:
                a, b = f"{backbone}_{first}{suffix}", f"{backbone}_{second}{suffix}"
                daily = pd.DataFrame({"date": held_eval.date, "difference": qlike(held_eval.target, held_eval[a]) - qlike(held_eval.target, held_eval[b])}).groupby("date").difference.mean()
                intervals[f"{a}-minus-{b}"] = {str(n): block_bootstrap(daily, n, 1000) for n in [20, 60]}
    pd.DataFrame(results).to_csv(ROOT / f"results/{result_stem}_V1.csv", index=False)
    pd.DataFrame(companies).to_csv(ROOT / f"results/{result_stem}_COMPANIES_V1.csv", index=False)
    pd.DataFrame(periods).to_csv(ROOT / f"results/{result_stem}_PERIODS_V1.csv", index=False)
    write_json(ROOT / f"results/{result_stem}_AUDIT_V1.json", {
        "completed_utc": utc_now(), "registered_plan": plan, "ridge_selection": ridge_choices,
        "calibration": scales, "training_history": histories, "paired_intervals": intervals,
        "predictions_sha256": sha256(dest / "predictions.csv"), "market_outcomes_2026_accessed": False})
    write_json(dest / "COMPLETED.json", {"completed_utc": utc_now(), "evaluation_rows": len(evaluation)})
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
