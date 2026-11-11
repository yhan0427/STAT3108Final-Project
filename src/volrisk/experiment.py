import importlib.metadata
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from .dataset import HAR_FEATURES, MARKET_FEATURES
from .fundamentals import FUND_FEATURES
from .io import read_json, sha256, utc_now, write_json
from .metrics import evaluate_predictions, forecast_metrics
from .models import Sequences, TrainOnlyScaler, VolatilityLSTM, predict_network, train_network


def load_panel(root, pilot=False):
    path = root / "data/processed" / ("pilot" if pilot else "main")
    manifest = read_json(path / "manifest.json")
    if sha256(path / "panel.pkl") != manifest["panel_sha256"]:
        raise ValueError("Processed panel checksum mismatch")
    return pd.read_pickle(path / "panel.pkl"), manifest, path


def positive_forecast(log_values):
    return np.exp(np.clip(log_values, -18, 5))


def arrays(panel, bundle):
    market = bundle["market_scaler"].transform(panel[MARKET_FEATURES])
    fund = bundle["fund_scaler"].transform(panel[FUND_FEATURES], missing=True) if bundle["has_fundamentals"] else np.zeros((len(panel), 0), dtype=np.float32)
    tab_market = bundle["tab_scaler"].transform(panel[MARKET_FEATURES + HAR_FEATURES])
    tab_full = np.concatenate([tab_market, fund], axis=1)
    har = bundle["har_scaler"].transform(panel[HAR_FEATURES])
    return market, fund, tab_market, tab_full, har


def base_frame(panel, indices, regime_threshold):
    out = panel.loc[indices, ["date", "ticker", "sector", "target", "target_end", "rv20", "market_rv20"]].copy()
    out["regime"] = np.where(out.market_rv20 > regime_threshold, "high", "low")
    out["Persistence"] = out.rv20.clip(lower=1e-8)
    return out


def model_predictions(root, run_dir, panel, indices, bundle):
    market, fund, tab_market, tab_full, har = arrays(panel, bundle)
    frame = base_frame(panel, indices, bundle["regime_threshold"])
    frame["HAR_type"] = positive_forecast(bundle["har_model"].predict(har[indices]))
    frame["Boosting_market"] = positive_forecast(bundle["boost_market"].predict(tab_market[indices]))
    if bundle["has_fundamentals"]:
        frame["Boosting_fundamentals"] = positive_forecast(bundle["boost_fund"].predict(tab_full[indices]))
    cfg = bundle["config"]
    for name, use_funds in [("LSTM_market", False), ("LSTM_fundamentals", True)]:
        if name not in bundle["models"]:
            continue
        predictions = []
        used_fund = fund if use_funds else np.zeros_like(fund)
        dataset = Sequences(market, used_fund, np.zeros(len(panel), dtype=np.float32), indices, cfg["lookback"])
        for seed in cfg["seeds"]:
            model = VolatilityLSTM(market.shape[1], fund.shape[1], cfg["lstm"]["hidden_size"])
            model.load_state_dict(torch.load(run_dir / f"{name}_{seed}.pt", map_location="cpu", weights_only=True))
            standardized = predict_network(model, dataset)
            predictions.append(positive_forecast(standardized * bundle["target_std"] + bundle["target_mean"]))
        frame[name] = np.mean(predictions, axis=0)
    return frame


def train(root, config, run_name, pilot=False, baselines_only=False):
    run_dir = root / "results/runs" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run {run_name} already exists; choose a new run name rather than overwrite evidence")
    panel, manifest, path = load_panel(root, pilot)
    if manifest["config"] != config:
        raise ValueError("Dataset configuration changed; rebuild before training")
    run_dir.mkdir(parents=True)
    train_idx = panel.index[panel.split == "train"].to_numpy()
    val_idx = panel.index[panel.split == "validation"].to_numpy()
    for idx in [train_idx, val_idx]:
        if not len(idx):
            raise ValueError("Empty chronological partition")
        # Guard against sequences crossing company boundaries.
        if not (panel.loc[idx, "ticker"].to_numpy() == panel.loc[idx - config["lookback"] + 1, "ticker"].to_numpy()).all():
            raise ValueError("Sequence crosses a company boundary")
    train_rows = panel.loc[train_idx]
    bundle = {"config": config, "pilot": pilot, "has_fundamentals": manifest["has_fundamentals"], "panel_path": str((path / "panel.pkl").relative_to(root)), "panel_sha256": manifest["panel_sha256"], "regime_threshold": float(train_rows.drop_duplicates("date").market_rv20.median())}
    bundle["market_scaler"] = TrainOnlyScaler().fit(train_rows[MARKET_FEATURES])
    bundle["tab_scaler"] = TrainOnlyScaler().fit(train_rows[MARKET_FEATURES + HAR_FEATURES])
    bundle["har_scaler"] = TrainOnlyScaler().fit(train_rows[HAR_FEATURES])
    if bundle["has_fundamentals"]:
        bundle["fund_scaler"] = TrainOnlyScaler().fit(train_rows[FUND_FEATURES])
    market, fund, tab_market, tab_full, har = arrays(panel, bundle)
    log_target = np.log(panel.target.clip(lower=1e-8).fillna(1)).to_numpy()
    bundle["target_mean"], bundle["target_std"] = float(log_target[train_idx].mean()), float(log_target[train_idx].std())
    bundle["har_model"] = LinearRegression().fit(har[train_idx], log_target[train_idx])
    bundle["boost_market"] = HistGradientBoostingRegressor(**config["boosting"], early_stopping=False, random_state=2026).fit(tab_market[train_idx], log_target[train_idx])
    bundle["models"] = ["Persistence", "HAR_type", "Boosting_market"]
    if bundle["has_fundamentals"]:
        bundle["boost_fund"] = HistGradientBoostingRegressor(**config["boosting"], early_stopping=False, random_state=2026).fit(tab_full[train_idx], log_target[train_idx])
        bundle["models"].append("Boosting_fundamentals")
    histories, seed_validation = {}, {}
    if not baselines_only:
        standardized = ((log_target - bundle["target_mean"]) / bundle["target_std"]).astype(np.float32)
        for name, with_funds in [("LSTM_market", False), ("LSTM_fundamentals", True)]:
            if with_funds and not bundle["has_fundamentals"]:
                continue
            bundle["models"].append(name)
            for seed in config["seeds"]:
                network, history = train_network(market, fund, standardized, train_idx, val_idx, config, seed, with_funds)
                torch.save(network.state_dict(), run_dir / f"{name}_{seed}.pt")
                histories[f"{name}_{seed}"] = history
                used_fund = fund if with_funds else np.zeros_like(fund)
                val_data = Sequences(market, used_fund, standardized, val_idx, config["lookback"])
                val_pred = positive_forecast(predict_network(network, val_data) * bundle["target_std"] + bundle["target_mean"])
                seed_validation[f"{name}_{seed}"] = forecast_metrics(panel.loc[val_idx, "target"], val_pred, panel.loc[val_idx, "rv20"])
    joblib.dump(bundle, run_dir / "models.joblib")
    val = model_predictions(root, run_dir, panel, val_idx, bundle)
    val.to_csv(run_dir / "validation_predictions.csv", index=False)
    validation = evaluate_predictions(val, bundle["models"], config["top_k"], config["bootstrap_replicates"])
    write_json(run_dir / "validation_metrics.json", validation)
    write_json(run_dir / "training_history.json", histories)
    write_json(run_dir / "seed_validation_metrics.json", seed_validation)
    packages = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "yfinance", "requests"]}
    code = {str(p.relative_to(root)): sha256(p) for p in (root / "src/volrisk").glob("*.py")}
    weights = {p.name: sha256(p) for p in run_dir.glob("*.pt")}
    weights["models.joblib"] = sha256(run_dir / "models.joblib")
    write_json(run_dir / "frozen_run.json", {"frozen_utc": utc_now(), "config": config, "dataset_manifest": manifest, "packages": packages, "python": platform.python_version(), "models": bundle["models"], "source_sha256": code, "weights_sha256": weights, "test_evaluated": False, "note": "Frozen before final-test evaluation. Early stopping uses validation only; no test-guided model selection."})
    print(pd.DataFrame(validation["metrics"]).T.to_string(), flush=True)


def evaluate(root, run_name):
    run_dir = root / "results/runs" / run_name
    if (run_dir / "test_metrics.json").exists() or (run_dir / "TEST_EVALUATION_STARTED.json").exists():
        raise FileExistsError("Final test was already opened for this run; do not silently rerun it")
    frozen = read_json(run_dir / "frozen_run.json")
    for filename, expected in frozen["source_sha256"].items():
        if sha256(root / filename) != expected:
            raise ValueError(f"Source changed since freeze: {filename}. Record a new experiment before evaluation.")
    for filename, expected in frozen["weights_sha256"].items():
        if sha256(run_dir / filename) != expected:
            raise ValueError("Model changed since freeze")
    bundle = joblib.load(run_dir / "models.joblib")
    if bundle["pilot"]:
        raise ValueError("Pilot test evaluation is disabled; use pilot validation only")
    panel_path = root / bundle["panel_path"]
    if sha256(panel_path) != bundle["panel_sha256"]:
        raise ValueError("Data changed since model fitting")
    panel = pd.read_pickle(panel_path)
    indices = panel.index[panel.split == "test"].to_numpy()
    write_json(run_dir / "TEST_EVALUATION_STARTED.json", {"started_utc": utc_now()})
    torch.set_num_threads(4)
    frame = model_predictions(root, run_dir, panel, indices, bundle)
    frame.to_csv(run_dir / "test_predictions.csv", index=False)
    results = evaluate_predictions(frame, bundle["models"], bundle["config"]["top_k"], bundle["config"]["bootstrap_replicates"])
    write_json(run_dir / "test_metrics.json", results)
    write_json(run_dir / "TEST_EVALUATION_COMPLETED.json", {"completed_utc": utc_now(), "prediction_sha256": sha256(run_dir / "test_predictions.csv")})
    print(pd.DataFrame(results["metrics"]).T.to_string(), flush=True)
