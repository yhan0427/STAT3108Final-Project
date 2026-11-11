from pathlib import Path

import numpy as np
import pandas as pd

from .fundamentals import FUND_FEATURES, align_snapshots, make_snapshots
from .io import read_json, sha256, utc_now, write_json

MARKET_FEATURES = ["return", "abs_return", "rv5", "rv20", "rv60", "log_range", "volume_change", "momentum20", "market_return", "market_rv20"]
HAR_FEATURES = ["log_sq_return", "log_mean_sq5", "log_mean_sq22"]


def forward_volatility(returns, horizon=20, annualization=252):
    # At row t+h, rolling(h) covers t+1..t+h. Shift it BACK to t.
    return returns.rolling(horizon, min_periods=horizon).std(ddof=0).shift(-horizon) * np.sqrt(annualization)


def assign_splits(frame, config):
    split = pd.Series("excluded", index=frame.index)
    boundaries = [("train", config["train_start"], config["validation_start"]), ("validation", config["validation_start"], config["test_start"]), ("test", config["test_start"], config["test_end_exclusive"])]
    for name, start, stop in boundaries:
        mask = (frame.date >= pd.Timestamp(start)) & (frame.date < pd.Timestamp(stop)) & (frame.target_end < pd.Timestamp(stop))
        split.loc[mask] = name
    return split


def market_features(raw, market, config):
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and sorted")
    if not raw.index.equals(market.index):
        raise ValueError("Stock history must match the SPY trading calendar; inspect missing dates")
    needed = ["adj_close", "close", "high", "low", "volume"]
    if raw[needed].isna().any().any() or (raw[["adj_close", "close", "high", "low"]] <= 0).any().any():
        raise ValueError("Invalid prices; refusing forward filling")
    r = np.log(raw.adj_close).diff()
    m = np.log(market.adj_close).diff()
    out = pd.DataFrame(index=raw.index)
    out["return"] = r
    out["abs_return"] = r.abs()
    for window in [5, 20, 60]:
        out[f"rv{window}"] = r.rolling(window).std(ddof=0) * np.sqrt(config["annualization"])
    out["log_range"] = np.log(raw.high / raw.low)
    out["volume_change"] = np.log1p(raw.volume).diff()
    out["momentum20"] = np.log(raw.adj_close).diff(20)
    out["market_return"] = m
    out["market_rv20"] = m.rolling(20).std(ddof=0) * np.sqrt(config["annualization"])
    sq = r.pow(2)
    out["log_sq_return"] = np.log(sq.clip(lower=1e-12))
    out["log_mean_sq5"] = np.log(sq.rolling(5).mean().clip(lower=1e-12))
    out["log_mean_sq22"] = np.log(sq.rolling(22).mean().clip(lower=1e-12))
    out["target"] = forward_volatility(r, config["horizon"], config["annualization"])
    out["target_end"] = pd.Series(out.index, index=out.index).shift(-config["horizon"])
    out = out.reset_index(names="date")
    valid_market = out[MARKET_FEATURES + HAR_FEATURES].notna().all(axis=1)
    out["sequence_ready"] = valid_market.rolling(config["lookback"], min_periods=config["lookback"]).sum().eq(config["lookback"])
    out["eligible"] = out.sequence_ready & out.target.notna() & (out.target > 0)
    out["split"] = assign_splits(out, config)
    out.loc[~out.eligible, "split"] = "excluded"
    return out


def build(root, config, universe, pilot=False, market_only=False):
    root = Path(root)
    output = root / "data/processed" / ("pilot" if pilot else "main")
    output.mkdir(parents=True, exist_ok=True)
    symbols = universe["pilot"] if pilot else list(universe["companies"])
    market = pd.read_csv(root / "data/raw/market/SPY.csv", parse_dates=["date"]).set_index("date")
    all_frames, sources, coverage = [], {}, {}
    for ticker in symbols:
        raw_path = root / "data/raw/market" / f"{ticker}.csv"
        raw = pd.read_csv(raw_path, parse_dates=["date"]).set_index("date")
        features = market_features(raw, market, config)
        features["ticker"] = ticker
        features["sector"] = universe["companies"][ticker]
        sources[str(raw_path.relative_to(root))] = sha256(raw_path)
        if not market_only:
            sec_path = root / read_json(root / "data/raw/sec/resolved_entities.json")[ticker]["path"]
            snapshots = make_snapshots(read_json(sec_path), latest=config["data_end_exclusive"])
            snapshots.to_csv(output / f"{ticker}_fundamental_snapshots.csv", index=False)
            aligned = align_snapshots(features.date, snapshots)
            features = features.merge(aligned, on="date", how="left", validate="one_to_one")
            sources[str(sec_path.relative_to(root))] = sha256(sec_path)
            coverage[ticker] = {name: float(features.loc[features.split != "excluded", name].notna().mean()) for name in FUND_FEATURES}
        all_frames.append(features)
        print(f"Built {ticker}", flush=True)
    panel = pd.concat(all_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    panel.to_pickle(output / "panel.pkl")
    sources["data/raw/market/SPY.csv"] = sha256(root / "data/raw/market/SPY.csv")
    counts = panel.loc[panel.split != "excluded"].groupby(["split", "ticker"]).size().unstack(fill_value=0)
    write_json(output / "manifest.json", {"built_utc": utc_now(), "config": config, "universe": symbols, "universe_description": universe["description"], "has_fundamentals": not market_only, "sources": sources, "panel_sha256": sha256(output / "panel.pkl"), "rows_per_split": panel.split.value_counts().to_dict(), "coverage": coverage, "balanced_eligible_panel": bool(counts.nunique(axis=1).eq(1).all())})
    print(counts.to_string(), flush=True)
    return panel
