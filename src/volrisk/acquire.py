"""Public-source downloads. Cached inputs are immutable unless explicitly removed."""
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from .io import read_json, sha256, utc_now, write_json


def fetch_json(url, path, user_agent):
    path = Path(path)
    if path.exists():
        return read_json(path)
    response = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    write_json(path, payload)
    write_json(path.with_suffix(".source.json"), {"url": url, "retrieved_utc": utc_now(), "sha256": sha256(path)})
    time.sleep(0.25)  # Sequential requests, below the SEC's published 10/sec ceiling.
    return payload


def download(root, config, universe, pilot=False, market_only=False):
    root = Path(root)
    raw = root / "data/raw"
    (raw / "market").mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(raw / "yfinance_cache"))
    symbols = universe["pilot"] if pilot else list(universe["companies"])
    for ticker in [*symbols, "SPY"]:
        path = raw / "market" / f"{ticker}.csv"
        if path.exists():
            meta = read_json(path.with_suffix(".source.json"))
            if meta["start"] != config["data_start"] or meta["end_exclusive"] != config["data_end_exclusive"]:
                raise ValueError(f"Cached {ticker} uses different dates; use a separate data directory.")
            if meta["sha256"] != sha256(path):
                raise ValueError(f"Cached {ticker} checksum mismatch")
            print(f"Cached market {ticker}", flush=True)
            continue
        frame = yf.Ticker(ticker).history(start=config["data_start"], end=config["data_end_exclusive"], auto_adjust=False, actions=True, raise_errors=True)
        if frame.empty or "Adj Close" not in frame:
            raise RuntimeError(f"No adjusted price history for {ticker}; refusing synthetic fallback")
        frame.index = frame.index.tz_localize(None).normalize()
        frame.index.name = "date"
        frame = frame.rename(columns=lambda x: x.lower().replace(" ", "_"))
        frame.to_csv(path)
        write_json(path.with_suffix(".source.json"), {"provider": "Yahoo Finance via yfinance", "url": f"https://finance.yahoo.com/quote/{ticker}/history/", "start": config["data_start"], "end_exclusive": config["data_end_exclusive"], "retrieved_utc": utc_now(), "rows": len(frame), "sha256": sha256(path), "auto_adjust": False})
        print(f"Downloaded market {ticker}: {len(frame)} rows", flush=True)
        time.sleep(0.2)
    if market_only:
        return
    private = read_json(root / ".local.json")
    user_agent = private["sec_user_agent"]
    if "@" not in user_agent:
        raise ValueError("SEC user agent must identify a real contact email in .local.json")
    tickers = fetch_json("https://www.sec.gov/files/company_tickers.json", raw / "sec/company_tickers.json", user_agent)
    cik_map = {x["ticker"]: int(x["cik_str"]) for x in tickers.values()}
    resolved_path = raw / "sec/resolved_entities.json"
    resolved = read_json(resolved_path) if resolved_path.exists() else {}
    for ticker in symbols:
        cik = universe.get("historical_cik_overrides", {}).get(ticker, cik_map[ticker])
        path = raw / "sec" / f"{ticker}-{cik}.json"
        legacy = raw / "sec" / f"{ticker}.json"
        if legacy.exists() and not path.exists() and int(read_json(legacy)["cik"]) == cik:
            path.write_bytes(legacy.read_bytes())
            path.with_suffix(".source.json").write_bytes(legacy.with_suffix(".source.json").read_bytes())
        data = fetch_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", path, user_agent)
        if int(data["cik"]) != cik:
            raise ValueError(f"Unexpected entity for {ticker}")
        resolved[ticker] = {"cik": cik, "entity_name": data["entityName"], "path": str(path.relative_to(root)), "sha256": sha256(path)}
        write_json(resolved_path, resolved)
        print(f"SEC {ticker}: {data['entityName']}", flush=True)
