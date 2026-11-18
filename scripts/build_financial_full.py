"""Join audited V3 financial quantities to the existing non-financial market cohort."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.fundamentals import align_snapshots
from volrisk.fundamentals_v3 import V3_FEATURES
from volrisk.io import read_json, sha256, utc_now, write_json


def main():
    dest = ROOT / "data/processed/financial-full-v1"
    if dest.exists():
        raise FileExistsError("Preserve the financial panel; version any rebuilt dataset")
    market_path = ROOT / "data/processed/full-cohort-v2/panel.pkl"
    market_meta = read_json(market_path.parent / "manifest.json")
    if sha256(market_path) != market_meta["panel_sha256"]:
        raise ValueError("Market panel changed")
    audit_path = ROOT / "results/FINANCIAL_COVERAGE_AUDIT_V1.json"
    audit = read_json(audit_path)
    if not audit["complete"]:
        raise ValueError("Financial acquisition incomplete")
    original_checks = read_json(ROOT / "results/FINANCIAL_ORIGINAL_CHECKS_V1.json")
    if any(r["status"] == "mismatch" for r in original_checks["counts"]):
        raise ValueError("Investigate original-value mismatches before fitting")
    panel = pd.read_pickle(market_path)
    if panel.date.max() > pd.Timestamp("2025-12-31"):
        raise ValueError("Sealed 2026 market data found")
    panel = panel.loc[panel.sector.ne("Financials")].reset_index(drop=True)
    for feature in V3_FEATURES:
        panel[feature] = np.nan
    panel["financial_filing_date"] = pd.NaT
    sources = {r["ticker"]: r for r in audit["companies"] if r["status"] == "ready"}
    ledger = []
    for ticker, group in panel.groupby("ticker", sort=True):
        source = sources.get(ticker)
        if source is None:
            ledger.append({"ticker": ticker, "status": "no_snapshot_source"})
            continue
        path = ROOT / source["snapshots_path"]
        if sha256(path) != source["snapshots_sha256"]:
            raise ValueError("Snapshot checksum changed")
        snaps = pd.read_csv(path, parse_dates=["filing_date", "report_end"])
        if snaps.filing_date.max() > pd.Timestamp("2025-12-31"):
            raise ValueError("Later disclosure in financial snapshots")
        aligned = align_snapshots(group.date, snaps)
        aligned.loc[aligned.filing_age_days.gt(180), V3_FEATURES[:-1]] = np.nan
        observed = aligned.filing_date.notna()
        if not (aligned.loc[observed, "filing_date"] < aligned.loc[observed, "date"]).all():
            raise ValueError("Same-day or future financial disclosure")
        panel.loc[group.index, V3_FEATURES] = aligned[V3_FEATURES].to_numpy()
        panel.loc[group.index, "financial_filing_date"] = aligned.filing_date.to_numpy()
        ledger.append({"ticker": ticker, "status": "aligned", "snapshot_sha256": source["snapshots_sha256"],
                       "first_available_snapshot": str(snaps.filing_date.min().date()),
                       "market_first_date": str(group.date.min().date()),
                       "entity_history_review_flag": bool(snaps.filing_date.min() > group.date.min() + pd.Timedelta(days=365)),
                       "flag_interpretation": "Late supported data or entity transition; not proof of incorporation date. No CIK stitching performed."})
    dest.mkdir(parents=True)
    panel.to_pickle(dest / "panel.pkl")
    counts = []
    for (role, split), group in panel.loc[panel.split.ne("excluded")].groupby(["role", "split"]):
        common = group.loc[group.context_ready.eq(True)]
        counts.append({"role": role, "split": split, "firms_context60": group.ticker.nunique(),
                       "origins_context60": len(group), "firms_context512": common.ticker.nunique(),
                       "origins_context512": len(common)})
    manifest = {"built_utc": utc_now(), "market_panel_sha256": sha256(market_path),
                "financial_coverage_audit_sha256": sha256(audit_path),
                "original_checks_sha256": sha256(ROOT / "results/FINANCIAL_ORIGINAL_CHECKS_V1.json"),
                "source_sha256": sha256(Path(__file__)), "panel_sha256": sha256(dest / "panel.pkl"),
                "features": V3_FEATURES, "retained_non_financial_firms": panel.ticker.nunique(),
                "market_rows": len(panel), "counts": counts, "source_ledger": ledger,
                "scope": "All retained non-financial issuers, unchanged company roles and market eligibility. No complete-case deletion, custom-tag backfill, CIK stitching or financial-sector ratio assumptions. Original quantities remain in earlier datasets; V3 indicators are separately defined alternatives.",
                "market_outcomes_2026_accessed": False}
    write_json(dest / "manifest.json", manifest)
    write_json(ROOT / "results/FINANCIAL_FULL_DATA_V1.json", manifest)
    print(pd.DataFrame(counts).to_string(index=False))


if __name__ == "__main__":
    main()
