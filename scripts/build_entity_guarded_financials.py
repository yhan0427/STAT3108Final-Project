"""Versioned sensitivity panel; never overwrite raw predecessor-series experiments."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.entity_spells import current_spell_windows, restrict_spell_disclosures
from volrisk.financial_coverage import audit_snapshots
from volrisk.fundamentals import align_snapshots
from volrisk.fundamentals_v3 import V3_FEATURES
from volrisk.io import read_json, sha256, utc_now, write_json


def main():
    parent = ROOT / "data/processed/financial-full-v1/panel.pkl"
    source_manifest = read_json(parent.parent / "manifest.json")
    if sha256(parent) != source_manifest["panel_sha256"]:
        raise ValueError("Parent financial panel changed")
    policy_path = ROOT / "config/entity-spells-v1.json"
    policy = read_json(policy_path)
    dest = ROOT / "data/processed/financial-entity-v1"
    if dest.exists():
        raise FileExistsError("Preserve the identity-sensitivity dataset")
    dest.mkdir(parents=True)
    panel = pd.read_pickle(parent)
    if panel.date.max() >= pd.Timestamp("2026-01-01"):
        raise ValueError("Sealed data found")
    coverage = read_json(ROOT / "results/FINANCIAL_COVERAGE_AUDIT_V1.json")
    sources = {r["ticker"]: r for r in coverage["companies"] if r["status"] == "ready"}
    records = []
    for ticker, spell in policy["spells"].items():
        group = panel.loc[panel.ticker.eq(ticker)]
        if group.empty:
            continue
        source = sources[ticker]
        if int(source["cik"]) != int(spell["cik"]):
            raise ValueError("Policy CIK mismatch")
        path = ROOT / source["source_path"]
        if sha256(path) != source["source_sha256"]:
            raise ValueError("Raw SEC source changed")
        payload = restrict_spell_disclosures(read_json(path), spell["start"])
        snaps = audit_snapshots(payload)
        snaps.to_csv(dest / f"{ticker}_snapshots.csv", index=False)
        aligned = align_snapshots(group.date, snaps)
        aligned.loc[aligned.filing_age_days.gt(180), V3_FEATURES[:-1]] = np.nan
        panel.loc[group.index, V3_FEATURES] = aligned[V3_FEATURES].to_numpy()
        panel.loc[group.index, "financial_filing_date"] = aligned.filing_date.to_numpy()
        market_ok, foundation_ok = current_spell_windows(group.date, spell["start"])
        prior = group.split.ne("excluded")
        excluded = group.loc[~market_ok & prior]
        panel.loc[group.index[~market_ok], "split"] = "excluded"
        panel.loc[group.index[~foundation_ok], "context_ready"] = False
        records.append({"ticker": ticker, "start": spell["start"], "role": group.role.iloc[0],
                        "excluded_origins_by_period": excluded.split.value_counts().to_dict(),
                        "first_price": str(group.date.min().date()),
                        "first_post_guard_origin": str(panel.loc[group.index[market_ok], "date"].min().date()) if market_ok.any() else None,
                        "snapshot_sha256": sha256(dest / f"{ticker}_snapshots.csv"),
                        "sources": spell["sources"]})
    panel.to_pickle(dest / "panel.pkl")
    counts = []
    for (role, split), group in panel.loc[panel.split.ne("excluded")].groupby(["role", "split"]):
        common = group.loc[group.context_ready.eq(True)]
        counts.append({"role": role, "split": split, "firms_context60": group.ticker.nunique(),
                       "origins_context60": len(group), "firms_context512": common.ticker.nunique(),
                       "origins_context512": len(common)})
    manifest = {"built_utc": utc_now(), "parent_panel_sha256": sha256(parent),
                "policy_sha256": sha256(policy_path), "source_sha256": sha256(Path(__file__)),
                "panel_sha256": sha256(dest / "panel.pkl"), "features": V3_FEATURES,
                "retained_non_financial_firms": panel.ticker.nunique(), "market_rows": len(panel),
                "counts": counts, "changes": records,
                "scope": "Conservative documented identity-spell sensitivity, not exhaustive entity certification. Raw and earlier predecessor-reference results preserved. No performance-based company replacement or date choice.",
                "market_outcomes_2026_accessed": False}
    write_json(dest / "manifest.json", manifest)
    write_json(ROOT / "results/FINANCIAL_ENTITY_DATA_V1.json", manifest)
    print(pd.DataFrame(counts).to_string(index=False))


if __name__ == "__main__":
    main()
