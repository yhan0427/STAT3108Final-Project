"""Freeze a role/sector/hash-selected text research cohort before text acquisition."""
from pathlib import Path
import hashlib
from math import floor
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.io import read_json, sha256, utc_now, write_json


def main():
    path = ROOT / "config/multimodal-100-v1.csv"
    plan_path = path.with_suffix(".json")
    if path.exists() or plan_path.exists():
        raise FileExistsError("Preserve frozen multimodal cohort")
    parent = ROOT / "config/full-cohort-v1.csv"
    if sha256(parent) != read_json(parent.with_suffix(".json"))["roster_sha256"]:
        raise ValueError("Frozen parent roster changed")
    roster = pd.read_csv(parent)
    failed = {r["ticker"] for r in read_json(ROOT / "results/FULL_COHORT_AUDIT_V2.json")["failed"]}
    roster = roster.loc[roster.sector.ne("Financials") & ~roster.ticker.isin(failed)].copy()
    selected = []
    for role, budget in [("training_company", 80), ("held_company", 20)]:
        pool = roster.loc[roster.role.eq(role)].copy()
        sizes = pool.groupby("sector").size().to_dict()
        total = sum(sizes.values())
        quotas = {sector: 1 for sector in sizes}
        remaining = budget - len(sizes)
        raw = {sector: remaining * n / total for sector, n in sizes.items()}
        for sector in sizes:
            quotas[sector] += floor(raw[sector])
        leftover = budget - sum(quotas.values())
        for sector in sorted(sizes, key=lambda k: (-(raw[k] - floor(raw[k])), k))[:leftover]:
            quotas[sector] += 1
        pool["sample_rank"] = pool.issuer_id.map(lambda x: hashlib.sha256(f"multimodal-100-v1:{x}".encode()).hexdigest())
        for sector, n in sorted(quotas.items()):
            selected.append(pool.loc[pool.sector.eq(sector)].sort_values("sample_rank").head(n))
    frame = pd.concat(selected).sort_values(["role", "sector", "sample_rank"]).reset_index(drop=True)
    if len(frame) != 100 or frame.cik.nunique() != 100:
        raise ValueError("Sample capacity or duplicate issuer failure")
    frame.to_csv(path, index=False)
    write_json(plan_path, {
        "frozen_utc": utc_now(), "roster_sha256": sha256(path), "source_sha256": sha256(Path(__file__)),
        "parent_roster_sha256": sha256(parent),
        "market_data_manifest_sha256": sha256(ROOT / "data/processed/full-cohort-v2/manifest.json"),
        "scope": "100 fixed retained non-financial reference-cohort issuers: 80 existing training-role, 20 existing held-role. Does not redefine historical membership or untouched status. No substitution for absent financials/text. Sampled for computational feasibility, not model outcomes.",
        "selection": "Within each existing role: at least one issuer per sector; allocate remaining seats by proportional largest remainders with alphabetical ties; fixed SHA256 multimodal-100-v1 issuer rank. No text-availability or performance threshold.",
        "role_counts": frame.role.value_counts().to_dict(),
        "filings": "Original 10-Ks disclosed 2013-01-01 through 2025-12-31; amendments/foreign/transition forms retained as separate metadata, no automatic primary-text substitution.",
        "encoder": "Pinned frozen FinBERT; complete accepted MD&A chunks, content-token weighting; no text-model fitting at corpus stage.",
        "dimensionality_reduction": "If applied, fit on unique documents actually available to training-role training-period origins only; never on all encoded documents.",
        "evaluation": "Four matched information groups market/financial/text/all plus availability/age/length controls. Same eligibility and budget; unchanged 2026 sealing. Full-market and full-financial studies remain separate scopes.",
        "market_outcomes_2026": "sealed"})
    print(frame.groupby(["role", "sector"]).size().to_string())


if __name__ == "__main__":
    main()
