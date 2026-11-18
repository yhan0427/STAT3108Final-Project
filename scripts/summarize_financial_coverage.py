"""Summarize availability against full frozen denominators, including SEC failures."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.financial_coverage import AUDIT_FEATURES, CONTENT_FEATURES
from volrisk.fundamentals import align_snapshots
from volrisk.fundamentals_v3 import V3_FEATURES
from volrisk.io import read_json, sha256, utc_now, write_json


def main():
    audit_path = ROOT / "results/FINANCIAL_COVERAGE_AUDIT_V1.json"
    audit = read_json(audit_path)
    if not audit["complete"]:
        raise ValueError("Wait for collection to finish; do not treat partial coverage as final")
    panel_path = ROOT / "data/processed/full-cohort-v2/panel.pkl"
    if sha256(panel_path) != audit["plan"]["market_panel_sha256"]:
        raise ValueError("Market denominator changed")
    for record in audit["companies"]:
        if record["status"] == "ready":
            for key in ["source", "snapshots", "coverage"]:
                if sha256(ROOT / record[f"{key}_path"]) != record[f"{key}_sha256"]:
                    raise ValueError(f"Cached {key} changed for {record['ticker']}")
    legacy_checks = []
    for path in sorted((ROOT / "data/processed/main").glob("*_fundamental_snapshots.csv")):
        ticker = path.name.split("_fundamental_snapshots")[0]
        record = next((r for r in audit["companies"]
                       if r["ticker"] == ticker and r["status"] == "ready"), None)
        if record is None:
            continue
        previous = pd.read_csv(path)
        current = pd.read_csv(ROOT / record["snapshots_path"])
        pd.testing.assert_frame_equal(previous, current[previous.columns], check_dtype=False)
        legacy_checks.append({"ticker": ticker, "legacy_snapshot_sha256": sha256(path),
                              "current_snapshot_sha256": record["snapshots_sha256"]})
    cols = ["ticker", "date", "split", "sector", "role", "context_ready"]
    panel = pd.read_pickle(panel_path)[cols]
    panel = panel.loc[panel.split.ne("excluded")].copy()
    successful = {r["ticker"]: r for r in audit["companies"] if r["status"] == "ready"}
    rows = []
    legacy = [f for f in AUDIT_FEATURES[:8] if f != "filing_age_days"]
    v3 = [f for f in V3_FEATURES if f != "filing_age_days"]
    sets = {"legacy_v1_content": legacy, "alternative_v3_content": v3}
    for ticker, origins in panel.groupby("ticker", sort=True):
        origins = origins.sort_values("date").reset_index(drop=True)
        flags = pd.DataFrame(False, index=origins.index, columns=AUDIT_FEATURES)
        record = successful.get(ticker)
        if record:
            snap_path = ROOT / record["snapshots_path"]
            if sha256(snap_path) != record["snapshots_sha256"]:
                raise ValueError(f"Snapshot changed: {ticker}")
            snaps = pd.read_csv(snap_path, parse_dates=["filing_date", "report_end"])
            aligned = align_snapshots(origins.date, snaps).reset_index(drop=True)
            aligned.loc[aligned.filing_age_days.gt(180), CONTENT_FEATURES] = np.nan
            flags = pd.DataFrame(np.isfinite(aligned[AUDIT_FEATURES].to_numpy(float)),
                                 index=origins.index, columns=AUDIT_FEATURES)
        counts = {f: flags[f] for f in AUDIT_FEATURES}
        for name, features in sets.items():
            counts[f"{name}_all_available"] = flags[features].all(axis=1)
            counts[f"{name}_any_available"] = flags[features].any(axis=1)
        for split, positions in origins.groupby("split").groups.items():
            for context in ["60_market_sessions", "512_rv20_common"]:
                idx = origins.loc[positions].index
                if context == "512_rv20_common":
                    idx = idx[origins.loc[idx, "context_ready"].eq(True)]
                if not len(idx):
                    continue
                for feature, mask in counts.items():
                    rows.append({"ticker": ticker, "sector": origins.sector.iloc[0],
                                 "role": origins.role.iloc[0], "split": split, "context": context,
                                 "feature": feature, "eligible_origins": len(idx),
                                 "available_origins": int(mask.loc[idx].sum()),
                                 "sec_snapshot_ready": record is not None})
    company = pd.DataFrame(rows)
    company.to_csv(ROOT / "results/FINANCIAL_COVERAGE_MATCHING_V1.csv", index=False)
    frames = []
    for group, selected in [
        ("all_retained", company),
        ("non_financial", company.loc[company.sector.ne("Financials")]),
        ("financial", company.loc[company.sector.eq("Financials")]),
    ]:
        grouped = selected.groupby(["role", "split", "context", "feature"], as_index=False).agg(
            eligible_origins=("eligible_origins", "sum"),
            available_origins=("available_origins", "sum"), companies=("ticker", "nunique"))
        grouped["group"] = group
        grouped["coverage"] = grouped.available_origins / grouped.eligible_origins
        frames.append(grouped)
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(ROOT / "results/FINANCIAL_COVERAGE_GROUPS_V1.csv", index=False)
    lines = ["# Financial availability and matched-sample feasibility", "",
             "Coverage diagnostics only; no financial models or text embeddings fitted. "
             "Company-year percentages in FINANCIAL_COVERAGE_V1.md are supplemented here with "
             "split, company-role, context and joint-availability comparisons. Failed SEC companies "
             "remain in every applicable market denominator.", "",
             "## Non-financial training companies: 60-session origins", "",
             "| Period | Feature | Available origins | Eligible origins | Coverage |",
             "|---|---|---:|---:|---:|"]
    selected = summary.loc[summary.group.eq("non_financial")
                           & summary.role.eq("training_company")
                           & summary.context.eq("60_market_sessions")]
    for row in selected.itertuples():
        lines.append(f"| {row.split} | {row.feature} | {row.available_origins:,} | "
                     f"{row.eligible_origins:,} | {row.coverage:.1%} |")
    lines += ["", "## Scope", "",
        "The V1 and V3 content sets are inherited from the original financial studies; this audit "
        "does not choose a new economic schema. All-available coverage is a diagnostic, not a decision "
        "to delete incomplete rows. Any-available coverage helps distinguish missing whole snapshots "
        "from missing individual fields. Financial companies are reported separately; the same "
        "accounting ratios are not assumed comparable across sectors.", "",
        "Market-only controls must be rerun on the exact origins used by each future multimodal "
        "comparison. The 512-rv20 subset is separately reported for compatibility with the existing "
        "foundation-model context requirement. A coverage-selected subset can change the population; "
        "retain full-coverage market results as a separate benchmark.", "",
        "Four-way market/financial/text/all comparisons additionally require a validated filing-document "
        "manifest and text extraction. Structured coverage alone does not establish text availability. "
        "No predictive benefit or final sample size is established here. 2026 market outcomes remain sealed.", ""]
    train = selected.loc[selected.split.eq("train")].set_index("feature")
    old = train.loc["legacy_v1_content_all_available", "coverage"]
    new = train.loc["alternative_v3_content_all_available", "coverage"]
    lines += ["## Next design decision", "",
        f"Among non-financial training-company origins, all seven legacy content fields are available "
        f"on {old:.1%} of origins; the seven V3 alternative content fields on {new:.1%}. "
        "These ratios describe availability, not predictive improvement. Dropping every incomplete row "
        "would discard a large share of the declared training population. Retain missingness and filing "
        "age as explicit controls, with imputation/clipping/scaling learned only on training-role, "
        "training-period observations.", "",
        "Before financial fusion, register the economic schema and verify a stratified set of source "
        "values and entity histories. Preserve the original quantities as separately named fields; "
        "V3 alternatives are not backfilled values for total liabilities or operating margin. Compare "
        "market-only, market+availability/age controls, and market+financial content with the same "
        "origins and backbone. Add text to the four-way matched design only after document extraction "
        "is validated. The historical evaluation remains development evidence.", ""]
    (ROOT / "results/FINANCIAL_MATCHING_V1.md").write_text("\n".join(lines))
    write_json(ROOT / "results/FINANCIAL_MATCHING_AUDIT_V1.json", {
        "completed_utc": utc_now(), "collection_audit_sha256": sha256(audit_path),
        "source_sha256": sha256(Path(__file__)), "market_panel_sha256": sha256(panel_path),
        "content_sets": sets, "market_outcomes_2026_accessed": False, "models_fitted": False,
        "company_rows": len(company), "all_retained_eligible_origins": len(panel),
        "source_snapshot_coverage_hashes_verified": len(successful),
        "legacy_snapshot_exact_agreement": legacy_checks,
    })
    print(selected[["split", "feature", "coverage"]].to_string(index=False))


if __name__ == "__main__":
    main()
