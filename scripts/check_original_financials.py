"""Compare a declared original-report subset with cached SEC facts and snapshot ratios."""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from bs4 import XMLParsedAsHTMLWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.inline_facts import extract_usd_facts
from volrisk.io import read_json, sha256, utc_now, write_json

TAGS = {"assets": ["Assets"], "parent_equity": ["StockholdersEquity"],
        "cash": ["CashAndCashEquivalentsAtCarryingValue"], "liabilities": ["Liabilities"],
        "current_assets": ["AssetsCurrent"], "current_liabilities": ["LiabilitiesCurrent"],
        "income": ["NetIncomeLoss", "ProfitLoss"], "cfo": ["NetCashProvidedByUsedInOperatingActivities"]}
RATIOS = {"parent_equity_to_assets": ("parent_equity", "assets"),
          "cash_to_assets": ("cash", "assets"), "liabilities_to_assets": ("liabilities", "assets"),
          "current_liabilities_to_assets": ("current_liabilities", "assets"),
          "current_ratio": ("current_assets", "current_liabilities"),
          "income_to_assets": ("income", "assets"), "cfo_to_assets": ("cfo", "assets")}


def main():
    warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)  # Inline XHTML declarations use the tolerant HTML parser.
    downloads_path = ROOT / "results/FILING_TEXT_DOWNLOAD_PILOT_V1.json"
    coverage_path = ROOT / "results/FINANCIAL_COVERAGE_AUDIT_V1.json"
    downloads = read_json(downloads_path)
    sources = {r["ticker"]: r for r in read_json(coverage_path)["companies"] if r["status"] == "ready"}
    records, document_audits = [], []
    for doc in downloads["documents"]:
        if doc["status"] != "downloaded":
            document_audits.append({"ticker": doc["ticker"], "accession": doc["accession"], "status": "download_failed"})
            continue
        path = ROOT / doc["path"]
        if sha256(path) != doc["sha256"]:
            raise ValueError("Original report changed")
        original = pd.DataFrame(extract_usd_facts(path.read_bytes(), doc["cik"]))
        base = {k: doc[k] for k in ["ticker", "cik", "accession", "filing_date", "report_period", "document_url"]}
        base["document_sha256"] = doc["sha256"]
        if original.empty:
            document_audits.append({**base, "status": "no_supported_inline_usd_facts"})
            continue
        source = sources[doc["ticker"]]
        source_path = ROOT / source["source_path"]
        if sha256(source_path) != source["source_sha256"]:
            raise ValueError("Cached Company Facts changed")
        payload = read_json(source_path)
        snapshots = pd.read_csv(ROOT / source["snapshots_path"])
        event = snapshots.loc[snapshots.filing_date.eq(doc["filing_date"]) & snapshots.report_end.eq(doc["report_period"])]
        selected = {}
        for concept, candidates in TAGS.items():
            subset = original.loc[original.tag.isin(candidates) & original.end.eq(doc["report_period"])].copy()
            if concept in {"income", "cfo"}:
                duration = (pd.to_datetime(subset.end) - pd.to_datetime(subset.start)).dt.days + 1
                subset = subset.loc[duration.between(330, 380)]
            else:
                subset = subset.loc[subset.start.eq("")]
            if subset.empty:
                records.append({**base, "check": "original_vs_api", "field": concept, "status": "original_field_unavailable"})
                continue
            tag = next(t for t in candidates if t in set(subset.tag))
            subset = subset.loc[subset.tag.eq(tag)]
            if subset.value.nunique() != 1 or subset.start.nunique() != 1:
                records.append({**base, "check": "original_vs_api", "field": concept, "status": "ambiguous_original_contexts"})
                continue
            fact = subset.iloc[0]
            selected[concept] = float(fact.value)
            known = [r["val"] for r in payload.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])
                     if r.get("accn") == doc["accession"] and r.get("end") == fact.end
                     and r.get("start", "") == fact.start and r.get("filed", "9999-12-31") <= "2025-12-31"]
            expected = float(fact.value)
            status = "match" if known and all(np.isclose(v, expected, rtol=1e-10, atol=.01) for v in known) else ("api_fact_unavailable" if not known else "mismatch")
            records.append({**base, "check": "original_vs_api", "field": concept, "tag": tag,
                            "status": status, "original_value": expected,
                            "cached_api_value": float(known[0]) if known else None, "fact_id": fact.fact_id,
                            "context_id": fact.context_id, "scale": int(fact.scale), "source_sha256": source["source_sha256"]})
        for feature, (numerator, denominator) in RATIOS.items():
            if numerator not in selected or denominator not in selected or selected[denominator] <= 0:
                continue
            expected = selected[numerator] / selected[denominator]
            actual = float(event.iloc[0][feature]) if len(event) == 1 else np.nan
            status = "snapshot_unavailable" if not np.isfinite(actual) else ("match" if np.isclose(actual, expected, rtol=1e-10, atol=1e-12) else "mismatch")
            records.append({**base, "check": "original_ratio_vs_snapshot", "field": feature, "status": status,
                            "original_value": expected, "snapshot_value": actual if np.isfinite(actual) else None,
                            "snapshot_sha256": source["snapshots_sha256"]})
        document_audits.append({**base, "status": "original_inline_fields_checked", "supported_usd_facts": len(original)})
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / "results/FINANCIAL_ORIGINAL_CHECKS_V1.csv", index=False)
    counts = frame.groupby(["check", "status"]).size().rename("n").reset_index()
    write_json(ROOT / "results/FINANCIAL_ORIGINAL_CHECKS_V1.json", {
        "completed_utc": utc_now(), "source_sha256": sha256(Path(__file__)),
        "parser_sha256": sha256(ROOT / "src/volrisk/inline_facts.py"),
        "downloads_sha256": sha256(downloads_path), "coverage_audit_sha256": sha256(coverage_path),
        "documents": document_audits, "counts": counts.to_dict("records"),
        "scope": "Fixed oldest/latest 20-document original SEC quality set, consolidated USD us-gaap fields only. No guessed transforms or dimensional sums. Sparse coverage is explicit; this is not full-cohort certification or a conformant XBRL validator.",
        "market_outcomes_2026_accessed": False,
        "reference": "https://www.xbrl.org/specification/inlinexbrl-part0/rec-2013-11-18/inlinexbrl-part0-rec-2013-11-18.html",
    })
    lines = ["# Original-filing financial spot checks", "",
             "Fixed original-report quality set; no documents selected by check success or forecasting outcomes. "
             "Original inline XBRL values are parsed independently from the Company Facts feature builder, "
             "with explicit CIK, non-dimensional context, USD unit, reporting period, scale and sign.", "",
             "| Check | Status | Count |", "|---|---|---:|"]
    for row in counts.itertuples():
        lines.append(f"| {row.check} | {row.status} | {row.n} |")
    lines += ["", "Original documents without supported inline facts remain listed in the JSON audit. "
              "Missing or ambiguous tags are not invented. These checks cover the named fields and documents "
              "only; they do not certify every historical feature, issuer transition, custom taxonomy or immutable "
              "point-in-time archive. Any mismatch must be investigated before those fields enter a new fit.", ""]
    (ROOT / "results/FINANCIAL_ORIGINAL_CHECKS_V1.md").write_text("\n".join(lines))
    print(counts.to_string(index=False))
    if frame.status.eq("mismatch").any():
        print(frame.loc[frame.status.eq("mismatch"), ["ticker", "accession", "check", "field", "original_value", "snapshot_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
