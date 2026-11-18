"""Publish the metadata pilot without claiming downloaded or validated report text."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.io import read_json


def main():
    frame = pd.read_csv(ROOT / "results/FILING_MANIFEST_PILOT_V1.csv", keep_default_na=False)
    audit = read_json(ROOT / "results/FILING_MANIFEST_AUDIT_PILOT_V1.json")
    if frame.duplicated(["cik", "accession"]).any() or not frame.filing_date.le("2025-12-31").all():
        raise ValueError("Duplicate accession or disallowed disclosure date")
    ready = sum(r["status"] == "ready" for r in audit["companies"])
    lines = ["# Annual-filing metadata pilot", "",
        "Fixed sector/hash selection: one retained training-role issuer in each of ten non-financial sectors. "
        "No model outcomes were used for this selection. Metadata covers filings dated 2013–2025, including "
        "historical submissions batches.", "",
        f"Successfully queried companies: {ready}/{len(audit['companies'])}. Annual filing records: {len(frame)}; "
        f"original 10-K primary-document candidates: {int(frame.primary_text_candidate.sum())}.", "",
        "| Company | Sector | Original 10-K candidates | Other annual records | First candidate filing | Last candidate filing |",
        "|---|---|---:|---:|---|---|"]
    for (ticker, sector), group in frame.groupby(["ticker", "sector"]):
        original = group.loc[group.primary_text_candidate]
        first = original.filing_date.min() if len(original) else "none"
        last = original.filing_date.max() if len(original) else "none"
        lines.append(f"| {ticker} | {sector} | {len(original)} | {len(group)-len(original)} | {first} | {last} |")
    lines += ["", "## What this establishes", "",
        "The recent submissions array alone is not sufficient for the full research history. The runner "
        "queries declared historical batches overlapping the permitted disclosure period, checks duplicate "
        "accession metadata and keeps amendments and foreign/transition forms separate. Original filing text "
        "is not replaced with a later amendment.", "",
        "A primary-document URL is a retrieval candidate, not proof that report text was downloaded or that "
        "Item 7 was correctly extracted. No body-text hashes, MD&A quality review, embeddings or model fitting "
        "are included. Acceptance timestamps are preserved as raw metadata; the eventual date-only join must "
        "conservatively use a later observed trading date. Report-period end is not disclosure time.", "",
        "CRH illustrates a scope issue: foreign annual forms appear in the history, whereas original 10-K "
        "coverage starts later. FOX and AVGO also have shorter CIK-specific histories than a naive "
        "thirteen-year-per-company assumption. Historical entities and reorganizations need review; "
        "do not silently stitch filings under another CIK.", "",
        "Next: retrieve and hash original documents for a small declared quality-review set, verify MD&A "
        "boundaries and extraction failures, then build frozen embeddings only after extraction quality passes. "
        "Preserve foreign forms as explicit scope/exclusions or design a separately validated extraction path. "
        "The 2026 market holdout remains sealed.", "",
        "Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces", ""]
    (ROOT / "results/FILING_MANIFEST_PILOT_V1.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
