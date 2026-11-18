"""Select the preregistered bulk quality sample; keep filing excerpts local only."""
from pathlib import Path
import hashlib
import sys
import warnings

import pandas as pd
from bs4 import XMLParsedAsHTMLWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.io import read_json, sha256, utc_now, write_json
from volrisk.mda import visible_text


def main():
    warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
    config_path = ROOT / "config/text-quality-100-v1.json"
    cfg = read_json(config_path)
    audit_path = ROOT / "results/TEXT_CORPUS_100_AUDIT_V1.json"
    audit = read_json(audit_path)
    if not audit["complete"]:
        raise ValueError("Complete the frozen corpus before freezing the final quality sample")
    if sha256(ROOT / "src/volrisk/mda.py") != cfg["extractor_sha256"]:
        raise ValueError("Frozen extraction rules changed")
    manifest_path = ROOT / "data/processed/text-corpus-100-v1/manifest.json"
    if sha256(manifest_path) != audit["local_manifest_sha256"]:
        raise ValueError("Corpus manifest changed")
    data = read_json(manifest_path)
    frame = pd.DataFrame(data["documents"])
    roster = pd.read_csv(ROOT / "config/multimodal-100-v1.csv").set_index("ticker")
    if frame.duplicated(["cik", "accession"]).any() or not frame.filing_date.between("2013-01-01", "2025-12-31").all():
        raise ValueError("Duplicate keys or disallowed disclosures")
    if not frame.form.eq("10-K").all():
        raise ValueError("Non-original form in primary corpus")
    for row in frame.itertuples():
        expected = roster.loc[row.ticker]
        if row.issuer_id != expected.issuer_id or row.role != expected.role or row.cik != expected.cik:
            raise ValueError("Cross-issuer/role document assignment")
    candidates = frame.loc[frame.status.eq("candidate_needs_review")].copy()
    candidates["quality_flags"] = candidates.quality_flags.fillna("")
    candidates["filing_year"] = pd.to_datetime(candidates.filing_date).dt.year
    candidates["review_rank"] = candidates.apply(lambda r: hashlib.sha256(
        f"text-quality-100-v1:{r.issuer_id}:{r.accession}".encode()).hexdigest(), axis=1)
    flagged = candidates.loc[candidates.quality_flags.ne("")].copy()
    flagged["selection_reason"] = "all_flagged"
    ordinary = candidates.loc[candidates.quality_flags.eq("")].sort_values("review_rank")
    ordinary = ordinary.groupby(["sector", "filing_year"], sort=True).head(1).copy()
    ordinary["selection_reason"] = "sector_year_hash_sample"
    sample = pd.concat([flagged, ordinary]).drop_duplicates(["cik", "accession"]).sort_values(["sector", "filing_year", "review_rank"])
    sample_path = ROOT / "results/TEXT_QUALITY_SAMPLE_100_V1.csv"
    sample.to_csv(sample_path, index=False)
    dest = ROOT / "data/processed/text-quality-100-v1"
    dest.mkdir(parents=True, exist_ok=True)
    plan = {"registered_config_sha256": sha256(config_path), "manifest_sha256": sha256(manifest_path),
            "sample_sha256": sha256(sample_path), "source_sha256": sha256(Path(__file__)),
            "sampled_ordinary": len(ordinary), "all_flagged": len(flagged), "total_selected": len(sample)}
    plan_path = dest / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Final quality sample changed")
    write_json(plan_path, plan)
    excerpts = []
    for row in sample.astype(object).where(sample.notna(), None).to_dict("records"):
        source, text = ROOT / row["document_path"], ROOT / row["text_path"]
        if sha256(source) != row["document_sha256"] or sha256(text) != row["text_sha256"]:
            raise ValueError("Selected original or section changed")
        full = visible_text(source.read_bytes())
        body = text.read_text()
        if full[int(row["start_char"]):int(row["end_char"])].strip() != body:
            raise ValueError("Stored section does not match source offsets")
        middle = len(body) // 2
        excerpts.append({**row, "before": full[max(0, int(row["start_char"]) - 80):int(row["start_char"])],
                         "start": body[:180], "middle": body[middle:middle + 120],
                         "tail": body[-100:], "after": full[int(row["end_char"]):int(row["end_char"]) + 260]})
    write_json(dest / "review_excerpts.json", {"plan": plan, "documents": excerpts})
    write_json(ROOT / "results/TEXT_QUALITY_SAMPLE_AUDIT_100_V1.json", {
        "created_utc": utc_now(), "plan": plan, "candidate_documents": len(candidates),
        "candidate_companies": candidates.ticker.nunique(),
        "review_complete": False, "accepted_for_training": 0,
        "excerpts_local_only": True, "market_outcomes_2026_accessed": False,
        "selection": "All flagged candidates plus min issuer/accession SHA per sector/filing year among ordinary candidates; no forecast outcomes or losses used."})
    print("Quality sample:", len(sample), "documents;", len(flagged), "flagged;", len(ordinary), "ordinary sector/year checks")


if __name__ == "__main__":
    main()
