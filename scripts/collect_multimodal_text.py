"""Resumable original annual-report corpus for a frozen, outcome-independent cohort."""
from pathlib import Path
import sys
import time
import warnings

import pandas as pd
import requests
from bs4 import XMLParsedAsHTMLWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_filing_manifest import fetch
from volrisk.filings import annual_records, historical_files
from volrisk.mda import extract_mda
from volrisk.io import read_json, sha256, utc_now, write_json


def publish(dest, plan, companies, documents):
    write_json(dest / "manifest.json", {"plan": plan, "companies": companies, "documents": documents})
    frame = pd.DataFrame(documents)
    if len(frame):
        frame.to_csv(ROOT / "results/TEXT_CORPUS_100_V1.csv", index=False)
    write_json(ROOT / "results/TEXT_CORPUS_100_AUDIT_V1.json", {
        "updated_utc": utc_now(), "plan": plan, "companies": companies,
        "companies_attempted": len(companies), "companies_planned": 100,
        "complete": len(companies) == 100,
        "original_documents": len(documents),
        "extraction_statuses": frame.status.value_counts().to_dict() if len(frame) else {},
        "local_manifest_sha256": sha256(dest / "manifest.json"),
        "accepted_for_training": 0, "models_fitted": False, "market_outcomes_2026_accessed": False,
        "scope": "Original annual 10-K metadata and frozen-parser candidates only. Every bulk candidate is unreviewed until the declared quality process. Foreign forms/amendments, extraction abstentions and failures are explicit, with no replacement or invented text.",
    })


def cached_report(doc, session, agent):
    paths = [ROOT / folder / str(doc["cik"]) / f"{doc['accession']}.html" for folder in [
        "data/raw/sec-filings-corpus-v1", "data/raw/sec-filings-pilot-v1",
        "data/raw/sec-filings-validation-v1", "data/raw/sec-filings-validation-v2"]]
    for path in paths:
        if path.exists():
            source = read_json(path.with_suffix(".source.json"))
            if source["sha256"] != sha256(path) or source["url"] != doc["document_url"]:
                raise ValueError("Cached report identity/hash changed")
            return path
    response = session.get(doc["document_url"], headers={"User-Agent": agent}, timeout=60)
    time.sleep(0.4)
    response.raise_for_status()
    if "html" not in response.headers.get("Content-Type", "").lower() or len(response.content) < 10000:
        raise ValueError("Expected original report HTML")
    path = paths[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    write_json(path.with_suffix(".source.json"), {
        "url": doc["document_url"], "sha256": sha256(path), "retrieved_utc": utc_now(),
        "cik": doc["cik"], "accession": doc["accession"], "content_type": response.headers.get("Content-Type")})
    return path


def main():
    warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
    roster_path = ROOT / "config/multimodal-100-v1.csv"
    cohort = read_json(roster_path.with_suffix(".json"))
    if sha256(roster_path) != cohort["roster_sha256"]:
        raise ValueError("Frozen text cohort changed")
    freeze_path = ROOT / "config/mda-extractor-v6.json"
    frozen = read_json(freeze_path)
    if sha256(ROOT / "src/volrisk/mda.py") != frozen["source_sha256"]:
        raise ValueError("Frozen parser changed")
    reviewed = read_json(ROOT / "results/MDA_VALIDATION_REVIEW_V2.json")
    if not reviewed["review_complete"] or reviewed["wrong_candidate_boundaries_observed"] != 0:
        raise ValueError("Resolve frozen-parser quality review before bulk collection")
    dest = ROOT / "data/processed/text-corpus-100-v1"
    dest.mkdir(parents=True, exist_ok=True)
    plan = {"registered_utc": utc_now(), "cohort_sha256": sha256(roster_path),
            "extractor_sha256": frozen["source_sha256"], "parser_review_sha256": sha256(ROOT / "results/MDA_VALIDATION_REVIEW_V2.json"),
            "runner_sha256": sha256(Path(__file__)),
            "metadata_parser_sha256": sha256(ROOT / "src/volrisk/filings.py"),
            "metadata_helper_sha256": sha256(ROOT / "scripts/build_filing_manifest.py"),
            "earliest_filing": "2013-01-01", "cutoff": "2025-12-31",
            "extraction": "Original 10-K only; frozen v6 standalone Item 7 to Item 7A/8 candidates; no silent full-report fallback",
            "quality_flags": "Multiple body candidates; <1500 or >30000 words; end heading Item 8; source fraction >0.65 or start fraction <0.03. Flagged candidates require explicit review before any fit.",
            "bulk_review": "All flagged candidates plus an outcome-independent sector/year/hash sample of ordinary candidates before supervised fitting; no automatic claim of full-corpus semantic correctness.",
            "scope": cohort["scope"]}
    plan_path = dest / "plan.json"
    if plan_path.exists():
        old = read_json(plan_path)
        if {k: v for k, v in old.items() if k != "registered_utc"} != {k: v for k, v in plan.items() if k != "registered_utc"}:
            raise ValueError("Corpus inputs/code changed; version the corpus")
        plan = old
    else:
        write_json(plan_path, plan)
    session = requests.Session()
    agent = read_json(ROOT / ".local.json")["sec_user_agent"]
    if "@" not in agent:
        raise ValueError("Identifying SEC contact required")
    documents, companies = [], []
    blocked = 0
    roster = pd.read_csv(roster_path).sort_values("ticker")
    for row in roster.to_dict("records"):
        ticker = row["ticker"]
        completed = dest / f"{ticker}_company.json"
        if completed.exists():
            done = read_json(completed)
            for record in done["documents"]:
                for key in ["document", "text"]:
                    if record.get(f"{key}_path") and sha256(ROOT / record[f"{key}_path"]) != record[f"{key}_sha256"]:
                        raise ValueError("Completed corpus artifact changed")
            companies.append(done["company"])
            documents += done["documents"]
            continue
        company = {k: row[k] for k in ["ticker", "cik", "issuer_id", "sector", "role"]}
        company_docs = []
        try:
            name = f"CIK{int(row['cik']):010d}.json"
            path = ROOT / "data/raw/sec-submissions-v1" / name
            current = fetch(f"https://data.sec.gov/submissions/{name}", path, session, agent)
            if int(current["cik"]) != int(row["cik"]):
                raise ValueError("SEC entity mismatch")
            metadata, source_files = annual_records(current, row["cik"]), {name: sha256(path)}
            for name in historical_files(current):
                path = ROOT / "data/raw/sec-submissions-v1" / name
                metadata += annual_records(fetch(f"https://data.sec.gov/submissions/{name}", path, session, agent), row["cik"])
                source_files[name] = sha256(path)
            table = pd.DataFrame(metadata)
            write_json(dest / f"{ticker}_annual_metadata.json", {"sources": source_files, "records": metadata})
            originals = table.loc[table.primary_text_candidate].drop_duplicates("accession").sort_values(["filing_date", "accession"]) if len(table) else pd.DataFrame()
            company.update(status="metadata_ready", annual_forms=len(table), original_10ks=len(originals),
                           sources=source_files, other_annual_forms=int((~table.primary_text_candidate).sum()) if len(table) else 0)
            for doc in originals.to_dict("records"):
                record = {**company, **doc}
                for key in ["sources", "annual_forms", "original_10ks", "other_annual_forms", "primary_text_candidate"]:
                    record.pop(key, None)
                checkpoint = dest / f"{doc['accession']}.json"
                if checkpoint.exists():
                    saved = read_json(checkpoint)
                    if saved.get("document_path") and sha256(ROOT / saved["document_path"]) != saved["document_sha256"]:
                        raise ValueError("Cached corpus document changed")
                    if saved.get("text_path") and sha256(ROOT / saved["text_path"]) != saved["text_sha256"]:
                        raise ValueError("Cached corpus text changed")
                    company_docs.append(saved)
                    continue
                try:
                    path = cached_report(doc, session, agent)
                    result = extract_mda(path.read_bytes())
                    text = result.pop("text")
                    record.update({k: v for k, v in result.items() if not k.endswith("preview") and k != "before_start"},
                                  document_path=str(path.relative_to(ROOT)), document_sha256=sha256(path))
                    if text:
                        text_path = dest / f"{doc['accession']}.txt"
                        text_path.write_text(text)
                        flags = []
                        if result["body_candidate_count"] > 1:
                            flags.append("multiple_body_candidates")
                        if not 1500 <= result["section_words"] <= 30000:
                            flags.append("atypical_word_count")
                        if result["end_heading"].lower().startswith("item 8"):
                            flags.append("item8_end")
                        if result["section_chars"] / result["visible_document_chars"] > .65 or result["start_char"] / result["visible_document_chars"] < .03:
                            flags.append("atypical_source_fraction")
                        record.update(text_path=str(text_path.relative_to(ROOT)), text_sha256=sha256(text_path),
                                      quality_flags=";".join(flags), quality_review_status="unreviewed_bulk_candidate")
                    blocked = 0
                except (ValueError, requests.RequestException) as exc:
                    record.update(status="download_or_extraction_failed", reason=str(exc))
                    if isinstance(exc, requests.RequestException):
                        blocked += 1
                write_json(checkpoint, record)
                company_docs.append(record)
                if blocked >= 3:
                    break
        except (ValueError, requests.RequestException) as exc:
            company.update(status="metadata_failed", reason=str(exc))
            if isinstance(exc, requests.RequestException):
                blocked += 1
        companies.append(company)
        documents += company_docs
        if blocked < 3:
            write_json(completed, {"company": company, "documents": company_docs})
        publish(dest, plan, companies, documents)
        counts = pd.Series([r["status"] for r in company_docs]).value_counts().to_dict()
        print(len(companies), ticker, company["status"], len(company_docs), counts, flush=True)
        if blocked >= 3:
            print("Stopped after three consecutive request failures; incomplete corpus remains explicit", flush=True)
            break
    publish(dest, plan, companies, documents)


if __name__ == "__main__":
    main()
