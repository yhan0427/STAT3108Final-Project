"""Fixed sector/hash pilot for annual filing metadata; does not download report text."""
from pathlib import Path
import hashlib
import sys
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.filings import annual_records, historical_files
from volrisk.io import read_json, sha256, utc_now, write_json


def fetch(url, path, session, agent):
    if path.exists():
        if sha256(path) != read_json(path.with_suffix(".source.json"))["sha256"]:
            raise ValueError("Cached submissions checksum mismatch")
        return read_json(path)
    response = session.get(url, headers={"User-Agent": agent}, timeout=45)
    time.sleep(0.3)
    response.raise_for_status()
    data = response.json()
    write_json(path, data)
    write_json(path.with_suffix(".source.json"), {
        "url": url, "sha256": sha256(path), "retrieved_utc": utc_now(),
    })
    return data


def main():
    roster_path = ROOT / "config/full-cohort-v1.csv"
    if sha256(roster_path) != read_json(ROOT / "config/full-cohort-v1.json")["roster_sha256"]:
        raise ValueError("Frozen roster changed")
    market_audit = read_json(ROOT / "results/FULL_COHORT_AUDIT_V2.json")
    failed = {r["ticker"] for r in market_audit["failed"]}
    roster = pd.read_csv(roster_path)
    roster = roster.loc[roster.role.eq("training_company") & roster.sector.ne("Financials")
                        & ~roster.ticker.isin(failed)].copy()
    roster["rank"] = roster.issuer_id.map(
        lambda value: hashlib.sha256(f"filing-pilot-v1:{value}".encode()).hexdigest())
    pilot = roster.sort_values(["sector", "rank"]).groupby("sector", sort=True).head(1)
    dest = ROOT / "data/processed/filing-manifest-pilot-v1"
    dest.mkdir(parents=True, exist_ok=True)
    plan = {"created_utc": utc_now(), "roster_sha256": sha256(roster_path),
            "selection": "One retained training-role issuer per non-financial sector, fixed SHA256 rank; no outcome-based selection",
            "companies": pilot[["ticker", "cik", "sector", "issuer_id"]].to_dict("records"),
            "earliest_filing": "2013-01-01", "last_filing": "2025-12-31",
            "source_sha256": sha256(Path(__file__)),
            "parser_sha256": sha256(ROOT / "src/volrisk/filings.py")}
    plan_path = dest / "plan.json"
    if plan_path.exists():
        existing = read_json(plan_path)
        if {k: v for k, v in existing.items() if k != "created_utc"} != {k: v for k, v in plan.items() if k != "created_utc"}:
            raise ValueError("Pilot plan changed; version this stage")
        plan = existing
    else:
        write_json(plan_path, plan)
    agent = read_json(ROOT / ".local.json")["sec_user_agent"]
    if "@" not in agent:
        raise ValueError("SEC contact identifier missing")
    records, audits = [], []
    session = requests.Session()
    for row in plan["companies"]:
        sources = []
        try:
            cik = int(row["cik"])
            filename = f"CIK{cik:010d}.json"
            path = ROOT / "data/raw/sec-submissions-v1" / filename
            data = fetch(f"https://data.sec.gov/submissions/{filename}", path, session, agent)
            if int(data["cik"]) != cik:
                raise ValueError("SEC submissions entity mismatch")
            blocks = [(path, data)]
            for filename in historical_files(data):
                path = ROOT / "data/raw/sec-submissions-v1" / filename
                blocks.append((path, fetch(f"https://data.sec.gov/submissions/{filename}", path, session, agent)))
            company = []
            for path, block in blocks:
                digest = sha256(path)
                sources.append({"path": str(path.relative_to(ROOT)), "sha256": digest})
                for record in annual_records(block, cik):
                    record.update({k: row[k] for k in ["ticker", "sector", "issuer_id"]})
                    record["submissions_source_sha256"] = digest
                    company.append(record)
            seen = {}
            for record in company:
                key = record["accession"]
                if key in seen:
                    compare = ["form", "filing_date", "report_period", "primary_document"]
                    if any(seen[key][k] != record[k] for k in compare):
                        raise ValueError("Conflicting annual metadata across submissions files")
                seen[key] = record
            records.extend(seen.values())
            audits.append({**row, "status": "ready", "annual_records": len(seen), "sources": sources})
        except (ValueError, requests.RequestException) as exc:
            audits.append({**row, "status": "failed", "reason": str(exc), "sources": sources})
        print(row["ticker"], audits[-1]["status"], audits[-1].get("annual_records", 0), flush=True)
    frame = pd.DataFrame(records)
    if len(frame):
        frame = frame.sort_values(["ticker", "filing_date", "accession"])
        frame.to_csv(ROOT / "results/FILING_MANIFEST_PILOT_V1.csv", index=False)
    write_json(ROOT / "results/FILING_MANIFEST_AUDIT_PILOT_V1.json", {
        "updated_utc": utc_now(), "plan": plan, "companies": audits,
        "annual_records": len(frame),
        "primary_10k_candidates": int(frame.primary_text_candidate.sum()) if len(frame) else 0,
        "report_texts_downloaded": 0, "embeddings_computed": 0,
        "market_outcomes_2026_accessed": False,
        "limits": "Metadata only, not text hashes or extraction validation. Amendments/foreign/transition forms retained separately; not substituted for original 10-K. Current SEC metadata is not an immutable historical archive. No report-period date is used as disclosure time.",
    })


if __name__ == "__main__":
    main()
