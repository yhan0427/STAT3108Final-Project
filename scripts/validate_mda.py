"""New-company frozen-parser quality sample, without forecast outcomes or auto approval."""
from pathlib import Path
import argparse
import hashlib
import sys
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_filing_manifest import fetch
from volrisk.filings import annual_records, historical_files
from volrisk.mda import extract_mda
from volrisk.io import read_json, sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()
    version = args.round
    parser_version = "v5" if version == "v1" else "v6"
    frozen_path = ROOT / f"config/mda-extractor-{parser_version}.json"
    frozen = read_json(frozen_path)
    if frozen["source_sha256"] != sha256(ROOT / "src/volrisk/mda.py"):
        raise ValueError("Frozen extraction rules changed")
    roster_path = ROOT / "config/full-cohort-v1.csv"
    if sha256(roster_path) != read_json(ROOT / "config/full-cohort-v1.json")["roster_sha256"]:
        raise ValueError("Frozen roster changed")
    pilot = read_json(ROOT / "data/processed/filing-manifest-pilot-v1/plan.json")
    excluded = {r["ticker"] for r in pilot["companies"]}
    if version == "v2":
        previous = read_json(ROOT / "data/processed/mda-validation-v1/plan.json")
        excluded.update(r["ticker"] for r in previous["companies"])
    excluded.update(r["ticker"] for r in read_json(ROOT / "results/FULL_COHORT_AUDIT_V2.json")["failed"])
    roster = pd.read_csv(roster_path)
    roster = roster.loc[roster.role.eq("training_company") & roster.sector.ne("Financials")
                        & ~roster.ticker.isin(excluded)].copy()
    roster["rank"] = roster.issuer_id.map(lambda x: hashlib.sha256(f"filing-pilot-v1:{x}".encode()).hexdigest())
    chosen = roster.sort_values(["sector", "rank"]).groupby("sector", sort=True).head(1)
    dest = ROOT / f"data/processed/mda-validation-{version}"
    dest.mkdir(parents=True, exist_ok=True)
    plan = {"registered_utc": utc_now(), "roster_sha256": sha256(roster_path),
            "extractor_sha256": frozen["source_sha256"], "frozen_config_sha256": sha256(frozen_path),
            "runner_sha256": sha256(Path(__file__)),
            "metadata_helper_sha256": sha256(ROOT / "scripts/build_filing_manifest.py"),
            "companies": chosen[["ticker", "cik", "sector", "issuer_id"]].to_dict("records"),
            "selection": "Next fixed issuer hash per non-financial training-company sector, excluding every development-quality issuer; oldest/latest original 10-K disclosed 2013–2025"}
    plan_path = dest / "plan.json"
    if plan_path.exists():
        old = read_json(plan_path)
        if {k: v for k, v in old.items() if k != "registered_utc"} != {k: v for k, v in plan.items() if k != "registered_utc"}:
            raise ValueError("New-company quality design changed")
        plan = old
    else:
        write_json(plan_path, plan)
    agent = read_json(ROOT / ".local.json")["sec_user_agent"]
    if "@" not in agent:
        raise ValueError("Identifying SEC contact required")
    session = requests.Session()
    documents, companies = [], []
    blocked = 0
    for company in plan["companies"]:
        try:
            cik = int(company["cik"])
            name = f"CIK{cik:010d}.json"
            path = ROOT / "data/raw/sec-submissions-v1" / name
            current = fetch(f"https://data.sec.gov/submissions/{name}", path, session, agent)
            if int(current["cik"]) != cik:
                raise ValueError("Submission CIK mismatch")
            metadata = annual_records(current, cik)
            source_hashes = {name: sha256(path)}
            for name in historical_files(current):
                path = ROOT / "data/raw/sec-submissions-v1" / name
                metadata += annual_records(fetch(f"https://data.sec.gov/submissions/{name}", path, session, agent), cik)
                source_hashes[name] = sha256(path)
            table = pd.DataFrame(metadata)
            originals = table.loc[table.primary_text_candidate].drop_duplicates("accession").sort_values(["filing_date", "accession"])
            if not len(originals):
                raise ValueError("No original 10-K in the declared disclosure interval")
            selected = pd.concat([originals.head(1), originals.tail(1)]).drop_duplicates("accession")
            for row in selected.to_dict("records"):
                record = {**row, **company, "submissions_sources": source_hashes}
                path = ROOT / f"data/raw/sec-filings-validation-{version}" / str(cik) / f"{row['accession']}.html"
                try:
                    if path.exists():
                        meta = read_json(path.with_suffix(".source.json"))
                        if sha256(path) != meta["sha256"] or meta["url"] != row["document_url"]:
                            raise ValueError("Cached filing identity/checksum changed")
                    else:
                        response = session.get(row["document_url"], headers={"User-Agent": agent}, timeout=60)
                        time.sleep(0.4)
                        response.raise_for_status()
                        if "html" not in response.headers.get("Content-Type", "").lower() or len(response.content) < 10000:
                            raise ValueError("Expected a substantive HTML report")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(response.content)
                        write_json(path.with_suffix(".source.json"), {"url": row["document_url"], "sha256": sha256(path), "retrieved_utc": utc_now()})
                    blocked = 0
                    result = extract_mda(path.read_bytes())
                    text = result.pop("text")
                    record.update(result, document_path=str(path.relative_to(ROOT)), document_sha256=sha256(path))
                    if text:
                        target = dest / f"{row['accession']}.txt"
                        target.write_text(text)
                        record.update(text_path=str(target.relative_to(ROOT)), text_sha256=sha256(target))
                except (ValueError, requests.RequestException) as exc:
                    record.update(status="download_or_extraction_failed", reason=str(exc))
                    if isinstance(exc, requests.RequestException):
                        blocked += 1
                documents.append(record)
                print(company["ticker"], row["filing_date"], record["status"], record.get("section_words", 0), flush=True)
                if blocked >= 3:
                    break
            companies.append({**company, "status": "metadata_ready", "original_10k_count": len(originals)})
        except (ValueError, requests.RequestException) as exc:
            companies.append({**company, "status": "failed", "reason": str(exc)})
            if isinstance(exc, requests.RequestException):
                blocked += 1
        write_json(dest / "candidates.json", {"plan": plan, "documents": documents, "companies": companies})
        public = [{k: v for k, v in r.items() if not k.endswith("preview") and k != "before_start"} for r in documents]
        write_json(ROOT / f"results/MDA_VALIDATION_{version.upper()}.json", {
            "updated_utc": utc_now(), "plan": plan, "documents": public, "companies": companies,
            "collection_complete": len(companies) == len(chosen) and blocked < 3,
            "review_complete": False, "accepted_for_embeddings": 0,
            "market_outcomes_2026_accessed": False,
            "limits": "Frozen-parser candidate generation only. Every boundary still needs review; a small stratified new-company quality set is not a forecast holdout or proof of universal extraction accuracy.",
        })
        if blocked >= 3:
            print("Stopped after three consecutive request failures", flush=True)
            break


if __name__ == "__main__":
    main()
