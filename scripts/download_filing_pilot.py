"""Retrieve a fixed oldest/latest original-10K quality set; no outcome selection."""
from pathlib import Path
import sys
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.io import read_json, sha256, utc_now, write_json


def main():
    source = ROOT / "results/FILING_MANIFEST_PILOT_V1.csv"
    manifest = pd.read_csv(source, keep_default_na=False)
    original = manifest.loc[manifest.primary_text_candidate].sort_values(["ticker", "filing_date", "accession"])
    selected = pd.concat([original.groupby("ticker").head(1), original.groupby("ticker").tail(1)])
    selected = selected.drop_duplicates(["cik", "accession"]).sort_values(["ticker", "filing_date"])
    if not selected.filing_date.le("2025-12-31").all() or not selected.form.eq("10-K").all():
        raise ValueError("Quality pilot must contain original 10-Ks disclosed through 2025")
    dest = ROOT / "data/processed/filing-text-pilot-v1"
    dest.mkdir(parents=True, exist_ok=True)
    plan = {"manifest_sha256": sha256(source), "source_sha256": sha256(Path(__file__)),
            "selection": "Oldest/latest original 10-K in each of the frozen ten sector/hash companies; no outcome-based selection",
            "documents": selected.to_dict("records"), "cutoff": "2025-12-31"}
    plan_path = dest / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Quality set changed; version this pilot")
    write_json(plan_path, plan)
    agent = read_json(ROOT / ".local.json")["sec_user_agent"]
    if "@" not in agent:
        raise ValueError("SEC contact required")
    session = requests.Session()
    records = []
    blocked = 0
    for row in plan["documents"]:
        record = dict(row)
        path = ROOT / "data/raw/sec-filings-pilot-v1" / str(row["cik"]) / f"{row['accession']}.html"
        try:
            if path.exists():
                meta = read_json(path.with_suffix(".source.json"))
                if sha256(path) != meta["sha256"] or meta["url"] != row["document_url"]:
                    raise ValueError("Cached filing identity/checksum changed")
            else:
                response = session.get(row["document_url"], headers={"User-Agent": agent}, timeout=60)
                time.sleep(0.4)
                response.raise_for_status()
                if "html" not in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Expected an HTML original filing")
                if len(response.content) < 10_000:
                    raise ValueError("Unexpectedly short filing response")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
                write_json(path.with_suffix(".source.json"), {
                    "url": row["document_url"], "retrieved_utc": utc_now(), "sha256": sha256(path),
                    "bytes": path.stat().st_size, "content_type": response.headers.get("Content-Type"),
                    "cik": row["cik"], "accession": row["accession"], "form": row["form"],
                })
            blocked = 0
            record.update(status="downloaded", path=str(path.relative_to(ROOT)), sha256=sha256(path), bytes=path.stat().st_size)
        except (ValueError, requests.RequestException) as exc:
            record.update(status="failed", reason=str(exc))
            if isinstance(exc, requests.RequestException):
                blocked += 1
        records.append(record)
        write_json(ROOT / "results/FILING_TEXT_DOWNLOAD_PILOT_V1.json", {
            "updated_utc": utc_now(), "plan": plan, "documents": records,
            "complete": len(records) == len(selected),
            "downloaded": sum(r["status"] == "downloaded" for r in records),
            "market_outcomes_2026_accessed": False,
        })
        print(row["ticker"], row["filing_date"], record["status"], record.get("reason", ""), flush=True)
        if blocked >= 3:
            print("Stopped after three consecutive request failures", flush=True)
            break


if __name__ == "__main__":
    main()
