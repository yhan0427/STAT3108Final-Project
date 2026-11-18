"""Generate versioned, reviewable MD&A candidates from original local SEC filings."""
from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.io import read_json, sha256, utc_now, write_json
from volrisk.mda import extract_mda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v6", choices=["v6"])
    args = parser.parse_args()
    version = args.version
    download_path = ROOT / "results/FILING_TEXT_DOWNLOAD_PILOT_V1.json"
    downloads = read_json(download_path)
    if not downloads["complete"]:
        raise ValueError("Finish the frozen download pilot before extraction")
    dest = ROOT / f"data/processed/mda-pilot-{version}"
    dest.mkdir(parents=True, exist_ok=True)
    plan = {"download_audit_sha256": sha256(download_path),
            "extractor_sha256": sha256(ROOT / "src/volrisk/mda.py"),
            "runner_sha256": sha256(Path(__file__)), "min_chars": 2500, "max_chars": 350000}
    plan_path = dest / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Parser/source changed; use a new extraction version")
    write_json(plan_path, plan)
    records = []
    for row in downloads["documents"]:
        record = {k: row[k] for k in ["ticker", "cik", "accession", "filing_date", "report_period", "document_url"]}
        if row["status"] != "downloaded":
            record.update(status="source_download_failed", reason=row.get("reason", ""))
        else:
            path = ROOT / row["path"]
            if sha256(path) != row["sha256"]:
                raise ValueError("Original document checksum changed")
            result = extract_mda(path.read_bytes())
            section = result.pop("text")
            record.update(result, document_sha256=row["sha256"])
            if section:
                text_path = dest / f"{row['accession']}.txt"
                text_path.write_text(section)
                record.update(text_path=str(text_path.relative_to(ROOT)), text_sha256=sha256(text_path))
        records.append(record)
        print(row["ticker"], row["filing_date"], record["status"], record.get("section_words", 0), flush=True)
    detailed_path = dest / "candidates.json"
    write_json(detailed_path, {"plan": plan, "documents": records})
    public = [{k: v for k, v in r.items() if not k.endswith("preview") and k != "before_start"} for r in records]
    write_json(ROOT / f"results/MDA_EXTRACTION_PILOT_{version.upper()}.json", {
        "updated_utc": utc_now(), "plan": plan,
        "documents": public, "private_review_sha256": sha256(detailed_path),
        "accepted_for_embeddings": 0, "review_complete": False,
        "scope": "Machine-generated candidate boundaries; agent/human quality review not yet completed; no embeddings computed",
    })
    pd.DataFrame(public).to_csv(
        ROOT / f"results/MDA_EXTRACTION_PILOT_{version.upper()}.csv", index=False)


if __name__ == "__main__":
    main()
