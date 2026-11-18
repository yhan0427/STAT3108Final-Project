"""Resumable SEC coverage audit on the frozen retained cohort; never trains models."""
import argparse
from pathlib import Path
import sys
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.financial_coverage import AUDIT_FEATURES, CUTOFF, audit_snapshots, coverage_by_year
from volrisk.io import read_json, sha256, utc_now, write_json


def cached_facts(root, cik):
    candidates = [root / f"data/raw/sec-financial-v1/CIK{cik:010d}.json"]
    candidates += sorted((root / "data/raw/sec").glob(f"*-{cik}.json"))
    for path in candidates:
        if not path.exists():
            continue
        meta = read_json(path.with_suffix(".source.json"))
        if sha256(path) != meta["sha256"]:
            raise ValueError(f"Cached SEC checksum mismatch: {path.name}")
        payload = read_json(path)
        if int(payload["cik"]) != cik:
            raise ValueError(f"Cached SEC entity mismatch: {path.name}")
        return path, payload
    return None, None


def collect(root, cik, user_agent, session):
    path, payload = cached_facts(root, cik)
    if path:
        return path, payload
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    response = session.get(url, headers={"User-Agent": user_agent}, timeout=45)
    time.sleep(0.3)  # Sequential requests, including unsuccessful responses.
    response.raise_for_status()
    payload = response.json()
    if int(payload["cik"]) != cik:
        raise ValueError("SEC returned the wrong CIK")
    path = root / f"data/raw/sec-financial-v1/CIK{cik:010d}.json"
    write_json(path, payload)
    write_json(path.with_suffix(".source.json"), {
        "url": url, "retrieved_utc": utc_now(), "sha256": sha256(path),
        "scope": "Current API cache; feature builder discards disclosures after 2025-12-31",
    })
    return path, payload


def publish(root, out, roster, panel, plan, records):
    successful = [r for r in records if r["status"] == "ready"]
    tables = [pd.read_csv(out / f"{r['ticker']}_coverage.csv") for r in successful]
    coverage = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if len(coverage):
        coverage.to_csv(root / "results/FINANCIAL_COVERAGE_V1.csv", index=False)
        groups = ["sector", "role", "year", "feature"]
        summary = coverage.groupby(groups, as_index=False).agg(
            available_origins=("available_origins", "sum"),
            eligible_origins=("eligible_origins", "sum"), companies=("ticker", "nunique"))
        summary["coverage"] = summary.available_origins / summary.eligible_origins
        summary.to_csv(root / "results/FINANCIAL_COVERAGE_SUMMARY_V1.csv", index=False)
    write_json(root / "results/FINANCIAL_COVERAGE_AUDIT_V1.json", {
        "updated_utc": utc_now(), "plan": plan, "retained_market_firms": len(roster),
        "attempted_firms": len(records), "successful_firms": len(successful),
        "complete": len(records) == len(roster), "companies": records,
        "market_outcomes_2026_accessed": False, "models_fitted": False,
    })
    covered = sum(int(r["eligible_origins"]) for r in successful)
    total = len(panel)
    lines = ["# Structured financial coverage audit", "",
        f"Status: {len(records)}/{len(roster)} retained-market companies attempted; "
        f"{len(successful)} produced snapshots. This is a coverage audit, not a forecast experiment.", "",
        f"Eligible 60-day market origins: {total:,}; origins in successful companies: {covered:,}. "
        "All-company percentages below retain failed and unattempted companies in the denominator as unavailable. "
        "The successful-company CSV alone is not the full-cohort denominator.", "",
        "| Feature | Available origins | Full retained-cohort coverage |", "|---|---:|---:|"]
    if len(coverage):
        for feature, group in coverage.groupby("feature", sort=True):
            available = int(group.available_origins.sum())
            lines.append(f"| {feature} | {available:,} | {available / total:.1%} |")
    lines += ["", "## Interpretation and limits", "",
        "The roster/roles and market eligibility are unchanged. The audit reads date/identity/split metadata, "
        "not market targets or model predictions. It does not choose a feature schema from forecast outcomes. "
        "Annual rows and sector/role summaries are included in the accompanying CSVs.", "",
        "Features preserve the original US-GAAP USD definitions; parent equity/assets and current liabilities/assets "
        "are separate alternatives, not substitutes for total liabilities or operating margin. Financial-sector "
        "comparability is not established by coverage. Missing custom tags, foreign forms and unsupported currencies "
        "remain missing; do not exclude companies or select fields merely to improve model scores.", "",
        "Only facts filed and reporting periods ending by 2025-12-31 enter the builder. "
        "Calendar filing dates become usable strictly on later trading dates; content older than 180 days is missing. "
        "Filing age is a separate diagnostic and can remain observed when content is stale. The current SEC API "
        "cache may contain later disclosures, which are removed before computation. This is not an immutable "
        "point-in-time archive and does not open the sealed 2026 market holdout.", "",
        "Snapshot mappings remain tied to the frozen CIK. Reorganizations and historical entity continuity require "
        "additional review before model fitting. Availability does not establish economic correctness; no ratios "
        "or annual-report embeddings have been validated across all issuers.", "",
        "Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces", ""]
    (root / "results/FINANCIAL_COVERAGE_V1.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Bound attempted companies for a feasibility check")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / "data/processed/financial-coverage-v1"
    out.mkdir(parents=True, exist_ok=True)
    roster_path = ROOT / "config/full-cohort-v1.csv"
    frozen = read_json(ROOT / "config/full-cohort-v1.json")
    if sha256(roster_path) != frozen["roster_sha256"]:
        raise ValueError("Frozen roster changed")
    panel_path = ROOT / "data/processed/full-cohort-v2/panel.pkl"
    manifest = read_json(ROOT / "data/processed/full-cohort-v2/manifest.json")
    if sha256(panel_path) != manifest["panel_sha256"]:
        raise ValueError("Frozen market panel changed")
    panel = pd.read_pickle(panel_path)[["ticker", "date", "split"]]
    if panel.date.max() > pd.Timestamp(CUTOFF):
        raise ValueError("Sealed 2026 dates in market panel")
    retained = set(panel.ticker)
    roster = pd.read_csv(roster_path).query("ticker in @retained").sort_values("ticker")
    panel = panel.loc[panel.split.ne("excluded")].copy()
    plan = {"roster_sha256": sha256(roster_path), "market_panel_sha256": sha256(panel_path),
            "builder_sha256": sha256(ROOT / "src/volrisk/financial_coverage.py"),
            "runner_sha256": sha256(Path(__file__)), "cutoff": CUTOFF,
            "features": AUDIT_FEATURES, "staleness_days": 180,
            "scope": "491 retained market histories; no role/sample replacement or model fitting"}
    plan_path = out / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Audit code/inputs changed; version the audit directory")
    write_json(plan_path, plan)
    agent = None if args.cache_only else read_json(ROOT / ".local.json")["sec_user_agent"]
    if agent is not None and "@" not in agent:
        raise ValueError("SEC requires an identifying contact")
    records = []
    session = requests.Session()
    blocked = 0
    for item in roster.to_dict("records"):
        if args.limit is not None and len(records) >= args.limit:
            break
        ticker = item["ticker"]
        record_path = out / f"{ticker}_audit.json"
        if record_path.exists():
            record = read_json(record_path)
            for key in ["coverage", "snapshots", "source"]:
                if sha256(ROOT / record[f"{key}_path"]) != record[f"{key}_sha256"]:
                    raise ValueError(f"Cached audit changed for {ticker}: {key}")
        else:
            record = {k: item[k] for k in ["ticker", "cik", "issuer_id", "sector", "role"]}
            try:
                if args.cache_only:
                    source, payload = cached_facts(ROOT, int(item["cik"]))
                    if source is None:
                        continue
                else:
                    source, payload = collect(ROOT, int(item["cik"]), agent, session)
                blocked = 0
                snaps = audit_snapshots(payload)
                origins = panel.loc[panel.ticker.eq(ticker), ["date", "split"]]
                coverage = coverage_by_year(origins, snaps)
                for key in ["ticker", "issuer_id", "sector", "role"]:
                    coverage[key] = item[key]
                paths = {"source": source, "snapshots": out / f"{ticker}_snapshots.csv",
                         "coverage": out / f"{ticker}_coverage.csv"}
                snaps.to_csv(paths["snapshots"], index=False)
                coverage.to_csv(paths["coverage"], index=False)
                record.update(status="ready", entity_name=payload.get("entityName"),
                              eligible_origins=len(origins), snapshots=len(snaps))
                for key, path in paths.items():
                    record[f"{key}_path"] = str(path.relative_to(ROOT))
                    record[f"{key}_sha256"] = sha256(path)
                write_json(record_path, record)
            except (ValueError, requests.RequestException) as exc:
                record.update(status="failed", reason=str(exc))
                if isinstance(exc, requests.RequestException):
                    blocked += 1
        records.append(record)
        if len(records) % 10 == 0:
            publish(ROOT, out, roster, panel, plan, records)
        print(len(records), ticker, record["status"], record.get("reason", ""), flush=True)
        if blocked >= 3:
            print("Stopping after three consecutive request failures; inspect access before retrying.", flush=True)
            break
    publish(ROOT, out, roster, panel, plan, records)


if __name__ == "__main__":
    main()
