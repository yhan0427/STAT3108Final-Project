"""Original annual-filing metadata, with explicit amendments and date limits."""
import re
from urllib.parse import quote


def historical_files(payload, earliest="2013-01-01", cutoff="2025-12-31"):
    files = []
    for item in payload.get("filings", {}).get("files", []):
        name = item["name"]
        if not re.fullmatch(r"CIK\d{10}-submissions-\d+\.json", name):
            raise ValueError("Unexpected SEC historical submissions filename")
        if item.get("filingFrom", earliest) <= cutoff and item.get("filingTo", cutoff) >= earliest:
            files.append(name)
    return sorted(set(files))


def annual_records(payload, cik, earliest="2013-01-01", cutoff="2025-12-31"):
    """Handle both recent and historical columnar submissions responses."""
    data = payload.get("filings", {}).get("recent", payload)
    fields = ["accessionNumber", "filingDate", "form", "primaryDocument"]
    if not all(k in data for k in fields):
        raise ValueError("SEC submissions response lacks required columns")
    length = len(data["accessionNumber"])
    if any(len(data[k]) != length for k in fields):
        raise ValueError("SEC submissions columns have unequal lengths")
    for optional in ["reportDate", "acceptanceDateTime"]:
        if optional in data and len(data[optional]) != length:
            raise ValueError("SEC optional submissions column has unequal length")
    records = []
    annual_forms = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A", "10-KT", "10-KT/A"}
    for i, filing_date in enumerate(data["filingDate"]):
        if not earliest <= filing_date <= cutoff:
            continue  # Filter before copying later-filing metadata into the manifest.
        form = data["form"][i]
        if form not in annual_forms:
            continue
        accession = data["accessionNumber"][i]
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            raise ValueError("Malformed SEC accession number")
        primary = data["primaryDocument"][i]
        if primary and (primary.startswith("/") or ".." in primary.split("/")):
            raise ValueError("Unsafe primary document path")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
        records.append({
            "cik": int(cik), "accession": accession, "form": form,
            "filing_date": filing_date,
            "report_period": data.get("reportDate", [""] * length)[i],
            "acceptance_timestamp_raw": data.get("acceptanceDateTime", [""] * length)[i],
            "primary_document": primary,
            "document_url": f"{base}/{quote(primary, safe='/')}" if primary else "",
            "index_url": f"{base}/{accession}-index.htm",
            "primary_text_candidate": form == "10-K" and bool(primary),
            "status": ("original_10k_metadata_only" if primary else "missing_primary_document")
                      if form == "10-K" else "amendment_or_other_annual_form_not_primary",
        })
    return records
