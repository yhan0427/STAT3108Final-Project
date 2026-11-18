import pytest

from volrisk.filings import annual_records, historical_files


def test_annual_metadata_uses_filing_date_and_preserves_amendments():
    payload = {"accessionNumber": ["0000000001-25-000001", "0000000001-25-000002", "0000000001-26-000001"],
               "filingDate": ["2025-02-01", "2025-03-01", "2026-02-01"],
               "reportDate": ["2024-12-31"] * 3,
               "form": ["10-K", "10-K/A", "10-K"],
               "primaryDocument": ["report.htm", "amendment.htm", "later.htm"]}
    rows = annual_records(payload, 1)
    assert len(rows) == 2
    assert rows[0]["primary_text_candidate"]
    assert not rows[1]["primary_text_candidate"]
    assert rows[0]["document_url"] == "https://www.sec.gov/Archives/edgar/data/1/000000000125000001/report.htm"
    assert rows[0]["acceptance_timestamp_raw"] == ""
    with pytest.raises(ValueError, match="unequal"):
        annual_records({**payload, "form": ["10-K"]}, 1)


def test_history_fetches_old_filings_without_fetching_2026_only_batch():
    payload = {"filings": {"files": [
        {"name": "CIK0000000001-submissions-001.json", "filingFrom": "2015-01-01", "filingTo": "2024-01-01"},
        {"name": "CIK0000000001-submissions-002.json", "filingFrom": "2026-01-01", "filingTo": "2026-10-01"},
    ]}}
    assert historical_files(payload) == ["CIK0000000001-submissions-001.json"]
    payload["filings"]["files"][0]["name"] = "../unexpected.json"
    with pytest.raises(ValueError, match="filename"):
        historical_files(payload)
