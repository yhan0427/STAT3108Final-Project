# Annual-filing metadata pilot

Fixed sector/hash selection: one retained training-role issuer in each of ten non-financial sectors. No model outcomes were used for this selection. Metadata covers filings dated 2013–2025, including historical submissions batches.

Successfully queried companies: 10/10. Annual filing records: 128; original 10-K primary-document candidates: 107.

| Company | Sector | Original 10-K candidates | Other annual records | First candidate filing | Last candidate filing |
|---|---|---:|---:|---|---|
| AVGO | Information Technology | 8 | 0 | 2018-12-21 | 2025-12-18 |
| CMS | Utilities | 13 | 0 | 2013-02-21 | 2025-02-11 |
| CRH | Materials | 2 | 13 | 2024-02-29 | 2025-02-26 |
| FANG | Energy | 13 | 2 | 2013-03-01 | 2025-02-26 |
| FOX | Communication Services | 7 | 0 | 2019-08-09 | 2025-08-06 |
| KR | Consumer Staples | 13 | 2 | 2013-04-02 | 2025-04-01 |
| LHX | Industrials | 12 | 1 | 2013-08-26 | 2025-02-14 |
| NKE | Consumer Discretionary | 13 | 0 | 2013-07-23 | 2025-07-17 |
| RMD | Health Care | 13 | 3 | 2013-08-13 | 2025-08-08 |
| WELL | Real Estate | 13 | 0 | 2013-02-26 | 2025-02-12 |

## What this establishes

The recent submissions array alone is not sufficient for the full research history. The runner queries declared historical batches overlapping the permitted disclosure period, checks duplicate accession metadata and keeps amendments and foreign/transition forms separate. Original filing text is not replaced with a later amendment.

A primary-document URL is a retrieval candidate, not proof that report text was downloaded or that Item 7 was correctly extracted. No body-text hashes, MD&A quality review, embeddings or model fitting are included. Acceptance timestamps are preserved as raw metadata; the eventual date-only join must conservatively use a later observed trading date. Report-period end is not disclosure time.

CRH illustrates a scope issue: foreign annual forms appear in the history, whereas original 10-K coverage starts later. FOX and AVGO also have shorter CIK-specific histories than a naive thirteen-year-per-company assumption. Historical entities and reorganizations need review; do not silently stitch filings under another CIK.

Next: retrieve and hash original documents for a small declared quality-review set, verify MD&A boundaries and extraction failures, then build frozen embeddings only after extraction quality passes. Preserve foreign forms as explicit scope/exclusions or design a separately validated extraction path. The 2026 market holdout remains sealed.

Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
