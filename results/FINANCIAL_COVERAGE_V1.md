# Structured financial coverage audit

Status: 491/491 retained-market companies attempted; 491 produced snapshots. This is a coverage audit, not a forecast experiment.

Eligible 60-day market origins: 1,280,028; origins in successful companies: 1,280,028. All-company percentages below retain failed and unattempted companies in the denominator as unavailable. The successful-company CSV alone is not the full-cohort denominator.

| Feature | Available origins | Full retained-cohort coverage |
|---|---:|---:|
| cash_to_assets | 1,108,388 | 86.6% |
| cfo_to_assets | 1,041,540 | 81.4% |
| current_liabilities_to_assets | 1,040,252 | 81.3% |
| current_ratio | 1,040,252 | 81.3% |
| filing_age_days | 1,252,084 | 97.8% |
| income_to_assets | 1,205,937 | 94.2% |
| liabilities_to_assets | 861,213 | 67.3% |
| operating_margin | 875,162 | 68.4% |
| parent_equity_to_assets | 1,154,449 | 90.2% |
| revenue_growth | 1,108,747 | 86.6% |

## Interpretation and limits

The roster/roles and market eligibility are unchanged. The audit reads date/identity/split metadata, not market targets or model predictions. It does not choose a feature schema from forecast outcomes. Annual rows and sector/role summaries are included in the accompanying CSVs.

Features preserve the original US-GAAP USD definitions; parent equity/assets and current liabilities/assets are separate alternatives, not substitutes for total liabilities or operating margin. Financial-sector comparability is not established by coverage. Missing custom tags, foreign forms and unsupported currencies remain missing; do not exclude companies or select fields merely to improve model scores.

Only facts filed and reporting periods ending by 2025-12-31 enter the builder. Calendar filing dates become usable strictly on later trading dates; content older than 180 days is missing. Filing age is a separate diagnostic and can remain observed when content is stale. The current SEC API cache may contain later disclosures, which are removed before computation. This is not an immutable point-in-time archive and does not open the sealed 2026 market holdout.

Snapshot mappings remain tied to the frozen CIK. Reorganizations and historical entity continuity require additional review before model fitting. Availability does not establish economic correctness; no ratios or annual-report embeddings have been validated across all issuers.

Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
