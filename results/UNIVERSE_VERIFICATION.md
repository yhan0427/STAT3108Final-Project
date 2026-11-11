# Reference-date universe verification

Reference: 2025-12-31. Status: **no full verified roster adopted; no expanded prices acquired**. All public-source snapshots below contain membership metadata only. The 2026 stock-outcome holdout remains sealed.

## Findings

1. `hanshof/sp500_constituents`, revision `a91ef88fad5ace83bed1f3452f451247295bcd18`: latest source date is 2025-08-23. Its 503-security candidate omits subsequent changes. Rejected for year-end use. The saved candidate CSV explicitly marks every row rejected.
2. `chinobing/historical_sp500_constituents`, revision `ef41f25cd1ff7e415648ee633c29c61bf035918c`: a 2025-12-31 row exists with 501 unique symbols, but it omits CRH/CVNA/FIX and retains SOLS. Those observations conflict with the [official December rebalance announcement](https://www.spglobal.com/spdji/en/documents/indexnews/announcements/20251205-1480992/1480992_december2025rebalance.pdf), effective December 22. Reject as a ready-to-use year-end roster pending reconciliation. The September addition checks pass, which demonstrates why checking only one rebalance or row count is insufficient.

The official December announcement adds CRH, CVNA and FIX and removes LKQ, SOLS and MHK. The [official September announcement](https://press.spglobal.com/2025-09-05-AppLovin,-Robinhood-Markets-and-Emcor-Group-Set-to-Join-S-P-500-Others-to-Join-S-P-100,-S-P-MidCap-400-and-S-P-SmallCap-600) supplies another spot check. These events do not constitute a full audited historical membership ledger.

[WRDS provides historical membership extraction from CRSP](https://wrds-www.wharton.upenn.edu/pages/grid-items/downloading-the-sp-500-constituents/), requiring subscription. Han prefers public sources for now. A [fund issuer's dated 2025-12-31 holdings report](https://www.guggenheiminvestments.com/GuggenheimInvestments/media/mutual-funds/holdings/RYSOX-S-P-500-2025-12-31.pdf) supplies independent name-level evidence, but fund holdings and sector classifications are not interchangeable with the official index roster. Cash, derivatives and security names/classes need reconciliation.

## Selection decision

Target all verified reference-date issuers eventually, subject to auditable coverage. First validate the pipeline on approximately 100 sector-stratified issuers. Sector-proportional allocation and deterministic hashing must be applied to a verified issuer registry before accessing new-company outcomes. Retain every candidate/exclusion reason. Include financial companies in market-only experiments; accounting-modal experiments need a separate suitable financial-sector schema. One eligible ordinary-equity class per issuer enters the main company-level panel, with other classes retained in the registry.

Do not pick the best-performing companies from the thirty-company study, select only technology giants, demand twelve complete years for every company, or silently replace difficult/missing names. A larger fixed-reference cohort still has survival/selection bias for its historical backfill. It is not historical index membership at every date.

## Remaining work

Reconcile an independent primary dated holdings/name anchor with the official change ledger; verify issuer and share-class mapping, sector definitions and reference-date coverage. No full roster or 100-company sample is yet frozen. Both source audit JSONs preserve revisions, hashes and limitations; raw multi-year membership files were processed in memory rather than saved redundantly. Never infer unique company count from security count alone.
