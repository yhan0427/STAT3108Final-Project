# Company expansion: dated-universe design

## Chosen reference and honest scope

Target reference date: **2025-12-31**, before the reserved 2026 evaluation. The dated constituent list has NOT yet been obtained or verified; no 100/500-company expansion is claimed. Acquire a date-stamped source and archive its provenance/hash before choosing the expanded cohort. If a reliable list for that date is unavailable, stop and document the gap; do not silently substitute today's members or label them as 2025 members.

This first design is a fixed reference-date cohort with historical price backfill, not a reconstruction of membership at every historical date. It conditions historical analyses on inclusion at the reference date and therefore has selection/survival bias. A genuine historical-membership study is a separate data task requiring membership entry/exit dates and delisting returns. Larger counts do not remove this bias.

The [official S&P 500 description](https://www.spglobal.com/spdji/en/indices/equity/sp-500/) describes 500 companies; the [S&P U.S. Indices methodology](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-us-indices.pdf) treats eligible listed share classes separately. Therefore count firms and securities separately; do not force a security count to 500.

## Registry and units

Keep one row per security with: issuer research ID, issuer name, CIK where applicable, security/share-class ID, provider symbol, exchange, reference-date membership, source effective date, observed date, source URL/hash, and known symbol/entity changes. CIK is an entity key, not a universal permanent security ID; reorganizations and ticker reuse require explicit mapping. Never join histories on ticker alone when identity changes are possible.

For the primary company-level panel choose one eligible ordinary-equity class per issuer using an outcome-independent deterministic rule: alphabetically first normalized class symbol at the fixed reference date. This is reproducible, not a claim of the most liquid class. Retain all classes in the registry and report the exclusions; sensitivity to class choice can be a later analysis. Count raw observations as unique security-date price rows, not OHLCV cells. Count model-ready labeled origins separately after history/label filters; report unbalanced coverage and excluded rows.

## Stages and selection

- Existing 30-company study remains a distinct convenience sample with its own provenance.
- Phase one targets 100 issuers from the verified reference cohort. Preserve existing names only if reference membership and identity are verified. Allocate remaining slots proportionally across source-defined sectors using largest remainders; within sector rank by SHA-256 of a fixed seed plus issuer ID. Publish the candidate roster before price/forecast scoring. Do not replace a company because its results are inconvenient.
- Phase two uses all verified reference issuers with available auditable histories; actual retained count can be below 500. Report coverage rather than manufacturing rows to reach 1M.

Price coverage is a reported outcome of collection. Do not forward-fill prices across IPO gaps, suspensions or delistings. Start at the first valid contiguous context; a 512-point rv20 input itself needs earlier returns. Score only origins with all required past features and complete future labels, preserving the same origins across compared methods. Keep short-history names in the coverage/exclusion ledger instead of erasing them from the candidate roster.

All sectors can enter market-only experiments. Financial institutions' accounting variables require a separate harmonized schema; do not reuse non-financial operating ratios blindly. A multimodal matched subset must be reported separately from the market-only universe.

## Cross-company and time generalization

A strict new-company evaluation excludes those issuers' labels from pooled fitting, hyperparameter selection and calibration; their past observed market history may still be used at inference. A per-company HAR fitted on that issuer's past labels is an adaptation baseline, not a zero-label held-company model, and must be labeled accordingly. Predeclare the two settings rather than pooling them into one headline.

Develop the expanded sample on pre-2026 data under the agreed sealed-holdout policy. Do not train on 2026 or inspect its performance. Forecast comparisons use common eligible origins, company-level summaries and paired resampling of whole dates. Overlapping firm-days and market shocks are not independent replications.

## Missing/dead securities

Keep a failure ledger: missing provider coverage, unresolved identifier, insufficient history, delisted before reference date, post-reference delisting, and missing terminal return. Do not replace a missing terminal return with zero or omit the last loss silently. If reliable dead-security histories/terminal returns require unavailable licensed data, label the fixed-cohort limitation and leave historical-membership evaluation incomplete. Index deletions and market delistings are different events.

Neither this design nor a future raw observation count supports an unqualified resume claim of a historical S&P 500 backtest. Only use verified retained counts, dates, and explicitly named scope after acquisition and filtering.
