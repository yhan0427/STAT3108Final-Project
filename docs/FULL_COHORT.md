# Full reference-cohort collection and unified training

## Frozen scope

All 500 CIK-defined issuers in the explicitly reconstructed 2025-12-31 registry are candidates, with one selected share class each. This is a fixed reference cohort with historical backfill, not a point-in-time historical S&P 500 membership study. Full independent official roster verification is qualified as described in EXPANSION_100.md.

Before full acquisition, select 100 held-company candidates only from outside the 100 issuers already scored during the pilot, using sector-proportional largest remainders and a fixed issuer-ID hash. The other 400 candidates form the fitting/development role. Pilot held firms already inspected now belong to development; they are not described as untouched again. Roster and plan hashes are in config/full-cohort-v1.json.

## Coverage and identity recovery

The initial collection retained 489 histories and 1,422,744 company-date rows. Version 2 recovers two more histories using verified symbol-only changes, yielding **491 retained histories and 1,428,780 company-date observations**. The 500 candidates are not all usable training firms. Record raw availability, eligible origins, roles and all failure reasons separately.

The [BNY issuer announcement](https://www.bny.com/corporate/global/en/about-us/newsroom/press-release/bny-announces-planned-change-of-stock-ticker-symbol-to-bny-130465.html) says BK changes to BNY with legal name, capital structure, CUSIPs and shareholder rights unchanged. The [Marsh issuer announcement](https://www.marsh.com/en/corp/about/news/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html) confirms MMC changes to MRSH with legal name/CUSIP unchanged. Preserved SEC CIK mappings agree. Reference-date tickers and frozen roles remain BK/MMC; provider aliases retrieve only 2014–2025 history. Symbol metadata is not a 2026 price/return input.

Other current CIK/ticker candidates are not automatically accepted as historical continuity: mergers and changed economic entities require stronger security-level evidence. Remaining nine unavailable reference histories stay in the failure ledger. Neither missing final losses nor unavailable prices are replaced with zero. Current provider coverage may create selection bias.

Versions 1 and 2 preserve independent manifests and coverage records. The alias-recovery script does not rewrite the initial collection. Model training uses only v2. Prices must be positive, calendars complete from first observed trading date through the reference-date close, and source checksums valid. IPO gaps are not filled. A successful raw history need not supply enough context for fitting or comparison.

## Unified procedure

Keep the previously used time boundaries: fitting origins from 2015 with labels ending before 2022, validation in 2022–2023, development evaluation in 2024–2025. The target is twenty log returns after next-close execution (t+2 through t+21). All transforms, Ridge alpha, neural early stopping and multiplicative volatility calibration exclude held-company labels. The 2026 temporal holdout remains sealed.

Fit on eligible sixty-session origins from training-role issuers. Require full 512-rv20 context for common validation and held-company comparison with Timestral. This can reduce firm/date counts relative to a sixty-day-only comparison; publish both eligibility counts. The full common held set has 97 issuers / 46,657 eligible origins, while 99 have some sixty-day eligible evaluation history.

Compare persistence, sequence Ridge and three-seed market LSTM, plus the same pinned Timestral 5M transfer procedure; report raw and validation-calibrated variants. Ridge and LSTM share sixty-session market information. Timestral has different context/point-output semantics and unverified pretraining overlap; it is a retrospective transfer comparator, not a historically available 2024 trading model. Do not interpret the comparison as equal capacity or prove architecture-wide superiority.

For memory efficiency, compute float64 Ridge cross-products in batches and solve regularized centered normal equations, retaining an unpenalized intercept. Validation selects among the same alphas (1, 100, 10,000). A regression test checks the batch solution against scikit-learn Ridge. Temporary Timestral contexts use a float32 memory-mapped file and are removed after successful completion. Raw inputs, provenance and fitted artifacts stay local; aggregate results are versioned.

These new-company outcomes remain conditional on the chosen cohort and earlier development choices. Date-block intervals retain market-level dependence but omit population and model-training uncertainty. The untouched 2026 evaluation is reserved for the final frozen method.
