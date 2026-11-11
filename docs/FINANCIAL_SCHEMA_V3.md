# Alternative accounting schema: scope and reproducibility

V3 investigates whether the prior financial-value result survives a more widely available set of accounting indicators. It retains thirty companies and defers text and company expansion. It is exploratory: earlier evaluation results were already inspected.

`config/robustness-v3.json` fixes three protocols: static 2024–25, the previously difficult 2020 origin, and eight companies excluded from fitting/validation. Same targets, sixty-day histories, model architecture, three seeds, thirteen model specifications and train-only preprocessing as V2. Each boundary purges forward labels crossing the boundary. Market-only predictions and all original data columns are checked for exact agreement with V2.

The new eight-column schema replaces total liabilities/assets and operating margin with parent stockholders’ equity/assets and current liabilities/assets. These are different quantities, not imputations. Standard USD US-GAAP concepts only; no invented issuer-specific mappings. Parent equity excludes noncontrolling interests, negative values remain valid, and it is not a substitute for total equity in the liabilities identity. Liabilities components require both same-period components. Pretax margin was rejected for inadequate coverage before model fitting; the candidate dataset is retained separately.

New snapshots use only facts filed by the snapshot date; daily alignment requires filing strictly before the forecast date. Accounting values stale after 180 days. The missingness control zeros seven standardized accounting values while retaining filing age and all eight missing flags. Its meaning changes with the feature schema; compare full values with the corresponding control. Coverage and economic content change together, so V3 cannot isolate a causal coverage effect.

The original `data/processed/main` and earlier results are preserved. New artifacts live in `data/processed/financial-v3` and `results/robustness/v3`. Both builds refuse overwrite. The fitted-run manifest records the plan, source hashes and panel checksum before fitting. Do not rerun into existing output directories.

```bash
.venv/bin/python -m volrisk.fundamentals_v3
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 .venv/bin/python -m volrisk.robustness --plan config/robustness-v3.json
.venv/bin/python scripts/summarize_financial_v3.py
```

The completed summary, all model metrics and conditional paired block intervals are in `results/FINANCIAL_V3.md`. `FINANCIAL_V3_AUDIT.json` verifies original-column equality, panel hashes, unchanged market controls and two current-liability ratios against previously transcribed original filings. Split-level coverage is in `FINANCIAL_V3_COVERAGE.csv`. These checks do not certify a complete immutable point-in-time financial archive.

Future scope should first audit missing concepts by issuer and preserve economic definitions. Then register common-coverage comparisons and a new external holdout. Do not choose company count or add text merely to seek a favorable evaluation result. The website retains V2 replay and clearly labels V3 as a separate research result.
