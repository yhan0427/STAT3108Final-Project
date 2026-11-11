# Execution-aligned method transfer

## Reference and scope

The starting reference is Brini (2026), [Forecasting Realized Volatility with Time Series Foundation Models](https://arxiv.org/abs/2607.05291), with [author code](https://github.com/Alessiobrini/tsfm-rv). It compares pretrained forecasters and econometric benchmarks on intraday-derived realized volatility. Its attention to per-asset loss ratios, calibration and pretraining overlap motivates our diagnostics. We transfer one comparator and evaluation ideas; we do not reproduce its dataset, targets, estimates or headline numbers.

[IBM's TTM model card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) documents the r2.1 daily-frequency models and training datasets. We use `512-96-ft-r2.1`, pinned to revision `3442406f106122f414a11f72b4ebfe4878bec89d`, as recorded in the [reference repository configuration](https://github.com/Alessiobrini/tsfm-rv/blob/main/code/config.py). TTM's architecture dates to 2024; its use in a 2026 finance paper makes it a relevant recent-study comparator, not a claim that we implemented the newest architecture.

## Our target and information timeline

At close t, observe prices and issue the signal. Execute at close t+1. The target is annualized population standard deviation of log returns t+2 through t+21 (twenty returns). This equals trailing rv20 observed at t+21. Predicting the twenty-first point of a future rv20 trajectory therefore targets exactly this marginal holding-period statistic. It is not a forecast of return direction or a probability of loss.

All original data and forecasts remain unchanged. The new panel is constructed in memory from the checksummed local panel. Features are computed through t only. Company boundaries are checked for sequence windows. Labels crossing train/validation/test boundaries are excluded using their delayed target-end timestamps. The original train period begins in 2015; validation is 2022–2023; evaluation is 2024–2025, with yearly slices. All are previously inspected historical periods, not new confirmatory data.

This fixes the marginal forecasting horizon. It does not make a historical correlation estimate a correlation forecast, eliminate weight drift, or equate log-return volatility with net simple-return strategy volatility. The portfolio experiment is still an approximation to a practical risk policy.

## Models and calibration

- Persistence: trailing rv20 at the signal close.
- LogHAR_proxy: separate company OLS of log target on log rv20, log mean of the last five rv20 observations, and log mean of the last twenty-two rv20 observations. Fit only on training rows. Recover a level forecast with the training residual log-normal adjustment. The name explicitly identifies a daily-return proxy, not the paper's intraday HAR specification.
- Ridge_sequence_market: pooled sixty-session histories of the ten existing market features. Train-only median imputation, clipping and standardization; float64 LSQR solver, tolerance 1e-8. Select alpha from 1, 100 and 10,000 by validation log MSE. This removes the earlier float32 linear-system stability concern for this new experiment; it does not revise the archived V2/V3 results.
- TTM_zero_shot: 512 historical rv20 observations, standardized separately using each context's mean and population standard deviation. Fixed checkpoint, CPU inference, daily frequency token 8. Invert scaling and extract step 21. A predeclared 1e-6 positive floor is counted in the audit. No local fitting or future label reaches the TTM inference worker. Input spans and covariates differ between models; this compares practical forecasting procedures, not a controlled architecture-only ablation.

For every model, also report a calibration variant. Fit one pooled multiplier on validation data: c = sqrt(mean((actual / forecast)^2)). This minimizes validation QLIKE among positive multiplicative volatility adjustments. Freeze c before evaluation. This is a scale-only diagnostic, not the paper's full Mincer–Zarnowitz procedure. Ridge selection and calibration share validation observations; test observations fit neither step. Calibration may improve QLIKE and worsen absolute error, so report both.

## Evaluation and practical application

All eight variants use identical evaluation rows. Report pooled QLIKE, MAE, RMSE and equal-company mean QLIKE ratios relative to raw LogHAR_proxy, plus company win counts. Raw and calibrated baseline rows remain visible, avoiding a comparison only against an uncalibrated weak baseline. Paired 20/60-date-block intervals retain market-level dependence, but do not incorporate model-training uncertainty, researcher selection or pretraining overlap.

Portfolio replay uses the same two baskets, fixed 10% risk target, and 0/10/25 bps costs. Keep every variant, including calibrated persistence. End at the final scheduled signal's complete t+21 target end, so every investment cycle has twenty return observations. Entry costs are included, cash yields zero, and no forced final liquidation is imposed. The primary application outcome remains risk-target error; return, drawdown, equity exposure and costs must be read alongside it. A lower-risk strategy is not automatically a better timing model.

## Pretraining audit and historical availability

The official model card lists Bitcoin among the r2 training sources and additional non-stock daily series for r2.1. We have not established the exact training cutoff or complete dataset disjointness from our test universe. The checkpoint postdates some test observations. Therefore the TTM results are **retrospective transfer diagnostics**, not a claim that this model could have been traded in 2024, and not proof of leakage-free pretraining. No inference about stock overlap follows merely from small model size or its observed performance.

A future untouched evaluation must be fixed before collecting/scoring its labels, after establishing the chosen model's release date and available training-data evidence. Merely adding more firm-day rows or reusing a calendar year already inspected does not produce independent confirmation.

## Reproduction

The original `.venv` runs the data, regressions and portfolio calculations. A separate `.venv-foundation` isolates TTM's dependency constraints from the existing demo:

```sh
.venv/bin/python -m venv .venv-foundation
.venv-foundation/bin/python -m pip install -r requirements-foundation-lock.txt
.venv-foundation/bin/python scripts/download_ttm.py
.venv/bin/python scripts/evaluate_aligned.py
```

The run refuses to overwrite `results/aligned/v1`. Its manifest records the plan, source hashes and original panel checksum before fitting. The worker records its input/output hashes, exact installed versions and checkpoint-file hashes. Row-level forecasts and fitted models stay local; aggregate reports and audit files are suitable for Git. The large temporary context matrix is deleted after successful completion; its checksum is retained and it can be reconstructed from the preserved panel with `align_panel` and `context_matrix` using validation then test index order. TTM receives only local numeric contexts; no paid API is involved. `config/aligned-v1.json` fixes the choices made before this run.

Results: `results/ALIGNED_V1.md`, `ALIGNED_V1.csv`, `ALIGNED_V1_COMPANIES.csv`, `ALIGNED_PORTFOLIO_V1.csv`, and `ALIGNED_V1_AUDIT.json`. These experiments are separate from the demo's clearly labeled V2 forecasts and Portfolio V1 replay.
