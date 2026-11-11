# Execution-aligned method transfer v1

Exploratory execution-aligned method transfer, fixed before this run after V1-V3 and portfolio V1 inspection; not an untouched holdout or exact paper replication.

Target: 20 daily log returns after next-close execution (t+2 through t+21). All models share evaluation origins. TTM is a pinned pretrained retrospective comparator; historical availability and pretraining disjointness are NOT established. LogHAR_proxy uses trailing daily-return volatility, not intraday realized variance. This is not an exact reproduction of the reference paper.

Raw and validation-calibrated models are retained. Calibration is a single volatility multiplier estimated exclusively on 2022–2023 observations with fully observed targets. Ridge alpha uses that same validation split; test observations do not fit either step.

## Forecast accuracy

| Period | Model | MAE (pp) | QLIKE | Mean company QLIKE ratio vs raw LogHAR | Companies better |
|---|---|---:|---:|---:|---:|
| 2024-2025 | Persistence | 9.761 | 0.5471 | 1.554 | 1/30 |
| 2024-2025 | LogHAR_proxy | 7.615 | 0.4065 | 1.000 | 0/30 |
| 2024-2025 | Ridge_sequence_market | 7.373 | 0.3856 | 1.076 | 11/30 |
| 2024-2025 | TTM_zero_shot | 7.575 | 0.3735 | 1.070 | 10/30 |
| 2024-2025 | Persistence_calibrated | 10.958 | 0.4436 | 1.289 | 4/30 |
| 2024-2025 | LogHAR_proxy_calibrated | 8.372 | 0.3328 | 0.905 | 23/30 |
| 2024-2025 | Ridge_sequence_market_calibrated | 8.064 | 0.2911 | 0.885 | 26/30 |
| 2024-2025 | TTM_zero_shot_calibrated | 7.949 | 0.3303 | 0.974 | 13/30 |
| 2024 | Persistence | 8.143 | 0.4905 | 1.853 | 0/30 |
| 2024 | LogHAR_proxy | 6.231 | 0.2683 | 1.000 | 0/30 |
| 2024 | Ridge_sequence_market | 6.027 | 0.2762 | 1.032 | 16/30 |
| 2024 | TTM_zero_shot | 6.521 | 0.2568 | 1.069 | 14/30 |
| 2024 | Persistence_calibrated | 9.161 | 0.3993 | 1.556 | 1/30 |
| 2024 | LogHAR_proxy_calibrated | 7.204 | 0.2514 | 1.077 | 13/30 |
| 2024 | Ridge_sequence_market_calibrated | 6.931 | 0.2362 | 1.038 | 16/30 |
| 2024 | TTM_zero_shot_calibrated | 7.134 | 0.2504 | 1.114 | 14/30 |
| 2025 | Persistence | 11.542 | 0.6093 | 1.355 | 4/30 |
| 2025 | LogHAR_proxy | 9.139 | 0.5586 | 1.000 | 0/30 |
| 2025 | Ridge_sequence_market | 8.854 | 0.5059 | 1.116 | 9/30 |
| 2025 | TTM_zero_shot | 8.734 | 0.5019 | 1.096 | 13/30 |
| 2025 | Persistence_calibrated | 12.935 | 0.4923 | 1.126 | 9/30 |
| 2025 | LogHAR_proxy_calibrated | 9.658 | 0.4225 | 0.824 | 27/30 |
| 2025 | Ridge_sequence_market_calibrated | 9.310 | 0.3516 | 0.816 | 24/30 |
| 2025 | TTM_zero_shot_calibrated | 8.846 | 0.4182 | 0.922 | 19/30 |

## Portfolio application, 10 bps

Same two preset baskets, 10% target, next-close execution, 20-session rebalance and zero cash yield. Full cycles only; final positions are marked to market. All costs (0/10/25 bps) are in ALIGNED_PORTFOLIO_V1.csv. Forecast target alignment does not remove uncertainty in historical correlation estimates or the difference between log-return forecast volatility and simple net strategy returns.

| Basket | Forecast model | Forecast risk MAE (pp) | Historical risk MAE (pp) | Difference (pp) | Net-return difference (pp) | Equity-exposure difference (pp) |
|---|---|---:|---:|---:|---:|---:|
| Diversified five | Persistence | 2.469 | 2.469 | +0.000 | +0.00 | +0.00 |
| Diversified five | LogHAR_proxy | 2.503 | 2.469 | +0.035 | +0.35 | +0.42 |
| Diversified five | Ridge_sequence_market | 2.486 | 2.469 | +0.017 | +1.88 | +3.65 |
| Diversified five | TTM_zero_shot | 2.791 | 2.469 | +0.323 | -0.97 | +0.40 |
| Diversified five | Persistence_calibrated | 2.432 | 2.469 | -0.037 | -3.99 | -8.13 |
| Diversified five | LogHAR_proxy_calibrated | 2.568 | 2.469 | +0.099 | -4.79 | -9.62 |
| Diversified five | Ridge_sequence_market_calibrated | 2.343 | 2.469 | -0.126 | -1.93 | -3.49 |
| Diversified five | TTM_zero_shot_calibrated | 2.762 | 2.469 | +0.294 | -3.12 | -4.37 |
| Technology five | Persistence | 3.429 | 3.429 | +0.000 | +0.00 | +0.00 |
| Technology five | LogHAR_proxy | 4.417 | 3.429 | +0.988 | -4.17 | +7.32 |
| Technology five | Ridge_sequence_market | 3.763 | 3.429 | +0.334 | -8.73 | +3.50 |
| Technology five | TTM_zero_shot | 3.479 | 3.429 | +0.050 | -10.31 | -0.17 |
| Technology five | Persistence_calibrated | 2.950 | 3.429 | -0.479 | -5.35 | -6.86 |
| Technology five | LogHAR_proxy_calibrated | 3.229 | 3.429 | -0.200 | -8.65 | -0.33 |
| Technology five | Ridge_sequence_market_calibrated | 2.820 | 3.429 | -0.609 | -13.21 | -4.96 |
| Technology five | TTM_zero_shot_calibrated | 3.069 | 3.429 | -0.360 | -12.35 | -3.97 |

## Interpretation boundary

These comparisons distinguish forecast accuracy from policy outcomes. A small average gain is not a population claim. Company outcomes share market shocks; rolling windows overlap. Paired 20/60-date-block intervals are conditional diagnostics, not a correction for researcher choice or pretraining contamination. The 2024 and 2025 slices partition an already inspected sample. Preset baskets and models were selected after prior project results. No new holdout or alpha claim is justified.

Method and reference details: docs/ALIGNED_METHOD_TRANSFER.md. Manifest, fitted models and row-level forecasts remain local under results/aligned/v1; aggregate results and audit are versioned. The current demo intentionally remains on its labeled V2/V1 evidence until this research branch is reviewed.
