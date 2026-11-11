# Where the modern forecasts fail

Question: are forecast differences concentrated in particular companies, periods, or a few extreme observations? All diagnostics use the previously inspected 2024–2025 development sample. 2026 remains sealed. Positive QLIKE differences mean worse than calibrated sequence Ridge.

## Company breadth and influence

| Model | Companies better / 30 | All-company loss difference | Exclude worst 1 | Exclude worst 3 | Exclude worst 5 |
|---|---:|---:|---:|---:|---:|
| LogHAR_proxy_calibrated | 13/30 | +0.0417 | +0.0033 | -0.0035 | -0.0082 |
| TTM_zero_shot_calibrated | 8/30 | +0.0392 | +0.0333 | +0.0243 | +0.0167 |
| Timestral5M | 10/30 | +0.0343 | +0.0305 | +0.0238 | +0.0178 |
| Timestral5M_calibrated | 10/30 | +0.0316 | +0.0278 | +0.0214 | +0.0155 |

Removing bad companies is a post-hoc sensitivity check, not a permissible way to report a better model score. It identifies concentration; it does not justify excluding those issuers from the next sample.

## Period and historical market-risk groups

SPY trailing 20-day risk threshold = 23.55% annualized, fixed at the validation-period 75th percentile. Groups use only information available at each origin.

| Model | Dimension | Group | N | QLIKE difference | MAE difference (pp) |
|---|---|---|---:|---:|---:|
| LogHAR_proxy_calibrated | year | 2024 | 7560 | +0.0152 | +0.273 |
| LogHAR_proxy_calibrated | year | 2025 | 6870 | +0.0709 | +0.348 |
| LogHAR_proxy_calibrated | regime | high_trailing_market_risk | 750 | -0.0525 | -3.454 |
| LogHAR_proxy_calibrated | regime | other_market_risk | 13680 | +0.0469 | +0.515 |
| TTM_zero_shot_calibrated | year | 2024 | 7560 | +0.0142 | +0.203 |
| TTM_zero_shot_calibrated | year | 2025 | 6870 | +0.0667 | -0.464 |
| TTM_zero_shot_calibrated | regime | high_trailing_market_risk | 750 | +0.0868 | -4.972 |
| TTM_zero_shot_calibrated | regime | other_market_risk | 13680 | +0.0366 | +0.152 |
| Timestral5M | year | 2024 | 7560 | +0.0189 | -0.268 |
| Timestral5M | year | 2025 | 6870 | +0.0514 | +0.673 |
| Timestral5M | regime | high_trailing_market_risk | 750 | +0.0084 | -0.919 |
| Timestral5M | regime | other_market_risk | 13680 | +0.0358 | +0.240 |
| Timestral5M_calibrated | year | 2024 | 7560 | +0.0174 | -0.211 |
| Timestral5M_calibrated | year | 2025 | 6870 | +0.0472 | +0.740 |
| Timestral5M_calibrated | regime | high_trailing_market_risk | 750 | +0.0097 | -0.746 |
| Timestral5M_calibrated | regime | other_market_risk | 13680 | +0.0328 | +0.296 |

## Timestral calibrated: company extremes

| Company | QLIKE difference | MAE difference (pp) |
|---|---:|---:|
| UPS | -0.0876 | +0.682 |
| NVDA | -0.0514 | -5.842 |
| DUK | -0.0411 | -1.085 |
| PEP | -0.0386 | -0.524 |
| NEE | -0.0355 | -1.012 |
| IBM | +0.0944 | +0.512 |
| COST | +0.0964 | +0.198 |
| ORCL | +0.1133 | -0.836 |
| AAPL | +0.1146 | +1.444 |
| TXN | +0.1420 | +3.837 |

The full CSV includes every company, sector and quarter, including unfavorable cases. FAILURE_DATES_V1.csv records post-hoc worst days. Date-block intervals are in the audit; subgroup rows are descriptive, not separate significance claims.

Next decision: use this analysis to define hypotheses for the expanded cohort, not to choose favorable stocks. Verify the dated issuer/security roster and its missing-data ledger before acquiring new-company outcomes. Keep identical origins within comparisons and preserve calibrated simple baselines.

Limits: survivor-selected thirty-company sample; common market shocks; reused development periods; TTM/Timestral pretraining overlap unverified; quantile-RMS versus point-head differences. These findings cannot establish that complex models are generally ineffective.
