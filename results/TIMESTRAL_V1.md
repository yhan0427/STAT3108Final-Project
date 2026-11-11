# Timestral 5M: exploratory Transformer transfer

Training cutoff and full overlap unverified. 2026 release is unavailable historically in 2024-2025; retrospective transfer only. No 2026 local prices or outcomes accessed.

Target and origins match aligned-v1. 512 past rv20 observations predict the twenty-first future rv20 point. Quantile RMS uses endpoint-constant numerical integration; it is an approximate variance-aligned summary, not the same point functional as TTM. Calibration uses only 2022–2023. All 2024–2025 findings remain exploratory.

| Period | Model | MAE (pp) | QLIKE | Equal-company QLIKE ratio vs raw LogHAR |
|---|---|---:|---:|---:|
| 2024-2025 | LogHAR_proxy | 7.615 | 0.4065 | 1.000 |
| 2024-2025 | LogHAR_proxy_calibrated | 8.372 | 0.3328 | 0.905 |
| 2024-2025 | Ridge_sequence_market_calibrated | 8.064 | 0.2911 | 0.885 |
| 2024-2025 | TTM_zero_shot_calibrated | 7.949 | 0.3303 | 0.974 |
| 2024-2025 | Timestral5M | 8.244 | 0.3255 | 0.978 |
| 2024-2025 | Timestral5M_calibrated | 8.306 | 0.3228 | 0.971 |
| 2024 | LogHAR_proxy | 6.231 | 0.2683 | 1.000 |
| 2024 | LogHAR_proxy_calibrated | 7.204 | 0.2514 | 1.077 |
| 2024 | Ridge_sequence_market_calibrated | 6.931 | 0.2362 | 1.038 |
| 2024 | TTM_zero_shot_calibrated | 7.134 | 0.2504 | 1.114 |
| 2024 | Timestral5M | 6.663 | 0.2551 | 1.042 |
| 2024 | Timestral5M_calibrated | 6.720 | 0.2536 | 1.038 |
| 2025 | LogHAR_proxy | 9.139 | 0.5586 | 1.000 |
| 2025 | LogHAR_proxy_calibrated | 9.658 | 0.4225 | 0.824 |
| 2025 | Ridge_sequence_market_calibrated | 9.310 | 0.3516 | 0.816 |
| 2025 | TTM_zero_shot_calibrated | 8.846 | 0.4182 | 0.922 |
| 2025 | Timestral5M | 9.984 | 0.4029 | 0.934 |
| 2025 | Timestral5M_calibrated | 10.051 | 0.3988 | 0.926 |

Portfolio metrics in TIMESTRAL_PORTFOLIO_V1.csv include risk-target error, returns, drawdown, mean equity, turnover and costs. Each model has its own validation-fixed constant-exposure control. They do not necessarily match realized development exposure. Intervals retain whole dates but do not establish multiplicity-adjusted significance.

Next: examine cross-company failures and pretraining provenance before selecting a final protocol; keep 2026 sealed.
