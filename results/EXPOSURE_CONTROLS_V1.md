# Fixed-exposure controls: development experiment

Question: does dynamic exposure improve risk-target tracking relative to holding a constant equity fraction chosen using validation-period signals only? This is exploratory development on previously inspected 2024–2025 data. The 2026 holdout remains sealed.

Constants are the mean scheduled target equity fractions from 2022–2023 for each model/basket, frozen before computing development outcomes. They match validation planned exposure, NOT realized development risk or average exposure. No development outcome selects the constants.

96 scenarios / 384 strategy records: two baskets, eight forecast variants, three cost assumptions, two cash-yield assumptions. Primary: 10 bps and 0% cash yield. The 4% cash-yield sensitivity is hypothetical and accrues ACT/365 on cash held between closes.

| Basket | Forecast | Fixed validation equity | Risk-error difference (pp) | 20-day block CI (pp) | Net-return difference (pp) | Equity difference (pp) |
|---|---|---:|---:|---|---:|---:|
| Diversified five | Persistence | 71.1% | -0.722 | [-1.343, -0.079] | +8.57 | +19.88 |
| Diversified five | LogHAR_proxy | 73.5% | -0.557 | [-1.157, +0.090] | +7.81 | +17.96 |
| Diversified five | Ridge_sequence_market | 81.1% | -0.257 | [-0.748, +0.232] | +5.73 | +13.64 |
| Diversified five | TTM_zero_shot | 68.6% | -0.554 | [-1.438, +0.705] | +8.80 | +22.86 |
| Diversified five | Persistence_calibrated | 62.7% | -1.345 | [-1.888, -0.725] | +8.46 | +20.15 |
| Diversified five | LogHAR_proxy_calibrated | 64.2% | -1.090 | [-1.583, -0.540] | +6.97 | +17.13 |
| Diversified five | Ridge_sequence_market_calibrated | 69.3% | -0.954 | [-1.478, -0.375] | +7.49 | +18.20 |
| Diversified five | TTM_zero_shot_calibrated | 63.5% | -0.954 | [-1.762, +0.196] | +8.98 | +23.14 |
| Technology five | Persistence | 37.6% | +0.712 | [-0.446, +1.950] | +14.08 | +14.03 |
| Technology five | LogHAR_proxy | 43.5% | +1.851 | [+0.990, +2.736] | +6.11 | +15.42 |
| Technology five | Ridge_sequence_market | 40.2% | +1.154 | [+0.144, +2.259] | +3.66 | +14.88 |
| Technology five | TTM_zero_shot | 37.9% | +0.779 | [-0.254, +1.852] | +3.56 | +13.53 |
| Technology five | Persistence_calibrated | 32.6% | -0.159 | [-1.133, +0.834] | +11.90 | +12.19 |
| Technology five | LogHAR_proxy_calibrated | 37.8% | +0.526 | [-0.371, +1.390] | +5.26 | +13.43 |
| Technology five | Ridge_sequence_market_calibrated | 34.0% | -0.152 | [-1.034, +0.794] | +3.11 | +12.61 |
| Technology five | TTM_zero_shot_calibrated | 35.1% | +0.181 | [-0.812, +1.222] | +3.30 | +12.53 |

Negative risk-error differences favor the dynamic forecast rule. Intervals are conditional date-block diagnostics, not multiplicity-adjusted significance claims. Read both 20/60-day intervals in the audit. The CSV includes risk error, net return, drawdown, average equity, turnover and trading costs for all strategies and sensitivities.

Limits: survivor selection, already inspected periods, overlapping windows, approximate covariance forecasts, and model pretraining overlap. A validation-matched constant does not isolate timing perfectly when realized equity exposure changes in the development period. No result establishes alpha or universal model superiority.
