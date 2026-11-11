# Portfolio Risk Lab — exploratory application results

Exploratory application evaluation after V1-V3 forecasting results and one portfolio smoke test; not a confirmatory holdout.

48 declared scenarios: two baskets, four overlapping evaluation protocols, two existing forecasting models and three proportional cost assumptions. Four strategies per scenario. No model is selected using these outcomes. No recent foundation model has yet been evaluated.

## Primary comparison: forecast rule minus historical-risk rule

Cost: 10 bps per dollar traded; target: 10% annual volatility. Negative risk-error difference favors the forecast rule. Return/exposure differences must be read together; lower exposure alone can reduce risk.

| Basket | Period | Forecast | Risk-error change (pp) | Net-return change (pp) | Equity-exposure change (pp) |
|---|---|---|---:|---:|---:|
| Diversified five | static_matched | Ridge_sequence_market | 0.010 | 1.78 | 3.57 |
| Diversified five | static_matched | LSTM_market | -0.169 | 1.36 | 1.83 |
| Diversified five | origin_2020 | Ridge_sequence_market | 0.160 | 1.08 | 4.74 |
| Diversified five | origin_2020 | LSTM_market | 1.746 | 8.34 | 18.31 |
| Diversified five | origin_2022 | Ridge_sequence_market | 1.190 | -1.41 | 13.15 |
| Diversified five | origin_2022 | LSTM_market | 0.966 | -0.54 | 11.59 |
| Diversified five | origin_2024 | Ridge_sequence_market | -0.124 | 0.61 | 1.10 |
| Diversified five | origin_2024 | LSTM_market | -0.157 | 0.55 | 0.72 |
| Technology five | static_matched | Ridge_sequence_market | 0.354 | -9.09 | 3.61 |
| Technology five | static_matched | LSTM_market | 1.386 | -2.39 | 10.43 |
| Technology five | origin_2020 | Ridge_sequence_market | 0.145 | 0.71 | 1.10 |
| Technology five | origin_2020 | LSTM_market | 2.132 | 11.10 | 13.05 |
| Technology five | origin_2022 | Ridge_sequence_market | -0.012 | -2.19 | 3.51 |
| Technology five | origin_2022 | LSTM_market | 1.830 | -4.01 | 10.16 |
| Technology five | origin_2024 | Ridge_sequence_market | -0.191 | -1.64 | 2.97 |
| Technology five | origin_2024 | LSTM_market | 0.879 | 2.30 | 9.27 |

## Interpretation and limits

These are scenario diagnostics, not independent replications or proof of alpha. Both risk rules use the same basket and trailing correlation estimator; only marginal volatility estimates change. They share a target, not necessarily realized risk or average exposure. Buy-and-hold and fixed half-equity supply context, not matched-exposure attribution.

Primary errors use overlapping twenty-session net strategy returns. Paired circular twenty-/sixty-date block intervals in the JSON are exploratory and conditional on fitted forecasts, selected baskets and simulation rules. They omit model-retraining and population-selection uncertainty. Do not count correlated scenarios as independent evidence.

The account starts in cash, pays entry costs at the first next-close execution, then lets holdings drift. Risk rules rebalance every twenty signal sessions. Cash earns zero; no leverage, taxes, market impact or forced terminal liquidation. The saved forecast label starts one session before the tradable holding window.

See docs/PORTFOLIO_PROTOCOL.md for formulas and timing; config/portfolio-v1.json for the full grid. Read all cost scenarios in PORTFOLIO_V1.csv, including unfavorable results. The previous forecasting results remain separate preliminary evidence.
## 2026-11-11 turnover erratum

Use `PORTFOLIO_V1_TURNOVER_CORRECTION.csv` for turnover: the original denominator was previous-close NAV; the corrected denominator is execution-day pre-trade NAV. All 192 records were replayed and every other metric verified unchanged. New execution-aligned foundation-model results are reported separately in `ALIGNED_V1.md`; the no-foundation-model statement above describes this original V1 run only.
