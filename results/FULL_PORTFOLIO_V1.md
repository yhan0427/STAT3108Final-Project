# Full-run forecasts to portfolio decisions

Exploratory application after full forecast scores inspected; basket selection uses only role, sector, history coverage and fixed hash, not forecast outcomes.

Baskets are selected from locally held companies, one per sector or five technology firms, using a fixed hash among histories with at least 400 eligible validation and evaluation origins. This availability-conditioned choice introduces coverage selection. Forecast scores were already inspected; this is not a new untouched application test.

Global model parameters and volatility calibration are reused unchanged. Basket validation prices/predictions set a constant stock fraction; basket target labels fit neither model nor calibration. The fixed fraction matches planned validation exposure, not realized evaluation exposure or risk. Zero cash yield is primary; 4% is a hypothetical ACT/365 sensitivity.

| Basket | Model | Benchmark | Risk-error difference (pp) | Net-return difference (pp) | Equity difference (pp) |
|---|---|---|---:|---:|---:|
| Held-sector basket | Persistence_calibrated | fixed_validation | -0.326 | -3.93 | +16.80 |
| Held-sector basket | Persistence_calibrated | historical | -0.492 | -2.59 | -10.17 |
| Held-sector basket | Ridge60_calibrated | fixed_validation | -0.067 | -3.80 | +18.52 |
| Held-sector basket | Ridge60_calibrated | historical | -0.569 | -1.49 | -4.01 |
| Held-sector basket | LSTM60_calibrated | fixed_validation | +0.026 | -3.06 | +16.24 |
| Held-sector basket | LSTM60_calibrated | historical | -0.472 | -0.77 | -6.36 |
| Held-sector basket | Timestral5M_calibrated | fixed_validation | +0.282 | -1.82 | +19.99 |
| Held-sector basket | Timestral5M_calibrated | historical | -0.126 | +0.18 | -3.95 |
| Held-technology basket | Persistence_calibrated | fixed_validation | +0.082 | +4.46 | +6.78 |
| Held-technology basket | Persistence_calibrated | historical | -0.440 | -5.93 | -5.97 |
| Held-technology basket | Ridge60_calibrated | fixed_validation | +0.251 | -0.84 | +6.75 |
| Held-technology basket | Ridge60_calibrated | historical | -0.496 | -7.26 | -2.12 |
| Held-technology basket | LSTM60_calibrated | fixed_validation | -0.008 | -0.19 | +3.86 |
| Held-technology basket | LSTM60_calibrated | historical | -0.762 | -5.93 | -4.37 |
| Held-technology basket | Timestral5M_calibrated | fixed_validation | +0.350 | +0.74 | +6.64 |
| Held-technology basket | Timestral5M_calibrated | historical | -0.345 | -7.38 | -3.88 |

Negative risk-error differences favor the forecast policy. All raw and calibrated variants and cost/cash sensitivities remain in the CSV, including returns, drawdown, average equity, turnover and costs. Both 20/60-date intervals are conditional diagnostics; comparisons are not multiplicity-adjusted.

Timing: signal after close, execute next close, earn subsequent returns; full twenty-session cycles only, entry costs included, no terminal forced sale. Historical correlation, revised prices, zero taxes/market impact, survivor selection, preset selection and pretrained-data overlap remain limitations. Results demonstrate policy tradeoffs, not guaranteed target risk, alpha or personal investment advice. The 2026 holdout remains sealed.

## Main observed tradeoff

At 10 bps and zero cash yield, calibrated LSTM reduces risk-target error relative to the historical-risk rule by 0.472 pp in the sector basket and 0.762 pp in technology. Net cumulative returns are lower by 0.77 pp and 5.93 pp, respectively, alongside lower mean equity exposure. Relative to each LSTM validation-fixed control, error differences are +0.026 pp and -0.008 pp; both 20/60-date intervals cross zero. This provides no reliable dynamic-policy advantage over these simple controls in this experiment. It does not prove equivalence, causality or universal failure. Validation-fixed and dynamic evaluation exposures differ substantially, so timing is not perfectly isolated.

The primary LSTM tradeoff plot is FULL_PORTFOLIO_V1.png. The chart is descriptive, not an uncertainty visualization. All 384 records, including raw/calibrated models and cost/cash sensitivity, remain available.
