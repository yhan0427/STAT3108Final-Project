# Multimodal forecasts to portfolio decisions

The frozen application grid evaluates1,632 strategy records:two baskets,34 raw/calibrated forecast procedures,three cost rates,two hypothetical cash yields andfour strategies. Basket membership and policy rules were registered during model training before the matched evaluation scores. The calendar periods were already inspected in prior research, so this remains exploratory.

Primary setting:10% annual volatility target,10bps per dollar traded,zero cash return,20-session rebalance and60-session historical correlations. Signal after close,execution at the following close; new holdings earn subsequent returns only. The comparison uses identical prices,dates and stock baskets. Stocks are long-only and unlevered. The constant baseline is frozen from mean planned validation equity; it does not match realized evaluation exposure.

## Decision findings

In the ten-sector basket, calibrated dynamic rules have modestly lower mean risk-target error than their model-specific fixed-equity controls, but the displayed date-block intervals cross zero. They also earn lower net returns than those controls. In the technology trio, calibrated LSTM dynamic rules have higher mean risk error than fixed equity; the all-modality LSTM difference is positive in both block intervals. The all-modality Ridge interval crosses zero. These results do not establish reliable dynamic superiority, equivalent risk exposure or alpha.

Improved forecast loss is not automatically improved allocation. Exposure bounds,historical correlation estimates,asset concentration and calibration mediate the decision. The language extension does not establish incremental practical benefit in this registered application.

| Basket | Calibrated model | Risk-error Δ(pp) | Net-return Δ(pp) | Mean-equity Δ(pp) |60-date risk-error95% interval(pp) |
|---|---|---:|---:|---:|---|
| Held nonfinancial sectors | Ridge_market | -0.219 | -0.34 | +20.69 | [-0.792, +0.443] |
| Held nonfinancial sectors | Ridge_financial | -0.280 | -0.45 | +20.86 | [-0.858, +0.394] |
| Held nonfinancial sectors | Ridge_text | -0.218 | -0.40 | +20.90 | [-0.789, +0.448] |
| Held nonfinancial sectors | Ridge_all | -0.279 | -0.51 | +20.92 | [-0.856, +0.397] |
| Held nonfinancial sectors | LSTM_market | -0.167 | -0.74 | +19.32 | [-0.614, +0.250] |
| Held nonfinancial sectors | LSTM_financial | -0.310 | -1.06 | +19.12 | [-0.718, +0.078] |
| Held nonfinancial sectors | LSTM_text | -0.171 | -1.29 | +19.77 | [-0.621, +0.274] |
| Held nonfinancial sectors | LSTM_all | -0.322 | -1.37 | +19.40 | [-0.762, +0.131] |
| Held technology trio | Ridge_market | +0.512 | +2.93 | +2.29 | [-0.206, +1.396] |
| Held technology trio | Ridge_financial | +0.441 | +3.18 | +1.89 | [-0.259, +1.295] |
| Held technology trio | Ridge_text | +0.457 | +2.75 | +2.30 | [-0.243, +1.330] |
| Held technology trio | Ridge_all | +0.392 | +2.85 | +1.90 | [-0.295, +1.227] |
| Held technology trio | LSTM_market | +1.277 | +4.33 | +7.18 | [+0.235, +2.330] |
| Held technology trio | LSTM_financial | +1.337 | +5.71 | +6.98 | [+0.274, +2.429] |
| Held technology trio | LSTM_text | +1.207 | +4.03 | +7.31 | [+0.194, +2.261] |
| Held technology trio | LSTM_all | +1.285 | +5.40 | +6.78 | [+0.260, +2.356] |

All differences are dynamic minus that procedure's validation-fixed equity control. The same market/financial/text/all information groups are shown for both backbones; metadata controls,raw forecasts and all unfavorable cost/cash outcomes remain in the complete CSV. The audit also reports comparisons against the historical-risk rule.

## Complete outcome contract

The CSV includes risk-target absolute error,total and annualized return,realized volatility,max drawdown,mean equity fraction,turnover and costs. Zero,10 and25bps costs are retained;4% cash yield is a hypothetical ACT/365 sensitivity,not observed Treasury returns. Risk tracking uses rolling20-session net portfolio simple-return volatility; forecast targets use single-stock log-return volatility. These are related,distinct outcomes.

Intervals use paired circular20/60-session blocks of daily risk-error differences,conditional on fixed procedures and selected baskets. They do not include full model retraining uncertainty,multiplicity adjustment or an uninspected temporal sample. Revised prices,provider/survivor selection,no taxes/liquidity/market-impact model and fixed correlation estimates remain limitations. The2026 market holdout remains sealed.
