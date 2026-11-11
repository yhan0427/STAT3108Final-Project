# Local experiment results

Historical, exploratory v1 study; fixed convenience sample, not historical S&P 500 membership.

Forecast origins: 2024-01-02 to 2025-12-02; 14,460 firm-days, 30 companies.

| Model | MAE | RMSE | QLIKE | R² vs persistence | Top-5 overlap |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.09687 | 0.14418 | 0.54128 | 0.000 | 0.457 |
| HAR_type | 0.07843 | 0.12453 | 0.48363 | 0.254 | 0.463 |
| Boosting_market | 0.07661 | 0.12402 | 0.41124 | 0.260 | 0.510 |
| Boosting_fundamentals | 0.07418 | 0.12084 | 0.42706 | 0.298 | 0.562 |
| LSTM_market | 0.07002 | 0.11518 | 0.36548 | 0.362 | 0.544 |
| LSTM_fundamentals | 0.06992 | 0.11572 | 0.38192 | 0.356 | 0.564 |

MAE/RMSE are annualized volatility fractions (0.01 = one percentage point). QLIKE uses variance. R² is relative to persistence, not to a constant mean. LSTM rows average predictions from three seeds with validation-selected stopping epochs.

## Paired loss comparisons

Negative differences favor the first model. Intervals resample entire dates and preserve the cross-section.

- HAR_type minus Persistence, 20-day blocks: mean -0.0577, 95% interval [-0.1046, 0.0088].
- HAR_type minus Persistence, 60-day blocks: mean -0.0577, 95% interval [-0.1049, -0.0005].
- Boosting_market minus Persistence, 20-day blocks: mean -0.1300, 95% interval [-0.1696, -0.0864].
- Boosting_market minus Persistence, 60-day blocks: mean -0.1300, 95% interval [-0.1624, -0.0980].
- Boosting_fundamentals minus Persistence, 20-day blocks: mean -0.1142, 95% interval [-0.1747, -0.0414].
- Boosting_fundamentals minus Persistence, 60-day blocks: mean -0.1142, 95% interval [-0.1776, -0.0444].
- LSTM_market minus Persistence, 20-day blocks: mean -0.1758, 95% interval [-0.2192, -0.1325].
- LSTM_market minus Persistence, 60-day blocks: mean -0.1758, 95% interval [-0.2120, -0.1371].
- LSTM_fundamentals minus Persistence, 20-day blocks: mean -0.1594, 95% interval [-0.2085, -0.1022].
- LSTM_fundamentals minus Persistence, 60-day blocks: mean -0.1594, 95% interval [-0.2055, -0.1081].
- LSTM_fundamentals minus LSTM_market, 20-day blocks: mean 0.0164, 95% interval [-0.0000, 0.0377].
- LSTM_fundamentals minus LSTM_market, 60-day blocks: mean 0.0164, 95% interval [-0.0003, 0.0372].
- Boosting_fundamentals minus Boosting_market, 20-day blocks: mean 0.0158, 95% interval [-0.0208, 0.0575].
- Boosting_fundamentals minus Boosting_market, 60-day blocks: mean 0.0158, 95% interval [-0.0288, 0.0670].

## Limits

Survivorship and fixed-universe selection; non-financial large caps only; daily-return volatility proxy; historical vendor data may be revised; SEC standard-tag coverage varies; missing financial features are imputed from training data with missing flags; no transaction costs, returns, live deployment or human-efficiency claims. Broad sector groups are manually assigned. Interpret market regimes using contemporaneously observed SPY volatility and a training-only threshold. Demo dates are user-selected examples; assess the full test aggregate rather than a selected success.

## Artifacts

Open index.html for offline historical replay. Full per-company/date predictions, seed histories, coverage, paired block-bootstrap statistics and run/source hashes are retained in the local run and processed-data folders.
