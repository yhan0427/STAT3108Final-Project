# Exploratory robustness results — v2

These experiments were designed after inspecting v1. They are diagnostics, not an untouched final test or a published-paper replication.

## Main findings

Matched-history Ridge is a competitive baseline. The LSTM advantage depends on year and metric. Complete accounting values do not show consistent incremental benefits over the missingness/filing-age control in these specifications. These results concern the implemented feature set with incomplete public accounting coverage, not all potential financial information.

| Evaluation | Persistence QLIKE | Sequence Ridge QLIKE | Market LSTM QLIKE |
|---|---:|---:|---:|
| origin_2020 | 1.515 | 1.469 | 1.966 |
| origin_2022 | 0.336 | 0.286 | 0.249 |
| origin_2024 | 0.500 | 0.283 | 0.294 |
| unseen_companies | 0.404 | 0.281 | 0.239 |

For the eight-company evaluation, full-fundamental LSTM has higher QLIKE than market-only LSTM; the paired intervals below quantify the difference. Do not compare its raw top-five overlap with the thirty-company protocols. Further scope expansion should first improve accounting coverage and predefine the new sample/evaluation design.

## static_matched

30 evaluation companies. Random top-five expected overlap: 16.7%. Compare screening scores only within this company pool.

| Model | MAE (pp) | RMSE (pp) | QLIKE | Top-five overlap |
|---|---:|---:|---:|---:|
| Persistence | 9.69 | 14.42 | 0.541 | 45.7% |
| Ridge_snapshot_market | 7.52 | 11.99 | 0.432 | 50.5% |
| Ridge_snapshot_missingness | 7.39 | 11.83 | 0.411 | 51.7% |
| Ridge_snapshot_fundamentals | 7.44 | 11.94 | 0.439 | 54.1% |
| Ridge_sequence_market | 7.34 | 11.87 | 0.383 | 58.7% |
| Ridge_sequence_missingness | 7.21 | 11.71 | 0.371 | 58.8% |
| Ridge_sequence_fundamentals | 7.20 | 11.72 | 0.379 | 58.8% |
| Boosting_market | 7.66 | 12.40 | 0.411 | 51.0% |
| Boosting_missingness | 7.37 | 12.03 | 0.379 | 55.5% |
| Boosting_fundamentals | 7.42 | 12.08 | 0.427 | 56.2% |
| LSTM_market | 7.00 | 11.52 | 0.365 | 54.4% |
| LSTM_missingness | 6.84 | 11.30 | 0.344 | 57.0% |
| LSTM_fundamentals | 6.99 | 11.57 | 0.382 | 56.4% |

Paired QLIKE intervals: negative favors the first model.

- LSTM_fundamentals minus LSTM_market, 20-day blocks: 0.0164, 95% CI [-0.0000, 0.0377].
- LSTM_fundamentals minus LSTM_market, 60-day blocks: 0.0164, 95% CI [-0.0003, 0.0372].
- LSTM_fundamentals minus LSTM_missingness, 20-day blocks: 0.0382, 95% CI [0.0199, 0.0597].
- LSTM_fundamentals minus LSTM_missingness, 60-day blocks: 0.0382, 95% CI [0.0159, 0.0654].
- LSTM_market minus Ridge_sequence_market, 20-day blocks: -0.0176, 95% CI [-0.0686, 0.0270].
- LSTM_market minus Ridge_sequence_market, 60-day blocks: -0.0176, 95% CI [-0.0593, 0.0152].
- Boosting_fundamentals minus Boosting_missingness, 20-day blocks: 0.0476, 95% CI [0.0232, 0.0728].
- Boosting_fundamentals minus Boosting_missingness, 60-day blocks: 0.0476, 95% CI [0.0182, 0.0810].

## origin_2020

30 evaluation companies. Random top-five expected overlap: 16.7%. Compare screening scores only within this company pool.

| Model | MAE (pp) | RMSE (pp) | QLIKE | Top-five overlap |
|---|---:|---:|---:|---:|
| Persistence | 19.05 | 28.38 | 1.515 | 47.6% |
| Ridge_snapshot_market | 14.54 | 24.87 | 1.577 | 43.6% |
| Ridge_snapshot_missingness | 14.57 | 25.06 | 1.588 | 43.9% |
| Ridge_snapshot_fundamentals | 15.07 | 25.71 | 1.738 | 39.0% |
| Ridge_sequence_market | 15.12 | 24.84 | 1.469 | 48.8% |
| Ridge_sequence_missingness | 15.12 | 24.96 | 1.484 | 49.6% |
| Ridge_sequence_fundamentals | 15.00 | 24.99 | 1.535 | 47.2% |
| Boosting_market | 15.36 | 25.82 | 1.830 | 37.1% |
| Boosting_missingness | 15.05 | 25.73 | 1.832 | 38.4% |
| Boosting_fundamentals | 16.82 | 27.88 | 2.377 | 27.0% |
| LSTM_market | 16.07 | 26.56 | 1.966 | 36.1% |
| LSTM_missingness | 15.85 | 26.53 | 1.962 | 42.6% |
| LSTM_fundamentals | 15.92 | 26.54 | 1.973 | 37.7% |

Paired QLIKE intervals: negative favors the first model.

- LSTM_fundamentals minus LSTM_market, 20-day blocks: 0.0069, 95% CI [-0.0582, 0.0907].
- LSTM_fundamentals minus LSTM_market, 60-day blocks: 0.0069, 95% CI [-0.0700, 0.1019].
- LSTM_fundamentals minus LSTM_missingness, 20-day blocks: 0.0116, 95% CI [-0.0286, 0.0394].
- LSTM_fundamentals minus LSTM_missingness, 60-day blocks: 0.0116, 95% CI [-0.0295, 0.0411].
- LSTM_market minus Ridge_sequence_market, 20-day blocks: 0.4976, 95% CI [0.2038, 0.8750].
- LSTM_market minus Ridge_sequence_market, 60-day blocks: 0.4976, 95% CI [0.2034, 0.9139].
- Boosting_fundamentals minus Boosting_missingness, 20-day blocks: 0.5452, 95% CI [0.1578, 1.1178].
- Boosting_fundamentals minus Boosting_missingness, 60-day blocks: 0.5452, 95% CI [0.1420, 1.1823].

## origin_2022

30 evaluation companies. Random top-five expected overlap: 16.7%. Compare screening scores only within this company pool.

| Model | MAE (pp) | RMSE (pp) | QLIKE | Top-five overlap |
|---|---:|---:|---:|---:|
| Persistence | 8.69 | 11.55 | 0.336 | 62.2% |
| Ridge_snapshot_market | 7.60 | 10.42 | 0.283 | 67.7% |
| Ridge_snapshot_missingness | 7.76 | 10.69 | 0.304 | 69.2% |
| Ridge_snapshot_fundamentals | 7.47 | 10.32 | 0.286 | 66.0% |
| Ridge_sequence_market | 7.79 | 11.62 | 0.286 | 74.6% |
| Ridge_sequence_missingness | 7.69 | 11.41 | 0.281 | 74.8% |
| Ridge_sequence_fundamentals | 7.58 | 11.23 | 0.273 | 74.4% |
| Boosting_market | 8.21 | 11.57 | 0.298 | 66.9% |
| Boosting_missingness | 8.00 | 11.24 | 0.292 | 64.6% |
| Boosting_fundamentals | 7.75 | 10.75 | 0.272 | 57.7% |
| LSTM_market | 7.13 | 9.92 | 0.249 | 75.2% |
| LSTM_missingness | 6.77 | 9.32 | 0.212 | 74.2% |
| LSTM_fundamentals | 7.01 | 9.74 | 0.226 | 68.4% |

Paired QLIKE intervals: negative favors the first model.

- LSTM_fundamentals minus LSTM_market, 20-day blocks: -0.0226, 95% CI [-0.0390, -0.0076].
- LSTM_fundamentals minus LSTM_market, 60-day blocks: -0.0226, 95% CI [-0.0363, -0.0085].
- LSTM_fundamentals minus LSTM_missingness, 20-day blocks: 0.0148, 95% CI [0.0045, 0.0283].
- LSTM_fundamentals minus LSTM_missingness, 60-day blocks: 0.0148, 95% CI [0.0040, 0.0297].
- LSTM_market minus Ridge_sequence_market, 20-day blocks: -0.0372, 95% CI [-0.0696, -0.0085].
- LSTM_market minus Ridge_sequence_market, 60-day blocks: -0.0372, 95% CI [-0.0680, -0.0133].
- Boosting_fundamentals minus Boosting_missingness, 20-day blocks: -0.0207, 95% CI [-0.0352, -0.0049].
- Boosting_fundamentals minus Boosting_missingness, 60-day blocks: -0.0207, 95% CI [-0.0317, -0.0084].

## origin_2024

30 evaluation companies. Random top-five expected overlap: 16.7%. Compare screening scores only within this company pool.

| Model | MAE (pp) | RMSE (pp) | QLIKE | Top-five overlap |
|---|---:|---:|---:|---:|
| Persistence | 8.19 | 11.83 | 0.500 | 45.9% |
| Ridge_snapshot_market | 6.43 | 9.75 | 0.346 | 53.3% |
| Ridge_snapshot_missingness | 6.27 | 9.51 | 0.323 | 54.5% |
| Ridge_snapshot_fundamentals | 6.24 | 9.48 | 0.338 | 57.2% |
| Ridge_sequence_market | 6.06 | 9.35 | 0.283 | 66.7% |
| Ridge_sequence_missingness | 5.97 | 9.19 | 0.273 | 67.2% |
| Ridge_sequence_fundamentals | 5.99 | 9.25 | 0.275 | 66.7% |
| Boosting_market | 6.29 | 9.67 | 0.343 | 56.9% |
| Boosting_missingness | 5.94 | 9.16 | 0.286 | 62.0% |
| Boosting_fundamentals | 5.95 | 9.16 | 0.303 | 61.6% |
| LSTM_market | 5.91 | 9.19 | 0.294 | 59.1% |
| LSTM_missingness | 5.75 | 8.92 | 0.267 | 63.7% |
| LSTM_fundamentals | 5.81 | 9.09 | 0.289 | 63.7% |

Paired QLIKE intervals: negative favors the first model.

- LSTM_fundamentals minus LSTM_market, 20-day blocks: -0.0057, 95% CI [-0.0180, 0.0047].
- LSTM_fundamentals minus LSTM_market, 60-day blocks: -0.0057, 95% CI [-0.0126, 0.0008].
- LSTM_fundamentals minus LSTM_missingness, 20-day blocks: 0.0216, 95% CI [0.0038, 0.0443].
- LSTM_fundamentals minus LSTM_missingness, 60-day blocks: 0.0216, 95% CI [0.0079, 0.0360].
- LSTM_market minus Ridge_sequence_market, 20-day blocks: 0.0116, 95% CI [-0.0010, 0.0245].
- LSTM_market minus Ridge_sequence_market, 60-day blocks: 0.0116, 95% CI [0.0045, 0.0182].
- Boosting_fundamentals minus Boosting_missingness, 20-day blocks: 0.0167, 95% CI [-0.0044, 0.0366].
- Boosting_fundamentals minus Boosting_missingness, 60-day blocks: 0.0167, 95% CI [-0.0004, 0.0349].

## unseen_companies

8 evaluation companies. Random top-five expected overlap: 62.5%. Compare screening scores only within this company pool.

| Model | MAE (pp) | RMSE (pp) | QLIKE | Top-five overlap |
|---|---:|---:|---:|---:|
| Persistence | 7.39 | 10.36 | 0.404 | 75.9% |
| Ridge_snapshot_market | 5.80 | 8.41 | 0.283 | 76.8% |
| Ridge_snapshot_missingness | 6.08 | 8.42 | 0.255 | 72.4% |
| Ridge_snapshot_fundamentals | 6.32 | 8.85 | 0.315 | 65.2% |
| Ridge_sequence_market | 5.68 | 8.38 | 0.281 | 80.0% |
| Ridge_sequence_missingness | 5.53 | 8.22 | 0.262 | 80.7% |
| Ridge_sequence_fundamentals | 5.60 | 8.39 | 0.286 | 80.1% |
| Boosting_market | 6.16 | 9.35 | 0.277 | 77.1% |
| Boosting_missingness | 6.02 | 8.93 | 0.242 | 75.2% |
| Boosting_fundamentals | 6.30 | 9.21 | 0.369 | 69.6% |
| LSTM_market | 5.39 | 7.98 | 0.239 | 80.4% |
| LSTM_missingness | 5.42 | 7.90 | 0.221 | 80.0% |
| LSTM_fundamentals | 5.77 | 8.46 | 0.278 | 73.7% |

Paired QLIKE intervals: negative favors the first model.

- LSTM_fundamentals minus LSTM_market, 20-day blocks: 0.0384, 95% CI [0.0230, 0.0555].
- LSTM_fundamentals minus LSTM_market, 60-day blocks: 0.0384, 95% CI [0.0267, 0.0511].
- LSTM_fundamentals minus LSTM_missingness, 20-day blocks: 0.0570, 95% CI [0.0396, 0.0765].
- LSTM_fundamentals minus LSTM_missingness, 60-day blocks: 0.0570, 95% CI [0.0425, 0.0720].
- LSTM_market minus Ridge_sequence_market, 20-day blocks: -0.0421, 95% CI [-0.0962, -0.0019].
- LSTM_market minus Ridge_sequence_market, 60-day blocks: -0.0421, 95% CI [-0.1027, 0.0031].
- Boosting_fundamentals minus Boosting_missingness, 20-day blocks: 0.1276, 95% CI [0.0782, 0.1862].
- Boosting_fundamentals minus Boosting_missingness, 60-day blocks: 0.1276, 95% CI [0.0753, 0.1792].

## Interpretation limits

Three selected annual origins do not establish universal robustness. The eight held-out firms were not used for this experiment’s fitting or selection, but were previously examined in v1. Missingness controls can reveal reporting/identity effects; they do not establish causation. SEC accounting coverage, survivorship selection, vendor revisions, non-financial large-cap scope and daily-return volatility measurement remain limitations. A different period or company pool is a different forecasting task. Do not pool overlapping protocol rows or choose a model based on the best evaluation seed.
