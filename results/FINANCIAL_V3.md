# Alternative accounting features — exploratory v3

Completed three registered protocols, thirteen model specifications each and twenty-seven neural seed fits. No company or text expansion. All original panel columns and all matched market-only forecasts are identical to v2. Original v1/v2 artifacts remain intact.

## Definitions and coverage

Replace total liabilities/assets with parent stockholders’ equity/assets; replace operating margin with current liabilities/assets. These are different economic features, not filled versions of the old features. The parent-equity measure excludes noncontrolling interests and must not be used to infer total liabilities. Negative equity is retained. The same eight-dimensional accounting input, missingness indicators, architecture, seeds and evaluation observations are used.

| Field | Mean coverage across thirty issuers |
|---|---:|
| liabilities_to_assets | 61.1% |
| operating_margin | 65.8% |
| parent_equity_to_assets | 85.5% |
| current_liabilities_to_assets | 100.0% |
| pretax_margin | 45.7% |
| liability_components_to_assets | 9.2% |

Coverage is measured over eligible calendar-split rows and averaged across issuers, not independent observations. Pretax margin was rejected on coverage alone before model fitting. Current plus noncurrent liabilities are not used to fill total liabilities: both same-period components are available too rarely. Split-level coverage is in FINANCIAL_V3_COVERAGE.csv.

## Matched QLIKE comparisons

Lower is better. Each LSTM forecast averages the same three registered seeds, without selecting evaluation winners.

| Evaluation | Market LSTM (both schemas) | V2 missingness | V3 missingness | V2 full ratios | V3 full ratios |
|---|---:|---:|---:|---:|---:|
| static_matched | 0.3655 | 0.3438 | 0.3471 | 0.3819 | 0.3754 |
| origin_2020 | 1.9662 | 1.9615 | 2.0123 | 1.9731 | 2.1354 |
| unseen_companies | 0.2392 | 0.2207 | 0.2353 | 0.2776 | 0.2677 |

## Within-v3 paired uncertainty

Daily mean loss differences; all firms on a date stay together in circular block resampling. Negative favors full financial values. These intervals are exploratory, conditional on the sample/fitted models, and do not capture retraining or unseen-company population uncertainty. Twenty-day labels overlap; no firm-day independence assumption.

| Evaluation | Full LSTM minus missingness | 20-day block 95% interval | 60-day block 95% interval |
|---|---:|---:|---:|
| static_matched | 0.0282 | [0.0153, 0.0438] | [0.0120, 0.0459] |
| origin_2020 | 0.1231 | [0.0532, 0.2182] | [0.0456, 0.2428] |
| unseen_companies | 0.0324 | [0.0215, 0.0451] | [0.0236, 0.0412] |

## Interpretation

Higher coverage alone has not established a financial-value benefit. Feature content and missingness patterns change together, so this experiment cannot identify a pure coverage effect or a causal mechanism. Compare the full ratios with their own schema’s missingness/filing-age control, not only the market baseline. Missingness may encode issuer/reporting differences.

The eight excluded companies were absent from this run’s fitting and validation but previously examined in v1. The 2020 protocol deliberately revisits a known failure period. The static and company-holdout evaluation periods overlap; do not pool them as independent replications. All three protocols were chosen after earlier results, so none is an untouched confirmatory test.

Next: audit missing tags by issuer and accounting definition, then preregister a comparable-feature common-coverage sensitivity and a genuinely new company/time holdout. Do not add text until there is a defensible incremental-information question. A larger sample alone cannot fix feature semantics or reporting bias.

## Verification

Original columns and parent panel hashes match. Market-only predictions match v2 exactly. Two current-liability ratios match previously transcribed original SEC statements. Synthetic timing/negative-equity tests cover the new concepts. This is limited verification, not certification of the full point-in-time archive. Ridge emitted ill-conditioned float32 matrix warnings in the registered alpha grid; fitted results are retained and this numerical limitation should be checked before confirmatory use.

## All registered model specifications

### static_matched

| Model | MAE (pp) | QLIKE | Top-five overlap |
|---|---:|---:|---:|
| Persistence | 9.69 | 0.5413 | 45.7% |
| Ridge_snapshot_market | 7.52 | 0.4323 | 50.5% |
| Ridge_snapshot_missingness | 7.48 | 0.4223 | 51.8% |
| Ridge_snapshot_fundamentals | 7.54 | 0.4488 | 53.3% |
| Ridge_sequence_market | 7.34 | 0.3831 | 58.7% |
| Ridge_sequence_missingness | 7.23 | 0.3722 | 58.8% |
| Ridge_sequence_fundamentals | 7.21 | 0.3795 | 58.7% |
| Boosting_market | 7.66 | 0.4112 | 51.0% |
| Boosting_missingness | 7.43 | 0.3776 | 54.0% |
| Boosting_fundamentals | 7.44 | 0.4290 | 55.1% |
| LSTM_market | 7.00 | 0.3655 | 54.4% |
| LSTM_missingness | 6.93 | 0.3471 | 55.6% |
| LSTM_fundamentals | 7.03 | 0.3754 | 57.1% |

Eight-firm raw top-five overlap must not be compared with thirty-firm overlap; random expectations are respectively 62.5% and 16.7%.

### origin_2020

| Model | MAE (pp) | QLIKE | Top-five overlap |
|---|---:|---:|---:|
| Persistence | 19.05 | 1.5147 | 47.6% |
| Ridge_snapshot_market | 14.54 | 1.5775 | 43.6% |
| Ridge_snapshot_missingness | 14.57 | 1.5889 | 43.6% |
| Ridge_snapshot_fundamentals | 15.05 | 1.7485 | 40.7% |
| Ridge_sequence_market | 15.12 | 1.4686 | 48.8% |
| Ridge_sequence_missingness | 15.14 | 1.4827 | 49.2% |
| Ridge_sequence_fundamentals | 14.95 | 1.5455 | 48.0% |
| Boosting_market | 15.36 | 1.8303 | 37.1% |
| Boosting_missingness | 15.17 | 1.8311 | 40.1% |
| Boosting_fundamentals | 16.98 | 2.4471 | 28.0% |
| LSTM_market | 16.07 | 1.9662 | 36.1% |
| LSTM_missingness | 15.98 | 2.0123 | 44.2% |
| LSTM_fundamentals | 16.32 | 2.1354 | 39.0% |

Eight-firm raw top-five overlap must not be compared with thirty-firm overlap; random expectations are respectively 62.5% and 16.7%.

### unseen_companies

| Model | MAE (pp) | QLIKE | Top-five overlap |
|---|---:|---:|---:|
| Persistence | 7.39 | 0.4043 | 75.9% |
| Ridge_snapshot_market | 5.80 | 0.2830 | 76.8% |
| Ridge_snapshot_missingness | 5.95 | 0.2799 | 72.0% |
| Ridge_snapshot_fundamentals | 6.41 | 0.3442 | 64.4% |
| Ridge_sequence_market | 5.68 | 0.2812 | 80.0% |
| Ridge_sequence_missingness | 5.54 | 0.2679 | 80.7% |
| Ridge_sequence_fundamentals | 5.59 | 0.2801 | 80.0% |
| Boosting_market | 6.16 | 0.2769 | 77.1% |
| Boosting_missingness | 6.02 | 0.2642 | 75.3% |
| Boosting_fundamentals | 6.32 | 0.3552 | 67.9% |
| LSTM_market | 5.39 | 0.2392 | 80.4% |
| LSTM_missingness | 5.37 | 0.2353 | 80.3% |
| LSTM_fundamentals | 5.62 | 0.2677 | 77.1% |

Eight-firm raw top-five overlap must not be compared with thirty-firm overlap; random expectations are respectively 62.5% and 16.7%.
