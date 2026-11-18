# Matched financial information experiment

Question: do financial values add information beyond the same market history, financial-field availability and filing age? This is an exploratory development study, not a fresh temporal test.

Data: 415 retained non-financial reference-cohort companies; 327 fitting firms / 550,253 origins; 327 non-held validation firms / 155,750 common-context origins; 81 held-role firms / 38,961 evaluation origins in 2024–2025. Missing financial values do not remove market observations. Company roles, execution-aligned target and training-only preprocessing are unchanged.

## Information controls

Every LSTM has the same sixteen-coordinate financial branch, architecture, three seeds and epoch budget. Market-only zeros every branch input. Availability retains filing age and eight missing flags while zeroing the seven economic values. Financial uses all values plus those controls. Ridge receives the same sixty-day market lag and current static coordinates. Each model has its own validation-only scalar calibration. The V3 quantities are alternatives with explicit economic definitions, not imputed total liabilities or operating margins.

## Primary frozen run

| Model | MAE (pp) | RMSE (pp) | QLIKE |
|---|---:|---:|---:|
| Persistence | 11.760 | 17.894 | 0.573315 |
| Ridge_market | 9.281 | 15.726 | 0.436758 |
| Ridge_availability | 9.096 | 15.459 | 0.403609 |
| Ridge_financial | 9.090 | 15.477 | 0.403805 |
| LSTM_market | 8.712 | 14.072 | 0.385260 |
| LSTM_availability | 8.267 | 13.592 | 0.341923 |
| LSTM_financial | 8.271 | 13.694 | 0.361475 |
| Persistence_calibrated | 13.225 | 20.011 | 0.463005 |
| Ridge_market_calibrated | 10.313 | 16.984 | 0.323855 |
| Ridge_availability_calibrated | 10.085 | 16.745 | 0.304354 |
| Ridge_financial_calibrated | 9.984 | 16.747 | 0.301540 |
| LSTM_market_calibrated | 9.525 | 14.010 | 0.303148 |
| LSTM_availability_calibrated | 9.197 | 13.532 | 0.268457 |
| LSTM_financial_calibrated | 8.895 | 13.306 | 0.261979 |

## Financial content beyond availability

| Backbone / regularization | Loss difference: financial − availability | 20-date 95% interval | 60-date 95% interval |
|---|---:|---|---|
| LSTM primary | -0.006478 | [-0.015555, 0.001771] | [-0.017107, 0.001604] |
| Ridge primary grid | -0.002814 | [-0.006432, 0.002445] | [-0.006308, 0.002476] |
| Ridge broad grid | -0.002926 | [-0.005539, 0.000680] | [-0.005443, 0.000581] |

Negative differences favor financial values. All financial-versus-availability intervals above cross zero. Calibrated LSTM QLIKE changes from 0.303148 (market) to 0.268457 (availability) to 0.261979 (financial). Most of this observed market-to-financial difference is also obtained by the availability/age control; this is not a causal percentage decomposition. The additional financial-content effect is uncertain, not proved absent.

Raw LSTM QLIKE is 0.341923 for availability versus 0.361475 for financial values: the ranking reverses after calibration. Calibrated financial MAE is lower in both development years, but two already-inspected years do not establish stable out-of-period improvement. Retain raw/calibrated metrics and all company/year outcomes.

## Stronger regularization audit

The primary Ridge grid selected its maximum alpha. A separate, explicitly exploratory broad-grid audit uses [0.01, 1, 100, 10000, 100000, 1000000, 10000000, 100000000] with validation log-MSE selection. All optima are now interior. Earlier experiments and scores are preserved.

| Population | Model | Selected alpha | MAE (pp) | QLIKE |
|---|---|---:|---:|---:|
| full_market | Ridge_market_calibrated | 1e+06 | 9.682 | 0.364635 |
| non_financial | Ridge_market_calibrated | 1e+06 | 9.839 | 0.313301 |
| non_financial | Ridge_availability_calibrated | 100000 | 9.848 | 0.301494 |
| non_financial | Ridge_financial_calibrated | 100000 | 9.744 | 0.298568 |

The full-market audit is a different population (97 evaluated firms) from the non-financial information experiment (81 firms). Do not compare their absolute losses as an information treatment. Expanded Ridge regularization does not establish a financial-content advantage over availability: its incremental intervals still cross zero. The earlier LSTM/Ridge evidence should be interpreted alongside this stronger-baseline audit rather than silently replacing the recorded historical grid.

## Interpretation and next decision

This motivates a separately registered age-versus-missingness diagnostic. Reporting-calendar information, taxonomy/industry differences and coverage can all influence the control; no one mechanism is established by the combined availability branch. Text integration must include analogous availability, age and length controls rather than interpreting every multimodal gain as semantic information.

Date-block intervals are conditional on fitted ensembles and the observed reference cohort. They do not capture all model-selection, company-population or data-source uncertainty, and no multiplicity adjustment or confirmatory claim is made. Original-filing spot checks are bounded; entity transitions, custom tags and the current SEC database's historical limitations remain. No portfolio benefit or alpha follows from these forecast losses alone. The 2026 market holdout remains sealed.
