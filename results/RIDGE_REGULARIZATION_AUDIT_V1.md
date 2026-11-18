# Ridge regularization robustness audit

The earlier full-market and primary financial Ridge grids selected their upper regularization endpoint. This exploratory audit uses one broad fixed grid, validation-only selection and unchanged inputs, splits, roles and preprocessing. All earlier results remain recorded. This is not a new evaluation sample.

| Population | Model | Alpha | MAE (pp) | QLIKE |
|---|---|---:|---:|---:|
| full_market | Ridge_market | 1e+06 | 8.686 | 0.493952 |
| full_market | Ridge_market_calibrated | 1e+06 | 9.682 | 0.364635 |
| non_financial | Ridge_market | 1e+06 | 8.974 | 0.423568 |
| non_financial | Ridge_market_calibrated | 1e+06 | 9.839 | 0.313301 |
| non_financial | Ridge_availability | 100000 | 8.965 | 0.401220 |
| non_financial | Ridge_availability_calibrated | 100000 | 9.848 | 0.301494 |
| non_financial | Ridge_financial | 100000 | 8.954 | 0.400098 |
| non_financial | Ridge_financial_calibrated | 100000 | 9.744 | 0.298568 |

All selected alphas are interior to the declared grid. Full-market evaluation has 97 firms / 46,657 origins; the non-financial information experiment has 81 / 38,961. Their absolute losses must not be interpreted as an information treatment.

## Full-market neural comparison after baseline strengthening

| Forecast minus broad Ridge | Mean QLIKE difference | 20-date 95% interval | 60-date 95% interval |
|---|---:|---|---|
| LSTM60_calibrated-minus-broad-Ridge | -0.011325 | [-0.045714, 0.025831] | [-0.057778, 0.034904] |
| Timestral5M_calibrated-minus-broad-Ridge | 0.047533 | [0.010596, 0.090603] | [0.011640, 0.093097] |

Calibrated Ridge improves from the original 0.373051 to 0.364635. LSTM retains a lower mean loss (0.353310), but both conditional date-block intervals for its difference cross zero. The pinned Timestral transfer procedure has higher loss (0.412168), with both intervals above zero in this audit. These are procedure-level findings on this reconstructed cohort; Timestral differs in context, input variables and point functional, with unverified pretraining overlap. They do not establish that contemporary architectures generally fail.

Independent numerical checks reconstruct the old alpha=10000 solution on 200 forecast cases with maximum absolute prediction difference 0. All full-market forecast keys/targets align, and all fourteen published primary financial summaries reconcile with the stored predictions.

The grid was broadened after inspecting earlier development results. Validation alone selects alpha; this does not convert the historical years or already-scored companies into untouched evidence. The 2026 market holdout remains sealed.
