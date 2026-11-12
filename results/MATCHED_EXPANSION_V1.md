# Matched training-cohort control

Exploratory follow-up: same held 34 firms, original-30 validation firms, target, feature history, seed set and training budget. Only the fitting cohort and its training-fitted transformations change: 30 versus 64 companies. Alpha and early stopping are selected independently by the same validation rule; this estimates the effect of the expanded training procedure, not identical fitted parameters. 2026 remains sealed.

| Training cohort | Model | MAE (pp) | QLIKE |
|---|---|---:|---:|
| original30 | Ridge60 | 7.619 | 0.4103 |
| original30 | LSTM60 | 7.365 | 0.3592 |
| original30 | Ridge60_calibrated | 8.711 | 0.3210 |
| original30 | LSTM60_calibrated | 7.978 | 0.3023 |
| expanded64 | Ridge60 | 7.687 | 0.4454 |
| expanded64 | LSTM60 | 7.330 | 0.3743 |
| expanded64 | Ridge60_calibrated | 8.449 | 0.3326 |
| expanded64 | LSTM60_calibrated | 7.829 | 0.2985 |

Negative expanded-minus-original QLIKE differences favor expansion. Conditional date-block intervals are in the audit. Do not infer a population effect from this selected cohort or interpret the shared calendar period as a new temporal test.

## Pilot interpretation and stopping point

Calibrated LSTM QLIKE changes from 0.3023 (original30 training) to 0.2985 (expanded64 training). The expanded-minus-original difference is -0.00383; both 20/60-date bootstrap intervals cross zero. This does not establish a reliable benefit from training expansion. Ridge also has no reliable improvement. This control completes the pilot design check; do not repeat the full training grid at every incremental sample size. Freeze the final cohort/roles and evaluation plan before the next full-scale comparison.
