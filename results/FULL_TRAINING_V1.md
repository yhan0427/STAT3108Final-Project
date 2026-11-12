# Full-cohort unified model training v1

All 500 reconstructed reference-date issuers were frozen before full acquisition. Coverage/exclusions and symbol-only recovery are retained in FULL_COHORT_AUDIT_V2.json. Only training-company labels fit parameters, hyperparameters and calibration; all previously scored pilot issuers are excluded from the new held-company role. Historical 2024-2025 development periods were already inspected on the original cohort. This is a new-company evaluation, not a new untouched time period.

| Role | Model | Companies | Origins | MAE (pp) | QLIKE |
|---|---|---:|---:|---:|---:|
| held_company | Persistence | 97 | 46657 | 11.472 | 0.6229 |
| held_company | Ridge60 | 97 | 46657 | 8.966 | 0.5057 |
| held_company | LSTM60 | 97 | 46657 | 8.299 | 0.4606 |
| held_company | Timestral5M | 97 | 46657 | 10.143 | 0.4167 |
| held_company | Persistence_calibrated | 97 | 46657 | 12.953 | 0.4992 |
| held_company | Ridge60_calibrated | 97 | 46657 | 10.139 | 0.3731 |
| held_company | LSTM60_calibrated | 97 | 46657 | 9.211 | 0.3533 |
| held_company | Timestral5M_calibrated | 97 | 46657 | 10.281 | 0.4122 |

Ridge and LSTM share the same 60-session market histories and training-only preprocessing. Three LSTM seeds are averaged; validation chooses epochs and Ridge alpha. Timestral uses 512 rv20 observations and pretrained weights; architecture, context and information sets differ. Its pretraining overlap is unverified. Calibrated variants use labels only from the non-held validation firms.

Scope: a reconstructed reference-date cohort with explicit official December corrections, not full historical membership. Unavailable histories and current-vendor coverage can induce selection bias. Streamed float64 Ridge sufficient statistics match the standard regularized intercept solution in regression checks. Timestral input is stored temporarily in float32 to bound memory usage. Raw and eligible counts are audited, not forced to a resume target. Company histories can start after 2014; no missing prices were fabricated. 2026 remains sealed. All company and conditional date-block diagnostics are retained.

## Uncertainty and aggregation

Calibrated LSTM minus calibrated Ridge mean QLIKE = -0.01974. The paired 20-date interval is [-0.06850, +0.02612]; the 60-date interval is [-0.08325, +0.03934]. Both cross zero. A lower point estimate is not a stable superiority claim. Calibrated Timestral minus Ridge = +0.03912; the 20-date interval crosses zero and the 60-date interval does not. These conditional diagnostics omit refitting, cohort-selection and multiplicity uncertainty.

All yearly rows are retained in FULL_TRAINING_PERIODS_V1.csv; equal-company normalized loss ratios and win counts are in FULL_TRAINING_COMPANY_RATIOS_V1.csv. Forecast errors are not portfolio returns or evidence of alpha. The next application stage must evaluate risk-target errors alongside returns, drawdown, equity exposure, turnover and costs using these frozen forecasts. The 2026 temporal holdout remains sealed.
