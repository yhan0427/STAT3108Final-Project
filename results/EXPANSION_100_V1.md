# Expanded-company training v1

100 candidates were frozen before downloading outcomes; 99 retained after WBD history acquisition failed. No replacement was selected. 64 firms actually supply training/validation labels from the 65-company fitting-role cohort (Q has insufficient early history); 34 held firms never supply fitting, selection or calibration labels. Historical 2024-2025 development periods were already inspected on the original cohort. This is a new-company evaluation, not a new untouched time period.

| Role | Model | Companies | Origins | MAE (pp) | QLIKE |
|---|---|---:|---:|---:|---:|
| held_company | Persistence | 34 | 16354 | 10.370 | 0.5556 |
| held_company | Ridge60 | 34 | 16354 | 7.687 | 0.4454 |
| held_company | LSTM60 | 34 | 16354 | 7.355 | 0.3688 |
| held_company | Timestral5M | 34 | 16354 | 9.051 | 0.3625 |
| held_company | Persistence_calibrated | 34 | 16354 | 11.666 | 0.4533 |
| held_company | Ridge60_calibrated | 34 | 16354 | 8.485 | 0.3318 |
| held_company | LSTM60_calibrated | 34 | 16354 | 7.983 | 0.2969 |
| held_company | Timestral5M_calibrated | 34 | 16354 | 9.116 | 0.3605 |
| new_training | Persistence | 34 | 16354 | 10.262 | 0.4971 |
| new_training | Ridge60 | 34 | 16354 | 7.681 | 0.3904 |
| new_training | LSTM60 | 34 | 16354 | 7.307 | 0.3084 |
| new_training | Timestral5M | 34 | 16354 | 9.171 | 0.3084 |
| new_training | Persistence_calibrated | 34 | 16354 | 11.489 | 0.4085 |
| new_training | Ridge60_calibrated | 34 | 16354 | 8.392 | 0.2905 |
| new_training | LSTM60_calibrated | 34 | 16354 | 7.824 | 0.2479 |
| new_training | Timestral5M_calibrated | 34 | 16354 | 9.236 | 0.3070 |
| original_training | Persistence | 30 | 14430 | 9.761 | 0.5471 |
| original_training | Ridge60 | 30 | 14430 | 7.443 | 0.4114 |
| original_training | LSTM60 | 30 | 14430 | 7.011 | 0.3580 |
| original_training | Timestral5M | 30 | 14430 | 8.244 | 0.3255 |
| original_training | Persistence_calibrated | 30 | 14430 | 10.958 | 0.4436 |
| original_training | Ridge60_calibrated | 30 | 14430 | 8.065 | 0.3058 |
| original_training | LSTM60_calibrated | 30 | 14430 | 7.366 | 0.2799 |
| original_training | Timestral5M_calibrated | 30 | 14430 | 8.291 | 0.3233 |

Ridge and LSTM share the same 60-session market histories and training-only preprocessing. Three LSTM seeds are averaged; validation chooses epochs and Ridge alpha. Timestral uses 512 rv20 observations and pretrained weights; architecture, context and information sets differ. Its pretraining overlap is unverified. Calibrated variants use labels only from the non-held validation firms.

Scope: a reconstructed reference-date cohort with explicit official December corrections, not full historical membership. Missing WBD data can induce selection bias. Raw and eligible counts are audited, not forced to a resume target. Company histories can start after 2014; no missing prices were fabricated. 2026 remains sealed. All company and conditional date-block diagnostics are retained.

Year-specific metrics are retained in EXPANSION_100_YEARS_V1.csv. The raw-data ledger counts 291,416 company-date rows, not OHLCV cells.
