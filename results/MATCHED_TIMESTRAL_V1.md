# Recent Transformer on the matched information-study cohort

Question: does the previously pinned Timestral5M transfer comparator change the interpretation when evaluated on exactly the same19 held companies and9,139 origins as the financial/text experiment?

This extension was registered after the matched Ridge/LSTM results were inspected. It is continued retrospective development,not an independent model-selection exercise or fresh confirmation. The checkpoint,code and point summary are unchanged. The runner reused46,869 raw full-market validation/evaluation forecasts after exact company-date,target andrv20 checks,and computed8,856 previously absent held-company validation forecasts. The broader-cohort calibration was discarded; calibration uses only the matched80 non-held training-company validation set.

| Procedure | Raw QLIKE | Validation-calibrated QLIKE |Calibrated MAE(volatility pp) |
|---|---:|---:|---:|
| Ridge_market | 0.388537 | 0.303285 | 11.286 |
| LSTM_market | 0.379093 | 0.294506 | 9.377 |
| Ridge_all | 0.358602 | 0.286470 | 11.050 |
| LSTM_all | 0.397257 | 0.307373 | 9.560 |
| Timestral5M | 0.322488 | 0.319908 | 11.026 |

The modern transfer model has higher mean calibrated QLIKE here than the market-only Ridge/LSTM,but both corresponding date-block interval lengths cross zero. Its difference from all-information Ridge is positive in both intervals. Different inputs,training procedures and point functionals prevent treating this as a pure architecture ablation or a claim that newer Transformers generally underperform.

| Calibrated Timestral minus comparator |Mean difference |20-date95% interval |60-date95% interval |
|---|---:|---|---|
| Ridge_market | +0.016623 | [-0.010187,+0.040295] | [-0.003773,+0.038804] |
| LSTM_market | +0.025402 | [-0.012900,+0.063185] | [-0.012022,+0.055991] |
| Ridge_all | +0.033438 | [+0.009860,+0.054062] | [+0.017223,+0.050722] |
| LSTM_all | +0.012535 | [-0.024304,+0.051377] | [-0.024573,+0.044622] |

## Decision application

The same registered baskets,cost/cash grid and execution rules yield96 additional strategy records. Raw and calibrated models have their own fixed equity fractions determined from earlier validation forecasts/prices,never held labels. Calibrated Timestral has lower mean risk-target error than its fixed control in the sector basket,and higher error in the technology trio;both20/60-date intervals cross zero in both baskets. Returns,drawdown,equity,turnover and costs remain in MATCHED_TIMESTRAL_PORTFOLIO_V1.csv. No reliable dynamic-policy advantage is established.

## Interpretation and temporal limit

Timestral receives512 trailingrv20 values and emits a quantile-derived RMS summary at the twenty-first future point. Ridge/LSTM receive60 sessions of ten market features,with optional financial/text inputs. Same origins and targets make the outcome comparison relevant; they do not equalize information sets. FinBERT is a frozen language encoder,not the modern temporal model. No supervised Transformer fusion was locally trained.

The2026 model release and unverified pretraining cutoff prevent a historical2024–2025 trading claim. Keeping local2026 prices sealed does not itself make2026 an unseen sample for a pretrained2026 model. Unless an independent cutoff excludes these outcomes,the modern model must remain outside a claimed fresh2026 primary validation(or be evaluated after its documented cutoff). The local2026 holdout remains unopened.

Sources: [pinned Timestral model](https://huggingface.co/mistralai/Timestral-5M/tree/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f),config/timestral-v1.json,docs/ALIGNED_METHOD_TRANSFER.md. Conditional bootstrap intervals do not include complete retraining uncertainty or multiplicity adjustment.
