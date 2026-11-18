# Matched financial and text information experiment

Research question: do financial values and annual-report language add predictive information beyond the same market history and disclosure metadata?

Fixed 100-issuer non-financial reference subset:80 fitting companies,130,065 fitting origins;80 non-held validation companies,37,730 origins;19 held companies,9,139 inspected2024–2025 origins. The twentieth held issuer(SW) lacks sufficient guarded current-security history; no replacement. Missing text remains in the sample. These are development results, not a fresh temporal confirmation.2026 remains sealed.

Ridge and three-seed LSTM use identical60-session market inputs. Each information group has the same52 static coordinates, masking excluded modalities. Frozen768-dimensional FinBERT vectors are reduced using475 unique fitting-period training-company documents only. Primary quality exclusions and identity boundaries were fixed before these scores. Financial/text availability,age and text-length controls are explicitly retained. Full details:TEXT_PIPELINE_100_V1.md and MULTIMODAL_DATA_CHECKS_100_V1.json.

| Information | Ridge raw QLIKE | Ridge calibrated | LSTM raw QLIKE | LSTM calibrated |
|---|---:|---:|---:|---:|
| Market | 0.388537 | 0.303285 | 0.379093 | 0.294506 |
| Financial metadata | 0.370332 | 0.291524 | 0.362995 | 0.281086 |
| Financial | 0.361725 | 0.287095 | 0.378116 | 0.287409 |
| Text metadata | 0.380973 | 0.301731 | 0.396303 | 0.304466 |
| Text | 0.378159 | 0.299644 | 0.396402 | 0.310104 |
| Combined metadata | 0.363354 | 0.290385 | 0.374381 | 0.289148 |
| Financial + text metadata | 0.358593 | 0.287084 | 0.378644 | 0.292849 |
| All | 0.358602 | 0.286470 | 0.397257 | 0.307373 |

## Findings within this specification

Frozen FinBERT representations do not show a stable incremental benefit here. Ridge text-versus-text-metadata and all-versus-financial+text-metadata differences are small and both block intervals cross zero. For LSTM, adding embeddings conditional on financial values and text metadata increases calibrated QLIKE, with both date-block intervals above zero. This applies to the frozen document pooling/PCA/fusion procedure and observed cohort; it does not establish that language is universally uninformative.

Ridge financial values improve calibrated QLIKE over its combined age/missingness control in both date intervals. The LSTM financial-content contrast crosses zero. A separately registered simple-age sensitivity addresses the earlier finding that age alone can fit better than the larger metadata control; do not select only the favorable comparator.

## Declared incremental contrasts

| Calibrated procedure difference | Mean QLIKE difference |20-date95% interval |60-date95% interval |
|---|---:|---|---|
| Ridge: financial minus financial_control | -0.004429 | [-0.007826, -0.001583] | [-0.008425, -0.001518] |
| Ridge: text minus text_control | -0.002087 | [-0.005013, +0.001182] | [-0.005330, +0.001024] |
| Ridge: all minus financial_text_control | -0.000614 | [-0.003127, +0.002058] | [-0.002915, +0.001609] |
| Ridge: all minus metadata | -0.003915 | [-0.007617, -0.000173] | [-0.008284, -0.000106] |
| Ridge: financial minus market | -0.016190 | [-0.023036, -0.009380] | [-0.022333, -0.009914] |
| Ridge: text minus market | -0.003641 | [-0.007502, +0.000640] | [-0.007733, +0.000625] |
| Ridge: all minus market | -0.016815 | [-0.024935, -0.007514] | [-0.024243, -0.008313] |
| LSTM: financial minus financial_control | +0.006323 | [-0.000739, +0.012926] | [-0.000736, +0.012231] |
| LSTM: text minus text_control | +0.005638 | [-0.003365, +0.014141] | [-0.004214, +0.015313] |
| LSTM: all minus financial_text_control | +0.014524 | [+0.007277, +0.022107] | [+0.008107, +0.021652] |
| LSTM: all minus metadata | +0.018226 | [+0.007663, +0.028033] | [+0.008733, +0.026949] |
| LSTM: financial minus market | -0.007097 | [-0.018122, +0.003440] | [-0.017425, +0.003156] |
| LSTM: text minus market | +0.015598 | [+0.006384, +0.024336] | [+0.008367, +0.022481] |
| LSTM: all minus market | +0.012867 | [-0.001596, +0.026665] | [+0.001137, +0.025005] |

Financial-control means filing age and missingness; text-control means document age,length,availability and newer-original-unavailable indicator. All-minus-financial+text-control isolates the embedding contribution conditional on financial values and text metadata. All-minus-market alone does not identify language content value.

Both raw and calibrated results are retained. Calibration and Ridge alpha/early stopping use only non-held validation outcomes. Member-seed diagnostics use the same ensemble calibration factor; ensemble loss is not the average member loss. Dates are resampled together across companies using20/60-session circular blocks. Intervals condition on the fitted procedure/cohort, omit full retraining uncertainty and are not multiplicity-adjusted. One negative or positive result cannot establish universal multimodal effectiveness.

Year/company tables and all seed outcomes accompany this report. The application grid is frozen separately in config/multimodal-portfolio-v1.json. Forecast loss is not net decision utility: examine risk-target error,return,drawdown,mean equity,turnover and costs against validation-fixed exposure and historical risk. No portfolio result is inferred from this table alone.
