# Multimodal failure and mechanism diagnostics

These are post-result descriptions of the fixed19-company development cohort. No model or encoder was retrained,company replaced or subgroup selected for a primary positive claim. Negative differences favor the first procedure.

| Calibrated contrast |2024 mean QLIKE difference |2025 mean QLIKE difference |Companies improved(out of19) |Leave-one-company-out mean range |
|---|---:|---:|---:|---|
| Ridge_financial-minus-Ridge_filing_age | -0.007067 | -0.001735 | 15 | [-0.005145, -0.002728] |
| Ridge_text-minus-Ridge_text_control | -0.003797 | -0.000205 | 15 | [-0.003056, -0.000844] |
| Ridge_all-minus-Ridge_financial_text_control | -0.001480 | +0.000338 | 10 | [-0.001421, +0.000432] |
| Ridge_all-minus-Ridge_market | -0.021251 | -0.011933 | 18 | [-0.019258, -0.015018] |
| LSTM_financial-minus-LSTM_filing_age | +0.005755 | +0.005586 | 8 | [+0.001492, +0.009425] |
| LSTM_text-minus-LSTM_text_control | +0.004957 | +0.006388 | 7 | [+0.000634, +0.016297] |
| LSTM_all-minus-LSTM_financial_text_control | +0.018387 | +0.010273 | 8 | [+0.006988, +0.019238] |
| LSTM_all-minus-LSTM_market | +0.012613 | +0.013147 | 8 | [+0.000986, +0.015712] |

The supplementary simple-age control was registered before the primary matched forecast scores were produced, following the larger-cohort age finding. Ridge financial values improve over both age-only and combined financial metadata in this cohort; LSTM financial values do not show a reliable improvement over age alone. Financial content value is therefore backbone/cohort dependent,not an established universal gain.

The CSV retains every company,year and text-availability/age bin. Unavailable text is retained,not deleted. A procedure difference within an unavailable-text subgroup can reflect how shared parameters were fitted on other companies; it is not a within-company causal text effect. Year signs and company counts are descriptive and do not make nineteen firms a representative sample of all equities.

Leave-one-company-out values remove one company only when rescoring fixed predictions. They detect concentrated average losses,not sensitivity to retraining or leave-one-company-out deployment. The existing date-block intervals retain shared market dates; fuller company/generalization uncertainty remains for later validation.

Next research decision: stabilize and audit the information contribution before expanding an expensive text grid. Full-section frozen pooled embeddings can dilute rare risk information; this is a hypothesis,not a demonstrated explanation. Any event-window,section-aware or alternative-encoder extension needs a new declared development protocol and must preserve the sealed2026 sample.
