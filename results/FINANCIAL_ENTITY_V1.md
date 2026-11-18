# Financial information: entity-spell sensitivity

The documented MTCH, DD, SW and KHC identity policy restricts unverified predecessor histories and disclosures before fitting. This is a conservative sensitivity, not exhaustive corporate-action certification. Earlier experiments remain preserved.

327 training companies /547,522 fitting origins;327 non-held validation companies /155,599 origins;80 held companies /38,480 evaluation origins. SW no longer supplies a512-observation current-spell context by2025. Companies are not replaced. Evaluation years2024–2025 were already inspected;2026 remains sealed.

| Information | Ridge calibrated QLIKE | LSTM calibrated QLIKE | LSTM raw QLIKE |
|---|---:|---:|---:|
| market | 0.312312 | 0.294007 | 0.358933 |
| filing_age | 0.298624 | 0.260836 | 0.312436 |
| availability | 0.297950 | 0.285014 | 0.333344 |
| financial | 0.294859 | 0.256343 | 0.317378 |

## Incremental information and uncertainty

| Calibrated LSTM difference | Mean QLIKE difference |20-date95% interval |60-date95% interval |
|---|---:|---|---|
| filing_age minus market | -0.033171 | [-0.047635, -0.018364] | [-0.048969, -0.015295] |
| financial minus availability | -0.028671 | [-0.053731, -0.009767] | [-0.053352, -0.010237] |
| financial minus filing_age | -0.004493 | [-0.012720, 0.003016] | [-0.012894, 0.004074] |

Lower loss is better. Financial-versus-combined-metadata evidence and financial-versus-age evidence answer different questions: adding seven missingness indicators can change neural fitting even though age is nested in the information set. The strongest substantive claim requires outperforming the simpler age control, not just one chosen metadata implementation. Report both raw and validation-calibrated losses; do not select whichever comparison is most favorable.

Intervals resample whole dates in20/60-session circular blocks and condition on the fitted ensemble. They do not capture all retraining/cohort uncertainty, adjust for multiple comparisons or create a fresh holdout. Same-origin old/new scores are in FINANCIAL_ENTITY_MATCHED_V1.csv; absolute81-versus80-company scores are not an identity correction effect.

Next: finish the already-frozen matched100-company multimodal experiment, including text age/length/availability controls. Translate candidate forecasts into decision outcomes under fixed rules and costs. A model-loss gain alone does not establish trading profit or alpha.
