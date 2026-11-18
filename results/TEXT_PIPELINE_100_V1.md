# Frozen annual-text corpus and quality gate

The fixed100-issuer sample contains80 training-role and20 held-role non-financial reference issuers, selected before inspecting text availability or forecast outcomes. It is a subset of the reconstructed2025-12-31 reference cohort, not historical index membership. Missing text does not cause company replacement.

## Collection and structural checks

- 1,206 original10-K filings;1,059 candidate sections;147 extractor abstentions. Unsupported foreign forms and amendments do not silently substitute for original10-K text.
- Frozen extractor v6. Review sample:44 flagged candidates plus130 sector/year hash-selected ordinary candidates.
- All44 flags inspected:13 accepted,31 excluded. Ordinary sample:129 accepted,1 excluded. Twelve additional BSX reports excluded following a supplementary attestation screen;26 total matches(BSX/PEP), all inspected.
- 44 candidate sections excluded in total. Reasons include separate Item7A market-risk content, auditor/ICFR report contamination and incorporation-by-reference stubs.
- Agent-assisted boundaries/interior checks are not independent human certification. Unreviewed ordinary sections are eligible under sampled QA; they are not individually certified. The stratified sample is not a simple random estimate of corpus-wide accuracy.

## Representations and timing

- 1,015 structurally eligible candidate vectors computed. Complete accepted sections are chunked into510 content tokens; frozen final-layer token means are weighted by content-token counts. Three sentiment probabilities are separate from768-dimensional embeddings.
- Conservative identity-spell filtering removes10 earlier disclosures before the matched dataset. It contains1005 retained document vectors.
- PCA uses475 unique documents actually available to pre2022 training-company fitting origins;32 components, explained variance fraction0.9258. No held-company or later document fits PCA. Daily component scaling also uses fitting origins only.
- Join strictly after the filing calendar date; text becomes unavailable after550 days. Filing age, length, availability and newer-original-unavailable status are explicit controls.
- All202 current FinBERT tensors are bitwise equal to the repository original2020 checkpoint. The2026 safetensors conversion is not new model training. This check does not certify all pretraining/tokenizer provenance or historical deployability of the full data pipeline.

## Matched experiment completed

80 fitting companies /130,065 origins;80 non-held validation companies /37,730 origins;19 held companies /9,139 inspected2024–2025 origins. SW cannot supply the guarded512-history context; it is not replaced. Held-company2022–2023 forecasts are saved for subsequent fixed-policy application evaluation, never used for forecast selection/calibration.

Eight equal-capacity information groups: market,financial-control,financial,text-control,text,combined-metadata,financial+text-control,all. Financial-control retains filing age and missingness; text-control retains age,length,availability and newer-unavailable indicator. Ridge and LSTM share60-session market input; every LSTM has52 static coordinates, masking unused inputs. Three seeds and identical epoch budgets. This is not a locally trained temporal Transformer.

Forecast and application results are now complete:see MULTIMODAL_100_V1.md and MULTIMODAL_PORTFOLIO_V1.md. Frozen embeddings do not establish stable incremental value in this specification.2026 prices remain sealed.

Artifacts: TEXT_CORPUS_REVIEW_100_V1.json;FINBERT_CORPUS_100_V1.json;FINBERT_WEIGHT_PROVENANCE_V1.json;MULTIMODAL_DATA_100_V1.json. Raw reports,review excerpts,weights,vectors and fitted models remain local/ignored.
