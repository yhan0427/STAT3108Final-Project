# Financial availability and matched-sample feasibility

Coverage diagnostics only; no financial models or text embeddings fitted. Company-year percentages in FINANCIAL_COVERAGE_V1.md are supplemented here with split, company-role, context and joint-availability comparisons. Failed SEC companies remain in every applicable market denominator.

## Non-financial training companies: 60-session origins

| Period | Feature | Available origins | Eligible origins | Coverage |
|---|---|---:|---:|---:|
| test | alternative_v3_content_all_available | 108,285 | 158,976 | 68.1% |
| test | alternative_v3_content_any_available | 158,830 | 158,976 | 99.9% |
| test | cash_to_assets | 143,853 | 158,976 | 90.5% |
| test | cfo_to_assets | 150,032 | 158,976 | 94.4% |
| test | current_liabilities_to_assets | 147,767 | 158,976 | 92.9% |
| test | current_ratio | 147,767 | 158,976 | 92.9% |
| test | filing_age_days | 158,935 | 158,976 | 100.0% |
| test | income_to_assets | 155,314 | 158,976 | 97.7% |
| test | legacy_v1_content_all_available | 66,044 | 158,976 | 41.5% |
| test | legacy_v1_content_any_available | 158,830 | 158,976 | 99.9% |
| test | liabilities_to_assets | 103,022 | 158,976 | 64.8% |
| test | operating_margin | 120,451 | 158,976 | 75.8% |
| test | parent_equity_to_assets | 148,476 | 158,976 | 93.4% |
| test | revenue_growth | 144,877 | 158,976 | 91.1% |
| train | alternative_v3_content_all_available | 287,133 | 550,253 | 52.2% |
| train | alternative_v3_content_any_available | 531,983 | 550,253 | 96.7% |
| train | cash_to_assets | 510,550 | 550,253 | 92.8% |
| train | cfo_to_assets | 399,280 | 550,253 | 72.6% |
| train | current_liabilities_to_assets | 494,467 | 550,253 | 89.9% |
| train | current_ratio | 494,467 | 550,253 | 89.9% |
| train | filing_age_days | 533,793 | 550,253 | 97.0% |
| train | income_to_assets | 506,801 | 550,253 | 92.1% |
| train | legacy_v1_content_all_available | 165,656 | 550,253 | 30.1% |
| train | legacy_v1_content_any_available | 531,983 | 550,253 | 96.7% |
| train | liabilities_to_assets | 326,585 | 550,253 | 59.4% |
| train | operating_margin | 402,308 | 550,253 | 73.1% |
| train | parent_equity_to_assets | 489,593 | 550,253 | 89.0% |
| train | revenue_growth | 472,576 | 550,253 | 85.9% |
| validation | alternative_v3_content_all_available | 104,443 | 157,431 | 66.3% |
| validation | alternative_v3_content_any_available | 156,486 | 157,431 | 99.4% |
| validation | cash_to_assets | 142,348 | 157,431 | 90.4% |
| validation | cfo_to_assets | 148,030 | 157,431 | 94.0% |
| validation | current_liabilities_to_assets | 145,446 | 157,431 | 92.4% |
| validation | current_ratio | 145,446 | 157,431 | 92.4% |
| validation | filing_age_days | 156,486 | 157,431 | 99.4% |
| validation | income_to_assets | 152,170 | 157,431 | 96.7% |
| validation | legacy_v1_content_all_available | 62,886 | 157,431 | 39.9% |
| validation | legacy_v1_content_any_available | 156,486 | 157,431 | 99.4% |
| validation | liabilities_to_assets | 100,030 | 157,431 | 63.5% |
| validation | operating_margin | 117,703 | 157,431 | 74.8% |
| validation | parent_equity_to_assets | 146,092 | 157,431 | 92.8% |
| validation | revenue_growth | 141,378 | 157,431 | 89.8% |

## Scope

The V1 and V3 content sets are inherited from the original financial studies; this audit does not choose a new economic schema. All-available coverage is a diagnostic, not a decision to delete incomplete rows. Any-available coverage helps distinguish missing whole snapshots from missing individual fields. Financial companies are reported separately; the same accounting ratios are not assumed comparable across sectors.

Market-only controls must be rerun on the exact origins used by each future multimodal comparison. The 512-rv20 subset is separately reported for compatibility with the existing foundation-model context requirement. A coverage-selected subset can change the population; retain full-coverage market results as a separate benchmark.

Four-way market/financial/text/all comparisons additionally require a validated filing-document manifest and text extraction. Structured coverage alone does not establish text availability. No predictive benefit or final sample size is established here. 2026 market outcomes remain sealed.

## Next design decision

Among non-financial training-company origins, all seven legacy content fields are available on 30.1% of origins; the seven V3 alternative content fields on 52.2%. These ratios describe availability, not predictive improvement. Dropping every incomplete row would discard a large share of the declared training population. Retain missingness and filing age as explicit controls, with imputation/clipping/scaling learned only on training-role, training-period observations.

Before financial fusion, register the economic schema and verify a stratified set of source values and entity histories. Preserve the original quantities as separately named fields; V3 alternatives are not backfilled values for total liabilities or operating margin. Compare market-only, market+availability/age controls, and market+financial content with the same origins and backbone. Add text to the four-way matched design only after document extraction is validated. The historical evaluation remains development evidence.
