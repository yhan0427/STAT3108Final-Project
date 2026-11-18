# Multimodal data and representation pilot

This stage validates data construction and frozen document representations. It has not established text forecast value, a completed four-way fusion model or a final temporal test.

## Structured financial evidence

491 retained companies produced filing-date snapshots. The fixed original-document subset has 11 reports with supported inline USD facts and 9 without supported inline facts. Independent original-source checks found 81 matching field values and 70 matching snapshot ratios, with no mismatches in the checked cases; seven named fields were unavailable. These are bounded spot checks, not certification of every historical feature or issuer transition.

## Text quality development

The first twenty original filings (oldest/latest from ten fixed sector/hash companies) were used to develop the parser. Nineteen bounded Item 7 sections passed agent-assisted boundary/interior-sample inspection; CMS 2013 was excluded because the selected section incorporated discussion by reference.

The first frozen-parser new-company sample contained 20 documents: 13 accepted boundaries, 2 wrong end boundaries, and 5 abstentions. Singular Disclosure and heading punctuation/prefix formats exposed failures. Those files became development data for the next revision.

The second new-company sample was collected after the revised v6 parser was frozen: 18 accepted boundaries, 0 observed wrong candidate boundaries, and 2 abstentions among 20 documents. This is a small targeted quality sample, not universal accuracy or independent human certification. Full-report substitution is not used.

## Frozen FinBERT pilot

- Model: ProsusAI/finbert, revision `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`; safetensors and tokenizer cached with hashes.
- 19 reviewed documents, 307,144 content tokens across 611 non-overlapping chunks.
- Mean of non-special attended last-hidden-state token vectors, then content-token-weighted document pooling; one 768-vector per document.
- Three sentiment probabilities are stored separately; they are not called embeddings.
- Encoder parameters are frozen in evaluation mode. No PCA, financial model, text model or dimensional reduction was fitted in this representation stage.
- All vector hashes/shapes/finiteness, probability sums and full token coverage checked. Batch/padding comparison maximum absolute vector difference: 1.19e-07.

## Boundaries and next work

FinBERT is a financial sentiment encoder associated with a 2019 paper, not the recent time-series comparator. Its model card describes Reuters TRC2 finance adaptation and Financial PhraseBank fine-tuning; complete document-level overlap/cutoff remains unverified. The current model card states CC BY-NC-SA 4.0; weights and raw reports remain local and are not covered by the project code license.

A separate 100-company text cohort is frozen by existing company role, sector and issuer hash: 80 training-role, 20 held-role issuers, without performance or text-availability selection. Annual reports through 2025 are being collected. Missing reports, foreign forms and extraction failures are retained; no company replacement. Bulk candidates require the declared quality process before fitting.

The full structured-financial comparison uses a separate non-financial cohort with market/availability/financial controls; it does not reuse document embeddings yet. Existing 2024–2025 outcomes are development evidence. The 2026 market holdout remains sealed, and no alpha or semantic predictive gain is claimed.

Sources: https://huggingface.co/ProsusAI/finbert ; https://www.sec.gov/search-filings/edgar-application-programming-interfaces
