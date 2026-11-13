# Next stage: financial features and FinBERT text representations

Status: planned implementation, not completed resume claims. 2026 remains sealed. Existing full market-only fits, forecasts and company roles are preserved as evidence.

## Incremental-information question

Do filing-available financial values and annual-report language improve execution-aligned twenty-session volatility forecasts beyond the same market history, and do any gains survive coverage controls and portfolio costs? A result of no gain is valid.

## Order and checkpoints

1. Extend SEC entity resolution and structured financial snapshots to the frozen retained market cohort. Preserve issuer identity changes, filing dates, report periods, taxonomy choices, currency/units and source hashes. Produce per-company/per-year/per-feature coverage before selecting a final schema or fitting. Common ratios need financial-sector interpretation; use a declared non-financial primary multimodal subset and a separately documented financial-sector schema/coverage analysis rather than silently treating unavailable current-ratio fields as zeros.
2. Build a dated document manifest for original annual 10-K filings with filing date through 2025 only, including accession, CIK, form, report period, filing/acceptance timestamp, source URL and hash. Retrieve historical submissions files as needed; the recent submissions array alone is insufficient. Later amendments never replace earlier available text retroactively. Foreign forms and missing 10-Ks remain explicit exclusions/secondary scope.
3. Extract Item 7 MD&A first. Validate a small stratified document set manually before bulk processing: headings, table of contents, repeated text, section boundaries, missing sections and abnormal lengths. Preserve extraction-quality and exclusion records. Add Item 1A only as a separately declared extension; do not expand text scope based on forecast outcomes.
4. Pin ProsusAI/finbert weights/tokenizer/code and audit disclosed pretraining data. Freeze the encoder initially. Extract contextual hidden-state representations; three sentiment probabilities are a separate baseline and must not be mislabeled embeddings. Chunk full selected sections, pool masked tokens and aggregate at document level. Record token counts/chunk weights. Do not silently truncate to the first 512 tokens. Any dimension reduction is fitted on training-role, training-period documents only, with duplicate documents counted once.
5. Align each filing strictly after its availability time to trading origins. If only a filing calendar date is reliable, use the next observed trading session conservatively. Period end is not disclosure time. Save filing age, availability and extraction indicators separately from economic/text content. Store embeddings once per document and join by document ID; never replicate 768-vectors physically over every daily row.
6. Freeze matched origins for four groups: market; market+financial; market+text; all three. Keep common companies, dates, targets, splits, seed budgets and hyperparameter selection rules. Use the same backbone and branch sizes, zeroing absent modalities so architecture/capacity changes do not masquerade as information gains. Maintain a market-only full-coverage benchmark to reveal selection effects, plus missingness/filing-age and document-length controls. Compare LSTM fusion and a supervised temporal Transformer fusion only after this data contract passes; retain the recent Timestral transfer benchmark with its distinct information set.
7. Evaluate forecasting, conditional whole-date block uncertainty and failure cases, then reuse fixed portfolio decision rules. These are continued development on inspected historical periods/company cohorts. Opening 2026 requires the later final frozen protocol and explicit authorization.

## FinBERT scope

ProsusAI/finbert is a financial-language/sentiment encoder associated with a 2019 paper. It is not the project's claim to the newest time-series architecture. The contemporary time-series model comparison is separate. The model card describes BERT-base hidden size 768 and three sentiment labels; use the backbone hidden states for embedding experiments and inspect the license/pretraining restrictions before distribution. No commercial deployment or completed FinBERT training is claimed.

## First deliverable

A coverage/identity audit and document manifest, followed by a small verified embedding pipeline. Do not begin the large multimodal training grid until financial coverage and text extraction quality are measured. Actual usable cohort size may be below 491 and must be reported for both full and matched samples. Resume bullets remain future claims until their corresponding artifacts and results exist.

Sources: https://www.sec.gov/search-filings/edgar-application-programming-interfaces ; https://huggingface.co/ProsusAI/finbert .
