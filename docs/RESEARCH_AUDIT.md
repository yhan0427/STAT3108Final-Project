# Independent research and UI review — 2026-11-09

Two independent subagents reviewed UI and research code. Additional artifact-level checks were executed by the primary agent. No models were retrained, and the existing final test was not rerun.

## What has and has not been reproduced

This is a literature-informed independent experiment, not a reproduction of a published result table. Corsi (2009) motivates multi-horizon HAR components, but the implementation uses daily squared-return proxies rather than the original high-frequency realized volatility, and targets forward 20-day daily-return standard deviation. LSTM is an implemented architecture; implementing it is not reproducing an empirical finance paper. FinBERT (Araci 2019) evaluates financial sentiment classification, not this equity-volatility task, and no text branch has been implemented. Gu et al. studies asset pricing; it is motivation for strong ML baselines, not a matching volatility experiment.

An exact replication requires a named target paper/table, the original asset universe, period, data frequency, label, split/refitting schedule, loss and model settings, then an explicit original-versus-reproduced comparison. We have not established that equivalence.

Sources: https://doi.org/10.1093/jjfinec/nbp001 ; https://arxiv.org/abs/1908.10063 ; https://doi.org/10.1093/rfs/hhaa009 .

## Timing and artifact checks

All 28 checks in results/AUDIT.json passed: source-data/weights/prediction hashes, balanced company windows, purged target boundaries, strict prior filing dates, exact reconstruction of the fitted training-only scalers and target normalization. For five pilot firms at two historical cutoffs, removing all later-filed raw facts produces identical historical snapshots. Existing unit tests passed. These checks support the absence of the tested leakage mechanisms; they are not a proof of all possible data issues.

Independent review found no confirmed use of future targets or later-disclosed financial facts in training. Before the audit fix, the original frozen Python source hashes also matched the current model code. A future-evaluation engineering gap was identified: source hashes were recorded but not enforced. The evaluator now rejects modified frozen source code, with a regression test. The historical freeze record remains unchanged; the guard added on 2026-11-09 is a subsequent source revision, not a revision to the old trained results.

The frozen test_evaluated=false field records status at freeze time. Actual completion is represented by TEST_EVALUATION_COMPLETED.json; it must not be interpreted as an unopened test today.

## Remaining limitations

- Convenience sample selected in 2026, including 10 technology companies out of 30; historical survival/selection bias remains. Adding current constituents does not fix historical membership.
- Current SEC Company Facts with filing timestamps is not an immutable historical archive. Verify a sample of original filings, financial definitions, units and dates manually before claiming full point-in-time financial accuracy.
- Mean feature coverage: total liabilities 61.1%, operating margin 65.8%, operating cash flow 80.0%. Missing masks can reveal company identity. Add a missingness/filing-age-only control before attributing gains to numeric fundamentals.
- One chronological split is insufficient evidence for stability across market regimes or unseen companies. Add rolling-origin evaluation and company-held-out tests.
- Three-seed ensembles are reported, but test performance for each seed is not separately documented; no seed-robustness conclusion should be inferred from the ensemble alone.
- The observed 2024–2025 period has been inspected. Further model selection on it is exploratory; renaming runs does not restore an untouched test.

## What the existing results support

Market-only LSTM improves top-five overlap over persistence by 8.71 percentage points. Post-hoc paired date-block bootstrap intervals from existing predictions are [4.69,12.82] pp with 20-day blocks, [5.02,12.66] pp with 60-day blocks.

Adding fundamentals improves observed overlap by 1.99 pp, but the 20-day-block interval is [-0.41,4.32] pp and the 60-day-block interval is [0.08,3.94] pp. This sensitivity does not support an unconditional claim of a stable gain. QLIKE becomes worse by 0.01644, with intervals including zero. Overall evidence for fundamental incremental value is mixed. These are retrospective screening metrics, not demonstrated human-efficiency or investment-return gains.

## Recommended next experiment sequence

1. Complete original-filing spot checks, missingness controls and stronger matched linear/tabular baselines. Register the specification and a genuinely new evaluation design before fitting.
2. Expand toward 100–150 companies with balanced broad sectors, a clear inclusion policy and a held-out company set. Keep financial firms separate until their accounting schema is appropriate. Do not promise that more companies improve predictive accuracy.
3. Run rolling-origin validation across several earlier periods. Preserve the existing v1 run and describe repeated use of its final period as exploratory. A later unseen period can supply a new final evaluation only after the protocol is fixed.
4. Pilot text extraction on at most 20 companies: historical 10-K Risk Factors/MD&A, exact accession/filing availability, section-quality checks, deduplication and immutable text hashes. Compare market, market+financials, market+text and all modalities on identical samples. Record embedding model release/training provenance; a recently pretrained model may encode information from the historical evaluation period. Use an appropriately dated encoder or explicitly disclose retrospective representation contamination.

## UI review

The first design's gradient was too low-contrast; oversized headers, a hexagonal badge, rings and grid decoration overwhelmed the data. Independent revision removes those elements, uses a visibly differentiated navy-to-steel-blue backdrop, frosted panels, a restrained silver action and a compact metric strip. Browser reload confirmed the new design and future-outcome reveal; the console showed no errors. All functional IDs and data bindings were retained.
