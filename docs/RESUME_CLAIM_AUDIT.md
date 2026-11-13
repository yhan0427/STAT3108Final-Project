# Resume claim audit — 2026-11-12

Audit against the supplied resume screenshot and actual experiment artifacts.

| Resume claim | Finding | Defensible current wording |
|---|---|---|
| Developed LSTM and Transformer models | LSTM trained locally; Timestral 5M is a pinned pretrained Transformer evaluated in transfer mode. No local Transformer training/fine-tuning completed. | Trained LSTM models and evaluated a pretrained Timestral Transformer |
| Integrated market, financial features and FinBERT embeddings | Market-only full cohort; financial feature experiments in a separate 30-firm pilot. No FinBERT code/artifacts or text pipeline. | Engineered filing-date-aligned SEC financial features in a separate 30-firm pilot |
| Conducted modality ablations | Structured market/financial/coverage controls exist; no four-way market/financial/text/all matched-sample ablation. | Evaluated financial-feature and missingness controls in pilot experiments |
| HAR-type, Ridge and gradient boosting baselines | Implemented across stages. Full-cohort unified comparison includes persistence, Ridge, LSTM and Timestral; not all pilot baselines were rerun at full scale. | Benchmarked across staged experiments; name the exact full-cohort comparator when describing that result |
| Purged temporal splits | Implemented, including execution-aligned target-end purges. | Purged temporal splits |
| Rolling-origin evaluation | Multiple protocol origins/windows with fixed fitted models per protocol, not daily rolling model refitting. | Multiple chronological evaluation windows |
| Gains vary across years/metrics | Supported in the stated exploratory studies. Full-run LSTM-Ridge intervals cross zero despite lower mean loss. | Assessed period/metric sensitivity and uncertainty; no universal model-superiority claim |
| 1M+ observations from 500 firms | 500 reconstructed reference-date candidates; 491 usable histories, 1,428,780 retained unique company-date price rows. Actual fitting: 386 firms. Common company holdout: 97 firms. | 1.43M daily price observations across 491 stocks from a reconstructed year-end-2025 cohort |
| Integrated full-scale market data with SEC financial data | Not done for 491 firms; SEC financial pilot is a separate 30-firm panel. | Explicitly distinguish full market panel from the financial pilot |
| Publication-date alignment and training-only preprocessing | Filing-date alignment and train-only preprocessing implemented. SEC snapshots use filing dates, not a complete immutable timestamped historical archive. Pretrained overlap remains unverified. | Filing-date alignment and training-only preprocessing to mitigate leakage |

## Current resume wording

**Equity Volatility Forecasting & Portfolio Risk Lab | Python**

- Built a reproducible Python pipeline covering 1.43M daily price observations across 491 stocks from a reconstructed year-end-2025 S&P 500 cohort; separately engineered filing-date-aligned SEC financial features for a 30-firm pilot.
- Trained LSTM models and evaluated a pretrained Timestral Transformer against statistical and machine-learning baselines across staged experiments; used purged temporal splits, 97 held-out companies, QLIKE/RMSE and date-block bootstrap to assess generalization and uncertainty.
- Built a portfolio-risk demo and evaluated 384 strategy records with execution delays, trading costs and validation-fixed exposure controls; found no reliable dynamic-policy improvement over fixed exposure in the exploratory application cases.

The second bullet describes staged experiments, not a claim that every baseline was evaluated on the same 97-firm full run. HAR-type and boosting belong to earlier pilots. The company's 2024–2025 holdout outcomes are now inspected; the final 2026 temporal holdout remains sealed. Full-scale financial/text multimodality is future work.

## Dates and title

Use an ongoing end date (Present) while the project is in progress. Sep 2026 is defensible only if problem formulation, reading or other actual project work began then; implementation/commit artifacts alone do not establish a September start. Current title avoids implying completed FinBERT/text fusion. Do not backdate commits or add planned milestones to completed bullets.

## Evidence

- results/FULL_COHORT_AUDIT_V2.json: actual market sample/observation counts.
- results/FULL_TRAINING_V1.md and AUDIT: local fitting, new-company evaluation and uncertainty.
- config/timestral-v1.json: pinned pretrained Transformer and point construction.
- results/FINANCIAL_V3.md and src/volrisk/fundamentals.py: financial pilot and filing-date alignment.
- config/robustness-v2.json: multiple chronological origins, not daily rolling refits.
- results/FULL_PORTFOLIO_V1.md: 384 application strategy records and interpretation limits.
