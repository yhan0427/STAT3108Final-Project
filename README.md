# Portfolio Risk Lab

**Research question:** Do better volatility forecasts improve portfolio risk targeting after execution delays and transaction costs, beyond simply reducing equity exposure?

Han · STAT UN3108, Applied Deep Learning and AI · Research in progress.

The project connects a public-data pipeline, chronological forecasting experiments and an equity/cash decision simulator. The current study compares sequence Ridge, LSTM and a pinned recent Transformer (Timestral 5M). Financial experiments and the fixed100-issuer annual-text data/quality/FinBERT pipeline are complete. The first matched multimodal Ridge/LSTM experiment and its registered portfolio evaluation are complete; frozen FinBERT embeddings do not establish stable incremental value in this specification.

## Current evidence

| Component | Completed scope |
|---|---|
| Reference cohort | 500 CIK-defined issuers reconstructed for 2025-12-31; 491 retained price histories |
| Market data | 1,428,780 unique company-date observations, ending in 2025 |
| Unified forecasting | 386 fitting companies; common held-company evaluation: 97 firms / 46,657 origins |
| Forecast comparison | Validation-calibrated QLIKE: LSTM 0.3533; Ridge 0.3646 after broad-grid audit (original grid: 0.3731); Timestral 0.4122 |
| Decision evaluation | Two declared baskets; costs, hypothetical cash-yield sensitivity and validation-fixed exposure controls; 384 strategy records |
| Financial/text extension | Structured coverage:491 firms; financial experiments81 held /80 under identity guard;100-issuer text pipeline and matched forecasting complete(19 evaluated firms);1,632 decision outcomes |
| Final temporal holdout | **2026 market outcomes remain sealed** |

The mean LSTM–Ridge QLIKE difference favors LSTM, but paired 20/60-date block intervals both cross zero, including after the [stronger regularization audit](results/RIDGE_REGULARIZATION_AUDIT_V1.md). The portfolio evaluation does not establish a reliable dynamic-policy advantage over validation-fixed equity exposure. These findings do not establish equivalence, universal model superiority or alpha.

This is a reconstructed **fixed reference-date cohort**, not historical S&P 500 membership at every date. Full independent official roster verification remains qualified. Missing histories and current-provider availability can induce selection bias. Company and market observation counts do not imply equally broad financial or text coverage.

## Read the research

1. [Cohort definition](docs/FULL_COHORT.md) and [actual coverage](results/FULL_COHORT_COVERAGE_V2.md).
2. [Full-cohort forecast results](results/FULL_TRAINING_V1.md), with [period diagnostics](results/FULL_TRAINING_PERIODS_V1.csv) and [company comparisons](results/FULL_TRAINING_COMPANY_RATIOS_V1.csv).
3. [Full-cohort portfolio results](results/FULL_PORTFOLIO_V1.md), [tradeoff figure](results/FULL_PORTFOLIO_V1.png) and [complete metric grid](results/FULL_PORTFOLIO_V1.csv).
4. [Decision protocol](docs/PORTFOLIO_PROTOCOL.md), [method-transfer definitions](docs/ALIGNED_METHOD_TRANSFER.md) and [sealed-holdout policy](docs/HOLDOUT_POLICY.md).
5. [Next financial/text stage](docs/MULTIMODAL_NEXT_STAGE.md) and [financial coverage audit](results/FINANCIAL_COVERAGE_V1.md). The structured coverage audit is complete; [joint availability and matching diagnostics](results/FINANCIAL_MATCHING_V1.md) define the next design decision. The [ten-company filing metadata pilot](results/FILING_MANIFEST_PILOT_V1.md) is complete; the [multimodal pipeline pilot](results/MULTIMODAL_PIPELINE_PILOT_V1.md) documents original-report checks, extraction failures and 19 completed frozen FinBERT embeddings. No text predictive improvement is established.

Completed smaller studies remain available through the [experiment index and reproduction guide](docs/REPRODUCIBILITY.md). They preserve earlier findings and specifications; they are not additional independent confirmations of the current study.

The [matched financial experiment](results/FINANCIAL_FULL_V1.md) separately compares market history, disclosure availability/age and financial content on the same non-financial origins. Much of its observed calibrated loss difference is also obtained by metadata controls; incremental financial-content intervals cross zero. These scores use a different population from the 97-firm full-market comparison.

The [documented identity sensitivity](results/FINANCIAL_ENTITY_V1.md) preserves prior results and restricts unverified predecessor histories. Financial-content gains over the simpler filing-age control still have intervals crossing zero. The [100-company text pipeline](results/TEXT_PIPELINE_100_V1.md) reports extraction failures, sampled QA and frozen FinBERT provenance; it does not yet establish text forecast improvement.

The [matched multimodal experiment](results/MULTIMODAL_100_V1.md) compares the four information groups and metadata controls on identical origins. Ridge gains from financial values in this subset; LSTM gains depend on the control. Embedding increments are small/uncertain for Ridge and worsen the financial LSTM. [Company/year diagnostics](results/MULTIMODAL_FAILURES_100_V1.md) retain the failures. The [registered decision grid](results/MULTIMODAL_PORTFOLIO_V1.md) does not establish reliable dynamic superiority over fixed equity; the concentrated technology basket exposes worse LSTM risk tracking.

The [recent Transformer comparison on the same 19 companies](results/MATCHED_TIMESTRAL_V1.md) recalibrates frozen Timestral forecasts on the matched non-held validation set. Its calibrated QLIKE is 0.3199; differences from market-only Ridge/LSTM remain uncertain in both date-block intervals. This is a different-information-set retrospective comparator, not locally trained Transformer fusion. Its unresolved pretraining cutoff also prevents assuming a sealed local2026 sample is unseen by the pretrained model.

## Timing and validation

The current execution-aligned forecast is issued after close t, trades at close t+1, and targets the annualized population standard deviation of daily log returns **t+2 through t+21**. The original forecasting pilot instead used t+1 through t+20; compare results only within their declared protocol.

- Training origins: 2015–2021. Validation origins: 2022–2023. Inspected development evaluation: 2024–2025.
- Labels crossing partition boundaries are purged. Preprocessing is fitted on training observations; calibration and model selection use non-held validation labels.
- Whole-date block bootstrap retains cross-company dependence, with 20/60-session sensitivity. Intervals are conditional on the fitted procedures and observed cohort.
- Ridge and LSTM share 60-session market inputs. Timestral uses 512 trailing-volatility observations and frozen pretrained weights; this is not an architecture-only ablation. TTM, used in an earlier experiment, is a Mixer rather than a Transformer.
- Timestral pretraining overlap remains unresolved. Its late-2026 release prevents claiming that this model could have been traded in 2024–2025. Its results are retrospective method transfer.
- Earlier company holdouts inspected during development are not untouched samples for subsequent design choices. **Do not download or score 2026 prices without the final frozen protocol and explicit authorization.**

## Run the local application

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
python -m volrisk.server --port 8766
```

Open http://127.0.0.1:8766/. Tabs: **Portfolio Lab**, **Research Agent**, **About the Project**.

A fresh checkout requires locally acquired price data and the five V2 prediction files before the application can start; optional matched-study access additionally requires its completed local forecasts, audit and application configuration. Follow the [reproduction guide](docs/REPRODUCIBILITY.md). Data and fitted artifacts are not bundled. With completed local matched artifacts, the replay defaults to the 19-held-company financial/text study. Its model menu includes market, financial, text and metadata controls, with a validation-fixed equity comparator. Earlier thirty-company V2 windows remain separately selectable. The stock pool, presets and models change together. The assistant receives only evidence for the applied selection; future outcomes are absent until explicitly revealed.

The simulator is long-only and unlevered, with drifting holdings, next-close execution and self-financing transaction costs. Defaults: 10% annual risk target, 20-session rebalancing, sixty-session correlation estimates, 10 bps per dollar traded and zero cash return. The [walkthrough](docs/DEMO_WALKTHROUGH.md) explains the controls and outcome reveal.

The Research Agent is optional. Configure a local OpenAI API key in its settings or set `OPENAI_API_KEY` in the server environment; `OPENAI_MODEL` overrides the default `gpt-4.1-mini`. The key is held in process memory and is not written to Git or browser storage. Numeric evidence is server-computed. Live model response quality and user efficiency have not been validated.

## Data and reproducibility

Prices come from [Yahoo Finance via yfinance](https://github.com/ranaroussi/yfinance); structured financials come from [SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces). Local manifests record resolved identities, retrieval metadata, source hashes and exclusions. Raw vendor data, model weights, per-stock predictions and credentials are excluded from Git. Source rights and pretrained-model licenses are separate from this repository's code license.

Financial information becomes usable strictly after its filing calendar date. Later disclosures must not flow backward; stale content and unsupported tags remain missing. Current SEC Company Facts is not a complete immutable point-in-time archive. Entity continuity, accounting definitions and extraction quality need additional review before full multimodal fitting.

There are separate core and foundation-model environments, pinned in `requirements-lock.txt` and `requirements-foundation-lock.txt`. Existing run directories are preserved; substantive revisions require new versioned runs. AI coding assistance is disclosed; the author is responsible for understanding, verifying and explaining the implementation and findings.

```text
config/          Frozen settings, company roles and source definitions
src/volrisk/     Acquisition, features, models, evaluation and application
scripts/         Versioned study runners, audits and report generators
tests/           Timing, leakage, accounting and numerical invariants
docs/            Methods, protocols and reproduction guide
results/         Shareable aggregate evidence and audits
poster/          Poster source; exported PDFs are historical drafts
output/pdf/      Earlier poster exports, not the current final findings
```

Local ignored artifacts live under `data/raw/`, `data/processed/`, `data/models/` and the run-specific result directories. Only load trusted local pickle/joblib files. The demo is a historical research tool; no live feed, trading reliability or investment-performance benefit has been established.
