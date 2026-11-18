# Reproduction guide and experiment index

Updated 2026-11-18. Run commands from the repository root. Completed artifacts and manifests specify the historical code/data contracts; current source plus a fresh vendor download may not reproduce a byte-identical historical snapshot. Inspect each runner's fixed output directory before execution. Do not overwrite existing experiments or use a new run name to disguise repeated inspection of an evaluation set.

## Core environment and acquisition

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
```

Create a gitignored `.local.json` with an identifying SEC contact:

```json
{"sec_user_agent": "Your name academic research your-email@example.org"}
```

Initial thirty-company experiment:

```bash
python -m volrisk.cli download
python -m volrisk.cli build
OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 python -m volrisk.cli train --run main-v1
python -m volrisk.cli evaluate --run main-v1
python -m volrisk.cli report --run main-v1
```

The original CLI defaults to the thirty-company scope in `config/universe.json`; it does not acquire the full reconstructed cohort. A five-company pipeline check uses `download --pilot`, `build --pilot` and `train --pilot --baselines-only --run pilot-baselines-v1`, without final-test scoring.

## Preliminary financial studies and application inputs

```bash
OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 python -m volrisk.robustness
python scripts/summarize_robustness.py
python scripts/check_sec_spotchecks.py
```

V2 produces the prediction files consumed by the current interactive application. See [V2 protocol](EXPERIMENT_V2.md), [results](../results/ROBUSTNESS_V2.md) and [SEC spot checks](../results/SEC_SPOTCHECKS.json). Its original target differs from the later execution-aligned target. Held-company top-five overlap has a different chance baseline from the full thirty-stock list.

The [V3 accounting study](FINANCIAL_SCHEMA_V3.md) has separate build, experiment and summary commands. V3 replaces two economic quantities with alternative indicators; it is not an imputation-only treatment or a clean causal test of coverage.

```bash
python scripts/evaluate_portfolio.py
```

The original application grid is defined in `config/portfolio-v1.json`. See [results](../results/PORTFOLIO_V1.md) and the separately retained [turnover correction](../results/PORTFOLIO_V1_TURNOVER_CORRECTION.csv). These are preliminary thirty-company application artifacts.

## Recent models and company expansion

| Stage | Protocol / implementation | Completed evidence |
|---|---|---|
| Execution-aligned TTM, Ridge, LogHAR | [Method definitions and commands](ALIGNED_METHOD_TRANSFER.md) | [Aligned results](../results/ALIGNED_V1.md) |
| Validation-fixed exposure controls | `scripts/evaluate_exposure_controls.py` | [Control results](../results/EXPOSURE_CONTROLS_V1.md) |
| Timestral 5M transfer | `scripts/evaluate_timestral.py`, `scripts/predict_timestral.py` | [Results and pinned revisions](../results/TIMESTRAL_V1.md) |
| Company/period diagnostics | `scripts/analyze_failures.py` | [Failure analysis](../results/FAILURE_ANALYSIS_V1.md) |
| Reference-list verification | [Universe design](UNIVERSE_EXPANSION.md) | [Rejected candidates and corrections](../results/UNIVERSE_VERIFICATION.md) |
| First expanded cohort | [100-candidate protocol](EXPANSION_100.md) | [Expanded results](../results/EXPANSION_100_V1.md) |
| Matched training-cohort control | `scripts/train_matched_expansion.py` | [Matched results](../results/MATCHED_EXPANSION_V1.md) |
| Full reconstructed cohort | [Full protocol and commands](FULL_COHORT.md) | [Coverage](../results/FULL_COHORT_COVERAGE_V2.md), [forecasting](../results/FULL_TRAINING_V1.md), [portfolio](../results/FULL_PORTFOLIO_V1.md) |

Full-cohort reconstruction depends on pinned source records, explicit December amendments and verified provider aliases. Follow the full protocol in order; `build_full_cohort.py` alone produces the earlier dataset version. Timestral inference uses `.venv-foundation`, its pinned code/weight revisions and the foundation dependency lock. It is frozen inference, not local Transformer training.

## Next financial/text stage

[Stage plan](MULTIMODAL_NEXT_STAGE.md). The financial audit reuses the frozen full-cohort-v2 panel's identity/date/split metadata and checksums; it does not fit models or read forecast outcomes.

```bash
# Feasibility using already cached SEC payloads only.
python scripts/audit_financial_coverage.py --cache-only --limit 5
# Continue the same audit with sequential public SEC downloads.
python scripts/audit_financial_coverage.py
# After all retained companies have been attempted.
python scripts/summarize_financial_coverage.py
```

The local audit directory preserves its plan and per-company checkpoints. Changed code/inputs require a new audit version. Public coverage reports explicitly distinguish partial collection from complete coverage. Download failures remain visible and do not cause company replacement. Current API payloads may contain later disclosures; only facts disclosed through 2025 are passed to feature construction. No 2026 market data is acquired.

A ten-company, sector/hash-selected annual-filing metadata pilot is implemented:

```bash
python scripts/build_filing_manifest.py
python scripts/report_filing_manifest.py
```

It retrieves original annual-filing metadata plus separately labeled amendments and foreign/transition forms, including historical submissions batches. See [pilot results](../results/FILING_MANIFEST_PILOT_V1.md). The subsequent fixed100-issuer text corpus,quality review,frozen FinBERT vectors,matched forecasting and application are complete. See the commands below. Text does not show a stable incremental gain in this specification.

## Completed financial and matched text experiments

Financial inputs and identity sensitivity, after the market panel and coverage audit:

```bash
python scripts/check_original_financials.py
python scripts/build_financial_full.py
python scripts/train_financial_full.py
python scripts/audit_ridge_regularization.py
python scripts/train_financial_metadata_ablation.py
python scripts/build_entity_guarded_financials.py
python scripts/train_financial_full.py --entity-guard
python scripts/report_financial_entity.py
```

These preserve different historical specifications. The earlier full financial runner source is copied into its ignored local run directory. A completed run refuses overwriting. See the manifest hashes and [identity policy](ENTITY_CONTINUITY.md) rather than treating all source versions as interchangeable.

Matched text stage (network downloads are sequential SEC requests; model inference uses the separate foundation environment):

```bash
python scripts/prepare_multimodal_cohort.py
python scripts/collect_multimodal_text.py
python scripts/prepare_text_quality_review.py
# Inspect the selected source sections and record explicit hash-bound decisions.
# finalize_text_quality.py requires that completed LOCAL review ledger.
python scripts/finalize_text_quality.py
.venv-foundation/bin/python scripts/download_finbert.py
.venv-foundation/bin/python scripts/encode_text_corpus.py
.venv-foundation/bin/python scripts/audit_finbert_provenance.py
python scripts/audit_finbert_tokenizer.py
python scripts/build_multimodal_panel.py
python scripts/audit_multimodal_data.py
python scripts/train_multimodal.py
python scripts/report_multimodal.py
python scripts/train_multimodal_age_control.py
python scripts/analyze_multimodal_failures.py
```

Collection requires the previously reviewed frozen parser gates. The encoder also retains the reviewed pilot as a reusable-vector source. A fresh checkout needs those preceding documented pilot stages and source artifacts; this is not a one-command download. Review excerpts and model binaries remain local. The public review JSON records completed decisions and hashes. Reacquisition can change retrieval metadata or vendor bytes: revalidate source/text identity and register a new run rather than weakening a frozen hash gate. The finalizer does not manufacture a fresh review ledger or automatically certify unseen documents.

The portfolio config was registered during model training before its first held scores; preserve it rather than reselecting baskets. The original command was `python scripts/prepare_multimodal_portfolio.py`. Once forecasts finish:

```bash
python scripts/evaluate_multimodal_portfolio.py
python scripts/report_multimodal_portfolio.py
```

Read [text quality/provenance](../results/TEXT_PIPELINE_100_V1.md),[matched results](../results/MULTIMODAL_100_V1.md),[failures](../results/MULTIMODAL_FAILURES_100_V1.md) and [decision results](../results/MULTIMODAL_PORTFOLIO_V1.md). The100-issuer subset is distinct from full-market491 and full-financial415 scopes; only19 held issuers meet its common guarded512-history evaluation. Text absence does not delete companies. No new2026 market prices are needed for any command in this stage.

## Recent model on matched financial/text origins

```bash
python scripts/evaluate_matched_timestral.py
python scripts/report_matched_timestral.py
```

The extension reuses hash-verified raw full-market Timestral forecasts for matching origins,recomputes missing held validation predictions with the pinned foundation worker,and recalibrates only on matched non-held validation companies. It does not reuse the broader population's scale or retrain the Transformer. See [matched results](../results/MATCHED_TIMESTRAL_V1.md). The policy was registered after earlier results were inspected,so it remains exploratory. Timestral's unresolved pretraining cutoff affects eligibility for any claimed fresh2026 confirmation.

## Application and historical presentation materials

```bash
python -m volrisk.server --port 8766
python scripts/update_demo_research.py
```

The generator updates About research summaries. The application independently loads the optional completed matched-study forecasts and audit; without them it retains preliminary V2 access. Its matched preset metrics reconcile exactly with the registered application grid. Optional AI access requires a separately configured API key; mock contract tests do not verify live answer quality.

The [portfolio poster](../output/pdf/portfolio_risk_lab_poster.pdf) and [earlier forecasting poster](../output/pdf/equity_volatility_research_poster.pdf) are historical exploratory drafts. They do not summarize the latest full-cohort findings. UI validation records in `results/COPILOT_VALIDATION.json` and `results/WORKSPACE_TABS_VALIDATION.json` are historical software checks, not current research metrics or live AI evaluation.

## Evaluation boundaries

2024–2025 results are inspected development evidence. Several protocols reuse dates or companies and are not independent replications. Date-block intervals are conditional diagnostics. The [2026 holdout](HOLDOUT_POLICY.md) remains sealed until a final frozen protocol and explicit authorization. Do not run new temporal acquisition or scoring as a reproduction shortcut.
