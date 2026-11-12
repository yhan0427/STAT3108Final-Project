# Portfolio Risk Lab

**Research question:** Can volatility forecasts improve portfolio risk targeting after execution delays and trading costs, relative to a historical-volatility rule?

Author: Han · STAT UN3108, Applied Deep Learning and AI.

A portfolio decision laboratory: define a stock basket, inspect correlation-aware risk, compare equity/cash allocation rules, and reveal historical outcomes. Forecasting experiments V1–V3 supply preliminary evidence; portfolio decision value is the new primary application.

## Start the application

```bash
.venv/bin/python -m volrisk.server --port 8766
```

Open http://127.0.0.1:8766/. The three tabs are **Portfolio Lab**, **Research Agent**, and **About the Project**. The homepage compares buy-and-hold, fixed 50% equity, historical-risk and forecast-risk strategies. Future results appear only when revealed. Export and all portfolio calculations work without an AI key.

The first version is a historical, long-only, unlevered simulation. Defaults: 10% annualized risk target, 20-session rebalancing, sixty-session correlation estimates, next-close execution, 10 bps cost per dollar traded and zero cash return. Weights drift between trades. Lower risk may come from lower exposure; no alpha or model superiority is assumed.

Fresh checkouts need local price snapshots and all five V2 prediction files before starting the server. Follow the acquisition/build and V2 commands below. Raw data, per-stock predictions, weights and credentials are not distributed with the repository. Aggregate results and source are shareable; exact vendor history may change on download.

## Research artifacts

- [Decision protocol](docs/PORTFOLIO_PROTOCOL.md): risk formulas, execution, self-financing costs and evaluation boundaries.
- [Application design](docs/PRACTICAL_APPLICATION.md) and [demo walkthrough](docs/DEMO_WALKTHROUGH.md).
- [Current A-theme poster](output/pdf/portfolio_risk_lab_poster.pdf): an exploratory application draft, not the final confirmed study.
- [Portfolio experiment results](results/PORTFOLIO_V1.md): complete declared scenario grid, cost sensitivity and interpretation limits.
- [Preliminary V2 forecasting study](results/ROBUSTNESS_V2.md) and [V3 accounting study](results/FINANCIAL_V3.md). These are not portfolio performance results.

```bash
python scripts/evaluate_portfolio.py
```

The scenario runner refuses to overwrite `results/portfolio/v1`. Its plan is in `config/portfolio-v1.json`. UI exploration does not create new confirmatory evidence. A recent-method comparator and a genuinely new evaluation period/company sample remain future research; existing baselines have not been relabelled as new methods.

## Forecasting foundation and reproducibility

## Quick start

Python 3.11+ is required. On this computer the environment is already installed as `.venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
```

For a fresh checkout, create a local `.local.json` (gitignored) containing your real SEC contact identifier:

```json
{"sec_user_agent": "Your name academic research your-email@example.org"}
```

Then run commands from the project root:

```bash
# Pipeline check only; no pilot test-set evaluation.
python -m volrisk.cli download --pilot
python -m volrisk.cli build --pilot
python -m volrisk.cli train --pilot --baselines-only --run pilot-baselines-v1

# Main study; cached pilot files are reused.
python -m volrisk.cli download
python -m volrisk.cli build
OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 python -m volrisk.cli train --run main-v1
python -m volrisk.cli evaluate --run main-v1
python -m volrisk.cli report --run main-v1
```

The earlier individual-stock demo can be regenerated at `results/demo/main-v1/index.html` with the report command above; its obsolete generated output was removed during workspace cleanup. Use the server above for the current Portfolio Risk Lab. The interface uses a navy/blue gradient, frosted-glass panels, metallic highlights and responsive layouts. The generated individual-stock demo requires no API keys, web service or internet connection. `results/RESULTS.md` contains the experiment summary. Full fitted objects and predictions live in `results/runs/main-v1/`.

Existing run names are never overwritten. Final-test evaluation is guarded against repetition; if you change the design after reading test results, record the change and treat the next experiment as exploratory. Do not rename a run to pretend that the same test period is untouched. A failed evaluation leaves an explicit start marker for audit rather than silently restarting.

## Reproducibility and data contract

| Item | Specification |
|---|---|
| Forecast time | After the observation day's close |
| Horizon | Next 20 observed trading days |
| Target | `sqrt(252) * std(r[t+1:t+21], ddof=0)` |
| Sequence | Most recent 60 trading days including observation date |
| Training origins | 2015–2021, with boundary-crossing targets removed |
| Validation origins | 2022–2023, with boundary-crossing targets removed |
| Test origins | 2024–2025, with incomplete/end-crossing targets removed |
| Preprocessing | Training-only median imputation, 1/99% clipping and standardization; financial missing flags |
| Early stopping | Validation standardized log-target MSE, fixed max epochs/patience |
| Regimes | Observed SPY trailing volatility, cutoff is training median |
| Uncertainty | Paired circular date-block bootstrap, 20 and 60 trading-day blocks |

All models use identical eligible observations, including rows with missing accounting features. Financial coverage is exported per company and field; missing values are not presented as real zero financial values. `filing_age_days` is elapsed calendar days since the available snapshot. Financial values older than 180 days are marked missing. Price gaps cause failure rather than silent forward filling or stock removal. The source manifest records timestamps, resolved entities, hashes and the exact study configuration.

Inputs are read only from local trusted artifacts after acquisition. Python pickle/joblib checkpoints must never be loaded from untrusted repositories.

## Sources and important limitations

- Prices: [Yahoo Finance through yfinance](https://github.com/ranaroussi/yfinance). Unofficial client; raw vendor data is excluded from Git. Respect source terms and verify redistribution rights separately.
- Financials: [SEC EDGAR Company Facts](https://www.sec.gov/search-filings/edgar-application-programming-interfaces). Standard USD US-GAAP tags only. Missing custom tags remain missing; we do not invent issuer-specific mappings or conflate differently defined accounting values merely to improve coverage.
- XOM historical entity correction: the current ticker map points to a new holding company. The 2015–2025 issuer is [CIK 34088, verified SEC filing](https://www.sec.gov/Archives/edgar/data/34088/000003408825000010/0000034088-25-000010-index.htm). The mapping override and evidence are recorded in `config/universe.json`.
- [HAR model reference: Corsi (2009)](https://doi.org/10.1093/jjfinec/nbp001). Our daily-return proxy baseline is explicitly labelled HAR-type.

This is a fixed convenience sample, not reconstructed historical S&P 500 membership; survivorship bias and technology overrepresentation apply. Broad sectors are manually assigned research groups, not historical licensed GICS classifications. SEC Company Facts is today's database with filing timestamps, not a complete immutable point-in-time archive. Original filing dates prevent known later facts from flowing backward, but lost historical tags, source corrections and incomplete coverage remain limitations. Raw volume changes can reflect corporate actions. Daily volatility targets overlap and are not independent firm-day observations.

A lower forecast loss need not imply a better top-five list. Higher volatility does not mean negative returns, poor corporate health or default risk. The demo uses held-out historical observations, not a live production data feed. User-selected replay dates are illustrations, not evidence in isolation.

## Layout

```text
config/              Frozen study settings and sample definition
src/volrisk/         Acquisition, point-in-time features, models, metrics, demo
tests/               Target, timing, leakage and evaluation invariants
docs/                Research decisions and data documentation
data/raw/            Local cached provider data and provenance (ignored)
data/processed/      Local feature panel, snapshots and coverage (ignored)
results/runs/        Model weights, validation/test predictions and manifests (ignored)
results/demo/        Local portable demo and research figures (ignored)
results/RESULTS.md   Shareable summary, generated from actual results
```

## Next research steps

Prioritize the decision-level portfolio evaluation, a reviewed recent-method comparator and a genuinely new holdout. Financial-feature work remains preliminary; text and stock-return prediction are deferred.

## Exploratory robustness extension (v2)

The v2 protocol adds matched-information Ridge baselines (including the same full 60-day sequence as LSTM), financial-missingness/filing-age controls, three selected expanding-window evaluation years, and eight companies excluded from both fitting and validation. These experiments were designed after reviewing v1; they must not be presented as an untouched confirmatory test.

```bash
OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 python -m volrisk.robustness
python scripts/summarize_robustness.py
python scripts/check_sec_spotchecks.py
```

The experiment refuses to overwrite `results/robustness/v2`. Read `docs/EXPERIMENT_V2.md` for the exact scope and `results/ROBUSTNESS_V2.md` for completed comparisons. Detailed fitted artifacts remain local and ignored. Compare top-five scores within a protocol: the held-out-company cohort has eight stocks, not thirty, so its random top-five overlap is much higher.

Original-statement spot checks currently cover ten ratios across two filings (Apple 2021 10-K and Caterpillar 2021 Q3 10-Q). This does not certify the full historical accounting panel. Check values and official source links are in `config/sec-spotchecks.json`, with computed matches in `results/SEC_SPOTCHECKS.json`.

## Optional Research Agent

Open AI settings in the Research Agent tab to configure a local OpenAI API key, or set `OPENAI_API_KEY` in the server environment. Default model: `gpt-4.1-mini`; `OPENAI_MODEL` overrides it. The key stays in process memory, is cleared on restart and never enters Git, browser storage or exports. A ChatGPT login does not supply this API key.

The server recomputes portfolio evidence; the model explains it. Hindsight outcomes enter the context only after reveal. An absent key produces an explicit unavailable message, not a simulated answer. Real model access and semantic answer quality require separate verification.

## Course deliverables and honest history

The final presentation is on December 10, 2026; the course report is due December 13 and repository December 15. The application pivot follows actual preliminary work, not a retrospective rewrite of the experiments. Do not fabricate dates or individual contributions.

AI coding assistance is disclosed. Han should review the implementation and be able to explain target construction, chronology, correlation, costs, and failure cases in the final Q&A.

## Execution-aligned foundation-model experiment

Current implementation ledger: [research status](docs/RESEARCH_STATUS.md). Expansion is a staged design, not an achieved sample size: [dated universe specification](docs/UNIVERSE_EXPANSION.md). The [2026 holdout remains sealed](docs/HOLDOUT_POLICY.md).

A separate exploratory extension now compares pinned IBM TinyTimeMixer r2.1, a LogHAR daily-return proxy, sequence Ridge and persistence on the next-close tradable horizon. It includes validation-only calibration, equal-company comparisons and cost-aware portfolio evaluation. The checkpoint's pretraining overlap and historical availability remain unresolved; this is method transfer, not exact paper replication or an untouched test. The demo still uses its labeled V2/V1 inputs.

- [Methods and reproduction](docs/ALIGNED_METHOD_TRANSFER.md)
- [Aligned forecast and portfolio results](results/ALIGNED_V1.md)
- [Run audit](results/ALIGNED_V1_AUDIT.json)
- [Portfolio V1 turnover correction](results/PORTFOLIO_V1_TURNOVER_CORRECTION.csv): all other metrics verified unchanged.
- [Fixed validation-exposure controls](results/EXPOSURE_CONTROLS_V1.md): development-only comparisons on 2024–2025, with transaction costs and hypothetical cash-yield sensitivity.

Recent Transformer development comparison: [Timestral 5M results](results/TIMESTRAL_V1.md), with pinned code/weights, common aligned origins and validation-only calibration. The 2026 holdout remains sealed.

[Company and period failure analysis](results/FAILURE_ANALYSIS_V1.md) reports all firms, sectors, quarters, validation-defined historical market-risk groups and post-hoc influence checks. These are descriptive development diagnostics, not a new holdout.

[Universe verification](results/UNIVERSE_VERIFICATION.md): public candidate lists have unresolved reference-date conflicts; 100/500-company expansion is not yet claimed.

First expanded training completed: [cohort construction and limitations](docs/EXPANSION_100.md), [training and held-company results](results/EXPANSION_100_V1.md). Frozen candidate cohort: 100 issuers; retained histories: 99; raw company-date observations: 291,416. Actual fitting/validation-label companies: 64; held-company evaluation: 34. These are reconstructed-reference-cohort results; the 2026 temporal holdout remains sealed.

[Matched pilot training-cohort control](results/MATCHED_EXPANSION_V1.md) holds validation/evaluation firms fixed. The small calibrated LSTM improvement has intervals crossing zero; no reliable benefit from expansion is yet established. Next work prioritizes full-cohort data verification before another comprehensive training grid.

Full-cohort stage completed: [coverage audit](results/FULL_COHORT_COVERAGE_V2.md) and [unified model results](results/FULL_TRAINING_V1.md). Actual collection: 500 reconstructed reference-date candidates, 491 retained histories, 1,428,780 company-date rows. Unified fitting uses 386 firms; common held-company evaluation uses 97 previously unscored firms / 46,657 origins. LSTM has lower calibrated QLIKE than Ridge on average, but both paired 20/60-date intervals cross zero. The existing demo remains on its clearly labeled preliminary evidence; the 2026 final holdout remains sealed.
