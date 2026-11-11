# Research decision log

## 2026-11-11: Application A — Portfolio Risk Lab

The current primary question is whether volatility forecasts improve portfolio risk-target decisions after execution delays and transaction costs. The original forecasting question below remains the record of the earlier stage, not a preregistration of the portfolio application. V1–V3 forecasting results had already been inspected before this pivot.

The portfolio application compares buy-and-hold, fixed 50% equity, historical-risk exposure and forecast-risk exposure. Both risk rules use the same historical correlation estimator. Long-only stock weights are scaled into equity and zero-yield cash, without leverage. Signals observed at close t execute at close t+1; new positions earn returns only after execution. Initial entry and subsequent rebalancing costs are deducted with self-financing accounting. Original forecast labels and delayed trading windows differ; this limitation is explicit.

The exploratory grid is saved in `config/portfolio-v1.json`: two baskets, four overlapping historical protocols, two models and three cost assumptions (48 scenarios, 192 strategy records). It was fixed after an initial application smoke test, not before all outcome inspection. All scenario results are retained. Overlapping periods are not independent replications, and neither higher returns nor lower exposure alone establishes forecasting value. See `docs/PORTFOLIO_PROTOCOL.md` and `results/PORTFOLIO_V1.md`.

A recent-method replication, an audit of any pretrained model's data overlap, and evaluation on new observations remain future work. The portfolio demo and poster are exploratory application artifacts. The earlier equity-volatility poster remains a historical artifact.

## Original forecasting-stage decisions

2026-11-07: Start locally. No Git initialization, commits or remote publishing until Han supplies the repository. Requested commit author: Han; confirm/use the supplied address when committing. Do not backdate commits or manufacture past contributions.

## Question
Do publicly available fundamentals improve 20-trading-day equity volatility forecasts over a market-only model? Does that translate into better top-five risk screening?

## Scope
Daily adjusted-price data, 30 fixed large-cap non-financial companies, 2015–2025 forecast period. Start with a five-company pipeline check; pilot checks use training and validation, not final-test scores. Exclude financial companies because accounting features need different interpretation. This convenience sample is not a historical S&P 500 panel. No claim of population representativeness.

Persistence, HAR-type OLS, histogram gradient boosting, market-only LSTM, LSTM plus fundamentals. Add a boosting-plus-fundamentals baseline for information-set fairness. Three fixed neural-network seeds; report ensemble results and seed validation results. No Transformer or text embeddings in v1.

## Labels and decision time
Forecast after the day's close; returns are adjusted-close log differences. The target is sqrt(252) times the population standard deviation (ddof=0) of returns t+1 through t+20, matching the proposal. Features include the current closing observation. This is a daily-return proxy, not intraday realized variance.

SEC features become usable strictly after the calendar filing date (the next observed trading date), conservatively excluding same-day disclosures. Select only values whose filing date is available at the snapshot; later amendments/restatements cannot flow backward. SEC Company Facts is a present-day collection with filing timestamps, not an immutable historical database; unavailable/deleted facts and issuer-specific taxonomy coverage remain limitations.

## Splits
Train origins 2015–2021; validation origins 2022–2023; test origins 2024–2025. Purge any origin whose target end reaches the next partition. Historical feature windows may use earlier partitions because those values were already observable. Do not discard all pre-boundary feature history. Scalers, imputation and regime threshold fit on training data only. Every model uses identical eligible origins.

## Evaluation
MAE/RMSE in annualized volatility units; QLIKE on annualized variance (ratio - log(ratio) - 1); out-of-sample R² relative to persistence on the same rows. Top-five overlap and cross-sectional Spearman ranking evaluated by date. Alert turnover is a workload diagnostic, not transaction cost. Paired circular moving-block bootstrap resamples whole dates, keeping companies together (20-day blocks; also 60-day sensitivity). Metrics describe retrospective forecasting, not investment returns or proven human efficiency.

Freeze architecture, features and hyperparameters before final test evaluation. Early stopping selects neural epochs using validation only. `evaluate` refuses to overwrite an existing test report. Revisions after viewing test results must be explicitly labelled exploratory, not a fresh untouched test.

## Availability and reproducibility
Public source URLs, retrieval times, file hashes, configuration, package versions and fitted artifacts accompany runs. Raw vendor prices and private configuration are gitignored. Synthetic test fixtures never enter empirical results. Data acquisition failures must remain explicit; never silently substitute invented data or fewer companies.

## 2026-11-11: Execution alignment and TTM method transfer

After Portfolio V1, register `config/aligned-v1.json` before the new run. Fix the tradable marginal target to t+2..t+21, preserve historical splits and purge by the delayed label end. Compare persistence, a per-company LogHAR daily-return proxy, stable float64 sequence Ridge and pinned TTM r2.1 inference. Fit a single multiplicative volatility calibration on validation data for every model, reporting both raw and calibrated versions. The model's pretraining overlap and historical availability are unresolved, so call this retrospective method transfer, not exact replication or a new holdout. Full methods and limitations are in `docs/ALIGNED_METHOD_TRANSFER.md`.

The run completed with 14,430 evaluation firm-days and 192 portfolio strategy records. Aggregate TTM loss and equal-company relative loss lead to different rankings; retain both. Calibration improves some risk-target errors while reducing equity exposure and returns. These findings do not establish alpha or a universal foundation-model advantage.

A separate accounting audit found that turnover used previous-close NAV as its denominator. Correct it to execution-day pre-trade NAV. Preserve the original Portfolio V1 CSV; publish a 192-row turnover correction and verify that every other saved metric is unchanged. This correction does not change holdings or deducted costs.

## 2026-11-11: Reserve 2026 for the final evaluation

Before any 2026 download or scoring, Han raised the risk of exhausting the final holdout. Keep the temporal configuration as a draft and defer access to 2026 outcomes. Finish fixed-exposure controls, cash sensitivity and debugging on already inspected 2024–2025 data first. Freeze the final experiment before opening 2026, with explicit authorization. See `docs/HOLDOUT_POLICY.md`; this local policy does not establish pretrained-data disjointness.

## 2026-11-11: Complete one recent Transformer comparison

Choose Timestral 5M for local resource feasibility before scoring; pin official code and weight revisions in config/timestral-v1.json. Use 512 past rv20 observations and approximate quantile RMS at step 21, with endpoint-constant tails and a declared positivity floor. Fit one calibration scale on validation only. Retain strong calibrated baselines, both baskets and validation-fixed exposure controls. Results remain exploratory and do not justify a universal modern-model advantage. The 2026 holdout was not accessed.

## 2026-11-11: Start and complete first expanded-company training

On Han's instruction to begin training promptly, reconstruct the incomplete dated page using explicit official December amendments rather than silently accepting a flawed list. Preserve source hashes, issuer mapping and every amendment; state that full independent roster verification is incomplete. Freeze 100 candidate issuers before accessing new-company prices, with sector-proportional/hash selection and distinct training versus held-company roles. Retain missing-data failures and short-history exclusions. One failure (WBD) leaves 99 histories, while Q supplies no early training labels. Fit Ridge and three LSTM seeds, and apply pinned Timestral inference, without any 2026 observations. Observe a stronger LSTM result on the new-company sample; retain it without rewriting the earlier thirty-company finding. A matched original-only control is required before attributing gains to expansion.
