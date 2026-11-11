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
