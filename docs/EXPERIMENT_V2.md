# V2: information controls and generalization

This exploratory extension was specified on 2026-11-09 after v1 test results were inspected. It is neither a paper replication nor an untouched confirmatory holdout. V1 artifacts remain intact. The exact machine-readable plan is config/robustness-v2.json; each execution saves that plan and source/data hashes before fitting.

## Models

Thirteen specifications: persistence; Ridge on current market/HAR summary features and Ridge on the complete flattened 60-day market history, each with market-only, missingness/filing-age controls, or full fundamentals; boosting with those three information sets; and matched-architecture LSTM with those three information sets. All LSTM modes use the same three prespecified seeds and validation-only early stopping.

Sequence Ridge and LSTM receive exactly the same preprocessed 60 x 10 historical market values. Flattening preserves order. Snapshot Ridge and boosting receive current market/HAR summaries instead; the latter are deterministic market-history functions. Do not attribute differences between summary and sequence specifications solely to nonlinear architecture.

Missingness controls provide the binary financial missing flags and elapsed filing age, while zeroing the seven financial-ratio values. They test whether gains are attributable to reporting coverage/freshness instead of the measured accounting values. This is not a full causal test: a missing pattern or financial ratio may identify a company.

Ridge alpha is chosen from 1, 100, 10000 by validation log-target MSE. This is the only new hyperparameter selection grid. Boosting and LSTM settings are unchanged from v1. Preprocessing and log-target scaling are refitted only on each protocol's training companies and dates. Windows cannot cross companies. Boundary-crossing target labels are purged.

## Protocols

- Static matched comparison: same dates and 30 companies as v1.
- Selected expanding-window origins: evaluate 2020, 2022, 2024; validate on the immediately previous year; train on earlier observations beginning 2015. Models remain fixed within each evaluation year. These are three selected origins, not a full annually refitted backtest.
- Company holdout: train/validate on 22 companies; evaluate 2024–2025 on AAPL, IBM, HD, PG, JNJ, CAT, XOM, NEE. Held-out company rows cannot influence preprocessing, alpha selection or early stopping. This is a simulated company holdout: those companies were examined in v1.

All eligible prediction rows are identical across models within each protocol. Metrics include annualized-volatility MAE/RMSE, variance QLIKE, persistence-relative R² and top-five overlap. Paired QLIKE uncertainty resamples whole dates with 20/60-day blocks. Seed-specific evaluation metrics and chosen Ridge alphas are recorded. Do not select the best seed using evaluation scores.

## Comparing screening results

The company-held-out pool contains eight stocks, whereas other pools contain thirty. Random top-five expected overlap is 5/8=62.5% versus 5/30=16.7%. Raw overlap cannot be compared across these pools as evidence of better forecasting. Compare against persistence and other models within the same pool. Do not aggregate the static, overlapping 2024 origin and holdout evaluations into an independent sample.

## Outputs

results/robustness/v2/metrics.csv, protocol-specific predictions.csv, fitted preprocessing/tabular models, LSTM weights and diagnostics with histories, seed scores, alpha selection and paired intervals. The folder is gitignored because it contains large fitted artifacts. A compact shareable narrative and figures are generated separately by scripts/summarize_robustness.py.
