# Research status and next decision

Updated 2026-11-11. This ledger separates completed evidence from resume ambitions.

## Question

Does better marginal volatility forecasting improve portfolio risk decisions beyond simply reducing stock exposure? Generalization across periods and companies, uncertainty and reproducibility are the research priorities. Portfolio Risk Lab supplies the decision application. We do not require a positive model effect.

## Implemented versus proposed

| Component | Actual status |
|---|---|
| Fixed 30-stock market panel and SEC filing-date features | Implemented; convenience survivor sample, not historical S&P 500 membership |
| Persistence, HAR-type proxies, Ridge, boosting, LSTM | Implemented; V1–V3 reports preserve original specifications |
| Same 60-day market history LSTM versus Ridge | Evaluated across protocols; advantage changes with period/metric, limited to these experiments |
| Financial feature coverage and missingness controls | Implemented; coverage/schema effects remain partly confounded |
| Execution-aligned Ridge and LogHAR proxy | Implemented, with training/validation boundary purges |
| TTM r2.1 | Actually evaluated; fixed checkpoint, raw and validation-calibrated forecasts; Mixer architecture, NOT Transformer |
| Fixed validation-exposure and cash-yield controls | Development runner implemented; results are in EXPOSURE_CONTROLS_V1 after completion |
| Recent Transformer | Timestral 5M actually evaluated on common aligned 2024–2025 origins; see TIMESTRAL_V1.md |
| FinBERT financial-text representations | Not implemented |
| Market / financial / text / combined matched-sample ablation | Not implemented |
| 500 firms / 1M+ observations | Aspirational, not achieved or audited |
| 2026 temporal holdout | SEALED; no acquisition/scoring in this stage; user reaffirmed this choice |

The portfolio demo still displays its labeled V2/V1 evidence. Separate new research reports must not be represented as already integrated into the application.

## One modern-model candidate, not a model sweep

Review completed: Mistral's [Timestral announcement](https://mistral.ai/news/timestral/) is dated 2026-10-15. Its [5M model card](https://huggingface.co/mistralai/Timestral-5M) describes a patch Transformer with time/variate attention and quantile output; [official inference code](https://github.com/mistralai/timestral) supports CPU execution. Select the 5M version for the next feasibility stage because it fits local resources, not because of a measured advantage on our stocks. It has now been installed and evaluated in a versioned development experiment; no outcome-based change of model size was made.

Before scoring, pin weights and code, inspect training-data disclosure, define how quantile predictions map to the variance loss, and freeze preprocessing/context. Start with the same past-only rv20 context and t+21 output used for TTM. Report that this isolates an application comparison, not a pure equal-capacity architecture ablation. Add a calibrated variant fitted on validation only. Keep strong calibrated simple baselines.

A release in late 2026 cannot be described as a model that could have been traded in 2024. An unseen local 2026 set does not establish absence of pretraining overlap. If cutoff/disjointness cannot be established, its result must remain a retrospective-transfer diagnostic; do not conflate it with a historically deployable strategy or the cleanly fitted local models.

## Fixed-exposure reasoning, in plain language

`freeze_validation_exposure` computes the average planned stock fraction on validation rebalance dates. `evaluate_policies` compares that constant fraction with dynamic forecast exposure on later development dates. Both restore the same within-stock weights, charge costs and earn returns only after execution. `simulate` accrues the hypothetical cash yield between closes using calendar days. A constant chosen on validation avoids using development outcomes to engineer a flattering comparison. It need not match realized exposure in the later period, so that remaining difference is reported.

## Efficient next steps

1. Finish and inspect the existing fixed-exposure controls, including failure cases and both date-block interval lengths.
2. Audit and pin one Timestral feasibility configuration. Run on development data only; don't install a collection of foundation models.
3. Verify the dated company/security registry in UNIVERSE_EXPANSION.md, then freeze phase-one company selection before querying outcomes.
4. Complete the modern-model comparison and selection design before requesting final 2026 access. Preserve the sealed holdout meanwhile.
5. Consider text only after a specific incremental-information hypothesis and consistent four-way samples are feasible.

Each update should state the question, completed artifacts, observed results/limits, and what decision follows. No claimed skill or dataset scale enters the resume until its corresponding artifact and count exist.

## Completed Timestral stage

Pinned code/weights and the quantile-to-point rule were recorded before scoring. The worker received 28,830 validation/development contexts and no labels; final development metrics use 14,430 observations. Calibration used validation only. Timestral-calibrated pooled QLIKE is 0.3228 versus calibrated Ridge 0.2911. Its paired loss difference is +0.0316: the 20-date interval crosses zero and the 60-date interval does not. These conditional diagnostics support no blanket superiority claim. In the diversified basket its risk-target error is 2.674 pp versus 3.485 pp for its validation-fixed control; in technology it is 3.120 versus 2.917 pp. All six portfolio metrics are retained.

The 5M choice was resource-motivated. The quantile-RMS approximation differs from TTM's point summary, and pretraining overlap remains unresolved. Context files were removed after success. Next priority: company-level failure analysis and verification of the dated expansion registry, before requesting final holdout access.

## Company/period diagnostics completed

`results/FAILURE_ANALYSIS_V1.md` reports every firm and sector, year/quarter, and SPY-risk groups with a threshold fitted on validation dates. Calibrated Timestral has lower QLIKE than calibrated Ridge in 10/30 companies. Removing the five worst relative companies leaves a positive average loss difference (+0.0155); this post-hoc check is descriptive and does not authorize removing those firms. In 2024, Timestral improves MAE but worsens QLIKE; in 2025 both comparisons favor Ridge. Company totals reconcile with the previously reported pooled scores. Next: verify the dated issuer/security registry before freezing the first expanded company roster. The 2026 holdout remains sealed.

## First expanded training completed

100 candidate issuers were frozen before price acquisition. The fixed Wikipedia historical page was explicitly corrected using the official December announcement, yielding a reconstructed 503-security/500-CIK registry; complete independent index verification remains a limitation. New companies were selected by sector and fixed issuer hash, never by model outcomes. Of 100 candidates, 99 histories were acquired (WBD failed, no replacement), totaling 291,416 raw company-date rows. Q has too little early history; 64 firms actually provide train/validation labels, and 34 retained held-company firms provide neither.

Ridge and three-seed LSTM were actually trained on the same sixty-day market input, using pre-2022 labels, training-only preprocessing and non-held 2022–2023 validation. The pinned Timestral procedure was evaluated on common origins. In the 34-company holdout, calibrated LSTM QLIKE is 0.2969 versus calibrated Ridge 0.3318, with paired 20/60-date intervals for the difference both below zero. Both yearly MAE and QLIKE favor LSTM in this run. These are exploratory new-company results on previously inspected calendar periods, not a new temporal holdout or proof that expanding training caused the improvement.

Next: run a matched original-30-only training control using the same aligned target, input preparation, common held-company evaluation origins and neural seeds. Separate training-data expansion from changes in preprocessing/targets/validation selection. Follow with portfolio translation before opening the still-sealed 2026 test. Full results: EXPANSION_100_V1.md and year/company/audit CSV/JSON outputs.

## Pilot control complete; prioritize data scale

The matched original30/expanded64 control is complete. Calibrated LSTM changes QLIKE 0.3023 to 0.2985 on identical held-company origins, with both date-block intervals crossing zero. No reliable expansion benefit is established. Following Han’s 2026-11-12 question, prioritize the full dated universe and coverage ledger rather than retraining every intermediate batch. Run the next comprehensive comparison after final cohort/roles and protocol freeze. Any company already scored during development is not an untouched company holdout for later design choices. The 2026 temporal holdout remains sealed.

## Full-cohort unified run completed

Full reconstructed cohort: 500 frozen issuers, 491 retained raw histories, 1,428,780 company-date observations. Nine unavailable histories remain excluded with reasons, without candidate replacement; BK/MMC were recovered via verified same-security ticker changes. Actual fitting uses 386 firms and 650,683 origins; non-held validation uses 387 firms / 183,780 common-context origins. New-company evaluation uses 97 previously unscored firms / 46,657 origins. Eligible-company counts differ by history requirements and are explicitly audited.

Ridge, three-seed LSTM and pinned Timestral have completed the unified comparison. Calibrated QLIKE: LSTM 0.3533, Ridge 0.3731, Timestral 0.4122. The LSTM-Ridge paired loss difference is -0.01974, with both 20/60-date intervals crossing zero. Retain this uncertainty and every company/year outcome. No universal neural-model superiority, reliable training-expansion benefit or alpha claim is justified. The 2026 temporal holdout is still sealed.

Next: translate the frozen full-run forecasts into portfolio decisions and evaluate against historical-risk and validation-fixed exposure controls, including costs and cash sensitivity. Keep company identity/selection limitations explicit; do not tune the forecast models to these newly seen company outcomes and call them untouched again. See docs/FULL_COHORT.md and results/FULL_TRAINING_V1.md.

## Full-run portfolio application completed

Frozen full-run parameters and non-held validation calibration were reused for two held-company baskets: one issuer per sector (11 stocks), and five technology issuers. Basket selection used fixed hashes and >=400 eligible validation/evaluation origins, not forecast scores. This application follows already inspected full forecast results and remains exploratory. Validation prices/predictions set model-specific constant-exposure controls without fitting held-company labels. Reissued evaluation predictions match the saved full-run outputs within numerical tolerance.

96 scenarios / 384 strategy records retain all raw/calibrated variants, 0/10/25 bps costs and 0%/hypothetical 4% cash yields. At the primary 10 bps/zero cash setting, calibrated LSTM improves risk-target MAE by 0.472 pp and 0.762 pp versus the historical-risk rule, with lower returns and lower average equity. Against validation-fixed equity, differences are +0.026 pp and -0.008 pp with both 20/60-date intervals crossing zero. No reliable dynamic-policy benefit over these constant-exposure controls is established, nor is equivalence proven.

Report: results/FULL_PORTFOLIO_V1.md; chart and all metric/audit files accompany it. Model-independent controls reconcile across model cases, all 384 rows are finite, and temporary inference contexts were removed. The 2026 holdout remains sealed; the demo still uses preliminary evidence until the latest research is explicitly integrated. Next: incorporate the decision tradeoffs and limitations into the demo/research presentation, then finalize the evaluation protocol before requesting 2026 access.
