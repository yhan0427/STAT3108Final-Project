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
