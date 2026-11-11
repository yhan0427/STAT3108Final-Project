# Final holdout policy

## Decision recorded 2026-11-11

The planned 2026 extension is reserved for final evaluation. No 2026 price acquisition or scoring was performed in this stage. `config/temporal-2026-v1.json` is a draft, not a completed preregistration or authorization to open the holdout.

Use the already inspected 2024–2025 periods for further development, including fixed-exposure controls, cash-return sensitivity, implementation checks and model-design changes. These results are exploratory. Keep the existing 2015–2021 fitting and 2022–2023 validation boundaries for current frozen coefficients, calibration and constant-exposure controls. Do not present 2024–2025 as an untouched final test merely because a new experiment version is created.

Before accessing 2026 outcomes, finish the implementation and freeze the model/checkpoint, parameters, preprocessing, calibration, universe, portfolio rules, primary comparisons, metrics and missing-data policy. Save timestamped source/configuration hashes and the fixed validation exposures. Opening the final evaluation requires explicit authorization after this development stage.

Score the frozen methods once and report all predeclared comparisons, including unfavorable findings. Once inspected, the observations remain the reported test set for that frozen experiment, but cannot supply an untouched test of later changes guided by its results. Any such revision must be labeled exploratory and needs later unseen observations for new confirmation. Do not manufacture a fresh holdout by hiding or deleting previously inspected scores.

Training leakage (future labels/features or test-fitted transformations) and adaptive test-set overfitting (changing choices after seeing test outcomes) are separate risks. The code must guard against the former; this evaluation policy guards against the latter. Neither policy proves that a pretrained model never encountered related financial data. The TTM corpus/cutoff limitations and the survivor-selected universe remain explicit.
