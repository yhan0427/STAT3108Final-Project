"""Reconcile and explain the recent Transformer comparison on matched origins."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now
from volrisk.metrics import forecast_metrics
from volrisk.aligned import variance_calibration


def main():
    audit = read_json(ROOT / 'results/MATCHED_TIMESTRAL_AUDIT_V1.json')
    path = ROOT / 'results/financial/matched-timestral-v1/predictions.csv'
    if sha256(path) != audit['predictions_sha256']:
        raise ValueError('Changed matched modern-model forecasts')
    frame = pd.read_csv(path)
    held = frame.loc[frame.role.eq('held_company') & frame.split.eq('test')]
    validation = frame.loc[frame.role.eq('training_company') & frame.split.eq('validation')]
    scale = variance_calibration(validation.target, validation.Timestral5M)
    if abs(scale - audit['nonheld_validation_calibration']) > 1e-12:
        raise ValueError('Calibration-source reconciliation failed')
    scores = pd.read_csv(ROOT / 'results/MATCHED_TIMESTRAL_V1.csv').set_index('model')
    error = 0.
    for name, row in scores.iterrows():
        metrics = forecast_metrics(held.target, held[name], held.Persistence)
        error = max(error, *(abs(row[k] - metrics[k]) for k in ['mae', 'rmse', 'qlike']))
    if error > 1e-10:
        raise ValueError('Metric reconciliation failed')
    parent = pd.read_csv(ROOT / 'results/MULTIMODAL_100_V1.csv').set_index('model')
    lines = ['# Recent Transformer on the matched information-study cohort', '',
        'Question: does the previously pinned Timestral5M transfer comparator change the interpretation when evaluated on exactly the same19 held companies and9,139 origins as the financial/text experiment?', '',
        'This extension was registered after the matched Ridge/LSTM results were inspected. It is continued retrospective development,not an independent model-selection exercise or fresh confirmation. The checkpoint,code and point summary are unchanged. The runner reused46,869 raw full-market validation/evaluation forecasts after exact company-date,target andrv20 checks,and computed8,856 previously absent held-company validation forecasts. The broader-cohort calibration was discarded; calibration uses only the matched80 non-held training-company validation set.', '',
        '| Procedure | Raw QLIKE | Validation-calibrated QLIKE |Calibrated MAE(volatility pp) |', '|---|---:|---:|---:|']
    for name in ['Ridge_market', 'LSTM_market', 'Ridge_all', 'LSTM_all']:
        lines.append(f'| {name} | {parent.loc[name,"qlike"]:.6f} | {parent.loc[name+"_calibrated","qlike"]:.6f} | {100*parent.loc[name+"_calibrated","mae"]:.3f} |')
    lines.append(f'| Timestral5M | {scores.loc["Timestral5M","qlike"]:.6f} | {scores.loc["Timestral5M_calibrated","qlike"]:.6f} | {100*scores.loc["Timestral5M_calibrated","mae"]:.3f} |')
    lines += ['', 'The modern transfer model has higher mean calibrated QLIKE here than the market-only Ridge/LSTM,but both corresponding date-block interval lengths cross zero. Its difference from all-information Ridge is positive in both intervals. Different inputs,training procedures and point functionals prevent treating this as a pure architecture ablation or a claim that newer Transformers generally underperform.', '',
        '| Calibrated Timestral minus comparator |Mean difference |20-date95% interval |60-date95% interval |', '|---|---:|---|---|']
    for other in ['Ridge_market', 'LSTM_market', 'Ridge_all', 'LSTM_all']:
        values = audit['paired_intervals'][f'Timestral5M_calibrated-minus-{other}_calibrated']
        a, b = values['20'], values['60']
        lines.append(f'| {other} | {a["mean_difference"]:+.6f} | [{a["ci95_low"]:+.6f},{a["ci95_high"]:+.6f}] | [{b["ci95_low"]:+.6f},{b["ci95_high"]:+.6f}] |')
    lines += ['', '## Decision application', '',
        'The same registered baskets,cost/cash grid and execution rules yield96 additional strategy records. Raw and calibrated models have their own fixed equity fractions determined from earlier validation forecasts/prices,never held labels. Calibrated Timestral has lower mean risk-target error than its fixed control in the sector basket,and higher error in the technology trio;both20/60-date intervals cross zero in both baskets. Returns,drawdown,equity,turnover and costs remain in MATCHED_TIMESTRAL_PORTFOLIO_V1.csv. No reliable dynamic-policy advantage is established.', '',
        '## Interpretation and temporal limit', '',
        'Timestral receives512 trailingrv20 values and emits a quantile-derived RMS summary at the twenty-first future point. Ridge/LSTM receive60 sessions of ten market features,with optional financial/text inputs. Same origins and targets make the outcome comparison relevant; they do not equalize information sets. FinBERT is a frozen language encoder,not the modern temporal model. No supervised Transformer fusion was locally trained.', '',
        'The2026 model release and unverified pretraining cutoff prevent a historical2024–2025 trading claim. Keeping local2026 prices sealed does not itself make2026 an unseen sample for a pretrained2026 model. Unless an independent cutoff excludes these outcomes,the modern model must remain outside a claimed fresh2026 primary validation(or be evaluated after its documented cutoff). The local2026 holdout remains unopened.', '',
        'Sources: [pinned Timestral model](https://huggingface.co/mistralai/Timestral-5M/tree/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f),config/timestral-v1.json,docs/ALIGNED_METHOD_TRANSFER.md. Conditional bootstrap intervals do not include complete retraining uncertainty or multiplicity adjustment.', '']
    (ROOT / 'results/MATCHED_TIMESTRAL_V1.md').write_text('\n'.join(lines))
    write_json(ROOT / 'results/MATCHED_TIMESTRAL_CHECKS_V1.json', {'completed_utc': utc_now(),
        'source_sha256': sha256(Path(__file__)), 'predictions_sha256': sha256(path),
        'calibration_recomputed': scale, 'max_metric_reconciliation_difference': error,
        'market_outcomes_2026_accessed': False})
    print('Reconciled matched transfer; QLIKE', scores.loc['Timestral5M_calibrated', 'qlike'])


if __name__ == '__main__':
    main()
