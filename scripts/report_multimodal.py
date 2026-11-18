"""Reconcile matched forecast scores and display all declared controls, including failures."""
from pathlib import Path
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now
from volrisk.metrics import forecast_metrics, qlike


def main():
    destination = ROOT / 'results/financial/multimodal-100-v1'
    if not (destination / 'COMPLETED.json').exists():
        raise ValueError('Await completed frozen run')
    audit = read_json(ROOT / 'results/MULTIMODAL_100_AUDIT_V1.json')
    path = destination / 'predictions.csv'
    if sha256(path) != audit['predictions_sha256']:
        raise ValueError('Forecast artifact changed')
    frame = pd.read_csv(path, parse_dates=['date'])
    held = frame.role.eq('held_company') & frame.split.eq('test')
    evaluation = frame.loc[held]
    scores = pd.read_csv(ROOT / 'results/MULTIMODAL_100_V1.csv').set_index('model')
    differences = []
    for name, row in scores.iterrows():
        check = forecast_metrics(evaluation.target, evaluation[name], evaluation.Persistence)
        for metric in ['mae', 'rmse', 'qlike', 'r2_vs_persistence']:
            differences.append(abs(check[metric] - row[metric]))
    if max(differences) > 1e-10:
        raise ValueError('Forecast metric reconciliation failed')
    seed_rows = []
    for backbone in ['LSTM']:
        for group in audit['registered_plan']['masks']:
            name = f'{backbone}_{group}'
            for seed in audit['registered_plan']['config']['seeds']:
                predictions = np.load(destination / f'{name}_{seed}.npz')['predictions']
                if len(predictions) != len(frame):
                    raise ValueError('Seed/order mismatch')
                for suffix, scale in [('', 1.), ('_calibrated', audit['calibration'][name])]:
                    seed_rows.append({'model': name + suffix, 'seed': seed,
                        'calibration_scope': 'Same non-held-validation ensemble factor for each member; diagnostic only',
                        **forecast_metrics(evaluation.target, predictions[held] * scale, evaluation.Persistence)})
    pd.DataFrame(seed_rows).to_csv(ROOT / 'results/MULTIMODAL_SEEDS_100_V1.csv', index=False)
    groups = list(audit['registered_plan']['masks'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    labels = ['Market', 'Financial\nmetadata', 'Financial', 'Text\nmetadata', 'Text', 'Combined\nmetadata', 'Financial +\ntext metadata', 'All']
    x = np.arange(len(groups))
    for axis, backbone in zip(axes, ['Ridge', 'LSTM']):
        raw = [scores.loc[f'{backbone}_{g}', 'qlike'] for g in groups]
        calibrated = [scores.loc[f'{backbone}_{g}_calibrated', 'qlike'] for g in groups]
        axis.bar(x - .19, raw, .38, color='#a5b4c5', label='Raw')
        axis.bar(x + .19, calibrated, .38, color='#173b67', label='Validation calibrated')
        axis.set_xticks(x, labels, rotation=35, ha='right', fontsize=8)
        axis.set_title(backbone)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    axes[0].set_ylabel('Held-company mean QLIKE (lower is better)')
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=9, ncol=2,
               loc='upper center', bbox_to_anchor=(.5, .91))
    fig.suptitle('Matched modalities and metadata controls · 19 held companies · 2024–2025', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .85])
    fig.savefig(ROOT / 'results/MULTIMODAL_100_V1.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    lines = ['# Matched financial and text information experiment', '',
        'Research question: do financial values and annual-report language add predictive information beyond the same market history and disclosure metadata?', '',
        'Fixed 100-issuer non-financial reference subset:80 fitting companies,130,065 fitting origins;80 non-held validation companies,37,730 origins;19 held companies,9,139 inspected2024–2025 origins. The twentieth held issuer(SW) lacks sufficient guarded current-security history; no replacement. Missing text remains in the sample. These are development results, not a fresh temporal confirmation.2026 remains sealed.', '',
        'Ridge and three-seed LSTM use identical60-session market inputs. Each information group has the same52 static coordinates, masking excluded modalities. Frozen768-dimensional FinBERT vectors are reduced using475 unique fitting-period training-company documents only. Primary quality exclusions and identity boundaries were fixed before these scores. Financial/text availability,age and text-length controls are explicitly retained. Full details:TEXT_PIPELINE_100_V1.md and MULTIMODAL_DATA_CHECKS_100_V1.json.', '',
        '| Information | Ridge raw QLIKE | Ridge calibrated | LSTM raw QLIKE | LSTM calibrated |', '|---|---:|---:|---:|---:|']
    for group, label in zip(groups, [x.replace('\n', ' ') for x in labels]):
        values = [scores.loc[f'{backbone}_{group}{suffix}', 'qlike'] for backbone in ['Ridge', 'LSTM'] for suffix in ['', '_calibrated']]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', '## Findings within this specification', '',
        'Frozen FinBERT representations do not show a stable incremental benefit here. Ridge text-versus-text-metadata and all-versus-financial+text-metadata differences are small and both block intervals cross zero. For LSTM, adding embeddings conditional on financial values and text metadata increases calibrated QLIKE, with both date-block intervals above zero. This applies to the frozen document pooling/PCA/fusion procedure and observed cohort; it does not establish that language is universally uninformative.', '',
        'Ridge financial values improve calibrated QLIKE over its combined age/missingness control in both date intervals. The LSTM financial-content contrast crosses zero. A separately registered simple-age sensitivity addresses the earlier finding that age alone can fit better than the larger metadata control; do not select only the favorable comparator.', '',
        '## Declared incremental contrasts', '',
        '| Calibrated procedure difference | Mean QLIKE difference |20-date95% interval |60-date95% interval |', '|---|---:|---|---|']
    for backbone in ['Ridge', 'LSTM']:
        for contrast in audit['registered_plan']['primary_comparisons']:
            a, b = contrast.split('-minus-')
            name = f'{backbone}_{a}_calibrated-minus-{backbone}_{b}_calibrated'
            values = audit['paired_intervals'][name]
            x, y = values['20'], values['60']
            lines.append(f'| {backbone}: {a} minus {b} | {x["mean_difference"]:+.6f} | [{x["ci95_low"]:+.6f}, {x["ci95_high"]:+.6f}] | [{y["ci95_low"]:+.6f}, {y["ci95_high"]:+.6f}] |')
    lines += ['', 'Financial-control means filing age and missingness; text-control means document age,length,availability and newer-original-unavailable indicator. All-minus-financial+text-control isolates the embedding contribution conditional on financial values and text metadata. All-minus-market alone does not identify language content value.', '',
        'Both raw and calibrated results are retained. Calibration and Ridge alpha/early stopping use only non-held validation outcomes. Member-seed diagnostics use the same ensemble calibration factor; ensemble loss is not the average member loss. Dates are resampled together across companies using20/60-session circular blocks. Intervals condition on the fitted procedure/cohort, omit full retraining uncertainty and are not multiplicity-adjusted. One negative or positive result cannot establish universal multimodal effectiveness.', '',
        'Year/company tables and all seed outcomes accompany this report. The application grid is frozen separately in config/multimodal-portfolio-v1.json. Forecast loss is not net decision utility: examine risk-target error,return,drawdown,mean equity,turnover and costs against validation-fixed exposure and historical risk. No portfolio result is inferred from this table alone.', '']
    (ROOT / 'results/MULTIMODAL_100_V1.md').write_text('\n'.join(lines))
    write_json(ROOT / 'results/MULTIMODAL_CHECKS_100_V1.json', {'completed_utc': utc_now(),
        'source_sha256': sha256(Path(__file__)), 'predictions_sha256': sha256(path),
        'scores_reconciled': len(scores), 'max_metric_difference': max(differences),
        'seed_diagnostics': len(seed_rows), 'market_outcomes_2026_accessed': False})
    print(scores[['mae', 'qlike']].to_string())


if __name__ == '__main__':
    main()
