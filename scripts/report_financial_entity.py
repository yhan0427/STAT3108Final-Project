"""Paired identity sensitivity and preregistered supplementary age comparisons."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now
from volrisk.metrics import block_bootstrap, qlike


def main():
    path = ROOT / 'results/financial/entity-v1/predictions.csv'
    audit = read_json(ROOT / 'results/FINANCIAL_ENTITY_AUDIT_V1.json')
    if sha256(path) != audit['predictions_sha256']:
        raise ValueError('Changed completed forecasts')
    plan_path = ROOT / 'config/entity-age-comparisons-v1.json'
    plan = read_json(plan_path)
    if sha256(ROOT / 'data/processed/financial-entity-v1/manifest.json') != plan['dataset_manifest_sha256']:
        raise ValueError('Changed sensitivity cohort')
    new = pd.read_csv(path, parse_dates=['date'])
    new = new.loc[new.role.eq('held_company') & new.split.eq('test')]
    intervals = {}
    for backbone in ['Ridge', 'LSTM']:
        for suffix in ['', '_calibrated']:
            for comparison in plan['comparisons']:
                a, b = comparison.split('-minus-')
                first, second = f'{backbone}_{a}{suffix}', f'{backbone}_{b}{suffix}'
                daily = pd.DataFrame({'date': new.date, 'difference': qlike(new.target, new[first]) - qlike(new.target, new[second])}).groupby('date').difference.mean()
                intervals[f'{first}-minus-{second}'] = {str(n): block_bootstrap(daily, n, plan['replicates']) for n in plan['blocks']}
    old_path = ROOT / 'results/financial/full-v1/predictions.csv'
    old = pd.read_csv(old_path, parse_dates=['date'])
    old = old.loc[old.role.eq('held_company') & old.split.eq('test')]
    matched = new[['ticker', 'date', 'target']].merge(old, on=['ticker', 'date'], suffixes=('_new', '_old'), validate='one_to_one')
    if len(matched) != len(new) or not np.allclose(matched.target_new, matched.target_old, atol=1e-12, rtol=0):
        raise ValueError('Unmatched targets in identity sensitivity')
    records = []
    for backbone in ['Ridge', 'LSTM']:
        for group in ['market', 'availability', 'financial']:
            for suffix in ['', '_calibrated']:
                name = f'{backbone}_{group}{suffix}'
                records.append({'model': name, 'companies': new.ticker.nunique(), 'origins': len(new),
                    'predecessor_reference_same_origins_qlike': float(qlike(matched.target_old, matched[name]).mean()),
                    'entity_guarded_same_origins_qlike': float(qlike(new.target, new[name]).mean())})
    pd.DataFrame(records).to_csv(ROOT / 'results/FINANCIAL_ENTITY_MATCHED_V1.csv', index=False)
    write_json(ROOT / 'results/FINANCIAL_ENTITY_AGE_V1.json', {'completed_utc': utc_now(),
        'plan_sha256': sha256(plan_path), 'predictions_sha256': sha256(path),
        'original_predictions_sha256': sha256(old_path), 'paired_intervals': intervals,
        'scope': 'Exploratory sensitivity; policy and training population changed. Paired original forecasts rescored on identical80held company-date keys; not a causal estimate or independent replication.',
        'market_outcomes_2026_accessed': False})
    scores = pd.read_csv(ROOT / 'results/FINANCIAL_ENTITY_V1.csv').set_index('model')
    lines = ['# Financial information: entity-spell sensitivity', '',
        'The documented MTCH, DD, SW and KHC identity policy restricts unverified predecessor histories and disclosures before fitting. This is a conservative sensitivity, not exhaustive corporate-action certification. Earlier experiments remain preserved.', '',
        '327 training companies /547,522 fitting origins;327 non-held validation companies /155,599 origins;80 held companies /38,480 evaluation origins. SW no longer supplies a512-observation current-spell context by2025. Companies are not replaced. Evaluation years2024–2025 were already inspected;2026 remains sealed.', '',
        '| Information | Ridge calibrated QLIKE | LSTM calibrated QLIKE | LSTM raw QLIKE |',
        '|---|---:|---:|---:|']
    for group in ['market', 'filing_age', 'availability', 'financial']:
        lines.append(f'| {group} | {scores.loc[f"Ridge_{group}_calibrated", "qlike"]:.6f} | {scores.loc[f"LSTM_{group}_calibrated", "qlike"]:.6f} | {scores.loc[f"LSTM_{group}", "qlike"]:.6f} |')
    lines += ['', '## Incremental information and uncertainty', '',
        '| Calibrated LSTM difference | Mean QLIKE difference |20-date95% interval |60-date95% interval |', '|---|---:|---|---|']
    all_intervals = {**audit['paired_intervals'], **intervals}
    for a, b in [('filing_age', 'market'), ('financial', 'availability'), ('financial', 'filing_age')]:
        row = all_intervals[f'LSTM_{a}_calibrated-minus-LSTM_{b}_calibrated']; x, y = row['20'], row['60']
        lines.append(f'| {a} minus {b} | {x["mean_difference"]:.6f} | [{x["ci95_low"]:.6f}, {x["ci95_high"]:.6f}] | [{y["ci95_low"]:.6f}, {y["ci95_high"]:.6f}] |')
    lines += ['', 'Lower loss is better. Financial-versus-combined-metadata evidence and financial-versus-age evidence answer different questions: adding seven missingness indicators can change neural fitting even though age is nested in the information set. The strongest substantive claim requires outperforming the simpler age control, not just one chosen metadata implementation. Report both raw and validation-calibrated losses; do not select whichever comparison is most favorable.', '',
        'Intervals resample whole dates in20/60-session circular blocks and condition on the fitted ensemble. They do not capture all retraining/cohort uncertainty, adjust for multiple comparisons or create a fresh holdout. Same-origin old/new scores are in FINANCIAL_ENTITY_MATCHED_V1.csv; absolute81-versus80-company scores are not an identity correction effect.', '',
        'Next: finish the already-frozen matched100-company multimodal experiment, including text age/length/availability controls. Translate candidate forecasts into decision outcomes under fixed rules and costs. A model-loss gain alone does not establish trading profit or alpha.', '']
    (ROOT / 'results/FINANCIAL_ENTITY_V1.md').write_text('\n'.join(lines))
    print('\n'.join(lines[16:27]))


if __name__ == '__main__':
    main()
