"""Descriptive period/company/availability failures; no refitting or outcome selection."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now
from volrisk.metrics import qlike


def main():
    path = ROOT / 'results/financial/multimodal-age-v1/predictions.csv'
    audit = read_json(ROOT / 'results/MULTIMODAL_AGE_100_AUDIT_V1.json')
    if sha256(path) != audit['predictions_sha256']:
        raise ValueError('Changed completed sensitivity forecasts')
    frame = pd.read_csv(path, parse_dates=['date'])
    frame = frame.loc[frame.role.eq('held_company') & frame.split.eq('test')].copy()
    panel_path = ROOT / 'data/processed/multimodal-100-v1/panel.pkl'
    panel = pd.read_pickle(panel_path)
    frame = frame.merge(panel[['date', 'ticker', 'text_age_days', 'text_available']], on=['date', 'ticker'], validate='one_to_one')
    frame['year'] = frame.date.dt.year
    frame['text_age_group'] = np.select([frame.text_available.eq(0), frame.text_age_days.le(90), frame.text_age_days.le(365)], ['unavailable', '1-90days', '91-365days'], default='366-550days')
    rows, leave_one_out = [], {}
    contrasts = [('financial', 'filing_age'), ('text', 'text_control'), ('all', 'financial_text_control'), ('all', 'market')]
    for backbone in ['Ridge', 'LSTM']:
        for a, b in contrasts:
            first, second = f'{backbone}_{a}_calibrated', f'{backbone}_{b}_calibrated'
            temporary = frame.copy()
            temporary['qlike_difference'] = qlike(temporary.target, temporary[first]) - qlike(temporary.target, temporary[second])
            temporary['absolute_error_difference'] = np.abs(temporary.target - temporary[first]) - np.abs(temporary.target - temporary[second])
            name = f'{first}-minus-{second}'
            for field in ['year', 'ticker', 'text_age_group']:
                for value, part in temporary.groupby(field):
                    rows.append({'contrast': name, 'dimension': field, 'group': str(value), 'origins': len(part),
                        'companies': part.ticker.nunique(), 'mean_qlike_difference': float(part.qlike_difference.mean()),
                        'mean_absolute_error_difference': float(part.absolute_error_difference.mean())})
            company = temporary.groupby('ticker').qlike_difference.agg(['sum', 'count', 'mean'])
            leave = (company['sum'].sum() - company['sum']) / (company['count'].sum() - company['count'])
            leave_one_out[name] = {'companies_with_lower_mean_loss': int(company['mean'].lt(0).sum()),
                'companies_with_higher_mean_loss': int(company['mean'].gt(0).sum()),
                'leave_one_company_out_min_difference': float(leave.min()),
                'leave_one_company_out_max_difference': float(leave.max()),
                'worst_mean_loss_increases': company['mean'].nlargest(3).to_dict(),
                'largest_mean_loss_reductions': company['mean'].nsmallest(3).to_dict()}
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / 'results/MULTIMODAL_FAILURES_100_V1.csv', index=False)
    write_json(ROOT / 'results/MULTIMODAL_FAILURES_100_V1.json', {'completed_utc': utc_now(),
        'predictions_sha256': sha256(path), 'source_sha256': sha256(Path(__file__)),
        'contrasts': leave_one_out, 'scope': 'Post-result descriptive diagnostics, not causal language/content effects or new confirmatory subgroups. Leave-one-company-out rescoring does not refit models or estimate leave-one-company-out generalization.',
        'market_outcomes_2026_accessed': False})
    lines = ['# Multimodal failure and mechanism diagnostics', '',
        'These are post-result descriptions of the fixed19-company development cohort. No model or encoder was retrained,company replaced or subgroup selected for a primary positive claim. Negative differences favor the first procedure.', '',
        '| Calibrated contrast |2024 mean QLIKE difference |2025 mean QLIKE difference |Companies improved(out of19) |Leave-one-company-out mean range |', '|---|---:|---:|---:|---|']
    for name, stats in leave_one_out.items():
        years = result.loc[result.contrast.eq(name) & result.dimension.eq('year')].set_index('group')
        lines.append(f'| {name.replace("_calibrated", "")} | {years.loc["2024", "mean_qlike_difference"]:+.6f} | {years.loc["2025", "mean_qlike_difference"]:+.6f} | {stats["companies_with_lower_mean_loss"]} | [{stats["leave_one_company_out_min_difference"]:+.6f}, {stats["leave_one_company_out_max_difference"]:+.6f}] |')
    lines += ['', 'The supplementary simple-age control was registered before the primary matched forecast scores were produced, following the larger-cohort age finding. Ridge financial values improve over both age-only and combined financial metadata in this cohort; LSTM financial values do not show a reliable improvement over age alone. Financial content value is therefore backbone/cohort dependent,not an established universal gain.', '',
        'The CSV retains every company,year and text-availability/age bin. Unavailable text is retained,not deleted. A procedure difference within an unavailable-text subgroup can reflect how shared parameters were fitted on other companies; it is not a within-company causal text effect. Year signs and company counts are descriptive and do not make nineteen firms a representative sample of all equities.', '',
        'Leave-one-company-out values remove one company only when rescoring fixed predictions. They detect concentrated average losses,not sensitivity to retraining or leave-one-company-out deployment. The existing date-block intervals retain shared market dates; fuller company/generalization uncertainty remains for later validation.', '',
        'Next research decision: stabilize and audit the information contribution before expanding an expensive text grid. Full-section frozen pooled embeddings can dilute rare risk information; this is a hypothesis,not a demonstrated explanation. Any event-window,section-aware or alternative-encoder extension needs a new declared development protocol and must preserve the sealed2026 sample.', '']
    (ROOT / 'results/MULTIMODAL_FAILURES_100_V1.md').write_text('\n'.join(lines))
    print(result.loc[result.dimension.eq('year'), ['contrast', 'group', 'mean_qlike_difference']].to_string(index=False))


if __name__ == '__main__':
    main()
