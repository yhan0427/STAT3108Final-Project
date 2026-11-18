"""Matched retrospective modern-model transfer; reuse raw inference, recalibrate fairly."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.aligned import context_matrix, variance_calibration
from volrisk.io import read_json, write_json, sha256, utc_now
from volrisk.metrics import forecast_metrics, qlike, block_bootstrap
from volrisk.temporal import evaluate_policies, freeze_validation_exposure


def main():
    policy_path = ROOT / 'config/matched-timestral-v1.json'
    policy = read_json(policy_path)
    parent = ROOT / 'results/financial/multimodal-age-v1/predictions.csv'
    source = ROOT / 'results/aligned/full-cohort-v1/predictions.csv'
    if sha256(parent) != policy['parent_predictions_sha256'] or sha256(source) != policy['full_market_predictions_sha256']:
        raise ValueError('Changed frozen forecast sources')
    for path, key in [('config/timestral-v1.json', 'timestral_config_sha256'),
                      ('scripts/predict_timestral.py', 'inference_source_sha256'),
                      ('config/multimodal-portfolio-v1.json', 'application_config_sha256'),
                      ('data/processed/multimodal-100-v1/manifest.json', 'matched_dataset_manifest_sha256')]:
        if sha256(ROOT / path) != policy[key]:
            raise ValueError('Changed registered transfer specification')
    destination = ROOT / 'results/financial/matched-timestral-v1'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'COMPLETED.json').exists():
        raise FileExistsError('Completed matched transfer is immutable')
    model_dir = ROOT / 'data/models/timestral-5m-bdd20cf'
    plan = {'policy_sha256': sha256(policy_path), 'source_sha256': sha256(Path(__file__)),
        'model_artifacts': {p.name: sha256(p) for p in model_dir.iterdir() if p.is_file()},
        'engine_hashes': {n: sha256(ROOT / f'src/volrisk/{n}.py') for n in ['temporal', 'portfolio']}}
    plan_path = destination / 'registered_plan.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('Transfer source/checkpoint changed')
    write_json(plan_path, plan)
    base = pd.read_csv(parent)
    original = pd.read_csv(source, usecols=['date', 'ticker', 'target', 'rv20', 'Timestral5M'])
    merged = base[['date', 'ticker', 'target', 'rv20']].merge(original, on=['date', 'ticker'], how='left', suffixes=('', '_source'), validate='one_to_one')
    missing = merged.Timestral5M.isna()
    if not (base.loc[missing, 'role'].eq('held_company') & base.loc[missing, 'split'].eq('validation')).all():
        raise ValueError('Unexpected missing reusable raw forecasts')
    for name in ['target', 'rv20']:
        if not np.allclose(merged.loc[~missing, name], merged.loc[~missing, name + '_source'], atol=1e-12, rtol=0):
            raise ValueError('Inconsistent reused forecast origins/targets: ' + name)
    panel_dir = ROOT / 'data/processed/multimodal-100-v1'
    manifest = read_json(panel_dir / 'manifest.json')
    if sha256(panel_dir / 'panel.pkl') != manifest['files']['panel.pkl']:
        raise ValueError('Changed context panel')
    panel = pd.read_pickle(panel_dir / 'panel.pkl')
    if panel.date.max() >= pd.Timestamp('2026-01-01'):
        raise ValueError('Sealed data found')
    keys = panel.reset_index()[['index', 'date', 'ticker']]
    keys['date'] = keys.date.dt.strftime('%Y-%m-%d')
    positions = base.loc[missing, ['date', 'ticker']].merge(keys, on=['date', 'ticker'], validate='one_to_one')['index'].to_numpy()
    if not panel.loc[positions, 'context_ready'].eq(True).all():
        raise ValueError('Unready identity-guarded context')
    contexts = destination / 'held_validation_contexts.npy'
    np.save(contexts, context_matrix(panel, positions).astype(np.float32))
    inference = destination / 'held_validation_timestral.npy'
    if inference.exists():
        checkpoint = read_json(str(inference) + '.json')
        if sha256(contexts) != checkpoint['input_sha256'] or sha256(inference) != checkpoint['output_sha256']:
            raise ValueError('Cached inference/source mismatch')
    else:
        subprocess.run([str(ROOT / '.venv-foundation/bin/python'), '-B', str(ROOT / 'scripts/predict_timestral.py'),
                        '--input', str(contexts), '--output', str(inference)], check=True)
    base['Timestral5M'] = merged.Timestral5M
    base.loc[missing, 'Timestral5M'] = np.load(inference)
    val = base.role.eq('training_company') & base.split.eq('validation')
    scale = variance_calibration(base.loc[val, 'target'], base.loc[val, 'Timestral5M'])
    base['Timestral5M_calibrated'] = base.Timestral5M * scale
    base.to_csv(destination / 'predictions.csv', index=False)
    evaluation = base.loc[base.role.eq('held_company') & base.split.eq('test')]
    scores, intervals, periods = [], {}, []
    models = ['Timestral5M', 'Timestral5M_calibrated']
    for name in models:
        scores.append({'model': name, **forecast_metrics(evaluation.target, evaluation[name], evaluation.Persistence)})
        for year, group in evaluation.groupby(evaluation.date.str[:4]):
            periods.append({'model': name, 'year': year, **forecast_metrics(group.target, group[name], group.Persistence)})
        for other in ['Ridge_market', 'LSTM_market', 'Ridge_all', 'LSTM_all']:
            comparator = other + ('_calibrated' if name.endswith('_calibrated') else '')
            daily = pd.DataFrame({'date': evaluation.date, 'difference': qlike(evaluation.target, evaluation[name]) - qlike(evaluation.target, evaluation[comparator])}).groupby('date').difference.mean()
            intervals[name + '-minus-' + comparator] = {str(n): block_bootstrap(daily, n, 1000) for n in [20, 60]}
    pd.DataFrame(scores).to_csv(ROOT / 'results/MATCHED_TIMESTRAL_V1.csv', index=False)
    pd.DataFrame(periods).to_csv(ROOT / 'results/MATCHED_TIMESTRAL_PERIODS_V1.csv', index=False)
    application = read_json(ROOT / 'config/multimodal-portfolio-v1.json')
    symbols = sorted({t for weights in application['baskets'].values() for t in weights})
    prices = pd.DataFrame({t: pd.read_csv(ROOT / f'data/raw/market/{t}.csv', parse_dates=['date']).set_index('date').adj_close for t in symbols})
    if prices.index.max() >= pd.Timestamp('2026-01-01'):
        raise ValueError('Sealed prices found')
    frame = base.loc[base.role.eq('held_company')].copy()
    rows, comparisons, constants = [], [], {}
    for basket, weights in application['baskets'].items():
        selected = frame.loc[frame.ticker.isin(weights)]
        counts = selected.groupby('date').ticker.nunique()
        selected = selected.loc[selected.date.isin(counts.index[counts.eq(len(weights))])]
        constants[basket] = {}
        for model in models:
            fixed = freeze_validation_exposure(prices, selected.loc[selected.split.eq('validation')], weights, model, application['target'])
            constants[basket][model] = fixed
            for cost in application['cost_bps']:
                for cash in application['cash_annual']:
                    out = evaluate_policies(prices, selected.loc[selected.split.eq('test')], weights, model, fixed, cost, cash, application['target'])
                    case = {'basket': basket, 'model': model, 'cost_bps': cost, 'cash_annual': cash, 'fixed_validation_exposure': fixed}
                    for strategy, value in out['strategies'].items():
                        rows.append({**case, 'strategy': strategy, **value['metrics']})
                    if cost == 10 and cash == 0:
                        dynamic = out['strategies']['forecast']
                        for benchmark in ['fixed_validation', 'historical']:
                            other = out['strategies'][benchmark]
                            av = np.asarray(dynamic['rolling_volatility'], dtype=float)
                            bv = np.asarray(other['rolling_volatility'], dtype=float)
                            comparisons.append({**case, 'benchmark': benchmark,
                                'risk_error_difference': dynamic['metrics']['risk_target_mae'] - other['metrics']['risk_target_mae'],
                                'return_difference': dynamic['metrics']['total_return'] - other['metrics']['total_return'],
                                'equity_difference': dynamic['metrics']['mean_equity_fraction'] - other['metrics']['mean_equity_fraction'],
                                'intervals': {str(n): block_bootstrap(np.abs(av - .1) - np.abs(bv - .1), n, 1000) for n in [20, 60]}})
        print('Application complete', basket, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / 'results/MATCHED_TIMESTRAL_PORTFOLIO_V1.csv', index=False)
    write_json(ROOT / 'results/MATCHED_TIMESTRAL_AUDIT_V1.json', {'completed_utc': utc_now(), 'plan': plan,
        'policy': policy, 'raw_forecasts_reused': int((~missing).sum()), 'new_held_validation_forecasts': int(missing.sum()),
        'nonheld_validation_calibration': scale, 'paired_intervals': intervals,
        'frozen_validation_exposures': constants, 'decision_comparisons': comparisons,
        'strategy_records': len(rows), 'predictions_sha256': sha256(destination / 'predictions.csv'),
        'market_outcomes_2026_accessed': False})
    contexts.unlink()
    write_json(destination / 'COMPLETED.json', {'completed_utc': utc_now()})
    print(pd.DataFrame(scores).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
