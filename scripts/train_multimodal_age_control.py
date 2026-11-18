"""Registered simple-age control using the parent experiment's exact preprocessing."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.aligned import variance_calibration
from volrisk.dataset import MARKET_FEATURES
from volrisk.fundamentals_v3 import V3_FEATURES
from volrisk.io import read_json, write_json, sha256, utc_now
from volrisk.metrics import forecast_metrics, qlike, block_bootstrap
from volrisk.models import train_network, predict_network, Sequences
from volrisk.stream_ridge import StreamRidge


def main():
    parent = ROOT / 'results/financial/multimodal-100-v1'
    if not (parent / 'COMPLETED.json').exists():
        raise ValueError('Parent experiment must finish first')
    policy_path = ROOT / 'config/multimodal-age-control-v1.json'
    policy = read_json(policy_path)
    if sha256(parent / 'registered_plan.json') != policy['parent_plan_sha256']:
        raise ValueError('Changed parent protocol')
    parent_audit = read_json(ROOT / 'results/MULTIMODAL_100_AUDIT_V1.json')
    panel_path = ROOT / 'data/processed/multimodal-100-v1/panel.pkl'
    if sha256(panel_path) != parent_audit['registered_plan']['panel_sha256'] or sha256(parent / 'predictions.csv') != parent_audit['predictions_sha256']:
        raise ValueError('Changed parent data/forecasts')
    destination = ROOT / 'results/financial/multimodal-age-v1'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'COMPLETED.json').exists():
        raise FileExistsError('Completed age sensitivity is immutable')
    plan = {'registered_utc': utc_now(), 'policy_sha256': sha256(policy_path),
        'parent_predictions_sha256': sha256(parent / 'predictions.csv'),
        'parent_preprocessing_sha256': sha256(parent / 'preprocessing.joblib'),
        'source_sha256': sha256(Path(__file__)), 'market_outcomes_2026_accessed': False}
    registered = destination / 'registered_plan.json'
    if registered.exists():
        old = read_json(registered)
        if {k: v for k, v in old.items() if k != 'registered_utc'} != {k: v for k, v in plan.items() if k != 'registered_utc'}:
            raise ValueError('Changed sensitivity contract')
    else:
        write_json(registered, plan)
    panel = pd.read_pickle(panel_path)
    preprocessing = joblib.load(parent / 'preprocessing.joblib')
    cfg = parent_audit['registered_plan']['config']
    held = panel.role.eq('held_company')
    train = panel.index[panel.split.eq('train') & ~held].to_numpy()
    validation = panel.index[panel.split.eq('validation') & ~held & panel.context_ready.eq(True)].to_numpy()
    held_validation = panel.index[panel.split.eq('validation') & held & panel.context_ready.eq(True)].to_numpy()
    evaluation = panel.index[panel.split.eq('test') & held & panel.context_ready.eq(True)].to_numpy()
    indices = np.r_[validation, held_validation, evaluation]
    base = pd.read_csv(parent / 'predictions.csv', parse_dates=['date'])
    if not np.array_equal(base.date.to_numpy(), panel.loc[indices, 'date'].to_numpy()) or not np.array_equal(base.ticker.to_numpy(), panel.loc[indices, 'ticker'].to_numpy()):
        raise ValueError('Prediction order mismatch')
    market = preprocessing['market'].transform(panel[MARKET_FEATURES])
    static = np.zeros((len(panel), 52), dtype=np.float32)
    static[:, :16] = preprocessing['financial'].transform(panel[V3_FEATURES], missing=True)
    static *= np.asarray(policy['mask'], dtype=np.float32)
    y = np.log(panel.target.clip(lower=1e-8).fillna(1)).to_numpy()
    mean, std = preprocessing['y_mean'], preprocessing['y_std']
    ys = ((y - mean) / std).astype(np.float32)
    choices, best = [], None
    with threadpool_limits(limits=4):
        statistics = StreamRidge.statistics(market, train, y, 60, static=static)
        for alpha in parent_audit['registered_plan']['ridge_alpha_grid']:
            model = StreamRidge(alpha).fit_statistics(statistics)
            error = float(np.mean((model.predict(market, validation, static=static) - y[validation]) ** 2))
            choices.append({'alpha': alpha, 'validation_log_mse': error})
            if best is None or error < best[0]:
                best = error, model
        base['Ridge_filing_age'] = np.exp(np.clip(best[1].predict(market, indices, static=static), -18, 5))
        joblib.dump(best[1], destination / 'Ridge_filing_age.joblib')
    predictions, histories = [], {}
    for seed in cfg['seeds']:
        cache = destination / f'LSTM_filing_age_{seed}.npz'
        meta_path = cache.with_suffix('.json')
        if cache.exists():
            meta = read_json(meta_path)
            if sha256(cache) != meta['predictions_sha256'] or sha256(destination / meta['weights_file']) != meta['weights_sha256']:
                raise ValueError('Changed checkpoint')
            prediction, history = np.load(cache)['predictions'], meta['history']
        else:
            model, history = train_network(market, static, ys, train, validation, cfg, seed, True)
            prediction = np.exp(np.clip(predict_network(model, Sequences(market, static, ys, indices, 60)) * std + mean, -18, 5))
            weights = destination / f'LSTM_filing_age_{seed}.pt'
            torch.save(model.state_dict(), weights)
            np.savez(cache, predictions=prediction)
            write_json(meta_path, {'history': history, 'predictions_sha256': sha256(cache), 'weights_file': weights.name, 'weights_sha256': sha256(weights)})
        predictions.append(prediction)
        histories[str(seed)] = history
    base['LSTM_filing_age'] = np.mean(predictions, axis=0)
    validation_rows = base.role.eq('training_company') & base.split.eq('validation')
    scales = {}
    for name in ['Ridge_filing_age', 'LSTM_filing_age']:
        scales[name] = variance_calibration(base.loc[validation_rows, 'target'], base.loc[validation_rows, name])
        base[name + '_calibrated'] = base[name] * scales[name]
    base.to_csv(destination / 'predictions.csv', index=False)
    evaluated = base.loc[base.role.eq('held_company') & base.split.eq('test')]
    scores, intervals = [], {}
    for backbone in ['Ridge', 'LSTM']:
        for suffix in ['', '_calibrated']:
            name = f'{backbone}_filing_age{suffix}'
            scores.append({'model': name, **forecast_metrics(evaluated.target, evaluated[name], evaluated.Persistence)})
            for comparison in policy['comparisons']:
                a, b = comparison.split('-minus-')
                first, second = f'{backbone}_{a}{suffix}', f'{backbone}_{b}{suffix}'
                daily = pd.DataFrame({'date': evaluated.date, 'difference': qlike(evaluated.target, evaluated[first]) - qlike(evaluated.target, evaluated[second])}).groupby('date').difference.mean()
                intervals[first + '-minus-' + second] = {str(n): block_bootstrap(daily, n, 1000) for n in [20, 60]}
    pd.DataFrame(scores).to_csv(ROOT / 'results/MULTIMODAL_AGE_100_V1.csv', index=False)
    write_json(ROOT / 'results/MULTIMODAL_AGE_100_AUDIT_V1.json', {'completed_utc': utc_now(), 'plan': plan,
        'policy': policy, 'ridge_choices': choices, 'calibration': scales, 'history': histories,
        'paired_intervals': intervals, 'predictions_sha256': sha256(destination / 'predictions.csv'),
        'market_outcomes_2026_accessed': False})
    write_json(destination / 'COMPLETED.json', {'completed_utc': utc_now()})
    print(pd.DataFrame(scores).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
