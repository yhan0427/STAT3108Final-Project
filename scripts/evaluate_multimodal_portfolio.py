"""Apply frozen multimodal forecasts to the previously registered decision grid."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now
from volrisk.metrics import block_bootstrap
from volrisk.temporal import evaluate_policies, freeze_validation_exposure


def main():
    destination = ROOT / 'results/portfolio/multimodal-v1'
    if destination.exists():
        raise FileExistsError('Preserve application experiment')
    config_path = ROOT / 'config/multimodal-portfolio-v1.json'
    plan = read_json(config_path)
    forecast_dir = ROOT / 'results/financial/multimodal-100-v1'
    if not (forecast_dir / 'COMPLETED.json').exists():
        raise ValueError('Forecast experiment must finish first')
    if sha256(forecast_dir / 'registered_plan.json') != plan['forecast_plan_sha256']:
        raise ValueError('Forecast protocol changed after application registration')
    for name, expected in plan['engine_hashes'].items():
        if sha256(ROOT / f'src/volrisk/{name}.py') != expected:
            raise ValueError('Decision engine changed')
    path = forecast_dir / 'predictions.csv'
    forecast_audit = read_json(ROOT / 'results/MULTIMODAL_100_AUDIT_V1.json')
    if sha256(path) != forecast_audit['predictions_sha256']:
        raise ValueError('Completed forecasts changed')
    frame = pd.read_csv(path, parse_dates=['date', 'target_end'])
    if frame.date.max() >= pd.Timestamp('2026-01-01') or frame.target_end.max() >= pd.Timestamp('2026-01-01'):
        raise ValueError('Sealed outcomes found')
    frame = frame.loc[frame.role.eq('held_company')].copy()
    if frame.duplicated(['ticker', 'date']).any():
        raise ValueError('Duplicate application forecasts')
    symbols = sorted({t for weights in plan['baskets'].values() for t in weights})
    prices = {}
    price_hashes = {}
    for ticker in symbols:
        source = ROOT / f'data/raw/market/{ticker}.csv'
        price = pd.read_csv(source, parse_dates=['date']).set_index('date').adj_close
        if price.index.max() >= pd.Timestamp('2026-01-01'):
            raise ValueError('Sealed price cache found')
        prices[ticker], price_hashes[ticker] = price, sha256(source)
    prices = pd.DataFrame(prices).sort_index()
    if prices.loc['2021-01-01':].isna().any().any() or (prices.dropna() <= 0).any().any():
        raise ValueError('Incomplete/nonpositive basket price calendar')
    destination.mkdir(parents=True)
    write_json(destination / 'registered_plan.json', {'config_sha256': sha256(config_path),
        'forecasts_sha256': sha256(path), 'source_sha256': sha256(Path(__file__)),
        'price_hashes': price_hashes, 'started_utc': utc_now()})
    rows, paths, comparisons, constants = [], [], [], {}
    for basket, weights in plan['baskets'].items():
        basket_frame = frame.loc[frame.ticker.isin(weights)]
        counts = basket_frame.groupby('date').ticker.nunique()
        basket_frame = basket_frame.loc[basket_frame.date.isin(counts.index[counts.eq(len(weights))])]
        validation = basket_frame.loc[basket_frame.split.eq('validation')]
        evaluation = basket_frame.loc[basket_frame.split.eq('test')]
        if validation.empty or evaluation.empty:
            raise ValueError('Missing basket validation/evaluation')
        constants[basket] = {m: freeze_validation_exposure(prices, validation, weights, m, plan['target']) for m in plan['models']}
        for model in plan['models']:
            for cost in plan['cost_bps']:
                for cash in plan['cash_annual']:
                    result = evaluate_policies(prices, evaluation, weights, model, constants[basket][model], cost, cash, plan['target'])
                    case = {'basket': basket, 'model': model, 'cost_bps': cost, 'cash_annual': cash,
                        'fixed_validation_exposure': constants[basket][model], 'start': result['start'], 'end': result['end']}
                    for strategy, values in result['strategies'].items():
                        if not all(k in values['metrics'] for k in plan['required_metrics']):
                            raise ValueError('Missing decision outcome')
                        rows.append({**case, 'strategy': strategy, **values['metrics']})
                    if cost == plan['primary']['cost_bps'] and cash == plan['primary']['cash_annual']:
                        dynamic = result['strategies']['forecast']
                        av = np.asarray(dynamic['rolling_volatility'], dtype=float)
                        for benchmark in ['fixed_validation', 'historical']:
                            other = result['strategies'][benchmark]
                            bv = np.asarray(other['rolling_volatility'], dtype=float)
                            comparisons.append({**case, 'benchmark': benchmark,
                                'risk_error_difference': dynamic['metrics']['risk_target_mae'] - other['metrics']['risk_target_mae'],
                                'return_difference': dynamic['metrics']['total_return'] - other['metrics']['total_return'],
                                'equity_difference': dynamic['metrics']['mean_equity_fraction'] - other['metrics']['mean_equity_fraction'],
                                'intervals': {str(n): block_bootstrap(np.abs(av - plan['target']) - np.abs(bv - plan['target']), n, 1000) for n in [20, 60]}})
                        for strategy, values in result['strategies'].items():
                            paths.extend({**case, 'strategy': strategy, **point} for point in values['path'])
        print('Completed', basket, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / 'results/MULTIMODAL_PORTFOLIO_V1.csv', index=False)
    pd.DataFrame(paths).to_csv(destination / 'paths.csv', index=False)
    write_json(ROOT / 'results/MULTIMODAL_PORTFOLIO_AUDIT_V1.json', {'completed_utc': utc_now(),
        'plan': plan, 'frozen_validation_exposures': constants, 'comparisons': comparisons,
        'strategy_records': len(rows), 'calibration_source': 'Frozen non-held validation only; no basket-label refit',
        'market_outcomes_2026_accessed': False})
    write_json(destination / 'COMPLETED.json', {'completed_utc': utc_now(), 'strategy_records': len(rows)})
    print('Strategy records', len(rows), flush=True)


if __name__ == '__main__':
    main()
