"""Select application baskets using only frozen roles, sector and history coverage."""
from pathlib import Path
import hashlib
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now


def main():
    destination = ROOT / 'config/multimodal-portfolio-v1.json'
    if destination.exists():
        raise FileExistsError('Application rules already frozen')
    panel_path = ROOT / 'data/processed/multimodal-100-v1/panel.pkl'
    manifest_path = panel_path.parent / 'manifest.json'
    panel = pd.read_pickle(panel_path)
    if sha256(panel_path) != read_json(manifest_path)['files']['panel.pkl']:
        raise ValueError('Matched panel changed')
    counts = panel.loc[panel.role.eq('held_company') & panel.context_ready.eq(True)].groupby(['ticker', 'split']).size().unstack(fill_value=0)
    eligible = counts.index[(counts.get('validation', 0) >= 400) & (counts.get('test', 0) >= 400)]
    roster = pd.read_csv(ROOT / 'config/multimodal-100-v1.csv')
    candidates = roster.loc[roster.ticker.isin(eligible)].copy()
    candidates['application_rank'] = candidates.issuer_id.map(lambda x: hashlib.sha256(('Han-multimodal-portfolio-v1/' + x).encode()).hexdigest())
    sector = candidates.sort_values('application_rank').groupby('sector', sort=True).head(1).ticker.tolist()
    technology = candidates.loc[candidates.sector.eq('Information Technology')].sort_values('application_rank').ticker.tolist()
    if len(sector) != 10 or len(technology) != 3:
        raise ValueError('Unexpected availability-conditioned basket sizes')
    models = ['Persistence'] + [f'{b}_{g}' for b in ['Ridge', 'LSTM'] for g in
        ['market', 'financial_control', 'financial', 'text_control', 'text', 'metadata', 'financial_text_control', 'all']]
    models += [m + '_calibrated' for m in models.copy()]
    plan = {'registered_utc': utc_now(), 'source_sha256': sha256(Path(__file__)),
        'dataset_manifest_sha256': sha256(manifest_path),
        'forecast_plan_sha256': sha256(ROOT / 'results/financial/multimodal-100-v1/registered_plan.json'),
        'selection': 'Held-role companies with>=400 common512-context origins in each2022-23validation/2024-25evaluation partition; one per nonfinancial sector or all3eligibletechnology companies, fixed issuer SHA rank; no text-availability/forecast-outcome ranking. Basket coverage conditioning remains a limitation.',
        'baskets': {'Held nonfinancial sectors': dict.fromkeys(sector, 1 / len(sector)),
                    'Held technology trio': dict.fromkeys(technology, 1 / len(technology))},
        'models': models, 'cost_bps': [0, 10, 25], 'cash_annual': [0, .04],
        'target': .1, 'rebalance_days': 20, 'correlation_history': 60,
        'execution': 'After-close origin signal, execute next close, earn following returns; unchanged temporal/portfolio engine. No terminal forced sale.',
        'fixed_control': 'For each model/basket mean scheduled validation equity; never fit forecasts or calibration to held labels. Does not match realized evaluation exposure.',
        'primary': {'cost_bps': 10, 'cash_annual': 0, 'metric': '20-session rolling net risk-target absolute error'},
        'required_metrics': ['risk_target_mae', 'total_return', 'annualized_return', 'realized_volatility', 'max_drawdown', 'mean_equity_fraction', 'turnover', 'cost_initial_capital'],
        'uncertainty': 'Primary dynamic-minus-fixed/historical risk-error differences, whole-date blocks20/60; conditional, not multiplicity-adjusted.',
        'scope': 'Exploratory application in inspected calendar periods; baskets frozen during first matched model run, before matched evaluation scores. Not a fresh holdout, alpha claim or historical-index strategy.',
        'engine_hashes': {n: sha256(ROOT / f'src/volrisk/{n}.py') for n in ['temporal', 'portfolio']},
        'market_outcomes_2026_accessed': False}
    write_json(destination, plan)
    print(plan['baskets'])


if __name__ == '__main__':
    main()
