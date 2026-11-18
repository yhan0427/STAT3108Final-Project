"""Artifact integration: scope isolation, hidden outcomes and frozen-policy reconciliation."""
from pathlib import Path
import copy
import json

import pandas as pd
import pytest

from volrisk.portfolio import PortfolioLab


@pytest.fixture(scope='module')
def matched_lab():
    root = Path(__file__).resolve().parents[1]
    required = ['results/financial/multimodal-age-v1/predictions.csv',
        'results/MULTIMODAL_AGE_100_AUDIT_V1.json', 'results/robustness/v2/static_matched/predictions.csv',
        'results/MULTIMODAL_PORTFOLIO_V1.csv']
    if not all((root / name).exists() for name in required):
        pytest.skip('Complete local matched research/application artifacts first')
    return PortfolioLab(root)


def body(lab):
    protocol = next(p for p in lab.catalogue()['protocols'] if p['id'] == 'multimodal_matched')
    return {'protocol': protocol['id'], 'model': 'MM_Ridge_all', 'weights': protocol['presets']['Held nonfinancial sectors'],
        'date': protocol['dates'][0], 'target': .1, 'cost_bps': 10}


def test_matched_scope_rejects_old_cohort_holdings_and_models(matched_lab):
    request = body(matched_lab)
    with pytest.raises(ValueError, match='cohort'):
        matched_lab.evaluate({**request, 'weights': {'AAPL': 1.}})
    with pytest.raises(ValueError, match='model'):
        matched_lab.evaluate({**request, 'model': 'Boosting_market'})
    protocol = next(p for p in matched_lab.catalogue()['protocols'] if p['id'] == 'multimodal_matched')
    assert protocol['companies'] == 19 and len(protocol['tickers']) == 19
    assert 'target' not in matched_lab.validation.columns


def test_matched_context_hides_future_until_explicit_reveal(matched_lab):
    request = body(matched_lab)
    hidden = matched_lab.evaluate(request)
    assert 'backtest' not in hidden
    assert not any(e['id'] == 'E7' for e in hidden['context']['evidence'])
    assert 'qlike' not in json.dumps(hidden['context']).lower()
    assert 't+2 through t+21' in hidden['assumptions']['forecast_alignment']
    coverage = next(e['data'] for e in hidden['context']['evidence'] if e['id'] == 'E4')
    assert all(r['financial_filing_date'] is None or r['financial_filing_date'] < request['date'] for r in coverage)


def test_interactive_preset_matches_registered_decision_grid(matched_lab):
    result = matched_lab.evaluate({**body(matched_lab), 'reveal': True})
    recorded = pd.read_csv(matched_lab.root / 'results/MULTIMODAL_PORTFOLIO_V1.csv')
    recorded = recorded.loc[recorded.basket.eq('Held nonfinancial sectors') & recorded.model.eq('Ridge_all_calibrated') & recorded.cost_bps.eq(10) & recorded.cash_annual.eq(0)].set_index('strategy')
    assert 'fixed_validation' in result['backtest']['strategies']
    for strategy, outcome in result['backtest']['strategies'].items():
        for metric in ['total_return', 'risk_target_mae', 'max_drawdown', 'mean_equity_fraction', 'turnover', 'cost_initial_capital']:
            assert outcome['metrics'][metric] == pytest.approx(recorded.loc[strategy, metric], abs=1e-12)


def test_future_returns_do_not_change_matched_decision(matched_lab):
    request = body(matched_lab)
    before = matched_lab.decision(request['protocol'], request['model'], request['weights'], .1, request['date'])
    mutated = copy.copy(matched_lab)
    mutated.log_returns = matched_lab.log_returns.copy()
    mutated.log_returns.loc[mutated.log_returns.index > request['date']] = 999.
    after = mutated.decision(request['protocol'], request['model'], request['weights'], .1, request['date'])
    assert before == after


def test_model_context_distinguishes_market_inputs_from_text(matched_lab):
    request=body(matched_lab)
    market=matched_lab.evaluate({**request,'model':'MM_Ridge_market'})
    text=matched_lab.evaluate({**request,'model':'MM_Ridge_text'})
    assert 'no financial values' in market['assumptions']['model_inputs']
    assert 'frozen text PCs' in text['assumptions']['model_inputs']


def test_recent_model_replay_matches_its_registered_grid(matched_lab):
    if not matched_lab.timestral_source:
        pytest.skip('Complete optional matched Timestral transfer first')
    request={**body(matched_lab),'model':'MM_Timestral5M','reveal':True}
    result=matched_lab.evaluate(request)
    assert 'not historically tradable' in result['assumptions']['pretraining_limit']
    assert 'no financial values,text' in result['assumptions']['model_inputs']
    recorded=pd.read_csv(matched_lab.root/'results/MATCHED_TIMESTRAL_PORTFOLIO_V1.csv')
    recorded=recorded.loc[recorded.basket.eq('Held nonfinancial sectors') & recorded.model.eq('Timestral5M_calibrated') & recorded.cost_bps.eq(10) & recorded.cash_annual.eq(0)].set_index('strategy')
    for strategy,outcome in result['backtest']['strategies'].items():
        for metric in ['total_return','risk_target_mae','max_drawdown','mean_equity_fraction','turnover','cost_initial_capital']:
            assert outcome['metrics'][metric]==pytest.approx(recorded.loc[strategy,metric],abs=1e-12)
