from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pytest
from volrisk.portfolio import PortfolioLab, PRESETS, covariance_from_history, risk_summary, rebalance, simulate


def test_covariance_preserves_marginal_vols_and_risk_contributions():
    history=np.random.default_rng(3).normal(size=(60,3))
    cov=covariance_from_history(history,[.2,.3,.4])
    np.testing.assert_allclose(np.diag(cov),[.04,.09,.16])
    assert np.linalg.eigvalsh(cov).min()>=0
    risk=risk_summary([.2,.3,.5],cov,.1)
    assert sum(risk['risk_contributions'])==pytest.approx(1.)
    assert risk['scaled_volatility']==pytest.approx(.1)


def test_risk_rule_never_leverages_low_risk_basket():
    risk=risk_summary([1.],np.array([[.0025]]),.10)
    assert risk['equity_fraction']==1 and risk['cash_fraction']==0
    assert risk['scaled_volatility']==.05


def test_rebalance_is_self_financing_and_costs_buys_and_sells():
    old=np.array([.8,.2]);weights=np.array([.25,.25])
    stocks,cash,cost,traded=rebalance(old,0,weights,.001)
    assert stocks.sum()+cash+cost==pytest.approx(1.)
    assert cost==pytest.approx(.001*traded)
    np.testing.assert_allclose(stocks/(stocks.sum()+cash),weights)
    stocks,cash,cost,traded=rebalance(np.zeros(2),1,np.array([.5,.5]),.001)
    assert cash==pytest.approx(0)
    assert stocks.sum()==pytest.approx(1/1.001)


def test_new_allocation_cannot_earn_execution_day_return():
    dates=pd.bdate_range('2024-01-02',periods=3)
    frame=pd.DataFrame({'A':[.9,.1,0.]},index=dates)
    out=simulate(frame,[1.],{dates[0]:1.},0)
    assert out['path'][0]['nav']==1.
    assert out['path'][1]['nav']==pytest.approx(1.1)


def test_buy_hold_weights_drift_and_constant_prices_only_lose_costs():
    dates=pd.bdate_range('2024-01-02',periods=3)
    frame=pd.DataFrame({'A':[0.,1.,0.],'B':[0.,0.,1.]},index=dates)
    out=simulate(frame,[.5,.5],{dates[0]:1.},0)
    assert out['path'][-1]['nav']==pytest.approx(2.)  # Daily reset would incorrectly give 2.25.
    flat=frame*0
    out=simulate(flat,[.5,.5],{dates[0]:.5,dates[1]:1.},.001)
    assert out['metrics']['total_return']==pytest.approx(-out['metrics']['cost_initial_capital'])


@pytest.fixture(scope='module')
def lab():
    root=Path(__file__).resolve().parents[1]
    if not (root/'data/raw/market/AAPL.csv').exists() or not (root/'results/robustness/v2/static_matched/predictions.csv').exists():
        pytest.skip('Local prices and V2 predictions required')
    return PortfolioLab(root)


@pytest.mark.parametrize('patch',[{'weights':{'AAPL':.5}}, {'weights':{'AAPL':-1,'MSFT':2}}, {'weights':{'AAPL':float('nan')}}, {'weights':{'BAD':1}}, {'target':float('nan')}, {'target':True}, {'cost_bps':-1}, {'reveal':'yes'}])
def test_invalid_requests_rejected(lab,patch):
    body={'weights':PRESETS['Diversified five'],**patch}
    with pytest.raises(ValueError):lab.evaluate(body)


def test_portfolio_future_outcomes_not_returned_before_reveal(lab):
    body={'weights':PRESETS['Diversified five']}
    hidden=lab.evaluate(body)
    assert 'backtest' not in hidden
    assert 'E7' not in [f['id'] for f in hidden['context']['evidence']]
    revealed=lab.evaluate({**body,'reveal':True})
    assert len(revealed['backtest']['strategies'])==4
    assert revealed['context']['evidence'][-1]['id']=='E7'
    assert all(s['signal_date']<s['execution_date'] for s in revealed['backtest']['signals'])


def test_future_mutations_do_not_change_selected_decision(lab):
    other=copy.copy(lab);other.log_returns=lab.log_returns.copy()
    date='2024-01-02';weights=PRESETS['Diversified five']
    before=lab.decision('static_matched','Ridge_sequence_market',weights,.1,date)
    other.log_returns.loc[other.log_returns.index>date]=999
    after=other.decision('static_matched','Ridge_sequence_market',weights,.1,date)
    assert before==after


def test_equal_marginal_forecasts_produce_equal_rules(lab):
    other=copy.copy(lab);other.source=copy.copy(lab.source);other.source.frames=lab.source.frames.copy()
    frame=lab.source.frames['static_matched'].copy();frame['Ridge_sequence_market']=frame.rv20
    other.source.frames['static_matched']=frame;other._cache={}
    result=other.backtest('static_matched','Ridge_sequence_market',PRESETS['Diversified five'],.1,10)
    assert result['strategies']['historical']['metrics']==result['strategies']['forecast']['metrics']
