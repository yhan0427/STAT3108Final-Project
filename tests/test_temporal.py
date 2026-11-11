import numpy as np
import pandas as pd
import pytest
from volrisk.portfolio import simulate
from volrisk.temporal import freeze_validation_exposure,full_cycle_frame,exposure_schedule


def test_cash_accrues_over_calendar_days_and_only_existing_cash():
    dates=pd.to_datetime(['2024-01-05','2024-01-08'])
    frame=pd.DataFrame({'A':[0.,0.]},index=dates)
    result=simulate(frame,[1.],{dates[0]:.5},0,.04)
    assert result['path'][-1]['nav']==pytest.approx(.5+.5*1.04**(3/365))
    fully_invested=simulate(frame,[1.],{dates[0]:1.},0,.04)
    assert fully_invested['metrics']['total_return']==pytest.approx(0.)


def test_fixed_exposure_rejects_test_dates():
    with pytest.raises(ValueError,match='validation period'):
        freeze_validation_exposure(None,pd.DataFrame({'date':['2026-01-02']}),{'A':1.},'model')


def test_complete_cycles_end_on_scheduled_signal_target():
    dates=pd.bdate_range('2026-01-02',periods=43)
    frame=pd.DataFrame({'date':dates,'target_end':dates+pd.offsets.BDay(21)})
    output=full_cycle_frame(frame)
    assert output.date.max()==dates[40]
    assert output.target_end.max()==dates[40]+pd.offsets.BDay(21)


def test_validation_control_does_not_read_future_prices():
    dates=pd.bdate_range('2021-01-01',periods=800)
    prices=pd.DataFrame({'A':100*np.exp(np.sin(np.arange(800))*.01)},index=dates)
    origins=dates[(dates>='2022-01-01') & (dates<'2023-01-01')]
    forecasts=pd.DataFrame({'date':origins,'ticker':'A','model':.2})
    before=freeze_validation_exposure(prices,forecasts,{'A':1.},'model')
    prices.loc[prices.index>origins.max(),'A']=1e9
    after=freeze_validation_exposure(prices,forecasts,{'A':1.},'model')
    assert before==pytest.approx(.5) and before==after
