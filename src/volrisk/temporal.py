"""Frozen validation controls and temporal decision evaluation."""
import numpy as np
import pandas as pd
from .portfolio import covariance_from_history, risk_summary, simulate


def exposure_schedule(prices, forecasts, weights, model, target=.1, every=20):
    """Allocation computation reads only trailing returns and origin forecasts."""
    dates=sorted(pd.to_datetime(forecasts.date.unique()))[::every]
    values=forecasts.assign(date=pd.to_datetime(forecasts.date)).set_index(['date','ticker'])
    logs=np.log(prices).diff();symbols=list(weights);w=np.array(list(weights.values()))
    targets={};signals=[]
    for date in dates:
        history=logs.loc[:date,symbols].tail(60)
        day=values.loc[date].loc[symbols]
        covariance=covariance_from_history(history,day[model].to_numpy())
        exposure=risk_summary(w,covariance,target)['equity_fraction']
        position=prices.index.get_loc(date)
        execution=prices.index[position+1]
        targets[execution]=exposure
        signals.append({'signal_date':str(date.date()),'execution_date':str(execution.date()),'target_equity':exposure})
    if not targets: raise ValueError('No eligible rebalance dates')
    return targets,signals


def freeze_validation_exposure(prices, forecasts, weights, model, target=.1):
    """Mean scheduled validation target, without using test exposure or returns."""
    dates=pd.to_datetime(forecasts.date)
    if len(dates)==0 or dates.min()<pd.Timestamp('2022-01-01') or dates.max()>=pd.Timestamp('2024-01-01'):
        raise ValueError('Fixed exposure may use only the 2022-2023 validation period')
    targets,_=exposure_schedule(prices,forecasts,weights,model,target)
    return float(np.mean(list(targets.values())))


def full_cycle_frame(forecasts,every=20):
    frame=forecasts.copy();frame['date']=pd.to_datetime(frame.date);frame['target_end']=pd.to_datetime(frame.target_end)
    dates=sorted(frame.date.unique())
    if not dates: raise ValueError('No complete evaluation targets')
    last_signal=dates[::every][-1]
    return frame[frame.date<=last_signal].copy()


def evaluate_policies(prices, forecasts, weights, model, fixed_exposure, cost_bps, cash_annual=0., target=.1):
    if not np.isfinite(fixed_exposure) or not 0<=fixed_exposure<=1:
        raise ValueError('Invalid frozen exposure')
    frame=full_cycle_frame(forecasts)
    dynamic,signals=exposure_schedule(prices,frame,weights,model,target)
    historical,_=exposure_schedule(prices,frame,weights,'rv20',target)
    start=min(dynamic);end=frame.target_end.max()
    returns=prices[list(weights)].pct_change(fill_method=None).loc[start:end]
    strategies={'buy_hold':{start:1.},'historical':historical,'fixed_validation':{d:fixed_exposure for d in dynamic},'forecast':dynamic}
    output={}
    for name,targets in strategies.items():
        result=simulate(returns,list(weights.values()),targets,cost_bps/10000,cash_annual)
        values=np.asarray([np.nan if x is None else x for x in result['rolling_volatility']])
        result['metrics']['risk_target_mae']=float(np.nanmean(np.abs(values-target)))
        output[name]=result
    return {'strategies':output,'signals':signals,'start':str(start.date()),'end':str(end.date())}
