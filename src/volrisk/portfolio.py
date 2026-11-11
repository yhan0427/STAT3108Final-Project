"""Historical portfolio experiments; signals at close, execution next close.

No future returns are consulted by the allocation rule. Earlier forecasting
experiments are inputs, not evidence of an untouched strategy holdout.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from .copilot import RiskEvidence

MODELS = ['Ridge_sequence_market', 'Boosting_market', 'LSTM_market', 'LSTM_missingness', 'LSTM_fundamentals']
STRATEGIES = {'buy_hold':'Buy & hold', 'fixed_half':'Fixed 50% equity', 'historical':'Historical-risk rule', 'forecast':'Forecast-risk rule'}
PRESETS = {
    'Diversified five': dict.fromkeys(['AAPL','JNJ','CAT','XOM','PG'], .2),
    'Technology five': dict.fromkeys(['AAPL','MSFT','NVDA','ORCL','ADBE'], .2),
}


def covariance_from_history(history, volatilities):
    """Same trailing correlation estimator for both marginal-volatility rules."""
    x=np.asarray(history,dtype=float); v=np.asarray(volatilities,dtype=float)
    if len(x)!=60 or not np.isfinite(x).all() or not np.isfinite(v).all() or (v<=0).any():
        raise ValueError('Need sixty valid historical returns and positive volatilities')
    cov=LedoitWolf().fit(x).covariance_
    scale=np.sqrt(np.maximum(np.diag(cov),1e-16))
    correlation=cov/np.outer(scale,scale)
    np.fill_diagonal(correlation,1.)
    return correlation*np.outer(v,v)


def risk_summary(weights,covariance,target):
    w=np.asarray(weights); variance=float(w@covariance@w)
    sigma=float(np.sqrt(max(variance,0)))
    exposure=min(1.,target/max(sigma,1e-12))
    contributions=w*(covariance@w)/max(variance,1e-16)
    return {'volatility':sigma,'equity_fraction':exposure,'cash_fraction':1-exposure,
            'scaled_volatility':sigma*exposure,'risk_contributions':contributions.tolist()}


def rebalance(holdings,cash,target_weights,cost_rate):
    """Self-financing trade to post-cost target weights; cost on dollars traded."""
    nav=float(np.sum(holdings)+cash)
    cost=0.
    for _ in range(50):
        updated=cost_rate*np.abs(target_weights*(nav-cost)-holdings).sum()
        if abs(updated-cost)<1e-13*nav:
            cost=float(updated);break
        cost=float(updated)
    after=nav-cost
    traded=float(np.abs(target_weights*after-holdings).sum())
    return target_weights*after,float((1-target_weights.sum())*after),cost,traded


def simulate(returns,weights,execution_targets,cost_rate,cash_annual=0.):
    """returns include the initial execution date, when the account starts in cash.

    Apply existing holdings' return first, then execute at that day's close.
    New holdings can only earn the following day's return. No forced final sale.
    """
    if not np.isfinite(cash_annual) or cash_annual <= -1:
        raise ValueError('Cash annual return must be finite and greater than -100%')
    w=np.asarray(weights); holdings=np.zeros(len(w));cash=1.;nav=1.
    previous_date=None
    rows=[];trades=[]
    for date,values in returns.iterrows():
        if previous_date is not None:
            cash *= (1+cash_annual)**((date-previous_date).days/365.)
        holdings=holdings*(1+values.to_numpy())
        pre_trade_nav=float(holdings.sum()+cash)
        cost=0.;traded=0.
        if date in execution_targets:
            exposure=execution_targets[date]
            holdings,cash,cost,traded=rebalance(holdings,cash,w*exposure,cost_rate)
            trades.append({'execution_date':str(date.date()),'target_equity':exposure,'cost':cost,'traded':traded})
        current=float(holdings.sum()+cash)
        rows.append({'date':str(date.date()),'nav':current,'return':current/nav-1,
                     'equity_fraction':float(holdings.sum()/current),'cost':cost,'turnover':traded/pre_trade_nav})
        nav=current
        previous_date=date
    frame=pd.DataFrame(rows)
    r=frame['return'].to_numpy(); wealth=np.r_[1.,frame.nav.to_numpy()]
    vol=float(np.std(r,ddof=0)*np.sqrt(252))
    rolling=pd.Series(r).rolling(20).std(ddof=0)*np.sqrt(252)
    return {'metrics':{'total_return':float(wealth[-1]-1),'annualized_return':float(wealth[-1]**(252/len(r))-1),
            'realized_volatility':vol,'sharpe_zero_rate':float(np.mean(r)*252/vol) if vol>1e-12 else None,
            'max_drawdown':float(np.min(wealth/np.maximum.accumulate(wealth)-1)),
            'mean_equity_fraction':float(frame.equity_fraction.mean()),'turnover':float(frame.turnover.sum()),
            'cost_initial_capital':float(frame.cost.sum()),'rebalances':len(trades)},
            'path':frame[['date','nav','equity_fraction']].to_dict('records'),
            'rolling_volatility':[None if pd.isna(v) else float(v) for v in rolling], 'trades':trades}


class PortfolioLab:
    def __init__(self,root,evidence=None):
        self.root=Path(root); self.source=evidence or RiskEvidence(root)
        tickers=sorted(self.source.frames['static_matched'].ticker.unique())
        prices={t:pd.read_csv(self.root/f'data/raw/market/{t}.csv',parse_dates=['date']).set_index('date').adj_close for t in tickers}
        self.prices=pd.DataFrame(prices)
        if self.prices.isna().any().any() or (self.prices<=0).any().any():
            raise ValueError('Price calendars must be complete and aligned')
        self.returns=self.prices.pct_change(fill_method=None)
        self.log_returns=np.log(self.prices).diff()
        self._cache={}

    def catalogue(self):
        return {'models':MODELS,'default_model':'Ridge_sequence_market','presets':PRESETS,
                'tickers':list(self.prices),'protocols':[p for p in self.source.catalogue()['protocols'] if p['id']!='unseen_companies'],
                'default_target':.10,'default_cost_bps':10,'rebalance_days':20}

    def validate(self,body):
        protocol=body.get('protocol','static_matched');model=body.get('model','Ridge_sequence_market')
        if protocol=='unseen_companies' or protocol not in self.source.frames or model not in MODELS:
            raise ValueError('Unknown portfolio protocol or forecast model')
        supplied=body.get('weights')
        if not isinstance(supplied,dict) or not 1<=len(supplied)<=30: raise ValueError('Provide one to thirty holdings')
        if any(t not in self.prices for t in supplied): raise ValueError('Holding is outside the research universe')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in supplied.values()):
            raise ValueError('Weights must be finite, nonnegative numbers')
        if not np.isclose(sum(supplied.values()),1.,atol=1e-6,rtol=0): raise ValueError('Stock weights must sum to 100%')
        weights={t:float(v) for t,v in sorted(supplied.items()) if v>0}
        target=body.get('target',.10);cost=body.get('cost_bps',10)
        if isinstance(target,bool) or not isinstance(target,(int,float)) or not np.isfinite(target) or not .02<=target<=.40:
            raise ValueError('Annualized risk target must be between 2% and 40%')
        if isinstance(cost,bool) or not isinstance(cost,(int,float)) or not np.isfinite(cost) or not 0<=cost<=100:
            raise ValueError('Trading cost must be between 0 and 100 basis points')
        reveal=body.get('reveal',False)
        if not isinstance(reveal,bool): raise ValueError('Reveal must be true or false')
        frame=self.source.frames[protocol];date=body.get('date',sorted(frame.date.unique())[0])
        if date not in set(frame.date): raise ValueError('Date has no saved predictions')
        return protocol,model,weights,float(target),float(cost),date,reveal

    def decision(self,protocol,model,weights,target,date):
        symbols=list(weights);w=np.array(list(weights.values()))
        frame=self.source.frames[protocol]
        day=frame.loc[frame.date.eq(date)].set_index('ticker').loc[symbols]
        history=self.log_returns.loc[:date,symbols].tail(60)
        historical=covariance_from_history(history,day.rv20.to_numpy())
        predicted=covariance_from_history(history,day[model].to_numpy())
        a=risk_summary(w,historical,target);b=risk_summary(w,predicted,target)
        pos=self.prices.index.get_loc(pd.Timestamp(date))
        holdings=[{'ticker':t,'weight':weights[t],'sector':str(day.loc[t,'sector']),
                   'forecast_volatility':float(day.loc[t,model]),'risk_contribution':b['risk_contributions'][i]} for i,t in enumerate(symbols)]
        return {'date':date,'execution_date':str(self.prices.index[pos+1].date()),'historical':a,'forecast':b,'holdings':holdings}

    def backtest(self,protocol,model,weights,target,cost):
        key=(protocol,model,tuple(weights.items()),target,cost)
        if key in self._cache:return self._cache[key]
        frame=self.source.frames[protocol];dates=sorted(frame.date.unique())
        schedule=dates[::20];historical={};predicted={};signals=[]
        for date in schedule:
            d=self.decision(protocol,model,weights,target,date)
            execution=pd.Timestamp(d['execution_date'])
            historical[execution]=d['historical']['equity_fraction'];predicted[execution]=d['forecast']['equity_fraction']
            signals.append({'signal_date':date,'execution_date':d['execution_date'],'historical_equity':historical[execution],'forecast_equity':predicted[execution]})
        start=min(historical);end=pd.Timestamp(frame.target_end.max())
        returns=self.returns.loc[start:end,list(weights)]
        strategies={}
        targets={'buy_hold':{start:1.},'fixed_half':{d:.5 for d in historical},'historical':historical,'forecast':predicted}
        for name,allocations in targets.items():
            result=simulate(returns,list(weights.values()),allocations,cost/10000)
            vols=np.array([v for v in result['rolling_volatility'] if v is not None])
            result['metrics']['risk_target_mae']=float(np.mean(np.abs(vols-target))) if len(vols) else None
            result['label']=STRATEGIES[name];strategies[name]=result
        output={'start':str(start.date()),'end':str(end.date()),'strategies':strategies,'signals':signals,
                'status':'Exploratory replay on previously inspected periods; not a new holdout or causal proof.'}
        if len(self._cache)>=32:self._cache.pop(next(iter(self._cache)))
        self._cache[key]=output
        return output

    def evaluate(self,body):
        protocol,model,weights,target,cost,date,reveal=self.validate(body)
        decision=self.decision(protocol,model,weights,target,date)
        assumptions={'target_volatility':target,'cost_bps_per_traded_dollar':cost,'cash_return':0.,'rebalance_days':20,
          'execution':'After-close signal; execute next trading close. New weights earn returns from the following session.',
          'constraints':'Long-only; no leverage; same relative stock weights across the two risk rules. Residual capital in cash.',
          'forecast_alignment':'Saved forecasts target the next twenty close-to-close returns; the execution delay shifts the tradable holding period. This is an implementation-lag experiment, not an exact portfolio-target forecast test.',
          'limits':'Fixed survivor sample; historical revised adjusted prices; constant proportional cost; no taxes or market impact; zero cash yield; no guarantee of target risk. Lower exposure can explain lower risk.'}
        facts=[{'id':'E1','label':'Portfolio experiment','data':{'protocol':protocol,'model':model,'date':date,'weights':weights}},
               {'id':'E2','label':'Risk estimates and allocation rules','data':decision},
               {'id':'E3','label':'Execution and limitations','data':assumptions}]
        result={'decision':decision,'assumptions':assumptions,'revealed':reveal}
        if reveal:
            result['backtest']=self.backtest(protocol,model,weights,target,cost)
            facts.append({'id':'E7','label':'Revealed full-period outcomes (hindsight)','data':{
                'start':result['backtest']['start'],'end':result['backtest']['end'],
                'strategies':{k:v['metrics'] for k,v in result['backtest']['strategies'].items()},'status':result['backtest']['status']}})
        result['context']={'evidence':facts,'provenance':{'methodology':'docs/PORTFOLIO_PROTOCOL.md','forecasts':f'results/robustness/v2/{protocol}/predictions.csv','prices':'Local adjusted daily price snapshots','context_policy':'No future outcomes or backtest metrics unless E7 is explicitly revealed.'}}
        return result
