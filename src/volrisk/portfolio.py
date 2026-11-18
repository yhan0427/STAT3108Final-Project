"""Historical portfolio experiments; signals at close, execution next close.

No future returns are consulted by the allocation rule. Earlier forecasting
experiments are inputs, not evidence of an untouched strategy holdout.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from .copilot import RiskEvidence
from .io import read_json, sha256

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
        self.frames=dict(self.source.frames)
        self.model_sets={p:MODELS for p in self.frames}
        self.validation=None
        self.multimodal_source=None
        self.timestral_source=None
        self.filing_metadata=None
        current=self.root/'results/financial/multimodal-age-v1/predictions.csv'
        audit=self.root/'results/MULTIMODAL_AGE_100_AUDIT_V1.json'
        if current.exists() and audit.exists():
            metadata=read_json(audit)
            if sha256(current)!=metadata['predictions_sha256']:
                raise ValueError('Matched research forecasts changed')
            original=pd.read_csv(current)
            if pd.to_datetime(original.target_end).max()>=pd.Timestamp('2026-01-01'):
                raise ValueError('Sealed outcomes cannot enter the replay')
            groups=['market','financial','text','all','financial_control','text_control','financial_text_control','filing_age']
            mapping={f'MM_{b}_{g}':f'{b}_{g}_calibrated' for b in ['Ridge','LSTM'] for g in groups}
            for public,saved in mapping.items(): original[public]=original[saved]
            held=original.role.eq('held_company')
            self.frames['multimodal_matched']=original.loc[held & original.split.eq('test')].copy()
            # Validation target labels are deliberately absent from the decision interface.
            self.validation=original.loc[held & original.split.eq('validation'),['date','ticker','rv20',*mapping]].copy()
            self.model_sets['multimodal_matched']=list(mapping)
            self.multimodal_source=str(current.relative_to(self.root))
            data_dir=self.root/'data/processed/multimodal-100-v1'
            if (data_dir/'manifest.json').exists():
                manifest=read_json(data_dir/'manifest.json')
                if sha256(data_dir/'panel.pkl')!=manifest['files']['panel.pkl']:
                    raise ValueError('Matched filing metadata changed')
                metadata_panel=pd.read_pickle(data_dir/'panel.pkl')
                metadata_panel=metadata_panel.loc[metadata_panel.role.eq('held_company') & metadata_panel.split.eq('test')].copy()
                metadata_panel['date']=metadata_panel.date.dt.strftime('%Y-%m-%d')
                self.filing_metadata=metadata_panel.set_index(['date','ticker'])
            transfer=self.root/'results/financial/matched-timestral-v1/predictions.csv'
            transfer_audit=self.root/'results/MATCHED_TIMESTRAL_AUDIT_V1.json'
            if transfer.exists() and transfer_audit.exists():
                if sha256(transfer)!=read_json(transfer_audit)['predictions_sha256']:
                    raise ValueError('Matched modern-model forecasts changed')
                modern=pd.read_csv(transfer,usecols=['date','ticker','Timestral5M_calibrated'])
                modern=modern.rename(columns={'Timestral5M_calibrated':'MM_Timestral5M'})
                for name in ['test','validation']:
                    frame=self.frames['multimodal_matched'] if name=='test' else self.validation
                    frame=frame.merge(modern,on=['date','ticker'],how='left',validate='one_to_one')
                    if frame.MM_Timestral5M.isna().any():raise ValueError('Missing matched modern-model origin')
                    if name=='test':self.frames['multimodal_matched']=frame
                    else:self.validation=frame
                self.model_sets['multimodal_matched'].append('MM_Timestral5M')
                self.timestral_source=str(transfer.relative_to(self.root))
        tickers=sorted(set().union(*(set(f.ticker) for p,f in self.frames.items() if p!='unseen_companies')))
        prices={t:pd.read_csv(self.root/f'data/raw/market/{t}.csv',parse_dates=['date']).set_index('date').adj_close for t in tickers}
        self.prices=pd.DataFrame(prices)
        if (self.prices<=0).any().any() or self.prices.index.max()>=pd.Timestamp('2026-01-01'):
            raise ValueError('Prices must be positive and end before the sealed period')
        # IPOs can have shorter histories; each selected sixty-return window is checked below.
        self.returns=self.prices.pct_change(fill_method=None)
        self.log_returns=np.log(self.prices).diff()
        self._cache={}

    def _frame(self,protocol):
        return self.frames[protocol] if protocol=='multimodal_matched' else self.source.frames[protocol]

    def catalogue(self):
        protocols=[{**p,'models':MODELS,'default_model':'Ridge_sequence_market','presets':PRESETS,
                    'tickers':sorted(self.frames[p['id']].ticker.unique()),'source_note':'Preliminary thirty-company V2 forecasts; fixed-half equity comparator.'}
                   for p in self.source.catalogue()['protocols'] if p['id']!='unseen_companies']
        if self.multimodal_source:
            frame=self.frames['multimodal_matched']
            baskets=read_json(self.root/'config/multimodal-portfolio-v1.json')['baskets']
            protocols.insert(0,{'id':'multimodal_matched','label':'2024–25 · matched financial + text study',
                'dates':sorted(frame.date.unique()),'companies':int(frame.ticker.nunique()),
                'models':self.model_sets['multimodal_matched'],'default_model':'MM_Ridge_market',
                'tickers':sorted(frame.ticker.unique()),'presets':baskets,
                'source_note':'Matched held-company forecasts · validation calibrated · next-close aligned · validation-fixed equity comparator. Interactive choices remain exploratory.'})
        return {'models':MODELS,'default_model':'Ridge_sequence_market','presets':PRESETS,
                'default_protocol':'multimodal_matched' if self.multimodal_source else 'static_matched',
                'tickers':sorted(self.frames['static_matched'].ticker.unique()),'protocols':protocols,
                'default_target':.10,'default_cost_bps':10,'rebalance_days':20}

    def validate(self,body):
        protocol=body.get('protocol','static_matched');model=body.get('model','Ridge_sequence_market')
        if protocol=='unseen_companies' or protocol not in self.frames or model not in self.model_sets.get(protocol,[]):
            raise ValueError('Unknown portfolio protocol or forecast model')
        supplied=body.get('weights')
        if not isinstance(supplied,dict) or not 1<=len(supplied)<=30: raise ValueError('Provide one to thirty holdings')
        eligible=set(self._frame(protocol).ticker)
        if any(t not in eligible for t in supplied): raise ValueError('Holding is outside the selected research cohort')
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
        frame=self._frame(protocol);date=body.get('date',sorted(frame.date.unique())[0])
        if date not in set(frame.date): raise ValueError('Date has no saved predictions')
        return protocol,model,weights,float(target),float(cost),date,reveal

    def decision(self,protocol,model,weights,target,date):
        symbols=list(weights);w=np.array(list(weights.values()))
        frame=self._frame(protocol)
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
        frame=self._frame(protocol);dates=sorted(frame.date.unique())
        if protocol=='multimodal_matched':
            from .temporal import evaluate_policies,freeze_validation_exposure
            validation=self.validation.loc[self.validation.ticker.isin(weights)].copy()
            counts=validation.groupby('date').ticker.nunique()
            validation=validation.loc[validation.date.isin(counts.index[counts.eq(len(weights))])]
            evaluation=frame.loc[frame.ticker.isin(weights)].copy()
            fixed=freeze_validation_exposure(self.prices,validation,weights,model,target)
            out=evaluate_policies(self.prices,evaluation,weights,model,fixed,cost,0.,target)
            labels={**STRATEGIES,'fixed_validation':'Validation-fixed equity'}
            for name,strategy in out['strategies'].items(): strategy['label']=labels[name]
            out['fixed_validation_exposure']=fixed
            out['status']='Exploratory matched held-company replay. Constant equity uses only earlier validation forecasts/prices; interactive baskets are not independent validation.'
            if len(self._cache)>=32:self._cache.pop(next(iter(self._cache)))
            self._cache[key]=out
            return out
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
        if protocol=='multimodal_matched':
            assumptions['forecast_alignment']='Forecast after close t, execute close t+1; single-stock targets cover t+2 through t+21. Portfolio risk also uses trailing correlations; covariance and net simple-return risk are distinct from marginal log-return targets.'
            assumptions['information_scope']='Fixed100-issuer nonfinancial subset,19 held firms meeting guarded common history; financial/text missingness retained. Frozen FinBERT pooling,training-only PCA,non-held validation calibration. No live prices or news. A model difference does not identify a causal effect of filing content.'
            assumptions['fixed_comparator']='Mean planned equity on earlier held-company validation forecasts and prices; no held target labels used for model fitting or calibration. Not realized test exposure matching.'
            input_groups={
                'market':'Sixty sessions of ten market features only; no financial values,filing metadata or text embeddings.',
                'financial':'Same market history plus seven financial values,filing age and eight missing-value indicators.',
                'text':'Same market history plus frozen text PCs and document age,length,availability and newer-unavailable metadata; no financial values.',
                'all':'Same market history plus financial values/metadata and frozen text PCs/document metadata.',
                'financial_control':'Same market history plus filing age and eight financial missing-value indicators; financial content values masked.',
                'text_control':'Same market history plus document age,length,availability and newer-unavailable metadata; embeddings masked.',
                'financial_text_control':'Same market history plus financial values/metadata and document metadata; embeddings masked.',
                'filing_age':'Same market history plus filing age and its missing-value indicator only.'}
            group=model.split('_',2)[2] if model!='MM_Timestral5M' else None
            assumptions['model_inputs']=input_groups[group] if group else 'Frozen Timestral:512 trailing rv20 observations only; no financial values,text or filing metadata. Quantile-derived RMS point summary.'
            if model=='MM_Timestral5M':
                assumptions['pretraining_limit']='Retrospective2026 model on earlier dates; not historically tradable in2024-25. Full pretraining cutoff/overlap unverified; local2026 sealing does not certify pretrained-model unseen status.'
        facts=[{'id':'E1','label':'Portfolio experiment','data':{'protocol':protocol,'model':model,'date':date,'weights':weights}},
               {'id':'E2','label':'Risk estimates and allocation rules','data':decision},
               {'id':'E3','label':'Execution and limitations','data':assumptions}]
        result={'decision':decision,'assumptions':assumptions,'revealed':reveal}
        if protocol=='multimodal_matched' and self.filing_metadata is not None:
            from .fundamentals_v3 import V3_FEATURES
            coverage=[]
            for ticker in weights:
                row=self.filing_metadata.loc[(date,ticker)]
                financial_date=row.financial_filing_date
                if pd.notna(financial_date) and pd.Timestamp(financial_date)>=pd.Timestamp(date):
                    raise ValueError('Same-day/future financial metadata')
                coverage.append({'ticker':ticker,
                    'financial_filing_date':str(pd.Timestamp(financial_date).date()) if pd.notna(financial_date) else None,
                    'financial_age_days':float(row.filing_age_days) if pd.notna(row.filing_age_days) else None,
                    'observed_financial_content_fields':int(row[V3_FEATURES[:-1]].notna().sum()),
                    'text_available':bool(row.text_available),
                    'text_age_days':float(row.text_age_days) if pd.notna(row.text_age_days) else None})
            facts.append({'id':'E4','label':'Point-in-time filing coverage','data':coverage})
        if reveal:
            result['backtest']=self.backtest(protocol,model,weights,target,cost)
            facts.append({'id':'E7','label':'Revealed full-period outcomes (hindsight)','data':{
                'start':result['backtest']['start'],'end':result['backtest']['end'],
                'strategies':{k:v['metrics'] for k,v in result['backtest']['strategies'].items()},'status':result['backtest']['status']}})
        result['context']={'evidence':facts,'provenance':{'methodology':'results/MULTIMODAL_PORTFOLIO_V1.md' if protocol=='multimodal_matched' else 'docs/PORTFOLIO_PROTOCOL.md','forecasts':(self.timestral_source if model=='MM_Timestral5M' else self.multimodal_source) if protocol=='multimodal_matched' else f'results/robustness/v2/{protocol}/predictions.csv','prices':'Local adjusted daily price snapshots','context_policy':'No future outcomes or backtest metrics unless E7 is explicitly revealed.'}}
        return result
