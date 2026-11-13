"""Frozen full-run forecasts to held-company portfolio decisions; no refitting."""
from pathlib import Path
import sys,subprocess,hashlib
import numpy as np
import pandas as pd
import torch,joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.io import read_json,write_json,sha256,utc_now
from volrisk.dataset import MARKET_FEATURES
from volrisk.models import VolatilityLSTM,Sequences,predict_network
from volrisk.aligned import context_matrix
from volrisk.temporal import evaluate_policies,freeze_validation_exposure
from volrisk.metrics import block_bootstrap

def main():
    dest=ROOT/'results/portfolio/full-v1'
    if dest.exists():raise FileExistsError('Preserve prior portfolio run')
    source=ROOT/'data/processed/full-cohort-v2/panel.pkl';p=pd.read_pickle(source)
    assert p.date.max()<pd.Timestamp('2026-01-01')
    roster=pd.read_csv(ROOT/'config/full-cohort-v1.csv')
    # Availability-only eligibility, fixed before portfolio outcomes; no score ranking.
    counts=p[p.role.eq('held_company') & p.context_ready.eq(True)].groupby(['ticker','split']).size().unstack(fill_value=0)
    eligible=set(counts.index[(counts.get('validation',0)>=400)&(counts.get('test',0)>=400)])
    candidates=roster[roster.ticker.isin(eligible)].copy()
    candidates['rank']=candidates.issuer_id.map(lambda v:hashlib.sha256(('Han-portfolio-full-v1/'+v).encode()).hexdigest())
    diversified=candidates.sort_values('rank').groupby('sector',sort=True).head(1).ticker.tolist()
    technology=candidates[candidates.sector.eq('Information Technology')].sort_values('rank').head(5).ticker.tolist()
    assert len(diversified)==11 and len(technology)==5
    baskets={'Held-sector basket':dict.fromkeys(diversified,1/len(diversified)),'Held-technology basket':dict.fromkeys(technology,.2)}
    models=['Persistence','Ridge60','LSTM60','Timestral5M','Persistence_calibrated','Ridge60_calibrated','LSTM60_calibrated','Timestral5M_calibrated']
    dest.mkdir(parents=True)
    plan={'started_utc':utc_now(),'status':'Exploratory application after full forecast scores inspected; basket selection uses only role, sector, history coverage and fixed hash, not forecast outcomes.','baskets':baskets,'models':models,'cost_bps':[0,10,25],'cash_annual':[0,.04],'primary':{'cost_bps':10,'cash_annual':0,'target':.1,'metric':'rolling20 net volatility absolute deviation from target'},'fixed_control':'Each model mean planned equity on common validation-origin basket forecasts; uses past prices/predictions, no held-company labels fit models or calibration.','source_panel_sha256':sha256(source),'fitted_bundle_sha256':sha256(ROOT/'results/aligned/full-cohort-v1/fitted.joblib'),'code_sha256':sha256(Path(__file__)),'holdout_2026_accessed':False}
    write_json(dest/'registered_plan.json',plan);write_json(ROOT/'config/full-portfolio-v1.json',plan)
    symbols=sorted(set(diversified+technology));mask=p.ticker.isin(symbols)&p.context_ready.eq(True)&p.split.isin(['validation','test']);idx=p.index[mask].to_numpy()
    bundle=joblib.load(ROOT/'results/aligned/full-cohort-v1/fitted.joblib');market=bundle['scaler'].transform(p[MARKET_FEATURES]);funds=np.zeros((len(p),0),dtype=np.float32)
    base=p.loc[idx,['date','ticker','split','target_end','rv20']].copy();base['Persistence']=base.rv20
    base['Ridge60']=np.exp(np.clip(bundle['ridge'].predict(market,idx),-18,5))
    neural=[]
    for seed in [7,42,2026]:
        net=VolatilityLSTM(10,0,24);net.load_state_dict(torch.load(ROOT/f'results/aligned/full-cohort-v1/LSTM60_{seed}.pt',map_location='cpu',weights_only=True))
        neural.append(np.exp(np.clip(predict_network(net,Sequences(market,funds,np.zeros(len(p),dtype=np.float32),idx,60))*bundle['std']+bundle['mean'],-18,5)))
    base['LSTM60']=np.mean(neural,axis=0)
    np.save(dest/'contexts.npy',context_matrix(p,idx).astype(np.float32))
    subprocess.run([str(ROOT/'.venv-foundation/bin/python'),'-B',str(ROOT/'scripts/predict_timestral.py'),'--input',str(dest/'contexts.npy'),'--output',str(dest/'timestral.npy')],check=True)
    base['Timestral5M']=np.load(dest/'timestral.npy')
    scales=read_json(ROOT/'results/FULL_TRAINING_AUDIT_V1.json')['calibration']
    for model in ['Persistence','Ridge60','LSTM60','Timestral5M']:base[model+'_calibrated']=base[model]*scales[model]
    base.to_csv(dest/'predictions.csv',index=False)
    prices=pd.DataFrame({t:pd.read_csv(ROOT/f'data/raw/market/{t}.csv',parse_dates=['date']).set_index('date').adj_close for t in symbols})
    rows=[];comparisons=[];constants={};paths=[]
    for basket,weights in baskets.items():
        frame=base[base.ticker.isin(weights)].copy();counts=frame.groupby('date').ticker.nunique();frame=frame[frame.date.isin(counts[counts.eq(len(weights))].index)]
        va=frame[frame.split.eq('validation')];te=frame[frame.split.eq('test')]
        assert len(va)>0 and len(te)>0
        constants[basket]={m:freeze_validation_exposure(prices,va,weights,m) for m in models}
        for model in models:
            for cost in [0,10,25]:
                for cash in [0.,.04]:
                    out=evaluate_policies(prices,te,weights,model,constants[basket][model],cost,cash)
                    case={'basket':basket,'model':model,'cost_bps':cost,'cash_annual':cash,'fixed_validation_exposure':constants[basket][model],'start':out['start'],'end':out['end']}
                    for strategy,value in out['strategies'].items():rows.append({**case,'strategy':strategy,**value['metrics']})
                    if cost==10 and cash==0:
                        a=out['strategies']['forecast']
                        for benchmark in ['fixed_validation','historical']:
                            b=out['strategies'][benchmark];av=np.asarray(a['rolling_volatility'],dtype=float);bv=np.asarray(b['rolling_volatility'],dtype=float)
                            comparisons.append({**case,'benchmark':benchmark,'risk_error_difference':a['metrics']['risk_target_mae']-b['metrics']['risk_target_mae'],'return_difference':a['metrics']['total_return']-b['metrics']['total_return'],'equity_difference':a['metrics']['mean_equity_fraction']-b['metrics']['mean_equity_fraction'],'intervals':{str(n):block_bootstrap(np.abs(av-.1)-np.abs(bv-.1),n,1000) for n in [20,60]}})
                        for strategy,v in out['strategies'].items():
                            paths.extend([{**case,'strategy':strategy,**point} for point in v['path']])
        print('Completed',basket,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/FULL_PORTFOLIO_V1.csv',index=False);pd.DataFrame(paths).to_csv(dest/'paths.csv',index=False)
    write_json(ROOT/'results/FULL_PORTFOLIO_AUDIT_V1.json',{'completed_utc':utc_now(),'plan':plan,'frozen_validation_exposures':constants,'comparisons':comparisons,'strategy_records':len(rows),'holdout_2026_accessed':False,'calibration_source':'Frozen non-held validation calibration from full training; never refitted on basket labels'})
    lines=['# Full-run forecasts to portfolio decisions','',plan['status'],'', 'Baskets are selected from locally held companies, one per sector or five technology firms, using a fixed hash among histories with at least 400 eligible validation and evaluation origins. This availability-conditioned choice introduces coverage selection. Forecast scores were already inspected; this is not a new untouched application test.','', 'Global model parameters and volatility calibration are reused unchanged. Basket validation prices/predictions set a constant stock fraction; basket target labels fit neither model nor calibration. The fixed fraction matches planned validation exposure, not realized evaluation exposure or risk. Zero cash yield is primary; 4% is a hypothetical ACT/365 sensitivity.','', '| Basket | Model | Benchmark | Risk-error difference (pp) | Net-return difference (pp) | Equity difference (pp) |','|---|---|---|---:|---:|---:|']
    for r in comparisons:
        if r['model'].endswith('_calibrated'):lines.append(f"| {r['basket']} | {r['model']} | {r['benchmark']} | {100*r['risk_error_difference']:+.3f} | {100*r['return_difference']:+.2f} | {100*r['equity_difference']:+.2f} |")
    lines+=['','Negative risk-error differences favor the forecast policy. All raw and calibrated variants and cost/cash sensitivities remain in the CSV, including returns, drawdown, average equity, turnover and costs. Both 20/60-date intervals are conditional diagnostics; comparisons are not multiplicity-adjusted.','', 'Timing: signal after close, execute next close, earn subsequent returns; full twenty-session cycles only, entry costs included, no terminal forced sale. Historical correlation, revised prices, zero taxes/market impact, survivor selection, preset selection and pretrained-data overlap remain limitations. Results demonstrate policy tradeoffs, not guaranteed target risk, alpha or personal investment advice. The 2026 holdout remains sealed.']
    (ROOT/'results/FULL_PORTFOLIO_V1.md').write_text('\n'.join(lines)+'\n');(dest/'contexts.npy').unlink();write_json(dest/'COMPLETED.json',{'completed_utc':utc_now(),'strategy_records':len(rows)})
if __name__=='__main__':main()
