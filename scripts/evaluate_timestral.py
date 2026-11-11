"""Development-only modern Transformer transfer; preserve all prior runs."""
from pathlib import Path
import sys,subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.aligned import align_panel,context_matrix,variance_calibration
from volrisk.io import read_json,write_json,sha256,utc_now
from volrisk.metrics import forecast_metrics,qlike,block_bootstrap
from volrisk.temporal import freeze_validation_exposure,evaluate_policies
from volrisk.portfolio import PRESETS

def main():
    dest=ROOT/'results/aligned/timestral-v1'
    if dest.exists():raise FileExistsError('Version new experiments; preserve this run')
    dest.mkdir(parents=True)
    plan=read_json(ROOT/'config/timestral-v1.json')
    write_json(dest/'manifest.json',{'started_utc':utc_now(),'plan':plan,'source_sha256':{p.name:sha256(p) for p in [Path(__file__),ROOT/'scripts/predict_timestral.py']},'panel_sha256':sha256(ROOT/'data/processed/main/panel.pkl'),'weights_sha256':sha256(ROOT/'data/models/timestral-5m-bdd20cf/model.safetensors')})
    base=pd.read_csv(ROOT/'results/aligned/v1/predictions_with_validation.csv',parse_dates=['date','target_end'])
    if base.target_end.max()>=pd.Timestamp('2026-01-01'):raise ValueError('2026 sealed')
    panel=align_panel(pd.read_pickle(ROOT/'data/processed/main/panel.pkl'))
    lookup=pd.Series(panel.index,index=pd.MultiIndex.from_frame(panel[['ticker','date']]))
    indices=lookup.loc[pd.MultiIndex.from_frame(base[['ticker','date']])].to_numpy()
    np.testing.assert_allclose(panel.loc[indices,'target'],base.target,rtol=1e-10)
    np.save(dest/'contexts.npy',context_matrix(panel,indices))
    subprocess.run([str(ROOT/'.venv-foundation/bin/python'),'-B',str(ROOT/'scripts/predict_timestral.py'),'--input',str(dest/'contexts.npy'),'--output',str(dest/'forecast.npy')],check=True)
    base['Timestral5M']=np.load(dest/'forecast.npy');validation=base.split.eq('validation')
    scale=variance_calibration(base.loc[validation,'target'],base.loc[validation,'Timestral5M'])
    base['Timestral5M_calibrated']=base.Timestral5M*scale
    base.to_csv(dest/'predictions.csv',index=False)
    test=base[~validation];models=['LogHAR_proxy','LogHAR_proxy_calibrated','Ridge_sequence_market_calibrated','TTM_zero_shot_calibrated','Timestral5M','Timestral5M_calibrated'];rows=[];intervals={}
    for period,g in [('2024-2025',test),('2024',test[test.date.dt.year.eq(2024)]),('2025',test[test.date.dt.year.eq(2025)])]:
        for model in models:
            ratios=[qlike(h.target,h[model]).mean()/qlike(h.target,h.LogHAR_proxy).mean() for _,h in g.groupby('ticker')]
            rows.append({'period':period,'model':model,**forecast_metrics(g.target,g[model],g.Persistence),'equal_company_qlike_ratio':float(np.mean(ratios))})
            diff=pd.DataFrame({'date':g.date,'difference':qlike(g.target,g[model])-qlike(g.target,g.Ridge_sequence_market_calibrated)}).groupby('date').difference.mean()
            intervals[period+'/'+model]={str(b):block_bootstrap(diff,b,1000) for b in [20,60]}
    pd.DataFrame(rows).to_csv(ROOT/'results/TIMESTRAL_V1.csv',index=False)
    prices=pd.DataFrame({t:pd.read_csv(ROOT/f'data/raw/market/{t}.csv',parse_dates=['date']).set_index('date').adj_close for t in sorted(base.ticker.unique())})
    portfolios=[]
    for name,weights in PRESETS.items():
        for model in models:
            constant=freeze_validation_exposure(prices,base[validation],weights,model)
            out=evaluate_policies(prices,test,weights,model,constant,10)
            for strategy,r in out['strategies'].items():portfolios.append({'preset':name,'model':model,'strategy':strategy,'fixed_validation_exposure':constant,**r['metrics']})
    pd.DataFrame(portfolios).to_csv(ROOT/'results/TIMESTRAL_PORTFOLIO_V1.csv',index=False)
    write_json(ROOT/'results/TIMESTRAL_V1_AUDIT.json',{'created_utc':utc_now(),'calibration_validation_only':scale,'holdout_2026_accessed':False,'worker':read_json(str(dest/'forecast.npy')+'.json'),'paired_qlike_intervals_vs_calibrated_ridge':intervals})
    lines=['# Timestral 5M: exploratory Transformer transfer','',plan['pretraining'],'','Target and origins match aligned-v1. 512 past rv20 observations predict the twenty-first future rv20 point. Quantile RMS uses endpoint-constant numerical integration; it is an approximate variance-aligned summary, not the same point functional as TTM. Calibration uses only 2022–2023. All 2024–2025 findings remain exploratory.','', '| Period | Model | MAE (pp) | QLIKE | Equal-company QLIKE ratio vs raw LogHAR |','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['period']} | {r['model']} | {100*r['mae']:.3f} | {r['qlike']:.4f} | {r['equal_company_qlike_ratio']:.3f} |")
    lines+=['','Portfolio metrics in TIMESTRAL_PORTFOLIO_V1.csv include risk-target error, returns, drawdown, mean equity, turnover and costs. Each model has its own validation-fixed constant-exposure control. They do not necessarily match realized development exposure. Intervals retain whole dates but do not establish multiplicity-adjusted significance.','', 'Next: examine cross-company failures and pretraining provenance before selecting a final protocol; keep 2026 sealed.']
    (ROOT/'results/TIMESTRAL_V1.md').write_text('\n'.join(lines)+'\n')
    write_json(dest/'COMPLETED.json',{'completed_utc':utc_now(),'test_rows':len(test),'portfolio_records':len(portfolios)})
    (dest/'contexts.npy').unlink()
    print(pd.DataFrame(rows).query("period == '2024-2025'").to_string(index=False),flush=True)
if __name__=='__main__':main()
