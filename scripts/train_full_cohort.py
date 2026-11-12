"""Actual expanded training and strict held-company evaluation, pre-2026 only."""
from pathlib import Path
import sys,subprocess
import numpy as np
import pandas as pd
import joblib,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.io import read_json,write_json,sha256,utc_now
from volrisk.stream_ridge import StreamRidge
from threadpoolctl import threadpool_limits
from volrisk.dataset import MARKET_FEATURES
from volrisk.models import TrainOnlyScaler,Sequences,train_network,predict_network
from volrisk.robustness import lag_matrix
from volrisk.aligned import context_matrix,variance_calibration
from volrisk.metrics import forecast_metrics,qlike,block_bootstrap

def main():
    dest=ROOT/'results/aligned/full-cohort-v1'
    if dest.exists():raise FileExistsError('Version new training runs')
    path=ROOT/'data/processed/full-cohort-v2/panel.pkl';manifest=read_json(path.parent/'manifest.json');assert sha256(path)==manifest['panel_sha256']
    panel=pd.read_pickle(path);assert panel.date.max()<pd.Timestamp('2026-01-01')
    cfg=read_json(ROOT/'config/study.json');dest.mkdir(parents=True)
    held=panel.role.eq('held_company')
    tr=panel.index[panel.split.eq('train') & ~held].to_numpy()
    va=panel.index[panel.split.eq('validation') & ~held & panel.context_ready.eq(True)].to_numpy()
    te=panel.index[panel.split.eq('test') & held & panel.context_ready.eq(True)].to_numpy()
    assert set(panel.loc[tr,'ticker']).isdisjoint(panel.loc[te[held.loc[te]],'ticker'])
    for idx in [tr,va,te]:assert np.array_equal(panel.loc[idx,'ticker'],panel.loc[idx-59,'ticker'])
    assert panel.loc[tr,'target_end'].max()<pd.Timestamp('2022-01-01') and panel.loc[va,'target_end'].max()<pd.Timestamp('2024-01-01')
    write_json(dest/'registered_plan.json',{'started_utc':utc_now(),'config':cfg,'scope':'Frozen full reconstructed reference cohort; actual role counts from data manifest. New held issuers exclude all pilot issuers. Pre-2026 only; historical calendar periods were inspected on other firms.','panel_sha256':sha256(path),'source_sha256':sha256(Path(__file__)),'train_companies':sorted(panel.loc[tr,'ticker'].unique()),'validation_companies':sorted(panel.loc[va,'ticker'].unique()),'held_companies':sorted(panel.loc[te[held.loc[te]],'ticker'].unique()),'models':['Persistence','Ridge60','LSTM60','Timestral5M'],'calibration':'Validation labels from non-held firms only','inference':'Timestral 512 rv20, quantile RMS point; retrospective pretrained transfer; not equal information/capacity architecture ablation'})
    scaler=TrainOnlyScaler().fit(panel.loc[tr,MARKET_FEATURES]);market=scaler.transform(panel[MARKET_FEATURES]);funds=np.zeros((len(panel),0),dtype=np.float32)
    y=np.log(panel.target.clip(lower=1e-8).fillna(1)).to_numpy();mean,std=y[tr].mean(),y[tr].std();ys=((y-mean)/std).astype(np.float32)
    indices=np.r_[va,te];base=panel.loc[indices,['date','ticker','sector','role','split','target','target_end','rv20']].copy();base['Persistence']=base.rv20
    print('TRAIN COUNTS',len(tr),len(va),len(te),flush=True)
    with threadpool_limits(limits=4):
        statistics=StreamRidge.statistics(market,tr,y,60);best=None;choices=[]
        for alpha in [1.,100.,10000.]:
            model=StreamRidge(alpha).fit_statistics(statistics);loss=float(np.mean((model.predict(market,va)-y[va])**2));choices.append({'alpha':alpha,'validation_log_mse':loss})
            if best is None or loss<best[0]:best=(loss,model)
        ridge=best[1];base['Ridge60']=np.exp(np.clip(ridge.predict(market,indices),-18,5))
    del statistics
    print('RIDGE COMPLETE',choices,flush=True)
    joblib.dump({'ridge':ridge,'scaler':scaler,'mean':mean,'std':std},dest/'fitted.joblib')
    predictions=[];history={}
    for seed in cfg['seeds']:
        model,h=train_network(market,funds,ys,tr,va,cfg,seed,False);history[str(seed)]=h
        torch.save(model.state_dict(),dest/f'LSTM60_{seed}.pt')
        predictions.append(np.exp(np.clip(predict_network(model,Sequences(market,funds,ys,indices,60))*std+mean,-18,5)))
        write_json(dest/'training_history.json',history)
    base['LSTM60']=np.mean(predictions,axis=0)
    contexts=np.lib.format.open_memmap(dest/'contexts.npy',mode='w+',dtype=np.float32,shape=(len(indices),512))
    for start in range(0,len(indices),1024):contexts[start:start+1024]=context_matrix(panel,indices[start:start+1024])
    contexts.flush();del contexts
    subprocess.run([str(ROOT/'.venv-foundation/bin/python'),'-B',str(ROOT/'scripts/predict_timestral.py'),'--input',str(dest/'contexts.npy'),'--output',str(dest/'timestral.npy')],check=True)
    base['Timestral5M']=np.load(dest/'timestral.npy');scales={}
    for name in ['Persistence','Ridge60','LSTM60','Timestral5M']:
        scales[name]=variance_calibration(base.loc[va,'target'],base.loc[va,name]);base[name+'_calibrated']=base[name]*scales[name]
    base.to_csv(dest/'predictions.csv',index=False)
    rows=[];intervals={};company=[]
    for role,g in base.loc[te].groupby('role'):
        for name in ['Persistence','Ridge60','LSTM60','Timestral5M','Persistence_calibrated','Ridge60_calibrated','LSTM60_calibrated','Timestral5M_calibrated']:
            rows.append({'role':role,'model':name,'companies':g.ticker.nunique(),**forecast_metrics(g.target,g[name],g.Persistence)})
            diff=pd.DataFrame({'date':g.date,'d':qlike(g.target,g[name])-qlike(g.target,g.Ridge60_calibrated)}).groupby('date').d.mean()
            intervals[role+'/'+name]={str(n):block_bootstrap(diff,n,1000) for n in [20,60]}
            for ticker,h in g.groupby('ticker'):company.append({'role':role,'ticker':ticker,'model':name,**forecast_metrics(h.target,h[name],h.Persistence)})
    pd.DataFrame(rows).to_csv(ROOT/'results/FULL_TRAINING_V1.csv',index=False);pd.DataFrame(company).to_csv(ROOT/'results/FULL_TRAINING_COMPANIES_V1.csv',index=False)
    write_json(ROOT/'results/FULL_TRAINING_AUDIT_V1.json',{'created_utc':utc_now(),'panel_manifest':manifest,'ridge_selection':choices,'calibration':scales,'train_rows':len(tr),'validation_rows':len(va),'evaluation_rows':len(te),'paired_intervals_vs_calibrated_ridge':intervals,'holdout_2026_accessed':False})
    lines=['# Full-cohort unified model training v1','', 'All 500 reconstructed reference-date issuers were frozen before full acquisition. Coverage/exclusions and symbol-only recovery are retained in FULL_COHORT_AUDIT_V2.json. Only training-company labels fit parameters, hyperparameters and calibration; all previously scored pilot issuers are excluded from the new held-company role. Historical 2024-2025 development periods were already inspected on the original cohort. This is a new-company evaluation, not a new untouched time period.','', '| Role | Model | Companies | Origins | MAE (pp) | QLIKE |','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['role']} | {r['model']} | {r['companies']} | {r['n']} | {100*r['mae']:.3f} | {r['qlike']:.4f} |")
    lines+=['','Ridge and LSTM share the same 60-session market histories and training-only preprocessing. Three LSTM seeds are averaged; validation chooses epochs and Ridge alpha. Timestral uses 512 rv20 observations and pretrained weights; architecture, context and information sets differ. Its pretraining overlap is unverified. Calibrated variants use labels only from the non-held validation firms.','', 'Scope: a reconstructed reference-date cohort with explicit official December corrections, not full historical membership. Unavailable histories and current-vendor coverage can induce selection bias. Streamed float64 Ridge sufficient statistics match the standard regularized intercept solution in regression checks. Timestral input is stored temporarily in float32 to bound memory usage. Raw and eligible counts are audited, not forced to a resume target. Company histories can start after 2014; no missing prices were fabricated. 2026 remains sealed. All company and conditional date-block diagnostics are retained.']
    (ROOT/'results/FULL_TRAINING_V1.md').write_text('\n'.join(lines)+'\n');write_json(dest/'COMPLETED.json',{'completed_utc':utc_now(),'retained_companies':manifest['retained_raw_histories'],'evaluation_rows':len(te)});(dest/'contexts.npy').unlink()
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
