"""Actual expanded training and strict held-company evaluation, pre-2026 only."""
from pathlib import Path
import sys,subprocess
import numpy as np
import pandas as pd
import joblib,torch
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.io import read_json,write_json,sha256,utc_now
from volrisk.dataset import MARKET_FEATURES
from volrisk.models import TrainOnlyScaler,Sequences,train_network,predict_network
from volrisk.robustness import lag_matrix
from volrisk.aligned import context_matrix,variance_calibration
from volrisk.metrics import forecast_metrics,qlike,block_bootstrap

def main():
    dest=ROOT/'results/aligned/expansion-100-v1'
    if dest.exists():raise FileExistsError('Version new training runs')
    path=ROOT/'data/processed/expansion-100-v1/panel.pkl';manifest=read_json(path.parent/'manifest.json');assert sha256(path)==manifest['panel_sha256']
    panel=pd.read_pickle(path);assert panel.date.max()<pd.Timestamp('2026-01-01')
    cfg=read_json(ROOT/'config/study.json');dest.mkdir(parents=True)
    held=panel.role.eq('held_company')
    tr=panel.index[panel.split.eq('train') & ~held].to_numpy()
    va=panel.index[panel.split.eq('validation') & ~held & panel.context_ready.eq(True)].to_numpy()
    te=panel.index[panel.split.eq('test') & panel.context_ready.eq(True)].to_numpy()
    assert set(panel.loc[tr,'ticker']).isdisjoint(panel.loc[te[held.loc[te]],'ticker'])
    for idx in [tr,va,te]:assert np.array_equal(panel.loc[idx,'ticker'],panel.loc[idx-59,'ticker'])
    assert panel.loc[tr,'target_end'].max()<pd.Timestamp('2022-01-01') and panel.loc[va,'target_end'].max()<pd.Timestamp('2024-01-01')
    write_json(dest/'registered_plan.json',{'started_utc':utc_now(),'config':cfg,'scope':'99 retained of 100 frozen candidates; 65 fitting firms, 34 held-company evaluation firms. Pre-2026 only; already inspected calendar periods.','panel_sha256':sha256(path),'source_sha256':sha256(Path(__file__)),'train_companies':sorted(panel.loc[tr,'ticker'].unique()),'validation_companies':sorted(panel.loc[va,'ticker'].unique()),'held_companies':sorted(panel.loc[te[held.loc[te]],'ticker'].unique()),'models':['Persistence','Ridge60','LSTM60','Timestral5M'],'calibration':'Validation labels from non-held firms only','inference':'Timestral 512 rv20, quantile RMS point; retrospective pretrained transfer; not equal information/capacity architecture ablation'})
    scaler=TrainOnlyScaler().fit(panel.loc[tr,MARKET_FEATURES]);market=scaler.transform(panel[MARKET_FEATURES]);funds=np.zeros((len(panel),0),dtype=np.float32)
    y=np.log(panel.target.clip(lower=1e-8).fillna(1)).to_numpy();mean,std=y[tr].mean(),y[tr].std();ys=((y-mean)/std).astype(np.float32)
    indices=np.r_[va,te];base=panel.loc[indices,['date','ticker','sector','role','split','target','target_end','rv20']].copy();base['Persistence']=base.rv20
    xtrain=lag_matrix(market,tr,60).astype(np.float64);xval=lag_matrix(market,va,60).astype(np.float64);best=None;choices=[]
    for alpha in [1.,100.,10000.]:
        model=Ridge(alpha=alpha,solver='lsqr',tol=1e-8).fit(xtrain,y[tr]);loss=float(np.mean((model.predict(xval)-y[va])**2));choices.append({'alpha':alpha,'validation_log_mse':loss})
        if best is None or loss<best[0]:best=(loss,model)
    del xtrain,xval
    ridge=best[1];base['Ridge60']=np.exp(np.clip(ridge.predict(lag_matrix(market,indices,60).astype(np.float64)),-18,5))
    joblib.dump({'ridge':ridge,'scaler':scaler,'mean':mean,'std':std},dest/'fitted.joblib')
    predictions=[];history={}
    for seed in cfg['seeds']:
        model,h=train_network(market,funds,ys,tr,va,cfg,seed,False);history[str(seed)]=h
        torch.save(model.state_dict(),dest/f'LSTM60_{seed}.pt')
        predictions.append(np.exp(np.clip(predict_network(model,Sequences(market,funds,ys,indices,60))*std+mean,-18,5)))
        write_json(dest/'training_history.json',history)
    base['LSTM60']=np.mean(predictions,axis=0)
    np.save(dest/'contexts.npy',context_matrix(panel,indices))
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
    pd.DataFrame(rows).to_csv(ROOT/'results/EXPANSION_100_V1.csv',index=False);pd.DataFrame(company).to_csv(ROOT/'results/EXPANSION_100_COMPANIES_V1.csv',index=False)
    write_json(ROOT/'results/EXPANSION_100_AUDIT_V1.json',{'created_utc':utc_now(),'panel_manifest':manifest,'ridge_selection':choices,'calibration':scales,'train_rows':len(tr),'validation_rows':len(va),'evaluation_rows':len(te),'paired_intervals_vs_calibrated_ridge':intervals,'holdout_2026_accessed':False})
    lines=['# Expanded-company training v1','', '100 candidates were frozen before downloading outcomes; 99 retained after WBD history acquisition failed. No replacement was selected. 65 firms supply training/validation labels; 34 held firms never supply fitting, selection or calibration labels. Historical 2024-2025 development periods were already inspected on the original cohort. This is a new-company evaluation, not a new untouched time period.','', '| Role | Model | Companies | Origins | MAE (pp) | QLIKE |','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['role']} | {r['model']} | {r['companies']} | {r['n']} | {100*r['mae']:.3f} | {r['qlike']:.4f} |")
    lines+=['','Ridge and LSTM share the same 60-session market histories and training-only preprocessing. Three LSTM seeds are averaged; validation chooses epochs and Ridge alpha. Timestral uses 512 rv20 observations and pretrained weights; architecture, context and information sets differ. Its pretraining overlap is unverified. Calibrated variants use labels only from the non-held validation firms.','', 'Scope: a reconstructed reference-date cohort with explicit official December corrections, not full historical membership. Missing WBD data can induce selection bias. Raw and eligible counts are audited, not forced to a resume target. Company histories can start after 2014; no missing prices were fabricated. 2026 remains sealed. All company and conditional date-block diagnostics are retained.']
    (ROOT/'results/EXPANSION_100_V1.md').write_text('\n'.join(lines)+'\n');write_json(dest/'COMPLETED.json',{'completed_utc':utc_now(),'retained_companies':99,'evaluation_rows':len(te)});(dest/'contexts.npy').unlink()
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
