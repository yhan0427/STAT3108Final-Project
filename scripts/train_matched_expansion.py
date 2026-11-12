"""Isolate training-cohort expansion with common original-firm validation."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch,joblib
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.io import read_json,write_json,sha256,utc_now
from volrisk.dataset import MARKET_FEATURES
from volrisk.models import TrainOnlyScaler,train_network,predict_network,Sequences
from volrisk.robustness import lag_matrix
from volrisk.aligned import variance_calibration
from volrisk.metrics import forecast_metrics,qlike,block_bootstrap

def main():
    dest=ROOT/'results/aligned/matched-expansion-v1'
    if dest.exists():raise FileExistsError('Preserve prior matched run')
    panel_path=ROOT/'data/processed/expansion-100-v1/panel.pkl';p=pd.read_pickle(panel_path);cfg=read_json(ROOT/'config/study.json')
    assert p.date.max()<pd.Timestamp('2026-01-01')
    va=p.index[p.split.eq('validation') & p.role.eq('original_training') & p.context_ready.eq(True)].to_numpy()
    te=p.index[p.split.eq('test') & p.role.eq('held_company') & p.context_ready.eq(True)].to_numpy()
    dest.mkdir(parents=True)
    write_json(dest/'registered_plan.json',{'started_utc':utc_now(),'question':'Does adding training firms help when evaluation and validation firms are held fixed?','scope':'Exploratory follow-up after viewing first expansion results; no 2026 access','training_cohorts':['original30','expanded64'],'validation':'Identical original 30 companies, 2022-2023 only','evaluation':'Identical 34 held companies, eligible 2024-2025 origins','seeds':cfg['seeds'],'config':cfg,'panel_sha256':sha256(panel_path),'source_sha256':sha256(Path(__file__))})
    y=np.log(p.target.clip(lower=1e-8).fillna(1)).to_numpy();metrics=[];outputs={};selections={}
    for cohort in ['original30','expanded64']:
        mask=p.role.eq('original_training') if cohort=='original30' else p.role.ne('held_company')
        tr=p.index[p.split.eq('train') & mask].to_numpy()
        assert p.loc[tr,'ticker'].nunique()==(30 if cohort=='original30' else 64)
        assert set(p.loc[tr,'ticker']).isdisjoint(p.loc[te,'ticker'])
        for idx in [tr,va,te]:assert np.array_equal(p.loc[idx,'ticker'],p.loc[idx-59,'ticker'])
        scaler=TrainOnlyScaler().fit(p.loc[tr,MARKET_FEATURES]);market=scaler.transform(p[MARKET_FEATURES]);funds=np.zeros((len(p),0),dtype=np.float32)
        mean,std=y[tr].mean(),y[tr].std();ys=((y-mean)/std).astype(np.float32)
        xtr=lag_matrix(market,tr,60).astype(np.float64);xva=lag_matrix(market,va,60).astype(np.float64);best=None;choices=[]
        for alpha in [1.,100.,10000.]:
            model=Ridge(alpha=alpha,solver='lsqr',tol=1e-8).fit(xtr,y[tr]);loss=float(np.mean((model.predict(xva)-y[va])**2));choices.append({'alpha':alpha,'validation_log_mse':loss})
            if best is None or loss<best[0]:best=(loss,model)
        ridge=best[1];del xtr,xva
        indices=np.r_[va,te];frame=p.loc[indices,['date','ticker','target','rv20']].copy()
        frame['Ridge60']=np.exp(np.clip(ridge.predict(lag_matrix(market,indices,60).astype(np.float64)),-18,5))
        neural=[];histories={}
        for seed in cfg['seeds']:
            net,h=train_network(market,funds,ys,tr,va,cfg,seed,False);histories[str(seed)]=h
            torch.save(net.state_dict(),dest/f'{cohort}_{seed}.pt')
            neural.append(np.exp(np.clip(predict_network(net,Sequences(market,funds,ys,indices,60))*std+mean,-18,5)))
            write_json(dest/f'{cohort}_history.json',histories)
        frame['LSTM60']=np.mean(neural,axis=0);scales={}
        for m in ['Ridge60','LSTM60']:
            scales[m]=variance_calibration(frame.loc[va,'target'],frame.loc[va,m]);frame[m+'_calibrated']=frame[m]*scales[m]
        frame.to_csv(dest/f'{cohort}_predictions.csv',index=False)
        joblib.dump({'scaler':scaler,'ridge':ridge,'target_mean':mean,'target_std':std},dest/f'{cohort}_fitted.joblib')
        selections[cohort]={'train_firms':p.loc[tr,'ticker'].nunique(),'train_rows':len(tr),'validation_rows':len(va),'held_rows':len(te),'ridge_choices':choices,'calibration':scales}
        out=frame.loc[te].copy();outputs[cohort]=out
        for m in ['Ridge60','LSTM60','Ridge60_calibrated','LSTM60_calibrated']:
            metrics.append({'cohort':cohort,'model':m,**forecast_metrics(out.target,out[m],out.rv20)})
    a,b=outputs['expanded64'],outputs['original30'];assert a[['date','ticker']].equals(b[['date','ticker']]);intervals={}
    for m in ['Ridge60','LSTM60','Ridge60_calibrated','LSTM60_calibrated']:
        diff=pd.DataFrame({'date':a.date,'difference':qlike(a.target,a[m])-qlike(b.target,b[m])}).groupby('date').difference.mean()
        intervals[m]={str(n):block_bootstrap(diff,n,1000) for n in [20,60]}
    pd.DataFrame(metrics).to_csv(ROOT/'results/MATCHED_EXPANSION_V1.csv',index=False)
    write_json(ROOT/'results/MATCHED_EXPANSION_V1_AUDIT.json',{'created_utc':utc_now(),'selections':selections,'paired_expanded_minus_original':intervals,'holdout_2026_accessed':False})
    lines=['# Matched training-cohort control','', 'Exploratory follow-up: same held 34 firms, original-30 validation firms, target, feature history, seed set and training budget. Only the fitting cohort and its training-fitted transformations change: 30 versus 64 companies. Alpha and early stopping are selected independently by the same validation rule; this estimates the effect of the expanded training procedure, not identical fitted parameters. 2026 remains sealed.','', '| Training cohort | Model | MAE (pp) | QLIKE |','|---|---|---:|---:|']
    for r in metrics:lines.append(f"| {r['cohort']} | {r['model']} | {100*r['mae']:.3f} | {r['qlike']:.4f} |")
    lines+=['','Negative expanded-minus-original QLIKE differences favor expansion. Conditional date-block intervals are in the audit. Do not infer a population effect from this selected cohort or interpret the shared calendar period as a new temporal test.']
    (ROOT/'results/MATCHED_EXPANSION_V1.md').write_text('\n'.join(lines)+'\n');write_json(dest/'COMPLETED.json',{'completed_utc':utc_now(),'held_companies':34,'held_rows':len(te)});print(pd.DataFrame(metrics).to_string(index=False),flush=True)
if __name__=='__main__':main()
