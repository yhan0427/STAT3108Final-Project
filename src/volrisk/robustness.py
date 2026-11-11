"""Versioned exploratory robustness experiments. Existing v1 runs stay intact."""
from pathlib import Path
import copy
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from .dataset import MARKET_FEATURES, HAR_FEATURES
from .fundamentals import FUND_FEATURES
from .models import TrainOnlyScaler, Sequences, predict_network, train_network
from .metrics import forecast_metrics, qlike, block_bootstrap, ranking_by_date
from .io import read_json, write_json, sha256, utc_now


def partition(panel, protocol, heldout):
    valid = panel.sequence_ready & panel.target.notna() & (panel.target > 0) & (panel.date >= '2015-01-01')
    a,b,c = [pd.Timestamp(protocol[k]) for k in ['train_end','validation_end','evaluation_end']]
    train = valid & (panel.date < a) & (panel.target_end < a)
    val = valid & (panel.date >= a) & (panel.date < b) & (panel.target_end < b)
    test = valid & (panel.date >= b) & (panel.date < c) & (panel.target_end < c)
    if protocol['company_holdout']:
        held = panel.ticker.isin(heldout)
        train &= ~held
        val &= ~held
        test &= held
    return tuple(panel.index[m].to_numpy() for m in [train,val,test])


def lag_matrix(market, indices, lookback):
    return np.stack([market[i-lookback+1:i+1].reshape(-1) for i in indices])


def missingness_control(funds):
    control=funds.copy()
    control[:,:funds.shape[1]//2-1]=0
    return control


def validate_inputs(root, config, relative_path='data/processed/main/panel.pkl'):
    path=root/relative_path
    manifest=read_json(path.parent/'manifest.json')
    if manifest['config'] != config:
        raise ValueError('Study configuration differs from the processed panel; rebuild or version the dataset.')
    if sha256(path) != manifest['panel_sha256']:
        raise ValueError('Processed panel checksum mismatch')


def run(root, plan_file='config/robustness-v2.json'):
    plan = read_json(root/plan_file)
    cfg = read_json(root/'config/study.json')
    panel_path=plan.get('panel_path','data/processed/main/panel.pkl')
    fund_features=plan.get('fundamental_features',FUND_FEATURES)
    validate_inputs(root,cfg,panel_path)
    output = root/'results/robustness'/plan.get('output_run','v2')
    if output.exists():
        raise FileExistsError('Robustness v2 already exists; preserve it and version any revised protocol.')
    output.mkdir(parents=True)
    panel = pd.read_pickle(root/panel_path)
    write_json(output/'registered_plan.json', {'started_utc':utc_now(),'plan':plan,'study':cfg,'panel_sha256':sha256(root/panel_path),'source_sha256':{str(p.relative_to(root)):sha256(p) for p in (root/'src/volrisk').glob('*.py')}})
    all_metrics=[]
    for protocol in plan['protocols']:
        name = protocol['name']
        dest=output/name; dest.mkdir()
        tr,va,te=partition(panel,protocol,plan['held_out_companies'])
        for idx in [tr,va,te]:
            assert len(idx) and np.array_equal(panel.loc[idx,'ticker'].to_numpy(),panel.loc[idx-cfg['lookback']+1,'ticker'].to_numpy())
        assert set(tr).isdisjoint(te) and set(va).isdisjoint(te)
        if protocol['company_holdout']:
            assert set(panel.loc[tr,'ticker']).isdisjoint(panel.loc[te,'ticker'])
            assert set(panel.loc[va,'ticker']).isdisjoint(panel.loc[te,'ticker'])
        print(f'PROTOCOL {name}: train={len(tr)} validation={len(va)} evaluation={len(te)}',flush=True)
        ms=TrainOnlyScaler().fit(panel.loc[tr,MARKET_FEATURES])
        fs=TrainOnlyScaler().fit(panel.loc[tr,fund_features])
        ts=TrainOnlyScaler().fit(panel.loc[tr,MARKET_FEATURES+HAR_FEATURES])
        market=ms.transform(panel[MARKET_FEATURES])
        funds=fs.transform(panel[fund_features],missing=True)
        snapshot=ts.transform(panel[MARKET_FEATURES+HAR_FEATURES])
        # Age is a control variable, not a monetary accounting value.
        control=missingness_control(funds)
        y=np.log(panel.target.clip(lower=1e-8).fillna(1)).to_numpy()
        mean,std=y[tr].mean(),y[tr].std()
        standardized=((y-mean)/std).astype(np.float32)
        base=panel.loc[te,['date','ticker','sector','target','target_end','rv20']].copy()
        base['Persistence']=base.rv20
        selections={}; fitted={}; histories={}; per_seed={}
        # Both point-in-time tabular and all-60-day sequence Ridge are explicit.
        lagged={k:lag_matrix(market,idx,cfg['lookback']) for k,idx in [('train',tr),('validation',va),('evaluation',te)]}
        for features_name in ['snapshot','sequence']:
            x={k:(snapshot[idx] if features_name=='snapshot' else lagged[k]) for k,idx in [('train',tr),('validation',va),('evaluation',te)]}
            for modality,f in [('market',np.zeros((len(panel),0),dtype=np.float32)),('missingness',control),('fundamentals',funds)]:
                label=f'Ridge_{features_name}_{modality}'
                xx={k:np.concatenate([x[k],f[idx]],axis=1) for k,idx in [('train',tr),('validation',va),('evaluation',te)]}
                best=None
                for alpha in plan['ridge_alphas']:
                    model=Ridge(alpha=alpha).fit(xx['train'],y[tr])
                    loss=float(np.mean((model.predict(xx['validation'])-y[va])**2))
                    if best is None or loss<best[0]: best=(loss,alpha,model)
                selections[label]={'alpha':best[1],'validation_log_mse':best[0]}
                fitted[label]=best[2]
                base[label]=np.exp(np.clip(best[2].predict(xx['evaluation']),-18,5))
        for modality,f in [('market',np.zeros((len(panel),0),dtype=np.float32)),('missingness',control),('fundamentals',funds)]:
            label=f'Boosting_{modality}'
            x=np.concatenate([snapshot,f],axis=1)
            model=HistGradientBoostingRegressor(**cfg['boosting'],early_stopping=False,random_state=2026).fit(x[tr],y[tr])
            fitted[label]=model
            base[label]=np.exp(np.clip(model.predict(x[te]),-18,5))
        # Same architecture and three seeds for every neural information set.
        for modality,f in [('market',np.zeros_like(funds)),('missingness',control),('fundamentals',funds)]:
            label=f'LSTM_{modality}'
            preds=[]
            for seed in cfg['seeds']:
                model,history=train_network(market,f,standardized,tr,va,cfg,seed,True)
                torch.save(model.state_dict(),dest/f'{label}_{seed}.pt')
                dataset=Sequences(market,f,standardized,te,cfg['lookback'])
                pred=np.exp(np.clip(predict_network(model,dataset)*std+mean,-18,5))
                per_seed[f'{label}_{seed}']=forecast_metrics(base.target,pred,base.Persistence)
                preds.append(pred);histories[f'{label}_{seed}']=history
            base[label]=np.mean(preds,axis=0)
        models=[c for c in base if c in ['Persistence'] or c.startswith(('Ridge_','Boosting_','LSTM_'))]
        base.to_csv(dest/'predictions.csv',index=False)
        for label in models:
            metrics=forecast_metrics(base.target,base[label],base.Persistence)
            ranking=ranking_by_date(base,label,5)
            all_metrics.append({'protocol':name,'model':label,**metrics,'top5_overlap':float(ranking.top_k_overlap.mean())})
        pairs=[('LSTM_fundamentals','LSTM_market'),('LSTM_fundamentals','LSTM_missingness'),('LSTM_market','Ridge_sequence_market'),('Boosting_fundamentals','Boosting_missingness')]
        intervals={}
        for model,benchmark in pairs:
            daily=pd.DataFrame({'date':base.date,'difference':qlike(base.target,base[model])-qlike(base.target,base[benchmark])}).groupby('date').difference.mean()
            intervals[f'{model} minus {benchmark}']={str(b):block_bootstrap(daily,b,1000) for b in [20,60]}
        joblib.dump({'market_scaler':ms,'fund_scaler':fs,'tab_scaler':ts,'target_mean':mean,'target_std':std,'tabular_models':fitted},dest/'fitted.joblib')
        write_json(dest/'diagnostics.json',{'protocol':protocol,'train_companies':sorted(panel.loc[tr,'ticker'].unique()),'validation_companies':sorted(panel.loc[va,'ticker'].unique()),'evaluation_companies':sorted(panel.loc[te,'ticker'].unique()),'ridge_selection':selections,'seed_evaluation':per_seed,'paired_qlike_intervals':intervals,'history':histories})
        pd.DataFrame(all_metrics).to_csv(output/'metrics.csv',index=False)
        print(pd.DataFrame(all_metrics).query('protocol == @name')[['model','mae','qlike','top5_overlap']].to_string(index=False),flush=True)
    write_json(output/'COMPLETED.json',{'completed_utc':utc_now(),'protocols':len(plan['protocols'])})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='config/robustness-v2.json')
    run(Path.cwd(),parser.parse_args().plan)
