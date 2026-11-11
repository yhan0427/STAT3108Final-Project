"""Frozen exploratory run: execution-aligned baselines and pinned zero-shot TTM."""
from pathlib import Path
import importlib.metadata
import subprocess
import sys
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from volrisk.aligned import align_panel, context_matrix, variance_calibration
from volrisk.dataset import MARKET_FEATURES, assign_splits
from volrisk.models import TrainOnlyScaler
from volrisk.robustness import lag_matrix, validate_inputs
from volrisk.metrics import forecast_metrics, qlike, block_bootstrap
from volrisk.portfolio import PortfolioLab, PRESETS
from volrisk.io import read_json, write_json, sha256, utc_now


def main():
    plan = read_json(ROOT/'config/aligned-v1.json')
    cfg = read_json(ROOT/'config/study.json')
    validate_inputs(ROOT,cfg)
    dest = ROOT/'results/aligned/v1'
    if dest.exists():
        raise FileExistsError('Preserve existing aligned run; use a new version for revisions')
    dest.mkdir(parents=True)
    write_json(dest/'manifest.json',{'started_utc':utc_now(),'plan':plan,
        'panel_sha256':sha256(ROOT/'data/processed/main/panel.pkl'),
        'sources':{str(p.relative_to(ROOT)):sha256(p) for p in list((ROOT/'src/volrisk').glob('*.py'))+[Path(__file__),ROOT/'scripts/predict_ttm.py']},
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn']}})
    panel = align_panel(pd.read_pickle(ROOT/'data/processed/main/panel.pkl'))
    panel['split'] = assign_splits(panel,plan)
    valid = panel.sequence_ready & panel.target.gt(0) & panel.target.notna()
    panel.loc[~valid,'split'] = 'excluded'
    tr = panel.index[panel.split.eq('train')].to_numpy()
    va = panel.index[panel.split.eq('validation') & panel.context_ready.eq(True)].to_numpy()
    te = panel.index[panel.split.eq('test') & panel.context_ready.eq(True)].to_numpy()
    assert panel.loc[tr,'target_end'].max() < pd.Timestamp(plan['validation_start'])
    assert panel.loc[va,'target_end'].max() < pd.Timestamp(plan['test_start'])
    assert panel.loc[te,'target_end'].max() < pd.Timestamp(plan['test_end_exclusive'])
    indices = np.r_[va,te]
    base = panel.loc[indices,['date','ticker','sector','target','target_start','target_end','rv20','split']].copy()
    base['Persistence'] = base.rv20
    loghar_features = ['loghar_day','loghar_week','loghar_month']
    y = np.log(panel.target.clip(lower=1e-8).fillna(1)).to_numpy()
    loghar_models = {}
    for ticker in sorted(panel.ticker.unique()):
        train = tr[panel.loc[tr,'ticker'].eq(ticker)]
        predict = indices[panel.loc[indices,'ticker'].eq(ticker)]
        model = LinearRegression().fit(panel.loc[train,loghar_features].to_numpy(), y[train])
        residual = y[train]-model.predict(panel.loc[train,loghar_features].to_numpy())
        adjustment = float(np.mean(residual**2)/2)
        base.loc[predict,'LogHAR_proxy'] = np.exp(model.predict(panel.loc[predict,loghar_features].to_numpy())+adjustment)
        loghar_models[ticker] = {'model':model,'lognormal_adjustment':adjustment}
    scaler = TrainOnlyScaler().fit(panel.loc[tr,MARKET_FEATURES])
    market = scaler.transform(panel[MARKET_FEATURES]).astype(np.float64)
    xs = {name:lag_matrix(market,idx,60) for name,idx in [('train',tr),('validation',va),('test',te)]}
    candidates = []
    best = None
    for alpha in plan['ridge_alphas']:
        model = Ridge(alpha=alpha,solver='lsqr',tol=1e-8).fit(xs['train'],y[tr])
        score = float(np.mean((model.predict(xs['validation'])-y[va])**2))
        candidates.append({'alpha':alpha,'validation_log_mse':score})
        if best is None or score < best[0]: best = (score,alpha,model)
    base['Ridge_sequence_market'] = np.exp(np.clip(best[2].predict(np.concatenate([xs['validation'],xs['test']])), -18, 5))
    del xs
    joblib.dump({'loghar':loghar_models,'ridge':best[2],'scaler':scaler},dest/'fitted.joblib')
    contexts = context_matrix(panel,indices)
    np.save(dest/'contexts.npy',contexts)
    del contexts
    subprocess.run([str(ROOT/'.venv-foundation/bin/python'),str(ROOT/'scripts/predict_ttm.py'),
        '--input',str(dest/'contexts.npy'),'--output',str(dest/'ttm.npy'),
        '--model-dir',str(ROOT/'data/models/ttm-3442406')], check=True,cwd=ROOT)
    base['TTM_zero_shot'] = np.load(dest/'ttm.npy',allow_pickle=False)
    scales = {}
    for model in plan['models']:
        scale = variance_calibration(base.loc[va,'target'],base.loc[va,model])
        scales[model] = scale
        base[model+'_calibrated'] = base[model]*scale
    base.to_csv(dest/'predictions_with_validation.csv',index=False)
    test = base.loc[te].copy()
    test.to_csv(dest/'predictions.csv',index=False)
    models = plan['models']+[m+'_calibrated' for m in plan['models']]
    company,summary,intervals = [],[],{}
    for period,part in [('2024-2025',test),('2024',test[test.date.dt.year.eq(2024)]),('2025',test[test.date.dt.year.eq(2025)])]:
        for model in models:
            ratios=[]; wins=0
            for ticker,g in part.groupby('ticker'):
                metric=forecast_metrics(g.target,g[model],g.Persistence)
                ratio=metric['qlike']/float(qlike(g.target,g.LogHAR_proxy).mean())
                ratios.append(ratio); wins+=int(ratio<1)
                company.append({'period':period,'ticker':ticker,'model':model,**metric,'qlike_ratio_to_loghar':ratio})
            summary.append({'period':period,'model':model,**forecast_metrics(part.target,part[model],part.Persistence),
                'equal_company_qlike_ratio':float(np.mean(ratios)),'companies_beating_loghar':wins,'companies':len(ratios)})
            difference=pd.DataFrame({'date':part.date,'difference':qlike(part.target,part[model])-qlike(part.target,part.LogHAR_proxy)}).groupby('date').difference.mean()
            intervals[period+'/'+model]={str(b):block_bootstrap(difference,b,plan['bootstrap_replicates']) for b in plan['bootstrap_blocks']}
    pd.DataFrame(summary).to_csv(ROOT/'results/ALIGNED_V1.csv',index=False)
    pd.DataFrame(company).to_csv(ROOT/'results/ALIGNED_V1_COMPANIES.csv',index=False)
    write_json(ROOT/'results/ALIGNED_V1_AUDIT.json',{'created_utc':utc_now(),'status':plan['status'],'plan_sha256':sha256(ROOT/'config/aligned-v1.json'),
        'split_counts':{'train':len(tr),'validation':len(va),'test':len(te)},
        'last_train_target_end':str(panel.loc[tr,'target_end'].max()),'last_validation_target_end':str(panel.loc[va,'target_end'].max()),
        'target_definition':'sqrt(252) * population std(log returns t+2 through t+21)',
        'old_vs_aligned_target_mean_abs_difference':float((panel.loc[te,'target']-panel.loc[te,'original_target']).abs().mean()),
        'ridge_candidates':candidates,'selected_alpha':best[1],'ridge_solver':'float64 lsqr, tol 1e-8',
        'validation_only_volatility_scales':scales,'ttm':read_json(str(dest/'ttm.npy')+'.json'),
        'paired_qlike_intervals_vs_raw_loghar':intervals})
    # End at the final scheduled signal's t+21: every holding cycle has 20 returns.
    dates=sorted(test.date.unique());last_signal=dates[::20][-1]
    portfolio_frame=test.loc[test.date.le(last_signal)].copy()
    portfolio_frame['date']=portfolio_frame.date.dt.strftime('%Y-%m-%d')
    portfolio_frame['target_end']=portfolio_frame.target_end.dt.strftime('%Y-%m-%d')
    lab=PortfolioLab(ROOT,SimpleNamespace(frames={'static_matched':portfolio_frame}))
    records=[]
    for preset in plan['application']['presets']:
        for model in models:
            for cost in plan['application']['cost_bps']:
                result=lab.backtest('static_matched',model,PRESETS[preset],plan['application']['target'],cost)
                for strategy,output in result['strategies'].items():
                    records.append({'preset':preset,'model':model,'cost_bps':cost,'strategy':strategy,'start':result['start'],'end':result['end'],**output['metrics']})
    pd.DataFrame(records).to_csv(ROOT/'results/ALIGNED_PORTFOLIO_V1.csv',index=False)
    lines=['# Execution-aligned method transfer v1','',plan['status'],'',
        'Target: 20 daily log returns after next-close execution (t+2 through t+21). All models share evaluation origins. TTM is a pinned pretrained retrospective comparator; historical availability and pretraining disjointness are NOT established. LogHAR_proxy uses trailing daily-return volatility, not intraday realized variance. This is not an exact reproduction of the reference paper.','',
        'Raw and validation-calibrated models are retained. Calibration is a single volatility multiplier estimated exclusively on 2022–2023 observations with fully observed targets. Ridge alpha uses that same validation split; test observations do not fit either step.','',
        '## Forecast accuracy','', '| Period | Model | MAE (pp) | QLIKE | Mean company QLIKE ratio vs raw LogHAR | Companies better |','|---|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['period']} | {row['model']} | {row['mae']*100:.3f} | {row['qlike']:.4f} | {row['equal_company_qlike_ratio']:.3f} | {row['companies_beating_loghar']}/{row['companies']} |")
    lines+=['','## Portfolio application, 10 bps','',
        'Same two preset baskets, 10% target, next-close execution, 20-session rebalance and zero cash yield. Full cycles only; final positions are marked to market. All costs (0/10/25 bps) are in ALIGNED_PORTFOLIO_V1.csv. Forecast target alignment does not remove uncertainty in historical correlation estimates or the difference between log-return forecast volatility and simple net strategy returns.','',
        '| Basket | Forecast model | Forecast risk MAE (pp) | Historical risk MAE (pp) | Difference (pp) | Net-return difference (pp) | Equity-exposure difference (pp) |','|---|---|---:|---:|---:|---:|---:|']
    table=pd.DataFrame(records)
    for (preset,model),group in table[table.cost_bps.eq(10)].groupby(['preset','model'],sort=False):
        f=group[group.strategy.eq('forecast')].iloc[0];h=group[group.strategy.eq('historical')].iloc[0]
        lines.append(f"| {preset} | {model} | {100*f.risk_target_mae:.3f} | {100*h.risk_target_mae:.3f} | {100*(f.risk_target_mae-h.risk_target_mae):+.3f} | {100*(f.total_return-h.total_return):+.2f} | {100*(f.mean_equity_fraction-h.mean_equity_fraction):+.2f} |")
    lines+=['','## Interpretation boundary','',
        'These comparisons distinguish forecast accuracy from policy outcomes. A small average gain is not a population claim. Company outcomes share market shocks; rolling windows overlap. Paired 20/60-date-block intervals are conditional diagnostics, not a correction for researcher choice or pretraining contamination. The 2024 and 2025 slices partition an already inspected sample. Preset baskets and models were selected after prior project results. No new holdout or alpha claim is justified.','',
        'Method and reference details: docs/ALIGNED_METHOD_TRANSFER.md. Manifest, fitted models and row-level forecasts remain local under results/aligned/v1; aggregate results and audit are versioned. The current demo intentionally remains on its labeled V2/V1 evidence until this research branch is reviewed.','']
    (ROOT/'results/ALIGNED_V1.md').write_text('\n'.join(lines))
    write_json(dest/'COMPLETED.json',{'completed_utc':utc_now(),'forecast_models':len(models),'test_rows':len(test),'portfolio_records':len(records)})
    # Rebuildable inference input; retain its checksum, predictions and provenance.
    (dest/'contexts.npy').unlink()
    print(pd.DataFrame(summary).query("period == '2024-2025'").to_string(index=False),flush=True)

if __name__=='__main__': main()
