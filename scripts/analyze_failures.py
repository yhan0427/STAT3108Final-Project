"""Exploratory company/time influence diagnostics; never accesses 2026 outcomes."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.metrics import qlike,block_bootstrap
from volrisk.io import write_json,sha256,utc_now


def main():
    source=ROOT/'results/aligned/timestral-v1/predictions.csv'
    dest=ROOT/'results/aligned/failures-v1'
    if dest.exists():raise FileExistsError('Preserve previous diagnostics')
    dest.mkdir(parents=True)
    frame=pd.read_csv(source,parse_dates=['date','target_end'])
    assert frame.target_end.max()<pd.Timestamp('2026-01-01')
    assert not frame.duplicated(['ticker','date']).any()
    validation=frame[frame.split.eq('validation')];test=frame[frame.split.eq('test')].copy()
    prices=pd.read_csv(ROOT/'data/raw/market/SPY.csv',parse_dates=['date']).set_index('date').adj_close
    market_risk=np.log(prices).diff().rolling(20).std(ddof=0)*np.sqrt(252)
    threshold=float(market_risk.loc[validation.date.unique()].quantile(.75))
    test['regime']=np.where(test.date.map(market_risk)>threshold,'high_trailing_market_risk','other_market_risk')
    test['year']=test.date.dt.year;test['quarter']=test.date.dt.to_period('Q').astype(str)
    benchmark='Ridge_sequence_market_calibrated'
    models=['LogHAR_proxy_calibrated','TTM_zero_shot_calibrated','Timestral5M','Timestral5M_calibrated']
    rows=[];influence=[];intervals={};tails=[]
    for model in models:
        data=test.copy();data['loss']=qlike(data.target,data[model]);data['base_loss']=qlike(data.target,data[benchmark]);data['difference']=data.loss-data.base_loss
        data['absolute_error_difference']=(data.target-data[model]).abs()-(data.target-data[benchmark]).abs()
        for dimension in ['ticker','sector','year','quarter','regime']:
            for group,g in data.groupby(dimension,sort=True):
                rows.append({'model':model,'dimension':dimension,'group':str(group),'n':len(g),'companies':g.ticker.nunique(),'dates':g.date.nunique(),
                    'qlike':float(g.loss.mean()),'ridge_qlike':float(g.base_loss.mean()),'difference':float(g.difference.mean()),
                    'mae_difference_pp':float(100*g.absolute_error_difference.mean())})
        daily=data.groupby('date').difference.mean()
        intervals[model]={str(n):block_bootstrap(daily,n,1000) for n in [20,60]}
        for date,value in daily.nlargest(5).items():tails.append({'model':model,'date':str(date.date()),'mean_company_loss_difference':float(value),'selection':'Five worst relative days selected after inspection; explanatory only'})
        company=data.groupby('ticker').difference.mean().sort_values(ascending=False)
        for k in [0,1,3,5]:
            excluded=company.head(k).index.tolist();remaining=data[~data.ticker.isin(excluded)]
            influence.append({'model':model,'excluded_worst_companies':k,'excluded':','.join(excluded),'remaining_mean_difference':float(remaining.difference.mean()),'note':'Post-hoc influence diagnostic, not a new performance estimate'})
    metrics=pd.DataFrame(rows);metrics.to_csv(ROOT/'results/FAILURE_ANALYSIS_V1.csv',index=False)
    pd.DataFrame(influence).to_csv(ROOT/'results/FAILURE_INFLUENCE_V1.csv',index=False)
    pd.DataFrame(tails).to_csv(ROOT/'results/FAILURE_DATES_V1.csv',index=False)
    write_json(ROOT/'results/FAILURE_ANALYSIS_V1_AUDIT.json',{'created_utc':utc_now(),'source_sha256':sha256(source),'script_sha256':sha256(Path(__file__)),
        'benchmark':benchmark,'models':models,'market_risk_threshold_validation_q75':threshold,'threshold_period':'2022-2023 saved validation origins only',
        'intervals':intervals,'holdout_2026_accessed':False,'test_rows':len(test),'companies':test.ticker.nunique(),
        'limitations':'Exploratory subgroup/influence analysis on inspected 2024-2025, no new-company holdout. No multiplicity-adjusted subgroup tests; firms share market shocks. Regime uses lagged observed SPY risk, not future market outcomes.'})
    lines=['# Where the modern forecasts fail','',
        'Question: are forecast differences concentrated in particular companies, periods, or a few extreme observations? All diagnostics use the previously inspected 2024–2025 development sample. 2026 remains sealed. Positive QLIKE differences mean worse than calibrated sequence Ridge.','',
        '## Company breadth and influence','', '| Model | Companies better / 30 | All-company loss difference | Exclude worst 1 | Exclude worst 3 | Exclude worst 5 |','|---|---:|---:|---:|---:|---:|']
    for model in models:
        co=metrics[(metrics.model==model)&metrics.dimension.eq('ticker')]
        values=[r['remaining_mean_difference'] for r in influence if r['model']==model]
        lines.append(f"| {model} | {int(co.difference.lt(0).sum())}/30 | "+' | '.join(f'{v:+.4f}' for v in values)+' |')
    lines+=['','Removing bad companies is a post-hoc sensitivity check, not a permissible way to report a better model score. It identifies concentration; it does not justify excluding those issuers from the next sample.','',
        '## Period and historical market-risk groups','',f'SPY trailing 20-day risk threshold = {100*threshold:.2f}% annualized, fixed at the validation-period 75th percentile. Groups use only information available at each origin.','',
        '| Model | Dimension | Group | N | QLIKE difference | MAE difference (pp) |','|---|---|---|---:|---:|---:|']
    for _,r in metrics[metrics.dimension.isin(['year','regime'])].iterrows():lines.append(f'| {r.model} | {r.dimension} | {r.group} | {r.n} | {r.difference:+.4f} | {r.mae_difference_pp:+.3f} |')
    lines+=['','## Timestral calibrated: company extremes','', '| Company | QLIKE difference | MAE difference (pp) |','|---|---:|---:|']
    co=metrics[(metrics.model=='Timestral5M_calibrated')&metrics.dimension.eq('ticker')].sort_values('difference')
    for _,r in pd.concat([co.head(5),co.tail(5)]).iterrows():lines.append(f'| {r.group} | {r.difference:+.4f} | {r.mae_difference_pp:+.3f} |')
    lines+=['','The full CSV includes every company, sector and quarter, including unfavorable cases. FAILURE_DATES_V1.csv records post-hoc worst days. Date-block intervals are in the audit; subgroup rows are descriptive, not separate significance claims.','',
        'Next decision: use this analysis to define hypotheses for the expanded cohort, not to choose favorable stocks. Verify the dated issuer/security roster and its missing-data ledger before acquiring new-company outcomes. Keep identical origins within comparisons and preserve calibrated simple baselines.','',
        'Limits: survivor-selected thirty-company sample; common market shocks; reused development periods; TTM/Timestral pretraining overlap unverified; quantile-RMS versus point-head differences. These findings cannot establish that complex models are generally ineffective.']
    (ROOT/'results/FAILURE_ANALYSIS_V1.md').write_text('\n'.join(lines)+'\n')
    write_json(dest/'COMPLETED.json',{'created_utc':utc_now(),'subgroup_rows':len(metrics),'holdout_2026_accessed':False})
    print(metrics[(metrics.model=='Timestral5M_calibrated')&metrics.dimension.eq('year')].to_string(index=False))

if __name__=='__main__':main()
