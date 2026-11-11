"""Development-only policy controls. Never opens the sealed 2026 holdout."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from volrisk.temporal import freeze_validation_exposure,evaluate_policies
from volrisk.portfolio import PRESETS
from volrisk.io import read_json,write_json,sha256,utc_now
from volrisk.metrics import block_bootstrap


def main():
    dest=ROOT/'results/temporal/development-v1'
    if dest.exists(): raise FileExistsError('Preserve completed development experiment')
    plan=read_json(ROOT/'config/aligned-v1.json')
    models=plan['models']+[m+'_calibrated' for m in plan['models']]
    source=ROOT/'results/aligned/v1/predictions_with_validation.csv'
    frame=pd.read_csv(source,parse_dates=['date','target_end'])
    if frame.date.max()>=pd.Timestamp('2026-01-01') or frame.target_end.max()>=pd.Timestamp('2026-01-01'):
        raise ValueError('2026 holdout is sealed')
    symbols=sorted(frame.ticker.unique())
    paths={t:ROOT/f'data/raw/market/{t}.csv' for t in symbols}
    prices=pd.DataFrame({t:pd.read_csv(p,parse_dates=['date']).set_index('date').adj_close for t,p in paths.items()})
    if prices.index.max()>=pd.Timestamp('2026-01-01'): raise ValueError('Use only the preserved pre-2026 price snapshot')
    if prices.isna().any().any(): raise ValueError('Incomplete price calendar')
    validation=frame[frame.split.eq('validation')].copy()
    development=frame[frame.split.eq('test')].copy()
    dest.mkdir(parents=True)
    constants={p:{m:freeze_validation_exposure(prices,validation,PRESETS[p],m) for m in models} for p in PRESETS}
    write_json(dest/'registered_plan.json',{'created_utc':utc_now(),'status':'Exploratory development on previously inspected 2024-2025; 2026 sealed.',
        'primary_cost_bps':10,'primary_cash_annual':0,'cost_bps':[0,10,25],'cash_annual':[0,.04],
        'cash_note':'4% is hypothetical constant annual yield, not observed Treasury data.',
        'models':models,'fixed_validation_exposure':constants,'predictions_sha256':sha256(source),
        'prices_sha256':{t:sha256(p) for t,p in paths.items()},
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/volrisk/temporal.py',ROOT/'src/volrisk/portfolio.py']}})
    records=[];comparisons=[]
    for preset,weights in PRESETS.items():
        for model in models:
            fixed=constants[preset][model]
            for cost in [0,10,25]:
                for cash in [0.,.04]:
                    result=evaluate_policies(prices,development,weights,model,fixed,cost,cash)
                    case={'preset':preset,'model':model,'cost_bps':cost,'cash_annual':cash,'fixed_validation_exposure':fixed}
                    for strategy,value in result['strategies'].items(): records.append({**case,'strategy':strategy,**value['metrics']})
                    if cost==10 and cash==0:
                        a=result['strategies']['forecast']
                        for benchmark in ['fixed_validation','historical']:
                            b=result['strategies'][benchmark]
                            av=np.asarray(a['rolling_volatility'],dtype=float);bv=np.asarray(b['rolling_volatility'],dtype=float)
                            difference=np.abs(av-.1)-np.abs(bv-.1)
                            comparisons.append({**case,'benchmark':benchmark,
                                'risk_error_difference':a['metrics']['risk_target_mae']-b['metrics']['risk_target_mae'],
                                'return_difference':a['metrics']['total_return']-b['metrics']['total_return'],
                                'equity_difference':a['metrics']['mean_equity_fraction']-b['metrics']['mean_equity_fraction'],
                                'intervals':{str(n):block_bootstrap(difference,n,1000) for n in [20,60]}})
            print('Completed',preset,model,flush=True)
    pd.DataFrame(records).to_csv(ROOT/'results/EXPOSURE_CONTROLS_V1.csv',index=False)
    write_json(ROOT/'results/EXPOSURE_CONTROLS_V1_AUDIT.json',{'created_utc':utc_now(),'fixed_validation_exposure':constants,'comparisons':comparisons,
        'rows':len(records),'holdout_2026_accessed':False,'development_period':'2024-2025','validation_period':'2022-2023',
        'manifest_sha256':sha256(dest/'registered_plan.json')})
    lines=['# Fixed-exposure controls: development experiment','',
        'Question: does dynamic exposure improve risk-target tracking relative to holding a constant equity fraction chosen using validation-period signals only? This is exploratory development on previously inspected 2024–2025 data. The 2026 holdout remains sealed.','',
        'Constants are the mean scheduled target equity fractions from 2022–2023 for each model/basket, frozen before computing development outcomes. They match validation planned exposure, NOT realized development risk or average exposure. No development outcome selects the constants.','',
        '96 scenarios / 384 strategy records: two baskets, eight forecast variants, three cost assumptions, two cash-yield assumptions. Primary: 10 bps and 0% cash yield. The 4% cash-yield sensitivity is hypothetical and accrues ACT/365 on cash held between closes.','',
        '| Basket | Forecast | Fixed validation equity | Risk-error difference (pp) | 20-day block CI (pp) | Net-return difference (pp) | Equity difference (pp) |','|---|---|---:|---:|---|---:|---:|']
    for row in comparisons:
        if row['benchmark']!='fixed_validation':continue
        ci=row['intervals']['20']
        lines.append(f"| {row['preset']} | {row['model']} | {row['fixed_validation_exposure']:.1%} | {100*row['risk_error_difference']:+.3f} | [{100*ci['ci95_low']:+.3f}, {100*ci['ci95_high']:+.3f}] | {100*row['return_difference']:+.2f} | {100*row['equity_difference']:+.2f} |")
    lines+=['','Negative risk-error differences favor the dynamic forecast rule. Intervals are conditional date-block diagnostics, not multiplicity-adjusted significance claims. Read both 20/60-day intervals in the audit. The CSV includes risk error, net return, drawdown, average equity, turnover and trading costs for all strategies and sensitivities.','',
        'Limits: survivor selection, already inspected periods, overlapping windows, approximate covariance forecasts, and model pretraining overlap. A validation-matched constant does not isolate timing perfectly when realized equity exposure changes in the development period. No result establishes alpha or universal model superiority.','']
    (ROOT/'results/EXPOSURE_CONTROLS_V1.md').write_text('\n'.join(lines))
    write_json(dest/'COMPLETED.json',{'created_utc':utc_now(),'scenarios':96,'records':len(records),'holdout_2026_accessed':False})

if __name__=='__main__':main()
