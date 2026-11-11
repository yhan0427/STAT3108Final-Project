"""Run the declared portfolio scenario grid; never select a winning test model."""
from pathlib import Path
import numpy as np
import pandas as pd
from volrisk.portfolio import PortfolioLab,PRESETS
from volrisk.metrics import block_bootstrap
from volrisk.io import read_json,write_json,sha256,utc_now

root=Path(__file__).resolve().parents[1]
plan=read_json(root/'config/portfolio-v1.json')
out=root/'results/portfolio/v1'
if out.exists():raise FileExistsError('Portfolio v1 already exists; version new experiments')
out.mkdir(parents=True)
write_json(out/'manifest.json',{'started_utc':utc_now(),'plan':plan,
 'source_sha256':{str(p.relative_to(root)):sha256(p) for p in (root/'src/volrisk').glob('*.py')},
 'price_sha256':{str(p.relative_to(root)):sha256(p) for p in (root/'data/raw/market').glob('*.csv')},
 'forecast_sha256':{p:sha256(root/f'results/robustness/v2/{p}/predictions.csv') for p in plan['protocols']}})
lab=PortfolioLab(root);rows=[];comparisons=[]
for preset in plan['presets']:
 for protocol in plan['protocols']:
  for model in plan['models']:
   for cost in plan['cost_bps']:
    r=lab.backtest(protocol,model,PRESETS[preset],plan['target'],cost)
    case=dict(preset=preset,protocol=protocol,model=model,cost_bps=cost)
    for strategy,value in r['strategies'].items():rows.append({**case,'strategy':strategy,**value['metrics']})
    if cost==10:
     a=r['strategies']['forecast'];b=r['strategies']['historical']
     av=np.array([np.nan if v is None else v for v in a['rolling_volatility']]);bv=np.array([np.nan if v is None else v for v in b['rolling_volatility']])
     differences=np.abs(av-plan['target'])-np.abs(bv-plan['target'])
     comparisons.append({**case,'risk_error_difference':a['metrics']['risk_target_mae']-b['metrics']['risk_target_mae'],
      'net_return_difference':a['metrics']['total_return']-b['metrics']['total_return'],
      'equity_exposure_difference':a['metrics']['mean_equity_fraction']-b['metrics']['mean_equity_fraction'],
      'paired_intervals':{str(block):block_bootstrap(differences,block=block,repetitions=1000) for block in [20,60]}})
    print(preset,protocol,model,cost,flush=True)
metrics=pd.DataFrame(rows);metrics.to_csv(root/'results/PORTFOLIO_V1.csv',index=False)
write_json(root/'results/PORTFOLIO_V1_COMPARISONS.json',comparisons)
lines=['# Portfolio Risk Lab — exploratory application results','',plan['status'],'',
 '48 declared scenarios: two baskets, four overlapping evaluation protocols, two existing forecasting models and three proportional cost assumptions. Four strategies per scenario. No model is selected using these outcomes. No recent foundation model has yet been evaluated.','',
 '## Primary comparison: forecast rule minus historical-risk rule','',
 'Cost: 10 bps per dollar traded; target: 10% annual volatility. Negative risk-error difference favors the forecast rule. Return/exposure differences must be read together; lower exposure alone can reduce risk.','',
 '| Basket | Period | Forecast | Risk-error change (pp) | Net-return change (pp) | Equity-exposure change (pp) |','|---|---|---|---:|---:|---:|']
for c in comparisons:lines.append(f'| {c["preset"]} | {c["protocol"]} | {c["model"]} | {100*c["risk_error_difference"]:.3f} | {100*c["net_return_difference"]:.2f} | {100*c["equity_exposure_difference"]:.2f} |')
lines += ['', '## Interpretation and limits','',
 'These are scenario diagnostics, not independent replications or proof of alpha. Both risk rules use the same basket and trailing correlation estimator; only marginal volatility estimates change. They share a target, not necessarily realized risk or average exposure. Buy-and-hold and fixed half-equity supply context, not matched-exposure attribution.','',
 'Primary errors use overlapping twenty-session net strategy returns. Paired circular twenty-/sixty-date block intervals in the JSON are exploratory and conditional on fitted forecasts, selected baskets and simulation rules. They omit model-retraining and population-selection uncertainty. Do not count correlated scenarios as independent evidence.','',
 'The account starts in cash, pays entry costs at the first next-close execution, then lets holdings drift. Risk rules rebalance every twenty signal sessions. Cash earns zero; no leverage, taxes, market impact or forced terminal liquidation. The saved forecast label starts one session before the tradable holding window.','',
 'See docs/PORTFOLIO_PROTOCOL.md for formulas and timing; config/portfolio-v1.json for the full grid. Read all cost scenarios in PORTFOLIO_V1.csv, including unfavorable results. The previous forecasting results remain separate preliminary evidence.']
(root/'results/PORTFOLIO_V1.md').write_text('\n'.join(lines))
write_json(out/'COMPLETED.json',{'completed_utc':utc_now(),'scenarios':48,'strategy_rows':len(rows)})
print('Completed',len(rows),'strategy rows')
