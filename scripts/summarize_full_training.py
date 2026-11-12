"""Full, unfiltered period/company diagnostics for the unified run."""
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.metrics import forecast_metrics
from volrisk.io import read_json
p=pd.read_csv(ROOT/'results/aligned/full-cohort-v1/predictions.csv',parse_dates=['date','target_end']);p=p[p.split.eq('test')]
assert p.target_end.max()<pd.Timestamp('2026-01-01') and not p.duplicated(['ticker','date']).any()
models=['Persistence','Ridge60','LSTM60','Timestral5M','Persistence_calibrated','Ridge60_calibrated','LSTM60_calibrated','Timestral5M_calibrated'];rows=[]
for period,g in [('2024-2025',p),('2024',p[p.date.dt.year.eq(2024)]),('2025',p[p.date.dt.year.eq(2025)])]:
    for model in models:rows.append({'period':period,'model':model,'companies':g.ticker.nunique(),**forecast_metrics(g.target,g[model],g.rv20)})
pd.DataFrame(rows).to_csv(ROOT/'results/FULL_TRAINING_PERIODS_V1.csv',index=False)
c=pd.read_csv(ROOT/'results/FULL_TRAINING_COMPANIES_V1.csv');b=c[c.model.eq('Ridge60_calibrated')].set_index('ticker').qlike
ratios=[]
for model,g in c.groupby('model'):
    ratio=g.qlike.to_numpy()/g.ticker.map(b).to_numpy();ratios.append({'model':model,'companies':len(g),'mean_company_qlike_ratio_vs_calibrated_ridge':ratio.mean(),'companies_with_lower_qlike':int((ratio<1).sum())})
pd.DataFrame(ratios).to_csv(ROOT/'results/FULL_TRAINING_COMPANY_RATIOS_V1.csv',index=False)
audit=read_json(ROOT/'results/FULL_TRAINING_AUDIT_V1.json')
path=ROOT/'results/FULL_TRAINING_V1.md';s=path.read_text();s+='\n## Uncertainty and aggregation\n\nCalibrated LSTM minus calibrated Ridge mean QLIKE = -0.01974. The paired 20-date interval is [-0.06850, +0.02612]; the 60-date interval is [-0.08325, +0.03934]. Both cross zero. A lower point estimate is not a stable superiority claim. Calibrated Timestral minus Ridge = +0.03912; the 20-date interval crosses zero and the 60-date interval does not. These conditional diagnostics omit refitting, cohort-selection and multiplicity uncertainty.\n\nAll yearly rows are retained in FULL_TRAINING_PERIODS_V1.csv; equal-company normalized loss ratios and win counts are in FULL_TRAINING_COMPANY_RATIOS_V1.csv. Forecast errors are not portfolio returns or evidence of alpha. The next application stage must evaluate risk-target errors alongside returns, drawdown, equity exposure, turnover and costs using these frozen forecasts. The 2026 temporal holdout remains sealed.\n';path.write_text(s)
print(pd.DataFrame(ratios).to_string(index=False))
