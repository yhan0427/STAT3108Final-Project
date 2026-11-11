"""Recompute V1 turnover only; verify all other saved metrics are unchanged."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from volrisk.portfolio import PortfolioLab,PRESETS
from volrisk.io import sha256,write_json,utc_now

source=ROOT/'results/PORTFOLIO_V1.csv'
output=ROOT/'results/PORTFOLIO_V1_TURNOVER_CORRECTION.csv'
if output.exists(): raise FileExistsError('Preserve existing correction')
old=pd.read_csv(source);lab=PortfolioLab(ROOT);records=[]
keys=['preset','protocol','model','cost_bps']
for case,group in old.groupby(keys,sort=False):
    preset,protocol,model,cost=case
    result=lab.backtest(protocol,model,PRESETS[preset],.1,cost)
    for _,row in group.iterrows():
        metric=result['strategies'][row.strategy]['metrics']
        for name,value in metric.items():
            if name!='turnover':
                assert np.isclose(row[name],value,rtol=1e-10,atol=1e-12), (case,name,row[name],value)
        records.append({**dict(zip(keys,case)),'strategy':row.strategy,'original_turnover':row.turnover,'corrected_turnover':metric['turnover']})
pd.DataFrame(records).to_csv(output,index=False)
write_json(ROOT/'results/PORTFOLIO_V1_TURNOVER_CORRECTION.json',{'created_utc':utc_now(),'original_csv_sha256':sha256(source),
    'corrected_csv_sha256':sha256(output),'source_sha256':sha256(ROOT/'src/volrisk/portfolio.py'),'rows':len(records),
    'correction':'Gross traded dollars divided by execution-day NAV after market movement and before costs, rather than previous close NAV.',
    'all_other_metrics_unchanged':True})
print('Verified and corrected',len(records),'rows',flush=True)
