"""Compare derived snapshots with manually read original statement amounts."""
from pathlib import Path
import numpy as np
import pandas as pd
from volrisk.io import read_json, write_json

root=Path(__file__).resolve().parents[1]
cases=read_json(root/'config/sec-spotchecks.json')['cases']
results=[]
for case in cases:
    snapshots=pd.read_csv(root/'data/processed/main'/f'{case["ticker"]}_fundamental_snapshots.csv')
    selected=snapshots.loc[(snapshots.filing_date==case['filing_date']) & (snapshots.report_end==case['report_end'])]
    assert len(selected)==1
    actual=selected.iloc[0]
    expected={
        'liabilities_to_assets':case['liabilities']/case['assets'],
        'cash_to_assets':case['cash']/case['assets'],
        'current_ratio':case['current_assets']/case['current_liabilities']
    }
    if 'revenue' in case:
        expected.update(operating_margin=case['operating_income']/case['revenue'],income_to_assets=case['income']/case['assets'],cfo_to_assets=case['cfo']/case['assets'],revenue_growth=case['revenue']/case['prior_revenue']-1)
    for feature,value in expected.items():
        match=bool(np.isclose(actual[feature],value,rtol=1e-9,atol=1e-12))
        results.append({'ticker':case['ticker'],'filing_date':case['filing_date'],'feature':feature,'snapshot':float(actual[feature]),'original_statement_ratio':value,'match':match,'source':case['url']})
write_json(root/'results/SEC_SPOTCHECKS.json',{'checks':results,'all_match':all(r['match'] for r in results),'scope':'Two original filings; ten ratios. No claim of full sample validation.'})
assert all(r['match'] for r in results)
print(f'{len(results)} original-statement ratio checks matched across {len(cases)} filings.')
