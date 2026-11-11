"""Reconstruct a dated cohort with explicit official corrections, then freeze roles."""
from pathlib import Path
import sys,io,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.io import read_json,write_json,sha256,utc_now

if (ROOT/'config/expansion-100-v1.json').exists():
    raise FileExistsError('Candidate cohort already frozen; version any selection revision')
source=ROOT/'data/raw/universe/wikipedia-1329075976.html'
f=pd.read_html(io.StringIO(source.read_text()))[0]
f=f.rename(columns={'Symbol':'ticker','Security':'company','GICS Sector':'sector','CIK':'cik'})
f=f[['ticker','company','sector','cik']].copy();f['correction']='historical_page'
f=f[~f.ticker.eq('SOLS')]
sec={v['ticker']:v for v in read_json(ROOT/'data/raw/sec/company_tickers.json').values()}
for t,sector in [('CRH','Materials'),('CVNA','Consumer Discretionary'),('FIX','Industrials')]:
    assert not f.ticker.eq(t).any()
    f.loc[len(f)+1000]=[t,sec[t]['title'],sector,sec[t]['cik_str'],'official_2025-12-22_addition']
assert len(f)==503 and f.ticker.nunique()==503 and f.cik.nunique()==500
assert not set(['LKQ','SOLS','MHK']) & set(f.ticker)
f['reference_date']='2025-12-31';f['issuer_id']=f.cik.map(lambda x:f'CIK{int(x):010d}')
f['provider_symbol']=f.ticker.str.replace('.','-',regex=False)
f['primary_company_security']=~f.sort_values('ticker').cik.duplicated().reindex(f.index)
f=f.sort_values('ticker').reset_index(drop=True)
f.to_csv(ROOT/'config/universe-2025-reconstructed.csv',index=False)
companies=f[f.primary_company_security].copy()
legacy=set(read_json(ROOT/'config/universe.json')['companies']);assert legacy<=set(companies.ticker)
remaining=companies[~companies.ticker.isin(legacy)].copy()
remaining['rank']=remaining.issuer_id.map(lambda x:hashlib.sha256(('Han-expansion-v1/'+x).encode()).hexdigest())
counts=remaining.groupby('sector').size();exact=70*counts/counts.sum();quota=exact.astype(int)
for sector in (exact-quota).sort_values(ascending=False).index[:70-quota.sum()]:quota[sector]+=1
new=pd.concat([remaining[remaining.sector.eq(s)].sort_values('rank').head(int(n)) for s,n in quota.items()])
new=new.sort_values('rank');new['role']=['new_training' if i%2==0 else 'held_company' for i in range(len(new))]
old=companies[companies.ticker.isin(legacy)].copy();old['role']='original_training'
selected=pd.concat([old,new]).sort_values('ticker').drop(columns=['rank'],errors='ignore')
assert len(selected)==100 and selected.issuer_id.nunique()==100
selected.to_csv(ROOT/'config/expansion-100-v1.csv',index=False)
write_json(ROOT/'config/expansion-100-v1.json',{'frozen_utc':utc_now(),'scope':'Reconstructed 2025-12-31 fixed cohort; not historical membership at every date. Independent full roster validation remains limited; corrections are explicit.','historical_page':'https://en.wikipedia.org/w/index.php?oldid=1329075976','historical_page_sha256':sha256(source),'official_corrections':'https://www.spglobal.com/spdji/en/documents/indexnews/announcements/20251205-1480992/1480992_december2025rebalance.pdf','corrections':{'add':['CRH','CVNA','FIX'],'remove':['SOLS'],'already_removed_in_page':['LKQ','MHK']},'sec_mapping_sha256':sha256(ROOT/'data/raw/sec/company_tickers.json'),'registry_sha256':sha256(ROOT/'config/universe-2025-reconstructed.csv'),'selection_sha256':sha256(ROOT/'config/expansion-100-v1.csv'),'selection':'Keep original 30; sector-proportional largest-remainder 70 additions, ranked by SHA256(Han-expansion-v1/issuer_id); alternate hash order into 35 training and 35 held companies. One lexicographically first listed class per CIK.','roles':selected.role.value_counts().to_dict(),'data_start':'2014-01-01','data_end_exclusive':'2026-01-01','train_start':'2015-01-01','validation_start':'2022-01-01','test_start':'2024-01-01','test_end_exclusive':'2026-01-01','target':'execution aligned t+2..t+21','missing_policy':'No fabricated/forward-filled prices. Record download/identity/coverage failures, never replace selected firms based on outcomes. Shorter histories allowed after full past context.','model_scope':'Market-only pooled Ridge and LSTM trained on original+new_training firms; held_company labels excluded from training, selection and calibration. Existing pretrained models remain retrospective comparators.','holdout_2026':'sealed'})
print(selected.role.value_counts().to_string())
