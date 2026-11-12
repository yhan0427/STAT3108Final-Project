"""Freeze all reconstructed issuers and untouched new-company roles before acquisition."""
from pathlib import Path
import sys,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.io import write_json,sha256,utc_now
out=ROOT/'config/full-cohort-v1.csv'
if out.exists():raise FileExistsError('Full cohort already frozen')
f=pd.read_csv(ROOT/'config/universe-2025-reconstructed.csv');f=f[f.primary_company_security].copy()
pilot=set(pd.read_csv(ROOT/'config/expansion-100-v1.csv').issuer_id)
f['prior_development']=f.issuer_id.isin(pilot)
new=f[~f.prior_development].copy();new['rank']=new.issuer_id.map(lambda x:hashlib.sha256(('Han-full-cohort-v1/'+x).encode()).hexdigest())
counts=new.groupby('sector').size();exact=100*counts/counts.sum();quota=exact.astype(int)
for s in (exact-quota).sort_values(ascending=False).index[:100-quota.sum()]:quota[s]+=1
held=set(pd.concat([new[new.sector.eq(s)].sort_values('rank').head(int(n)) for s,n in quota.items()]).issuer_id)
f['role']=f.issuer_id.map(lambda x:'held_company' if x in held else 'training_company')
assert len(f)==500 and f.issuer_id.nunique()==500 and len(held)==100 and not held & pilot
f.sort_values('ticker').to_csv(out,index=False)
write_json(ROOT/'config/full-cohort-v1.json',{'frozen_utc':utc_now(),'roster_sha256':sha256(out),'source_registry_sha256':sha256(ROOT/'config/universe-2025-reconstructed.csv'),'scope':'All 500 CIK-defined issuers in the explicitly reconstructed 2025-12-31 cohort, one class per issuer; independent official full-list verification remains qualified. Not historical membership at every date.','roles':{'training_company':400,'held_company':100},'selection':'100 held firms selected only from outside the 100 previously scored pilot issuers, sector-proportional largest remainders and fixed SHA256 rank. Earlier pilot held firms now belong to development/training; no longer untouched.','dates':{'start':'2014-01-01','end_exclusive':'2026-01-01'},'coverage':'Preserve all candidate rows; do not replace unavailable firms. Require complete observed trading calendar from first price through 2025-12-31. Report model-ready origins separately.','holdout_2026':'sealed','stage':'Data collection/quality audit only. No new full training grid until coverage checks pass.'})
print(f.role.value_counts().to_string())
