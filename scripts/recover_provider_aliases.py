"""Recover verified symbol-only changes; preserve initial collection as v1."""
from pathlib import Path
import sys
import pandas as pd
import yfinance as yf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.io import read_json,write_json,sha256,utc_now
from volrisk.dataset import market_features,assign_splits
from volrisk.aligned import align_panel

def main():
    out=ROOT/'data/processed/full-cohort-v2'
    if out.exists():raise FileExistsError('Preserve alias recovery version')
    roster=pd.read_csv(ROOT/'config/full-cohort-v1.csv').set_index('ticker');aliases=read_json(ROOT/'config/provider-aliases-v1.json');cfg=read_json(ROOT/'config/study.json')
    market=pd.read_csv(ROOT/'data/raw/market/SPY.csv',parse_dates=['date']).set_index('date');yf.set_tz_cache_location(str(ROOT/'data/raw/yfinance_cache'))
    panel=pd.read_pickle(ROOT/'data/processed/full-cohort-v1/panel.pkl');frames=[panel];ledger=read_json(ROOT/'results/FULL_COHORT_COVERAGE_V1.json');recoveries=[]
    for t,a in aliases.items():
        r=roster.loc[t];assert r.issuer_id==a['issuer_id'];assert t not in set(panel.ticker)
        path=ROOT/f'data/raw/market/{t}.csv'
        raw=yf.Ticker(a['provider_symbol']).history(start='2014-01-01',end='2026-01-01',auto_adjust=False,actions=True,raise_errors=True)
        if raw.empty or 'Adj Close' not in raw:raise ValueError('Alias history unavailable; do not substitute another company')
        raw.index=raw.index.tz_localize(None).normalize();raw.index.name='date';raw=raw.rename(columns=lambda v:v.lower().replace(' ','_'))
        expected=market.loc[raw.index.min():raw.index.max()]
        assert raw.index.equals(expected.index) and raw.index.max()==pd.Timestamp('2025-12-31')
        raw.to_csv(path);write_json(path.with_suffix('.source.json'),{'provider':'Yahoo Finance via yfinance','start':'2014-01-01','end_exclusive':'2026-01-01','auto_adjust':False,'retrieved_utc':utc_now(),'sha256':sha256(path),'rows':len(raw),'reference_ticker':t,**a})
        f=market_features(raw,expected,cfg)
        for key in ['ticker','sector','role','issuer_id','prior_development']:f[key]=t if key=='ticker' else r[key]
        f=align_panel(f);f['split']=assign_splits(f,cfg);f.loc[~f.sequence_ready|f.target.isna()|f.target.le(0),'split']='excluded';frames.append(f)
        record=next(v for v in ledger if v['ticker']==t);record.update(status='ready',provider_symbol=a['provider_symbol'],recovery_source=a['source'],raw_rows=len(raw),first_date=str(raw.index.min().date()),last_date=str(raw.index.max().date()),sha256=sha256(path));record['initial_failure']=record.pop('reason')
        for split in ['train','validation','test']:record['eligible_'+split]=int(f.split.eq(split).sum());record['context512_'+split]=int((f.split.eq(split)&f.context_ready.eq(True)).sum())
        recoveries.append(t);print('Recovered',t,a['provider_symbol'],len(raw),flush=True)
    out.mkdir(parents=True);result=pd.concat(frames,ignore_index=True).sort_values(['ticker','date']).reset_index(drop=True);result.to_pickle(out/'panel.pkl')
    summary={'completed_utc':utc_now(),'candidate_issuers':500,'retained_raw_histories':sum(r['status']=='ready' for r in ledger),'raw_company_date_rows_retained':sum(r.get('raw_rows',0) for r in ledger if r['status']=='ready'),'failed':[r for r in ledger if r['status']!='ready'],'verified_alias_recoveries':recoveries,'panel_sha256':sha256(out/'panel.pkl'),'roster_sha256':sha256(ROOT/'config/full-cohort-v1.csv'),'previous_manifest_sha256':sha256(ROOT/'data/processed/full-cohort-v1/manifest.json'),'model_ready_counts':[],'holdout_2026_accessed':False}
    for (role,split),g in result[result.split.ne('excluded')].groupby(['role','split']):summary['model_ready_counts'].append({'role':role,'split':split,'firms_context60':g.ticker.nunique(),'origins_context60':len(g),'firms_context512':g[g.context_ready.eq(True)].ticker.nunique(),'origins_context512':int(g.context_ready.eq(True).sum())})
    write_json(out/'manifest.json',summary);write_json(ROOT/'results/FULL_COHORT_AUDIT_V2.json',summary);write_json(ROOT/'results/FULL_COHORT_COVERAGE_V2.json',ledger)
    print('COMPLETE',summary['retained_raw_histories'],summary['raw_company_date_rows_retained'],flush=True)
if __name__=='__main__':main()
