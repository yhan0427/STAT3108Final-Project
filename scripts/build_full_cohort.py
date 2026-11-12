"""Collect and audit the frozen full cohort without touching 2026 outcomes."""
from pathlib import Path
import sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
import yfinance as yf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.io import read_json,write_json,sha256,utc_now
from volrisk.dataset import market_features,assign_splits
from volrisk.aligned import align_panel

def main():
    roster=pd.read_csv(ROOT/'config/full-cohort-v1.csv');plan=read_json(ROOT/'config/full-cohort-v1.json');assert sha256(ROOT/'config/full-cohort-v1.csv')==plan['roster_sha256']
    out=ROOT/'data/processed/full-cohort-v1'
    if (out/'panel.pkl').exists():raise FileExistsError('Version any rebuilt full panel')
    cfg=read_json(ROOT/'config/study.json');market=pd.read_csv(ROOT/'data/raw/market/SPY.csv',parse_dates=['date']).set_index('date')
    yf.set_tz_cache_location(str(ROOT/'data/raw/yfinance_cache'))
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'acquisition_plan.json',{'started_utc':utc_now(),'plan':plan,'source_sha256':sha256(Path(__file__)),'market_sha256':sha256(ROOT/'data/raw/market/SPY.csv')})
    def collect(row):
        t=row['ticker'];path=ROOT/f'data/raw/market/{t}.csv';record={k:row[k] for k in ['ticker','issuer_id','sector','role','prior_development']}
        try:
            if not path.exists():
                error=None
                for attempt in range(3):
                    try:
                        raw=yf.Ticker(row['provider_symbol']).history(start='2014-01-01',end='2026-01-01',auto_adjust=False,actions=True,raise_errors=True)
                        if raw.empty or 'Adj Close' not in raw:raise ValueError('No adjusted price history')
                        raw.index=raw.index.tz_localize(None).normalize();raw.index.name='date';raw=raw.rename(columns=lambda v:v.lower().replace(' ','_'))
                        raw.to_csv(path)
                        write_json(path.with_suffix('.source.json'),{'provider':'Yahoo Finance via yfinance','url':f"https://finance.yahoo.com/quote/{row['provider_symbol']}/history/",'provider_symbol':row['provider_symbol'],'issuer_id':row['issuer_id'],'start':'2014-01-01','end_exclusive':'2026-01-01','retrieved_utc':utc_now(),'rows':len(raw),'sha256':sha256(path),'auto_adjust':False})
                        break
                    except Exception as e:error=e;time.sleep(1+attempt)
                if not path.exists():raise error
            raw=pd.read_csv(path,parse_dates=['date']).set_index('date');meta=read_json(path.with_suffix('.source.json'))
            if sha256(path)!=meta['sha256']:raise ValueError('Source checksum mismatch')
            if raw.index.max()>=pd.Timestamp('2026-01-01'):raise ValueError('Sealed 2026 data found')
            record.update(raw_rows=len(raw),first_date=str(raw.index.min().date()),last_date=str(raw.index.max().date()),sha256=sha256(path))
            if raw.index.max()!=market.index.max():raise ValueError('Incomplete terminal history relative to 2025-12-31 reference')
            expected=market.loc[raw.index.min():raw.index.max()]
            if not raw.index.equals(expected.index):raise ValueError('Missing/duplicate/misaligned trading sessions; no forward fill')
            f=market_features(raw,expected,cfg);f['ticker']=t;f['sector']=row['sector'];f['role']=row['role'];f['issuer_id']=row['issuer_id'];f['prior_development']=row['prior_development']
            f=align_panel(f);f['split']=assign_splits(f,cfg)
            f.loc[~f.sequence_ready | f.target.isna() | f.target.le(0),'split']='excluded'
            for split in ['train','validation','test']:
                record['eligible_'+split]=int(f.split.eq(split).sum());record['context512_'+split]=int((f.split.eq(split)&f.context_ready.eq(True)).sum())
            record['status']='ready';return record,f
        except Exception as e:record.update(status='failed',reason=str(e));return record,None
    ledger=[];frames=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(collect,row) for row in roster.to_dict('records')]
        for future in as_completed(futures):
            record,frame=future.result();ledger.append(record)
            if frame is not None:frames.append(frame)
            write_json(ROOT/'results/FULL_COHORT_COVERAGE_V1.json',sorted(ledger,key=lambda r:r['ticker']))
            print(len(ledger),record['ticker'],record['status'],record.get('reason',''),flush=True)
    if len(frames)<350:raise RuntimeError('Coverage too low; inspect failure ledger before building')
    panel=pd.concat(frames,ignore_index=True).sort_values(['ticker','date']).reset_index(drop=True);panel.to_pickle(out/'panel.pkl')
    summary={'completed_utc':utc_now(),'candidate_issuers':500,'retained_raw_histories':len(frames),'raw_company_date_rows_retained':sum(r['raw_rows'] for r in ledger if r['status']=='ready'),'failed':[r for r in ledger if r['status']!='ready'],'model_ready_counts':[],'panel_sha256':sha256(out/'panel.pkl'),'roster_sha256':sha256(ROOT/'config/full-cohort-v1.csv'),'holdout_2026_accessed':False}
    for (role,split),g in panel[panel.split.ne('excluded')].groupby(['role','split']):summary['model_ready_counts'].append({'role':role,'split':split,'firms_context60':g.ticker.nunique(),'origins_context60':len(g),'firms_context512':g[g.context_ready.eq(True)].ticker.nunique(),'origins_context512':int(g.context_ready.eq(True).sum())})
    write_json(out/'manifest.json',summary);write_json(ROOT/'results/FULL_COHORT_AUDIT_V1.json',summary);print('COMPLETE',summary['retained_raw_histories'],summary['raw_company_date_rows_retained'],flush=True)
if __name__=='__main__':main()
