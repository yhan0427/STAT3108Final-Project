"""Download only pre-2026 market history for the frozen expansion roster."""
from pathlib import Path
import sys,time
import pandas as pd
import yfinance as yf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.io import read_json,write_json,sha256,utc_now
from volrisk.dataset import market_features,assign_splits
from volrisk.aligned import align_panel

roster=pd.read_csv(ROOT/'config/expansion-100-v1.csv');cfg=read_json(ROOT/'config/study.json')
yf.set_tz_cache_location(str(ROOT/'data/raw/yfinance_cache'))
market=pd.read_csv(ROOT/'data/raw/market/SPY.csv',parse_dates=['date']).set_index('date')
frames=[];ledger=[]
for _,r in roster.iterrows():
    path=ROOT/f'data/raw/market/{r.ticker}.csv'
    try:
        if not path.exists():
            last=None
            for attempt in range(3):
                try:
                    raw=yf.Ticker(r.provider_symbol).history(start='2014-01-01',end='2026-01-01',auto_adjust=False,actions=True,raise_errors=True)
                    if raw.empty or 'Adj Close' not in raw:raise ValueError('No adjusted history')
                    raw.index=raw.index.tz_localize(None).normalize();raw.index.name='date';raw=raw.rename(columns=lambda v:v.lower().replace(' ','_'))
                    raw.to_csv(path)
                    write_json(path.with_suffix('.source.json'),{'provider':'Yahoo Finance via yfinance','url':f'https://finance.yahoo.com/quote/{r.provider_symbol}/history/','start':'2014-01-01','end_exclusive':'2026-01-01','retrieved_utc':utc_now(),'rows':len(raw),'sha256':sha256(path),'auto_adjust':False,'issuer_id':r.issuer_id,'provider_symbol':r.provider_symbol})
                    break
                except Exception as e:last=e;time.sleep(1)
            if not path.exists():raise last
        raw=pd.read_csv(path,parse_dates=['date']).set_index('date')
        meta=read_json(path.with_suffix('.source.json'));assert meta['sha256']==sha256(path)
        assert raw.index.max()<pd.Timestamp('2026-01-01')
        expected=market.loc[raw.index.min():raw.index.max()]
        if not raw.index.equals(expected.index):raise ValueError('Incomplete/misaligned trading calendar')
        f=market_features(raw,expected,cfg);f['ticker']=r.ticker;f['sector']=r.sector;f['role']=r.role;frames.append(f)
        ledger.append({'ticker':r.ticker,'role':r.role,'status':'ready','raw_rows':len(raw),'first_date':str(raw.index.min().date()),'last_date':str(raw.index.max().date()),'sha256':sha256(path)})
        print('Ready',r.ticker,len(raw),flush=True)
    except Exception as e:
        ledger.append({'ticker':r.ticker,'role':r.role,'status':'failed','reason':str(e)});print('FAILED',r.ticker,str(e),flush=True)
    write_json(ROOT/'results/EXPANSION_100_COVERAGE.json',ledger)
if len(frames)<80:raise RuntimeError('Too few collected firms; inspect coverage ledger')
panel=align_panel(pd.concat(frames,ignore_index=True));panel['split']=assign_splits(panel,cfg)
panel.loc[~panel.sequence_ready | panel.target.isna() | panel.target.le(0),'split']='excluded'
out=ROOT/'data/processed/expansion-100-v1';out.mkdir(parents=True,exist_ok=True)
if (out/'panel.pkl').exists():raise FileExistsError('Preserve existing expansion panel')
panel.to_pickle(out/'panel.pkl')
write_json(out/'manifest.json',{'built_utc':utc_now(),'roster_sha256':sha256(ROOT/'config/expansion-100-v1.csv'),'panel_sha256':sha256(out/'panel.pkl'),'raw_observations':sum(r.get('raw_rows',0) for r in ledger),'candidate_companies':100,'retained_companies':len(frames),'excluded':[r for r in ledger if r['status']!='ready'],'split_role_counts':panel.groupby(['split','role']).size().to_dict().__str__(),'holdout_2026_accessed':False})
print('COMPLETE',len(panel),'raw observations',flush=True)
