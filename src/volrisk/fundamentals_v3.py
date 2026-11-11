"""Explicit alternative accounting schema, without mutating the v1/v2 panel."""
from pathlib import Path
import numpy as np
import pandas as pd
from .fundamentals import TAGS,flatten_facts,asof_periods,trailing_flow,safe_ratio,align_snapshots
from .io import read_json,write_json,sha256,utc_now

V3_FEATURES=['parent_equity_to_assets','cash_to_assets','current_ratio','current_liabilities_to_assets','income_to_assets','cfo_to_assets','revenue_growth','filing_age_days']
EXTRA={'parent_equity':['StockholdersEquity'],'pretax_income':['IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'],'noncurrent_liabilities':['LiabilitiesNoncurrent']}

def extra_snapshot(facts,date,end):
    known=asof_periods(facts,pd.Timestamp(date));end=pd.Timestamp(end)
    def instant(concept):
        rows=known.loc[(known.concept==concept)&known.start.isna()&known.end.eq(end)]
        return float(rows.iloc[-1].value) if len(rows) else np.nan
    assets=instant('assets');revenue=trailing_flow(known,'revenue',end)
    current,noncurrent=instant('current_liabilities'),instant('noncurrent_liabilities')
    return {'parent_equity_to_assets':safe_ratio(instant('parent_equity'),assets),'current_liabilities_to_assets':safe_ratio(current,assets),'pretax_margin':safe_ratio(trailing_flow(known,'pretax_income',end),revenue),'liability_components_to_assets':safe_ratio(current+noncurrent,assets)}

def build(root):
    output=root/'data/processed/financial-v3'
    if output.exists(): raise FileExistsError('Preserve existing v3 data; version any revision.')
    output.mkdir(parents=True)
    original=root/'data/processed/main/panel.pkl';panel=pd.read_pickle(original)
    entities=read_json(root/'data/raw/sec/resolved_entities.json')
    coverage=[]
    for ticker in sorted(panel.ticker.unique()):
        snapshots=pd.read_csv(root/f'data/processed/main/{ticker}_fundamental_snapshots.csv',parse_dates=['filing_date','report_end'])
        facts=flatten_facts(read_json(root/entities[ticker]['path']),{**TAGS,**EXTRA})
        extra=pd.DataFrame([extra_snapshot(facts,r.filing_date,r.report_end) for r in snapshots.itertuples()])
        snaps=pd.concat([snapshots.reset_index(drop=True),extra],axis=1)
        snaps.to_csv(output/f'{ticker}_snapshots.csv',index=False)
        idx=panel.index[panel.ticker.eq(ticker)]
        aligned=align_snapshots(panel.loc[idx,'date'],snaps)
        # The common aligner stales original fields; explicitly stale added ones.
        aligned.loc[aligned.filing_age_days>180,list(extra.columns)]=np.nan
        for col in extra.columns: panel.loc[idx,col]=aligned[col].to_numpy()
        eligible=panel.loc[idx].query('split != "excluded"')
        coverage.append({'ticker':ticker,**{f:float(eligible[f].notna().mean()) for f in ['liabilities_to_assets','operating_margin',*extra.columns]}})
        print('Accounting schema audit',ticker,flush=True)
    panel.to_pickle(output/'panel.pkl')
    pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False)
    manifest=read_json(root/'data/processed/main/manifest.json')
    manifest.update({'built_utc':utc_now(),'parent_panel_sha256':sha256(original),'panel_sha256':sha256(output/'panel.pkl'),'feature_schema':'V3: replace total-liability ratio with parent equity ratio and operating margin with current liabilities/assets; economic definitions differ. This is an alternative feature set, not imputation of the same quantities. Pretax margin was rejected on coverage alone before any model fitting.','fundamental_features':V3_FEATURES})
    write_json(output/'manifest.json',manifest)
    print(pd.DataFrame(coverage).drop(columns='ticker').mean().to_string(),flush=True)

if __name__=='__main__':build(Path.cwd())
