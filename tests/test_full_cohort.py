from pathlib import Path
import json
import pandas as pd
import pytest
from volrisk.io import sha256
ROOT=Path(__file__).resolve().parents[1]


def test_full_cohort_reserves_only_unseen_issuers():
    roster=pd.read_csv(ROOT/'config/full-cohort-v1.csv')
    pilot=pd.read_csv(ROOT/'config/expansion-100-v1.csv')
    plan=json.loads((ROOT/'config/full-cohort-v1.json').read_text())
    assert len(roster)==500 and roster.issuer_id.nunique()==500
    assert sha256(ROOT/'config/full-cohort-v1.csv')==plan['roster_sha256']
    held=set(roster[roster.role.eq('held_company')].issuer_id)
    assert len(held)==100 and not held & set(pilot.issuer_id)
    assert roster[roster.role.eq('training_company')].issuer_id.nunique()==400
    assert set(roster.sector)==set(roster[roster.role.eq('held_company')].sector)


def test_full_collection_manifest_reconciles_counts_and_date_boundaries():
    manifest=ROOT/'data/processed/full-cohort-v1/manifest.json'
    if not manifest.exists():pytest.skip('Full local collection not complete')
    meta=json.loads(manifest.read_text());ledger=json.loads((ROOT/'results/FULL_COHORT_COVERAGE_V1.json').read_text())
    assert len(ledger)==500 and len({r['issuer_id'] for r in ledger})==500
    ready=[r for r in ledger if r['status']=='ready']
    assert len(ready)==meta['retained_raw_histories']
    assert sum(r['raw_rows'] for r in ready)==meta['raw_company_date_rows_retained']
    assert all(r['last_date']=='2025-12-31' for r in ready)
    assert meta['holdout_2026_accessed'] is False
    assert sha256(manifest.parent/'panel.pkl')==meta['panel_sha256']
