"""Guards for the frozen company expansion and delayed-label partitions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from volrisk.io import sha256
ROOT=Path(__file__).resolve().parents[1]


def test_frozen_expansion_preserves_company_separation_and_official_repairs():
    r=pd.read_csv(ROOT/'config/universe-2025-reconstructed.csv')
    assert len(r)==503 and r.issuer_id.nunique()==500
    assert {'CRH','CVNA','FIX'}<=set(r.ticker)
    assert not {'SOLS','LKQ','MHK'} & set(r.ticker)
    selected=pd.read_csv(ROOT/'config/expansion-100-v1.csv')
    plan=json.loads((ROOT/'config/expansion-100-v1.json').read_text())
    assert sha256(ROOT/'config/expansion-100-v1.csv')==plan['selection_sha256']
    assert len(selected)==100 and selected.issuer_id.nunique()==100
    original=set(json.loads((ROOT/'config/universe.json').read_text())['companies'])
    assert original<=set(selected[selected.role.eq('original_training')].ticker)
    assert not set(selected[selected.role.eq('held_company')].ticker)&original
    assert selected.role.value_counts().to_dict()=={'new_training':35,'held_company':35,'original_training':30}


def test_expanded_labels_and_histories_respect_boundaries():
    path=ROOT/'data/processed/expansion-100-v1/panel.pkl'
    if not path.exists():pytest.skip('Local expansion data required')
    p=pd.read_pickle(path)
    assert p.date.max()<pd.Timestamp('2026-01-01')
    for split,stop in [('train','2022-01-01'),('validation','2024-01-01'),('test','2026-01-01')]:
        q=p[p.split.eq(split)]
        assert (q.target_end<pd.Timestamp(stop)).all()
        idx=q.index.to_numpy()
        assert np.array_equal(p.loc[idx,'ticker'],p.loc[idx-59,'ticker'])
    assert p[p.role.ne('held_company')].ticker.nunique()==65
    assert p[p.role.eq('held_company')].ticker.nunique()==34
