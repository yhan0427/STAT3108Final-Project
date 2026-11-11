import numpy as np
import pandas as pd
import pytest
from volrisk.aligned import execution_target, align_panel, context_matrix, variance_calibration, restore_ttm_prediction
from volrisk.dataset import assign_splits


def test_execution_target_uses_only_post_execution_returns():
    r = pd.Series(np.arange(40, dtype=float)/100)
    target = execution_target(r)
    assert target.iloc[0] == pytest.approx(np.std(r.iloc[2:22])*np.sqrt(252))
    changed = r.copy(); changed.iloc[1] = 1000
    assert execution_target(changed).iloc[0] == pytest.approx(target.iloc[0])
    changed = r.copy(); changed.iloc[21] = 1000
    assert execution_target(changed).iloc[0] != pytest.approx(target.iloc[0])
    assert target.iloc[-21:].isna().all()


def test_label_purge_respects_delayed_end():
    dates = pd.bdate_range('2021-11-01', periods=70)
    frame = pd.DataFrame({'date':dates, 'target_end':pd.Series(dates).shift(-21)})
    cfg = {'train_start':'2021-01-01','validation_start':'2022-01-01','test_start':'2023-01-01','test_end_exclusive':'2024-01-01'}
    splits = assign_splits(frame,cfg)
    assert (frame.loc[splits.eq('train'),'target_end'] < pd.Timestamp('2022-01-01')).all()
    assert splits.iloc[30] == 'excluded'


def test_context_is_past_only_and_company_bounded():
    frame = pd.DataFrame({'ticker':['A']*600+['B']*600,'rv20':np.arange(1200,dtype=float)})
    before = context_matrix(frame,[550])
    frame.loc[551:,'rv20'] = -100
    np.testing.assert_array_equal(context_matrix(frame,[550]),before)
    assert before[0,-1] == 550
    with pytest.raises(ValueError,match='company boundary'):
        context_matrix(frame,[700])


def test_calibration_minimizes_variance_qlike():
    from volrisk.metrics import qlike
    actual = np.array([.1,.2,.4]); pred = np.array([.15,.15,.15])
    scale = variance_calibration(actual,pred)
    loss = qlike(actual,pred*scale).mean()
    assert loss < qlike(actual,pred*scale*.9).mean()
    assert loss < qlike(actual,pred*scale*1.1).mean()
    assert variance_calibration(actual,actual/2) == pytest.approx(2)
    with pytest.raises(ValueError): variance_calibration(actual,[0,1,2])


def test_ttm_selects_21st_point_and_inverts_context_scaling():
    output = np.arange(96,dtype=float)[None,:,None]
    pred, count = restore_ttm_prediction(output,[.1],[.01])
    assert pred[0] == pytest.approx(.3)
    assert count == 0
    pred, count = restore_ttm_prediction(-output,[.1],[.01])
    assert pred[0] == 1e-6 and count == 1


def test_alignment_preserves_input_and_does_not_cross_tickers():
    dates = pd.bdate_range('2020-01-01', periods=600)
    panel = pd.concat([pd.DataFrame({'ticker':t,'date':dates,'return':np.arange(600)/10000,'rv20':.2,'target':.3,'target_end':dates}) for t in ['A','B']],ignore_index=True)
    aligned = align_panel(panel)
    assert panel.target.eq(.3).all()
    assert aligned.groupby('ticker').tail(21).target.isna().all()
    assert aligned.loc[0,'target_start'] == dates[2]
    assert aligned.loc[0,'target_end'] == dates[21]
