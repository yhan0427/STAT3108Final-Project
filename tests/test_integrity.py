import numpy as np
import pandas as pd
import pytest

from volrisk.dataset import assign_splits, forward_volatility, market_features
from volrisk.fundamentals import align_snapshots, asof_periods, flatten_facts, trailing_flow
from volrisk.metrics import block_bootstrap, forecast_metrics, qlike, ranking_by_date
from volrisk.models import TrainOnlyScaler


def test_forward_target_excludes_current_return_and_uses_exact_horizon():
    r = pd.Series([100.0, 1., 2., 3., 4., 5.])
    target = forward_volatility(r, 3, 252)
    assert target.iloc[0] == pytest.approx(np.std([1., 2., 3.]) * np.sqrt(252))
    assert target.iloc[2] == pytest.approx(np.std([3., 4., 5.]) * np.sqrt(252))
    assert target.iloc[-3:].isna().all()


def test_split_purges_targets_crossing_boundary():
    frame = pd.DataFrame({"date": pd.to_datetime(["2021-12-01", "2021-12-20", "2022-01-03", "2023-12-29"]), "target_end": pd.to_datetime(["2021-12-30", "2022-01-20", "2022-02-01", "2024-01-30"])})
    config = dict(train_start="2015-01-01", validation_start="2022-01-01", test_start="2024-01-01", test_end_exclusive="2026-01-01")
    assert assign_splits(frame, config).tolist() == ["train", "excluded", "validation", "excluded"]


def test_same_day_filing_is_not_used():
    snap = pd.DataFrame({"filing_date": pd.to_datetime(["2020-02-01", "2020-05-01"]), "operating_margin": [0.1, 0.9]})
    for name in ["liabilities_to_assets", "cash_to_assets", "current_ratio", "income_to_assets", "cfo_to_assets", "revenue_growth"]:
        snap[name] = 1.0
    aligned = align_snapshots(pd.to_datetime(["2020-05-01", "2020-05-04"]), snap)
    assert aligned.operating_margin.tolist() == [0.1, 0.9]


def test_later_restatement_cannot_change_earlier_snapshot():
    payload = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2020-12-31", "filed": "2021-02-01", "form": "10-K", "accn": "a", "val": 100},
        {"end": "2020-12-31", "filed": "2022-02-01", "form": "10-K", "accn": "b", "val": 500}
    ]}}}}}
    facts = flatten_facts(payload)
    assert asof_periods(facts, pd.Timestamp("2021-06-01")).iloc[0].value == 100
    assert asof_periods(facts, pd.Timestamp("2022-06-01")).iloc[0].value == 500


def test_cashflow_ytd_is_differenced_before_ttm_sum():
    rows = pd.DataFrame([
        ["cfo", "2020-01-01", "2020-09-30", 60],
        ["cfo", "2020-01-01", "2020-12-31", 100],
        ["cfo", "2021-01-01", "2021-03-31", 11],
        ["cfo", "2021-01-01", "2021-06-30", 33],
        ["cfo", "2021-01-01", "2021-09-30", 66],
    ], columns=["concept", "start", "end", "value"])
    for col in ["start", "end"]:
        rows[col] = pd.to_datetime(rows[col])
    rows["filed"] = pd.Timestamp("2021-11-01")
    assert trailing_flow(rows, "cfo", pd.Timestamp("2021-09-30")) == 40 + 11 + 22 + 33


def test_scaler_does_not_refit_on_future_extremes():
    scaler = TrainOnlyScaler().fit([[1., np.nan], [2., np.nan], [3., np.nan]])
    before = scaler.mean.copy()
    transformed = scaler.transform([[1e9, 50.]], missing=True)
    assert np.array_equal(scaler.mean, before)
    assert np.isfinite(transformed).all()
    assert transformed.shape == (1, 4)


def test_qlike_uses_variance_and_is_zero_for_perfect_forecast():
    assert qlike([0.2], [0.1])[0] == pytest.approx(4 - np.log(4) - 1)
    assert qlike([0.2, 0.3], [0.2, 0.3]).sum() == pytest.approx(0)


def test_persistence_relative_r2():
    metrics = forecast_metrics([0.1, 0.3], [0.2, 0.2], [0.2, 0.2])
    assert metrics["r2_vs_persistence"] == pytest.approx(0)


def test_top_k_and_no_trivial_pilot_score():
    frame = pd.DataFrame({"date": ["2024-01-01"] * 6, "ticker": list("ABCDEF"), "target": [1, 2, 3, 4, 5, 6], "model": [6, 5, 4, 3, 2, 1]})
    assert ranking_by_date(frame, "model", 2).iloc[0].top_k_overlap == 0
    assert ranking_by_date(frame.iloc[:5], "model", 5).empty


def test_block_bootstrap_is_paired_and_reproducible():
    a = block_bootstrap(np.ones(100) * -0.1, repetitions=20)
    assert a == block_bootstrap(np.ones(100) * -0.1, repetitions=20)
    assert a["ci95_low"] == pytest.approx(-0.1)
    assert a["ci95_high"] == pytest.approx(-0.1)


def test_future_price_mutation_does_not_change_past_features():
    dates = pd.bdate_range("2020-01-01", periods=220)
    rng = np.random.default_rng(4)
    price = 100 * np.exp(np.cumsum(rng.normal(0, .01, len(dates))))
    raw = pd.DataFrame({"adj_close": price, "close": price, "high": price * 1.01, "low": price / 1.01, "volume": 1000}, index=dates)
    cfg = dict(horizon=20, annualization=252, lookback=60, train_start="2020-01-01", validation_start="2021-01-01", test_start="2022-01-01", test_end_exclusive="2023-01-01")
    original = market_features(raw, raw, cfg)
    changed = raw.copy()
    changed.loc[dates[180]:, "adj_close"] *= 3
    alternate = market_features(changed, changed, cfg)
    from volrisk.dataset import MARKET_FEATURES, HAR_FEATURES
    pd.testing.assert_frame_equal(original.loc[:179, MARKET_FEATURES + HAR_FEATURES], alternate.loc[:179, MARKET_FEATURES + HAR_FEATURES])
    assert original.loc[175, "target"] != alternate.loc[175, "target"]


def test_evaluation_rejects_source_modified_after_freeze(tmp_path):
    from volrisk.experiment import evaluate
    from volrisk.io import sha256, write_json
    source = tmp_path / 'inference.py'
    source.write_text('original')
    run = tmp_path / 'results/runs/audit-fixture'
    write_json(run / 'frozen_run.json', {'source_sha256': {'inference.py': sha256(source)}})
    source.write_text('modified after freeze')
    with pytest.raises(ValueError, match='Source changed since freeze'):
        evaluate(tmp_path, 'audit-fixture')


def test_company_holdout_and_purged_robustness_partitions():
    from volrisk.robustness import partition
    rows=[]
    for ticker in ['SEEN','UNSEEN']:
        for date,end in [('2021-12-01','2021-12-29'),('2021-12-28','2022-01-25'),('2022-06-01','2022-06-29'),('2023-06-01','2023-06-29')]:
            rows.append(dict(ticker=ticker,date=pd.Timestamp(date),target_end=pd.Timestamp(end),sequence_ready=True,target=.2))
    panel=pd.DataFrame(rows)
    protocol=dict(train_end='2022-01-01',validation_end='2023-01-01',evaluation_end='2024-01-01',company_holdout=True)
    train,val,test=partition(panel,protocol,['UNSEEN'])
    assert train.tolist()==[0]
    assert val.tolist()==[2]
    assert test.tolist()==[7]


def test_sequence_ridge_receives_same_ordered_history_as_lstm():
    from volrisk.robustness import lag_matrix
    from volrisk.models import Sequences
    x=np.arange(60,dtype=np.float32).reshape(20,3)
    indices=np.array([7,12])
    sequences=Sequences(x,np.zeros((20,0)),np.zeros(20),indices,5)
    lagged=lag_matrix(x,indices,5)
    for j in range(len(indices)):
        np.testing.assert_array_equal(lagged[j],sequences[j][0].reshape(-1))


def test_missingness_control_erases_financial_values_but_preserves_age_and_masks():
    from volrisk.robustness import missingness_control
    funds=np.arange(32,dtype=np.float32).reshape(2,16)
    control=missingness_control(funds)
    assert np.all(control[:,:7]==0)
    np.testing.assert_array_equal(control[:,7:],funds[:,7:])
    assert funds[0,1]!=0  # Original array is not mutated.


def test_robustness_rejects_stale_configuration_and_changed_panel(tmp_path):
    from volrisk.robustness import validate_inputs
    from volrisk.io import sha256, write_json
    path=tmp_path/'data/processed/main/panel.pkl'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'original trusted local fixture')
    write_json(path.parent/'manifest.json',{'config':{'horizon':20},'panel_sha256':sha256(path)})
    with pytest.raises(ValueError,match='configuration differs'):
        validate_inputs(tmp_path,{'horizon':5})
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum mismatch'):
        validate_inputs(tmp_path,{'horizon':20})


def test_alternative_equity_preserves_negative_values_and_asof_time():
    from volrisk.fundamentals_v3 import extra_snapshot
    facts=pd.DataFrame([
        ['assets',None,'2020-12-31','2021-02-01','a',100],
        ['parent_equity',None,'2020-12-31','2021-02-01','a',-20],
        ['current_liabilities',None,'2020-12-31','2021-02-01','a',40],
        ['parent_equity',None,'2020-12-31','2022-02-01','b',60],
    ],columns=['concept','start','end','filed','accn','value'])
    for col in ['start','end','filed']:facts[col]=pd.to_datetime(facts[col])
    facts['tag']=facts.concept;facts['priority']=0
    result=extra_snapshot(facts,'2021-06-01','2020-12-31')
    assert result['parent_equity_to_assets']==-.2
    assert result['current_liabilities_to_assets']==.4
    assert np.isnan(result['liability_components_to_assets'])
