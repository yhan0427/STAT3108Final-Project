import pandas as pd

from volrisk.entity_spells import current_spell_windows, restrict_spell_disclosures


def test_feature_history_embargo_covers_both_sequence_and_rolling_input():
    dates = pd.bdate_range("2019-01-01", periods=900)
    start = dates[100]
    sequence, foundation = current_spell_windows(dates, start)
    assert not sequence[218] and sequence[219]
    assert not foundation[630] and foundation[631]
    assert dates[219-119] >= start
    assert dates[631-531] >= start
    ipo_sequence, ipo_foundation = current_spell_windows(dates, dates[0])
    assert ipo_sequence.all() and ipo_foundation.all()


def test_later_disclosed_comparatives_survive_but_old_entity_disclosures_do_not():
    old = {"filed": "2019-02-01", "end": "2018-12-31", "val": 100}
    recast = {"filed": "2021-02-01", "end": "2018-12-31", "val": 40}
    payload = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [old, recast]}}}}}
    restricted = restrict_spell_disclosures(payload, "2020-07-01")
    assert restricted["facts"]["us-gaap"]["Assets"]["units"]["USD"] == [recast]
    assert len(payload["facts"]["us-gaap"]["Assets"]["units"]["USD"]) == 2
