import numpy as np
import pandas as pd
import pytest

from volrisk.financial_coverage import (
    AUDIT_FEATURES, coverage_by_year, restrict_disclosures,
)


def test_restrict_by_disclosure_not_report_period_and_do_not_mutate_cache():
    rows = [{"filed": "2025-02-01", "end": "2024-12-31", "val": 10},
            {"filed": "2026-02-01", "end": "2024-12-31", "val": 999}]
    payload = {"facts": {"us-gaap": {"Assets": {"units": {"USD": rows}}}}}
    clean = restrict_disclosures(payload)
    assert clean["facts"]["us-gaap"]["Assets"]["units"]["USD"] == rows[:1]
    assert len(payload["facts"]["us-gaap"]["Assets"]["units"]["USD"]) == 2


def test_coverage_keeps_missing_denominators_excludes_same_day_and_stales_extras():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-08-01"])
    origins = pd.DataFrame({"date": dates, "split": "test"})
    snapshot = {f: [1.] for f in AUDIT_FEATURES if f != "filing_age_days"}
    snapshot.update(filing_date=[dates[0]], report_end=[pd.Timestamp("2024-12-31")])
    result = coverage_by_year(origins, pd.DataFrame(snapshot)).set_index("feature")
    assert result.loc["parent_equity_to_assets", "eligible_origins"] == 3
    assert result.loc["parent_equity_to_assets", "available_origins"] == 1
    assert result.loc["parent_equity_to_assets", "coverage"] == pytest.approx(1 / 3)
    assert result.loc["filing_age_days", "available_origins"] == 2
    with pytest.raises(ValueError, match="without outcomes"):
        coverage_by_year(origins.assign(target=np.ones(3)), pd.DataFrame(snapshot))
