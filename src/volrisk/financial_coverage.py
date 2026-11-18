"""Coverage diagnostics only: no labels, fitting, or outcome-based feature selection."""
import copy

import numpy as np
import pandas as pd

from .fundamentals import (
    FUND_FEATURES, TAGS, align_snapshots, asof_periods, flatten_facts,
    make_snapshots, safe_ratio,
)

EXTRA_TAGS = {"parent_equity": ["StockholdersEquity"]}
AUDIT_FEATURES = list(dict.fromkeys([
    *FUND_FEATURES, "parent_equity_to_assets", "current_liabilities_to_assets",
]))
CONTENT_FEATURES = [f for f in AUDIT_FEATURES if f != "filing_age_days"]
CUTOFF = "2025-12-31"


def restrict_disclosures(payload, cutoff=CUTOFF):
    """Filter every taxonomy/unit before any feature computation, not by period end."""
    result = copy.deepcopy(payload)
    for taxonomy in result.get("facts", {}).values():
        for concept in taxonomy.values():
            for unit, facts in concept.get("units", {}).items():
                concept["units"][unit] = [
                    r for r in facts
                    if r.get("filed") and r["filed"] <= cutoff
                    and r.get("end", "9999-12-31") <= cutoff
                ]
    return result


def audit_snapshots(payload):
    """Retain legacy feature definitions; audit alternatives without replacing them."""
    payload = restrict_disclosures(payload)
    snapshots = make_snapshots(payload, latest=CUTOFF)
    facts = flatten_facts(payload, {**TAGS, **EXTRA_TAGS})
    additions = []
    for row in snapshots.itertuples():
        known = asof_periods(facts, row.filing_date)
        def instant(concept):
            found = known.loc[
                known.concept.eq(concept) & known.start.isna()
                & known.end.eq(row.report_end)
            ]
            return float(found.iloc[-1].value) if len(found) else np.nan
        assets = instant("assets")
        additions.append({
            "parent_equity_to_assets": safe_ratio(instant("parent_equity"), assets),
            "current_liabilities_to_assets": safe_ratio(instant("current_liabilities"), assets),
        })
    return pd.concat([snapshots.reset_index(drop=True), pd.DataFrame(additions)], axis=1)


def coverage_by_year(origins, snapshots):
    """Count finite values on eligible 60-day origins, including every missing row."""
    forbidden = {"target", "target_end", "original_target", "return"}
    if forbidden.intersection(origins.columns):
        raise ValueError("Coverage input must contain metadata only, without outcomes")
    origins = origins.sort_values("date").reset_index(drop=True)
    aligned = align_snapshots(origins.date, snapshots).reset_index(drop=True)
    aligned.loc[aligned.filing_age_days.gt(180), CONTENT_FEATURES] = np.nan
    if (aligned.filing_date.dropna() >= aligned.loc[aligned.filing_date.notna(), "date"]).any():
        raise ValueError("Same-day or future filing entered coverage")
    rows = []
    for year, positions in origins.groupby(origins.date.dt.year).groups.items():
        group = aligned.loc[positions]
        for feature in AUDIT_FEATURES:
            values = group[feature].to_numpy(dtype=float)
            count = int(np.isfinite(values).sum())
            rows.append({"year": int(year), "feature": feature,
                         "eligible_origins": len(group), "available_origins": count,
                         "coverage": count / len(group)})
    return pd.DataFrame(rows)
