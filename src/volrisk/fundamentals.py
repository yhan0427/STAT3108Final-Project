"""Build filing-date snapshots without importing subsequently disclosed values."""
import numpy as np
import pandas as pd

TAGS = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "operating_income": ["OperatingIncomeLoss"],
    "income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
}
FUND_FEATURES = ["liabilities_to_assets", "cash_to_assets", "current_ratio", "operating_margin", "income_to_assets", "cfo_to_assets", "revenue_growth", "filing_age_days"]


def flatten_facts(payload, tags=None):
    rows = []
    gaap = payload.get("facts", {}).get("us-gaap", {})
    for concept, candidates in (TAGS if tags is None else tags).items():
        for priority, tag in enumerate(candidates):
            for fact in gaap.get(tag, {}).get("units", {}).get("USD", []):
                if fact.get("form") not in {"10-K", "10-Q", "10-K/A", "10-Q/A"} or "filed" not in fact:
                    continue
                rows.append({"concept": concept, "tag": tag, "priority": priority, "start": fact.get("start"), "end": fact["end"], "filed": fact["filed"], "accn": fact["accn"], "value": fact["val"]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No supported SEC USD accounting facts")
    for col in ["start", "end", "filed"]:
        frame[col] = pd.to_datetime(frame[col])
    return frame


def asof_periods(facts, date):
    available = facts.loc[(facts.filed <= date) & (facts.end <= date)].copy()
    # First choose latest disclosure for each individual tag/period, then preferred
    # concept mapping. No present-day restatement is backfilled into older snapshots.
    available = available.sort_values(["filed", "accn"]).drop_duplicates(["concept", "tag", "start", "end"], keep="last")
    return available.sort_values("priority").drop_duplicates(["concept", "start", "end"], keep="first")


def trailing_flow(periods, concept, end):
    rows = periods.loc[(periods.concept == concept) & periods.start.notna()].copy()
    if rows.empty:
        return np.nan
    rows["days"] = (rows.end - rows.start).dt.days + 1
    annual = rows.loc[(rows.end == end) & rows.days.between(330, 380)]
    if not annual.empty:
        return float(annual.sort_values("filed").iloc[-1].value)
    quarters = {}
    for period_end, group in rows.groupby("end"):
        direct = group.loc[group.days.between(60, 110)]
        if not direct.empty:
            quarters[period_end] = float(direct.sort_values("filed").iloc[-1].value)
            continue
        for current in group.sort_values("days").itertuples():
            if not 110 < current.days <= 380:
                continue
            previous = rows.loc[(rows.start == current.start) & ((current.end - rows.end).dt.days.between(60, 110))]
            if not previous.empty:
                prior = previous.sort_values("end").iloc[-1]
                quarters[period_end] = float(current.value - prior.value)
                break
    ends = sorted([d for d in quarters if d <= end])[-4:]
    if len(ends) != 4 or ends[-1] != end:
        return np.nan
    if not all(60 <= (b - a).days <= 110 for a, b in zip(ends, ends[1:])):
        return np.nan
    return sum(quarters[d] for d in ends)


def safe_ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def make_snapshots(payload, earliest="2013-01-01", latest="2025-12-31"):
    facts = flatten_facts(payload)
    records = []
    for date in sorted(facts.loc[facts.filed.between(earliest, latest), "filed"].unique()):
        date = pd.Timestamp(date)
        known = asof_periods(facts, date)
        asset_rows = known.loc[(known.concept == "assets") & known.start.isna()]
        if asset_rows.empty:
            continue
        end = asset_rows.end.max()
        if (date - end).days > 180:
            continue

        def instant(concept):
            found = known.loc[(known.concept == concept) & known.start.isna() & (known.end == end)]
            return float(found.iloc[-1].value) if not found.empty else np.nan

        assets = instant("assets")
        revenue = trailing_flow(known, "revenue", end)
        older_ends = known.loc[(known.concept == "revenue") & ((end - known.end).dt.days.between(350, 380)), "end"]
        prior_revenue = trailing_flow(known, "revenue", older_ends.max()) if not older_ends.empty else np.nan
        event = {"filing_date": date, "report_end": end, "liabilities_to_assets": safe_ratio(instant("liabilities"), assets), "cash_to_assets": safe_ratio(instant("cash"), assets), "current_ratio": safe_ratio(instant("current_assets"), instant("current_liabilities")), "operating_margin": safe_ratio(trailing_flow(known, "operating_income", end), revenue), "income_to_assets": safe_ratio(trailing_flow(known, "income", end), assets), "cfo_to_assets": safe_ratio(trailing_flow(known, "cfo", end), assets), "revenue_growth": safe_ratio(revenue, prior_revenue) - 1}
        records.append(event)
    if not records:
        raise ValueError("No usable point-in-time fundamental snapshots")
    return pd.DataFrame(records).sort_values("filing_date")


def align_snapshots(dates, snapshots):
    left = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date")
    aligned = pd.merge_asof(left, snapshots.sort_values("filing_date"), left_on="date", right_on="filing_date", direction="backward", allow_exact_matches=False)
    aligned["filing_age_days"] = (aligned.date - aligned.filing_date).dt.days.astype(float)
    # Stale snapshots are missing, not silently treated as fresh financials.
    stale = aligned.filing_age_days > 180
    aligned.loc[stale, [c for c in FUND_FEATURES if c != "filing_age_days"]] = np.nan
    return aligned
