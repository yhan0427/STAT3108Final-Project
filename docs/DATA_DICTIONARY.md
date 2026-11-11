# Features and availability

All returns use adjusted-close log differences. Same-day OHLC/volume are allowed because the forecast decision is after market close.

| Field | Definition |
|---|---|
| return / abs_return | Current daily log return and absolute value |
| rv5 / rv20 / rv60 | Past 5/20/60 daily returns, population standard deviation times sqrt(252) |
| log_range | log(raw high / raw low) |
| volume_change | Change in log(1 + reported volume); split effects are not removed |
| momentum20 | 20-day adjusted log-price difference |
| market_return / market_rv20 | SPY current log return and trailing 20-day volatility |
| HAR log components | log squared daily return and logs of 5/22-day mean squared returns, floor 1e-12 |
| liabilities_to_assets | Reported total liabilities / assets, same accounting date |
| cash_to_assets | Reported cash and cash equivalents / assets, same accounting date |
| current_ratio | Current assets / current liabilities, same accounting date |
| operating_margin | Trailing-four-quarter operating income / revenue |
| income_to_assets | Trailing-four-quarter net income / assets at latest report end, not average-asset ROA |
| cfo_to_assets | Trailing-four-quarter operating cash flow / assets at latest report end |
| revenue_growth | Trailing-four-quarter revenue / corresponding prior-year trailing revenue - 1 |
| filing_age_days | Calendar days from latest known snapshot filing to forecast origin |

Only positive denominators produce ratios. Missing denominators/tags/periods stay missing. The cash-flow mapping uses total operating cash flow, not continuing-operations cash flow as an unlabelled substitute. A full annual flow at the current period end can supply a trailing flow directly; otherwise four contiguous quarters are constructed. Standalone 60–110 day quarters are used directly; cumulative values require a same-fiscal-start earlier period to derive a standalone quarter. Annual-minus-nine-month values recover quarter four. Incomplete four-quarter histories are missing, never annualized by multiplying an incomplete quarter by four.

Availability uses each fact's SEC `filed` date and excludes same-day facts. Within a historical snapshot only disclosures filed by that snapshot date may be selected. The selected report end is the latest available asset balance. Duration and instant values must match that report end. This avoids combining old assets with new flows. Published comparative facts may be used after they become public, never before. Original units, tags, periods, accession numbers and filing timestamps remain in the raw SEC cache; derived snapshots are saved separately.

Coverage is a diagnostic, not a filter selected on outcomes. Some companies use custom tags not returned as standard concepts in Company Facts. As a result the experiment tests the implemented public-data feature set, not the proposition that all possible fundamental information is useless/useful.
