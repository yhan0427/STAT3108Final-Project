# Original-filing financial spot checks

Fixed original-report quality set; no documents selected by check success or forecasting outcomes. Original inline XBRL values are parsed independently from the Company Facts feature builder, with explicit CIK, non-dimensional context, USD unit, reporting period, scale and sign.

| Check | Status | Count |
|---|---|---:|
| original_ratio_vs_snapshot | match | 70 |
| original_vs_api | match | 81 |
| original_vs_api | original_field_unavailable | 7 |

Original documents without supported inline facts remain listed in the JSON audit. Missing or ambiguous tags are not invented. These checks cover the named fields and documents only; they do not certify every historical feature, issuer transition, custom taxonomy or immutable point-in-time archive. Any mismatch must be investigated before those fields enter a new fit.
