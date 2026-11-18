"""Conservative identity restrictions, separate from source text extraction quality."""
import copy

import numpy as np
import pandas as pd


def restrict_spell_disclosures(payload, first_filing):
    result = copy.deepcopy(payload)
    for taxonomy in result.get("facts", {}).values():
        for concept in taxonomy.values():
            for unit, rows in concept.get("units", {}).items():
                concept["units"][unit] = [r for r in rows if first_filing <= r.get("filed", "") <= "2025-12-31"]
    return result


def current_spell_windows(dates, start, sequence_days=60, longest_return_feature=60):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Unique chronological trading dates required")
    first = int(dates.searchsorted(pd.Timestamp(start)))
    positions = np.arange(len(dates))
    # No pre-spell observations exist in an IPO-only provider cache: use the
    # parent's declared warm-up, rather than adding an arbitrary extra embargo.
    if first == 0:
        return positions >= 0, positions >= 0
    # Earliest price in a60-session sequence ofrv60 is i-119; the first512-rv20
    # point needs a price at i-531. Exclude contamination through derived inputs.
    sequence_offset = sequence_days + longest_return_feature - 1
    foundation_offset = 512 + 20 - 1
    return positions >= first + sequence_offset, positions >= first + foundation_offset
