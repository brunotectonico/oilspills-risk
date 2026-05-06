"""Periods to be analyzed - used for downloading and for exporting"""

from __future__ import annotations
import warnings

from datetime import date, datetime
import pandas as pd

def seasonal_periods(start_date: str, end_date: str, season_length_months: int = 3) -> list[tuple[date, date, str]]:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    s_date = datetime.strptime(start_date, fmt).date()
    e_date = datetime.strptime(end_date, fmt).date()

    span = max(1, min(12, season_length_months))
    periods: list[tuple[date, date, str]] = []

    year, month = s_date.year, s_date.month

    def _add_months(y: int, m: int, delta: int) -> tuple[int, int]:
        total = y * 12 + (m - 1) + delta
        return total // 12, total % 12 + 1

    idx = 1
    while True:
        period_start = date(year, month, 1)
        end_year, end_month = _add_months(year, month, span)  # end is exclusive
        period_end = date(end_year, end_month, 1) - pd.offsets.Day(1)
        period_end = period_end.date()

        if period_start > e_date:
            break

        if period_start >= s_date and period_end <= e_date:
            pid = f"S{idx}_{period_start:%Y%m}_{period_end:%Y%m}"
            periods.append((period_start, period_end, pid))
            idx += 1
        else:
            warnings.warn(
                f"Skipping incomplete season {idx}: [{period_start:%Y-%m} to {period_end:%Y-%m}] "
                f"vs range [{s_date:%Y-%m-%d} to {e_date:%Y-%m-%d}]"
            )

        year, month = _add_months(year, month, span)

    return periods