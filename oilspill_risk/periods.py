"""Date-window helpers for OSCAR downloads and exports."""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta


def _parse_utc_date(value: str) -> date:
    """Parse the UTC timestamp format used by the PO.DAAC downloader."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").date()


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + delta
    return total // 12, total % 12 + 1


def seasonal_periods(
    start_date: str,
    end_date: str,
    season_length_months: int = 3,
) -> list[tuple[date, date, str]]:
    """Return complete month-aligned periods inside a UTC date range.

    Incomplete leading/trailing windows are skipped so each period represents
    exactly ``season_length_months`` calendar months.
    """
    s_date = _parse_utc_date(start_date)
    e_date = _parse_utc_date(end_date)
    if s_date > e_date:
        raise ValueError(f"start_date must be before end_date: {start_date} > {end_date}")

    span = max(1, min(12, int(season_length_months)))
    periods: list[tuple[date, date, str]] = []
    year, month = s_date.year, s_date.month
    window_idx = 1

    while True:
        period_start = date(year, month, 1)
        end_year, end_month = _add_months(year, month, span)
        period_end = date(end_year, end_month, 1) - timedelta(days=1)

        if period_start > e_date:
            break

        if period_start >= s_date and period_end <= e_date:
            period_id = f"S{window_idx}_{period_start:%Y%m}_{period_end:%Y%m}"
            periods.append((period_start, period_end, period_id))
            window_idx += 1
        else:
            warnings.warn(
                f"Skipping incomplete season {window_idx}: "
                f"[{period_start:%Y-%m-%d} to {period_end:%Y-%m-%d}] vs "
                f"range [{s_date:%Y-%m-%d} to {e_date:%Y-%m-%d}]",
                stacklevel=2,
            )

        year, month = _add_months(year, month, span)

    return periods
