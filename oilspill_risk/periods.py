"""Date-window helpers for OSCAR downloads and exports."""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from typing import Literal

PeriodFrequency = Literal["daily", "monthly", "seasonal", "season", "yearly"]


def _parse_utc_date(value: str) -> date:
    """Parse the UTC timestamp format used by the PO.DAAC downloader."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").date()


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + delta
    return total // 12, total % 12 + 1


def _month_end(year: int, month: int, span: int = 1) -> date:
    end_year, end_month = _add_months(year, month, span)
    return date(end_year, end_month, 1) - timedelta(days=1)


def _warn_incomplete_period(
    period_id: str,
    period_start: date,
    period_end: date,
    s_date: date,
    e_date: date,
) -> None:
    warnings.warn(
        f"Skipping incomplete period {period_id}: "
        f"[{period_start:%Y-%m-%d} to {period_end:%Y-%m-%d}] vs "
        f"range [{s_date:%Y-%m-%d} to {e_date:%Y-%m-%d}]",
        stacklevel=2,
    )


def _validate_date_range(start_date: str, end_date: str) -> tuple[date, date]:
    s_date = _parse_utc_date(start_date)
    e_date = _parse_utc_date(end_date)
    if s_date > e_date:
        raise ValueError(f"start_date must be before end_date: {start_date} > {end_date}")
    return s_date, e_date


def analysis_periods(
    start_date: str,
    end_date: str,
    frequency: PeriodFrequency = "seasonal",
    season_length_months: int = 3,
) -> list[tuple[date, date, str]]:
    """Return period windows with calendar labels that repeat across years.

    Frequencies map to period IDs as follows:
    - ``daily``: one period per day, labeled ``DMMDD`` (for example ``D0101``).
    - ``monthly``: complete calendar months, labeled ``M01``..``M12``.
    - ``seasonal``/``season``: complete month windows starting at ``start_date``'s
      month, labeled ``S1``..``Sn`` and repeated in later years.
    - ``yearly``: complete 12-month windows starting at ``start_date``'s month,
      labeled ``Y1`` and repeated in later years.
    """
    s_date, e_date = _validate_date_range(start_date, end_date)
    normalized_frequency = "seasonal" if frequency == "season" else frequency

    if normalized_frequency == "daily":
        periods: list[tuple[date, date, str]] = []
        current = s_date
        while current <= e_date:
            periods.append((current, current, f"D{current:%m%d}"))
            current += timedelta(days=1)
        return periods

    if normalized_frequency == "monthly":
        span = 1
        prefix = "M"
        windows_per_year = 12
    elif normalized_frequency == "seasonal":
        span = max(1, min(12, int(season_length_months)))
        if 12 % span != 0:
            raise ValueError("season_length_months must divide 12 for repeating seasonal labels")
        prefix = "S"
        windows_per_year = 12 // span
    elif normalized_frequency == "yearly":
        span = 12
        prefix = "Y"
        windows_per_year = 1
    else:
        raise ValueError(
            "frequency must be one of: daily, monthly, seasonal, season, yearly"
        )

    periods = []
    year, month = s_date.year, s_date.month
    window_idx = 0

    while True:
        period_start = date(year, month, 1)
        period_end = _month_end(year, month, span)
        if period_start > e_date:
            break

        label_idx = (window_idx % windows_per_year) + 1
        if prefix == "M":
            period_id = f"M{period_start.month:02d}"
        else:
            period_id = f"{prefix}{label_idx}"

        if period_start >= s_date and period_end <= e_date:
            periods.append((period_start, period_end, period_id))
        else:
            _warn_incomplete_period(period_id, period_start, period_end, s_date, e_date)

        year, month = _add_months(year, month, span)
        window_idx += 1

    return periods


def seasonal_periods(
    start_date: str,
    end_date: str,
    season_length_months: int = 3,
    frequency: PeriodFrequency = "seasonal",
) -> list[tuple[date, date, str]]:
    """Return complete analysis periods inside a UTC date range.

    This keeps the historical function name used by the download workflow while
    supporting daily, monthly, seasonal, and yearly period definitions. Period
    labels repeat across years so downstream aggregation can average all matching
    labels together (for example, all Januaries as ``M01`` or all Jan-Mar windows
    as ``S1`` when the analysis starts in January).
    """
    return analysis_periods(
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        season_length_months=season_length_months,
    )
