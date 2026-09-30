"""Actual-date calendar rules; empirical dates are never inferred from seasons."""
from datetime import date, datetime, timedelta


def parse_date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise TypeError("Use an explicit calendar date, not a timestamp")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError("An actual ISO date (YYYY-MM-DD) is required")
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD")
    return result


FALLBACK_ROLLOVER = (8, 1)


def season_of(value: date | str, rollover: tuple[int, int] = FALLBACK_ROLLOVER) -> int:
    """Fallback end-year label (D-016): the year rolls on 1 August by default.

    This is the last resort of ``fpp.data.seasons.assign_season``; verified
    provider competition-season identifiers and documented competition
    periods take precedence. The earlier June 1 rule placed June playoff
    games in the following season. Labels never determine chronological
    eligibility, which uses actual dates and release dates.
    """
    day = parse_date(value)
    month, day_of_month = rollover
    if not (1 <= month <= 12 and 1 <= day_of_month <= 31):
        raise ValueError("Rollover must be a (month, day) pair")
    return day.year + ((day.month, day.day) >= (month, day_of_month))


def forecast_cutoff(season: int) -> date:
    """October 1 of the opening year, for an end-year season label."""
    return date(season - 1, 10, 1)


def training_cutoff(season: int) -> date:
    return forecast_cutoff(season) - timedelta(days=1)


def assumed_release(value: date | str, lag_days: int = 1) -> date:
    """A1 assumption, not an observed publication timestamp."""
    if not isinstance(lag_days, int) or lag_days < 0:
        raise ValueError("lag_days must be a nonnegative integer")
    return parse_date(value) + timedelta(days=lag_days)


def add_years(value: date | str, years: int) -> date:
    day = parse_date(value)
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # February 29 maps to February 28.
        return day.replace(year=day.year + years, day=28)


def age(dob: date | str, on: date | str) -> float:
    birth, day = parse_date(dob), parse_date(on)
    if day < birth:
        raise ValueError("Age cannot be negative")
    return (day - birth).days / 365.2425
