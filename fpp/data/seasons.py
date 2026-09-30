"""Competition-aware season assignment (D-016).

A season label indexes records for grouping; it is not a date, and
chronological eligibility always uses actual dates and release dates.
Assignment order: a verified provider competition-season identifier, then
the competition's documented period boundaries, then an audited fallback
rollover that applies only when neither exists.

The earlier global June 1 rule (mapping version ``season-map-v1``) placed
European playoff games from June in the following season; cached provider
pages hold 25,940 June game rows. Those rows are a review population, not a
proof that every June record was misclassified. The fallback rollover is now
1 August. Every assignment retains the original label, the assigned season,
the method, and the mapping version so relabelling is auditable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .calendar import parse_date, season_of

FALLBACK_ROLLOVER = (8, 1)
MAPPING_VERSION = "season-map-v2"
METHODS = ("provider_identifier", "competition_period", "fallback_rollover")


def _season_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1900:
        raise ValueError(f"{name} must be an integer end-year season label")
    return value


@dataclass(frozen=True)
class CompetitionSeasonPeriod:
    """Documented period of one competition-season, from the rules catalog."""
    competition: str
    season: int
    start: date
    end: date

    def __post_init__(self):
        if not self.competition:
            raise ValueError("A competition identifier is required")
        _season_int(self.season, "season")
        object.__setattr__(self, "start", parse_date(self.start))
        object.__setattr__(self, "end", parse_date(self.end))
        if self.start > self.end:
            raise ValueError("Competition period start exceeds end")

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


@dataclass(frozen=True)
class SeasonAssignment:
    season: int
    method: str
    mapping_version: str = MAPPING_VERSION
    original_label: int | None = None

    def __post_init__(self):
        _season_int(self.season, "season")
        if self.method not in METHODS:
            raise ValueError(f"Unknown season assignment method: {self.method!r}")
        if not self.mapping_version:
            raise ValueError("A mapping version is required")
        if self.original_label is not None:
            _season_int(self.original_label, "original_label")

    @property
    def changed(self) -> bool:
        return self.original_label is not None and self.original_label != self.season

    @property
    def verified(self) -> bool:
        """True when the season comes from provider or period evidence, not the fallback."""
        return self.method != "fallback_rollover"


def assign_season(*, date, competition, original_label=None, provider_season=None,
                  periods: Iterable[CompetitionSeasonPeriod] = (),
                  rollover=FALLBACK_ROLLOVER, mapping_version=MAPPING_VERSION) -> SeasonAssignment:
    """Resolve one record's season; the fallback is never applied silently."""
    day = parse_date(date)
    if not competition:
        raise ValueError("A competition identifier is required")
    if provider_season is not None:
        return SeasonAssignment(_season_int(provider_season, "provider_season"),
                                "provider_identifier", mapping_version, original_label)
    matches = [p for p in periods if p.competition == competition and p.contains(day)]
    if len(matches) > 1:
        raise ValueError(f"Overlapping period definitions for {competition!r} on {day.isoformat()}")
    if matches:
        return SeasonAssignment(matches[0].season, "competition_period", mapping_version, original_label)
    return SeasonAssignment(season_of(day, rollover), "fallback_rollover", mapping_version, original_label)


def audit_assignments(assignments: Iterable[SeasonAssignment]) -> dict:
    """Summarize methods and relabelling for the data-contract audit."""
    items = list(assignments)
    by_method = Counter(a.method for a in items)
    changed = [a for a in items if a.changed]
    return {
        "total": len(items),
        "by_method": {m: by_method.get(m, 0) for m in METHODS},
        "fallback": by_method.get("fallback_rollover", 0),
        "changed": len(changed),
        "changed_pairs": sorted(Counter((a.original_label, a.season) for a in changed).items()),
        "mapping_versions": sorted({a.mapping_version for a in items}),
    }
