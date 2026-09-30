"""Immutable input records and explicit completeness gates.

A season label never supplies a date. Period boundaries identify the exact
competition/team/horizon scope used for summary reconciliation.
"""
from dataclasses import dataclass
from datetime import date
from math import isfinite
from .calendar import parse_date
from .seasons import SeasonAssignment

BOX_FIELDS = ("a2", "k2", "a3", "k3", "af", "kf", "orb", "drb", "ast", "stl", "blk", "tov", "pf")


def _dates(obj, names):
    for name in names:
        value = getattr(obj, name)
        if value is not None:
            object.__setattr__(obj, name, parse_date(value))


def _nonnegative(value, name):
    if value is not None and (not isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class SeasonPeriod:
    season: int
    start: date
    end: date

    def __post_init__(self):
        _dates(self, ("start", "end"))
        if self.start > self.end:
            raise ValueError("Period start exceeds end")


@dataclass(frozen=True)
class GameRecord:
    player_id: str
    record_id: str
    date: date
    release_date: date
    competition: str
    season: int
    minutes: float
    team_id: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    horizon: str = "full"
    domain: str = "intl"
    appeared: bool | None = None
    starter: bool | None = None
    provider_status: str | None = None
    status_mapping_version: str | None = None
    release_is_assumed: bool = True
    source_version: str = "unspecified"
    participation_conflict: bool = False
    boxscore_counts: tuple[int | None, ...] = ()
    season_assignment: SeasonAssignment | None = None

    def __post_init__(self):
        _dates(self, ("date", "release_date", "period_start", "period_end"))
        # D-016: no global date rule decides the season. The label must be
        # within a year of the game date; an attached assignment record
        # (provider identifier, competition period, or audited fallback)
        # must agree with it, and its method is exposed as season_method.
        if isinstance(self.season, bool) or not isinstance(self.season, int):
            raise ValueError("season must be an integer end-year label")
        if abs(self.season - self.date.year) > 1:
            raise ValueError("Season label is not within a year of the game date")
        if self.season_assignment is not None:
            if not isinstance(self.season_assignment, SeasonAssignment):
                raise TypeError("season_assignment must be a SeasonAssignment")
            if self.season_assignment.season != self.season:
                raise ValueError("Season label contradicts its assignment record")
        if self.release_date < self.date:
            raise ValueError("Game cannot be released before it occurs")
        if (self.period_start is None) != (self.period_end is None):
            raise ValueError("Supply both verified period boundaries or neither")
        if self.period_start is not None and not self.period_start <= self.date <= self.period_end:
            raise ValueError("Game lies outside its declared period")
        if self.domain not in {"intl", "ncaa"} or self.horizon not in {"regular", "full"}:
            raise ValueError("Unknown domain or horizon")
        if self.minutes is None:
            raise ValueError("GameRecord requires observed minutes; unresolved minutes need an upstream field mask")
        _nonnegative(self.minutes, "minutes")
        object.__setattr__(self, "boxscore_counts", tuple(self.boxscore_counts))
        for count in self.boxscore_counts:
            _nonnegative(count, "boxscore count")
        if self.participation_conflict:
            object.__setattr__(self, "appeared", None)
            object.__setattr__(self, "starter", None)
            return
        positive = self.minutes > 0 or any(c is not None and c > 0 for c in self.boxscore_counts)
        if positive and self.appeared is False:
            raise ValueError("Positive minutes contradict nonparticipation")
        if self.starter is True and self.appeared is False:
            raise ValueError("Starter contradicts nonparticipation")
        if positive or self.starter is True:
            object.__setattr__(self, "appeared", True)

    @property
    def season_method(self) -> str:
        """How the season label was established; 'unverified_label' if unknown."""
        return self.season_assignment.method if self.season_assignment else "unverified_label"

    @property
    def scope(self):
        return (self.player_id, self.competition, self.season, self.team_id,
                self.period_start, self.period_end, self.horizon)


@dataclass(frozen=True)
class SummaryRecord:
    player_id: str
    record_id: str
    period_start: date
    period_end: date
    release_date: date
    competition: str
    season: int
    games_played: int
    minutes: float
    team_id: str | None = None
    horizon: str = "full"
    complete: bool = False
    domain: str = "intl"
    release_is_assumed: bool = True
    source_version: str = "unspecified"

    def __post_init__(self):
        _dates(self, ("period_start", "period_end", "release_date"))
        if self.period_start > self.period_end or self.release_date < self.period_end:
            raise ValueError("Invalid summary period or release date")
        if self.domain not in {"intl", "ncaa"} or self.horizon not in {"regular", "full"}:
            raise ValueError("Unknown domain or horizon")
        if isinstance(self.games_played, bool) or not isinstance(self.games_played, int) or self.games_played < 0:
            raise ValueError("games_played must be a nonnegative integer")
        _nonnegative(self.minutes, "minutes")
        if self.games_played == 0 and self.minutes > 0:
            raise ValueError("Positive minutes contradict zero appearances")

    @property
    def scope(self):
        return (self.player_id, self.competition, self.season, self.team_id,
                self.period_start, self.period_end, self.horizon)


@dataclass(frozen=True)
class ReconciliationTolerance:
    """Must be supplied explicitly; the package chooses no production tolerance."""
    version: str
    games: int
    minutes: float

    def __post_init__(self):
        if not self.version or isinstance(self.games, bool) or not isinstance(self.games, int) or self.games < 0:
            raise ValueError("A version and nonnegative game tolerance are required")
        if self.minutes is None:
            raise ValueError("A numeric minute tolerance is required")
        _nonnegative(self.minutes, "minutes tolerance")


@dataclass(frozen=True)
class CoverageEvidence:
    scope: tuple
    route: str
    game_ids: tuple[str, ...]
    summary_id: str | None
    observed_appearances: int | None
    observed_minutes: float
    reported_appearances: int | None = None
    reported_minutes: float | None = None
    appearance_coverage: float | None = None
    minute_coverage: float | None = None
    tolerance_version: str | None = None
    audit_reason: str | None = None


@dataclass(frozen=True)
class RoutedHistory:
    games: tuple[GameRecord, ...] = ()
    summaries: tuple[SummaryRecord, ...] = ()
    coverage_summaries: tuple[SummaryRecord, ...] = ()
    coverage: tuple[CoverageEvidence, ...] = ()
    augmentation_deleted_ids: tuple[str, ...] = ()

    @property
    def nonempty(self):
        return bool(self.games or self.summaries)

    @property
    def game_anchor(self):
        return max((g.date for g in self.games), default=None)

    @property
    def source_anchor(self):
        if not self.nonempty:
            return None
        return max([g.date for g in self.games] +
                   [s.period_end for s in self.summaries + self.coverage_summaries])

    @property
    def identity(self):
        # Metadata affects eligibility and the source clock, even when not encoded.
        game = tuple(sorted((g.record_id, g.date.isoformat(), g.release_date.isoformat(), g.source_version) for g in self.games))
        def summaries(items):
            return tuple(sorted((s.record_id, s.period_start.isoformat(), s.period_end.isoformat(), s.release_date.isoformat(), s.source_version) for s in items))
        return game, summaries(self.summaries), summaries(self.coverage_summaries)


@dataclass(frozen=True)
class ForecastWindow:
    player_id: str
    target_season: int
    direction: str
    requested_k: tuple[int, ...]
    intended_start: date | None
    intended_end: date
    cutoff: date
    partial: bool
    history: RoutedHistory

    @property
    def source_anchor(self):
        return self.history.source_anchor


@dataclass(frozen=True)
class OutcomeUnit:
    player_id: str
    season: int
    horizon: str
    period_start: date
    period_end: date
    release_date: date
    scheduled_games: int
    games_played: int | None = None
    starts: int | None = None
    minutes: float | None = None
    counts: tuple[int, ...] | None = None
    roster_confirmed: bool = True
    completeness: str = "INCOMPLETE_COVERAGE"
    participation_verified: bool = False
    no_participation_verified: bool = False
    minutes_provenance: str | None = None
    provenance_verified: bool = False
    evidence: str | None = None

    def __post_init__(self):
        _dates(self, ("period_start", "period_end", "release_date"))
        if self.horizon not in {"regular", "full"} or self.period_start > self.period_end:
            raise ValueError("Invalid outcome horizon")
        if self.release_date < self.period_end:
            raise ValueError("Complete season outcomes cannot be released before period end")
        for name in ("scheduled_games", "games_played", "starts"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        _nonnegative(self.minutes, "minutes")
        if self.games_played is not None and self.games_played > self.scheduled_games:
            raise ValueError("G exceeds S")
        if self.starts is not None and self.games_played is not None and self.starts > self.games_played:
            raise ValueError("J exceeds G")
        if self.counts is not None:
            object.__setattr__(self, "counts", tuple(self.counts))
            if len(self.counts) != len(BOX_FIELDS) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.counts):
                raise ValueError("counts must contain 13 nonnegative integers in BOX_FIELDS order")
            if any(self.counts[k] > self.counts[a] for a, k in ((0, 1), (2, 3), (4, 5))):
                raise ValueError("Made shots exceed attempts")
            if self.games_played is not None and self.counts[-1] > 5 * self.games_played:
                raise ValueError("Personal fouls exceed 5G")
        if self.games_played == 0 and (self.minutes not in (None, 0) or self.starts not in (None, 0) or (self.counts and any(self.counts))):
            raise ValueError("Nonzero outcomes contradict G=0")

    @property
    def exclusion_reasons(self):
        reasons = []
        if not self.roster_confirmed:
            reasons.append("NO_CONFIRMED_ROSTER")
        if self.completeness != "COMPLETE" or not self.evidence:
            reasons.append("INCOMPLETE_COVERAGE")
        if any(value is None for value in (self.games_played, self.starts, self.minutes, self.counts)):
            reasons.append("UNRESOLVED_OUTCOME")
        if not self.participation_verified:
            reasons.append("UNVERIFIED_PARTICIPATION")
        if self.games_played == 0 and not self.no_participation_verified:
            reasons.append("UNVERIFIED_NONPARTICIPATION")
        if not self.minutes_provenance or not self.provenance_verified:
            reasons.append("UNVERIFIED_MINUTES_PROVENANCE")
        return tuple(reasons)

    @property
    def likelihood_eligible(self):
        return not self.exclusion_reasons


@dataclass(frozen=True)
class ParticipationEvidence:
    appeared: bool | None
    starter: bool | None
    reason: str
    conflict: bool = False


def classify_participation(*, minutes, counts=(), provider_status=None, status_map=None):
    """Combine observed positive evidence and a caller-supplied verified map.

    Map entries must be 'starter', 'played', or 'dnp'. Unknown provider
    values remain unknown; raw zero minutes never establishes nonparticipation.
    Conflicting provider DNP and positive statistics remain unresolved.
    """
    _nonnegative(minutes, "minutes")
    for count in counts:
        _nonnegative(count, "boxscore count")
    mapped = (status_map or {}).get(provider_status)
    if mapped not in {None, "starter", "played", "dnp"}:
        raise ValueError("Unknown value in provider participation mapping")
    positive = (minutes is not None and minutes > 0) or any(c is not None and c > 0 for c in counts)
    if mapped == "dnp" and positive:
        return ParticipationEvidence(None, None, "CONTRADICTORY_PROVIDER_DNP", True)
    if mapped == "starter":
        return ParticipationEvidence(True, True, "PROVIDER_STARTER")
    if mapped == "played":
        return ParticipationEvidence(True, None, "PROVIDER_PARTICIPATION")
    if mapped == "dnp":
        return ParticipationEvidence(False, False, "PROVIDER_NONPARTICIPATION")
    if positive:
        return ParticipationEvidence(True, None, "POSITIVE_OBSERVED_STATISTICS")
    return ParticipationEvidence(None, None, "UNRESOLVED_PARTICIPATION")
