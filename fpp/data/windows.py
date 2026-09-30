"""Eligibility before routing, atomic summaries, and post-clip deduplication."""
from collections import defaultdict
from dataclasses import replace
from datetime import date, timedelta
from .calendar import parse_date, add_years
from .records import (CoverageEvidence, ForecastWindow, RoutedHistory,
                      ReconciliationTolerance)


def _unique(records):
    by_id = {}
    for record in records:
        if record.record_id in by_id and record != by_id[record.record_id]:
            raise ValueError(f"Conflicting duplicate source ID: {record.record_id}")
        by_id[record.record_id] = record
    return tuple(by_id.values())


def _ratio(numerator, denominator):
    return None if numerator is None or denominator in (None, 0) else numerator / denominator


def route_records(games, summaries, tolerance: ReconciliationTolerance | None = None):
    """Route already eligible records; callers must apply cutoffs first.

    Coverage comparisons require exact scope and a registered tolerance. A
    summary overlapping existing games can never become a second encoder input.
    """
    games, summaries = _unique(games), _unique(summaries)
    groups = defaultdict(list)
    for game in games:
        groups[game.scope].append(game)
    complete = {}
    for summary in summaries:
        if summary.complete:
            if summary.scope in complete:
                raise ValueError("Multiple complete summaries for the same scope require upstream reconciliation")
            complete[summary.scope] = summary
    # Two complete summaries of one player/competition/season/team whose periods overlap (a regular-season
    # line inside a full-season line) would double count that season; a canonical scope must be chosen upstream.
    items = list(complete.values())
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            same = (a.player_id, a.competition, a.season, a.team_id) == (b.player_id, b.competition, b.season, b.team_id)
            if same and a.period_start <= b.period_end and b.period_start <= a.period_end:
                raise ValueError(f"Overlapping complete summaries {a.record_id!r} and {b.record_id!r} require a canonical scope")
    encoded, coverage_summaries, evidence = [], [], []
    for scope, items in groups.items():
        items = sorted(items, key=lambda g: (g.date, g.record_id))
        observed_g = None if any(g.appeared is None for g in items) else sum(g.appeared for g in items)
        observed_m = sum(g.minutes for g in items)
        summary = complete.get(scope)
        status, reason = "games_unknown", None
        if summary is not None and tolerance is not None:
            coverage_summaries.append(summary)
            if observed_g is None:
                reason = "UNRESOLVED_PARTICIPATION"
            else:
                dg, dm = summary.games_played - observed_g, summary.minutes - observed_m
                if abs(dg) <= tolerance.games and abs(dm) <= tolerance.minutes:
                    status = "games_complete"
                elif dg >= -tolerance.games and dm >= -tolerance.minutes and dg > tolerance.games:
                    status = "games_partial"
                else:
                    status, reason = "games_unreconciled", "INCONSISTENT_REPORTED_EXPOSURE"
            evidence.append(CoverageEvidence(scope, status, tuple(g.record_id for g in items),
                summary.record_id, observed_g, observed_m, summary.games_played, summary.minutes,
                _ratio(observed_g, summary.games_played), _ratio(observed_m, summary.minutes),
                tolerance.version, reason))
        else:
            reason = "NO_REGISTERED_TOLERANCE" if summary is not None else "NO_ADMISSIBLE_MATCHING_SUMMARY"
            evidence.append(CoverageEvidence(scope, status, tuple(g.record_id for g in items),
                                              None, observed_g, observed_m, audit_reason=reason))
    for scope, summary in complete.items():
        # Missing or conflicting period/horizon definitions do not license
        # double counting the same player/competition/team season.
        overlapping_games = [g for g in games if (g.player_id, g.competition, g.season, g.team_id) ==
                             (summary.player_id, summary.competition, summary.season, summary.team_id)
                             and ((g.period_start is not None and g.period_start <= summary.period_end
                                   and summary.period_start <= g.period_end)
                                  or summary.period_start <= g.date <= summary.period_end)]
        if not overlapping_games:
            encoded.append(summary)
            evidence.append(CoverageEvidence(scope, "summary_only", (), summary.record_id,
                None, 0.0, summary.games_played, summary.minutes))
    return RoutedHistory(tuple(sorted(games, key=lambda g: (g.date, g.record_id))),
                         tuple(sorted(encoded, key=lambda s: s.record_id)),
                         tuple(sorted(coverage_summaries, key=lambda s: s.record_id)), tuple(evidence))


def build_windows(*, player_id, target_season, target_start, target_end,
                  forecast_cutoff, training_cutoff, direction, games=(), summaries=(),
                  season_periods=(), heldout_players=(), k_values=(1, 2, 3),
                  max_years=4, tolerance=None, label_release_date=None):
    """Construct one player's forward or reconstruction training windows.

    Pass training_cutoff=None only for inference. Reconstruction requires a
    training cutoff. Explicit season periods define intended reconstruction
    windows and partial flags; their dates are never inferred from game IDs.
    """
    if player_id in set(heldout_players):
        return ()
    start, end, forecast = map(parse_date, (target_start, target_end, forecast_cutoff))
    if start > end or direction not in {"-", "+"}:
        raise ValueError("Invalid target boundaries or direction")
    train = None if training_cutoff is None else parse_date(training_cutoff)
    if train is not None and label_release_date is not None and parse_date(label_release_date) > train:
        return ()
    if direction == "+" and train is None:
        raise ValueError("Reconstruction is training-only and needs a training cutoff")
    cutoff = min(forecast, train) if direction == "-" and train else (forecast if direction == "-" else train)
    # Reconstruction horizon: ``max_years`` after the season end, or, when None, every game released by the
    # training cutoff (D-038); the far upper bound keeps the partial-season flag meaningful when clipped.
    if direction == "-":
        upper = start - timedelta(days=1)
    else:
        upper = add_years(end, int(max_years)) if max_years is not None else date(9999, 12, 31)
    lower = None if direction == "-" else end + timedelta(days=1)
    eligible_g = tuple(g for g in _unique(games) if g.player_id == player_id and g.domain == "intl"
                       and g.release_date <= cutoff and g.date <= min(upper, cutoff)
                       and (lower is None or g.date >= lower))
    eligible_s = tuple(s for s in _unique(summaries) if s.player_id == player_id and s.domain == "intl"
                       and s.release_date <= cutoff and s.period_end <= min(upper, cutoff)
                       and (lower is None or s.period_start >= lower))
    if direction == "-":
        history = route_records(eligible_g, eligible_s, tolerance)
        return (ForecastWindow(player_id, target_season, direction, (), None, upper, cutoff,
                               False, history),) if history.nonempty else ()
    if not season_periods:
        if eligible_g or eligible_s:
            raise ValueError("Reconstruction needs verified season periods; no inferred dates")
        return ()
    seasons_with_evidence = {g.season for g in eligible_g} | {s.season for s in eligible_s if s.complete}
    periods = sorted((p for p in season_periods if p.season in seasons_with_evidence and p.end >= lower
                      and p.start <= min(upper, cutoff)), key=lambda p: (p.start, p.season))
    if len({p.season for p in periods}) != len(periods):
        raise ValueError("Each reconstruction season needs one registered window period")
    by_identity = {}
    for k in k_values:
        if k is not None and (not isinstance(k, int) or k <= 0):
            raise ValueError("Window sizes must be positive integers or None (every admissible season)")
        chosen = periods if k is None else periods[:k]
        if not chosen:
            continue
        intended_start = max(lower, chosen[0].start)
        intended_end = min(upper, chosen[-1].end)
        allowed = {p.season for p in chosen}
        gs = [g for g in eligible_g if g.season in allowed and intended_start <= g.date <= intended_end]
        ss = [s for s in eligible_s if s.season in allowed and intended_start <= s.period_start and s.period_end <= intended_end]
        history = route_records(gs, ss, tolerance)
        if not history.nonempty:
            continue
        window = ForecastWindow(player_id, target_season, direction, (k,), intended_start,
                                intended_end, cutoff, intended_end > cutoff, history)
        key = history.identity
        if key in by_identity:
            previous = by_identity[key]
            by_identity[key] = replace(previous, requested_k=previous.requested_k + (k,),
                                       partial=previous.partial or window.partial,
                                       intended_end=max(previous.intended_end, window.intended_end))
        else:
            by_identity[key] = window
    return tuple(by_identity.values())


def player_loss_weights(windows):
    """Equal mass to players, seasons within player, then deduplicated windows.

    Each direction separately sums to one. Direction weights are applied by
    the training objective, not hidden in these row weights.
    """
    windows = tuple(windows)
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen = set()
    for i, w in enumerate(windows):
        key = (w.direction, w.player_id, w.target_season, w.history.identity)
        if key in seen:
            raise ValueError("Deduplicate windows before assigning weights")
        seen.add(key)
        groups[w.direction][w.player_id][w.target_season].append(i)
    result = [0.0] * len(windows)
    for players in groups.values():
        for seasons in players.values():
            for indices in seasons.values():
                for i in indices:
                    result[i] = 1.0 / (len(players) * len(seasons) * len(indices))
    return tuple(result)


def build_pretraining_pool(*, games=(), summaries=(), cutoff, heldout_players=(), tolerance=None, prediction_date=None):
    """Source-only histories: no NCAA targets, destination fields, or target clocks.

    For next-game training supply prediction_date. That date is a strict
    upper bound on both game dates and complete summary endpoints. Encoders
    must subsequently recompute prefix clocks and features from these records.
    """
    cutoff, heldout = parse_date(cutoff), set(heldout_players)
    prediction = None if prediction_date is None else parse_date(prediction_date)
    gs = [g for g in _unique(games) if g.domain == "intl" and g.player_id not in heldout
          and g.date <= cutoff and g.release_date <= cutoff
          and (prediction is None or g.date < prediction)]
    ss = [s for s in _unique(summaries) if s.domain == "intl" and s.player_id not in heldout
          and s.period_end <= cutoff and s.release_date <= cutoff
          and (prediction is None or s.period_end < prediction)]
    players = sorted({g.player_id for g in gs} | {s.player_id for s in ss})
    result = {}
    for player in players:
        history = route_records([g for g in gs if g.player_id == player],
                                [s for s in ss if s.player_id == player], tolerance)
        if history.nonempty:
            result[player] = history
    return result


def augment_history(history, *, delete_game_ids=()):
    """Thin a routed history without changing its original routing decision.

    Original coverage evidence stays as provenance. Active coverage summaries
    require at least one retained game from their scope; deletion never admits
    a formerly excluded summary encoder input. Encoders must compute observed
    exposure from retained games, not from the original provenance record.
    """
    deleted = set(delete_game_ids)
    original_ids = {g.record_id for g in history.games}
    if not deleted <= original_ids:
        raise ValueError("Augmentation can delete only present game IDs")
    retained = tuple(g for g in history.games if g.record_id not in deleted)
    active_scopes = {g.scope for g in retained}
    active_coverage = tuple(s for s in history.coverage_summaries if s.scope in active_scopes)
    return replace(history, games=retained, coverage_summaries=active_coverage,
                   augmentation_deleted_ids=tuple(sorted(set(history.augmentation_deleted_ids) | deleted)))


def eligible_training_outcomes(units, *, cutoff, heldout_players=()):
    """Require complete verified labels released by the fitting cutoff."""
    cutoff, heldout = parse_date(cutoff), set(heldout_players)
    return tuple(u for u in units if u.player_id not in heldout and u.release_date <= cutoff
                 and u.likelihood_eligible)
