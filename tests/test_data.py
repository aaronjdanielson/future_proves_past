"""Synthetic source-contract tests; no production DB or heldout outcomes read."""
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from fpp.data.calendar import parse_date, season_of, forecast_cutoff, training_cutoff, assumed_release
from fpp.data.records import (GameRecord, SummaryRecord, SeasonPeriod, OutcomeUnit,
                              ReconciliationTolerance, classify_participation)
from fpp.data.windows import (build_windows, route_records, player_loss_weights, build_pretraining_pool,
                              augment_history, eligible_training_outcomes)
from fpp.data.source import ReadOnlyRoster, audit_database
from fpp.data.manifests import write_manifest, load_manifest, file_sha256
from fpp.data.seasons import (CompetitionSeasonPeriod, SeasonAssignment, assign_season,
                              audit_assignments)

TOL = ReconciliationTolerance("fixture-exact-v1", 0, 0.0)


def game(record_id="g1", day="2022-10-10", **overrides):
    d = parse_date(day)
    values = dict(player_id="p1", record_id=record_id, date=d, release_date=assumed_release(d),
                  competition="L", season=season_of(d), minutes=10.0, appeared=True,
                  period_start=date(2022, 6, 1), period_end=date(2023, 5, 31))
    values.update(overrides)
    return GameRecord(**values)


def summary(record_id="s1", **overrides):
    values = dict(player_id="p1", record_id=record_id, period_start=date(2022, 6, 1),
                  period_end=date(2023, 5, 31), release_date=date(2023, 6, 1), competition="L",
                  season=2023, games_played=2, minutes=20.0, complete=True)
    values.update(overrides)
    return SummaryRecord(**values)


def forward(**overrides):
    args = dict(player_id="p1", target_season=2024, target_start="2023-11-01",
                target_end="2024-04-01", forecast_cutoff="2023-10-01",
                training_cutoff="2023-09-30", direction="-", tolerance=TOL)
    args.update(overrides)
    return build_windows(**args)


class CalendarTests(unittest.TestCase):
    def test_end_year_fallback_rollover(self):
        # D-016: the fallback rolls on 1 August; June playoff games keep their season.
        self.assertEqual(season_of("2024-06-15"), 2024)
        self.assertEqual(season_of("2024-07-31"), 2024)
        self.assertEqual(season_of("2024-08-01"), 2025)
        self.assertEqual(season_of("2024-06-01", rollover=(6, 1)), 2025)
        with self.assertRaises(ValueError):
            season_of("2024-06-01", rollover=(13, 1))
        self.assertEqual(forecast_cutoff(2026), date(2025, 10, 1))
        self.assertEqual(training_cutoff(2026), date(2025, 9, 30))

    def test_actual_iso_dates_required(self):
        for value in (2023, "2023", "20231001", None):
            with self.assertRaises((ValueError, TypeError)):
                parse_date(value)
        with self.assertRaises(ValueError):
            game(season=1900)

    def test_zero_minutes_not_nonparticipation(self):
        self.assertIsNone(game(minutes=0, appeared=None).appeared)


class SeasonAssignmentTests(unittest.TestCase):
    def test_provider_identifier_takes_precedence(self):
        a = assign_season(date="2024-06-15", competition="ACB", original_label=2025,
                          provider_season=2024)
        self.assertEqual((a.season, a.method), (2024, "provider_identifier"))
        self.assertTrue(a.changed)
        self.assertTrue(a.verified)

    def test_competition_period_before_fallback(self):
        period = CompetitionSeasonPeriod("ACB", 2024, "2023-09-20", "2024-06-25")
        a = assign_season(date="2024-06-20", competition="ACB", periods=(period,))
        self.assertEqual((a.season, a.method), (2024, "competition_period"))
        b = assign_season(date="2024-06-20", competition="OTHER", periods=(period,))
        self.assertEqual((b.season, b.method), (2024, "fallback_rollover"))
        self.assertFalse(b.verified)
        overlap = CompetitionSeasonPeriod("ACB", 2025, "2024-06-01", "2025-06-30")
        with self.assertRaises(ValueError):
            assign_season(date="2024-06-20", competition="ACB", periods=(period, overlap))

    def test_fallback_and_audit(self):
        items = [assign_season(date="2024-08-01", competition="X", original_label=2025),
                 assign_season(date="2024-06-15", competition="X", original_label=2025),
                 assign_season(date="2024-03-01", competition="X", provider_season=2024)]
        self.assertEqual([a.season for a in items], [2025, 2024, 2024])
        audit = audit_assignments(items)
        self.assertEqual(audit["total"], 3)
        self.assertEqual(audit["fallback"], 2)
        self.assertEqual(audit["changed"], 1)
        self.assertEqual(audit["changed_pairs"], [((2025, 2024), 1)])
        with self.assertRaises(ValueError):
            SeasonAssignment(2024, "guess")

    def test_game_record_carries_assignment_and_rejects_contradictions(self):
        a = assign_season(date="2022-10-10", competition="L", provider_season=2023)
        self.assertEqual(game(season=2023, season_assignment=a).season_method, "provider_identifier")
        self.assertEqual(game(season=2022).season_method, "unverified_label")
        with self.assertRaises(ValueError):
            game(season=2022, season_assignment=a)
        with self.assertRaises(ValueError):
            game(season=2025)
        self.assertTrue(game(minutes=0, appeared=None, starter=True).appeared)
        self.assertTrue(classify_participation(minutes=0, counts=(1,)).appeared)
        self.assertIsNone(classify_participation(minutes=0).appeared)
        self.assertFalse(classify_participation(minutes=0, provider_status="D", status_map={"D": "dnp"}).appeared)
        self.assertTrue(classify_participation(minutes=0, provider_status="S", status_map={"S": "starter"}).starter)
        result = classify_participation(minutes=1, provider_status="D", status_map={"D": "dnp"})
        self.assertTrue(result.conflict)
        self.assertIsNone(result.appeared)
        self.assertIsNone(game(minutes=1, appeared=result.appeared, participation_conflict=result.conflict).appeared)
        with self.assertRaisesRegex(ValueError, "requires observed minutes"):
            game(minutes=None)
        self.assertTrue(game(minutes=0, appeared=None, boxscore_counts=(1,)).appeared)


class RoutingTests(unittest.TestCase):
    def test_complete_partial_and_unknown(self):
        g1, g2, s = game(), game("g2", "2022-10-20"), summary()
        complete = route_records((g1, g2), (s,), TOL)
        self.assertEqual(complete.coverage[0].route, "games_complete")
        self.assertFalse(complete.summaries)
        self.assertEqual(complete.coverage_summaries, (s,))
        partial = route_records((g1,), (s,), TOL)
        self.assertEqual(partial.coverage[0].route, "games_partial")
        self.assertEqual(partial.coverage[0].appearance_coverage, .5)
        self.assertEqual(partial.source_anchor, s.period_end)
        self.assertEqual(partial.game_anchor, g1.date)
        self.assertFalse(partial.summaries)
        unknown = route_records((g1,), (), TOL)
        self.assertEqual(unknown.coverage[0].route, "games_unknown")
        self.assertIsNone(unknown.coverage[0].reported_minutes)
        self.assertEqual(unknown.source_anchor, g1.date)

    def test_summary_only_and_empty(self):
        only = route_records((), (summary(),), TOL)
        self.assertTrue(only.nonempty)
        self.assertEqual(only.source_anchor, date(2023, 5, 31))
        self.assertIsNone(only.game_anchor)
        self.assertFalse(route_records((), (summary(complete=False),), TOL).nonempty)
        self.assertIsNone(route_records((), (), TOL).source_anchor)

    def test_scope_mismatch_cannot_double_count(self):
        for changed in (replace(summary(), horizon="regular"),
                        replace(summary(), period_start=date(2022, 7, 1))):
            history = route_records((game(),), (changed,), TOL)
            self.assertEqual(history.coverage[0].route, "games_unknown")
            self.assertFalse(history.summaries)
            self.assertFalse(history.coverage_summaries)

    def test_augmentation_cannot_enable_summary_channel(self):
        original = route_records((game(),), (summary(),), TOL)
        thinned = augment_history(original, delete_game_ids=("g1",))
        self.assertFalse(thinned.nonempty)
        self.assertFalse(thinned.summaries)
        self.assertFalse(thinned.coverage_summaries)
        self.assertIsNone(thinned.source_anchor)
        self.assertEqual(thinned.coverage, original.coverage)
        self.assertEqual(thinned.augmentation_deleted_ids, ("g1",))
        with self.assertRaises(ValueError):
            augment_history(original, delete_game_ids=("not-present",))

    def test_nonoverlapping_periods_can_use_different_channels(self):
        early = summary(period_end=date(2022, 8, 31), release_date=date(2022, 9, 1))
        later = game(period_start=date(2022, 9, 1))
        history = route_records((later,), (early,), TOL)
        self.assertEqual(history.games, (later,))
        self.assertEqual(history.summaries, (early,))

    def test_conflicts_not_clipped_to_full_coverage(self):
        history = route_records((game(minutes=30),), (summary(),), TOL)
        self.assertEqual(history.coverage[0].route, "games_unreconciled")
        self.assertEqual(history.coverage[0].minute_coverage, 1.5)

    def test_explicit_tolerance_required(self):
        history = route_records((game(),), (summary(),))
        self.assertEqual(history.coverage[0].audit_reason, "NO_REGISTERED_TOLERANCE")
        self.assertFalse(history.coverage_summaries)
        self.assertIsNone(history.coverage[0].reported_minutes)

    def test_conflicting_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            route_records((game(), game(minutes=4)), (), TOL)
        self.assertEqual(len(route_records((game(), game()), (), TOL).games), 1)


class WindowTests(unittest.TestCase):
    def test_training_and_forecast_cutoffs(self):
        records = (game(), game("late", "2023-09-30", period_start=date(2023, 6, 1), period_end=date(2024, 5, 31)))
        windows = forward(games=records)
        self.assertEqual([g.record_id for g in windows[0].history.games], ["g1"])
        # The same record is released by the inference cutoff, but not the preceding training cutoff.
        inference = forward(games=records, training_cutoff=None)
        self.assertEqual(len(inference[0].history.games), 2)

    def test_unavailable_summary_cannot_change_features(self):
        baseline = forward(games=(game(),), summaries=())
        result = forward(games=(game(),), summaries=(summary(release_date=date(2023, 10, 1)),))
        self.assertEqual(result, baseline)

    def test_cross_boundary_summary_atomic(self):
        crossing = summary(period_end=date(2023, 11, 10), release_date=date(2023, 11, 11))
        self.assertFalse(forward(summaries=(crossing,), forecast_cutoff="2023-12-01", training_cutoff=None))
        valid = forward(summaries=(summary(),))
        self.assertEqual(valid[0].source_anchor, date(2023, 5, 31))

    def test_heldout_excluded_all_training_sources(self):
        self.assertFalse(forward(games=(game(),), heldout_players=("p1",)))
        self.assertFalse(build_pretraining_pool(games=(game(),), summaries=(summary(),),
                                               cutoff="2023-10-01", heldout_players=("p1",), tolerance=TOL))
        self.assertFalse(forward(games=(game(),), label_release_date="2024-06-01"))

    def test_next_game_prefix_excludes_same_day_released_target(self):
        target = game("target", "2022-10-20", release_date=date(2022, 10, 20), release_is_assumed=False)
        pool = build_pretraining_pool(games=(game(), target), cutoff="2023-10-01",
                    summaries=(summary(),), prediction_date="2022-10-20", tolerance=TOL)
        self.assertEqual(tuple(g.record_id for g in pool["p1"].games), ("g1",))
        self.assertFalse(pool["p1"].coverage_summaries)
        self.assertFalse(pool["p1"].summaries)

    def test_reconstruction_clipping_and_dedup(self):
        records = (game(), game("future", "2023-02-01"))
        windows = build_windows(player_id="p1", target_season=2022, target_start="2021-11-01",
                  target_end="2022-04-01", forecast_cutoff="2021-10-01", training_cutoff="2023-01-01",
                  direction="+", games=records, summaries=(summary(),),
                  season_periods=(SeasonPeriod(2023, "2022-06-01", "2023-05-31"),), tolerance=TOL)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].requested_k, (1, 2, 3))
        self.assertTrue(windows[0].partial)
        self.assertEqual(len(windows[0].history.games), 1)
        self.assertFalse(windows[0].history.summaries)
        self.assertFalse(windows[0].history.coverage_summaries)

    def test_reconstruction_summary_must_start_after_ncaa_end(self):
        result = build_windows(player_id="p1", target_season=2022, target_start="2021-11-01",
                  target_end="2022-07-01", forecast_cutoff="2021-10-01", training_cutoff="2023-10-01",
                  direction="+", summaries=(summary(),),
                  season_periods=(SeasonPeriod(2023, "2022-06-01", "2023-05-31"),), tolerance=TOL)
        self.assertFalse(result)

    def test_reconstruction_requires_calendar_and_cap(self):
        kwargs = dict(player_id="p1", target_season=2022, target_start="2021-11-01",
                  target_end="2022-04-01", forecast_cutoff="2021-10-01", training_cutoff="2030-01-01",
                  direction="+", tolerance=TOL)
        with self.assertRaises(ValueError):
            build_windows(**kwargs, games=(game(),))
        later = game("later", "2027-01-01", period_start=date(2026, 6, 1), period_end=date(2027, 5, 31))
        self.assertFalse(build_windows(**kwargs, games=(later,), season_periods=(SeasonPeriod(2027, "2026-06-01", "2027-05-31"),)))
        # D-038: without the years cap the same game is admitted (released by the 2030 cutoff); k=None spans every season.
        full = build_windows(**kwargs, games=(later,), season_periods=(SeasonPeriod(2027, "2026-06-01", "2027-05-31"),),
                             max_years=None, k_values=(1, None))
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0].requested_k, (1, None))
        self.assertFalse(full[0].partial)
        with self.assertRaises(ValueError):
            build_windows(**kwargs, games=(later,), season_periods=(SeasonPeriod(2027, "2026-06-01", "2027-05-31"),), k_values=(0,))

    def test_player_season_window_weights(self):
        a = forward(games=(game(),))[0]
        b = replace(a, target_season=2025)
        c = replace(a, player_id="p2")
        self.assertEqual(player_loss_weights((a, b, c)), (.25, .25, .5))
        with self.assertRaises(ValueError):
            player_loss_weights((a, a))
        no_metadata = forward(games=(game(),))[0]
        metadata = forward(games=(game(),), summaries=(summary(),))[0]
        self.assertNotEqual(no_metadata.history.identity, metadata.history.identity)
        self.assertEqual(player_loss_weights((no_metadata, metadata)), (.5, .5))


class OutcomeTests(unittest.TestCase):
    def complete(self, **kwargs):
        values = dict(player_id="p1", season=2023, horizon="regular", period_start="2022-11-01",
                      period_end="2023-03-01", release_date="2023-03-02", scheduled_games=30,
                      games_played=1, starts=0, minutes=0, counts=(0,) * 13,
                      completeness="COMPLETE", participation_verified=True,
                      evidence="fixture-provider-certified", minutes_provenance="fixture-v1", provenance_verified=True)
        values.update(kwargs)
        return OutcomeUnit(**values)

    def test_positive_appearance_with_recorded_zero(self):
        self.assertTrue(self.complete().likelihood_eligible)
        self.assertFalse(self.complete(completeness="INCOMPLETE_COVERAGE").likelihood_eligible)
        self.assertFalse(self.complete(provenance_verified=False).likelihood_eligible)
        self.assertFalse(self.complete(games_played=0).likelihood_eligible)
        self.assertTrue(self.complete(games_played=0, no_participation_verified=True).likelihood_eligible)

    def test_training_label_release_and_holdout_gates(self):
        unit = self.complete()
        self.assertFalse(eligible_training_outcomes((unit,), cutoff="2023-03-01"))
        self.assertFalse(eligible_training_outcomes((unit,), cutoff="2023-03-02", heldout_players=("p1",)))
        self.assertEqual(eligible_training_outcomes((unit,), cutoff="2023-03-02"), (unit,))

    def test_support_constraints(self):
        for kwargs in ({"games_played": 31}, {"starts": 2}, {"games_played": 0, "minutes": 1},
                       {"counts": (0, 1) + (0,) * 11}):
            with self.assertRaises(ValueError):
                self.complete(**kwargs)


class StorageTests(unittest.TestCase):
    def test_sqlite_is_readonly_and_schema_audit_is_not_outcome_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.db"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE intl_gamelogs (season INTEGER, min REAL)")
            con.execute("INSERT INTO intl_gamelogs VALUES (2026, 99)")
            con.commit()
            con.close()
            before = file_sha256(path)
            audit = audit_database(path)
            self.assertFalse(audit["temporal_eligibility_verified"])
            self.assertNotIn("table_row_counts", audit)
            self.assertIn("date", audit["contract_checks"]["intl_gamelogs"]["missing_columns"])
            self.assertEqual(audit_database(path, include_counts=True)["table_row_counts"]["intl_gamelogs"], 1)
            with ReadOnlyRoster(path) as source:
                with self.assertRaises(sqlite3.OperationalError):
                    source.connection.execute("DELETE FROM intl_gamelogs")
                with self.assertRaises(ValueError):
                    source.count('intl_gamelogs"; DROP TABLE players; --')
            self.assertEqual(file_sha256(path), before)
            missing = Path(tmp) / "missing.db"
            with self.assertRaises(FileNotFoundError):
                audit_database(missing)
            self.assertFalse(missing.exists())

    def test_manifest_write_once_hash_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fold.json"
            payload = {"config": {"rating_tier": "T0"}, "source_sha256": "fixture-hash"}
            write_manifest(path, payload)
            self.assertEqual(load_manifest(path, expected_config=payload["config"], expected_source_sha256="fixture-hash"), payload)
            with self.assertRaises(FileExistsError):
                write_manifest(path, payload)
            with self.assertRaises(ValueError):
                load_manifest(path, expected_config={"rating_tier": "T2"})
            changed = json.loads(path.read_text())
            changed["payload"]["config"]["rating_tier"] = "T2"
            path.chmod(0o644)
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
