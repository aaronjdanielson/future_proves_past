"""Regression tests for the second review round (D-027): production adapters enforce the contracts."""
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fpp.data.assemble import BOX_FIELDS, _gate, cluster_periods, counts_valid
from fpp.data.folds import Fold, fold_membership
from fpp.data.records import SummaryRecord
from fpp.data.stores import Stores
from fpp.data.tables import read_tables, write_tables
from fpp.data.windows import route_records


class SeasonInferenceTests(unittest.TestCase):
    def test_stray_summer_game_cannot_flip_the_season(self):
        # A July game 57 days before a September start joins the cluster; majority labelling keeps 2025.
        dates = pd.to_datetime(["2024-07-20"] + [f"2024-{m:02d}-15" for m in (9, 10, 11, 12)] + ["2025-01-15", "2025-03-15"])
        p = cluster_periods(dates, gap_days=60)
        self.assertEqual(list(p["season"]), [2025])
        self.assertLess(float(p["majority_share"].iloc[0]), 1.0)     # the dissent is recorded for the audit

    def test_summer_league_keeps_its_year_past_august(self):
        dates = pd.to_datetime(["2025-03-10", "2025-05-20", "2025-06-15", "2025-07-25", "2025-08-05", "2025-08-12"])
        p = cluster_periods(dates, gap_days=60)
        self.assertEqual(list(p["season"]), [2025])

    def test_covid_shifted_summer_season_is_not_merged_into_the_next(self):
        # A July–September 2021 season votes 2022 by majority; the guard hands it 2021 because the span
        # of the merged label would exceed one season and 2021 is free.
        late = [f"2021-07-{d:02d}" for d in (10, 20, 30)] + [f"2021-08-{d:02d}" for d in (5, 12, 19, 26)] + ["2021-09-04", "2021-09-11"]
        regular = [f"2022-0{m}-15" for m in (3, 4, 5, 6, 7)] + ["2022-08-10", "2022-08-20"]
        p = cluster_periods(pd.to_datetime(late + regular), gap_days=60)
        self.assertEqual(list(p["season"]), [2021, 2022])
        self.assertEqual(p.loc[p["season"] == 2021, "guard"].iloc[0], "relabelled_preceding_year")
        self.assertEqual(p.loc[p["season"] == 2021, "end"].iloc[0], pd.Timestamp("2021-09-11"))

    def test_split_season_competition_is_flagged_not_silently_merged(self):
        # Two tournaments in one calendar year (Apertura/Clausura) plus the next spring: the preceding label is
        # taken, so the overlong merge remains and is flagged for the audit.
        spring = [f"2022-0{m}-15" for m in (3, 4, 5, 6)]                    # 2022 (label taken)
        autumn = ["2022-08-20"] + [f"2022-{m:02d}-15" for m in (9, 10, 11, 12)]   # 66 days after spring; votes 2023
        next_spring = [f"2023-0{m}-15" for m in (3, 4, 5, 6)] + ["2023-07-25"]   # 2023; merged span 339 days
        p = cluster_periods(pd.to_datetime(spring + autumn + next_spring), gap_days=60)
        self.assertIn("overlong_merge_unresolved", set(p["guard"]))
        self.assertEqual(list(p["season"]), [2022, 2023])

    def test_sparse_cup_rounds_merge_by_label(self):
        dates = pd.to_datetime(["2024-10-01", "2024-10-02", "2025-02-20"])   # 141-day gap, same season
        p = cluster_periods(dates, gap_days=60)
        self.assertEqual(list(p["season"]), [2025])
        self.assertEqual(p["end"].iloc[0], pd.Timestamp("2025-02-20"))


class PointsCertificateTests(unittest.TestCase):
    """D-029: an opponent's lines filed under a team read as 400 minutes / eight overtimes; points expose it."""

    def test_extra_lines_fail_coverage_and_overtime(self):
        from fpp.data.assemble import build_ncaa_games

        class Fake:
            def ncaa_gamelogs(self):
                rows = []
                for i in range(20):                       # two teams' players under one team id: 400 minutes
                    rows.append({"player_id": 1000 + i, "game_id": 1, "season": 2023, "date": "2022-11-25", "team_id": 160,
                                 "opp_id": 9, "status": "Bench", "minutes": 20.0, "pts": 6, "fg2a": 3, "fg2m": 3, "fg3a": 0, "fg3m": 0,
                                 "fta": 0, "ftm": 0, "orb": 1, "drb": 1, "ast": 1, "stl": 0, "blk": 0, "tov": 1, "pf": 1, "game_type": "regular_season"})
                for i in range(10):                       # a clean game
                    rows.append({**rows[0], "player_id": 2000 + i, "game_id": 2, "date": "2022-12-01", "minutes": 20.0, "pts": 7})
                return pd.DataFrame(rows)

            def ncaa_team_games(self):
                return pd.DataFrame({"game_id": [1, 1, 2, 2], "team_id": [160, 9, 160, 9], "season": 2023, "home": [1, 0, 1, 0],
                                     "pts": [75, 45, 70, 60], "poss": 65.0, "game_type": "regular_season"})

            def ncaa_team_ratings(self):
                return pd.DataFrame({"team_id": [160, 9], "season": [2023, 2023], "adj_o": [110.0, 100.0], "adj_d": [100.0, 105.0], "adj_pace": [68.0, 70.0]})

        games, _, team = build_ncaa_games(Fake(), lag_days=1)
        g1 = games[games["game_id"] == 1].iloc[0]
        g2 = games[games["game_id"] == 2].iloc[0]
        self.assertFalse(bool(g1["points_consistent"]))
        self.assertFalse(bool(g1["coverage_certified"]))
        self.assertFalse(bool(g1["overtime_certified"]))
        self.assertTrue(np.isnan(g1["overtime_periods"]))
        self.assertTrue(bool(g2["points_consistent"]) and bool(g2["coverage_certified"]) and g2["overtime_periods"] == 0.0)


class CountIntegrityTests(unittest.TestCase):
    def test_fractional_counts_are_invalid(self):
        counts = pd.DataFrame([[5, 2, 2, 1, 2, 2, 1, 2, 0, 0, 0, 1, 2], [5, 2.5, 2, 1, 2, 2, 1, 2, 0, 0, 0, 1, 2]],
                              columns=list(BOX_FIELDS))
        self.assertEqual(list(counts_valid(counts, pd.Series([20.0, 20.0]))), [True, False])


def _unit_row(**over):
    base = dict(player_id=1, season=2024, period_start=pd.Timestamp("2023-11-06"), period_end=pd.Timestamp("2024-03-10"),
                release_date=pd.Timestamp("2024-03-11"), scheduled_games=30, games_played=10, starts=3.0, minutes=200.0,
                coverage_complete=True, invalid_rows=0, unresolved_rows=0, summary_reconciled=True, summary_gp=10,
                **{n: 0 for n in BOX_FIELDS})
    base.update(over)
    return base


class GateTests(unittest.TestCase):
    def test_unresolved_starts_without_summary_are_excluded(self):
        units = pd.DataFrame([_unit_row(starts=np.nan, summary_reconciled=False, summary_gp=np.nan)])
        reasons, eligible, evidence = _gate(units)
        self.assertFalse(eligible[0])
        self.assertIn("UNRESOLVED_OUTCOME", reasons[0])

    def test_summary_starts_disagreement_is_not_reconciled(self):
        # summary_reconciled is computed upstream from games, minutes, and starts; a False value must exclude.
        units = pd.DataFrame([_unit_row(summary_reconciled=False)])
        _, eligible, _ = _gate(units)
        self.assertFalse(eligible[0])

    def test_participant_without_season_line_is_not_certified_by_minutes(self):
        units = pd.DataFrame([_unit_row(summary_reconciled=False, summary_gp=np.nan, coverage_complete=True)])
        _, eligible, evidence = _gate(units)
        self.assertFalse(eligible[0])
        self.assertIsNone(evidence[0])

    def test_zero_row_unit_requires_consistency_and_no_listed_lines(self):
        ok = pd.DataFrame([_unit_row(games_played=0, starts=0.0, minutes=0.0, summary_reconciled=False, summary_gp=np.nan)])
        _, eligible, evidence = _gate(ok)
        self.assertTrue(eligible[0])
        self.assertEqual(evidence[0], "NO_SEASON_LINE_AND_TEAM_MINUTE_SUM_CONSISTENT")
        listed = pd.DataFrame([_unit_row(games_played=0, starts=0.0, minutes=0.0, summary_reconciled=False, summary_gp=np.nan, unresolved_rows=1)])
        _, eligible, _ = _gate(listed)
        self.assertFalse(eligible[0])                                   # a listed zero-stat line is unresolved, not absent


class FoldReleaseTests(unittest.TestCase):
    def test_late_released_outcome_is_excluded_from_training(self):
        units = pd.DataFrame([
            {"unit_id": "a", "player_id": 1, "season": 2022, "release_date": pd.Timestamp("2022-04-01"), "likelihood_eligible": True,
             "intl_entrant": False, "period_end": pd.Timestamp("2022-03-31"), "forecast_cutoff": pd.Timestamp("2021-10-01")},
            {"unit_id": "b", "player_id": 2, "season": 2022, "release_date": pd.Timestamp("2022-10-15"), "likelihood_eligible": True,
             "intl_entrant": False, "period_end": pd.Timestamp("2022-03-31"), "forecast_cutoff": pd.Timestamp("2021-10-01")},
        ])
        games = pd.DataFrame({"record_id": [], "source": [], "player_id": [], "date": [], "release_date": [], "season": []})
        periods = pd.DataFrame({"source": ["intl"], "season": [2022], "start": [pd.Timestamp("2021-10-01")], "end": [pd.Timestamp("2022-06-01")]})
        m = fold_membership({"units": units, "games": games, "competition_periods": periods}, Fold(2023, "development"))
        self.assertEqual(list(m["train_units"]["unit_id"]), ["a"])
        self.assertEqual(m["late_released_excluded"], 1)


class TableIntegrityTests(unittest.TestCase):
    def _tables(self):
        units = pd.DataFrame({"unit_id": ["u"], "exclusion_reasons": [""], "evidence": ["x"]})
        games = pd.DataFrame({"record_id": ["r"], "player_id": [1]})
        periods = pd.DataFrame({"source": ["intl"], "season": [2024]})
        return {"units": units, "games": games, "competition_periods": periods, "audit": {"n": 1}}

    def test_tampered_table_and_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "t"
            write_tables(self._tables(), out, config={}, fingerprints={})
            read_tables(out)                                            # intact: loads
            with open(out / "games.parquet", "ab") as handle:
                handle.write(b"\0")
            with self.assertRaises(ValueError):
                read_tables(out)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "t"
            write_tables(self._tables(), out, config={}, fingerprints={})
            path = out / "manifest.json"
            text = path.read_text().replace('"n": 1', '"n": 2')
            path.chmod(0o644)
            path.write_text(text)
            with self.assertRaises(ValueError):
                read_tables(out)


class OverlappingSummaryTests(unittest.TestCase):
    def test_regular_and_full_season_summaries_are_rejected(self):
        common = dict(player_id="1", release_date=date(2024, 7, 1), competition="L", season=2024, complete=True)
        regular = SummaryRecord(record_id="s1", period_start=date(2023, 10, 1), period_end=date(2024, 4, 1), games_played=30, minutes=600.0, **common)
        full = SummaryRecord(record_id="s2", period_start=date(2023, 10, 1), period_end=date(2024, 6, 1), games_played=36, minutes=720.0, **common)
        with self.assertRaises(ValueError):
            route_records([], [regular, full])
        route_records([], [regular])                                     # a single summary routes normally


class ContentProbeTests(unittest.TestCase):
    def test_probe_sees_row_changes_that_file_stats_may_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "intl.db"
            con = sqlite3.connect(db)
            con.execute("CREATE TABLE boxscores (GameID INTEGER, TeamID INTEGER)")
            con.execute("INSERT INTO boxscores VALUES (1, 1)")
            con.commit()
            stores = Stores({"intl": db})
            before = stores.content_probe("intl")
            con.execute("INSERT INTO boxscores VALUES (2, 2)")
            con.commit()
            after = stores.content_probe("intl")
            stores.close()
            con.close()
            self.assertNotEqual(before, after)
            self.assertEqual(after["boxscores"]["rows"], 2)


if __name__ == "__main__":
    unittest.main()


class LabelScopeTests(unittest.TestCase):
    """D-060: first-season labels only — training units and reconstruction windows restricted; evaluation unchanged."""

    def test_first_year_scope(self):
        base = {"likelihood_eligible": True, "intl_entrant": False, "period_end": pd.Timestamp("2021-03-31"),
                "forecast_cutoff": pd.Timestamp("2020-10-01"), "release_date": pd.Timestamp("2021-04-01")}
        units = pd.DataFrame([
            {**base, "unit_id": "f", "player_id": 1, "season": 2021, "first_year": True},
            {**base, "unit_id": "s", "player_id": 1, "season": 2022, "first_year": False, "period_end": pd.Timestamp("2022-03-31"),
             "forecast_cutoff": pd.Timestamp("2021-10-01"), "release_date": pd.Timestamp("2022-04-01")},
            {**base, "unit_id": "e", "player_id": 2, "season": 2023, "first_year": True, "period_end": pd.Timestamp("2023-03-31"),
             "forecast_cutoff": pd.Timestamp("2022-10-01"), "release_date": pd.Timestamp("2023-04-01")},
            {**base, "unit_id": "e2", "player_id": 3, "season": 2023, "first_year": False, "period_end": pd.Timestamp("2023-03-31"),
             "forecast_cutoff": pd.Timestamp("2022-10-01"), "release_date": pd.Timestamp("2023-04-01")},
        ])
        games = pd.DataFrame({"record_id": [], "source": [], "player_id": [], "date": [], "release_date": [], "season": []})
        periods = pd.DataFrame({"source": ["intl"], "season": [2022], "start": [pd.Timestamp("2021-10-01")], "end": [pd.Timestamp("2022-06-01")]})
        tabs = {"units": units, "games": games, "competition_periods": periods}
        full = fold_membership(tabs, Fold(2023, "development"))
        fy = fold_membership(tabs, Fold(2023, "development"), label_scope="first_year")
        self.assertEqual(sorted(full["train_units"]["unit_id"]), ["f", "s"])
        self.assertEqual(list(fy["train_units"]["unit_id"]), ["f"])
        self.assertEqual(sorted(fy["evaluation_units"]["unit_id"]), ["e", "e2"])          # evaluation set unchanged
        with self.assertRaises(ValueError):
            fold_membership(tabs, Fold(2023, "development"), label_scope="freshmen")
