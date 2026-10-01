"""Track-1 assembly helpers and fold membership on synthetic frames.

``windows.build_windows`` is the oracle for reconstruction nesting; the
vectorized fold membership must reproduce its game sets, requested k, and
partial flags on a synthetic player.
"""
import unittest
from datetime import date

import numpy as np
import pandas as pd

from fpp.data.assemble import (assign_by_periods, cluster_periods, counts_valid, end_year_label,
                               overtime_from_team_minutes, pair_opponents, regulation_from_minutes)
from fpp.data.folds import Fold, reconstruction_windows, season_periods
from fpp.data.records import GameRecord, SeasonPeriod
from fpp.data.windows import build_windows


class OvertimeTests(unittest.TestCase):
    def test_periods_and_certificates(self):
        minutes = pd.Series([200.0, 199.0, 225.0, 251.0, 210.0, 172.0, 240.0, 213.0])
        regulation = pd.Series([40, 40, 40, 40, 40, 40, 48, 40])
        periods, overtime, coverage = overtime_from_team_minutes(minutes, regulation)
        # A rounding excess up to +10 (210) still identifies the pattern; +13 (213) is ambiguous; 172 is a missing line.
        self.assertEqual(list(overtime), [True, True, True, True, True, False, True, False])
        self.assertEqual(list(coverage), [True, True, True, True, True, False, True, True])
        self.assertEqual([p for p, c in zip(periods, overtime) if c], [0.0, 0.0, 1.0, 2.0, 0.0, 0.0])
        self.assertTrue(np.isnan(periods.iloc[5]) and np.isnan(periods.iloc[7]))

    def test_regulation_snap(self):
        self.assertEqual(regulation_from_minutes(pd.Series([200, 199, 201, 225])), 40)
        self.assertEqual(regulation_from_minutes(pd.Series([240, 240, 241, 265])), 48)
        self.assertEqual(regulation_from_minutes(pd.Series([], dtype=float)), 40)


class SeasonTests(unittest.TestCase):
    def test_end_year_label_rolls_on_august_first(self):
        labels = end_year_label(pd.Series(["2025-07-31", "2025-08-01", "2025-11-15", "2026-06-20"]))
        self.assertEqual(list(labels), [2025, 2026, 2026, 2026])

    def test_cluster_periods_summer_league_stays_whole(self):
        # A summer league (March–August) plus a winter league would be split by a June-1 label;
        # clustering keeps each season whole and labels it at its start.
        dates = pd.to_datetime(["2024-03-10", "2024-05-20", "2024-06-15", "2024-08-05",
                                "2025-03-08", "2025-06-30", "2025-08-02"])
        periods = cluster_periods(dates, gap_days=60)
        self.assertEqual(list(periods["season"]), [2024, 2025])
        self.assertEqual(periods["end"].iloc[0], pd.Timestamp("2024-08-05"))

    def test_cluster_periods_merges_interrupted_season(self):
        dates = pd.to_datetime(["2019-10-20", "2020-01-15", "2020-06-25", "2020-08-10"])  # five-month pause
        periods = cluster_periods(dates, gap_days=60)
        self.assertEqual(len(periods), 1)
        self.assertEqual(int(periods["season"].iloc[0]), 2020)

    def test_assign_by_periods(self):
        periods = pd.DataFrame({"competition_id": [7, 7], "season": [2024, 2025],
                                "start": pd.to_datetime(["2023-10-01", "2024-10-01"]), "end": pd.to_datetime(["2024-06-30", "2025-06-30"])})
        dates = pd.Series(pd.to_datetime(["2024-03-01", "2024-12-01", "2024-08-15"]))
        keys = pd.Series([7, 7, 7])
        self.assertEqual(list(assign_by_periods(dates, keys, periods)), [2024, 2025, -1])


class CountTests(unittest.TestCase):
    def test_counts_valid_rejects_makes_over_attempts(self):
        counts = pd.DataFrame({"a2": [5, 3], "k2": [2, 4], "a3": [1, 1], "k3": [1, 0], "af": [2, 2], "kf": [2, 1],
                               "orb": [1, 1], "drb": [2, 2], "ast": [0, 0], "stl": [0, 0], "blk": [0, 0], "tov": [1, 1], "pf": [2, 2]})
        self.assertEqual(list(counts_valid(counts, pd.Series([20.0, 15.0]))), [True, False])

    def test_pair_opponents(self):
        team = pd.DataFrame({"GameID": [1, 1, 2], "TeamID": [10, 20, 30], "PTS": [80, 75, 90]})
        paired = pair_opponents(team).sort_values("TeamID").reset_index(drop=True)
        self.assertEqual(list(paired["opp_id"].iloc[:2]), [20, 10])
        self.assertTrue(np.isnan(paired["opp_id"].iloc[2]))

    def test_pair_opponents_keeps_one_row_with_three_team_ids(self):
        # A resolved opponent (scored line) and an untracked one in the same game: one row per team, scored partner kept.
        team = pd.DataFrame({"GameID": [1, 1, 1], "TeamID": [10, 20, 30], "PTS": [80, 75, np.nan]})
        paired = pair_opponents(team)
        self.assertEqual(len(paired), 3)
        self.assertEqual(int(paired.loc[paired["TeamID"] == 10, "opp_id"].iloc[0]), 20)


def _synthetic_player(dates_labels):
    rows = []
    for i, (d, label) in enumerate(dates_labels):
        rows.append({"record_id": f"intl:{i}:1", "source": "intl", "player_id": 1, "date": pd.Timestamp(d),
                     "release_date": pd.Timestamp(d) + pd.Timedelta(days=1), "season": label, "minutes": 20.0})
    return pd.DataFrame(rows)


def _records(frame):
    out = []
    for r in frame.itertuples(index=False):
        out.append(GameRecord(player_id="1", record_id=r.record_id, date=r.date.date(), release_date=r.release_date.date(),
                              competition="L", season=int(r.season), minutes=r.minutes))
    return out


class ReconstructionWindowTests(unittest.TestCase):
    def test_matches_build_windows_oracle(self):
        games = _synthetic_player([("2020-10-05", 2021), ("2021-03-01", 2021), ("2021-10-12", 2022), ("2022-05-20", 2022),
                                   ("2022-10-01", 2023), ("2023-04-01", 2023), ("2024-11-11", 2025)])
        periods = pd.DataFrame({"source": "intl", "competition_id": 1, "competition": "L",
                                "season": [2021, 2022, 2023, 2025],
                                "start": pd.to_datetime(["2020-09-15", "2021-09-15", "2022-09-15", "2024-09-15"]),
                                "end": pd.to_datetime(["2021-06-15", "2022-06-15", "2023-06-15", "2025-06-15"])})
        labels = season_periods(periods)

        class Unit:
            unit_id, player_id, season = "1:2020:9", 1, 2020
            period_end = pd.Timestamp("2020-03-10")
            forecast_cutoff = pd.Timestamp("2019-10-01")

        fold = Fold(2023, "development")  # training cutoff 2022-09-30 clips the third season
        ours = reconstruction_windows(Unit(), games, fold, labels)
        oracle = build_windows(player_id="1", target_season=2020, target_start=date(2019, 11, 5), target_end=date(2020, 3, 10),
                               forecast_cutoff=date(2019, 10, 1), training_cutoff=fold.training_cutoff.date(), direction="+",
                               games=_records(games), season_periods=[SeasonPeriod(int(s), st.date(), en.date()) for s, st, en in
                                                                       zip(periods["season"], periods["start"], periods["end"])])
        self.assertEqual(len(ours), len(oracle))
        ours_sets = sorted((tuple(sorted(w["record_ids"])), w["k"], w["partial"]) for w in ours)
        oracle_sets = sorted((tuple(sorted(g.record_id for g in w.history.games)), w.requested_k, w.partial) for w in oracle)
        self.assertEqual(ours_sets, oracle_sets)
        # k=1 is the 2021 season alone; k=2 adds 2022; k=3 would add 2023 but nothing is released by the cutoff.
        self.assertEqual([w["k"] for w in ours], [(1,), (2, 3)])

    def test_full_horizon_admits_late_careers_and_matches_oracle(self):
        # A 2015 first-year unit whose tracked games resume five to seven years later (a four-year player turned
        # professional): the registered four-year horizon yields nothing; the full horizon (D-038) admits every
        # game released by the cutoff, nested by season, with an "all seasons" window deduplicated against k=3.
        games = _synthetic_player([("2020-10-05", 2021), ("2021-03-01", 2021), ("2021-10-12", 2022), ("2022-05-20", 2022),
                                   ("2022-09-01", 2023)])
        periods = pd.DataFrame({"source": "intl", "competition_id": 1, "competition": "L", "season": [2021, 2022, 2023],
                                "start": pd.to_datetime(["2020-09-15", "2021-09-15", "2022-08-15"]),
                                "end": pd.to_datetime(["2021-06-15", "2022-06-15", "2023-06-15"])})
        labels = season_periods(periods)

        class Unit:
            unit_id, player_id, season = "1:2015:9", 1, 2015
            period_end = pd.Timestamp("2015-03-10")
            forecast_cutoff = pd.Timestamp("2014-10-01")

        fold = Fold(2023, "development")
        self.assertEqual(reconstruction_windows(Unit(), games, fold, labels), [])
        self.assertEqual(reconstruction_windows(Unit(), games, fold, labels, max_years=4.0), [])   # a float horizon must not crash
        ours = reconstruction_windows(Unit(), games, fold, labels, k_values=(1, 2, 3, None), max_years=None)
        sp = [SeasonPeriod(int(s), st.date(), en.date()) for s, st, en in zip(periods["season"], periods["start"], periods["end"])]
        oracle = build_windows(player_id="1", target_season=2015, target_start=date(2014, 11, 5), target_end=date(2015, 3, 10),
                               forecast_cutoff=date(2014, 10, 1), training_cutoff=fold.training_cutoff.date(), direction="+",
                               games=_records(games), season_periods=sp, k_values=(1, 2, 3, None), max_years=None)
        ours_sets = sorted((tuple(sorted(w["record_ids"])), w["k"], w["partial"]) for w in ours)
        oracle_sets = sorted((tuple(sorted(g.record_id for g in w.history.games)), w.requested_k, w.partial) for w in oracle)
        self.assertEqual(ours_sets, oracle_sets)
        self.assertEqual([w["k"] for w in ours], [(1,), (2,), (3, None)])
        self.assertEqual([w["n_games"] for w in ours], [2, 4, 5])
        self.assertTrue(ours[-1]["partial"])          # the 2023 season runs past the 2022-09-30 cutoff


if __name__ == "__main__":
    unittest.main()
