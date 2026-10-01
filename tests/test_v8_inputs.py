"""D-039 data side: rating uncertainty on game rows and the latest pre-cutoff own-team strength per unit."""
import unittest

import numpy as np
import pandas as pd

from fpp.data.assemble import _attach_intl_strength, _evidence_profile


def _team_rows():
    return pd.DataFrame({"GameID": [1, 2], "TeamID": [10, 10], "opp_id": [20, 30], "LeagueID": [5, 5], "Season": [2022, 2022]})


def _bucket_end():
    return pd.DataFrame({"LeagueID": [5, 5], "Season": [2021, 2022], "bucket_end": pd.to_datetime(["2021-06-01", "2022-06-01"])})


class RatingUncertaintyTests(unittest.TestCase):
    def test_sd_columns_follow_the_rating(self):
        ratings = pd.DataFrame({"team_id": [10, 20, 20], "season": [2022, 2022, 2021], "league_id": 5,
                                "adj_o": [110.0, 100.0, 98.0], "adj_d": [100.0, 105.0, 106.0], "adj_pace": [70.0, 68.0, 67.0],
                                "adj_o_sd": [1.5, 3.0, 3.5], "adj_d_sd": [1.2, 2.8, 3.1]})
        out = _attach_intl_strength(_team_rows(), ratings, _bucket_end(), lag_days=1)
        g1 = out[out["GameID"] == 1].iloc[0]
        self.assertEqual((g1["team_adj_o"], g1["team_adj_o_sd"], g1["team_adj_d_sd"]), (110.0, 1.5, 1.2))
        self.assertEqual((g1["opp_adj_o"], g1["opp_adj_o_sd"]), (100.0, 3.0))
        self.assertEqual((g1["opp_prior_adj_o"], g1["opp_prior_adj_o_sd"]), (98.0, 3.5))   # 2021 rating as the prior
        self.assertEqual(g1["team_strength_available"], pd.Timestamp("2022-06-02"))
        g2 = out[out["GameID"] == 2].iloc[0]                                             # unrated opponent
        self.assertTrue(np.isnan(g2["opp_adj_o"]) and np.isnan(g2["opp_adj_o_sd"]))

    def test_rating_store_without_uncertainty_is_tolerated(self):
        ratings = pd.DataFrame({"team_id": [10], "season": [2022], "league_id": 5, "adj_o": [110.0], "adj_d": [100.0], "adj_pace": [70.0]})
        out = _attach_intl_strength(_team_rows(), ratings, _bucket_end(), lag_days=1)
        self.assertEqual(out["team_adj_o"].iloc[0], 110.0)
        self.assertTrue(out["team_adj_o_sd"].isna().all())

    def test_no_rating_store(self):
        out = _attach_intl_strength(_team_rows(), None, _bucket_end(), lag_days=1)
        self.assertTrue(out["team_adj_o_sd"].isna().all() and out["opp_prior_adj_d_sd"].isna().all())


class LastTeamStrengthTests(unittest.TestCase):
    def test_latest_available_team_before_cutoff(self):
        games = pd.DataFrame({
            "player_id": [1, 1, 1, 1], "source": ["intl", "intl", "intl", "ncaa"],
            "date": pd.to_datetime(["2020-02-01", "2021-02-01", "2021-05-01", "2021-11-20"]),
            "release_date": pd.to_datetime(["2020-02-02", "2021-02-02", "2021-05-02", "2021-11-21"]),
            "minutes": [20.0, 25.0, 30.0, 15.0], "competition_id": [5, 5, 6, 0], "team_is_usa": False, "season": [2020, 2021, 2021, 2022],
            "team_adj_o": [100.0, 108.0, 112.0, np.nan], "team_adj_d": [104.0, 101.0, 99.0, np.nan], "team_adj_pace": [70.0, 71.0, 72.0, np.nan],
            "team_adj_o_sd": [2.0, 1.5, 4.0, np.nan], "team_adj_d_sd": [2.1, 1.4, 3.9, np.nan],
            # 2020 season rating available 2020-06-02; 2021 ratings available 2021-06-02 (after the 2021-10-01 cutoff? no: before)
            "team_strength_available": pd.to_datetime(["2020-06-02", "2021-06-02", "2021-10-15", pd.NaT]),
        })
        units = pd.DataFrame({"player_id": [1, 1, 2], "season": [2022, 2021, 2022],
                              "forecast_cutoff": pd.to_datetime(["2021-10-01", "2020-10-01", "2021-10-01"])})
        out = _evidence_profile(units, games)
        # Unit 2022 (cutoff 2021-10-01): the May 2021 game's rating (competition 6) is not available until 2021-10-15,
        # so the latest available team is the February 2021 one.
        self.assertEqual((out.loc[0, "last_team_adj_o"], out.loc[0, "last_team_adj_o_sd"]), (108.0, 1.5))
        self.assertEqual(out.loc[0, "last_team_strength_date"], pd.Timestamp("2021-02-01"))
        self.assertEqual(out.loc[0, "n_intl_games"], 3)
        # Unit 2021 (cutoff 2020-10-01): only the 2020 game and its rating are released.
        self.assertEqual(out.loc[1, "last_team_adj_o"], 100.0)
        # A player with no games: NaN strength, zero counts.
        self.assertTrue(np.isnan(out.loc[2, "last_team_adj_o"]) and out.loc[2, "n_intl_games"] == 0)

    def test_tables_without_strength_columns(self):
        games = pd.DataFrame({"player_id": [1], "source": ["intl"], "date": pd.to_datetime(["2020-02-01"]),
                              "release_date": pd.to_datetime(["2020-02-02"]), "minutes": [20.0], "competition_id": [5],
                              "team_is_usa": False, "season": [2020]})
        units = pd.DataFrame({"player_id": [1], "season": [2021], "forecast_cutoff": pd.to_datetime(["2020-10-01"])})
        out = _evidence_profile(units, games)
        self.assertTrue(np.isnan(out.loc[0, "last_team_adj_o"]))
        self.assertEqual(out.loc[0, "n_intl_games"], 1)


class ShowcaseProfileTests(unittest.TestCase):
    def test_selection_flags_respect_cutoff_and_side(self):
        from fpp.data.assemble import _showcase_profile
        games = pd.DataFrame({
            "player_id": [1, 1, 2, 3], "source": ["events", "events", "events", "intl"],
            "release_date": pd.to_datetime(["2022-04-10", "2022-03-30", "2023-04-09", "2022-01-01"]),
            "competition": ["nike_hoop_summit", "mcdonalds", "nike_hoop_summit", "L"],
            "team_is_usa": [False, False, True, False], "starter": [True, False, True, False],
            "minutes": [24.0, 18.0, 30.0, 20.0], "pts": [12.0, 6.0, 20.0, 4.0]})
        units = pd.DataFrame({"unit_id": ["1:2023", "2:2023", "2:2024", "3:2023", "4:2023"], "player_id": [1, 2, 2, 3, 4],
                              "forecast_cutoff": pd.to_datetime(["2022-10-01", "2022-10-01", "2023-10-01", "2022-10-01", "2022-10-01"])})
        out = _showcase_profile(units, games).set_index("unit_id")
        u1 = out.loc["1:2023"]
        self.assertTrue(bool(u1["showcase_hoop_summit_world"]) and bool(u1["showcase_mcdonalds"]) and not bool(u1["showcase_hoop_summit_usa"]))
        self.assertEqual((int(u1["n_showcase_events"]), int(u1["showcase_games"]), int(u1["showcase_starts"]), float(u1["showcase_minutes"])), (2, 2, 1, 42.0))
        self.assertFalse(bool(out.loc["2:2023", "showcase_hoop_summit_usa"]))     # the 2023 edition is after the 2022 cutoff
        self.assertTrue(bool(out.loc["2:2024", "showcase_hoop_summit_usa"]))      # ... and before the 2023 cutoff
        self.assertEqual(int(out.loc["3:2023", "showcase_games"]), 0)              # club games are not showcases
        self.assertFalse(bool(out.loc["4:2023", "showcase_biosteel"]))            # untracked player: False, not NaN
        self.assertEqual(out["showcase_hoop_summit_world"].dtype, bool)

    def test_no_showcase_source(self):
        from fpp.data.assemble import _showcase_profile
        games = pd.DataFrame({"player_id": [1], "source": ["intl"], "release_date": pd.to_datetime(["2022-01-01"]), "competition": ["L"],
                              "team_is_usa": [False], "starter": [False], "minutes": [20.0], "pts": [4.0]})
        units = pd.DataFrame({"unit_id": ["1:2023"], "player_id": [1], "forecast_cutoff": pd.to_datetime(["2022-10-01"])})
        out = _showcase_profile(units, games)
        self.assertFalse(bool(out.loc[0, "showcase_mcdonalds"]))
        self.assertEqual(int(out.loc[0, "showcase_games"]), 0)


if __name__ == "__main__":
    unittest.main()


class RecruitStatusTests(unittest.TestCase):
    def test_unranked_and_not_rankable_are_distinct(self):
        from fpp.data.assemble import recruit_status
        u = pd.DataFrame({
            "recruit_national_rank": [5.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "recruit_composite_score": [0.99, 0.90, np.nan, np.nan, np.nan, np.nan],
            "recruit_star_rating": [5.0, 3.0, np.nan, np.nan, np.nan, np.nan],
            "nationality": ["United States", "United States", "Serbia", "United States", "Canada", "United States/Nigeria"],
            "n_intl_games": [0, 0, 40, 0, 0, 12], "n_national_nonusa_games": [0, 0, 20, 0, 0, 0],
            "season": [2026, 2016, 2026, 2026, 2026, 2026], "ncaa_seasons_completed": [0, 0, 0, 0, 0, 0]})
        self.assertEqual(list(recruit_status(u)), ["ranked", "rated_not_ranked", "not_rankable_international",
                                                   "unranked_domestic", "unranked_domestic", "not_rankable_international"])
        old = u.iloc[[3]].assign(season=2002)
        self.assertEqual(recruit_status(old).iloc[0], "class_not_covered")


class FeatureSchemaTests(unittest.TestCase):
    def test_v7_tables_keep_four_static_categoricals_and_v8_adds_status(self):
        from fpp.model import features as F
        base = pd.DataFrame({"position": ["G", "F"], "conference": ["X", "Y"], "role": ["freshman", "freshman"], "class": ["Fr", "Fr"]})
        v7 = F.static_vocab(base)
        self.assertEqual(list(v7), F.STATIC_CATEGORICAL)
        self.assertEqual(F.static_codes(base, v7).shape, (2, 4))
        v8_units = base.assign(recruit_status=["ranked", "not_rankable_international"])
        v8 = F.static_vocab(v8_units)
        self.assertEqual(list(v8), F.STATIC_CATEGORICAL_V8)
        codes = F.static_codes(v8_units, v8)
        self.assertEqual(codes.shape, (2, 5))
        self.assertNotEqual(codes[0, 4], codes[1, 4])                     # the two statuses are distinct inputs
        self.assertEqual(F.static_codes(v8_units, v7).shape, (2, 4))      # a v7 model ignores the extra column

    def test_player_dobs_fill_from_players_table(self):
        from fpp.model import features as F
        units = pd.DataFrame({"player_id": [1, 1, 2], "dob": [pd.Timestamp("2005-01-01")] * 2 + [pd.NaT]})
        players = pd.DataFrame({"player_id": [2, 3], "dob": [pd.Timestamp("2006-02-02"), pd.Timestamp("2007-03-03")]})
        d = F.player_dobs({"units": units, "players": players})
        self.assertEqual((d[1], d[2], d[3]), (pd.Timestamp("2005-01-01"), pd.Timestamp("2006-02-02"), pd.Timestamp("2007-03-03")))
        self.assertTrue(pd.isna(F.player_dobs({"units": units})[2]))       # v7 tables: units only

    def test_tables_round_trip_with_optional_players(self):
        import tempfile
        from pathlib import Path
        from fpp.data.tables import read_tables, write_tables
        tables = {"units": pd.DataFrame({"unit_id": ["u"], "exclusion_reasons": [""], "evidence": ["x"]}),
                  "games": pd.DataFrame({"record_id": ["r"], "player_id": [1]}),
                  "competition_periods": pd.DataFrame({"source": ["intl"], "season": [2024]}),
                  "players": pd.DataFrame({"player_id": [1], "dob": [pd.Timestamp("2006-01-01")]}), "audit": {"n": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            write_tables(tables, Path(tmp) / "t", config={}, fingerprints={})
            back = read_tables(Path(tmp) / "t")
            self.assertIn("players", back)
            self.assertEqual(int(back["players"]["player_id"].iloc[0]), 1)
            tables.pop("players")
            write_tables(tables, Path(tmp) / "t7", config={}, fingerprints={})
            self.assertNotIn("players", read_tables(Path(tmp) / "t7"))

    def test_pool_collate_sizes_static_categoricals_from_the_model(self):
        from fpp.model.pool import collate_pool
        from tests.test_pool_and_arms import _pool_games
        from fpp.model import features as F
        from fpp.model.pool import PoolSampler
        games = _pool_games()
        store = F.GameStore.from_frame(games)
        cutoff = F._days(pd.Series([pd.Timestamp("2022-09-30")]))[0]
        norm = F.Normalizer.fit(store, np.arange(len(games)), cutoff, np.zeros((2, len(F.STATIC_CONTINUOUS))))
        ex = PoolSampler(store, norm, cutoff_days=cutoff, heldout_players=set(), seed=1).sample(4)
        self.assertEqual(tuple(collate_pool(ex)["static_cats"].shape), (4, 4))
        self.assertEqual(tuple(collate_pool(ex, 5)["static_cats"].shape), (4, 5))


class ProviderTeamMinutesTests(unittest.TestCase):
    """D-053: NCAA overtime certified from provider team minutes, with the player-minute sum as fallback."""

    def _games(self, minutes_col):
        from fpp.data.assemble import build_ncaa_games

        class Fake:
            def ncaa_gamelogs(self):
                rows = []
                for gid, per_player in ((1, 20.0), (2, 22.5), (3, 20.0), (4, 21.3)):     # sums: 200, 225, 200, 213
                    for i in range(10):
                        rows.append({"player_id": gid * 100 + i, "game_id": gid, "season": 2023, "date": "2022-12-01", "team_id": 160,
                                     "opp_id": 9, "status": "Bench", "minutes": per_player, "pts": 6, "fg2a": 3, "fg2m": 3, "fg3a": 0, "fg3m": 0,
                                     "fta": 0, "ftm": 0, "orb": 1, "drb": 1, "ast": 1, "stl": 0, "blk": 0, "tov": 1, "pf": 1, "game_type": "regular_season"})
                return pd.DataFrame(rows)

            def ncaa_team_games(self):
                t = pd.DataFrame({"game_id": [1, 1, 2, 2, 3, 3, 4, 4], "team_id": [160, 9] * 4, "season": 2023, "home": [1, 0] * 4,
                                  "pts": [60, 55] * 4, "poss": 65.0, "game_type": "regular_season"})
                if minutes_col is not None:
                    t["minutes"] = minutes_col
                return t

            def ncaa_team_ratings(self):
                return pd.DataFrame({"team_id": [160, 9], "season": [2023, 2023], "adj_o": [110.0, 100.0], "adj_d": [100.0, 105.0], "adj_pace": [68.0, 70.0]})

        games, _, team = build_ncaa_games(Fake(), lag_days=1)
        return games, team.set_index(["game_id", "team_id"])

    def test_provider_minutes_certify_and_fall_back(self):
        #                     game1 exact regulation, game2 exact 1 OT, game3 provider 224 (=1 OT, rounding) vs sum 200 (disagree),
        #                     game4 provider 213 (neither) -> sum path (213 is outside the sum tolerance too -> uncertified)
        games, team = self._games([200, 200, 225, 225, 224, 224, 213, 213])
        t = team.loc[(1, 160)]; self.assertEqual((t["overtime_periods"], t["overtime_source"]), (0.0, "provider_team_minutes"))
        t = team.loc[(2, 160)]; self.assertEqual((t["overtime_periods"], t["overtime_source"], bool(t["overtime_agrees"])), (1.0, "provider_team_minutes", True))
        t = team.loc[(3, 160)]; self.assertEqual((t["overtime_periods"], t["overtime_source"], bool(t["overtime_agrees"])), (1.0, "provider_team_minutes", False))
        t = team.loc[(4, 160)]; self.assertEqual(t["overtime_source"], "none"); self.assertTrue(np.isnan(t["overtime_periods"]))
        g = games[(games["game_id"] == 2) & (games["team_id"] == 160)].iloc[0]
        self.assertEqual(float(g["overtime_periods"]), 1.0)

    def test_without_the_column_the_sum_path_is_unchanged(self):
        games, team = self._games(None)
        t = team.loc[(2, 160)]; self.assertEqual((t["overtime_periods"], t["overtime_source"]), (1.0, "player_minute_sum"))
        t = team.loc[(4, 160)]; self.assertEqual(t["overtime_source"], "none")


class LeakedRankTests(unittest.TestCase):
    def test_rank_above_limit_is_nulled_but_the_row_is_kept(self):
        from fpp.data.assemble import RANK_LIMIT, _latest_before
        import numpy as np
        rec = pd.DataFrame({"player_id": [1, 2], "class_year": [2015, 2015], "national_rank": [50.0, RANK_LIMIT + 1000.0],
                            "pos_rank": [5.0, 300.0], "composite_score": [0.99, 0.80], "star_rating": [5.0, 3.0]})
        leaked = rec["national_rank"] > RANK_LIMIT
        rec.loc[leaked, "national_rank"] = np.nan; rec.loc[leaked, "pos_rank"] = np.nan          # the assembler's rule
        units = pd.DataFrame({"player_id": [1, 2], "season": [2016, 2016]})
        out = _latest_before(units, rec, on="player_id", time_col="class_year", limit=units["season"] - 1,
                             columns=["national_rank", "pos_rank", "composite_score", "star_rating", "class_year"])
        self.assertEqual(out.loc[0, "recruit_national_rank"], 50.0)
        self.assertTrue(np.isnan(out.loc[1, "recruit_national_rank"]))
        self.assertEqual((out.loc[1, "recruit_composite_score"], out.loc[1, "recruit_star_rating"]), (0.80, 3.0))   # kept
