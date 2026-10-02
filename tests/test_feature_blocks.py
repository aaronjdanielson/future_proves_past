"""D-065: feature schemas with optional blocks, and the data-side blocks (usage/role, rolling RAPM, prior league,
destination context, international RAPM anchor, On3 fill)."""
import unittest

import numpy as np
import pandas as pd
import torch

from fpp.data.assemble import (PRIOR_LEAGUE_BY_ROLE, attach_role_fields, attach_rolling_rapm, attach_unit_blocks,
                               destination_context, prior_league_table)
from fpp.model import features as F
from fpp.model.tower import ReferenceTower
from tests.test_pool_and_arms import _pool_games


class SchemaTests(unittest.TestCase):
    def test_schema_widths(self):
        v7, v8, v9 = F.feature_schema("v7"), F.feature_schema("v8"), F.feature_schema("v9")
        self.assertEqual((len(v7.continuous), len(v7.static_continuous), len(v7.static_categorical)), (39, 12, 4))
        self.assertEqual(v7.field_names, F.FIELD_NAMES)
        self.assertEqual(v7.static_names, F.STATIC_CONTINUOUS)
        self.assertEqual(list(v8.static_categorical), F.STATIC_CATEGORICAL_V8)
        self.assertEqual(len(v9.continuous), 39 + 9 + 4)
        self.assertEqual(len(v9.static_continuous), 12 + 1 + 15 + 4 + 1)
        self.assertEqual(list(v9.static_categorical), F.STATIC_CATEGORICAL_V8 + ["prior_league"])
        self.assertEqual(v9.blocks, F.BLOCKS)
        sub = F.feature_schema("v9", ("on3", "usage"))
        self.assertEqual(sub.blocks, ("usage", "on3"))                                  # canonical order
        self.assertEqual((len(sub.continuous), len(sub.static_continuous), len(sub.static_categorical)), (48, 13, 5))
        self.assertEqual(sub.field_names[:39], F.FIELD_NAMES)                            # base fields keep their positions
        with self.assertRaises(ValueError):
            F.feature_schema("v9", ("nope",))
        with self.assertRaises(ValueError):
            F.feature_schema("v8", ("usage",))
        self.assertEqual(F.feature_schema("v9", ()).describe()["game_fields"], 39)

    def test_schema_from_config_and_auto(self):
        units_v8 = pd.DataFrame({"recruit_status": ["ranked"]})
        units_v7 = pd.DataFrame({"role": ["freshman"]})
        self.assertEqual(F.schema_from_config({}, units_v8).name, "v8")
        self.assertEqual(F.schema_from_config(None, units_v7).name, "v7")
        rec = F.feature_schema("v9", ("usage",)).describe()
        self.assertEqual(F.schema_from_config({"feature_schema": rec}, units_v7).blocks, ("usage",))
        self.assertEqual(F.resolve_schema("auto", None, units_v8).name, "v8")
        with self.assertRaises(ValueError):
            F.resolve_schema("auto", ("usage",), units_v8)

    def test_static_continuous_reproduces_the_registered_columns(self):
        units = pd.DataFrame({"age_at_cutoff": [18.5, np.nan], "ncaa_seasons_completed": [0, 2], "first_year": [True, False],
                              "height_cm": [200.0, 190.0], "weight_kg": [90.0, np.nan], "recruit_national_rank": [10.0, np.nan],
                              "recruit_composite_score": [0.99, np.nan], "recruit_star_rating": [5.0, np.nan],
                              "dest_prior_adj_o": [110.0, 100.0], "dest_prior_adj_d": [95.0, 100.0], "dest_prior_adj_pace": [70.0, 65.0],
                              "dest_prior_games": [31.0, np.nan]})
        z = F.static_continuous(units)
        self.assertEqual(z.shape, (2, 12))
        np.testing.assert_allclose(z[0], [18.5, 0, 1, 200, 90, np.log1p(10), 0.99, 5, 110, 95, 70, np.log1p(31)])
        self.assertTrue(np.isnan(z[1, [0, 4, 5, 6, 7, 11]]).all())
        z9 = F.static_continuous(units, F.feature_schema("v9"))
        self.assertEqual(z9.shape, (2, 33))
        np.testing.assert_allclose(z9[:, :12], z)
        self.assertTrue(np.isnan(z9[:, 12:]).all())                                     # block columns absent: masked

    def test_store_channel_and_tower_under_v9(self):
        games = _pool_games(n_players=2, n_games=5)
        fs = F.feature_schema("v9")
        store = F.GameStore.from_frame(games, schema=fs)
        self.assertEqual(store.raw.shape[1], 52)
        self.assertEqual(len(store.missing_fields), 13)                                  # the fixture predates the blocks
        cutoff = float(store.date.max()) + 1.0
        rows = store.rows(1)
        norm = F.Normalizer.fit(store, np.arange(len(games)), cutoff, np.zeros((2, 33)))
        ch = F.build_channel(store, rows, cutoff_days=cutoff, target_days=cutoff, normalizer=norm)
        self.assertEqual(ch.x.shape[1], 52)
        self.assertFalse(ch.mask[:, 39:].any())                                          # block fields masked, base fields not
        self.assertTrue(ch.mask[:, 0].all())
        t_v, t_m = F.time_basis(-1.0, 18.0, 19.0)
        targets = {"games": 3, "starts": 1, "minutes": 60.0, "counts": np.zeros(13, dtype=np.int64), "overtime": 0.0,
                   "overtime_known": True, "schedule": 5}
        ex = F.Example("u", "-", (), ch, F.Channel.empty(52), np.zeros(33, np.float32), np.zeros(33, bool), np.zeros(6, np.int32),
                       t_v, t_m, targets)
        batch = F.collate([ex, ex])
        self.assertEqual(tuple(batch["intl"]["x"].shape[1:]), (5, 52))
        self.assertEqual(tuple(batch["ncaa"]["x"].shape[1:]), (1, 52))
        tower = ReferenceTower([len(store.vocab[c]) for c in F.CATEGORICAL], [2] * 6, n_fields=52, n_static=33)
        with torch.no_grad():
            params = tower(batch)
        self.assertIn("pi_g", params)
        self.assertEqual(params["pi_g"].shape[0], 2)


class RoleFieldTests(unittest.TestCase):
    def test_usage_shares_rank_and_contributors(self):
        g = pd.DataFrame({"game_id": [1, 1, 1, 1], "team_id": [10, 10, 10, 20], "minutes": [30.0, 10.0, 0.0, 40.0],
                          "a2": [8, 2, 0, 10], "a3": [2, 0, 0, 5], "af": [5, 0, 0, 0], "tov": [3, 1, 0, 2], "ast": [4, 0, 0, 8],
                          "orb": [1, 0, 0, 2], "drb": [5, 1, 0, 6], "pts": [20, 4, 0, 30],
                          "team_pts": [70, 70, 70, 60], "team_minutes": [200.0, 200.0, 200.0, 200.0],
                          "team_fga": [50, 50, 50, 60], "team_fta": [20, 20, 20, 10], "team_tov": [12, 12, 12, 8],
                          "team_ast": [15, 15, 15, 16], "team_reb": [35, 35, 35, 40], "team_poss": [70.0, 70.0, 70.0, 70.0]})
        out = attach_role_fields(g)
        used = (10 + 0.44 * 5 + 3) * (200 / 5)
        team_used = 50 + 0.44 * 20 + 12
        self.assertAlmostEqual(out.loc[0, "usage_game"], 100 * used / (30 * team_used))
        self.assertTrue(np.isnan(out.loc[2, "usage_game"]))                               # no minutes: undefined
        self.assertAlmostEqual(out.loc[0, "pts_share"], 20 / 70)
        self.assertAlmostEqual(out.loc[3, "fga_share"], 15 / 60)
        self.assertAlmostEqual(out.loc[0, "reb_share"], 6 / 35)
        np.testing.assert_array_equal(out["min_rank_on_team"].to_numpy()[[0, 1, 3]], [1, 2, 1])
        self.assertTrue(np.isnan(out.loc[2, "min_rank_on_team"]))
        np.testing.assert_array_equal(out["n_team_contributors"].to_numpy(), [2, 2, 2, 1])

    def test_shares_are_clipped_and_missing_totals_stay_nan(self):
        g = pd.DataFrame({"game_id": [1], "team_id": [10], "minutes": [20.0], "a2": [30], "a3": [0], "af": [0], "tov": [0], "ast": [0],
                          "orb": [0], "drb": [0], "pts": [90], "team_pts": [70], "team_minutes": [200.0], "team_fga": [20], "team_fta": [0],
                          "team_tov": [0], "team_ast": [np.nan], "team_reb": [0], "team_poss": [np.nan]})
        out = attach_role_fields(g)
        self.assertEqual(out.loc[0, "pts_share"], 1.0)
        self.assertEqual(out.loc[0, "fga_share"], 1.0)
        self.assertTrue(np.isnan(out.loc[0, "ast_share"]))
        self.assertTrue(np.isnan(out.loc[0, "reb_share"]))                                # zero denominator: undefined
        self.assertEqual(out.loc[0, "usage_game"], 100.0)


class RollingRapmTests(unittest.TestCase):
    def test_latest_snapshot_strictly_before_the_game_in_the_same_league_season(self):
        games = pd.DataFrame({"player_id": [1, 1, 1, 2], "original_label": [2020, 2020, 2020, 2020], "competition_id": [5, 5, 6, 5],
                              "date": pd.to_datetime(["2020-01-10", "2020-01-20", "2020-01-20", "2020-01-20"])})
        rolling = pd.DataFrame({"player_id": [1, 1, 1], "season": [2020, 2020, 2020], "league_id": [5, 5, 5],
                                "cutoff_date": ["2020-01-05", "2020-01-10", "2020-01-19"], "orapm": [0.1, 0.2, 0.3],
                                "drapm": [-0.1, -0.2, -0.3], "rapm": [0.0, 0.0, 0.0], "off_equiv": [10.0, 20.0, 30.0]})
        out = attach_rolling_rapm(games, rolling)
        self.assertAlmostEqual(out.loc[0, "rapm_orapm"], 0.1)                            # the 01-10 snapshot is not before 01-10
        self.assertEqual(out.loc[0, "rapm_snapshot_age_days"], 5)
        self.assertAlmostEqual(out.loc[1, "rapm_orapm"], 0.3)
        self.assertAlmostEqual(out.loc[1, "rapm_off_equiv"], 30.0)
        self.assertTrue(np.isnan(out.loc[2, "rapm_orapm"]))                               # other league
        self.assertTrue(np.isnan(out.loc[3, "rapm_orapm"]))                               # other player
        none = attach_rolling_rapm(games.copy(), None)
        self.assertTrue(none["rapm_orapm"].isna().all())


class _Stub:
    def __init__(self, rosters, summaries, rapm, team_seasons, prior=None):
        self._r, self._s, self._p, self._t, self._pl = rosters, summaries, rapm, team_seasons, prior

    def rosters(self):
        return self._r.copy()

    def ncaa_summaries(self):
        return self._s.copy()

    def player_rapm(self):
        return None if self._p is None else self._p.copy()

    def team_seasons(self):
        return self._t.copy()

    def player_prior_league(self):
        return None if self._pl is None else self._pl.copy()


class DestinationContextTests(unittest.TestCase):
    def test_returning_shares_rapm_and_style(self):
        rosters = pd.DataFrame({"team_id": [1, 1, 1, 2], "season": [2024, 2024, 2024, 2024], "player_id": [11, 12, 13, 21]})
        # 2023 lines: team 1 had players 11 (returns), 12 (returns, no minutes), 14 (left); player 13 is new; team 2 has no 2023 lines.
        summaries = pd.DataFrame({"player_id": [11, 12, 14, 13], "season": [2023, 2023, 2023, 2023], "team_id": [1, 1, 1, 3],
                                  "gp": [30, 5, 30, 30], "gs": [30, 0, 10, 30], "minutes": [900.0, 0.0, 600.0, 800.0],
                                  "pts": [400.0, 0.0, 200.0, 300.0], "trb": [150.0, 0.0, 100.0, 100.0]})
        rapm = pd.DataFrame({"player_id": [11, 14], "season": [2023, 2023], "orapm": [1.0, 3.0], "drapm": [0.5, 1.0], "rapm": [1.5, 4.0], "n_poss": [1000, 900]})
        ts = pd.DataFrame({"team_id": [1, 1], "season": [2023, 2024], "fg3_rate": [0.4, 0.5], "fta_rate": [0.3, 0.2], "to_pct": [0.18, 0.15],
                           "oreb_pct": [0.3, 0.25], "dreb_pct": [0.7, 0.75]})
        out = destination_context(_Stub(rosters, summaries, rapm, ts), (2024, 2024)).set_index(["team_id", "season"])
        row = out.loc[(1, 2024)]
        self.assertAlmostEqual(row["dest_ret_min_share"], 900 / 1500)
        self.assertAlmostEqual(row["dest_ret_pts_share"], 400 / 600)
        self.assertAlmostEqual(row["dest_ret_starts_share"], 30 / 40)
        self.assertEqual(row["dest_ret_players"], 2)
        self.assertEqual(row["dest_ret_rapm_n"], 1)
        self.assertAlmostEqual(row["dest_ret_rapm_mean"], 1.5)                            # only player 11 carries RAPM
        self.assertEqual(row["dest_ret_measurable"], 1.0)
        self.assertAlmostEqual(row["dest_fg3_rate"], 0.4)                                 # the prior season's style
        new = out.loc[(2, 2024)]
        self.assertEqual(new["dest_ret_measurable"], 0.0)
        self.assertTrue(np.isnan(new["dest_ret_min_share"]))
        self.assertTrue(np.isnan(new["dest_fg3_rate"]))


class UnitBlockTests(unittest.TestCase):
    def test_prior_league_pooling_and_role_fallback(self):
        pl = pd.DataFrame({"player_id": [1, 2, 3], "season": [2024, 2024, 2024], "prior_league": ["juco_naia_other", "unknown", None],
                           "prior_league_known": [1, 0, None]})
        table = prior_league_table(_Stub(None, None, None, None, pl))
        self.assertEqual(list(table["prior_league"]), ["juco_naia_other", "hs_or_none", "hs_or_none"])
        self.assertEqual(list(table["prior_league_known"]), [1, 0, 0])
        units = pd.DataFrame({"player_id": [1, 4, 5], "season": [2024, 2024, 2027], "team_id": [1, 1, 1], "role": ["transfer_nond1", "returning", "freshman"],
                              "recruit_national_rank": [np.nan, np.nan, np.nan], "recruit_star_rating": [np.nan, np.nan, np.nan]})
        out = attach_unit_blocks(units, {"player_prior_league": table})
        self.assertEqual(list(out["prior_league"]), ["juco_naia_other", "ncaa_d1", "hs_or_none"])
        self.assertEqual(list(out["prior_league_known"]), [1, 1, 0])
        self.assertEqual(PRIOR_LEAGUE_BY_ROLE["freshman"], ("hs_or_none", 0))

    def test_intl_anchor_and_on3_fill(self):
        units = pd.DataFrame({"player_id": [1, 2, 3], "season": [2024, 2024, 2024], "team_id": [1, 1, 1], "role": ["freshman"] * 3,
                              "recruit_national_rank": [np.nan, 50.0, np.nan], "recruit_star_rating": [np.nan, 4.0, np.nan]})
        anchors = pd.DataFrame({"player_id": [1, 1, 1], "season": [2022, 2023, 2024], "orapm": [0.1, 0.2, 0.9], "drapm": [0.0, -0.1, 0.5],
                                "rapm": [0.1, 0.1, 1.4], "n_poss": [200.0, 300.0, 400.0], "n_leagues": [1, 1, 1]})
        on3 = pd.DataFrame({"player_id": [1, 2, 3], "class_year": [2023, 2023, 2024], "on3_rank": [120.0, 30.0, 5.0], "on3_stars": [3.0, 5.0, 5.0],
                            "on3_rating": [88.0, 95.0, 99.0]})
        out = attach_unit_blocks(units, {"intl_rapm_anchor": anchors, "on3_rankings": on3})
        self.assertAlmostEqual(out.loc[0, "intl_rapm_orapm"], 0.2)                       # season 2024 is not before the unit
        self.assertEqual(out.loc[0, "intl_rapm_season_gap"], 1)
        self.assertAlmostEqual(out.loc[0, "intl_rapm_poss"], 300.0)
        self.assertTrue(np.isnan(out.loc[1, "intl_rapm_orapm"]))
        self.assertEqual(out.loc[0, "recruit_national_rank"], 120.0)                     # filled from On3
        self.assertEqual(out.loc[0, "recruit_rank_from_on3"], 1.0)
        self.assertEqual(out.loc[0, "recruit_star_rating"], 3.0)
        self.assertEqual(out.loc[1, "recruit_national_rank"], 50.0)                      # 247 rank kept
        self.assertEqual(out.loc[1, "recruit_rank_from_on3"], 0.0)
        self.assertTrue(np.isnan(out.loc[2, "recruit_national_rank"]))                    # class 2024 is not before season 2024
        self.assertEqual(out.loc[2, "recruit_rank_from_on3"], 0.0)

    def test_national_team_only_is_kept_as_its_own_level(self):
        pl = pd.DataFrame({"player_id": [1, 2], "season": [2027, 2027], "prior_league": ["national_team_only", "intl"],
                           "prior_league_known": [1, 1]})
        table = prior_league_table(_Stub(None, None, None, None, pl))
        self.assertEqual(list(table["prior_league"]), ["national_team_only", "intl"])

    def test_strict_control_neutralises_post_first_season_summaries(self):
        from fpp.model.dataset import mask_post_first_season_statics
        units = pd.DataFrame({"player_id": [1, 1, 2, 3], "season": [2022, 2025, 2023, 2024], "first_season": [2022, 2022, 2023, 2022],
                              "first_year": [True, False, True, False],
                              "prior_league": ["intl", "intl", "national_team_only", "juco_naia_other"], "prior_league_known": [1, 1, 1, 1],
                              "intl_rapm_orapm": [0.5, 0.7, np.nan, 0.2], "intl_rapm_drapm": [0.1, 0.2, np.nan, 0.0],
                              "intl_rapm_poss": [300.0, 500.0, np.nan, 100.0], "intl_rapm_season_gap": [1, 1, np.nan, 3]})
        out = mask_post_first_season_statics(units)
        # First-season units keep everything; the returner's intl label and post-first-season anchor are neutralised.
        self.assertEqual(list(out["prior_league"]), ["intl", "hs_or_none", "national_team_only", "juco_naia_other"])
        self.assertEqual(list(out["prior_league_known"]), [1, 0, 1, 1])
        self.assertAlmostEqual(out.loc[0, "intl_rapm_orapm"], 0.5)
        self.assertTrue(np.isnan(out.loc[1, "intl_rapm_orapm"]))                          # anchor 2024 >= first season 2022
        self.assertAlmostEqual(out.loc[3, "intl_rapm_orapm"], 0.2)                        # anchor 2021 < first season 2022
        v7 = pd.DataFrame({"player_id": [1], "season": [2022], "first_year": [False]})
        self.assertIs(mask_post_first_season_statics(v7), v7)                              # v7 tables: untouched

    def test_career_history_uses_every_roster_season(self):
        from fpp.data.assemble import career_history
        all_rosters = pd.DataFrame({"player_id": [1, 1, 1, 2, 2, 3], "season": [2001, 2002, 2003, 2003, 2004, 2027],
                                    "team_id": [5, 5, 5, 6, 6, 7]})
        units = pd.DataFrame({"player_id": [1, 2, 2], "season": [2003, 2003, 2004], "first_season": [2003, 2003, 2003]})
        out = career_history(all_rosters, units)
        self.assertEqual(list(out["first_season"]), [2001, 2003, 2003])
        self.assertEqual(list(out["first_year"]), [False, True, False])                 # the 2003 returner is not a first season
        self.assertEqual(list(out["ncaa_seasons_completed"]), [2, 0, 1])

    def test_career_history_normalizes_first_year_roles(self):
        """D-074: a first D1 season in a freshman class is a freshman whatever the upstream role rule read from tracked
        international games; later-year units and non-freshman classes keep the upstream role."""
        from fpp.data.assemble import career_history
        all_rosters = pd.DataFrame({"player_id": [1, 2, 2, 3, 4, 5], "season": [2027, 2026, 2027, 2027, 2027, 2027],
                                    "team_id": [5, 6, 7, 8, 9, 10]})
        units = pd.DataFrame({"player_id": [1, 2, 3, 4, 5], "season": [2027] * 5,
                              "role": ["transfer_d1", "transfer_d1", "transfer_d1", "transfer_nond1", "transfer_d1"],
                              "class": ["Fr", "Fr", "Jr", "Jr", "RS-Fr"]})
        out = career_history(all_rosters, units)
        self.assertEqual(list(out["first_year"]), [True, False, True, True, True])
        self.assertEqual(list(out["role"]), ["freshman", "transfer_d1", "transfer_d1", "transfer_nond1", "freshman"])
        self.assertEqual(list(out["role_upstream"]), ["transfer_d1", "transfer_d1", "transfer_d1", "transfer_nond1", "transfer_d1"])
        no_role = career_history(all_rosters, units.drop(columns=["role", "class"]))
        self.assertNotIn("role_upstream", no_role)                                      # fixtures without roles: unchanged

    def test_strict_mask_boundary_for_a_first_season_before_the_tables(self):
        from fpp.model.dataset import post_first_season_mask
        games = _pool_games(n_players=1, n_games=4)                                     # non-NCAA games 2021-10-01 .. 2021-12-10
        store = F.GameStore.from_frame(games)
        base = {"unit_id": ["1:2023"], "player_id": [1], "season": [2023],
                "period_start": pd.to_datetime(["2022-11-07"]), "period_end": pd.to_datetime(["2023-03-10"])}
        in_range = post_first_season_mask(store, pd.DataFrame({**base, "first_season": [2023]}))
        self.assertFalse(in_range.any())                                                # games precede the first season
        earlier = post_first_season_mask(store, pd.DataFrame({**base, "first_season": [2021]}))
        self.assertTrue(earlier.all())                                                  # after 30 April 2021: post-first-season

    def test_missing_tables_leave_columns_absent(self):
        units = pd.DataFrame({"player_id": [1], "season": [2024], "team_id": [1], "role": ["freshman"],
                              "recruit_national_rank": [np.nan], "recruit_star_rating": [np.nan]})
        out = attach_unit_blocks(units, {})
        self.assertNotIn("prior_league", out)
        self.assertNotIn("dest_ret_min_share", out)


if __name__ == "__main__":
    unittest.main()
