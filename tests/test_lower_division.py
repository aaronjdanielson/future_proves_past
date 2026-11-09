"""D-101: the lower-division season lines and the static ``ld_*`` fields of first-year units."""
import unittest

import numpy as np
import pandas as pd

from fpp.data import assemble as A

COUNTS = {c: 0 for c in A.LOWER_DIV_COUNTS}


def line(pid, season, team, school, level, **counts):
    return {"player_id": pid, "season": season, "team_id": team, "school": school, "level": level, **{**COUNTS, **counts}}


class LowerDivisionLines(unittest.TestCase):
    def test_level_rule_and_mid_season_move(self):
        raw = pd.DataFrame([
            line(1, 2020, -5, "Pearl River", "JUCO", GP=30, MIN=600, PTS=300),
            line(1, 2021, 77, "Duke", "D1", GP=20, MIN=200, PTS=50),             # a D1 row: never lower-division
            line(2, 2021, 1550, "Lynn", "D2", GP=10, MIN=200, PTS=80),
            line(2, 2021, 1551, "Tampa", "D2", GP=12, MIN=250, PTS=100),         # moved mid-season: summed
            line(3, 2022, 77, "Duke", None, GP=25, MIN=500, PTS=200),            # missing Level on a D1 team id: dropped
            line(4, 2022, 9001, "Chipola", None, GP=28, MIN=700, PTS=400),       # missing Level, never D1: kept
            line(5, 2022, -9, "Kilgore", "OTHER", GP=0, MIN=0, PTS=0),           # listed without stats: dropped
        ])
        out = A.lower_division_lines(raw).set_index(["player_id", "season"])
        self.assertEqual(sorted(out.index), [(1, 2020), (2, 2021), (4, 2022)])
        self.assertEqual(out.loc[(2, 2021), "GP"], 22)
        self.assertEqual(out.loc[(2, 2021), "MIN"], 450)
        self.assertIsNone(A.lower_division_lines(None))


def jc(source, season, school, name, **counts):
    return {"source": source, "season": season, "school": school, "name": name, **{**COUNTS, **counts}}


class JuniorCollegeMatch(unittest.TestCase):
    """RealGM rows that list a junior-college season without stats take the full-population line that matches on
    season, player name and school (D-101)."""

    def setUp(self):
        self.names = pd.DataFrame({"player_id": [20, 21, 22, 23, 24, 25, 26],
                                   "name": ["D.J. Smith", "Al Jones", "Bo Cole", "Chris Lee", "Chris Lee", "Ed Fry",
                                            "Gil Hart"]})

    def lines(self, rows):
        return A.lower_division_lines(pd.DataFrame(rows), pd.DataFrame(self.junior), self.names)

    def test_match_rules(self):
        self.junior = [
            jc("njcaa", 2016, "Paris", "DJ  Smith", GP=30, GS=None, MIN=800, PTS=400),        # matches player 20
            jc("njcaa", 2016, "Navarro", "Al Jones", GP=30, MIN=800, PTS=400),                # school differs from Tyler
            jc("njcaa", 2016, "Chipola", "Bo Cole", GP=30, MIN=800, PTS=400),
            jc("njcaa", 2016, "Chipola", "Bo Cole", GP=10, MIN=100, PTS=40),                  # two candidates: ambiguous
            jc("cccaa", 2016, "Tyler", "Chris Lee", GP=30, MIN=800, PTS=400),                 # claimed by two players
            jc("njcaa", 2016, "Pearl River", "Ed Fry", GP=30, MIN=800, PTS=999),              # RealGM has the stats
            jc("njcaa", 2016, "Kilgore", "Gil Hart", GP=12, MIN=300, PTS=150),                # second school, same season
        ]
        out = self.lines([
            line(20, 2016, -1, "Paris JC", "JUCO"),
            line(21, 2016, -2, "Tyler", "JUCO"),
            line(22, 2016, -3, "Chipola", "JUCO"),
            line(23, 2016, -4, "Tyler JC", "JUCO"), line(24, 2016, -4, "Tyler JC", "JUCO"),
            line(25, 2016, -5, "Pearl River", "JUCO", GP=28, MIN=700, PTS=350), line(25, 2016, -5, "Pearl River", "JUCO"),
            line(26, 2016, 1550, "Lynn", "D2", GP=10, MIN=200, PTS=80), line(26, 2016, -6, "Kilgore College", "OTHER"),
            line(27, 2016, -7, "DNP", "OTHER"),
        ]).set_index(["player_id", "season"])
        self.assertEqual(sorted(out.index), [(20, 2016), (25, 2016), (26, 2016)])
        self.assertEqual(out.loc[(20, 2016), "source"], "njcaa")
        self.assertTrue(np.isnan(out.loc[(20, 2016), "GS"]))                 # a blank count stays missing
        self.assertEqual(out.loc[(25, 2016), "PTS"], 350)                    # not doubled by the listed duplicate
        self.assertEqual(out.loc[(25, 2016), "source"], "realgm")
        self.assertEqual(out.loc[(26, 2016), "GP"], 22)                      # D2 stats + matched JUCO line summed
        self.assertEqual(out.loc[(26, 2016), "source"], "njcaa+realgm")

    def test_made_attempted_text_and_no_junior_table(self):
        raw = pd.DataFrame([{"source": "njcaa", "season": 2015, "school": "Vincennes", "name": "Al Jones", "GP": "15",
                             "GS": "", "MIN": "275", "FG": "45-118", "FG3": "23-68", "FT": "9-16", "TRB": "40",
                             "AST": "10", "STL": "5", "BLK": "1", "TOV": "12", "PF": "20", "PTS": 122}])
        f = A.junior_college_frame(raw).iloc[0]
        self.assertEqual((f.FGM, f.FGA, f.FG3M, f.FG3A, f.FTM, f.FTA, f.MIN), (45, 118, 23, 68, 9, 16, 275))
        self.assertTrue(np.isnan(f.GS))
        rows = [line(21, 2015, -2, "Vincennes", "JUCO")]
        self.assertEqual(len(A.lower_division_lines(pd.DataFrame(rows))), 0)               # nothing with stats
        out = A.lower_division_lines(pd.DataFrame(rows), raw, self.names)
        self.assertEqual(out.iloc[0].PTS, 122)
        self.assertEqual(out.iloc[0].FGA, 118)


class AttachLowerDivision(unittest.TestCase):
    def setUp(self):
        self.lines = pd.DataFrame([
            {**COUNTS, "player_id": 10, "season": 2019, "GP": 30, "GS": 30, "MIN": 900, "PTS": 450, "FGA": 300, "FG3A": 90,
             "FTA": 100, "TRB": 180, "AST": 60, "STL": 30, "BLK": 15, "TOV": 45},
            {**COUNTS, "player_id": 10, "season": 2020, "GP": 32, "GS": 16, "MIN": 800, "PTS": 400, "FGA": 320, "FG3A": 80,
             "FTA": 80, "TRB": 160, "AST": 80, "STL": 20, "BLK": 20, "TOV": 40},
            {**COUNTS, "player_id": 11, "season": 2021, "GP": 5, "GS": 0, "MIN": 30, "PTS": 12, "FGA": 10},   # under 40 min
            {**COUNTS, "player_id": 12, "season": 2022, "GP": 30, "GS": 30, "MIN": 900, "PTS": 450},           # same season
        ])
        self.units = pd.DataFrame({"player_id": [10, 10, 11, 12, 13], "season": [2021, 2022, 2022, 2022, 2022],
                                   "first_year": [True, False, True, True, True]})

    def test_last_season_before_the_unit_first_year_only(self):
        u = A.attach_lower_division(self.units, self.lines)
        self.assertEqual(list(u.columns[-len(A.LOWER_DIV_COLUMNS):]), A.LOWER_DIV_COLUMNS)
        r = u.iloc[0]                                             # player 10, first D1 season 2021: last line 2020
        self.assertEqual(r.ld_gap, 1)
        self.assertEqual(r.ld_seasons, 2)
        self.assertEqual(r.ld_gp, 32)
        self.assertAlmostEqual(r.ld_mpg, 25.0)
        self.assertAlmostEqual(r.ld_gs_share, 0.5)
        self.assertAlmostEqual(r.ld_pts40, 20.0)
        self.assertAlmostEqual(r.ld_stlblk40, 2.0)
        self.assertAlmostEqual(r.ld_fg3a_share, 0.25)
        self.assertAlmostEqual(r.ld_fta_rate, 0.25)
        self.assertAlmostEqual(r.ld_ts, 400 / (2 * (320 + 0.44 * 80)))
        self.assertTrue(u.iloc[1][A.LOWER_DIV_COLUMNS].isna().all())          # later D1 season: not eligible
        small = u.iloc[2]                                                     # 30 minutes: counts kept, rates masked
        self.assertEqual(small.ld_gp, 5)
        self.assertTrue(np.isnan(small.ld_pts40) and np.isnan(small.ld_ts) and np.isnan(small.ld_fg3a_share))
        self.assertTrue(u.iloc[3][A.LOWER_DIV_COLUMNS].isna().all())          # a line of the unit's own season is not prior
        self.assertTrue(u.iloc[4][A.LOWER_DIV_COLUMNS].isna().all())          # no line at all

    def test_no_table_leaves_masked_columns_and_reattach_is_idempotent(self):
        u = A.attach_lower_division(self.units, None)
        self.assertTrue(u[A.LOWER_DIV_COLUMNS].isna().all().all())
        once = A.attach_lower_division(self.units, self.lines)
        twice = A.attach_lower_division(once, self.lines)
        pd.testing.assert_frame_equal(once, twice)

    def test_forecast_frame_without_first_year_column(self):
        u = A.attach_lower_division(self.units.drop(columns="first_year"), self.lines)
        self.assertEqual(u.iloc[1].ld_gap, 2)                                 # every row eligible


class LowerDivBlock(unittest.TestCase):
    def test_block_adds_fourteen_static_fields_read_from_the_units(self):
        from fpp.model import features as F
        base = F.feature_schema("v9")
        s = F.feature_schema("v9", F.V9_DEFAULT_BLOCKS + ("lower_div",))
        self.assertEqual(s.blocks, F.V9_DEFAULT_BLOCKS + ("lower_div",))
        self.assertEqual(len(s.continuous), len(base.continuous))                  # no game fields
        self.assertEqual(len(s.static_continuous), len(base.static_continuous) + 14)
        self.assertEqual([c for _, c, _ in s.static_continuous[-14:]], A.LOWER_DIV_COLUMNS)
        self.assertNotIn("lower_div", base.blocks)                                  # opt-in
        units = A.attach_lower_division(pd.DataFrame({"player_id": [10, 13], "season": [2021, 2021],
                                                      "first_year": [True, True]}),
                                        pd.DataFrame([{**COUNTS, "player_id": 10, "season": 2020, "GP": 31, "GS": 31,
                                                       "MIN": 900, "PTS": 450, "FGA": 300}]))
        z = F.static_continuous(units, s)[:, -14:]
        self.assertAlmostEqual(z[0, 2], np.log1p(31))                              # ld_gp_log
        self.assertAlmostEqual(z[0, 1], np.log1p(1))                               # ld_seasons_log
        self.assertTrue(np.isnan(z[1]).all())                                       # no line: masked


if __name__ == "__main__":
    unittest.main()
