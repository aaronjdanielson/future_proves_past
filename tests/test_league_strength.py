"""D-100: league strength on game rows (minus the cross-league RAPM calibration shift of the prior season) and its block."""
import unittest

import numpy as np
import pandas as pd

from fpp.data.assemble import attach_league_strength
from fpp.model import features as F


class LeagueStrengthTests(unittest.TestCase):
    def test_prior_season_shift_negated_on_international_rows(self):
        games = pd.DataFrame({"source": ["intl", "intl", "intl", "intl", "national", "intl"],
                              "competition_id": [4, 4, 4, 1, 4, 99],
                              "original_label": [2020, 2021, 2019, 2021, 2021, 2021]})
        shifts = pd.DataFrame({"league_id": [4, 4, 1, 1], "season": [2019, 2020, 2019, 2020],
                               "shift": [1.0, 1.2, 0.0, 0.0], "n_players": [300, 310, 130, 135], "pinned": [0, 0, 1, 1]})
        out = attach_league_strength(games, shifts)
        self.assertAlmostEqual(out.loc[0, "league_strength"], -1.0)       # the 2020 label reads the 2019 shift
        self.assertAlmostEqual(out.loc[1, "league_strength"], -1.2)
        self.assertTrue(np.isnan(out.loc[2, "league_strength"]))           # a league's first fitted season has no prior
        self.assertAlmostEqual(out.loc[3, "league_strength"], 0.0)         # the pinned reference league
        self.assertTrue(np.isnan(out.loc[4, "league_strength"]))           # national-team row
        self.assertTrue(np.isnan(out.loc[5, "league_strength"]))           # league without a fitted shift
        none = attach_league_strength(games.copy(), None)
        self.assertTrue(none["league_strength"].isna().all())

    def test_block_adds_one_game_field_and_nothing_static(self):
        s = F.feature_schema("v9", ("league_strength",))
        self.assertEqual(s.blocks, ("league_strength",))
        self.assertEqual(len(s.continuous), 39 + 1)
        self.assertEqual(s.field_names[-1], "league_strength")
        self.assertEqual(len(s.static_continuous), 12)
        full = F.feature_schema("v9", F.V9_DEFAULT_BLOCKS + ("league_strength",))
        self.assertEqual(len(full.continuous), 39 + 1)                         # the production set has no game blocks
        self.assertEqual(F.feature_schema("v9").blocks, F.V9_DEFAULT_BLOCKS)   # opt-in: the default set is unchanged


if __name__ == "__main__":
    unittest.main()
