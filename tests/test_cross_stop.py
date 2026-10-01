"""D-067: cross-stopping splits the validation season's players into two fixed halves; the test season is untouched."""
import unittest

import numpy as np
import pandas as pd

from fpp.experiments.train import cross_stop_halves, season_masks


class CrossStopTests(unittest.TestCase):
    def test_halves_are_balanced_complements_and_seed_independent(self):
        players = list(range(100, 207))
        a0, a1 = cross_stop_halves(players)
        b0, b1 = cross_stop_halves(reversed(players))             # input order does not matter
        self.assertEqual((a0, a1), (b0, b1))
        self.assertFalse(a0 & a1)
        self.assertEqual(a0 | a1, set(players))
        self.assertLessEqual(abs(len(a0) - len(a1)), 1)

    def test_masks(self):
        tu = pd.DataFrame({"player_id": [1, 1, 2, 3, 3], "season": [2020, 2021, 2021, 2019, 2021]})
        train, stop = season_masks(tu, train_max_season=2020, stop_season=2021)
        np.testing.assert_array_equal(train, [True, False, False, True, False])
        np.testing.assert_array_equal(stop, [False, True, True, False, True])
        train, stop = season_masks(tu, train_max_season=2020, stop_season=2021, stop_players={1, 3})
        np.testing.assert_array_equal(train, [True, False, True, True, False])      # player 2's validation season trains
        np.testing.assert_array_equal(stop, [False, True, False, False, True])
        self.assertFalse((train & stop).any())


if __name__ == "__main__":
    unittest.main()
