"""D-063: one international starter flag from the page value and the listing-order value."""
import unittest

import numpy as np
import pandas as pd

from fpp.data.stores import STARTER_SOURCE, resolve_intl_starter


def _frame(page, derived, game=1, team=1):
    n = len(page)
    return pd.DataFrame({"GameID": [game] * n, "TeamID": [team] * n, "PlayerID": list(range(n)), "PTS": [0] * n,
                         "Starter": page, "StarterDerived": derived})


class StarterResolutionTests(unittest.TestCase):
    def test_complete_page_with_five_starters_wins(self):
        page = [1, 1, 1, 1, 1, 0, 0]
        derived = [1, 1, 1, 1, 0, 1, 0]                 # listing order disagrees on two rows
        out = resolve_intl_starter(_frame(page, derived))
        np.testing.assert_array_equal(out["Starter"].to_numpy(), page)
        self.assertTrue((out["starter_source"] == 1).all())
        self.assertNotIn("StarterDerived", out)

    def test_complete_page_with_wrong_count_falls_back_to_listing_order(self):
        page = [1, 1, 1, 1, 0, 0, 0]                    # four marked: a page error
        derived = [1, 1, 1, 1, 1, 0, 0]
        out = resolve_intl_starter(_frame(page, derived))
        np.testing.assert_array_equal(out["Starter"].to_numpy(), derived)
        self.assertTrue((out["starter_source"] == 2).all())

    def test_partial_page_keeps_its_values_and_fills_the_rest(self):
        page = [1, None, None, 0, None]                 # only linked players carry a page value
        derived = [1, 1, 1, 1, 1]
        out = resolve_intl_starter(_frame(page, derived))
        np.testing.assert_array_equal(out["Starter"].to_numpy(), [1, 1, 1, 0, 1])
        np.testing.assert_array_equal(out["starter_source"].to_numpy(), [1, 2, 2, 1, 2])

    def test_missing_both_is_unknown_and_missing_columns_are_tolerated(self):
        out = resolve_intl_starter(_frame([None, None], [None, 1]))
        self.assertTrue(np.isnan(out["Starter"].iloc[0]))
        self.assertEqual(out["Starter"].iloc[1], 1)
        np.testing.assert_array_equal(out["starter_source"].to_numpy(), [0, 2])
        bare = resolve_intl_starter(pd.DataFrame({"GameID": [1], "TeamID": [1], "PlayerID": [1], "PTS": [2]}))
        self.assertTrue(np.isnan(bare["Starter"].iloc[0]))
        self.assertEqual(STARTER_SOURCE[int(bare["starter_source"].iloc[0])], "none")

    def test_team_games_are_resolved_independently(self):
        a = _frame([1, 1, 1, 1, 0], [1, 1, 1, 1, 0], game=1, team=1)          # complete, four marked -> listing order
        b = _frame([1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1], game=1, team=2)    # complete, five marked -> page
        out = resolve_intl_starter(pd.concat([a, b], ignore_index=True))
        self.assertTrue((out.loc[out["TeamID"] == 1, "starter_source"] == 2).all())
        self.assertTrue((out.loc[out["TeamID"] == 2, "starter_source"] == 1).all())
        np.testing.assert_array_equal(out.loc[out["TeamID"] == 2, "Starter"].to_numpy(), [1, 1, 1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
