"""D-062: season-balanced loss weights with a first-season mass share, against the registered player-balanced rule."""
import unittest

import numpy as np
import pandas as pd

from fpp.model.dataset import FoldDataset, example_weights, first_year_flags, stratum_weights, validate_weighting


def _units():
    # Player 1: first season 2023 then 2024; player 2: one season (2023, first); player 3: two later seasons only.
    return pd.DataFrame({"unit_id": ["1:2023", "1:2024", "2:2023", "3:2023", "3:2024"], "player_id": [1, 1, 2, 3, 3],
                         "season": [2023, 2024, 2023, 2023, 2024], "first_year": [True, False, True, False, False]})


def _windows():
    # Two nested windows for 1:2023, one for 1:2024, one for 3:2024.
    return pd.DataFrame({"unit_id": ["1:2023", "1:2023", "1:2024", "3:2024"], "player_id": [1, 1, 1, 3],
                         "season": [2023, 2023, 2024, 2024], "k": [(1,), (2,), (1,), (1,)]})


class StratumWeightTests(unittest.TestCase):
    def test_share_is_enforced_and_natural_share_is_untouched(self):
        base = np.array([1.0, 1.0, 1.0, 1.0])
        fy = np.array([True, False, False, False])
        w = stratum_weights(base, fy, 0.5)
        self.assertAlmostEqual(w[fy].sum() / w.sum(), 0.5)
        self.assertAlmostEqual(w.sum(), base.sum())                                   # total mass preserved
        np.testing.assert_array_equal(stratum_weights(base, fy, None), base)
        np.testing.assert_array_equal(stratum_weights(base, np.ones(4, dtype=bool), 0.5), base)   # single stratum: unchanged
        np.testing.assert_array_equal(stratum_weights(base, np.zeros(4, dtype=bool), 0.5), base)

    def test_first_year_flags_default_to_false_without_the_column(self):
        self.assertFalse(first_year_flags(pd.DataFrame({"unit_id": ["a"]})).any())
        np.testing.assert_array_equal(first_year_flags(_units()), [True, False, True, False, False])


class ExampleWeightTests(unittest.TestCase):
    def test_player_balanced_rule_is_unchanged(self):
        w_fwd, w_rec = example_weights(_units(), _windows())
        # Base: players -> seasons: (1/2, 1/2, 1, 1/2, 1/2) over 3 players; rescaled to mean one.
        expected = np.array([0.5, 0.5, 1.0, 0.5, 0.5]) / 3
        np.testing.assert_allclose(w_fwd, expected * len(expected) / expected.sum())
        self.assertAlmostEqual(w_fwd.mean(), 1.0)
        # Reconstruction: players 1 (two seasons) and 3 (one): 1:2023 windows 1/(2*2*2) each, 1:2024 1/(2*2), 3:2024 1/2.
        base = np.array([1 / 8, 1 / 8, 1 / 4, 1 / 2])
        np.testing.assert_allclose(w_rec, base * len(base) / base.sum())
        self.assertAlmostEqual(w_rec.mean(), 1.0)

    def test_season_balanced_with_first_year_share(self):
        units, windows = _units(), _windows()
        w_fwd, w_rec = example_weights(units, windows, weighting="season_balanced", first_year_share=0.5)
        fy = first_year_flags(units)
        self.assertAlmostEqual(w_fwd[fy].sum() / w_fwd.sum(), 0.5)
        self.assertAlmostEqual(w_fwd.mean(), 1.0)
        np.testing.assert_allclose(w_fwd[fy], w_fwd[fy][0])                          # equal within the first-season stratum
        np.testing.assert_allclose(w_fwd[~fy], w_fwd[~fy][0])                        # and within the later-season stratum
        # Reconstruction: unit mass split over windows, first-season windows carry half the direction's mass.
        fy_w = np.array([True, True, False, False])
        self.assertAlmostEqual(w_rec[fy_w].sum() / w_rec.sum(), 0.5)
        self.assertAlmostEqual(w_rec[0], w_rec[1])                                   # the two nested windows of 1:2023 share its mass
        self.assertAlmostEqual(w_rec[2], w_rec[3])                                   # one window each: equal
        self.assertAlmostEqual(w_rec.mean(), 1.0)

    def test_season_balanced_natural_share_is_uniform(self):
        w_fwd, w_rec = example_weights(_units(), _windows(), weighting="season_balanced")
        np.testing.assert_allclose(w_fwd, 1.0)
        base = np.array([0.5, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(w_rec, base * len(base) / base.sum())

    def test_invalid_weighting_and_share_are_rejected(self):
        with self.assertRaises(ValueError):
            example_weights(_units(), _windows(), weighting="career_balanced")
        for bad in (0.0, 1.0, -0.2, 1.5):
            with self.assertRaises(ValueError):
                validate_weighting("season_balanced", bad)
        validate_weighting("season_balanced", None)
        validate_weighting("player_balanced", 0.5)
        self.assertEqual(FoldDataset.WEIGHTINGS, ("player_balanced", "season_balanced"))


if __name__ == "__main__":
    unittest.main()
