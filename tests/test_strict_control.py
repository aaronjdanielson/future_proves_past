"""D-041: the strict control removes every non-NCAA game dated on/after the player's first NCAA season from the
input channels, the pool and reconstruction; freshman evaluation inputs are unchanged."""
import unittest

import numpy as np
import pandas as pd

from fpp.model import features as F
from fpp.model.dataset import FoldDataset, post_first_season_mask
from fpp.model.pool import PoolSampler
from tests.test_pool_and_arms import _pool_games


def _games():
    g = _pool_games(n_players=2, n_games=6)                      # intl games 2021-10-01 .. 2021-12-10, season 2022
    # Player 1 enters the NCAA in 2022-23 (period starts 2022-11-07); add two summer national-team games after
    # the first season and one NCAA game row.
    later = g[g["player_id"] == 1].iloc[:3].copy()
    later["record_id"] = ["national:901:1", "national:902:1", "national:903:1"]
    later["source"] = "national"
    # Two summer games after the first season (post-freshman) and one national-team window game played DURING
    # the first season (2023-02-24): the latter is not post-freshman under the research definition and stays.
    later["date"] = pd.to_datetime(["2023-07-01", "2023-07-03", "2023-02-24"])
    later["release_date"] = later["date"] + pd.Timedelta(days=1)
    later["season"] = [2024, 2024, 2023]
    ncaa = g[g["player_id"] == 1].iloc[:1].copy()
    ncaa["record_id"] = "ncaa:801:1"
    ncaa["source"] = "ncaa"
    ncaa["date"] = pd.Timestamp("2022-11-20")
    ncaa["release_date"] = pd.Timestamp("2022-11-21")
    ncaa["season"] = 2023
    return pd.concat([g, later, ncaa], ignore_index=True).sort_values(["player_id", "date"]).reset_index(drop=True)


def _units():
    return pd.DataFrame({"unit_id": ["1:2023", "1:2024"], "player_id": [1, 1], "season": [2023, 2024],
                         "period_start": pd.to_datetime(["2022-11-07", "2023-11-06"]),
                         "period_end": pd.to_datetime(["2023-03-10", "2024-03-09"]),
                         "forecast_cutoff": pd.to_datetime(["2022-10-01", "2023-10-01"])})


class StrictControlTests(unittest.TestCase):
    def test_mask_marks_only_post_first_season_non_ncaa_rows(self):
        games = _games()
        store = F.GameStore.from_frame(games)
        mask = post_first_season_mask(store, _units())
        rec = store.record_id
        self.assertTrue(all(mask[rec == r] for r in ("national:901:1", "national:902:1")))
        self.assertFalse(mask[rec == "national:903:1"][0])                               # in-season game: not post-freshman
        self.assertFalse(mask[rec == "ncaa:801:1"][0])                                   # NCAA rows are never masked
        self.assertFalse(mask[(store.player_id == 1) & (store.date < F._days(pd.Series([pd.Timestamp("2022-11-07")]))[0])].any())
        self.assertFalse(mask[store.player_id == 2].any())                                 # no NCAA season: nothing excluded

    def test_channels_and_pool_respect_the_mask(self):
        games = _games()
        store = F.GameStore.from_frame(games)
        units = _units()
        mask = post_first_season_mask(store, units)
        empty = pd.DataFrame(columns=["unit_id", "player_id", "season", "k", "record_ids"])
        strict = FoldDataset(None, store, None, None, units, units, empty, set(), mask)
        loose = FoldDataset(None, store, None, None, units, units, empty, set(), None)
        soph = units.iloc[1]
        cutoff = F._days(pd.Series([soph["forecast_cutoff"]]))[0]
        intl_strict, ncaa_strict = strict._channels(soph, cutoff)
        intl_loose, ncaa_loose = loose._channels(soph, cutoff)
        self.assertEqual(len(intl_loose) - len(intl_strict), 2)                          # the two summer games; the in-season game stays
        self.assertEqual(len(ncaa_strict), len(ncaa_loose))                                # NCAA channel untouched
        fresh = units.iloc[0]
        cutoff0 = F._days(pd.Series([fresh["forecast_cutoff"]]))[0]
        self.assertEqual(len(strict._channels(fresh, cutoff0)[0]), len(loose._channels(fresh, cutoff0)[0]))   # freshman inputs identical
        norm = F.Normalizer.fit(store, np.arange(len(games)), cutoff, np.zeros((2, len(F.STATIC_CONTINUOUS))))
        with_pool = PoolSampler(store, norm, cutoff_days=cutoff, heldout_players=set(), seed=1)
        without = PoolSampler(store, norm, cutoff_days=cutoff, heldout_players=set(), seed=1, excluded_rows=mask)
        self.assertEqual(int(with_pool.admissible.sum()) - int(without.admissible.sum()), 2)


if __name__ == "__main__":
    unittest.main()


class SubsetKeepsMaskTests(unittest.TestCase):
    def test_subset_and_examples_carry_the_exclusion(self):
        from fpp.experiments.train import _subset
        games = _games()
        store = F.GameStore.from_frame(games)
        units = _units()
        mask = post_first_season_mask(store, units)
        empty = pd.DataFrame(columns=["unit_id", "player_id", "season", "k", "record_ids"])
        ds = FoldDataset(None, store, None, None, units, units, empty, set(), mask)
        sub = _subset(ds, np.array([False, True]))
        self.assertIs(sub.excluded_rows, mask)
        soph = sub.train_units.iloc[0]
        cutoff = F._days(pd.Series([soph["forecast_cutoff"]]))[0]
        loose = FoldDataset(None, store, None, None, units, units, empty, set(), None)
        self.assertEqual(len(loose._channels(soph, cutoff)[0]) - len(sub._channels(soph, cutoff)[0]), 2)
