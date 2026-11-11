"""D-109: the 'near' reader for reconstruction windows — recency forward from the earliest post-target game, the cap keeps
the games nearest the season, and the time basis is anchored on a game the encoder sees. The registered 'latest' reader
is unchanged to the last digit."""
import unittest

import numpy as np
import pandas as pd
import pytest

from fpp.model import features as F
from tests.test_features import _games


def _many_games(n: int) -> pd.DataFrame:
    base = _games().iloc[0].to_dict()
    rows = []
    for i in range(n):
        d = pd.Timestamp("2021-10-01") + pd.Timedelta(days=4 * i)
        r = dict(base, record_id=f"intl:{i}:7", game_id=i, date=d, release_date=d + pd.Timedelta(days=1), season=2022 + (i // 60))
        rows.append(r)
    return pd.DataFrame(rows)


class BackwardReaderTests(unittest.TestCase):
    def setUp(self):
        self.games = _games()
        self.store = F.GameStore.from_frame(self.games, dobs=pd.Series({7: pd.Timestamp("2004-03-01")}))
        self.rows = self.store.rows(7)
        self.cutoff = F._days(pd.Series([pd.Timestamp("2024-09-30")]))[0]
        self.norm = F.Normalizer.fit(self.store, self.rows, self.cutoff, np.zeros((1, len(F.STATIC_CONTINUOUS))))

    def test_latest_reader_is_the_registered_channel(self):
        target = F._days(pd.Series([pd.Timestamp("2024-11-05")]))[0]
        a = F.build_channel(self.store, self.rows, cutoff_days=self.cutoff, target_days=target, normalizer=self.norm)
        b = F.build_channel(self.store, self.rows, cutoff_days=self.cutoff, target_days=target, normalizer=self.norm, reader="latest")
        for k in ("x", "mask", "clocks", "weight", "recency", "evidence"):
            np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
        self.assertAlmostEqual(float(a.recency[-1]), 0.0)
        self.assertEqual(a.anchor_days, float(self.store.date[self.rows[-1]]))
        with self.assertRaises(ValueError):
            F.build_channel(self.store, self.rows, cutoff_days=self.cutoff, target_days=target, normalizer=self.norm, reader="middle")

    def test_near_reader_anchors_on_the_game_nearest_the_season(self):
        target = F._days(pd.Series([pd.Timestamp("2022-06-01")]))[0]       # a season before every game: a backward window
        ch = F.build_channel(self.store, self.rows, cutoff_days=self.cutoff, target_days=target, normalizer=self.norm, reader="near")
        self.assertAlmostEqual(float(ch.recency[0]), 0.0)                    # earliest post-target game anchors recency
        self.assertTrue(np.all(np.diff(ch.recency) > 0))                      # growing with time after the season
        self.assertEqual(ch.anchor_days, float(self.store.date[self.rows[0]]))
        self.assertTrue(np.all(ch.clocks[:, 0] > 0))                          # tau still the true signed distance to the target
        self.assertAlmostEqual(float(ch.evidence[4]), (target - ch.anchor_days) / F.YEAR_DAYS, places=5)   # gap to the nearest game
        # The pooling weights the tower derives, 2^(-recency / half-life), now favour the games nearest the season.
        w = 2.0 ** (-ch.recency / F.HALF_LIFE_YEARS)
        self.assertGreater(w[0], w[-1])
        latest = F.build_channel(self.store, self.rows, cutoff_days=self.cutoff, target_days=target, normalizer=self.norm)
        w_latest = 2.0 ** (-latest.recency / F.HALF_LIFE_YEARS)
        self.assertLess(w_latest[0], w_latest[-1])
        np.testing.assert_array_equal(ch.x, latest.x)                         # the games themselves are encoded identically

    def test_cap_keeps_the_nearest_games_and_the_time_basis_follows(self):
        games = _many_games(F.MAX_GAMES + 40)
        store = F.GameStore.from_frame(games, dobs=pd.Series({7: pd.Timestamp("2004-03-01")}))
        rows = store.rows(7)
        cutoff = F._days(pd.Series([pd.Timestamp("2026-09-30")]))[0]
        norm = F.Normalizer.fit(store, rows, cutoff, np.zeros((1, len(F.STATIC_CONTINUOUS))))
        target = F._days(pd.Series([pd.Timestamp("2021-06-01")]))[0]
        near = F.build_channel(store, rows, cutoff_days=cutoff, target_days=target, normalizer=norm, reader="near")
        late = F.build_channel(store, rows, cutoff_days=cutoff, target_days=target, normalizer=norm, reader="latest")
        self.assertEqual(len(near.weight), F.MAX_GAMES)
        self.assertEqual(float(near.evidence[9]), 1.0)                        # truncation recorded
        self.assertEqual(near.anchor_days, float(store.date[rows[0]]))        # the earliest games are the ones kept
        self.assertEqual(late.anchor_days, float(store.date[rows[-1]]))
        self.assertEqual(int(near.season[-1]), int(store.season[rows[F.MAX_GAMES - 1]]))
        # build_example dates the time basis at the channel's anchor, not at a dropped game.
        unit = pd.Series({"unit_id": "u", "period_start": pd.Timestamp("2021-06-01"), "dob": pd.Timestamp("2004-03-01"), "age_at_cutoff": 17.2,
                          "minutes": 100.0, "games_played": 10, "starts": 1, "overtime_periods": 0.0, "overtime_known": True, "scheduled_games": 30,
                          **{c: 0 for c in F.TARGET_COUNTS}})
        static = np.zeros(len(F.STATIC_CONTINUOUS), dtype=np.float32)
        cats = np.zeros(len(F.STATIC_CATEGORICAL), dtype=np.int32)
        ex_near = F.build_example(unit, store=store, normalizer=norm, static=static, static_mask=static > -1, static_cats=cats,
                                  intl_rows=rows, ncaa_rows=rows[:0], cutoff_days=cutoff, direction="+", reader="near")
        ex_late = F.build_example(unit, store=store, normalizer=norm, static=static, static_mask=static > -1, static_cats=cats,
                                  intl_rows=rows, ncaa_rows=rows[:0], cutoff_days=cutoff, direction="+")
        gap_near = (ex_near.intl.anchor_days - target) / F.YEAR_DAYS
        gap_late = (ex_late.intl.anchor_days - target) / F.YEAR_DAYS
        self.assertLess(gap_near, gap_late)
        self.assertGreater(gap_near, 0)
        self.assertFalse(np.array_equal(ex_near.time, ex_late.time))          # the signed gap and source age moved with the anchor

    def test_dataset_and_run_arm_validate_the_reader(self):
        from fpp.experiments.train import run_arm
        from fpp.model.dataset import FoldDataset
        self.assertEqual(FoldDataset.__dataclass_fields__["backward_reader"].default, "latest")
        with pytest.raises(ValueError, match="backward_reader"):
            run_arm(tables="no_tables", fold_season=2023, arm="AR", seed=1, out="no_out", backward_reader="middle")


if __name__ == "__main__":
    unittest.main()
