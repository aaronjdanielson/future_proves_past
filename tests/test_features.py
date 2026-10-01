"""Feature construction on a synthetic game table."""
import unittest

import numpy as np
import pandas as pd

from fpp.model import features as F


def _games():
    rows = []
    dates = ["2022-10-01", "2022-11-15", "2023-01-20", "2023-10-05", "2024-02-01"]
    for i, d in enumerate(dates):
        rows.append({
            "record_id": f"intl:{i}:7", "source": "intl", "player_id": 7, "game_id": i, "date": pd.Timestamp(d),
            "release_date": pd.Timestamp(d) + pd.Timedelta(days=1), "season": 2023 if d < "2023-08" else 2024,
            "competition": "L", "competition_kind": "league", "country": "X", "age_group": "senior", "division": "",
            "minutes": 20.0 + i, "a2": 5, "k2": 2, "a3": 2, "k3": 1, "af": 2, "kf": 2, "orb": 1, "drb": 2, "ast": 1,
            "stl": 0, "blk": 0, "tov": 1, "pf": 2, "pts": 9, "home": 1, "starter": True, "team_pts": 80, "opp_pts": 75,
            "regulation_minutes": 40, "overtime_periods": 0.0, "coverage_certified": True,
            "opp_adj_o": 110.0, "opp_adj_d": 105.0, "opp_adj_pace": 70.0, "opp_strength_available": pd.Timestamp("2023-06-30") if d < "2023-08" else pd.Timestamp("2024-06-30"),
            "opp_prior_adj_o": 108.0, "opp_prior_adj_d": 106.0, "opp_prior_adj_pace": 69.0, "opp_prior_available": pd.Timestamp("2022-06-30") if d < "2023-08" else pd.Timestamp("2023-06-30"),
            "team_adj_o": 112.0, "team_adj_d": 104.0, "team_adj_pace": 71.0, "team_strength_available": pd.Timestamp("2023-06-30"),
            "opp_hist_winpct": 0.6, "opp_hist_placement": 3.0, "opp_hist_available": pd.Timestamp("2023-06-30"),
            "team_hist_winpct": 0.5, "team_hist_placement": 5.0, "team_hist_available": pd.Timestamp("2023-06-30"),
            "opp_record_winpct": np.nan, "opp_record_margin": np.nan,
        })
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.games = _games()
        self.store = F.GameStore.from_frame(self.games, dobs=pd.Series({7: pd.Timestamp("2004-03-01")}))

    def test_cutoff_aware_strength(self):
        rows = self.store.rows(7)
        col = F.FIELD_NAMES.index("opp_adj_o")
        # Cutoff 2023-09-30: season-2023 ratings (available 2023-07-01) are usable; 2024 ratings are not, and
        # their prior-season fallback (available 2023-07-01) is.
        x = self.store.continuous(rows, F._days(pd.Series([pd.Timestamp("2023-09-30")]))[0])
        self.assertEqual(list(x[:3, col]), [110.0] * 3)
        self.assertEqual(list(x[3:, col]), [108.0] * 2)
        # Cutoff 2023-01-01: nothing is available yet for the 2023 rows except the prior-season fallback.
        x = self.store.continuous(rows[:3], F._days(pd.Series([pd.Timestamp("2023-01-01")]))[0])
        self.assertEqual(list(x[:, col]), [108.0] * 3)

    def test_channel_clocks_and_evidence(self):
        rows = self.store.rows(7)
        cutoff = F._days(pd.Series([pd.Timestamp("2024-09-30")]))[0]
        target = F._days(pd.Series([pd.Timestamp("2024-11-05")]))[0]
        norm = F.Normalizer.fit(self.store, rows, cutoff, np.zeros((1, len(F.STATIC_CONTINUOUS))))
        ch = F.build_channel(self.store, rows, cutoff_days=cutoff, target_days=target, normalizer=norm)
        self.assertTrue(ch.present)
        self.assertEqual(ch.x.shape, (5, len(F.FIELD_NAMES)))
        tau = ch.clocks[:, 0]
        self.assertTrue(np.all(tau < 0) and np.all(np.diff(tau) > 0))            # all before the target, increasing
        self.assertAlmostEqual(float(ch.recency[-1]), 0.0)                        # latest game anchors recency
        self.assertEqual(list(ch.clocks[:, 4]), [1.0, 0.0, 0.0, 1.0, 0.0])        # season boundaries
        self.assertEqual(list(ch.clocks[:, 5]), [1.0, 0.0, 0.0, 0.0, 0.0])        # first game
        self.assertEqual(list(ch.weight), [21.0, 22.0, 23.0, 24.0, 25.0])         # minutes + 1
        self.assertEqual(float(ch.evidence[5]), 2.0)                              # two seasons
        self.assertEqual(float(ch.evidence[7]), 1.0)                              # strength known everywhere
        self.assertEqual(float(ch.evidence[8]), 1.0)                              # age known (dob supplied)

    def test_empty_channel_and_collate(self):
        import torch
        empty = F.Channel.empty(len(F.FIELD_NAMES))
        self.assertFalse(empty.present)
        rows = self.store.rows(7)
        cutoff = F._days(pd.Series([pd.Timestamp("2024-09-30")]))[0]
        norm = F.Normalizer.fit(self.store, rows, cutoff, np.zeros((1, len(F.STATIC_CONTINUOUS))))
        full = F.build_channel(self.store, rows, cutoff_days=cutoff, target_days=cutoff + 30, normalizer=norm)
        static = np.zeros(len(F.STATIC_CONTINUOUS), dtype=np.float32)
        t_v, t_m = F.time_basis(-0.3, 20.5, 20.8)
        targets = {"games": 10, "starts": 2, "minutes": 150.0, "counts": np.zeros(13, dtype=np.int64), "overtime": 0.0,
                   "overtime_known": True, "schedule": 30}
        ex1 = F.Example("u1", "-", (), full, empty, static, static > -1, np.zeros(len(F.STATIC_CATEGORICAL), dtype=np.int32), t_v, t_m, targets)
        ex2 = F.Example("u2", "+", (1,), empty, empty, static, static > -1, np.zeros(len(F.STATIC_CATEGORICAL), dtype=np.int32), *F.time_basis(np.nan, np.nan, np.nan), targets)
        batch = F.collate([ex1, ex2])
        self.assertEqual(tuple(batch["intl"]["x"].shape), (2, 5, len(F.FIELD_NAMES)))
        self.assertEqual(batch["intl"]["valid"].sum().item(), 5)
        self.assertEqual(batch["intl"]["present"].tolist(), [True, False])
        self.assertEqual(batch["direction"].tolist(), [-1.0, 1.0])
        self.assertTrue(torch.isfinite(batch["intl"]["x"]).all())
        # Forward gating of the time basis: negative delta fills the first bump block only.
        self.assertTrue(t_m[3:7].all() and not t_m[7:].any())


if __name__ == "__main__":
    unittest.main()
