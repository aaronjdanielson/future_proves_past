"""Milestone 0.4: the pool objective, the shuffled-correspondence control, and the component decomposition."""
import unittest

import numpy as np
import pandas as pd
import torch

from fpp.experiments.gradcheck import structural_cases
from fpp.experiments.train import shuffle_reconstruction_targets
from fpp.model import features as F
from fpp.model.distributions import SeasonParameters
from fpp.model.pool import CONTEXT_FIELDS, PoolSampler, collate_pool
from fpp.model.tower import ReferenceTower
from fpp.model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch, expand_parameters


def _pool_games(n_players=3, n_games=8):
    rows = []
    for p in range(1, n_players + 1):
        for i in range(n_games):
            d = pd.Timestamp("2021-10-01") + pd.Timedelta(days=14 * i)
            rows.append({"record_id": f"intl:{p * 100 + i}:{p}", "source": "intl", "player_id": p, "game_id": p * 100 + i, "date": d,
                         "release_date": d + pd.Timedelta(days=1), "season": 2022, "competition": "L", "competition_kind": "league",
                         "country": "X", "age_group": "senior", "division": "", "minutes": 20.5 + i, "a2": 5, "k2": 2, "a3": 2, "k3": 1,
                         "af": 2, "kf": 2, "orb": 1, "drb": 2, "ast": 1, "stl": 0, "blk": 0, "tov": 1, "pf": 2, "pts": 9, "home": i % 2,
                         "starter": True if i % 3 else np.nan, "team_pts": 80, "opp_pts": 75, "regulation_minutes": 40,
                         "overtime_periods": 0.0, "coverage_certified": True,
                         "opp_adj_o": 110.0, "opp_adj_d": 105.0, "opp_adj_pace": 70.0, "opp_strength_available": pd.Timestamp("2022-06-30"),
                         "opp_prior_adj_o": 108.0, "opp_prior_adj_d": 106.0, "opp_prior_adj_pace": 69.0, "opp_prior_available": pd.Timestamp("2021-06-30"),
                         "team_adj_o": 112.0, "team_adj_d": 104.0, "team_adj_pace": 71.0, "team_strength_available": pd.Timestamp("2022-06-30"),
                         "opp_hist_winpct": 0.6, "opp_hist_placement": 3.0, "opp_hist_available": pd.Timestamp("2022-06-30"),
                         "team_hist_winpct": 0.5, "team_hist_placement": 5.0, "team_hist_available": pd.Timestamp("2022-06-30"),
                         "opp_record_winpct": np.nan, "opp_record_margin": np.nan})
    return pd.DataFrame(rows)


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.games = _pool_games()
        self.store = F.GameStore.from_frame(self.games, dobs=pd.Series({1: pd.Timestamp("2002-05-01")}))
        rows = np.arange(len(self.games))
        cutoff = F._days(pd.Series([pd.Timestamp("2022-09-30")]))[0]
        self.normalizer = F.Normalizer.fit(self.store, rows, cutoff, np.zeros((2, len(F.STATIC_CONTINUOUS))))
        self.sampler = PoolSampler(self.store, self.normalizer, cutoff_days=cutoff, heldout_players={3}, seed=1)

    def test_impossible_minutes_are_not_targets_and_score_minus_inf(self):
        games = self.games.copy()
        games.loc[games["record_id"] == "intl:107:1", "minutes"] = 96.0          # a season total filed as a game
        store = F.GameStore.from_frame(games)
        rows = np.arange(len(games))
        cutoff = F._days(pd.Series([pd.Timestamp("2022-09-30")]))[0]
        norm = F.Normalizer.fit(store, rows, cutoff, np.zeros((2, len(F.STATIC_CONTINUOUS))))
        sampler = PoolSampler(store, norm, cutoff_days=cutoff, heldout_players=set(), seed=1)
        self.assertNotIn(int(np.flatnonzero(store.record_id == "intl:107:1")[0]), set(sampler.targets.tolist()))
        lik = DifferentiableSeasonLikelihood(None, quadrature_nodes=8)
        beyond = 40.0 + 5.0 * lik.accuracy.max_overtime_terms + 1.0           # more minutes than any budgeted overtime allows
        sb = SeasonBatch(torch.ones(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), torch.tensor([beyond], dtype=torch.float64),
                         torch.zeros(1, 13, dtype=torch.long), torch.ones(1, dtype=torch.long))
        c = expand_parameters(InterceptOnlySeasonModel(SeasonParameters()).constrained(), 1)
        lp, _ = lik.log_prob(sb, params=c)                                   # no budget error: impossible → -inf
        self.assertTrue(torch.isinf(lp).all())
        bad = {k: v.clone() for k, v in c.items()}
        bad["alpha_m"][0] = float("nan")
        with self.assertRaises(ValueError):
            lik.log_prob(SeasonBatch(torch.ones(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), torch.tensor([20.0], dtype=torch.float64),
                                     torch.zeros(1, 13, dtype=torch.long), torch.ones(1, dtype=torch.long)), params=bad)

    def test_targets_exclude_heldout_and_first_games(self):
        players = set(self.store.player_id[self.sampler.targets].tolist())
        self.assertEqual(players, {1, 2})                                   # player 3 held out
        self.assertEqual(len(self.sampler.targets), 2 * 7)                  # first game of each player has no prefix

    def test_prefix_is_strictly_earlier_and_clocks_anchor_at_the_target(self):
        row = int(self.sampler.targets[-1])
        ex = self.sampler.example(row)
        target = self.store.date[row]
        self.assertTrue(np.all(self.store.date[self.store.rows(ex.player_id)][: len(ex.intl.weight)] < target))
        self.assertTrue(np.all(ex.intl.clocks[:, 0] < 0))                    # tau relative to the prediction date
        self.assertAlmostEqual(float(ex.intl.recency[-1]), 0.0)
        self.assertEqual(ex.targets["games"], 1)
        self.assertTrue(np.all(ex.targets["counts"] == np.array([5, 2, 2, 1, 2, 2, 1, 2, 1, 0, 0, 1, 2])))
        self.assertAlmostEqual(ex.targets["minutes"] * 60, round(ex.targets["minutes"] * 60), places=6)
        # The target's minutes are the source value (20.5 + game index), decoded from the log1p-stored field.
        source_minutes = float(self.games.loc[self.games["record_id"] == self.store.record_id[row], "minutes"].iloc[0])
        self.assertAlmostEqual(ex.targets["minutes"], source_minutes, places=6)
        # NCAA- and destination-derived static inputs are hidden; only age survives when known.
        self.assertTrue(ex.static_mask[0] if ex.player_id == 1 else not ex.static_mask[0])
        self.assertFalse(ex.static_mask[1:].any())

    def test_collate_and_pretrain_forward(self):
        torch.manual_seed(0)
        examples = self.sampler.sample(6)
        batch = collate_pool(examples)
        tower = ReferenceTower([5, 4, 6, 4, 3, 4], [4, 6, 4, 5], SeasonParameters(), n_context=len(CONTEXT_FIELDS))
        params = tower.pretrain_forward(batch)
        self.assertEqual(tuple(params["rates"].shape), (6, 10))
        lik = DifferentiableSeasonLikelihood(None, quadrature_nodes=16)
        from fpp.model.observation import OnceRoundedMinutesKernel
        lik = DifferentiableSeasonLikelihood(None, kernel=OnceRoundedMinutesKernel(step=1 / 60), quadrature_nodes=16)
        sb = SeasonBatch(batch["games"], batch["starts"], batch["minutes"], batch["counts"], batch["schedule"],
                         batch["overtime"].long(), batch["overtime_known"])
        lp, _ = lik.log_prob(sb, params=params)
        self.assertTrue(torch.isfinite(lp).all())
        (-lp).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in tower.parameters() if p.grad is not None))
        # Forward heads receive no gradient from the pool objective.
        self.assertTrue(all(p.grad is None for p in tower.heads.parameters()))


class ShuffleTests(unittest.TestCase):
    def test_targets_permute_within_strata_only(self):
        class E:
            def __init__(self, uid, direction, t):
                self.unit_id, self.direction, self.targets = uid, direction, t
        units = pd.DataFrame({"unit_id": ["a", "b", "c", "d", "e"], "season": [2020, 2020, 2020, 2021, 2021],
                              "first_year": [True, True, False, True, True]})
        examples = [E("a", "+", 1), E("b", "+", 2), E("c", "+", 3), E("d", "+", 4), E("e", "+", 5), E("a", "-", 9)]
        info = shuffle_reconstruction_targets(examples, units, np.random.default_rng(0))
        self.assertEqual(info["strata"], 3)
        self.assertEqual({examples[0].targets, examples[1].targets}, {1, 2})   # stratum (2020, True)
        self.assertEqual(examples[2].targets, 3)                               # singleton stratum untouched
        self.assertEqual({examples[3].targets, examples[4].targets}, {4, 5})
        self.assertEqual(examples[5].targets, 9)                               # forward example untouched


class OpportunityDecompositionTests(unittest.TestCase):
    def test_total_equals_opportunity_plus_production(self):
        params = SeasonParameters()
        model = InterceptOnlySeasonModel(params)
        cases, draws = structural_cases(params, np.random.default_rng(3), 32)
        batch = SeasonBatch.from_outcomes(list(cases.values()) + draws, 32)
        lik = DifferentiableSeasonLikelihood(model)
        total, _ = lik.log_prob(batch)
        opp = lik.log_prob_opportunity(batch)
        finite = torch.isfinite(total)
        self.assertTrue(bool(((total - opp)[finite] <= 1e-9).all()))           # production|opportunity is a log probability
        zero = batch.games == 0
        self.assertTrue(torch.allclose(opp[zero], total[zero]))
        c = lik.log_prob_components(batch)
        self.assertTrue(bool(((opp - c["games"] - c["starts"])[finite & ~zero] <= 1e-9).all()))


if __name__ == "__main__":
    unittest.main()
