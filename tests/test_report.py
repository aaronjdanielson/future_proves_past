"""Report helpers, likelihood decomposition, and the D-028 operational rules."""
import unittest

import numpy as np
import pandas as pd
import torch

from fpp.data.folds import UNRESOLVED_GUARD, player_index, unresolved_period_games
from fpp.experiments.gradcheck import structural_cases
from fpp.experiments.report import interval_coverage, reliability, row_parameters
from fpp.model.distributions import SeasonParameters
from fpp.model.likelihood import SeasonModel
from fpp.model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch, expand_parameters


class DecompositionTests(unittest.TestCase):
    def test_components_sum_to_total_and_reference_parameters_round_trip(self):
        params = SeasonParameters()
        model = InterceptOnlySeasonModel(params)
        cases, draws = structural_cases(params, np.random.default_rng(2), 32)
        batch = SeasonBatch.from_outcomes(list(cases.values()) + draws, 32)
        lik = DifferentiableSeasonLikelihood(model)
        c = lik.log_prob_components(batch)
        finite = torch.isfinite(c["total"])
        recombined = c["participation"] + c["games_given_participation"] + c["starts"] + c["rest"]
        self.assertLess(float((recombined[finite] - c["total"][finite]).abs().max()), 1e-9)
        zero = batch.games == 0
        self.assertTrue(torch.allclose(c["participation"][zero], torch.log1p(-model.constrained()["pi_g"]).expand_as(c["participation"][zero])))
        self.assertTrue(bool((c["games_given_participation"][zero] == 0).all()))
        # Per-row parameters rebuilt as SeasonParameters give the same SciPy score as the shared model.
        per_row = {k: v.detach().numpy() for k, v in expand_parameters(model.constrained(), len(batch)).items()}
        rebuilt = row_parameters(per_row, 0)
        outcome = list(cases.values())[1]
        self.assertAlmostEqual(SeasonModel(rebuilt).log_prob(outcome, 32), SeasonModel(params).log_prob(outcome, 32), places=10)


class HelperTests(unittest.TestCase):
    def test_reliability_and_coverage(self):
        prob = np.linspace(0.05, 0.95, 100)
        outcome = (np.random.default_rng(0).random(100) < prob).astype(float)
        r = reliability(prob, outcome, bins=5)
        self.assertEqual(sum(b["n"] for b in r["bins"]), 100)
        self.assertTrue(0 <= r["brier"] <= 1)
        self.assertAlmostEqual(interval_coverage(np.array([0, 0]), np.array([10, 10]), np.array([5, 11])), 0.5)


class UnresolvedPeriodTests(unittest.TestCase):
    def test_unresolved_guard_games_are_dropped_from_the_player_index(self):
        games = pd.DataFrame({"record_id": ["a", "b", "c"], "source": ["intl", "intl", "ncaa"], "player_id": [1, 1, 1],
                              "date": pd.to_datetime(["2021-08-01", "2022-04-01", "2022-12-01"]),
                              "season_guard": [UNRESOLVED_GUARD, "", ""]})
        index = player_index(games)
        self.assertEqual(list(index[1]["record_id"]), ["b"])
        self.assertEqual(unresolved_period_games(games), 1)


if __name__ == "__main__":
    unittest.main()
