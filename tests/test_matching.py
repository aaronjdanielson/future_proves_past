"""D-033: pool objective without attendance, unit-level shuffle, matched minibatches, exact conditional exposure."""
import unittest

import numpy as np
import pandas as pd
import torch

from fpp.experiments.gradcheck import structural_cases
from fpp.experiments.train import _batches, shuffle_reconstruction_targets
from fpp.model.distributions import SeasonParameters
from fpp.model.observation import OnceRoundedMinutesKernel
from fpp.model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch, expand_parameters


class PoolObjectiveTests(unittest.TestCase):
    def test_participation_removed_exactly(self):
        # With S = G = 1 the games factor equals log pi_g, so lp - log pi_g does not depend on pi_g.
        params = SeasonParameters()
        lik = DifferentiableSeasonLikelihood(None, kernel=OnceRoundedMinutesKernel(step=1 / 60), quadrature_nodes=16)
        counts = torch.tensor([[6, 3, 2, 1, 2, 2, 1, 3, 2, 0, 0, 1, 2]] * 3, dtype=torch.long)
        sb = SeasonBatch(torch.ones(3, dtype=torch.long), torch.tensor([1, 0, 1]), torch.tensor([22.5, 31.0 + 1 / 60, 40.0], dtype=torch.float64),
                         counts, torch.ones(3, dtype=torch.long))
        base = expand_parameters(InterceptOnlySeasonModel(params).constrained(), 3)
        out = []
        for pi in (0.2, 0.9):
            c = {k: v.clone() for k, v in base.items()}
            c["pi_g"] = torch.full((3,), pi, dtype=torch.float64)
            lp, _ = lik.log_prob(sb, params=c)
            out.append(lp - torch.log(c["pi_g"]))
        self.assertTrue(torch.allclose(out[0], out[1], atol=1e-10))
        self.assertTrue(torch.isfinite(out[0]).all())


class InvalidRowGradientTests(unittest.TestCase):
    """D-037: an impossible observation in a batch must not poison the gradients of the valid rows."""

    def test_mixed_batch_has_finite_gradients(self):
        params = SeasonParameters()
        lik = DifferentiableSeasonLikelihood(None, kernel=OnceRoundedMinutesKernel(step=1 / 60), quadrature_nodes=16)
        counts = torch.tensor([[6, 3, 2, 1, 2, 2, 1, 3, 2, 0, 0, 1, 2],
                               [2, 5, 2, 1, 2, 2, 1, 3, 2, 0, 0, 1, 2],        # makes > attempts: impossible
                               [6, 3, 2, 1, 2, 2, 1, 3, 2, 0, 0, 1, 2]], dtype=torch.long)
        sb = SeasonBatch(torch.ones(3, dtype=torch.long), torch.tensor([1, 0, 1]), torch.tensor([22.5, 31.0, 40.0], dtype=torch.float64),
                         counts, torch.ones(3, dtype=torch.long))
        c = {k: v.detach().clone().requires_grad_(True) for k, v in expand_parameters(InterceptOnlySeasonModel(params).constrained(), 3).items()}
        lp, _ = lik.log_prob(sb, params=c)
        self.assertTrue(torch.isinf(lp[1]) and torch.isfinite(lp[0]) and torch.isfinite(lp[2]))
        lp[torch.isfinite(lp)].sum().backward()
        for name, leaf in c.items():
            self.assertTrue(torch.isfinite(leaf.grad).all(), name)


class UnitLevelShuffleTests(unittest.TestCase):
    def test_nested_windows_share_one_donor_bundle(self):
        class E:
            def __init__(self, uid, k, t):
                self.unit_id, self.direction, self.k, self.targets = uid, "+", k, t
        units = pd.DataFrame({"unit_id": ["a", "b", "c"], "season": [2020, 2020, 2020], "first_year": [True, True, True]})
        bundles = {u: {"games": i, "schedule": 30 + i, "overtime": float(i)} for i, u in enumerate(["a", "b", "c"])}
        examples = [E("a", (1,), bundles["a"]), E("a", (2, 3), bundles["a"]), E("b", (1,), bundles["b"]), E("c", (1,), bundles["c"]), E("c", (2,), bundles["c"])]
        info = shuffle_reconstruction_targets(examples, units, np.random.default_rng(1))
        self.assertEqual(info["units_shuffled"], 3)
        self.assertIs(examples[0].targets, examples[1].targets)              # both windows of unit a: same bundle
        self.assertIs(examples[3].targets, examples[4].targets)
        received = {examples[0].targets["games"], examples[2].targets["games"], examples[3].targets["games"]}
        self.assertEqual(received, {0, 1, 2})                                # a permutation of complete bundles


class ReconstructionHoldoutTests(unittest.TestCase):
    def test_holdout_removes_players_entirely_and_keeps_their_windows_as_test(self):
        from fpp.experiments.train import split_reconstruction_holdout

        class E:
            def __init__(self, uid, d, k=()):
                self.unit_id, self.direction, self.k = uid, d, k
        train = [E("1:2020:9", "-"), E("1:2020:9", "+", (1,)), E("1:2021:9", "-"), E("2:2020:9", "-"), E("2:2020:9", "+", (1,)),
                 E("3:2020:9", "-"), E("4:2020:9", "-"), E("4:2020:9", "+", (1,)), E("4:2020:9", "+", (2,))]
        kept, test, held = split_reconstruction_holdout(train, 1 / 3, np.random.default_rng(3))
        self.assertEqual(len(held), 1)
        p = next(iter(held))
        self.assertTrue(all(int(e.unit_id.split(":")[0]) != p for e in kept))              # no example of the held player trains
        self.assertTrue(all(e.direction == "+" and int(e.unit_id.split(":")[0]) == p for e in test))
        self.assertEqual(len(kept) + len(test) + sum(1 for e in train if e.direction == "-" and int(e.unit_id.split(":")[0]) == p), len(train))
        self.assertIn(E("3:2020:9", "-").unit_id, [e.unit_id for e in kept])                 # forward-only players untouched
        k0, t0, h0 = split_reconstruction_holdout(train, 0.0, np.random.default_rng(3))
        self.assertEqual((len(k0), len(t0), len(h0)), (len(train), 0, 0))


class MatchedMinibatchTests(unittest.TestCase):
    def test_forward_batch_order_depends_only_on_the_seed(self):
        items = list(range(1000))
        a = [tuple(g) for g in _batches(items, 64, np.random.default_rng(20260901))]
        b = [tuple(g) for g in _batches(items, 64, np.random.default_rng(20260901))]
        c = [tuple(g) for g in _batches(items, 64, np.random.default_rng(20260902))]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class LatentExposureTests(unittest.TestCase):
    def test_moment_matches_direct_integration(self):
        from scipy.integrate import quad
        from scipy.stats import beta as beta_dist
        params = SeasonParameters(overtime_rate_per_game=1e-4, minutes_max_probability=1e-4)   # one overtime term, no atom
        model = InterceptOnlySeasonModel(params)
        lik = DifferentiableSeasonLikelihood(model, quadrature_nodes=64)
        cases, draws = structural_cases(params, np.random.default_rng(4), 32)
        batch = SeasonBatch.from_outcomes(draws, 32)
        expected = lik.latent_minutes_given_opportunity(batch).detach().numpy()
        a, b = params.minutes_alpha, params.minutes_beta
        for i in range(min(4, len(draws))):
            g, m = int(batch.games[i]), float(batch.recorded_minutes[i])
            if g == 0 or m <= 0 or m >= 40 * g:
                continue
            maximum = 40.0 * g
            lo, hi = (m - 0.5) / maximum, (m + 0.5) / maximum
            mass = quad(lambda u: beta_dist.pdf(u, a, b), lo, hi)[0]
            first = quad(lambda u: u * beta_dist.pdf(u, a, b), lo, hi)[0]
            self.assertAlmostEqual(expected[i], maximum * first / mass, delta=1e-3)
            self.assertTrue(m - 0.5 <= expected[i] <= m + 0.5)


if __name__ == "__main__":
    unittest.main()
