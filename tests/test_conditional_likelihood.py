"""Per-row parameters and observed overtime in the differentiable likelihood."""
import unittest

import numpy as np
import torch

from fpp.experiments.gradcheck import structural_cases
from fpp.model.distributions import SeasonParameters
from fpp.model.torch_likelihood import (DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch,
                                        expand_parameters)

SCHEDULE = 32


def _batch(params, seed=20260901):
    rng = np.random.default_rng(seed)
    cases, draws = structural_cases(params, rng, SCHEDULE)
    outcomes = list(cases.values()) + draws
    return SeasonBatch.from_outcomes(outcomes, SCHEDULE)


class ConditionalLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.params = SeasonParameters()
        self.model = InterceptOnlySeasonModel(self.params)
        self.lik = DifferentiableSeasonLikelihood(self.model)
        self.batch = _batch(self.params)

    def test_per_row_parameters_equal_expanded_intercepts(self):
        shared, _ = self.lik.log_prob(self.batch, overtime_terms=12)
        c = expand_parameters(self.model.constrained(), len(self.batch))
        per_row = {k: v.clone() for k, v in c.items()}                    # genuinely [B]-shaped tensors
        free = DifferentiableSeasonLikelihood(None, quadrature_nodes=32)  # no model at all
        own, _ = free.log_prob(self.batch, overtime_terms=12, params=per_row)
        finite = torch.isfinite(shared)
        self.assertTrue(torch.equal(finite, torch.isfinite(own)))
        self.assertLess(float((shared[finite] - own[finite]).abs().max()), 1e-12)

    def test_per_row_parameters_vary_by_row(self):
        c = expand_parameters(self.model.constrained(), len(self.batch))
        varied = {k: v.clone() for k, v in c.items()}
        varied["nu"] = varied["nu"] * torch.linspace(0.5, 2.0, len(self.batch), dtype=torch.float64)
        base, _ = self.lik.log_prob(self.batch, overtime_terms=12)
        out, _ = self.lik.log_prob(self.batch, overtime_terms=12, params=varied)
        positive = (self.batch.games > 0) & torch.isfinite(base)
        self.assertTrue(bool((out[positive] != base[positive]).any()))
        # Row-wise independence: changing row i's parameter must not move row j.
        single = {k: v.clone() for k, v in c.items()}
        i = int(positive.nonzero()[0])
        single["alpha_m"][i] = single["alpha_m"][i] * 1.5
        out2, _ = self.lik.log_prob(self.batch, overtime_terms=12, params=single)
        others = torch.arange(len(self.batch)) != i
        self.assertLess(float((out2[others & torch.isfinite(base)] - base[others & torch.isfinite(base)]).abs().max()), 1e-12)
        self.assertNotAlmostEqual(float(out2[i]), float(base[i]))

    def test_observed_overtime_terms_sum_to_the_marginal(self):
        """logsumexp over o of the joint with O = o observed equals the summed-out log probability."""
        marginal, diag = self.lik.log_prob(self.batch)
        positive = (self.batch.games > 0) & torch.isfinite(marginal)
        n_terms = int(diag.overtime_terms_per_row.max())
        pieces = []
        for o in range(n_terms + 2):
            b = SeasonBatch(self.batch.games, self.batch.starts, self.batch.recorded_minutes, self.batch.counts, self.batch.schedule,
                            overtime=torch.full((len(self.batch),), o, dtype=torch.long),
                            overtime_known=torch.ones(len(self.batch), dtype=torch.bool))
            lp, _ = self.lik.log_prob(b)
            pieces.append(lp)
        stacked = torch.stack(pieces)                                   # [O, B]
        recombined = torch.logsumexp(stacked, dim=0)
        err = (recombined[positive] - marginal[positive]).abs().max()
        self.assertLess(float(err), 5e-5)                               # the registered overtime log tolerance
        # An observed overtime that cannot hold the recorded minutes is impossible.
        heavy = self.batch.recorded_minutes > self.batch.games.to(torch.float64) * 40.0
        if bool(heavy.any()):
            b0 = SeasonBatch(self.batch.games, self.batch.starts, self.batch.recorded_minutes, self.batch.counts, self.batch.schedule,
                             overtime=torch.zeros(len(self.batch), dtype=torch.long), overtime_known=torch.ones(len(self.batch), dtype=torch.bool))
            lp0, _ = self.lik.log_prob(b0)
            self.assertTrue(torch.isinf(lp0[heavy]).all())

    def test_mixed_known_and_unknown_rows(self):
        B = len(self.batch)
        known = torch.zeros(B, dtype=torch.bool)
        known[::2] = True
        marginal, _ = self.lik.log_prob(self.batch)
        b = SeasonBatch(self.batch.games, self.batch.starts, self.batch.recorded_minutes, self.batch.counts, self.batch.schedule,
                        overtime=torch.zeros(B, dtype=torch.long), overtime_known=known)
        mixed, _ = self.lik.log_prob(b)
        unknown = ~known & torch.isfinite(marginal)
        self.assertLess(float((mixed[unknown] - marginal[unknown]).abs().max()), 1e-12)
        b_all = SeasonBatch(self.batch.games, self.batch.starts, self.batch.recorded_minutes, self.batch.counts, self.batch.schedule,
                            overtime=torch.zeros(B, dtype=torch.long), overtime_known=torch.ones(B, dtype=torch.bool))
        all_known, _ = self.lik.log_prob(b_all)
        k = known & torch.isfinite(all_known)
        self.assertLess(float((mixed[k] - all_known[k]).abs().max()), 1e-12)

    def test_gradients_flow_to_per_row_parameters(self):
        c = expand_parameters(self.model.constrained(), len(self.batch))
        leaves = {k: v.detach().clone().requires_grad_(True) for k, v in c.items()}
        lp, _ = self.lik.log_prob(self.batch, overtime_terms=8, params=leaves)
        finite = torch.isfinite(lp)
        lp[finite].sum().backward()
        for name, leaf in leaves.items():
            self.assertTrue(torch.isfinite(leaf.grad).all(), name)
        self.assertTrue(bool((leaves["nu"].grad[self.batch.games > 0] != 0).any()))


if __name__ == "__main__":
    unittest.main()
