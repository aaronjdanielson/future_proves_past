"""Differentiable likelihood against the SciPy reference (milestone 0.2, track 3)."""
import math
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from fpp.model.distributions import SeasonParameters
from fpp.model.likelihood import SeasonModel
from fpp.model.observation import OnceRoundedMinutesKernel
from fpp.model.production import ProductionParameters, RATE_NAMES

if torch is not None:
    from fpp.experiments.gradcheck import structural_cases
    from fpp.model.torch_likelihood import (DifferentiableSeasonLikelihood, InterceptOnlySeasonModel,
                                            SeasonBatch, finite_difference_check, value_agreement)

SCHEDULE = 32


@unittest.skipIf(torch is None, "torch not installed")
class TorchLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.params = SeasonParameters()
        self.rng = np.random.default_rng(20260901)
        self.model = InterceptOnlySeasonModel(self.params)

    def test_reference_parameters_round_trip(self):
        ref = self.model.reference()
        self.assertAlmostEqual(ref.participation_probability, self.params.participation_probability, places=12)
        self.assertAlmostEqual(ref.minutes_beta, self.params.minutes_beta, places=12)
        self.assertAlmostEqual(ref.production.rates["A2"], self.params.production.rates["A2"], places=12)

    def test_values_agree_on_structural_cases_and_draws(self):
        cases, draws = structural_cases(self.params, self.rng, SCHEDULE)
        outcomes = list(cases.values()) + draws
        lik = DifferentiableSeasonLikelihood(self.model, quadrature_nodes=32)
        result = value_agreement(lik, outcomes, SCHEDULE)
        self.assertTrue(result["same_support"])
        self.assertLess(result["max_abs_diff_nats"], 2e-4, result)
        # every structural case has finite mass in both implementations
        for i, name in enumerate(cases):
            self.assertTrue(math.isfinite(result["reference"][i]), name)
            self.assertTrue(math.isfinite(result["ours"][i]), name)

    def test_refinement_converges_toward_reference(self):
        cases, draws = structural_cases(self.params, self.rng, SCHEDULE)
        outcomes = list(cases.values()) + draws
        diffs = [value_agreement(DifferentiableSeasonLikelihood(self.model, quadrature_nodes=k),
                                 outcomes, SCHEDULE)["max_abs_diff_nats"] for k in (8, 32, 64)]
        self.assertLess(diffs[-1], 2e-4)
        self.assertLessEqual(diffs[-1], diffs[0] + 1e-9)

    def test_endpoint_heavy_minutes_shapes_remain_finite_and_close(self):
        for alpha, beta in ((0.4, 0.6), (30.0, 0.7), (0.6, 30.0)):
            params = SeasonParameters(minutes_alpha=alpha, minutes_beta=beta, minutes_max_probability=0.2)
            model = InterceptOnlySeasonModel(params)
            ref = SeasonModel(params, OnceRoundedMinutesKernel())
            outcomes = [ref.sample(SCHEDULE, self.rng) for _ in range(20)]
            lik = DifferentiableSeasonLikelihood(model, quadrature_nodes=64)
            result = value_agreement(lik, outcomes, SCHEDULE)
            self.assertTrue(result["same_support"], (alpha, beta))
            self.assertLess(result["max_abs_diff_nats"], 1e-3, (alpha, beta, result["max_abs_diff_nats"]))
            batch = SeasonBatch.from_outcomes(outcomes, SCHEDULE)
            lp, _ = lik.log_prob(batch)
            self.assertFalse(bool(torch.isnan(lp).any()), (alpha, beta))
            (-lp[torch.isfinite(lp)]).sum().backward()
            for name, p in model.named_parameters():
                self.assertTrue(bool(torch.isfinite(p.grad).all()), (alpha, beta, name))

    def test_autodiff_matches_finite_differences(self):
        cases, draws = structural_cases(self.params, self.rng, SCHEDULE)
        outcomes = [o for o in list(cases.values()) + draws if o.games > 0][:8]
        lik = DifferentiableSeasonLikelihood(self.model, quadrature_nodes=32)
        result = finite_difference_check(lik, SeasonBatch.from_outcomes(outcomes, SCHEDULE),
                                         overtime_terms=30, eps=1e-6)
        self.assertLess(result["max_rel_err"], 1e-5, result["per_parameter"])

    def test_rare_counts_gradient_is_finite(self):
        cases, _ = structural_cases(self.params, self.rng, SCHEDULE)
        batch = SeasonBatch.from_outcomes([cases["rare_counts"]], SCHEDULE)
        lik = DifferentiableSeasonLikelihood(self.model, quadrature_nodes=32)
        lp, diag = lik.log_prob(batch)
        self.assertTrue(bool(torch.isfinite(lp).all()))
        (-lp).sum().backward()
        grads = torch.cat([p.grad.flatten() for p in self.model.parameters()])
        self.assertTrue(bool(torch.isfinite(grads).all()))
        self.assertLessEqual(diag.max_overtime_log_error_bound, lik.accuracy.overtime_log_tolerance)

    def test_invalid_observations_score_minus_infinity_without_error(self):
        cases, _ = structural_cases(self.params, self.rng, SCHEDULE)
        good = cases["atom_full_minutes"]
        bad_grid = type(good)(good.games, good.starts, good.recorded_minutes - 0.3, good.counts)
        bad_starts = type(good)(good.games, good.games + 1, good.recorded_minutes, good.counts)
        lik = DifferentiableSeasonLikelihood(self.model)
        lp, diag = lik.log_prob(SeasonBatch.from_outcomes([good, bad_grid, bad_starts], SCHEDULE))
        self.assertTrue(math.isfinite(float(lp[0])))
        self.assertEqual(float(lp[1]), -math.inf)
        self.assertEqual(float(lp[2]), -math.inf)
        self.assertEqual(diag.invalid_observations, 2)

    def test_overtime_budget_exhaustion_is_an_error(self):
        from fpp.model.observation import NumericalAccuracy
        from fpp.model.torch_likelihood import OvertimeBudgetError
        cases, _ = structural_cases(self.params, self.rng, SCHEDULE)
        lik = DifferentiableSeasonLikelihood(self.model, accuracy=NumericalAccuracy(max_overtime_terms=1))
        with self.assertRaises(OvertimeBudgetError):
            lik.log_prob(SeasonBatch.from_outcomes([cases["overtime_endpoint_o1"]], SCHEDULE))

    def test_batch_cost_is_measured(self):
        from fpp.model.torch_likelihood import time_batch
        ref = SeasonModel(self.params, OnceRoundedMinutesKernel())
        outcomes = [ref.sample(SCHEDULE, self.rng) for _ in range(64)]
        lik = DifferentiableSeasonLikelihood(self.model, quadrature_nodes=32)
        timing = time_batch(lik, SeasonBatch.from_outcomes(outcomes, SCHEDULE), repeats=2)
        self.assertGreater(timing["median_seconds"], 0.0)
        self.assertLess(timing["median_seconds"], 60.0)



@unittest.skipIf(torch is None, "torch not installed")
class RecoveryMachineryTests(unittest.TestCase):
    """Fast checks of the fitting/information machinery; the study itself runs via `fpp recovery`."""

    def test_per_row_adaptive_overtime_matches_fixed_terms(self):
        params = SeasonParameters()
        rng = np.random.default_rng(7)
        ref = SeasonModel(params, OnceRoundedMinutesKernel())
        outcomes = [ref.sample(SCHEDULE, rng) for _ in range(30)]
        lik = DifferentiableSeasonLikelihood(InterceptOnlySeasonModel(params))
        batch = SeasonBatch.from_outcomes(outcomes, SCHEDULE)
        with torch.no_grad():
            adaptive, diag = lik.log_prob(batch)
            fixed, _ = lik.log_prob(batch, overtime_terms=60)
        self.assertTrue(bool((adaptive - fixed).abs().max() < 1e-4))
        self.assertLess(diag.mean_overtime_terms, diag.overtime_terms + 1)

    def test_fit_reduces_nll_and_information_is_symmetric(self):
        from fpp.experiments.recovery import fit, observed_information, standard_errors, _perturbed
        params = SeasonParameters()
        rng = np.random.default_rng(11)
        ref = SeasonModel(params, OnceRoundedMinutesKernel())
        outcomes = [ref.sample(SCHEDULE, rng) for _ in range(40)]
        batch = SeasonBatch.from_outcomes(outcomes, SCHEDULE)
        model = InterceptOnlySeasonModel(params)
        _perturbed(model, np.random.default_rng(12), 0.3)
        lik = DifferentiableSeasonLikelihood(model, quadrature_nodes=16)
        result = fit(lik, batch, frozen=("log_nu", "logit_pi_m"), max_iter=15, overtime_terms=12)
        self.assertLess(result["nll_end"], result["nll_start"])
        info = observed_information(lik, batch, overtime_terms=12, eps=1e-3)
        self.assertLess(info["asymmetry"], 1e-3)
        ses = standard_errors(info, [n for n, _ in model.named_parameters() if n not in ("log_nu", "logit_pi_m")])
        self.assertNotIn("log_nu", ses["se"])
        self.assertIn("logit_pi_g", ses["se"])


if __name__ == "__main__":
    unittest.main()
