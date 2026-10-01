"""Normalization, endpoint, rare-event and integration checks on synthetic data."""
from dataclasses import replace
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, betaincc, betaln, logsumexp
from scipy.stats import beta, poisson

from fpp.model.distributions import (SeasonParameters, beta_binomial_log_prob,
                                    hurdle_beta_binomial_log_prob)
from fpp.model.likelihood import SeasonModel, SeasonOutcome
from fpp.model.observation import (KernelProvenance, NumericalAccuracy,
                                  NumericalAccuracyError, OnceRoundedMinutesKernel,
                                  conditional_minutes_log_prob, marginalize_overtime,
                                  poisson_log_tail_upper_bound)
from fpp.model.production import COUNT_NAMES


class ObservationTests(unittest.TestCase):
    def test_kernel_partition_with_partial_endpoint_cell(self):
        kernel = OnceRoundedMinutesKernel(step=1.0)
        for maximum in (0.0, 0.7, 3.2, 40.0):
            support = kernel.support(maximum)
            for latent in np.linspace(0, maximum, 101):
                self.assertEqual(sum(kernel.probability(m, latent, maximum) for m in support), 1)
                self.assertIn(kernel.sample(latent, maximum), support)
        self.assertEqual(kernel.record(0.5, 40), 1)
        self.assertFalse(kernel.provenance.empirical_allowed)
        self.assertTrue(kernel.provenance.synthetic_only)
        with self.assertRaises(ValueError):
            KernelProvenance(aggregation="sum_of_rounded_game_minutes")

    def test_exact_recorded_marginal_normalization(self):
        kernel = OnceRoundedMinutesKernel()
        for alpha, shape_b in ((0.08, 4), (4, 0.07), (40, 80), (1, 1)):
            for maximum in (0.7, 40.0, 43.2):
                for atom in (0, 0.23, 1):
                    masses = [conditional_minutes_log_prob(m, maximum, alpha, shape_b,
                                                           atom, kernel).log_probability
                              for m in kernel.support(maximum)]
                    self.assertAlmostEqual(float(logsumexp(masses)), 0, places=11)

    def test_endpoint_atom_adds_to_continuous_cell_mass(self):
        kernel = OnceRoundedMinutesKernel()
        result = conditional_minutes_log_prob(40, 40, 2.5, 0.4, 0.2, kernel)
        expected = 0.2 + 0.8 * betaincc(2.5, 0.4, 39.5 / 40)
        self.assertAlmostEqual(math.exp(result.log_probability), expected, places=13)
        weighted = conditional_minutes_log_prob(40, 40, 2.5, 0.4, 0.2, kernel,
                                                lambda m: math.log(0.3))
        self.assertAlmostEqual(math.exp(weighted.log_probability), expected * 0.3, places=11)

    def test_zero_recorded_minutes_are_positive_probability_for_positive_games(self):
        kernel = OnceRoundedMinutesKernel()
        result = conditional_minutes_log_prob(0, 40, 0.2, 2, 0.1, kernel)
        self.assertGreater(math.exp(result.log_probability), 0)
        expected = 0.9 * betainc(0.2, 2, 0.5 / 40)
        self.assertAlmostEqual(math.exp(result.log_probability), expected, places=13)

    def test_independent_qags_quadrature_endpoint_and_rare_counts(self):
        # Our integrator uses algebraic-weight quadrature; this independent
        # calculation uses QAGS after a power substitution at a singular
        # endpoint, with log scaling for rare production events.
        kernel = OnceRoundedMinutesKernel()
        cases = ((0.12, 3.0, 0.0, 0), (2.0, 0.13, 40.0, 2),
                 (2.5, 4.0, 2.0, 40), (90.0, 110.0, 18.0, 8))
        for alpha, shape_b, recorded, count in cases:
            maximum, atom = 40.0, 0.13
            production = lambda m: float(poisson.logpmf(count, 0.17 * m))
            result = conditional_minutes_log_prob(recorded, maximum, alpha, shape_b,
                                                  atom, kernel, production)
            lo, hi = kernel.cell(recorded, maximum)
            lo, hi = lo / maximum, hi / maximum
            scaling = production(maximum * (lo + hi) / 2)
            if lo == 0:
                constant = alpha * math.log(hi) - math.log(alpha) - betaln(alpha, shape_b)
                def integrand(t):
                    u = hi * t ** (1 / alpha)
                    return math.exp(production(maximum * u) - scaling + constant +
                                    (shape_b - 1) * math.log1p(-u))
                integral = quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-10, limit=500)[0]
            elif hi == 1:
                constant = shape_b * math.log1p(-lo) - math.log(shape_b) - betaln(alpha, shape_b)
                def integrand(t):
                    u = 1 - (1 - lo) * t ** (1 / shape_b)
                    return math.exp(production(maximum * u) - scaling + constant +
                                    (alpha - 1) * math.log(u))
                integral = quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-10, limit=500)[0]
            else:
                integrand = lambda u: math.exp(production(maximum * u) - scaling) * beta.pdf(u, alpha, shape_b)
                integral = quad(integrand, lo, hi, epsabs=1e-12, epsrel=1e-10, limit=500)[0]
            continuous = math.log1p(-atom) + math.log(integral) + scaling
            endpoint = (math.log(atom) + production(maximum)
                        if kernel.probability(recorded, maximum, maximum) else -math.inf)
            expected = float(np.logaddexp(continuous, endpoint))
            self.assertLess(abs(result.log_probability - expected), 5e-5)
            self.assertTrue(result.used_quadrature)

    def test_minute_only_projected_model_normalizes_including_zero_games(self):
        params = SeasonParameters(overtime_rate_per_game=0.0)
        kernel = OnceRoundedMinutesKernel(step=5)
        terms = []
        schedule = 2
        for games in range(schedule + 1):
            game_log = hurdle_beta_binomial_log_prob(games, schedule,
                                                      params.participation_probability,
                                                      params.games_alpha, params.games_beta)
            if games == 0:
                terms.append(game_log)
                continue
            for starts in range(games + 1):
                start_log = beta_binomial_log_prob(starts, games, params.starts_alpha, params.starts_beta)
                for recorded in kernel.support(40 * games):
                    observation = marginalize_overtime(recorded, games, params.minutes_alpha,
                                                       params.minutes_beta, params.minutes_max_probability,
                                                       0, kernel)
                    terms.append(game_log + start_log + observation.log_probability)
        self.assertAlmostEqual(float(logsumexp(terms)), 0, places=11)

    def test_overtime_bound_and_explicit_resource_failure(self):
        for mean in (0.001, 0.8, 12.0, 100.0):
            for last in (0, 2, 20, 150):
                bound = poisson_log_tail_upper_bound(last, mean)
                actual = poisson.logsf(last, mean)
                self.assertGreaterEqual(bound + 1e-11, actual)
        self.assertTrue(math.isfinite(poisson_log_tail_upper_bound(700, 0.01)))
        kernel = OnceRoundedMinutesKernel()
        observed = marginalize_overtime(20, 1, 2, 2, 0.1, 0.5, kernel)
        self.assertLessEqual(observed.diagnostics.overtime_log_error_bound, 5e-5)
        self.assertLessEqual(observed.diagnostics.poisson_log_tail_bound, math.log(1e-8))
        self.assertFalse(observed.diagnostics.total_error_certified)
        with self.assertRaises(NumericalAccuracyError):
            marginalize_overtime(20, 1, 2, 2, 0.1, 0.5, kernel,
                                 accuracy=NumericalAccuracy(max_overtime_terms=1))

    def test_overtime_is_not_renormalized_and_support_extends_past_regulation(self):
        kernel = OnceRoundedMinutesKernel()
        result = marginalize_overtime(45, 1, 2, 3, 1.0, 0.3, kernel)
        # With all mass at maximum minutes, exactly one OT produces 45.
        self.assertAlmostEqual(result.log_probability, poisson.logpmf(1, 0.3), places=12)
        impossible = marginalize_overtime(20, 1, 2, 3, 1.0, 0.3, kernel)
        self.assertEqual(impossible.log_probability, -math.inf)

    def test_rare_production_requires_more_than_absolute_poisson_tail(self):
        kernel = OnceRoundedMinutesKernel()
        ordinary = marginalize_overtime(20, 1, 2, 2, 0.1, 0.4, kernel)
        rare = marginalize_overtime(20, 1, 2, 2, 0.1, 0.4, kernel,
                                    production_log_prob=lambda _: -150.0)
        self.assertGreater(rare.diagnostics.overtime_terms, ordinary.diagnostics.overtime_terms)
        self.assertLessEqual(rare.diagnostics.overtime_log_error_bound, 5e-5)
        self.assertLess(abs(rare.log_probability - ordinary.log_probability + 150), 5e-5)

    def test_invalid_production_and_unattainable_quadrature_accuracy_fail(self):
        kernel = OnceRoundedMinutesKernel()
        with self.assertRaises(ValueError):
            conditional_minutes_log_prob(40, 40, 2, 2, 1, kernel, lambda _: 0.1)
        with self.assertRaises(NumericalAccuracyError):
            conditional_minutes_log_prob(20, 40, 2, 2, 0, kernel, lambda m: -m,
                                          NumericalAccuracy(quadrature_log_tolerance=1e-20,
                                                            max_quadrature_refinements=1))

    def test_full_sampler_scorer_and_invalid_observations(self):
        model = SeasonModel(SeasonParameters(overtime_rate_per_game=0))
        zero = SeasonOutcome(0, 0, 0, {key: 0 for key in COUNT_NAMES})
        self.assertAlmostEqual(model.log_prob(zero, 10), math.log(0.15))
        positive_zero = SeasonOutcome(1, 0, 0, dict(zero.counts))
        self.assertTrue(math.isfinite(model.log_prob(positive_zero, 10)))
        for outcome in (replace(zero, starts=1), replace(zero, recorded_minutes=1),
                        replace(positive_zero, recorded_minutes=0.4),
                        replace(zero, counts={"A2": 1}),
                        replace(positive_zero, counts={**zero.counts, "PF": 6})):
            self.assertEqual(model.log_prob(outcome, 10), -math.inf)
        rng = np.random.default_rng(319)
        for _ in range(4):
            outcome = model.sample(3, rng)
            result = model.score(outcome, 3)
            self.assertTrue(math.isfinite(result.log_prob))
            self.assertEqual(outcome.points, 2 * outcome.counts["K2"] +
                             3 * outcome.counts["K3"] + outcome.counts["Kf"])
        with self.assertRaises(ValueError):
            model.log_prob(zero, -1)


if __name__ == "__main__":
    unittest.main()
