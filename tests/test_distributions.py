"""Normalization and generative-contract tests for the scientific reference."""
import dataclasses
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import betabinom, nbinom, poisson

from fpp.model import distributions as d
from fpp.model import production as p


class DiscreteDistributionTests(unittest.TestCase):
    def test_beta_binomial_normalized_and_matches_independent_reference(self):
        for n, alpha, beta in ((0, 1, 1), (1, 0.2, 4), (32, 2, 3), (34, 0.01, 0.03)):
            logs = np.array([d.beta_binomial_log_prob(k, n, alpha, beta) for k in range(n + 1)])
            self.assertAlmostEqual(float(logsumexp(logs)), 0, places=11)
            np.testing.assert_allclose(logs, betabinom.logpmf(np.arange(n + 1), n, alpha, beta), atol=1e-11)

    def test_hurdle_handles_nearly_all_zero_base_distribution(self):
        for participation in (0, 0.17, 1):
            for schedule, alpha, beta in ((0, 2, 3), (1, 1, 1), (33, 1e-18, 40)):
                probs = np.array([math.exp(d.hurdle_beta_binomial_log_prob(g, schedule, participation, alpha, beta)) for g in range(schedule + 1)])
                self.assertAlmostEqual(float(probs.sum()), 1, places=12)
                self.assertAlmostEqual(probs[0], 1 if schedule == 0 else 1 - participation, places=13)

    def test_hurdle_sampling_agrees_with_pmf(self):
        rng = np.random.default_rng(828)
        draws = np.array([d.hurdle_beta_binomial_sample(5, 0.65, 2.2, 1.3, rng) for _ in range(12000)])
        frequencies = np.bincount(draws, minlength=6) / len(draws)
        probabilities = np.exp([d.hurdle_beta_binomial_log_prob(k, 5, 0.65, 2.2, 1.3) for k in range(6)])
        self.assertTrue(np.all(abs(frequencies - probabilities) < 6 * np.sqrt(probabilities * (1 - probabilities) / len(draws)) + 1 / len(draws)))

    def test_nb_and_poisson_normalization_including_zero_mean(self):
        for mean, dispersion in ((0, 3), (0.1, 0.3), (8, 2), (80, 100)):
            upper = 0 if mean == 0 else int(nbinom.ppf(1 - 1e-12, dispersion, dispersion / (dispersion + mean)))
            logs = np.array([d.negative_binomial_log_prob(k, mean, dispersion) for k in range(upper + 1)])
            self.assertAlmostEqual(float(np.exp(logs).sum()), 1, places=10)
            np.testing.assert_allclose(logs, nbinom.logpmf(np.arange(upper + 1), dispersion, dispersion / (dispersion + mean)), rtol=1e-10, atol=1e-10)
        for mean in (0, 0.2, 25):
            upper = int(poisson.ppf(1 - 1e-12, mean))
            self.assertAlmostEqual(sum(math.exp(d.poisson_log_prob(k, mean)) for k in range(upper + 1)), 1, places=10)

    def test_nb_sampler_mean_variance_and_zero_mass(self):
        rng = np.random.default_rng(246)
        mean, dispersion = 9.0, 4.0
        draws = np.array([d.negative_binomial_sample(mean, dispersion, rng) for _ in range(25000)])
        variance = mean + mean**2 / dispersion
        self.assertLess(abs(draws.mean() - mean), 6 * math.sqrt(variance / len(draws)))
        self.assertLess(abs(draws.var() / variance - 1), 0.10)
        zero_mass = math.exp(d.negative_binomial_log_prob(0, mean, dispersion))
        self.assertLess(abs(np.mean(draws == 0) - zero_mass), 6 * math.sqrt(zero_mass * (1 - zero_mass) / len(draws)))

    def test_truncated_nb_normalization_even_when_retained_mass_is_tiny(self):
        for mean, dispersion, upper in ((0, 1, 0), (0.1, 0.2, 5), (40, 5, 10), (1e12, 500, 5)):
            logs = np.array([d.truncated_negative_binomial_log_prob(k, mean, dispersion, upper) for k in range(upper + 1)])
            self.assertAlmostEqual(float(logsumexp(logs)), 0, places=10)
            self.assertLessEqual(float(logs.max()), 1e-10)
        rng = np.random.default_rng(133)
        draws = [d.truncated_negative_binomial_sample(200, 2, 5, rng) for _ in range(200)]
        self.assertTrue(all(0 <= k <= 5 for k in draws))
        self.assertGreater(len(set(draws)), 1)

    def test_overtime_sampler_is_not_capped(self):
        rng = np.random.default_rng(447)
        draws = [d.poisson_sample(35, rng) for _ in range(500)]
        self.assertGreater(max(draws), 45)
        self.assertTrue(all(math.isfinite(d.poisson_log_prob(k, 35)) for k in draws))

    def test_impossible_count_values_score_negative_infinity(self):
        for value in (-1, 0.5, float("nan"), float("inf"), True, "1", None):
            self.assertEqual(d.beta_binomial_log_prob(value, 5, 2, 3), -np.inf)
            self.assertEqual(d.negative_binomial_log_prob(value, 2, 3), -np.inf)
            self.assertEqual(d.poisson_log_prob(value, 2), -np.inf)
        self.assertEqual(d.beta_binomial_log_prob(6, 5, 2, 3), -np.inf)
        self.assertEqual(d.truncated_negative_binomial_log_prob(6, 2, 3, 5), -np.inf)
        self.assertEqual(d.poisson_log_prob(1, 0), -np.inf)
        self.assertEqual(d.negative_binomial_log_prob(1, 0, 3), -np.inf)

    def test_invalid_distribution_parameters_raise(self):
        for call in (
            lambda: d.beta_binomial_log_prob(0, -1, 2, 3),
            lambda: d.beta_binomial_log_prob(0, 2, 0, 3),
            lambda: d.negative_binomial_log_prob(0, -1, 2),
            lambda: d.poisson_log_prob(0, float("nan")),
            lambda: d.hurdle_beta_binomial_log_prob(0, 3, 1.1, 2, 3),
            lambda: d.SeasonParameters(minutes_alpha=0),
            lambda: d.SeasonParameters(overtime_rate_per_game=-0.1),
            lambda: d.SeasonParameters(regulation_minutes=0),
        ):
            with self.assertRaises(ValueError):
                call()


class LatentMinutesTests(unittest.TestCase):
    def test_continuous_mass_plus_endpoint_atom_normalizes(self):
        for alpha, beta, atom in ((2, 3, 0), (0.5, 0.7, 0.3), (5, 4, 1)):
            params = d.SeasonParameters(minutes_alpha=alpha, minutes_beta=beta, minutes_max_probability=atom)
            maximum = d.available_minutes(3, 2, params)
            integral, error = quad(lambda m: math.exp(d.latent_minutes_log_prob(m, 3, 2, params)), 0, maximum, epsabs=2e-10)
            endpoint = math.exp(d.latent_minutes_log_prob(maximum, 3, 2, params, at_maximum=True))
            self.assertAlmostEqual(integral + endpoint, 1, places=8)
            self.assertLess(error, 1e-7)
            self.assertEqual(d.latent_minutes_log_prob(maximum, 3, 2, params), -np.inf)
            self.assertEqual(d.latent_minutes_log_prob(maximum - 1, 3, 2, params, at_maximum=True), -np.inf)

    def test_endpoint_sampler_and_fraction_mean(self):
        rng = np.random.default_rng(816)
        params = d.SeasonParameters(minutes_alpha=2, minutes_beta=4, minutes_max_probability=0.25)
        draws = [d.latent_minutes_sample(2, 0, params, rng) for _ in range(6000)]
        atom_fraction = np.mean([atom for _, atom in draws])
        self.assertLess(abs(atom_fraction - 0.25), 0.03)
        continuous = [m / 80 for m, atom in draws if not atom]
        self.assertLess(abs(np.mean(continuous) - 1 / 3), 0.02)
        self.assertTrue(all(m == 80 if atom else 0 < m < 80 for m, atom in draws))

    def test_zero_game_branch_and_impossible_exposure(self):
        params = d.SeasonParameters()
        self.assertEqual(d.latent_minutes_sample(0, 0, params, np.random.default_rng(11)), (0.0, False))
        self.assertEqual(d.latent_minutes_log_prob(0, 0, 0, params), 0)
        self.assertEqual(d.latent_minutes_log_prob(1, 0, 0, params), -np.inf)
        with self.assertRaises(ValueError):
            d.available_minutes(0, 1, params)

    def test_boundary_rounded_continuous_draw_fails_without_retry(self):
        class EndpointGenerator:
            def __init__(self, value):
                self.value = value
                self.beta_calls = 0

            def random(self):
                return 0.5

            def beta(self, alpha, beta):
                self.beta_calls += 1
                return self.value

        params = d.SeasonParameters(minutes_max_probability=0)
        for value in (0.0, 1.0):
            rng = EndpointGenerator(value)
            with self.assertRaisesRegex(FloatingPointError, "without resampling or clamping"):
                d.latent_minutes_sample(2, 0, params, rng)
            self.assertEqual(rng.beta_calls, 1)

    def test_endpoint_heavy_beta_fails_explicitly_on_unrepresentable_draw(self):
        params = d.SeasonParameters(minutes_alpha=0.1, minutes_beta=0.01, minutes_max_probability=0)
        rng = np.random.default_rng(42)
        d.latent_minutes_sample(2, 0, params, rng)
        d.latent_minutes_sample(2, 0, params, rng)
        # The third float64 beta draw is exactly 1 for this fixed seed. It
        # must fail, rather than being discarded and biasing accepted draws.
        with self.assertRaises(FloatingPointError):
            d.latent_minutes_sample(2, 0, params, rng)


class ProductionAndSeasonTests(unittest.TestCase):
    def test_conditional_attempt_make_pair_normalizes(self):
        params = p.ProductionParameters()
        mean = 2.0 * params.rates["A2"]
        dispersion = params.dispersions["A2"]
        upper = int(nbinom.ppf(1 - 1e-12, dispersion, dispersion / (dispersion + mean)))
        total = 0.0
        for attempts in range(upper + 1):
            total += sum(math.exp(d.negative_binomial_log_prob(attempts, mean, dispersion) + d.beta_binomial_log_prob(makes, attempts, params.makes_alpha["K2"], params.makes_beta["K2"])) for makes in range(attempts + 1))
        self.assertAlmostEqual(total, 1, places=10)

    def test_sampling_preserves_joint_support_and_points(self):
        params = d.SeasonParameters()
        rng = np.random.default_rng(98)
        for _ in range(250):
            outcome = d.sample_latent_season(32, params, rng)
            self.assertTrue(0 <= outcome.starts <= outcome.games <= 32)
            self.assertLessEqual(outcome.latent_minutes, d.available_minutes(outcome.games, outcome.overtime, params))
            self.assertTrue(p.valid_counts(outcome.counts, outcome.games, params.production))
            self.assertEqual(tuple(outcome.counts), p.COUNT_NAMES)
            self.assertLessEqual(p.log_prob(outcome.counts, outcome.latent_minutes, outcome.games, params.production), 0)
            self.assertTrue(math.isfinite(d.latent_season_log_prob(outcome, 32, params)))
            self.assertEqual(p.points(outcome.counts), 2 * outcome.counts["K2"] + 3 * outcome.counts["K3"] + outcome.counts["Kf"])

    def test_zero_schedule_and_nonparticipation_force_all_zeros(self):
        for schedule, params in ((0, d.SeasonParameters(participation_probability=1)), (32, d.SeasonParameters(participation_probability=0))):
            outcome = d.sample_latent_season(schedule, params, np.random.default_rng(312))
            self.assertEqual((outcome.games, outcome.starts, outcome.overtime, outcome.latent_minutes), (0, 0, 0, 0))
            self.assertTrue(all(v == 0 for v in outcome.counts.values()))
            self.assertEqual(d.latent_season_log_prob(outcome, schedule, params), 0)
        counts = dict.fromkeys(p.COUNT_NAMES, 0)
        counts["A2"] = 1
        self.assertEqual(p.log_prob(counts, 0, 0, p.ProductionParameters()), -np.inf)

    def test_bad_production_is_rejected_without_clipping(self):
        params = p.ProductionParameters()
        zeros = dict.fromkeys(p.COUNT_NAMES, 0)
        for counts in ({}, dict(zeros, K2=1), dict(zeros, PF=6), dict(zeros, AST=-1), dict(zeros, ORB=1.5), dict(zeros, extra=0), None):
            self.assertEqual(p.log_prob(counts, 5, 1, params), -np.inf)
        for minutes in (-1, float("nan"), "bad", None):
            self.assertEqual(p.log_prob(zeros, minutes, 1, params), -np.inf)
        with self.assertRaises(ValueError):
            p.sample(20, 0, params, np.random.default_rng(5))
        with self.assertRaises(ValueError):
            p.ProductionParameters(rates={})
        with self.assertRaises(ValueError):
            p.ProductionParameters(foul_limit_per_game=0)

    def test_generator_reproducibility_and_no_parameter_aliasing(self):
        rates = dict(p.ProductionParameters().rates)
        production_params = p.ProductionParameters(rates=rates)
        rates["A2"] = 1000
        self.assertEqual(production_params.rates["A2"], 9)
        with self.assertRaises(TypeError):
            production_params.rates["A2"] = 3
        params = d.SeasonParameters(production=production_params)
        left = np.random.default_rng(45)
        right = np.random.default_rng(45)
        self.assertEqual([d.sample_latent_season(8, params, left) for _ in range(12)], [d.sample_latent_season(8, params, right) for _ in range(12)])


if __name__ == "__main__":
    unittest.main()
