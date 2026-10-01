from dataclasses import replace
import math
import unittest

import numpy as np

from fpp.model.distributions import SeasonParameters
from fpp.model.forecast import ForecastDistribution, ScheduleDistribution
from fpp.model.likelihood import SeasonModel, SeasonOutcome
from fpp.model.production import COUNT_NAMES


class ForecastTests(unittest.TestCase):
    def test_schedule_marginalization_differs_from_realized_schedule_score(self):
        params = replace(SeasonParameters(), participation_probability=0.8)
        model = SeasonModel(params)
        forecast = ForecastDistribution(model, ScheduleDistribution({0: 0.3, 32: 0.7}))
        zero = SeasonOutcome(0, 0, 0.0, {key: 0 for key in COUNT_NAMES})
        self.assertAlmostEqual(math.exp(forecast.log_prob(zero)), 0.3 + 0.7 * 0.2)
        self.assertAlmostEqual(math.exp(model.log_prob(zero, 32)), 0.2)

    def test_unknown_schedule_is_drawn_before_opportunity(self):
        model = SeasonModel(replace(SeasonParameters(), participation_probability=1.0))
        forecast = ForecastDistribution(model, ScheduleDistribution({0: 0.4, 2: 0.6}))
        rng = np.random.default_rng(291)
        outcomes = [forecast.sample(rng) for _ in range(2000)]
        self.assertAlmostEqual(np.mean([o.games == 0 for o in outcomes]), 0.4, delta=0.04)
        self.assertTrue(all(0 <= o.starts <= o.games <= 2 for o in outcomes))

    def test_schedule_weights_are_validated_not_renormalized(self):
        for probabilities in ({}, {1: 0.5}, {True: 1.0}, {-1: 1.0}, {1: float("nan")}):
            with self.assertRaises(ValueError):
                ScheduleDistribution(probabilities)


if __name__ == "__main__":
    unittest.main()
