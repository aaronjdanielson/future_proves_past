"""Marginalize a supplied discrete schedule forecast in the reference model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from .likelihood import SeasonModel, SeasonOutcome


@dataclass(frozen=True)
class ScheduleDistribution:
    """Explicit normalized schedule PMF; weights are never silently normalized."""
    probabilities: Mapping[int, float]

    def __post_init__(self):
        if not self.probabilities:
            raise ValueError("A schedule distribution must contain positive mass")
        probabilities = {}
        for schedule, probability in self.probabilities.items():
            if isinstance(schedule, bool) or not isinstance(schedule, int) or schedule < 0:
                raise ValueError("Schedule support must contain nonnegative integers")
            if not math.isfinite(probability) or probability < 0:
                raise ValueError("Schedule probabilities must be finite and nonnegative")
            probabilities[schedule] = float(probability)
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Schedule probabilities must sum to one")
        object.__setattr__(self, "probabilities", MappingProxyType(dict(sorted(probabilities.items()))))

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(list(self.probabilities), p=list(self.probabilities.values())))


class ForecastDistribution:
    """Propagate schedule uncertainty into season draws and observed scores."""

    def __init__(self, model: SeasonModel, schedule: ScheduleDistribution):
        self.model, self.schedule = model, schedule

    def sample(self, rng: np.random.Generator) -> SeasonOutcome:
        return self.model.sample(self.schedule.sample(rng), rng)

    def log_prob(self, outcome: SeasonOutcome) -> float:
        terms = [math.log(probability) + self.model.log_prob(outcome, schedule)
                 for schedule, probability in self.schedule.probabilities.items() if probability > 0]
        return float(logsumexp(terms))
