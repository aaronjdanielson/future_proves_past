"""Observed season likelihood: scalar SciPy reference, not a trained tower.

All public scoring paths marginalize latent minutes and overtime.  The
recorded outcome never supplies a plug-in latent production exposure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

from . import production
from .distributions import (SeasonParameters, available_minutes,
                            beta_binomial_log_prob, hurdle_beta_binomial_log_prob,
                            sample_latent_season)
from .observation import (NumericalAccuracy, ObservationDiagnostics,
                          OnceRoundedMinutesKernel, marginalize_overtime)


@dataclass(frozen=True)
class SeasonOutcome:
    games: int
    starts: int
    recorded_minutes: float
    counts: Mapping[str, int]

    @property
    def points(self) -> int:
        return 2 * self.counts["K2"] + 3 * self.counts["K3"] + self.counts["Kf"]

    def as_dict(self) -> dict:
        return {"games": self.games, "starts": self.starts,
                "recorded_minutes": self.recorded_minutes,
                "counts": dict(self.counts), "points": self.points}


@dataclass(frozen=True)
class LikelihoodResult:
    log_prob: float
    diagnostics: ObservationDiagnostics | None


class SeasonModel:
    """Normalized reference season family, conditional on supplied schedule.

    Unknown-schedule marginalization belongs to the forecast/scenario layer.
    A default kernel is a registered *synthetic fixture*, not an empirical
    observation model.  This object does not fit or imply trained parameters.
    """

    def __init__(self, params: SeasonParameters | None = None,
                 kernel: OnceRoundedMinutesKernel | None = None,
                 accuracy: NumericalAccuracy | None = None):
        self.params = params or SeasonParameters()
        self.kernel = kernel or OnceRoundedMinutesKernel()
        self.accuracy = accuracy or NumericalAccuracy()

    def sample(self, schedule: int, rng: np.random.Generator) -> SeasonOutcome:
        latent = sample_latent_season(schedule, self.params, rng)
        maximum = available_minutes(latent.games, latent.overtime, self.params)
        recorded = self.kernel.sample(latent.latent_minutes, maximum, rng)
        return SeasonOutcome(latent.games, latent.starts, recorded, dict(latent.counts))

    def _valid_observation(self, outcome: SeasonOutcome, schedule: int) -> bool:
        if not isinstance(outcome, SeasonOutcome):
            return False
        for value in (outcome.games, outcome.starts):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                return False
        if not 0 <= outcome.starts <= outcome.games <= schedule:
            return False
        if not isinstance(outcome.counts, Mapping):
            return False
        try:
            if not production.valid_counts(outcome.counts, outcome.games, self.params.production):
                return False
            recorded = float(outcome.recorded_minutes)
        except (TypeError, ValueError, KeyError, OverflowError):
            return False
        if not math.isfinite(recorded) or recorded < 0:
            return False
        if outcome.games == 0:
            return recorded == 0 and all(value == 0 for value in outcome.counts.values())
        # The support is the grid plus possible exact maximum-minute atoms.
        on_grid = math.isclose(recorded / self.kernel.step,
                               round(recorded / self.kernel.step), abs_tol=1e-10, rel_tol=0)
        possible_o = ((recorded - outcome.games * self.params.regulation_minutes)
                      / self.params.overtime_minutes)
        on_endpoint = possible_o >= 0 and math.isclose(possible_o, round(possible_o),
                                                      abs_tol=1e-10, rel_tol=0)
        return on_grid or on_endpoint

    def score(self, outcome: SeasonOutcome, schedule: int) -> LikelihoodResult:
        if (isinstance(schedule, (bool, np.bool_)) or
                not isinstance(schedule, (int, np.integer)) or schedule < 0):
            raise ValueError("Schedule must be a nonnegative integer")
        if not self._valid_observation(outcome, schedule):
            return LikelihoodResult(-math.inf, None)
        p = self.params
        games_log = hurdle_beta_binomial_log_prob(outcome.games, schedule,
                                                   p.participation_probability,
                                                   p.games_alpha, p.games_beta)
        if outcome.games == 0 or games_log == -math.inf:
            return LikelihoodResult(games_log, None)
        starts_log = beta_binomial_log_prob(outcome.starts, outcome.games,
                                            p.starts_alpha, p.starts_beta)

        def production_at(latent_minutes):
            return production.log_prob(outcome.counts, latent_minutes, outcome.games,
                                       p.production, regulation_minutes=p.regulation_minutes)

        observed = marginalize_overtime(outcome.recorded_minutes, outcome.games,
                                         p.minutes_alpha, p.minutes_beta,
                                         p.minutes_max_probability, p.overtime_rate_per_game,
                                         self.kernel, production_at,
                                         p.regulation_minutes, p.overtime_minutes,
                                         self.accuracy)
        return LikelihoodResult(games_log + starts_log + observed.log_probability,
                                observed.diagnostics)

    def log_prob(self, outcome: SeasonOutcome, schedule: int) -> float:
        return self.score(outcome, schedule).log_prob

    def numerical_manifest(self) -> dict:
        return {"implementation": "scipy-reference; no autograd or fitting",
                "kernel": self.kernel.as_dict(), "accuracy": asdict(self.accuracy),
                "total_error_certified": False,
                "quadrature_validation": "adaptive refinement; empirical convergence",
                "overtime": "untruncated Poisson; no retained-mass renormalization"}
