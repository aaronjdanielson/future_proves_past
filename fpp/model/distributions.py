"""Scalar NumPy/SciPy reference for the paper's latent season distribution.

All randomness is supplied explicitly. Count scores are normalized log PMFs;
``latent_minutes_log_prob`` uses Lebesgue measure inside the support and a
separate unit atom at its upper endpoint. These are scientific reference
functions, not differentiable neural training code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite, log, log1p
from typing import TYPE_CHECKING, Mapping

import numpy as np
from scipy.special import betaln, gammaln, logsumexp, xlogy

if TYPE_CHECKING:
    from .production import ProductionParameters


def _positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")


def _nonnegative(value: float, name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _probability(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _is_count(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and isfinite(float(value))
        and float(value) >= 0
        and float(value).is_integer()
    )


def _count_parameter(value: int, name: str) -> int:
    if not _is_count(value):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def beta_binomial_log_prob(k: int, n: int, alpha: float, beta: float) -> float:
    n = _count_parameter(n, "n")
    _positive(alpha, "alpha")
    _positive(beta, "beta")
    if not _is_count(k) or k > n:
        return -np.inf
    if n == 0:
        return 0.0
    return float(
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        + betaln(k + alpha, n - k + beta) - betaln(alpha, beta)
    )


def beta_binomial_sample(n: int, alpha: float, beta: float, rng: np.random.Generator) -> int:
    n = _count_parameter(n, "n")
    _positive(alpha, "alpha")
    _positive(beta, "beta")
    if n == 0:
        return 0
    return int(rng.binomial(n, rng.beta(alpha, beta)))


@lru_cache(maxsize=256)
def _positive_beta_binomial_logs(n: int, alpha: float, beta: float) -> tuple[float, ...]:
    # Summing the positive support avoids cancellation in 1 - b(0) when
    # virtually all unconditional beta-binomial mass lies at zero.
    logs = np.array([beta_binomial_log_prob(k, n, alpha, beta) for k in range(1, n + 1)])
    logs -= logsumexp(logs)
    return tuple(float(x) for x in logs)


def hurdle_beta_binomial_log_prob(
    games: int, schedule: int, participation_probability: float, alpha: float, beta: float
) -> float:
    schedule = _count_parameter(schedule, "schedule")
    _probability(participation_probability, "participation_probability")
    _positive(alpha, "alpha")
    _positive(beta, "beta")
    if not _is_count(games) or games > schedule:
        return -np.inf
    if schedule == 0:
        return 0.0
    if games == 0:
        return log1p(-participation_probability) if participation_probability < 1 else -np.inf
    if participation_probability == 0:
        return -np.inf
    return log(participation_probability) + _positive_beta_binomial_logs(schedule, alpha, beta)[int(games) - 1]


def hurdle_beta_binomial_sample(
    schedule: int, participation_probability: float, alpha: float, beta: float,
    rng: np.random.Generator,
) -> int:
    schedule = _count_parameter(schedule, "schedule")
    _probability(participation_probability, "participation_probability")
    _positive(alpha, "alpha")
    _positive(beta, "beta")
    if schedule == 0 or rng.random() >= participation_probability:
        return 0
    probabilities = np.exp(_positive_beta_binomial_logs(schedule, alpha, beta))
    return int(rng.choice(schedule, p=probabilities)) + 1


def poisson_log_prob(k: int, mean: float) -> float:
    _nonnegative(mean, "mean")
    if not _is_count(k):
        return -np.inf
    if mean == 0:
        return 0.0 if k == 0 else -np.inf
    return float(xlogy(k, mean) - mean - gammaln(k + 1))


def poisson_sample(mean: float, rng: np.random.Generator) -> int:
    _nonnegative(mean, "mean")
    return int(rng.poisson(mean))


def negative_binomial_log_prob(k: int, mean: float, dispersion: float) -> float:
    """NB(mean, dispersion), with variance mean + mean**2 / dispersion."""
    _nonnegative(mean, "mean")
    _positive(dispersion, "dispersion")
    if not _is_count(k):
        return -np.inf
    if mean == 0:
        return 0.0 if k == 0 else -np.inf
    if k == 0:
        return float(-dispersion * np.log1p(mean / dispersion))
    return float(
        gammaln(k + dispersion) - gammaln(dispersion) - gammaln(k + 1)
        - dispersion * np.log1p(mean / dispersion)
        - k * np.log1p(dispersion / mean)
    )


def negative_binomial_sample(mean: float, dispersion: float, rng: np.random.Generator) -> int:
    _nonnegative(mean, "mean")
    _positive(dispersion, "dispersion")
    if mean == 0:
        return 0
    return int(rng.negative_binomial(dispersion, dispersion / (dispersion + mean)))


def _truncated_nb_logs(mean: float, dispersion: float, upper: int) -> np.ndarray:
    upper = _count_parameter(upper, "upper")
    _nonnegative(mean, "mean")
    _positive(dispersion, "dispersion")
    if mean == 0:
        logs = np.full(upper + 1, -np.inf)
        logs[0] = 0.0
        return logs
    # Vectorizing the finite support matters because the observed likelihood
    # evaluates this normalization at every latent-minute quadrature node.
    support = np.arange(upper + 1, dtype=float)
    logs = (
        gammaln(support + dispersion) - gammaln(dispersion) - gammaln(support + 1)
        - dispersion * np.log1p(mean / dispersion)
        - support * np.log1p(dispersion / mean)
    )
    return logs - logsumexp(logs)


def truncated_negative_binomial_log_prob(k: int, mean: float, dispersion: float, upper: int) -> float:
    upper = _count_parameter(upper, "upper")
    _nonnegative(mean, "mean")
    _positive(dispersion, "dispersion")
    if not _is_count(k) or k > upper:
        return -np.inf
    return float(_truncated_nb_logs(mean, dispersion, upper)[int(k)])


def truncated_negative_binomial_sample(
    mean: float, dispersion: float, upper: int, rng: np.random.Generator
) -> int:
    probabilities = np.exp(_truncated_nb_logs(mean, dispersion, upper))
    return int(rng.choice(len(probabilities), p=probabilities))


def _default_production() -> ProductionParameters:
    from .production import ProductionParameters
    return ProductionParameters()


@dataclass(frozen=True)
class SeasonParameters:
    """Fixed conditional heads for executable likelihood/simulation checks.

    A future learned model emits these parameters given player, history,
    destination, age and time features; this dataclass does not train that model.
    """
    participation_probability: float = 0.85
    games_alpha: float = 5.0
    games_beta: float = 1.5
    starts_alpha: float = 2.0
    starts_beta: float = 3.0
    overtime_rate_per_game: float = 0.04
    minutes_alpha: float = 2.5
    minutes_beta: float = 2.0
    minutes_max_probability: float = 0.01
    regulation_minutes: float = 40.0
    overtime_minutes: float = 5.0
    production: ProductionParameters = field(default_factory=_default_production)

    def __post_init__(self) -> None:
        from .production import ProductionParameters
        _probability(self.participation_probability, "participation_probability")
        _probability(self.minutes_max_probability, "minutes_max_probability")
        for name in ("games_alpha", "games_beta", "starts_alpha", "starts_beta", "minutes_alpha", "minutes_beta", "regulation_minutes", "overtime_minutes"):
            _positive(getattr(self, name), name)
        _nonnegative(self.overtime_rate_per_game, "overtime_rate_per_game")
        if not isinstance(self.production, ProductionParameters):
            raise TypeError("production must be ProductionParameters")


def available_minutes(games: int, overtime: int, params: SeasonParameters) -> float:
    games = _count_parameter(games, "games")
    overtime = _count_parameter(overtime, "overtime")
    if games == 0 and overtime != 0:
        raise ValueError("overtime must be zero when games is zero")
    return params.regulation_minutes * games + params.overtime_minutes * overtime


def latent_minutes_log_prob(
    minutes: float, games: int, overtime: int, params: SeasonParameters,
    *, at_maximum: bool = False,
) -> float:
    """Mixed-measure score with an explicitly identified upper-endpoint atom."""
    maximum = available_minutes(games, overtime, params)
    if not isfinite(minutes):
        return -np.inf
    if games == 0:
        return 0.0 if minutes == 0 and not at_maximum else -np.inf
    if at_maximum:
        if minutes != maximum or params.minutes_max_probability == 0:
            return -np.inf
        return log(params.minutes_max_probability)
    if not 0 < minutes < maximum or params.minutes_max_probability == 1:
        return -np.inf
    fraction = minutes / maximum
    return float(
        log1p(-params.minutes_max_probability)
        + xlogy(params.minutes_alpha - 1, fraction)
        + xlogy(params.minutes_beta - 1, 1 - fraction)
        - betaln(params.minutes_alpha, params.minutes_beta) - log(maximum)
    )


def latent_minutes_sample(
    games: int, overtime: int, params: SeasonParameters, rng: np.random.Generator
) -> tuple[float, bool]:
    """Draw once, failing if the continuous value is not representable.

    This float64 reference supports sampling only when the sampled continuous
    minute value can be represented strictly between zero and its maximum.
    Endpoint-heavy beta shapes can violate that condition. A boundary-rounded
    continuous draw raises ``FloatingPointError``; retrying or clamping would
    change the requested probability distribution.
    """
    maximum = available_minutes(games, overtime, params)
    if games == 0:
        return 0.0, False
    if rng.random() < params.minutes_max_probability:
        return maximum, True
    minutes = maximum * float(rng.beta(params.minutes_alpha, params.minutes_beta))
    if not 0 < minutes < maximum:
        raise FloatingPointError(
            "continuous beta minute draw is not representable strictly inside "
            "the support; sampling stopped without resampling or clamping"
        )
    return minutes, False


@dataclass(frozen=True)
class LatentSeasonOutcome:
    games: int
    starts: int
    overtime: int
    latent_minutes: float
    minutes_at_maximum: bool
    counts: Mapping[str, int]


def sample_latent_season(schedule: int, params: SeasonParameters, rng: np.random.Generator) -> LatentSeasonOutcome:
    from . import production
    games = hurdle_beta_binomial_sample(schedule, params.participation_probability, params.games_alpha, params.games_beta, rng)
    starts = beta_binomial_sample(games, params.starts_alpha, params.starts_beta, rng)
    overtime = poisson_sample(games * params.overtime_rate_per_game, rng)
    minutes, at_maximum = latent_minutes_sample(games, overtime, params, rng)
    counts = production.sample(minutes, games, params.production, rng, regulation_minutes=params.regulation_minutes)
    return LatentSeasonOutcome(games, starts, overtime, minutes, at_maximum, counts)


def latent_season_log_prob(outcome: LatentSeasonOutcome, schedule: int, params: SeasonParameters) -> float:
    from . import production
    game_score = hurdle_beta_binomial_log_prob(outcome.games, schedule, params.participation_probability, params.games_alpha, params.games_beta)
    if not np.isfinite(game_score):
        return -np.inf
    if not _is_count(outcome.overtime) or (outcome.games == 0 and outcome.overtime != 0):
        return -np.inf
    return float(
        game_score
        + beta_binomial_log_prob(outcome.starts, outcome.games, params.starts_alpha, params.starts_beta)
        + poisson_log_prob(outcome.overtime, outcome.games * params.overtime_rate_per_game)
        + latent_minutes_log_prob(outcome.latent_minutes, outcome.games, outcome.overtime, params, at_maximum=outcome.minutes_at_maximum)
        + production.log_prob(outcome.counts, outcome.latent_minutes, outcome.games, params.production, regulation_minutes=params.regulation_minutes)
    )
