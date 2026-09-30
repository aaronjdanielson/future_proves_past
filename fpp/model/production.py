"""Normalized, ordered primitive-count distribution from paper eq:boxscore.

This initial executable submodel has constant rate/shape heads. It implements
the correct conditional sampling order, attempt/make constraints and foul
truncation; learned dependence of head parameters on preceding counts is a
subsequent neural-model phase. It never propagates conditional means as draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .distributions import (
    _count_parameter, _is_count, _positive,
    beta_binomial_log_prob, beta_binomial_sample,
    negative_binomial_log_prob, negative_binomial_sample,
    truncated_negative_binomial_log_prob, truncated_negative_binomial_sample,
)

COUNT_NAMES = ("A2", "K2", "A3", "K3", "Af", "Kf", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF")
RATE_NAMES = ("A2", "A3", "Af", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF")
MAKE_ATTEMPTS = {"K2": "A2", "K3": "A3", "Kf": "Af"}


def _rates() -> dict[str, float]:
    return dict(zip(RATE_NAMES, (9.0, 6.0, 4.0, 1.5, 4.0, 3.0, 1.0, 0.6, 2.0, 3.0)))


def _freeze_positive_mapping(values: Mapping[str, float], keys: tuple[str, ...], name: str) -> Mapping[str, float]:
    if set(values) != set(keys):
        raise ValueError(f"{name} must contain exactly {keys}")
    copied = {}
    for key in keys:
        value = float(values[key])
        _positive(value, f"{name}[{key}]")
        copied[key] = value
    return MappingProxyType(copied)


@dataclass(frozen=True)
class ProductionParameters:
    """Rates per regulation game and NB dispersions; BB shapes for makes."""
    rates: Mapping[str, float] = field(default_factory=_rates)
    dispersions: Mapping[str, float] = field(default_factory=lambda: {key: 15.0 for key in RATE_NAMES})
    makes_alpha: Mapping[str, float] = field(default_factory=lambda: {"K2": 10.0, "K3": 7.0, "Kf": 15.0})
    makes_beta: Mapping[str, float] = field(default_factory=lambda: {"K2": 10.0, "K3": 13.0, "Kf": 5.0})
    foul_limit_per_game: int = 5

    def __post_init__(self) -> None:
        for name, keys in (("rates", RATE_NAMES), ("dispersions", RATE_NAMES), ("makes_alpha", tuple(MAKE_ATTEMPTS)), ("makes_beta", tuple(MAKE_ATTEMPTS))):
            object.__setattr__(self, name, _freeze_positive_mapping(getattr(self, name), keys, name))
        _count_parameter(self.foul_limit_per_game, "foul_limit_per_game")
        if self.foul_limit_per_game == 0:
            raise ValueError("foul_limit_per_game must be positive")


def valid_counts(counts: Mapping[str, int], games: int, params: ProductionParameters) -> bool:
    if not isinstance(counts, Mapping) or set(counts) != set(COUNT_NAMES):
        return False
    if not _is_count(games) or any(not _is_count(counts[key]) for key in COUNT_NAMES):
        return False
    if any(counts[make] > counts[attempt] for make, attempt in MAKE_ATTEMPTS.items()):
        return False
    if counts["PF"] > params.foul_limit_per_game * games:
        return False
    return games > 0 or all(counts[key] == 0 for key in COUNT_NAMES)


def log_prob(
    counts: Mapping[str, int], latent_minutes: float, games: int,
    params: ProductionParameters, regulation_minutes: float = 40.0,
) -> float:
    """Return the joint discrete count log PMF, or -inf off support."""
    _positive(regulation_minutes, "regulation_minutes")
    if (
        not valid_counts(counts, games, params)
        or not isinstance(latent_minutes, (int, float, np.integer, np.floating))
        or not isfinite(latent_minutes)
        or latent_minutes < 0
    ):
        return -np.inf
    if games == 0:
        return 0.0 if latent_minutes == 0 else -np.inf
    exposure = latent_minutes / regulation_minutes
    result = 0.0
    for name in COUNT_NAMES:
        if name in MAKE_ATTEMPTS:
            result += beta_binomial_log_prob(counts[name], counts[MAKE_ATTEMPTS[name]], params.makes_alpha[name], params.makes_beta[name])
        elif name == "PF":
            result += truncated_negative_binomial_log_prob(counts[name], exposure * params.rates[name], params.dispersions[name], params.foul_limit_per_game * games)
        else:
            result += negative_binomial_log_prob(counts[name], exposure * params.rates[name], params.dispersions[name])
    return float(result)


def sample(
    latent_minutes: float, games: int, params: ProductionParameters,
    rng: np.random.Generator, regulation_minutes: float = 40.0,
) -> dict[str, int]:
    games = _count_parameter(games, "games")
    _positive(regulation_minutes, "regulation_minutes")
    if not isfinite(latent_minutes) or latent_minutes < 0:
        raise ValueError("latent_minutes must be finite and nonnegative")
    if games == 0:
        if latent_minutes != 0:
            raise ValueError("zero games requires zero latent minutes")
        return dict.fromkeys(COUNT_NAMES, 0)
    exposure = latent_minutes / regulation_minutes
    counts = {}
    for name in COUNT_NAMES:
        if name in MAKE_ATTEMPTS:
            counts[name] = beta_binomial_sample(counts[MAKE_ATTEMPTS[name]], params.makes_alpha[name], params.makes_beta[name], rng)
        elif name == "PF":
            counts[name] = truncated_negative_binomial_sample(exposure * params.rates[name], params.dispersions[name], params.foul_limit_per_game * games, rng)
        else:
            counts[name] = negative_binomial_sample(exposure * params.rates[name], params.dispersions[name], rng)
    return counts


def derived_totals(counts: Mapping[str, int]) -> dict[str, int]:
    """Deterministic identities; these are not additional likelihood factors."""
    if set(counts) != set(COUNT_NAMES) or any(not _is_count(counts[key]) for key in COUNT_NAMES):
        raise ValueError("counts must contain exactly the 13 nonnegative integer primitives")
    if any(counts[make] > counts[attempt] for make, attempt in MAKE_ATTEMPTS.items()):
        raise ValueError("makes exceed attempts")
    return {
        "points": int(2 * counts["K2"] + 3 * counts["K3"] + counts["Kf"]),
        "rebounds": int(counts["ORB"] + counts["DRB"]),
        "field_goals_made": int(counts["K2"] + counts["K3"]),
        "field_goals_attempted": int(counts["A2"] + counts["A3"]),
    }


def points(counts: Mapping[str, int]) -> int:
    """Exact points identity from the 13 primitive count fields."""
    return derived_totals(counts)["points"]
