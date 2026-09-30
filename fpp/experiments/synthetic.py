"""Seeded fixture checks; these outputs are not estimates of transfer gain."""

from __future__ import annotations

import platform
import numpy as np
import scipy

from .. import __version__
from ..model.distributions import SeasonParameters
from ..model.likelihood import SeasonModel
from ..model.observation import OnceRoundedMinutesKernel
from ..model.production import points, valid_counts
from ..serialization import to_jsonable


def run_smoke(seed: int = 20260901, draws: int = 200, score_first: int = 3,
              schedule: int = 32) -> dict:
    """Generate coherent outcomes and score a few draws under the same model."""
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not isinstance(score_first, int) or isinstance(score_first, bool) or not 1 <= score_first <= draws:
        raise ValueError("score_first must be between 1 and draws")
    if not isinstance(schedule, int) or isinstance(schedule, bool) or schedule < 0:
        raise ValueError("schedule must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    params = SeasonParameters()
    kernel = OnceRoundedMinutesKernel()
    model = SeasonModel(params, kernel)
    outcomes = [model.sample(schedule, rng) for _ in range(draws)]
    for outcome in outcomes:
        if not 0 <= outcome.starts <= outcome.games <= schedule:
            raise AssertionError("Sample violates opportunity support")
        if not valid_counts(outcome.counts, outcome.games, params.production):
            raise AssertionError("Sample violates count support")
        if outcome.recorded_minutes < 0:
            raise AssertionError("Sample has negative recorded minutes")
    scored = []
    for outcome in outcomes[:score_first]:
        result = model.score(outcome, schedule)
        if not np.isfinite(result.log_prob):
            raise AssertionError("Generated outcome has non-finite observed-data likelihood")
        scored.append({"outcome": to_jsonable(outcome), "points": points(outcome.counts),
                       "score": to_jsonable(result)})
    return {
        "kind": "synthetic_reference_check",
        "empirical_transfer_gain": None,
        "trained_model": False,
        "seed": seed, "draws": draws, "schedule": schedule,
        "versions": {"fpp": __version__, "python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "parameters": to_jsonable(params),
        "kernel_provenance": to_jsonable(kernel.provenance),
        "numerical_settings": to_jsonable(model.numerical_manifest()),
        "summary": {
            "mean_games": float(np.mean([y.games for y in outcomes])),
            "mean_recorded_minutes": float(np.mean([y.recorded_minutes for y in outcomes])),
            "mean_points": float(np.mean([points(y.counts) for y in outcomes])),
            "zero_game_fraction": float(np.mean([y.games == 0 for y in outcomes])),
        },
        "scored_draws": scored,
    }
