"""Milestone 0.2 validation of the differentiable likelihood against the reference.

Produces one JSON report: value agreement on a seeded synthetic batch that
covers the structural cases (zero games, full-minute atoms, recorded zero
minutes with appearances, overtime-only recorded values, rare counts),
autodiff-versus-finite-difference gradient agreement, node-count refinement,
and measured forward+backward cost for a realistic batch. It fits nothing.
"""
from __future__ import annotations

import platform

import numpy as np

from ..model.distributions import SeasonParameters
from ..model.likelihood import SeasonModel, SeasonOutcome
from ..model.observation import OnceRoundedMinutesKernel
from ..model.production import COUNT_NAMES
from ..serialization import to_jsonable


def structural_cases(params: SeasonParameters, rng, schedule: int):
    """Hand-built observations exercising every branch of the observation model."""
    ref = SeasonModel(params, OnceRoundedMinutesKernel())
    draws = [ref.sample(schedule, rng) for _ in range(12)]
    positive = next(d for d in draws if d.games >= 8)
    g = positive.games
    reg, ot = params.regulation_minutes, params.overtime_minutes
    zero_counts = dict.fromkeys(COUNT_NAMES, 0)
    cases = {
        "zero_games": SeasonOutcome(0, 0, 0.0, zero_counts),
        "atom_full_minutes": SeasonOutcome(g, positive.starts, reg * g, positive.counts),
        "recorded_zero_with_appearances": SeasonOutcome(g, 0, 0.0, zero_counts),
        "overtime_endpoint_o1": SeasonOutcome(g, positive.starts, reg * g + ot, positive.counts),
        "overtime_interior_o1": SeasonOutcome(g, positive.starts, reg * g + ot - 2.0, positive.counts),
        "rare_counts": SeasonOutcome(g, positive.starts, positive.recorded_minutes,
                                     {**positive.counts, "A2": positive.counts["A2"] + 120,
                                      "K2": positive.counts["K2"] + 60}),
        "last_grid_cell": SeasonOutcome(g, positive.starts, reg * g - 1.0, positive.counts),
    }
    return cases, draws


def run_gradcheck(seed: int = 20260901, batch_size: int = 256, schedule: int = 32,
                  nodes: int = 32, params: SeasonParameters | None = None) -> dict:
    import torch
    from ..model.torch_likelihood import (DifferentiableSeasonLikelihood, InterceptOnlySeasonModel,
                                          SeasonBatch, finite_difference_check, time_batch,
                                          value_agreement)
    params = params or SeasonParameters()
    rng = np.random.default_rng(seed)
    cases, draws = structural_cases(params, rng, schedule)
    outcomes = list(cases.values()) + draws
    model = InterceptOnlySeasonModel(params)
    report = {"kind": "differentiable_likelihood_validation", "seed": seed, "schedule": schedule,
              "empirical_transfer_gain": None, "trained_model": False,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "torch": torch.__version__},
              "kernel_provenance": to_jsonable(OnceRoundedMinutesKernel().provenance)}

    refinement = {}
    for k in (16, 32, 64):
        lik = DifferentiableSeasonLikelihood(model, quadrature_nodes=k)
        agreement = value_agreement(lik, outcomes, schedule)
        refinement[str(k)] = {key: agreement[key] for key in
                              ("n", "same_support", "max_abs_diff_nats", "mean_abs_diff_nats", "overtime_terms")}
    report["value_agreement_by_nodes"] = refinement
    lik = DifferentiableSeasonLikelihood(model, quadrature_nodes=nodes)
    agreement = value_agreement(lik, outcomes, schedule)
    report["structural_cases"] = {name: {"reference": agreement["reference"][i], "ours": agreement["ours"][i]}
                                  for i, name in enumerate(cases)}

    valid_for_grad = [o for o in outcomes if o.games > 0][:8]
    report["gradient_check"] = finite_difference_check(
        lik, SeasonBatch.from_outcomes(valid_for_grad, schedule), overtime_terms=30, eps=1e-6)

    big = [SeasonModel(params, OnceRoundedMinutesKernel()).sample(schedule, rng) for _ in range(batch_size)]
    report["timing"] = time_batch(lik, SeasonBatch.from_outcomes(big, schedule))
    return report
