"""The calibration layer (D-107): identity at zero, floors preserved, MAP recovers an injected offset, Laplace and PSIS pieces."""
import numpy as np
import pytest
import torch

from fpp.calibration import (COMPONENTS, Correction, Design, draw_posterior, fit_map, gaussian_logpdf, laplace_covariance,
                             log_likelihood, mixture_log_scores, psis_khat)
from fpp.experiments.report import row_parameters
from fpp.model.likelihood import SeasonModel
from fpp.model.production import COUNT_NAMES, MAKE_ATTEMPTS, RATE_NAMES
from fpp.model.torch_likelihood import DifferentiableSeasonLikelihood, SeasonBatch
from fpp.model.tower import PARAMETER_FLOORS, PROBABILITY_BOUNDS

D = torch.float64


def base_params(n: int, rng: np.random.Generator) -> dict:
    """Per-unit head parameters well inside their floors, with some spread."""
    u = lambda lo, hi, *shape: torch.as_tensor(rng.uniform(lo, hi, size=(n, *shape) if shape else n), dtype=D)  # noqa: E731
    return {"pi_g": u(0.6, 0.98), "alpha_g": u(2, 8), "beta_g": u(0.5, 3), "alpha_j": u(0.5, 4), "beta_j": u(0.5, 4),
            "nu": u(0.02, 0.06), "alpha_m": u(1.5, 4), "beta_m": u(1.5, 4), "pi_m": u(0.005, 0.03),
            "rates": u(0.5, 6, len(RATE_NAMES)), "dispersions": u(3, 20, len(RATE_NAMES)),
            "makes_alpha": u(5, 15, len(MAKE_ATTEMPTS)), "makes_beta": u(5, 15, len(MAKE_ATTEMPTS))}


def design_for(n: int, rng: np.random.Generator, league=True):
    no_hist = rng.random(n) < 0.5
    leagues = rng.choice(["intl", "hs_or_none", "juco"], size=n)
    des = Design.from_units(no_hist, leagues, use_history=True, use_league=league)
    return des, des.matrix(no_hist, leagues)


def test_identity_at_zero_and_floors_under_random_offsets():
    rng = np.random.default_rng(1)
    params = base_params(50, rng)
    des, z = design_for(50, rng)
    corr = Correction(des)
    same = corr.transform(params, torch.zeros(corr.size, dtype=D), z)
    for k, v in params.items():
        assert torch.allclose(same[k], v, atol=1e-12), k
    phi = torch.as_tensor(rng.normal(0, 1.5, corr.size), dtype=D)
    out = corr.transform(params, phi, z)
    lo, hi = PROBABILITY_BOUNDS
    assert bool((out["pi_g"] >= lo).all() and (out["pi_g"] <= hi).all())
    for k, f in PARAMETER_FLOORS.items():
        assert bool((out[k] > f).all()), k
    # Sizes agree across the layout, the prior and the description.
    assert corr.size == len(corr.prior_scales()) == len(corr.describe(np.zeros(corr.size)))
    assert torch.isfinite(corr.log_prior(phi))
    # A component left out passes through untouched.
    part_only = Correction(des, components=("part",))
    out2 = part_only.transform(params, torch.full((part_only.size,), 0.7, dtype=D), z)
    assert torch.allclose(out2["alpha_g"], params["alpha_g"]) and not torch.allclose(out2["pi_g"], params["pi_g"])


def synthetic_batch(params: dict, corr: Correction, phi_true: torch.Tensor, z: torch.Tensor, rng: np.random.Generator):
    """Seasons drawn from the corrected parameters under the reference sampler."""
    truth = {k: v.numpy() for k, v in corr.transform(params, phi_true, z).items()}
    n = len(truth["pi_g"])
    games, starts, minutes, counts, sched = [], [], [], [], []
    for i in range(n):
        s = int(rng.integers(28, 34))
        o = SeasonModel(row_parameters(truth, i)).sample(s, rng)
        games.append(o.games); starts.append(o.starts); minutes.append(o.recorded_minutes); sched.append(s)
        counts.append([o.counts[c] for c in COUNT_NAMES])
    return SeasonBatch(torch.tensor(games), torch.tensor(starts), torch.tensor(minutes, dtype=D), torch.tensor(counts), torch.tensor(sched), None, None)


def test_map_recovers_injected_offsets_and_laplace_is_proper():
    rng = np.random.default_rng(3)
    n = 240
    params = base_params(n, rng)
    des, z = design_for(n, rng, league=False)          # intercept + no-history
    corr = Correction(des, components=("part", "games"))
    phi_true = torch.zeros(corr.size, dtype=D)
    a, _ = corr.layout["part"]["mean"]
    phi_true[a] = 0.6                                   # participation intercept: logit +0.6
    g, _ = corr.layout["games"]["mean"]
    phi_true[g] = -0.5                                  # games mean: logit −0.5
    batch = synthetic_batch(params, corr, phi_true, z, rng)
    lik = DifferentiableSeasonLikelihood(None, quadrature_nodes=16)
    phi_map, info = fit_map(lik, batch, params, corr, z, max_iter=60)
    assert info["final"] < info["start"] and info["grad_norm"] < 1e-3
    # Average corrections over the units (intercept plus the no-history share times its coefficient). Participation is
    # weakly informed at this n (posterior sd about 0.2, prior sd 0.25): require the direction and the shrunk magnitude;
    # games, with about thirty trials per unit, is recovered closely.
    share = float(z[:, 1].mean())
    part = float(phi_map[a] + share * phi_map[a + 1])
    games = float(phi_map[g] + share * phi_map[g + 1])
    assert 0.05 < part < 0.9, part
    assert abs(games + 0.5) < 0.2, games
    cov, hinfo = laplace_covariance(lik, batch, params, corr, phi_map, z)
    assert hinfo["min_eigenvalue"] > 0
    sd = torch.sqrt(torch.diagonal(cov))
    assert 0.02 < float(sd[g]) < 0.3
    draws = draw_posterior(phi_map, cov, 8, seed=0)
    assert draws.shape == (8, corr.size)
    # The mixture over identical draws equals the point score; a real mixture is a proper log score.
    point = log_likelihood(lik, batch, params, corr, phi_map, z)
    mix, per = mixture_log_scores(lik, batch, params, corr, phi_map[None, :].repeat(3, 1), z)
    assert torch.allclose(mix, point, atol=1e-9)
    mix2, _ = mixture_log_scores(lik, batch, params, corr, draws, z)
    assert torch.isfinite(mix2).all()


def test_gaussian_logpdf_matches_scipy():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(5)
    a = rng.normal(size=(3, 3))
    cov = torch.as_tensor(a @ a.T + np.eye(3), dtype=D)
    mean = torch.as_tensor(rng.normal(size=3), dtype=D)
    x = torch.as_tensor(rng.normal(size=(4, 3)), dtype=D)
    ours = gaussian_logpdf(x, mean, cov).numpy()
    ref = multivariate_normal(mean.numpy(), cov.numpy()).logpdf(x.numpy())
    assert np.allclose(ours, ref, atol=1e-9)


def test_psis_khat_separates_good_and_bad_proposals():
    rng = np.random.default_rng(11)
    good = psis_khat(rng.normal(0, 0.1, 2000))          # near-uniform weights
    heavy = psis_khat(rng.exponential(1.0, 2000))       # Pareto(1) weights: k = 1
    degenerate = psis_khat(rng.standard_t(1, 2000) * 2.0)   # a few weights carry everything
    assert good["khat"] < 0.5 and good["ess"] > 1500
    assert 0.8 < heavy["khat"] < 1.3 and heavy["ess"] < 200
    assert degenerate["khat"] > 0.7                      # inf when no tail can be fitted
