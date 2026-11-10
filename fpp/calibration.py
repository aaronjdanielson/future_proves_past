"""Selective calibration of a frozen model's season forecasts (D-107; the 5 November proposals).

A small parameter vector φ corrects the head parameters of a frozen fit through offsets on the natural scales of the
season family, in coordinates that preserve the registered floors (``PARAMETER_FLOORS``) and probability bounds, with
the identity at φ = 0. Each corrected mean receives a linear predictor over cutoff-available descriptors (an intercept, a
no-tracked-history flag and prior-league levels); each concentration or dispersion receives one global log offset.
Proper fixed-scale normal priors regularize everything; the point comparator is the MAP, the Bayesian comparator a
Laplace approximation at the MAP whose adequacy is checked by Pareto-smoothed importance weights against the exact
posterior. Predictive scores use the exact integrated season likelihood with the posterior mixture evaluated by
``logsumexp`` over draws. Nothing here touches the tower; the inputs are cached per-unit head parameters.

Families and coordinates (``T_φ``):

* probabilities (``pi_g``, ``pi_m``): logit offset, result kept inside ``PROBABILITY_BOUNDS``;
* beta shape pairs (games, starts, minute share, makes): excess shapes above their floors,
  ``s = α + β − f_α − f_β``, ``r = (α − f_α)/s``, ``r' = sigmoid(logit r + δ)``, ``s' = s·exp(d)``,
  ``α' = f_α + r's'``, ``β' = f_β + (1 − r')s'``;
* positive rates and dispersions: ``x' = f + (x − f)·exp(δ)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .model.production import MAKE_ATTEMPTS, RATE_NAMES
from .model.tower import PARAMETER_FLOORS, PROBABILITY_BOUNDS

DTYPE = torch.float64
PRIOR_SCALES = {"intercept": 0.25, "no_history": 0.25, "league": 0.20, "dispersion": 0.50}
# Component name → (kind, parameter keys, columns, whether a dispersion/concentration offset exists).
COMPONENTS = {
    "part": ("prob", ("pi_g",), 1, False),
    "games": ("beta", ("alpha_g", "beta_g"), 1, True),
    "starts": ("beta", ("alpha_j", "beta_j"), 1, True),
    "minutes": ("beta", ("alpha_m", "beta_m"), 1, True),
    "maxshare": ("prob", ("pi_m",), 1, False),
    "rates": ("positive", ("rates", "dispersions"), len(RATE_NAMES), True),
    "makes": ("beta", ("makes_alpha", "makes_beta"), len(MAKE_ATTEMPTS), True),
}
DEFAULT_COMPONENTS = ("part", "games", "starts", "minutes", "rates", "makes")   # pi_m fixed in the first version (proposal)
EPS = 1e-6


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(EPS, 1 - EPS)
    return torch.log(p) - torch.log1p(-p)


@dataclass
class Design:
    """Descriptor matrix for the mean offsets: intercept, no-history flag, one column per prior-league level."""
    columns: list
    levels: list

    @classmethod
    def from_units(cls, no_history: np.ndarray, league: np.ndarray, use_history: bool = True, use_league: bool = True) -> "Design":
        levels = sorted(set(str(x) for x in league)) if use_league else []
        cols = ["intercept"] + (["no_history"] if use_history else []) + [f"league={lv}" for lv in levels]
        return cls(cols, levels)

    def matrix(self, no_history: np.ndarray, league: np.ndarray) -> torch.Tensor:
        n = len(no_history)
        z = np.zeros((n, len(self.columns)))
        z[:, 0] = 1.0
        j = 1
        if "no_history" in self.columns:
            z[:, j] = np.asarray(no_history, dtype=float)
            j += 1
        pos = {lv: j + i for i, lv in enumerate(self.levels)}
        for i, lv in enumerate(league):
            k = pos.get(str(lv))
            if k is not None:                       # an unseen level keeps the intercept and history terms only
                z[i, k] = 1.0
        return torch.as_tensor(z, dtype=DTYPE)

    @property
    def scales(self) -> torch.Tensor:
        s = [PRIOR_SCALES["intercept"]] + ([PRIOR_SCALES["no_history"]] if "no_history" in self.columns else []) + [PRIOR_SCALES["league"]] * len(self.levels)
        return torch.tensor(s, dtype=DTYPE)


@dataclass
class Correction:
    """Parameter layout of ``T_φ`` for a set of components and a design; ``size`` is the length of φ."""
    design: Design
    components: tuple = DEFAULT_COMPONENTS
    layout: dict = field(default_factory=dict)
    size: int = 0

    def __post_init__(self):
        unknown = [c for c in self.components if c not in COMPONENTS]
        if unknown:
            raise ValueError(f"unknown components {unknown}; choose from {sorted(COMPONENTS)}")
        p = len(self.design.columns)
        k = 0
        for name in self.components:
            _, _, ncol, has_disp = COMPONENTS[name]
            self.layout[name] = {"mean": (k, k + ncol * p), "disp": None}
            k += ncol * p
            if has_disp:
                self.layout[name]["disp"] = (k, k + ncol)
                k += ncol
        self.size = k

    # -- pieces ---------------------------------------------------------------------------------------------------------
    def prior_scales(self) -> torch.Tensor:
        p = len(self.design.columns)
        parts = []
        for name in self.components:
            _, _, ncol, has_disp = COMPONENTS[name]
            parts.append(self.design.scales.repeat(ncol))
            if has_disp:
                parts.append(torch.full((ncol,), PRIOR_SCALES["dispersion"], dtype=DTYPE))
        s = torch.cat(parts)
        assert len(s) == self.size and p == len(self.design.scales)
        return s

    def log_prior(self, phi: torch.Tensor) -> torch.Tensor:
        s = self.prior_scales()
        return -0.5 * ((phi / s) ** 2).sum() - torch.log(s).sum() - 0.5 * self.size * math.log(2 * math.pi)

    def offsets(self, name: str, phi: torch.Tensor, z: torch.Tensor):
        """(δ [N, ncol], d [ncol] or None) for one component."""
        _, _, ncol, has_disp = COMPONENTS[name]
        a, b = self.layout[name]["mean"]
        beta = phi[a:b].reshape(ncol, -1)              # [ncol, P]
        delta = z @ beta.T                              # [N, ncol]
        d = None
        if has_disp:
            c, e = self.layout[name]["disp"]
            d = phi[c:e]
        return delta, d

    def transform(self, params: dict, phi: torch.Tensor, z: torch.Tensor) -> dict:
        """Corrected head parameters; every key of ``params`` is returned (untouched families pass through)."""
        out = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in params.items()}
        lo, hi = PROBABILITY_BOUNDS
        for name in self.components:
            kind, keys, ncol, _ = COMPONENTS[name]
            delta, d = self.offsets(name, phi, z)
            if kind == "prob":
                p = out[keys[0]]
                out[keys[0]] = torch.sigmoid(_logit(p) + delta[:, 0]).clamp(lo, hi)
            elif kind == "positive":
                rate_key, disp_key = keys
                fr, fd = PARAMETER_FLOORS[rate_key], PARAMETER_FLOORS[disp_key]
                out[rate_key] = fr + (out[rate_key] - fr).clamp(min=EPS) * torch.exp(delta)
                out[disp_key] = fd + (out[disp_key] - fd).clamp(min=EPS) * torch.exp(d)[None, :]
            else:   # beta pair(s)
                ka, kb = keys
                fa, fb = PARAMETER_FLOORS[ka], PARAMETER_FLOORS[kb]
                alpha, beta = out[ka], out[kb]
                squeeze = alpha.dim() == 1
                if squeeze:
                    alpha, beta, delta_ = alpha[:, None], beta[:, None], delta
                else:
                    delta_ = delta
                s = (alpha + beta - fa - fb).clamp(min=EPS)
                r = ((alpha - fa) / s).clamp(EPS, 1 - EPS)
                r2 = torch.sigmoid(_logit(r) + delta_)
                s2 = s * torch.exp(d)[None, :]
                a2, b2 = fa + r2 * s2, fb + (1 - r2) * s2
                out[ka], out[kb] = (a2[:, 0], b2[:, 0]) if squeeze else (a2, b2)
        return out

    def describe(self, phi: np.ndarray, sd: np.ndarray | None = None) -> list[dict]:
        """One row per parameter: component, column (for rates/makes), descriptor or 'dispersion', value, sd."""
        rows = []
        p = len(self.design.columns)
        for name in self.components:
            kind, keys, ncol, has_disp = COMPONENTS[name]
            labels = list(RATE_NAMES) if name == "rates" else list(MAKE_ATTEMPTS) if name == "makes" else [""]
            a, _ = self.layout[name]["mean"]
            for j in range(ncol):
                for i, col in enumerate(self.design.columns):
                    k = a + j * p + i
                    rows.append({"component": name, "column": labels[j], "term": col, "value": float(phi[k]),
                                 "sd": None if sd is None else float(sd[k])})
            if has_disp:
                c, _ = self.layout[name]["disp"]
                for j in range(ncol):
                    rows.append({"component": name, "column": labels[j], "term": "log dispersion/concentration", "value": float(phi[c + j]),
                                 "sd": None if sd is None else float(sd[c + j])})
        return rows


# -- objective, MAP, Laplace, predictive ----------------------------------------------------------------------------------
def log_likelihood(lik, batch, params: dict, correction: Correction, phi: torch.Tensor, z: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Per-unit log probability [N] under the corrected parameters (exact integrated likelihood, overtime summed out)."""
    from .model.torch_likelihood import SeasonBatch
    n = len(batch.games)
    out = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        sub = SeasonBatch(batch.games[s:e], batch.starts[s:e], batch.recorded_minutes[s:e], batch.counts[s:e], batch.schedule[s:e], None, None)
        p = correction.transform({k: v[s:e] for k, v in params.items()}, phi, z[s:e])
        lp, _ = lik.log_prob(sub, params=p)
        out.append(lp)
    return torch.cat(out)


def negative_log_posterior(lik, batch, params, correction, phi, z, mask: torch.Tensor | None = None) -> torch.Tensor:
    lp = log_likelihood(lik, batch, params, correction, phi, z)
    if mask is not None:
        lp = lp[mask]
    return -(lp.sum() + correction.log_prior(phi))


def fit_map(lik, batch, params, correction, z, mask=None, max_iter: int = 200, tol: float = 1e-7, verbose: bool = False) -> tuple[torch.Tensor, dict]:
    """MAP by L-BFGS with strong-Wolfe line search from φ = 0; returns (φ, info)."""
    phi = torch.zeros(correction.size, dtype=DTYPE, requires_grad=True)
    opt = torch.optim.LBFGS([phi], lr=1.0, max_iter=max_iter, tolerance_grad=tol, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")
    trace = []

    def closure():
        opt.zero_grad()
        obj = negative_log_posterior(lik, batch, params, correction, phi, z, mask)
        obj.backward()
        trace.append(float(obj.detach()))
        if verbose and len(trace) % 10 == 1:
            print(f"  L-BFGS eval {len(trace)}: {trace[-1]:.4f}", flush=True)
        return obj

    opt.step(closure)
    with torch.no_grad():
        final = float(negative_log_posterior(lik, batch, params, correction, phi, z, mask))
        start = trace[0] if trace else final
    grad = torch.autograd.grad(negative_log_posterior(lik, batch, params, correction, phi, z, mask), phi)[0]
    return phi.detach(), {"evaluations": len(trace), "start": start, "final": final, "grad_norm": float(grad.norm())}


def laplace_covariance(lik, batch, params, correction, phi_map, z, mask=None, jitter: float = 1e-8, method: str = "fd",
                       step: float = 1e-4, verbose: bool = False) -> tuple[torch.Tensor, dict]:
    """Inverse Hessian of the negative log posterior at the MAP, made positive definite if needed. ``method="fd"`` takes
    central differences of the exact gradient (2·dim gradient passes; adequate for a Gaussian approximation);
    ``"autograd"`` is the exact double-backward Hessian (slower through the quadrature)."""
    f = lambda p: negative_log_posterior(lik, batch, params, correction, p, z, mask)   # noqa: E731
    if method == "autograd":
        h = torch.autograd.functional.hessian(f, phi_map.clone().requires_grad_(True))
    else:
        def grad_at(p):
            p = p.clone().requires_grad_(True)
            return torch.autograd.grad(f(p), p)[0].detach()
        dim = len(phi_map)
        h = torch.zeros(dim, dim, dtype=DTYPE)
        for j in range(dim):
            e = torch.zeros(dim, dtype=DTYPE)
            e[j] = step
            h[:, j] = (grad_at(phi_map + e) - grad_at(phi_map - e)) / (2 * step)
            if verbose and j % 20 == 0:
                print(f"  Hessian column {j + 1}/{dim}", flush=True)
    h = 0.5 * (h + h.T)
    eig = torch.linalg.eigvalsh(h)
    added = 0.0
    if float(eig.min()) <= 0:
        added = float(-eig.min()) + jitter
        h = h + added * torch.eye(len(h), dtype=DTYPE)
    cov = torch.linalg.inv(h)
    cov = 0.5 * (cov + cov.T)
    return cov, {"min_eigenvalue": float(eig.min()), "max_eigenvalue": float(eig.max()), "jitter_added": added}


def draw_posterior(phi_map: torch.Tensor, cov: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    chol = torch.linalg.cholesky(cov + 1e-12 * torch.eye(len(cov), dtype=DTYPE))
    eps = torch.randn(n, len(cov), generator=gen, dtype=DTYPE)
    return phi_map[None, :] + eps @ chol.T


def gaussian_logpdf(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    d = x - mean[None, :]
    chol = torch.linalg.cholesky(cov + 1e-12 * torch.eye(len(cov), dtype=DTYPE))
    sol = torch.cholesky_solve(d.T, chol).T
    logdet = 2 * torch.log(torch.diagonal(chol)).sum()
    return -0.5 * ((d * sol).sum(1) + logdet + len(cov) * math.log(2 * math.pi))


def mixture_log_scores(lik, batch, params, correction, draws: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(log mixture predictive [N] = logsumexp_r log p(y | φ_r) − log R, per-draw log scores [R, N])."""
    with torch.no_grad():
        per = torch.stack([log_likelihood(lik, batch, params, correction, phi, z) for phi in draws])
    return torch.logsumexp(per, dim=0) - math.log(len(draws)), per


def psis_khat(log_weights: np.ndarray) -> dict:
    """Pareto-k diagnostic of importance weights (generalized-Pareto fit to the largest 20%, Zhang & Stephens 2009 estimator)
    and the effective sample size of the self-normalized weights. k < 0.5 good, < 0.7 acceptable, above: the proposal
    distribution misses posterior mass."""
    lw = np.asarray(log_weights, dtype=float)
    lw = lw[np.isfinite(lw)]
    lw = lw - lw.max()
    w = np.sort(np.exp(lw))
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    n = len(w)
    m = max(int(np.ceil(min(0.2 * n, 3 * np.sqrt(n)))), 5)
    if n < m + 5:
        return {"khat": float("inf"), "ess": ess, "tail": 0}
    cutoff = w[-(m + 1)]                                # largest non-tail weight; exceedances are positive
    x = w[-m:] - cutoff
    x = x[x > 0]
    n_x = len(x)
    if n_x < 5:                                         # a handful of weights carry everything: no usable tail (loo: Inf)
        return {"khat": float("inf"), "ess": ess, "tail": int(n_x)}
    prior = 3.0
    mgrid = 30 + int(np.floor(np.sqrt(n_x)))
    jj = np.arange(1, mgrid + 1, dtype=float)
    xstar = x[max(int(np.floor(n_x / 4 + 0.5)) - 1, 0)]
    theta = 1.0 / x[-1] + (1 - np.sqrt(mgrid / (jj - 0.5))) / prior / xstar
    k_j = np.mean(np.log1p(-theta[:, None] * x[None, :]), axis=1)        # loo's lx: k(b) = mean log(1 − b x)
    l_j = n_x * (np.log(-theta / k_j) - k_j - 1)
    w_j = np.exp(l_j - l_j.max())
    w_j = w_j / w_j.sum()
    theta_hat = float(np.sum(theta * w_j))
    khat = float(np.mean(np.log1p(-theta_hat * x)))
    khat = (n_x * khat + 10 * 0.5) / (n_x + 10)        # the loo package's weakly informative adjustment toward 0.5
    return {"khat": khat, "ess": ess, "tail": int(n_x)}
