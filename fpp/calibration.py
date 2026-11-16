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
import torch.nn.functional as Fn

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


@dataclass
class HierarchicalCorrection(Correction):
    """B1 of the Bayesian follow-up (D-114): the league effects of every mean/rate coordinate are partially pooled with a
    learned scale. In the mean block the league columns of the design hold standard-normal z's and the effect is
    b_{ℓk} = τ_k (z_{ℓk} − Σ_r ω_r z_{rk}) with fixed centering weights ω (the calibration cohort's league shares), so the
    intercept keeps its meaning; τ_k ~ HalfNormal(tau_scale) per mean coordinate, parameterized as log τ_k and appended after
    the fixed layout (non-centred, so a small τ is not a funnel for MAP or Laplace). Intercept, history and dispersion offsets
    keep their fixed-scale priors. A unit whose league level was unseen in calibration gets intercept and history terms only;
    callers check that none exists (the prior-league categories are fixed across seasons)."""
    omega: np.ndarray | None = None
    tau_scale: float = PRIOR_SCALES["league"]
    base_size: int = 0
    n_mean: int = 0
    tau_offset: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.design.levels:
            raise ValueError("the hierarchical correction needs league levels in the design")
        self.base_size = self.size
        k = 0
        for name in self.components:
            self.tau_offset[name] = k
            k += COMPONENTS[name][2]
        self.n_mean = k
        self.size = self.base_size + k
        L = len(self.design.levels)
        if self.omega is None:
            self.omega = np.full(L, 1.0 / L)
        self._omega = torch.as_tensor(np.asarray(self.omega, dtype=float), dtype=DTYPE)
        self._n_fixed_cols = len(self.design.columns) - L

    def tau(self, phi: torch.Tensor) -> torch.Tensor:
        return torch.exp(phi[self.base_size:])

    def prior_scales(self) -> torch.Tensor:
        """Scales of the Gaussian part only (the fixed layout): league z's have unit scale; τ's prior is separate."""
        L = len(self.design.levels)
        col_scales = torch.cat([self.design.scales[:self._n_fixed_cols], torch.ones(L, dtype=DTYPE)])
        parts = []
        for name in self.components:
            _, _, ncol, has_disp = COMPONENTS[name]
            parts.append(col_scales.repeat(ncol))
            if has_disp:
                parts.append(torch.full((ncol,), PRIOR_SCALES["dispersion"], dtype=DTYPE))
        s = torch.cat(parts)
        assert len(s) == self.base_size
        return s

    def log_prior(self, phi: torch.Tensor) -> torch.Tensor:
        s = self.prior_scales()
        base = phi[:self.base_size]
        gauss = -0.5 * ((base / s) ** 2).sum() - torch.log(s).sum() - 0.5 * self.base_size * math.log(2 * math.pi)
        log_tau = phi[self.base_size:]
        tau = torch.exp(log_tau)
        half_normal = (math.log(2.0) - 0.5 * math.log(2 * math.pi * self.tau_scale ** 2) - 0.5 * (tau / self.tau_scale) ** 2).sum()
        return gauss + half_normal + log_tau.sum()          # + log τ: the Jacobian of τ → log τ

    def league_effects(self, name: str, phi: torch.Tensor) -> torch.Tensor:
        """[ncol, L] centred league effects b_{ℓk} = τ_k (z_{ℓk} − ω·z_k) of one component."""
        _, _, ncol, _ = COMPONENTS[name]
        a, b = self.layout[name]["mean"]
        beta = phi[a:b].reshape(ncol, -1)
        zed = beta[:, self._n_fixed_cols:]
        k0 = self.tau_offset[name]
        tau = self.tau(phi)[k0:k0 + ncol]
        return tau[:, None] * (zed - (zed * self._omega[None, :]).sum(1, keepdim=True))

    def offsets(self, name: str, phi: torch.Tensor, z: torch.Tensor):
        _, _, ncol, has_disp = COMPONENTS[name]
        a, b = self.layout[name]["mean"]
        beta = phi[a:b].reshape(ncol, -1)
        beta_mod = torch.cat([beta[:, :self._n_fixed_cols], self.league_effects(name, phi)], dim=1)
        delta = z @ beta_mod.T
        d = None
        if has_disp:
            c, e = self.layout[name]["disp"]
            d = phi[c:e]
        return delta, d

    def describe(self, phi: np.ndarray, sd: np.ndarray | None = None) -> list[dict]:
        """Fixed terms as in ``Correction`` (league rows report the standard-normal z), then τ per coordinate and the implied
        centred league effects at the point estimate."""
        rows = super().describe(phi[:self.base_size], None if sd is None else sd[:self.base_size])
        for r in rows:
            if r["term"].startswith("league="):
                r["term"] = r["term"] + " (z)"
        phi_t = torch.as_tensor(phi, dtype=DTYPE)
        tau = self.tau(phi_t).numpy()
        for name in self.components:
            _, _, ncol, _ = COMPONENTS[name]
            labels = list(RATE_NAMES) if name == "rates" else list(MAKE_ATTEMPTS) if name == "makes" else [""]
            k0 = self.tau_offset[name]
            eff = self.league_effects(name, phi_t).detach().numpy()
            for j in range(ncol):
                idx = self.base_size + k0 + j
                rows.append({"component": name, "column": labels[j], "term": "tau (league scale)", "value": float(tau[k0 + j]),
                             "sd": None if sd is None else float(tau[k0 + j] * sd[idx])})        # delta method on log τ
                for i, lv in enumerate(self.design.levels):
                    rows.append({"component": name, "column": labels[j], "term": f"league={lv} (effect)", "value": float(eff[j, i]), "sd": None})
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


def _gpd_fit(x: np.ndarray) -> tuple[float, float]:
    """Generalized-Pareto (k, σ) for positive exceedances x (Zhang & Stephens 2009 estimator as in loo's gpdfit, with the
    weakly informative adjustment of k toward 0.5)."""
    x = np.sort(np.asarray(x, dtype=float))
    n_x = len(x)
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
    sigma = -khat / theta_hat
    khat = (n_x * khat + 10 * 0.5) / (n_x + 10)        # the loo package's weakly informative adjustment toward 0.5
    return khat, float(sigma)


def psis_threshold(n_draws: int) -> float:
    """loo's sample-size-dependent Pareto-k threshold, min(1 − 1/log10 S, 0.7): 0.5 at 100 draws, 0.67 at 1,000."""
    return float(min(1 - 1 / math.log10(max(n_draws, 11)), 0.7))


def psis_khat(log_weights: np.ndarray) -> dict:
    """Pareto-k diagnostic of importance weights (generalized-Pareto fit to the largest 20%) and the effective sample size of
    the self-normalized weights. Compare k̂ with ``psis_threshold(S)`` (0.5 at S = 100 draws): above it the proposal misses
    posterior mass and the unweighted mixture under-represents it; below it the smoothed weights are reliable, which is not a
    certificate that the unweighted proposal mixture represents the posterior."""
    return {k: v for k, v in psis_smooth(log_weights).items() if k in ("khat", "ess", "tail")}


def psis_smooth(log_weights: np.ndarray) -> dict:
    """Pareto-smoothed importance weights (Vehtari, Simpson, Gelman, Yao & Gabry): the largest M = min(0.2 S, 3 √S) weights
    are replaced by the expected order statistics of the fitted generalized Pareto, capped at the largest raw weight, then
    self-normalized. Returns ``khat``, ``ess`` (raw weights), ``ess_smoothed``, ``tail`` and ``log_weights`` (normalized
    smoothed log weights in the input order; non-finite inputs get −inf)."""
    lw_in = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(lw_in)
    lw = lw_in[finite] - lw_in[finite].max()
    w = np.exp(lw)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    n = len(w)
    out = np.full(len(lw_in), -np.inf)
    m = max(int(np.ceil(min(0.2 * n, 3 * np.sqrt(n)))), 5)
    order = np.argsort(w)
    ws = w[order]
    if n < m + 5:
        out[finite] = lw - np.log(np.exp(lw).sum())
        return {"khat": float("inf"), "ess": ess, "ess_smoothed": ess, "tail": 0, "log_weights": out}
    cutoff = ws[-(m + 1)]                               # largest non-tail weight; exceedances are positive
    x = ws[-m:] - cutoff
    keep = x > 0
    n_x = int(keep.sum())
    if n_x < 5:                                         # a handful of weights carry everything: no usable tail (loo: Inf)
        out[finite] = lw - np.log(np.exp(lw).sum())
        return {"khat": float("inf"), "ess": ess, "ess_smoothed": ess, "tail": n_x, "log_weights": out}
    khat, sigma = _gpd_fit(x[keep])
    p = (np.arange(1, n_x + 1) - 0.5) / n_x
    if abs(khat) < 1e-12:
        q = cutoff - sigma * np.log1p(-p)
    else:
        q = cutoff + sigma / khat * ((1 - p) ** (-khat) - 1)
    smoothed = ws.copy()
    tail_idx = np.arange(n - m, n)[keep]
    smoothed[tail_idx] = np.minimum(np.sort(q), ws[-1])  # expected order statistics, capped at the largest raw weight
    back = np.empty_like(smoothed)
    back[order] = smoothed
    back = back / back.sum()
    out[finite] = np.log(back)
    return {"khat": khat, "ess": ess, "ess_smoothed": float(1.0 / np.sum(back ** 2)), "tail": n_x, "log_weights": out}


def weighted_mixture(per_draw: torch.Tensor, log_weights: np.ndarray) -> torch.Tensor:
    """log Σ_r w̃_r p(y_i | φ_r) from per-draw log scores [R, N] and normalized log weights [R] (the importance-sampling
    corrected predictive; with uniform weights it is the plain mixture)."""
    lw = torch.as_tensor(log_weights, dtype=per_draw.dtype)
    return torch.logsumexp(per_draw + lw[:, None], dim=0)


def block_mixtures(per_draw: torch.Tensor, blocks: int) -> torch.Tensor:
    """[K, N] mixtures over K consecutive disjoint blocks of draws: the spread of a cohort mean across blocks, divided by √K,
    is the Monte Carlo standard error of the full-draw mixture's cohort mean (separate from player-sampling uncertainty)."""
    r = per_draw.shape[0]
    size = r // blocks
    return torch.stack([torch.logsumexp(per_draw[k * size:(k + 1) * size], dim=0) - math.log(size) for k in range(blocks)])


class MakesFactorLikelihood:
    """The makes | attempts factor of the season likelihood alone, with the ``log_prob(batch, params=...)`` interface of
    ``DifferentiableSeasonLikelihood``. For a correction that changes only the shooting parameters the season law factorizes
    exactly, log p_φ(Y_i) = c_i + s_i(φ) with s_i the beta-binomial makes terms and c_i = log p_raw(Y_i) − s_i(0) fixed, so
    fitting, curvature, PSIS and mixtures need only beta-binomial evaluations (no latent-minute quadrature or overtime sum);
    ``shooting_constant`` returns c_i from one pass of the full likelihood. Rows with zero attempts contribute zero."""

    @staticmethod
    def log_prob(batch, params: dict | None = None, **_):
        from .model.production import COUNT_NAMES
        from .model.torch_likelihood import beta_binomial_logpmf
        ci = {n: i for i, n in enumerate(COUNT_NAMES)}
        counts = batch.counts
        total = torch.zeros(len(counts), dtype=DTYPE)
        for h, (make, attempt) in enumerate(MAKE_ATTEMPTS.items()):
            total = total + beta_binomial_logpmf(counts[:, ci[make]], counts[:, ci[attempt]],
                                                 params["makes_alpha"][:, h].to(DTYPE), params["makes_beta"][:, h].to(DTYPE))
        return total, None


def shooting_constant(lik_full, batch, params: dict, correction, z) -> torch.Tensor:
    """c_i = log p_raw(Y_i) − s_i(0): the non-shooting part of each unit's season log probability under the frozen forecast."""
    zero = torch.zeros(correction.size, dtype=DTYPE)
    with torch.no_grad():
        full = log_likelihood(lik_full, batch, params, correction, zero, z)
        makes = log_likelihood(MakesFactorLikelihood(), batch, params, correction, zero, z)
    return full - makes


# -- B2: Bayesian increments to the frozen linear head (D-114; the specification's §4) ----------------------------------------
HEAD_OUTPUTS = ("pi_g", "alpha_g", "beta_g", "alpha_j", "beta_j", "nu", "alpha_m", "beta_m", "pi_m",
                *(f"rate {n}" for n in RATE_NAMES), *(f"dispersion {n}" for n in RATE_NAMES),
                *(f"makes_alpha {k}" for k in MAKE_ATTEMPTS), *(f"makes_beta {k}" for k in MAKE_ATTEMPTS))
HEAD_PRIOR = {"weight": 0.20, "bias": 0.20}      # ΔW_kr ~ N(0, weight² / P) on the standardized representation, Δb_k ~ N(0, bias²)
MAKES_ROWS = tuple(range(len(HEAD_OUTPUTS) - 2 * len(MAKE_ATTEMPTS), len(HEAD_OUTPUTS)))   # the six shooting rows (α then β)


def head_parameters(eta: torch.Tensor) -> dict:
    """The registered head's valid-parameter map T: raw linear outputs [N, 35] → the season family, exactly as
    ``OutcomeHeads.forward`` (bounded logistic links for the two probabilities, floor + softplus for everything else)."""
    k, r, m = 9, len(RATE_NAMES), len(MAKE_ATTEMPTS)
    if eta.shape[-1] != k + 2 * r + 2 * m:
        raise ValueError(f"expected {k + 2 * r + 2 * m} head outputs, got {eta.shape[-1]}")
    eta = eta.to(DTYPE)
    fl = PARAMETER_FLOORS
    lo, hi = PROBABILITY_BOUNDS
    prob = lambda x: lo + (hi - lo) * torch.sigmoid(x)     # noqa: E731
    pos = lambda x, name: fl[name] + Fn.softplus(x)        # noqa: E731
    return {"pi_g": prob(eta[:, 0]), "alpha_g": pos(eta[:, 1], "alpha_g"), "beta_g": pos(eta[:, 2], "beta_g"),
            "alpha_j": pos(eta[:, 3], "alpha_j"), "beta_j": pos(eta[:, 4], "beta_j"), "nu": pos(eta[:, 5], "nu"),
            "alpha_m": pos(eta[:, 6], "alpha_m"), "beta_m": pos(eta[:, 7], "beta_m"), "pi_m": prob(eta[:, 8]),
            "rates": pos(eta[:, k:k + r], "rates"), "dispersions": pos(eta[:, k + r:k + 2 * r], "dispersions"),
            "makes_alpha": pos(eta[:, k + 2 * r:k + 2 * r + m], "makes_alpha"), "makes_beta": pos(eta[:, k + 2 * r + m:], "makes_beta")}


@dataclass
class HeadCorrection:
    """B2 of the Bayesian follow-up (D-114; specification §4): a Gaussian increment to the frozen linear head,
    η_i = Ŵ u_i + b̂ + ΔW ũ_i + Δb, with ũ = (u − mean) / scale the representation standardized by fixed statistics and T
    (``head_parameters``) the registered parameter map; zero increments reproduce the neural forecasts exactly. φ holds, for
    each active output row k in ``rows`` (all 35 by default), its P weight increments followed by its bias increment; priors
    ΔW_kr ~ N(0, weight_scale² / P), Δb_k ~ N(0, bias_scale²). In the shared objective functions the ``z`` argument carries the
    raw representation u [N, P]; ``transform`` ignores the cached parameters (callers verify that T(Ŵu + b̂) reproduces them).
    Shares the interface of ``Correction`` that ``log_likelihood``/``fit_map``/``mixture_log_scores`` use."""
    W: torch.Tensor
    b: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    rows: tuple | None = None
    weight_scale: float = HEAD_PRIOR["weight"]
    bias_scale: float = HEAD_PRIOR["bias"]
    size: int = 0

    def __post_init__(self):
        self.W, self.b = torch.as_tensor(self.W, dtype=DTYPE), torch.as_tensor(self.b, dtype=DTYPE)
        self.mean, self.scale = torch.as_tensor(self.mean, dtype=DTYPE), torch.as_tensor(self.scale, dtype=DTYPE)
        K, P = self.W.shape
        if K != len(HEAD_OUTPUTS) or self.b.shape != (K,) or self.mean.shape != (P,) or self.scale.shape != (P,):
            raise ValueError(f"head shapes do not match the registered linear head: W {tuple(self.W.shape)}, b {tuple(self.b.shape)}, "
                             f"mean {tuple(self.mean.shape)}, scale {tuple(self.scale.shape)}")
        if bool((self.scale <= 0).any()):
            raise ValueError("standardization scales must be positive")
        self.rows = tuple(range(K)) if self.rows is None else tuple(int(r) for r in self.rows)
        if len(set(self.rows)) != len(self.rows) or any(r < 0 or r >= K for r in self.rows):
            raise ValueError(f"rows must be distinct indices below {K}")
        self.size = len(self.rows) * (P + 1)
        sel = torch.zeros(len(self.rows), K, dtype=DTYPE)
        for i, r in enumerate(self.rows):
            sel[i, r] = 1.0
        self._select = sel

    @property
    def width(self) -> int:
        return int(self.W.shape[1])

    def features(self, u) -> torch.Tensor:
        """[N, P + 1]: the standardized representation and a constant one for the bias increment."""
        u = torch.as_tensor(u, dtype=DTYPE)
        return torch.cat([(u - self.mean) / self.scale, torch.ones(len(u), 1, dtype=DTYPE)], dim=1)

    def increments(self, phi: torch.Tensor) -> torch.Tensor:
        """[R, P + 1]: per active row, the weight increments then the bias increment."""
        return phi.reshape(len(self.rows), self.width + 1)

    def eta(self, phi: torch.Tensor, u) -> torch.Tensor:
        """Raw head outputs [N, 35] under the increment φ."""
        u = torch.as_tensor(u, dtype=DTYPE)
        return u @ self.W.T + self.b + (self.features(u) @ self.increments(phi).T) @ self._select

    def prior_scales(self) -> torch.Tensor:
        row = torch.cat([torch.full((self.width,), self.weight_scale / math.sqrt(self.width), dtype=DTYPE),
                         torch.tensor([self.bias_scale], dtype=DTYPE)])
        return row.repeat(len(self.rows))

    def log_prior(self, phi: torch.Tensor) -> torch.Tensor:
        s = self.prior_scales()
        return -0.5 * ((phi / s) ** 2).sum() - torch.log(s).sum() - 0.5 * self.size * math.log(2 * math.pi)

    def transform(self, params: dict, phi: torch.Tensor, z: torch.Tensor) -> dict:
        return head_parameters(self.eta(phi, z))

    def describe(self, phi: np.ndarray, sd: np.ndarray | None = None) -> list[dict]:
        """One row per active output: bias increment (± sd), the Euclidean norm of its weight increments, the largest weight
        increment and how many weights lie beyond two posterior sd."""
        inc = np.asarray(phi, dtype=float).reshape(len(self.rows), -1)
        s = None if sd is None else np.asarray(sd, dtype=float).reshape(len(self.rows), -1)
        rows = []
        for i, k in enumerate(self.rows):
            w = inc[i, :-1]
            rows.append({"output": HEAD_OUTPUTS[k], "row": int(k), "bias": float(inc[i, -1]), "bias_sd": None if s is None else float(s[i, -1]),
                         "weight_norm": float(np.linalg.norm(w)), "largest_weight": float(w[int(np.argmax(np.abs(w)))]) if len(w) else 0.0,
                         "weights_beyond_2sd": None if s is None else int(np.sum(np.abs(w) > 2 * s[i, :-1]))})
        return rows


def eta_gradient(lik, batch, eta: torch.Tensor, mask: torch.Tensor | None = None, chunk: int = 512) -> torch.Tensor:
    """∇_η Σ_i ℓ_i(η_i) at η [N, 35] → [N, 35]. Units are independent, so the summed gradient is the per-unit gradient;
    masked (non-finite) units contribute zero rows."""
    from .model.torch_likelihood import SeasonBatch
    n = len(batch.games)
    out = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        sub = SeasonBatch(batch.games[s:e], batch.starts[s:e], batch.recorded_minutes[s:e], batch.counts[s:e], batch.schedule[s:e], None, None)
        x = eta[s:e].detach().clone().requires_grad_(True)
        lp, _ = lik.log_prob(sub, params=head_parameters(x))
        if mask is not None:
            lp = torch.where(mask[s:e], lp, torch.zeros_like(lp))
        g = torch.autograd.grad(lp.sum(), x)[0].detach()
        if mask is not None:
            g = torch.where(mask[s:e, None], g, torch.zeros_like(g))
        out.append(torch.nan_to_num(g, nan=0.0))
    return torch.cat(out)


def head_laplace_covariance(lik, batch, correction: HeadCorrection, phi_map: torch.Tensor, u, mask=None, step: float = 1e-4,
                            jitter: float = 1e-8, verbose: bool = False) -> tuple[torch.Tensor, dict]:
    """Laplace covariance of a ``HeadCorrection`` by the chain rule. The negative log posterior's Hessian in φ is
    Σ_i x_i x_iᵀ ⊗ (−H_{η,i}) over the active rows plus the prior precision, where x_i = (ũ_i, 1) and H_{η,i} is unit i's
    Hessian of the log likelihood in the head's output space, taken by central differences of the exact gradient — 2 × 35
    batched gradient passes instead of 2 × |φ| (1,155 for the full head). Same positive-definiteness handling and return
    shape as ``laplace_covariance``; ``jitter_added`` reports any damping."""
    eta = correction.eta(phi_map, u).detach()
    n, K = eta.shape
    H = torch.zeros(n, K, K, dtype=DTYPE)
    for j in range(K):
        d = torch.zeros(K, dtype=DTYPE)
        d[j] = step
        H[:, :, j] = (eta_gradient(lik, batch, eta + d, mask) - eta_gradient(lik, batch, eta - d, mask)) / (2 * step)
        if verbose and j % 5 == 0:
            print(f"  head Hessian: output {j + 1}/{K}", flush=True)
    H = 0.5 * (H + H.transpose(1, 2))
    rows = list(correction.rows)
    Hr = H[:, rows][:, :, rows]                                   # [N, R, R]
    x = correction.features(u)                                    # [N, P + 1]
    R, Q = len(rows), x.shape[1]
    M = torch.zeros(R, Q, R, Q, dtype=DTYPE)
    for s in range(0, n, 256):
        e = min(n, s + 256)
        M += torch.einsum("ikl,ia,ib->kalb", Hr[s:e], x[s:e], x[s:e])
    h = -M.reshape(R * Q, R * Q) + torch.diag(1.0 / correction.prior_scales() ** 2)
    h = 0.5 * (h + h.T)
    eig = torch.linalg.eigvalsh(h)
    added = 0.0
    if float(eig.min()) <= 0:
        added = float(-eig.min()) + jitter
        h = h + added * torch.eye(len(h), dtype=DTYPE)
    cov = torch.linalg.inv(h)
    cov = 0.5 * (cov + cov.T)
    return cov, {"min_eigenvalue": float(eig.min()), "max_eigenvalue": float(eig.max()), "jitter_added": added}
