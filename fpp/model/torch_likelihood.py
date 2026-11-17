"""Differentiable (PyTorch) observed-data season likelihood — milestone 0.2, track 3.

Tensor implementation of the reference in ``distributions``, ``production``,
``observation`` and ``likelihood``. The SciPy reference remains the
independent oracle: agreement of observed-data log probabilities and of
autodiff gradients with central finite differences is required
(``fpp.experiments.gradcheck``). Same model, same recording kernel (reused
verbatim, it is data-only), same untruncated overtime; nothing renormalized.

Quadrature. Each recorded-minute cell is an interval of the latent fraction
u = M*/M̄. Interior cells use Gauss–Legendre nodes in u. A cell touching u=0
uses the substitution u = h t^{1/α}, which removes the u^{α-1} factor
exactly: ∫_0^h u^{α-1}(1-u)^{β-1}P(u)du = (h^α/α)∫_0^1 (1-u(t))^{β-1}P(u(t))dt.
A cell touching u=1 uses 1-u = (1-l) t^{1/β} symmetrically. Node positions
therefore depend smoothly on (α, β), and the whole integrand is smooth in t,
so fixed nodes give a differentiable, well-conditioned approximation whose
accuracy is validated against the adaptive reference rather than assumed.

Overtime. Terms o = 0, 1, ... are added for the whole batch until every
element satisfies the registered log-score bound and the absolute tail
check; exhaustion raises. ``overtime_terms`` fixes the count instead, which
the gradient check needs so that the function is smooth in the parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from torch import nn

from .distributions import SeasonParameters
from .likelihood import SeasonOutcome
from .observation import NumericalAccuracy, OnceRoundedMinutesKernel, poisson_log_tail_upper_bound
from .production import COUNT_NAMES, MAKE_ATTEMPTS, RATE_NAMES, ProductionParameters

DTYPE = torch.float64
NEG_INF = -math.inf
# Finite stand-in for log(0) inside reductions. logsumexp/logaddexp over
# all-(-inf) inputs have NaN gradients even when the result is later masked
# (0 * NaN = NaN in backward), so reductions see this sentinel and the
# -inf is restored on the output side, where a where() gives zero gradient.
LOG_ZERO = -1e300


def _finite_log(x):
    return torch.where(torch.isfinite(x), x, torch.full_like(x, LOG_ZERO))


class OvertimeBudgetError(RuntimeError):
    """The registered overtime budget could not be met; no truncated result is returned."""


# --------------------------------------------------------------------------- #
# Elementwise distributions (all broadcast; -inf off support; no clamping)
# --------------------------------------------------------------------------- #

def _lbeta(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def beta_binomial_logpmf(k, n, alpha, beta):
    k, n = k.to(DTYPE), n.to(DTYPE)
    valid = (k >= 0) & (k <= n)
    ks, ns = torch.where(valid, k, torch.zeros_like(k)), torch.where(valid, n, torch.zeros_like(n))
    logp = (torch.lgamma(ns + 1) - torch.lgamma(ks + 1) - torch.lgamma(ns - ks + 1)
            + _lbeta(ks + alpha, ns - ks + beta) - _lbeta(alpha, beta))
    return torch.where(valid, logp, torch.full_like(logp, NEG_INF))


def _safe_batch(batch, valid):
    """Replace invalid rows by zero-game rows (zero starts, minutes, counts, overtime) for the internal pieces."""
    if bool(valid.all()):
        return batch
    z = torch.zeros_like(batch.games)
    games = torch.where(valid, batch.games, z)
    starts = torch.where(valid, batch.starts, z)
    minutes = torch.where(valid, batch.recorded_minutes, torch.zeros_like(batch.recorded_minutes))
    counts = torch.where(valid[:, None], batch.counts, torch.zeros_like(batch.counts))
    overtime = None if batch.overtime is None else torch.where(valid, batch.overtime, torch.zeros_like(batch.overtime))
    known = None if batch.overtime_known is None else (batch.overtime_known & valid)
    return SeasonBatch(games, starts, minutes, counts, batch.schedule, overtime, known)


def _col(v):
    """[B] → [B, 1] so a per-row parameter broadcasts against [B, K] nodes; scalars pass through."""
    v = torch.as_tensor(v, dtype=DTYPE)
    return v[:, None] if v.dim() == 1 else v


def hurdle_beta_binomial_logpmf(games, schedule, pi, alpha, beta):
    """Hurdle at zero; positive support renormalized by a logsumexp over 1..S."""
    games, schedule = games.long(), schedule.long()
    s_max = max(int(schedule.max().item()), 1)
    ks = torch.arange(1, s_max + 1, dtype=DTYPE)
    logs = beta_binomial_logpmf(ks[None, :], schedule.to(DTYPE)[:, None], _col(alpha), _col(beta))
    lognorm = torch.logsumexp(logs, dim=1)
    idx = (games - 1).clamp(min=0, max=s_max - 1)
    positive = logs.gather(1, idx[:, None]).squeeze(1) - lognorm + torch.log(pi)
    zero = torch.log1p(-pi).expand_as(positive)
    out = torch.where(games == 0, zero, positive)
    empty = torch.where(games == 0, torch.zeros_like(out), torch.full_like(out, NEG_INF))
    out = torch.where(schedule == 0, empty, out)
    invalid = (games < 0) | (games > schedule)
    return torch.where(invalid, torch.full_like(out, NEG_INF), out)


def negative_binomial_logpmf(k, mean, dispersion):
    """NB(mean, dispersion) with variance mean + mean**2 / dispersion; mean > 0 at nodes."""
    k = k.to(DTYPE)
    mean = torch.as_tensor(mean, dtype=DTYPE)
    positive = mean > 0
    # Evaluate the formula at a safe mean so the unselected branch never
    # produces inf/NaN (whose backward would poison the selected branch).
    safe_mean = torch.where(positive, mean, torch.ones_like(mean))
    # k*(log m - log(m+d)) equals -k*log1p(d/m) but its derivative stays finite
    # when m is tiny (the log1p form differentiates through d/m**2, which
    # overflows to inf and yields 0*inf = NaN for k = 0).
    logp = (torch.lgamma(k + dispersion) - torch.lgamma(dispersion) - torch.lgamma(k + 1)
            - dispersion * torch.log1p(safe_mean / dispersion)
            + k * (torch.log(safe_mean) - torch.log(safe_mean + dispersion)))
    zero_mean = torch.where(k == 0, torch.zeros_like(logp), torch.full_like(logp, NEG_INF))
    logp = torch.where(positive, logp, zero_mean)
    return torch.where(k >= 0, logp, torch.full_like(logp, NEG_INF))


def truncated_negative_binomial_logpmf(k, mean, dispersion, cap, fast: bool = False):
    """NB restricted to 0..cap and renormalized there; cap is a data tensor [B].

    ``fast`` computes the same normalizer with the node-independent part of the mass function taken out of the
    [B, K, C] support tensor: log NB(c; mean, d) = A[b, c] − d·log1p(mean/d) + c·x[b, k] with A = lgamma(c + d) −
    lgamma(d) − lgamma(c + 1) and x = log(mean) − log(mean + d), so the sum over the support is one product and one
    logsumexp instead of the whole mass function on [B, K, C]. Used by the vectorized path for per-row dispersions and
    positive means; anything else takes the general route below."""
    cap_max = int(cap.max().item())
    support = torch.arange(cap_max + 1, dtype=DTYPE)                       # [C]
    shape = mean.shape                                                       # [B, K]
    dispersion = torch.as_tensor(dispersion, dtype=DTYPE)
    if fast and len(shape) == 2 and dispersion.shape == (shape[0], 1) and bool((mean > 0).all()):
        base = torch.lgamma(support[None, :] + dispersion) - torch.lgamma(dispersion) - torch.lgamma(support[None, :] + 1)   # [B, C]
        base = torch.where(support[None, :] <= cap.view(-1, 1), base, torch.full_like(base, NEG_INF))
        x = torch.log(mean) - torch.log(mean + dispersion)                                                                  # [B, K]
        lognorm = -dispersion * torch.log1p(mean / dispersion) + torch.logsumexp(base[:, None, :] + support * x[:, :, None], dim=-1)
        kk = k.to(DTYPE).expand(shape)
        logp = negative_binomial_logpmf(kk, mean, dispersion) - lognorm
        valid = (kk <= cap.view(-1, *([1] * (len(shape) - 1)))) & (kk >= 0)
        return torch.where(valid, logp, torch.full_like(logp, NEG_INF))
    disp3 = dispersion.unsqueeze(-1) if dispersion.dim() == len(shape) else dispersion   # per-row [B, 1] → [B, 1, 1]
    logs = negative_binomial_logpmf(support.view(*([1] * len(shape)), -1),  # [B, K, C]
                                    mean.unsqueeze(-1), disp3)
    within = support.view(*([1] * len(shape)), -1) <= cap.view(-1, *([1] * (len(shape) - 1)), 1)
    logs = torch.where(within, logs, torch.full_like(logs, NEG_INF))
    lognorm = torch.logsumexp(logs, dim=-1)
    kk = k.to(DTYPE).expand(shape)
    logp = negative_binomial_logpmf(kk, mean, dispersion) - lognorm
    valid = (kk <= cap.view(-1, *([1] * (len(shape) - 1)))) & (kk >= 0)
    return torch.where(valid, logp, torch.full_like(logp, NEG_INF))


def poisson_logpmf(k, mean):
    k = torch.as_tensor(k, dtype=DTYPE)
    mean = torch.as_tensor(mean, dtype=DTYPE)
    positive = mean > 0
    safe_mean = torch.where(positive, mean, torch.ones_like(mean))   # xlogy(0, 0) has a 0/0 backward
    logp = torch.xlogy(k, safe_mean) - safe_mean - torch.lgamma(k + 1)
    zero_mean = torch.where(k == 0, torch.zeros_like(logp), torch.full_like(logp, NEG_INF))
    return torch.where(positive, logp, zero_mean)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

def _logit(p):
    return math.log(p) - math.log1p(-p)


class InterceptOnlySeasonModel(nn.Module):
    """Unconstrained parameters with the same links as the paper's heads.

    Probabilities use the logistic link, positive parameters the exponential
    link (softplus would do; exp keeps finite-difference checks simplest).
    ``reference()`` returns the equal SciPy ``SeasonParameters`` so the two
    implementations can be compared on identical values.
    """

    def __init__(self, params: SeasonParameters | None = None):
        super().__init__()
        p = params or SeasonParameters()
        prod = p.production

        def raw(value):
            return nn.Parameter(torch.tensor(value, dtype=DTYPE))

        self.logit_pi_g = raw(_logit(p.participation_probability))
        self.log_alpha_g = raw(math.log(p.games_alpha))
        self.log_beta_g = raw(math.log(p.games_beta))
        self.log_alpha_j = raw(math.log(p.starts_alpha))
        self.log_beta_j = raw(math.log(p.starts_beta))
        self.log_nu = raw(math.log(p.overtime_rate_per_game))
        self.log_alpha_m = raw(math.log(p.minutes_alpha))
        self.log_beta_m = raw(math.log(p.minutes_beta))
        self.logit_pi_m = raw(_logit(p.minutes_max_probability))
        self.log_rates = nn.Parameter(torch.tensor([math.log(prod.rates[n]) for n in RATE_NAMES], dtype=DTYPE))
        self.log_dispersions = nn.Parameter(torch.tensor([math.log(prod.dispersions[n]) for n in RATE_NAMES], dtype=DTYPE))
        self.log_makes_alpha = nn.Parameter(torch.tensor([math.log(prod.makes_alpha[n]) for n in MAKE_ATTEMPTS], dtype=DTYPE))
        self.log_makes_beta = nn.Parameter(torch.tensor([math.log(prod.makes_beta[n]) for n in MAKE_ATTEMPTS], dtype=DTYPE))
        self.regulation_minutes = float(p.regulation_minutes)
        self.overtime_minutes = float(p.overtime_minutes)
        self.foul_limit_per_game = int(prod.foul_limit_per_game)

    def constrained(self) -> dict:
        return {
            "pi_g": torch.sigmoid(self.logit_pi_g), "alpha_g": self.log_alpha_g.exp(),
            "beta_g": self.log_beta_g.exp(), "alpha_j": self.log_alpha_j.exp(),
            "beta_j": self.log_beta_j.exp(), "nu": self.log_nu.exp(),
            "alpha_m": self.log_alpha_m.exp(), "beta_m": self.log_beta_m.exp(),
            "pi_m": torch.sigmoid(self.logit_pi_m),
            "rates": self.log_rates.exp(), "dispersions": self.log_dispersions.exp(),
            "makes_alpha": self.log_makes_alpha.exp(), "makes_beta": self.log_makes_beta.exp(),
        }

    def reference(self) -> SeasonParameters:
        c = {k: v.detach() for k, v in self.constrained().items()}
        prod = ProductionParameters(
            rates=dict(zip(RATE_NAMES, c["rates"].tolist())),
            dispersions=dict(zip(RATE_NAMES, c["dispersions"].tolist())),
            makes_alpha=dict(zip(MAKE_ATTEMPTS, c["makes_alpha"].tolist())),
            makes_beta=dict(zip(MAKE_ATTEMPTS, c["makes_beta"].tolist())),
            foul_limit_per_game=self.foul_limit_per_game)
        return SeasonParameters(
            participation_probability=float(c["pi_g"]), games_alpha=float(c["alpha_g"]),
            games_beta=float(c["beta_g"]), starts_alpha=float(c["alpha_j"]),
            starts_beta=float(c["beta_j"]), overtime_rate_per_game=float(c["nu"]),
            minutes_alpha=float(c["alpha_m"]), minutes_beta=float(c["beta_m"]),
            minutes_max_probability=float(c["pi_m"]),
            regulation_minutes=self.regulation_minutes, overtime_minutes=self.overtime_minutes,
            production=prod)


# --------------------------------------------------------------------------- #
# Batches
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SeasonBatch:
    games: torch.Tensor            # [B] long
    starts: torch.Tensor           # [B] long
    recorded_minutes: torch.Tensor  # [B] double
    counts: torch.Tensor           # [B, 13] long, COUNT_NAMES order
    schedule: torch.Tensor         # [B] long
    overtime: torch.Tensor | None = None        # [B] long, observed overtime periods where known
    overtime_known: torch.Tensor | None = None  # [B] bool; False rows sum overtime out

    @classmethod
    def from_outcomes(cls, outcomes, schedules) -> "SeasonBatch":
        outcomes = list(outcomes)
        schedules = [int(s) for s in (schedules if not isinstance(schedules, int) else [schedules] * len(outcomes))]
        return cls(
            games=torch.tensor([o.games for o in outcomes], dtype=torch.long),
            starts=torch.tensor([o.starts for o in outcomes], dtype=torch.long),
            recorded_minutes=torch.tensor([float(o.recorded_minutes) for o in outcomes], dtype=DTYPE),
            counts=torch.tensor([[int(o.counts[n]) for n in COUNT_NAMES] for o in outcomes], dtype=torch.long),
            schedule=torch.tensor(schedules, dtype=torch.long))

    def __len__(self):
        return int(self.games.shape[0])

    def subset(self, index: torch.Tensor) -> "SeasonBatch":
        return SeasonBatch(self.games[index], self.starts[index], self.recorded_minutes[index],
                           self.counts[index], self.schedule[index],
                           None if self.overtime is None else self.overtime[index],
                           None if self.overtime_known is None else self.overtime_known[index])


@dataclass(frozen=True)
class BatchDiagnostics:
    overtime_terms: int                 # maximum over rows
    max_overtime_log_error_bound: float
    max_poisson_log_tail_bound: float
    quadrature_nodes: int
    invalid_observations: int
    mean_overtime_terms: float = float("nan")
    overtime_terms_per_row: torch.Tensor | None = None   # [B] long; 0 for zero-game rows


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #

PARAMETER_SHAPES = {"pi_g": (), "alpha_g": (), "beta_g": (), "alpha_j": (), "beta_j": (), "nu": (),
                    "alpha_m": (), "beta_m": (), "pi_m": (), "rates": (len(RATE_NAMES),),
                    "dispersions": (len(RATE_NAMES),), "makes_alpha": (len(MAKE_ATTEMPTS),),
                    "makes_beta": (len(MAKE_ATTEMPTS),)}


def expand_parameters(c: dict, B: int) -> dict:
    """Give every parameter a leading batch dimension: [B] for scalars, [B, n] for vectors.

    An intercept-only model yields 0-d / [n] tensors that are expanded (a
    view, so gradients still reach the shared parameters); conditional heads
    yield per-row tensors that pass through after a shape check.
    """
    out = {}
    for name, tail in PARAMETER_SHAPES.items():
        v = torch.as_tensor(c[name], dtype=DTYPE)
        if v.dim() == len(tail):
            v = v.expand(B, *tail) if tail else v.expand(B)
        if tuple(v.shape) != (B, *tail):
            raise ValueError(f"Parameter {name!r} must have shape {(B, *tail)}, got {tuple(v.shape)}")
        if not bool(torch.isfinite(v).all()):
            # Non-finite heads would otherwise surface as a spurious overtime-budget failure.
            raise ValueError(f"Parameter {name!r} contains non-finite values (diverged model)")
        out[name] = v
    return out


class DifferentiableSeasonLikelihood:
    """Observed-data log probability of a batch.

    Parameters come either from an ``InterceptOnlySeasonModel`` (shared
    across rows) or per row from conditional heads, passed as ``params`` to
    ``log_prob``; both are expanded to a leading batch dimension so one code
    path serves the validated intercept-only case and the tower.
    """

    def __init__(self, model: InterceptOnlySeasonModel | None = None, kernel: OnceRoundedMinutesKernel | None = None,
                 accuracy: NumericalAccuracy | None = None, quadrature_nodes: int = 32, *,
                 regulation_minutes: float = 40.0, overtime_minutes: float = 5.0, foul_limit_per_game: int = 5,
                 vectorized: bool = True):
        if quadrature_nodes < 2:
            raise ValueError("At least two quadrature nodes are required")
        self.model = model
        self.regulation_minutes = float(model.regulation_minutes) if model is not None else float(regulation_minutes)
        self.overtime_minutes = float(model.overtime_minutes) if model is not None else float(overtime_minutes)
        self.foul_limit_per_game = int(model.foul_limit_per_game) if model is not None else int(foul_limit_per_game)
        self.kernel = kernel or OnceRoundedMinutesKernel()
        self.accuracy = accuracy or NumericalAccuracy()
        self.nodes = quadrature_nodes
        x, w = np.polynomial.legendre.leggauss(quadrature_nodes)
        self.t = torch.tensor((x + 1.0) / 2.0, dtype=DTYPE)          # nodes in (0, 1)
        self.logw = torch.log(torch.tensor(w / 2.0, dtype=DTYPE))    # weights on [0, 1]
        # ``vectorized`` evaluates the same law with far fewer tensor operations (2026-11-16): the count types in one
        # call, the three quadrature pieces and the atom in one production evaluation, and every row whose overtime is
        # observed in one component call. ``vectorized=False`` is the original loop, kept as the reference the
        # equivalence test compares against; the two agree to floating-point reassociation.
        self.vectorized = bool(vectorized)
        self._count_index = {name: j for j, name in enumerate(COUNT_NAMES)}
        self._rate_index = {name: i for i, name in enumerate(RATE_NAMES)}
        self._make_index = {name: i for i, name in enumerate(MAKE_ATTEMPTS)}
        plain = [name for name in COUNT_NAMES if name in self._rate_index and name != "PF"]
        self._nb_cols = [self._count_index[name] for name in plain]
        self._nb_rates = [self._rate_index[name] for name in plain]

    # -- production at latent minutes m [B, K] ------------------------------------
    def _production_logpmf(self, counts, m, games, c):
        return self._production_logpmf_vectorized(counts, m, games, c) if self.vectorized else self._production_logpmf_loop(counts, m, games, c)

    def _production_logpmf_vectorized(self, counts, m, games, c):
        """The same sum as the loop below with one negative-binomial call for the nine untruncated count types."""
        exposure = m / self.regulation()                                    # [B, K]
        kf = counts.to(DTYPE)
        total = torch.zeros_like(m)
        for name, attempt in MAKE_ATTEMPTS.items():                         # makes do not depend on the latent minutes
            i = self._make_index[name]
            total = total + beta_binomial_logpmf(kf[:, self._count_index[name], None], kf[:, self._count_index[attempt], None],
                                                 c["makes_alpha"][:, i, None], c["makes_beta"][:, i, None])
        k = kf[:, self._nb_cols][:, None, :]                                # [B, 1, R]
        rates, disps = c["rates"][:, self._nb_rates][:, None, :], c["dispersions"][:, self._nb_rates][:, None, :]
        total = total + negative_binomial_logpmf(k, exposure[:, :, None] * rates, disps).sum(dim=-1)
        i = self._rate_index["PF"]
        return total + truncated_negative_binomial_logpmf(kf[:, self._count_index["PF"], None], exposure * c["rates"][:, i, None],
                                                          c["dispersions"][:, i, None], self.foul_limit_per_game * games, fast=True)

    def _production_logpmf_loop(self, counts, m, games, c):
        exposure = m / self.regulation()
        total = torch.zeros_like(m)
        rate_index = {name: i for i, name in enumerate(RATE_NAMES)}
        make_index = {name: i for i, name in enumerate(MAKE_ATTEMPTS)}
        for j, name in enumerate(COUNT_NAMES):
            k = counts[:, j].to(DTYPE)[:, None]
            if name in MAKE_ATTEMPTS:
                attempts = counts[:, COUNT_NAMES.index(MAKE_ATTEMPTS[name])].to(DTYPE)[:, None]
                i = make_index[name]
                total = total + beta_binomial_logpmf(k, attempts, c["makes_alpha"][:, i, None], c["makes_beta"][:, i, None])
            elif name == "PF":
                i = rate_index[name]
                cap = self.foul_limit_per_game * games
                total = total + truncated_negative_binomial_logpmf(k, exposure * c["rates"][:, i, None], c["dispersions"][:, i, None], cap)
            else:
                i = rate_index[name]
                total = total + negative_binomial_logpmf(k, exposure * c["rates"][:, i, None], c["dispersions"][:, i, None])
        return total

    def regulation(self):
        return self.regulation_minutes

    def overtime_terms_for_tail(self, batch: SeasonBatch, nu: torch.Tensor, tolerance: float, margin: int = 6, cap: int = 128) -> torch.Tensor:
        """Per-row overtime term counts for optimization — the fixed-count mode of ``log_prob`` (2026-11-16).

        The adaptive sum stops a row when the omitted Poisson mass is small *relative to the row's joint probability*,
        bounding every omitted component by one. That bound is rigorous and very loose: a season's joint probability is
        of order e^-60, so a row with unobserved overtime evaluates about thirty terms where the mass beyond the
        thirteenth is below 1e-8 of the total. For gradient steps this returns instead the smallest count whose omitted
        Poisson mass Pr(O > o) is at most ``tolerance``, never fewer than ``margin`` terms beyond the first overtime
        count that can hold the recorded minutes. Scoring keeps the adaptive rule. No gradient flows through the counts."""
        with torch.no_grad():
            g = batch.games.to(DTYPE)
            mu = (g * nu.detach().to(DTYPE)).clamp(min=1e-12)
            needed = torch.ceil(((batch.recorded_minutes - g * self.regulation()) / self.overtime_minutes).clamp(min=0)).long()
            grid = torch.arange(cap, dtype=DTYPE)
            tail = torch.special.gammainc(grid[None, :] + 1.0, mu[:, None])            # Pr(O > o) for o = 0 .. cap - 1
            ok = tail <= tolerance
            first = torch.where(ok.any(dim=1), ok.to(torch.long).argmax(dim=1), torch.full((len(mu),), cap - 1, dtype=torch.long))
            return torch.maximum(first + 1, needed + 1 + margin)

    # -- one overtime component: log of (1-pi_m)∫cell + atom, per element ----------
    def _component(self, batch: SeasonBatch, maximum: torch.Tensor, c: dict, production: bool = True, moment: bool = False):
        """With ``production=False`` the integrand is the minutes density alone (P ≡ 1), giving the
        marginal probability of the recorded minutes given games and overtime; the difference between
        the two is the exact log probability of the counts given the observed opportunity. With
        ``moment=True`` the integrand is additionally multiplied by the latent fraction u (the atom
        sits at u = 1), so that exp(moment − opportunity) is E[u | G, O, M]."""
        B = len(batch)
        # Rows without a valid cell are masked out below; they still flow through
        # every piece, so give them a benign interior placeholder cell rather
        # than [0, 0], which would evaluate production at ~1e-300 minutes and
        # poison the backward pass of the selected rows through 0 * inf.
        lo = np.full(B, 0.25)
        hi = np.full(B, 0.75)
        valid = np.zeros(B, dtype=bool)
        atom = np.zeros(B, dtype=bool)
        recorded = batch.recorded_minutes.tolist()
        maxima = maximum.tolist()
        games_list = batch.games.tolist()
        games_aware = bool(getattr(self.kernel, "games_aware", False))
        for i in range(B):
            if maxima[i] <= 0:
                continue
            cell = (self.kernel.cell(recorded[i], maxima[i], games_list[i]) if games_aware
                    else self.kernel.cell(recorded[i], maxima[i]))
            if cell is None:
                continue
            valid[i] = True
            lo[i], hi[i] = cell[0] / maxima[i], cell[1] / maxima[i]
            atom[i] = bool(self.kernel.probability(recorded[i], maxima[i], maxima[i], games_list[i]) if games_aware
                           else self.kernel.probability(recorded[i], maxima[i], maxima[i]))
        maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))  # masked rows only
        lo_t, hi_t = torch.tensor(lo, dtype=DTYPE), torch.tensor(hi, dtype=DTYPE)
        valid_t, atom_t = torch.tensor(valid), torch.tensor(atom)
        alpha, beta, pi_m = c["alpha_m"], c["beta_m"], c["pi_m"]          # [B]
        lbeta = _lbeta(alpha, beta)[:, None]                                # [B, 1]
        a1, b1 = (alpha - 1)[:, None], (beta - 1)[:, None]
        t, logw = self.t[None, :], self.logw[None, :]
        games = batch.games

        K = self.nodes
        stacked = None
        if production and self.vectorized:
            # One production evaluation at [B, 3K + 1] latent fractions: the nodes of pieces A, B and C (the same
            # expressions as below) and u = 1 for the atom.
            width_ = (hi_t - lo_t).clamp(min=1e-300)
            h_ = torch.where(hi_t < 1, hi_t, torch.full_like(hi_t, 0.5)).clamp(min=1e-300)
            l_ = torch.where(lo_t > 0, lo_t, torch.full_like(lo_t, 0.5))
            span_ = (1 - l_).clamp(min=1e-300)
            fractions = torch.cat([lo_t[:, None] + width_[:, None] * t, h_[:, None] * t ** (1.0 / alpha)[:, None],
                                   1 - span_[:, None] * t ** (1.0 / beta)[:, None], torch.ones(B, 1, dtype=DTYPE)], dim=1)
            stacked = self._production_logpmf(batch.counts, maximum[:, None] * fractions, games, c)

        def logP(u, piece: int):  # production at latent minutes maximum*u, [B, K]; optionally times u for the first moment
            if not production:
                out = torch.zeros_like(u)
            elif stacked is not None:
                out = stacked[:, piece * K:(piece + 1) * K]
            else:
                out = self._production_logpmf(batch.counts, maximum[:, None] * u, games, c)
            return out + torch.log(u.clamp(min=1e-300)) if moment else out

        # Piece A: interior cells (0 < lo, hi < 1), plain Gauss-Legendre in u.
        a_mask = valid_t & (lo_t > 0) & (hi_t < 1)
        width = (hi_t - lo_t).clamp(min=1e-300)
        u_a = lo_t[:, None] + width[:, None] * t
        log_a = (torch.log(width)[:, None] + a1 * torch.log(u_a) + b1 * torch.log1p(-u_a)
                 - lbeta + logP(u_a, 0) + logw)
        piece_a = torch.logsumexp(log_a, dim=1)

        # Piece B: lower endpoint, u = h t^{1/alpha}, h = hi (or 1/2 when the cell spans [0,1]).
        b_mask = valid_t & (lo_t == 0)
        h = torch.where(hi_t < 1, hi_t, torch.full_like(hi_t, 0.5)).clamp(min=1e-300)
        u_b = h[:, None] * t ** (1.0 / alpha)[:, None]
        log_b = ((alpha * torch.log(h) - torch.log(alpha))[:, None] - lbeta
                 + b1 * torch.log1p(-u_b) + logP(u_b, 1) + logw)
        piece_b = torch.logsumexp(log_b, dim=1)

        # Piece C: upper endpoint, 1-u = (1-l) t^{1/beta}, l = lo (or 1/2 when the cell spans [0,1]).
        c_mask = valid_t & (hi_t == 1)
        l = torch.where(lo_t > 0, lo_t, torch.full_like(lo_t, 0.5))
        span = (1 - l).clamp(min=1e-300)
        u_c = 1 - span[:, None] * t ** (1.0 / beta)[:, None]
        log_c = ((beta * torch.log(span) - torch.log(beta))[:, None] - lbeta
                 + a1 * torch.log(u_c) + logP(u_c, 2) + logw)
        piece_c = torch.logsumexp(log_c, dim=1)

        zero = torch.full_like(piece_a, LOG_ZERO)
        continuous = torch.logsumexp(torch.stack([
            torch.where(a_mask, _finite_log(piece_a), zero),
            torch.where(b_mask, _finite_log(piece_b), zero),
            torch.where(c_mask, _finite_log(piece_c), zero)]), dim=0)
        continuous = torch.log1p(-pi_m) + continuous

        # Atom at full available minutes, weighted by production at maximum.
        if not production:
            prod_max = torch.zeros_like(piece_a)
        elif stacked is not None:
            prod_max = stacked[:, 3 * K]
        else:
            prod_max = self._production_logpmf(batch.counts, maximum[:, None], games, c).squeeze(1)
        atom_log = torch.where(atom_t, torch.log(pi_m) + _finite_log(prod_max), zero)
        return torch.where(valid_t, torch.logaddexp(continuous, atom_log), zero)

    # -- structural validity (data only), mirroring the reference ---------------
    def _valid(self, batch: SeasonBatch) -> torch.Tensor:
        g, j, s, m, k = batch.games, batch.starts, batch.schedule, batch.recorded_minutes, batch.counts
        ok = (0 <= j) & (j <= g) & (g <= s) & (m >= 0) & torch.isfinite(m) & (k >= 0).all(dim=1)
        for make, attempt in MAKE_ATTEMPTS.items():
            ok &= k[:, COUNT_NAMES.index(make)] <= k[:, COUNT_NAMES.index(attempt)]
        ok &= k[:, COUNT_NAMES.index("PF")] <= self.foul_limit_per_game * g
        zero = (g == 0)
        ok &= ~zero | ((m == 0) & (k == 0).all(dim=1))
        step = self.kernel.step
        on_grid = torch.isclose(m / step, torch.round(m / step), atol=1e-10, rtol=0)
        possible_o = (m - g.to(DTYPE) * self.regulation()) / self.overtime_minutes
        on_endpoint = (possible_o >= 0) & torch.isclose(possible_o, torch.round(possible_o), atol=1e-10, rtol=0)
        ok &= zero | on_grid | on_endpoint
        # Recorded minutes beyond what the registered overtime budget can ever accommodate are an
        # impossible observation (a data defect), scored -inf rather than exhausting the budget (D-036).
        ok &= zero | (m <= g.to(DTYPE) * self.regulation() + self.overtime_minutes * self.accuracy.max_overtime_terms)
        if batch.overtime_known is not None:
            # An observed overtime count must be a nonnegative integer that leaves the recorded
            # minutes within the available maximum.
            o = batch.overtime.to(DTYPE)
            known = batch.overtime_known
            fits = (o >= 0) & (m <= g.to(DTYPE) * self.regulation() + self.overtime_minutes * o + 1e-9)
            ok &= ~known | (fits & (o == torch.round(o)))
        return ok

    def log_prob_components(self, batch: SeasonBatch, params: dict | None = None) -> dict:
        """Factors of the observed-data log probability, each [B], summing to ``total``:
        ``participation`` (the hurdle: log(1-π) for zero games, log π otherwise),
        ``games_given_participation`` (the renormalized beta-binomial count, zero on the zero branch),
        ``starts``, and ``rest`` (overtime, latent and recorded minutes, production).
        ``games`` = participation + games_given_participation is kept for convenience."""
        total, _ = self.log_prob(batch, params=params)
        B = len(batch)
        c = expand_parameters(self.model.constrained() if params is None else params, B)
        positive = batch.games > 0
        games = _finite_log(hurdle_beta_binomial_logpmf(batch.games, batch.schedule, c["pi_g"], c["alpha_g"], c["beta_g"]))
        participation = torch.where(positive, torch.log(c["pi_g"]), torch.log1p(-c["pi_g"]))
        participation = torch.where(batch.schedule > 0, participation, torch.zeros_like(participation))
        games_given = torch.where(positive, games - participation, torch.zeros_like(games))
        starts = _finite_log(beta_binomial_logpmf(batch.starts, batch.games, c["alpha_j"], c["beta_j"]))
        starts = torch.where(positive, starts, torch.zeros_like(starts))
        rest = torch.where(torch.isfinite(total), total - games - starts, torch.full_like(total, NEG_INF))
        return {"total": total, "games": games, "participation": participation, "games_given_participation": games_given,
                "starts": starts, "rest": rest}

    def log_prob_opportunity(self, batch: SeasonBatch, params: dict | None = None, overtime_terms=None) -> torch.Tensor:
        """Marginal log probability of the *opportunity* outcome (games, starts, recorded minutes) with
        production integrated out: the same overtime sum and minutes quadrature as ``log_prob`` with the
        production factor set to one. ``log_prob − log_prob_opportunity`` is then the exact log
        probability of the 13 counts given the observed opportunity (D-032)."""
        return self.log_prob(batch, overtime_terms=overtime_terms, params=params, production=False)[0]

    def latent_minutes_given_opportunity(self, batch: SeasonBatch, params: dict | None = None) -> torch.Tensor:
        """$E[M^\\ast \\mid G, J, M, x]$: latent minutes integrated over the recording cell and the overtime
        sum with production excluded from the weights (D-033). Zero-game rows return zero.

        Both sums stop under their own relative bound (the numerator's uses the first-moment tail, D-095), each at most
        the registered overtime tolerance, so the log of the returned ratio is within twice that tolerance of the
        exact conditional mean (quadrature error aside)."""
        opp = self.log_prob(batch, params=params, production=False)[0]
        moment = self.log_prob(batch, params=params, production=False, moment=True)[0]
        expected = torch.exp(moment - opp)
        return torch.where(batch.games > 0, expected, torch.zeros_like(expected))

    def log_prob(self, batch: SeasonBatch, overtime_terms=None, params: dict | None = None, production: bool = True,
                 moment: bool = False):
        """Return ([B] log probabilities, BatchDiagnostics).

        Overtime is summed per row until that row meets the registered bound
        (as the reference does); converged rows leave the active set, so a
        rare observation does not make every row pay for its tail. With a
        fixed ``overtime_terms`` — an int for every positive-game row, or a
        [B] long tensor of per-row counts — the term counts do not depend on
        the parameters, which keeps the objective smooth for gradient checks
        and optimization. Per-row counts are typically taken from one
        adaptive pass plus a margin and re-verified at the optimum.

        Rows whose overtime is observed (``batch.overtime_known``) contribute
        the single term $o = O$, i.e. $p(O\\mid g,u)$ times the component at
        that overtime — the joint observed-data probability with $O$ as an
        observed outcome — and take part in no convergence check.

        ``params`` supplies per-row parameters from conditional heads; without
        it the shared parameters of ``self.model`` are expanded to the batch.
        """
        B = len(batch)
        if params is None:
            if self.model is None:
                raise ValueError("Either a model or per-row params is required")
            params = self.model.constrained()
        c = expand_parameters(params, B)
        valid = self._valid(batch)
        # Invalid observations are scored -inf on the output only. Their pieces are computed on a benign
        # substitute (a zero-game row): an impossible count would make a whole quadrature row -inf, and the
        # backward of logsumexp over such a row is NaN even when the loss excludes it (D-037).
        batch = _safe_batch(batch, valid)
        games_log = hurdle_beta_binomial_logpmf(batch.games, batch.schedule, c["pi_g"], c["alpha_g"], c["beta_g"])
        starts_log = beta_binomial_logpmf(batch.starts, batch.games, c["alpha_j"], c["beta_j"])
        positive = batch.games > 0
        mean = batch.games.to(DTYPE) * c["nu"]
        log_total = torch.full((B,), LOG_ZERO, dtype=DTYPE)
        acc = self.accuracy
        fixed = overtime_terms is not None
        known = (batch.overtime_known & positive) if batch.overtime_known is not None else torch.zeros(B, dtype=torch.bool)
        observed = batch.overtime.to(torch.long) if batch.overtime is not None else torch.zeros(B, dtype=torch.long)
        per_row = None
        if isinstance(overtime_terms, torch.Tensor):
            per_row = overtime_terms.to(torch.long)
            if per_row.shape != (B,):
                raise ValueError("Per-row overtime term counts must have shape [B]")
            limit = int(per_row.max().item()) if B else 0
        else:
            limit = int(overtime_terms) if fixed else acc.max_overtime_terms
        if bool(known.any()):
            limit = max(limit, int(observed[known].max().item()) + 1)
        unknown = positive & ~known
        active = ((valid & unknown) if not fixed else unknown.clone())
        terms_used = torch.zeros(B, dtype=torch.long)
        bounds = torch.zeros(B, dtype=DTYPE)
        tails = torch.full((B,), NEG_INF, dtype=DTYPE)
        pending = known                                   # rows with observed overtime still to be scored inside the loop
        if self.vectorized and bool(known.any()):
            # Every row with observed overtime contributes the single term o = O: all of them in one component call,
            # each at its own available maximum, instead of one call per distinct overtime count.
            idx = known.nonzero(as_tuple=False).squeeze(1)
            o_row = observed[idx]
            sub = batch.subset(idx)
            c_sub = {k: v[idx] for k, v in c.items()}
            maximum = sub.games.to(DTYPE) * self.regulation() + self.overtime_minutes * o_row.to(DTYPE)
            component = self._component(sub, maximum, c_sub, production=production, moment=moment)
            if moment:
                component = component + torch.log(maximum)
            updated = torch.logaddexp(log_total[idx], _finite_log(poisson_logpmf(o_row.to(DTYPE), mean[idx])) + component)
            log_total = log_total.index_put((idx,), updated)
            terms_used[idx] = o_row + 1
            pending = torch.zeros_like(known)
        for o in range(limit):
            if per_row is not None:
                active = unknown & (per_row > o)
            current = active | (pending & (observed == o))
            idx = current.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                if not bool(active.any()) and not bool((pending & (observed >= o)).any()):
                    break
                continue
            sub = batch.subset(idx)
            c_sub = {k: v[idx] for k, v in c.items()}
            maximum = sub.games.to(DTYPE) * self.regulation() + self.overtime_minutes * o
            component = self._component(sub, maximum, c_sub, production=production, moment=moment)
            if moment:
                component = component + torch.log(maximum)      # E[m*] = maximum × E[u], summed over overtime
            updated = torch.logaddexp(log_total[idx], _finite_log(poisson_logpmf(float(o), mean[idx])) + component)
            log_total = log_total.index_put((idx,), updated)
            terms_used[idx] = o + 1
            if fixed:
                continue
            check = active[idx]
            if not bool(check.any()):
                continue
            with torch.no_grad():
                cidx = idx[check]
                mus = mean[cidx].tolist()
                sub_tails = torch.tensor([poisson_log_tail_upper_bound(o, mu) for mu in mus], dtype=DTYPE)   # log Pr(O > o)
                omitted = sub_tails
                if moment:
                    # D-095: a moment term is Pois(o') × maximum(o') × E[u | ...] with maximum(o') = L g + t o' > 1, so the
                    # probability tail does not bound the omitted numerator. With E[u | ...] <= 1 and o' Pois(o'; mu) =
                    # mu Pois(o' - 1; mu): omitted <= L g Pr(O > o) + t mu Pr(O >= o).
                    ge = torch.tensor([0.0 if o == 0 else poisson_log_tail_upper_bound(o - 1, mu) for mu in mus], dtype=DTYPE)
                    lg = torch.log(batch.games[cidx].to(DTYPE) * self.regulation())
                    tmu = torch.log(self.overtime_minutes * mean[cidx].detach())
                    omitted = torch.logaddexp(lg + sub_tails, tmu + ge)
                sub_bound = torch.logaddexp(torch.zeros_like(omitted), omitted - log_total[cidx].detach())
                converged = (sub_tails <= math.log(acc.absolute_tail_tolerance)) & (sub_bound <= acc.overtime_log_tolerance)
                bounds[cidx], tails[cidx] = sub_bound, sub_tails
                active[cidx[converged]] = False
        else:
            if not fixed and bool(active.any()):
                raise OvertimeBudgetError(
                    f"Overtime failed its registered budget after {limit} terms for "
                    f"{int(active.sum())} rows; worst log-score bound={float(bounds[active].max()):g}. "
                    "No truncated result was returned.")
        neg = torch.full_like(games_log, NEG_INF)
        total = torch.where(positive, _finite_log(games_log) + _finite_log(starts_log) + log_total,
                            _finite_log(games_log))
        # Restore exact -inf for impossible observations; the where() gives
        # the sentinel path zero gradient, so no NaN can reach the parameters.
        total = torch.where(valid & (total > -1e200), total, neg)
        scored = valid & positive & ~known
        diag = BatchDiagnostics(
            int(terms_used.max()) if B else 0,
            float(bounds[scored].max()) if (not fixed and bool(scored.any())) else float("nan"),
            float(tails[scored].max()) if (not fixed and bool(scored.any())) else float("nan"),
            self.nodes, int((~valid).sum()),
            float(terms_used[scored].to(DTYPE).mean()) if bool(scored.any()) else float("nan"),
            terms_used)
        return total, diag


# --------------------------------------------------------------------------- #
# Validation harness: values against the reference, gradients against finite differences
# --------------------------------------------------------------------------- #

def reference_log_probs(model: InterceptOnlySeasonModel, outcomes, schedules, kernel=None, accuracy=None):
    from .likelihood import SeasonModel
    ref = SeasonModel(model.reference(), kernel or OnceRoundedMinutesKernel(), accuracy or NumericalAccuracy())
    if isinstance(schedules, int):
        schedules = [schedules] * len(outcomes)
    return np.array([ref.log_prob(o, int(s)) for o, s in zip(outcomes, schedules)], dtype=float)


def value_agreement(likelihood: DifferentiableSeasonLikelihood, outcomes, schedules) -> dict:
    batch = SeasonBatch.from_outcomes(outcomes, schedules)
    with torch.no_grad():
        ours, diag = likelihood.log_prob(batch)
    ref = reference_log_probs(likelihood.model, outcomes, schedules, likelihood.kernel, likelihood.accuracy)
    ours_np = ours.numpy()
    both_finite = np.isfinite(ours_np) & np.isfinite(ref)
    same_support = np.all(np.isfinite(ours_np) == np.isfinite(ref))
    diff = np.abs(ours_np[both_finite] - ref[both_finite])
    return {"n": len(outcomes), "same_support": bool(same_support),
            "max_abs_diff_nats": float(diff.max()) if diff.size else 0.0,
            "mean_abs_diff_nats": float(diff.mean()) if diff.size else 0.0,
            "overtime_terms": diag.overtime_terms, "quadrature_nodes": diag.quadrature_nodes,
            "reference": ref.tolist(), "ours": ours_np.tolist()}


def finite_difference_check(likelihood: DifferentiableSeasonLikelihood, batch: SeasonBatch,
                            overtime_terms: int = 40, eps: float = 1e-6) -> dict:
    """Compare autodiff gradients of the summed log probability with central differences.

    The overtime term count is fixed so the objective is smooth in the
    parameters; adaptive stopping is a control decision, not part of the model.
    """
    model = likelihood.model
    params = [p for p in model.parameters()]
    names = [n for n, _ in model.named_parameters()]

    def objective():
        lp, _ = likelihood.log_prob(batch, overtime_terms=overtime_terms)
        finite = torch.isfinite(lp)
        if not bool(finite.all()):
            raise ValueError("Gradient check requires a batch of valid observations")
        return lp.sum()

    model.zero_grad(set_to_none=True)
    objective().backward()
    autograd = torch.cat([p.grad.detach().flatten() for p in params]).clone()
    flat = torch.cat([p.detach().flatten() for p in params]).clone()
    fd = torch.zeros_like(flat)
    with torch.no_grad():
        for i in range(flat.numel()):
            for sign in (+1.0, -1.0):
                bumped = flat.clone()
                bumped[i] += sign * eps
                offset = 0
                for p in params:
                    n = p.numel()
                    p.copy_(bumped[offset:offset + n].view_as(p))
                    offset += n
                fd[i] += sign * objective() / (2 * eps)
        offset = 0
        for p in params:                    # restore
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n
    abs_err = (autograd - fd).abs()
    scale = torch.maximum(autograd.abs(), fd.abs()).clamp(min=1e-8)
    rel_err = abs_err / scale
    per_param = []
    offset = 0
    for name, p in zip(names, params):
        n = p.numel()
        per_param.append({"parameter": name, "max_abs_err": float(abs_err[offset:offset + n].max()),
                          "max_rel_err": float(rel_err[offset:offset + n].max())})
        offset += n
    return {"max_abs_err": float(abs_err.max()), "max_rel_err": float(rel_err.max()),
            "eps": eps, "overtime_terms": overtime_terms, "n_parameters": int(flat.numel()),
            "per_parameter": per_param}


def time_batch(likelihood: DifferentiableSeasonLikelihood, batch: SeasonBatch, repeats: int = 3) -> dict:
    """Seconds per forward+backward pass (adaptive overtime), median over repeats."""
    times = []
    for _ in range(repeats):
        likelihood.model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        lp, diag = likelihood.log_prob(batch)
        (-lp[torch.isfinite(lp)]).sum().backward()
        times.append(time.perf_counter() - start)
    return {"batch_size": len(batch), "median_seconds": float(np.median(times)),
            "seconds_per_unit": float(np.median(times) / max(len(batch), 1)),
            "overtime_terms": diag.overtime_terms, "quadrature_nodes": diag.quadrature_nodes,
            "threads": torch.get_num_threads()}
