"""Discrete recorded-minute observations and a SciPy reference likelihood.

This is a synthetic, non-autograd implementation of paper equations
``kernel``, ``observation`` and ``otbound``.  Once-rounding a season total
is explicitly NOT a claim about NCAA provider provenance.  Empirical
comparisons require the separate data/provenance gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import warnings
from typing import Callable

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import minimize_scalar
from scipy.special import betaln, betainc, betaincc, gammaln
from scipy.stats import poisson


class NumericalAccuracyError(RuntimeError):
    """A registered numerical budget could not be met within resource limits."""


@dataclass(frozen=True)
class KernelProvenance:
    registration_id: str = "synthetic-once-rounded-total-v1"
    provider: str = "synthetic fixture"
    rule: str = "nearest bounded grid point; ties upward; exact endpoints"
    aggregation: str = "once_rounded_season_total"
    verified: bool = False
    validated_working_model: bool = False
    synthetic_only: bool = True

    def __post_init__(self):
        if not self.registration_id or not self.provider:
            raise ValueError("Kernel registration and provider are required")
        if self.aggregation != "once_rounded_season_total":
            raise ValueError("This kernel implements only once-rounded totals")

    @property
    def empirical_allowed(self) -> bool:
        return not self.synthetic_only and (self.verified or self.validated_working_model)


@dataclass(frozen=True)
class OnceRoundedMinutesKernel:
    """Deterministic nearest quantization, including both bounded endpoints.

    Support is ``{0, step, ..., floor(maximum/step)*step} union {maximum}``.
    Midpoints between adjacent support points define cells.  This partition
    yields a normalized discrete kernel even when the final cell is shorter.
    """

    step: float = 1.0
    provenance: KernelProvenance = field(default_factory=KernelProvenance)

    def __post_init__(self):
        if not math.isfinite(self.step) or self.step <= 0:
            raise ValueError("Recorded grid resolution must be positive and finite")

    def as_dict(self) -> dict:
        return asdict(self)

    def _last_grid(self, maximum: float) -> float:
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError("Minute maximum must be finite and nonnegative")
        return min(maximum, math.floor(maximum / self.step) * self.step)

    def support(self, maximum: float) -> np.ndarray:
        last = self._last_grid(maximum)
        values = np.arange(round(last / self.step) + 1, dtype=float) * self.step
        if last < maximum and not math.isclose(last, maximum, abs_tol=1e-12):
            values = np.append(values, maximum)
        else:
            values[-1] = maximum
        return values

    def record(self, latent_minutes: float, maximum: float) -> float:
        last = self._last_grid(maximum)
        if not math.isfinite(latent_minutes) or not 0 <= latent_minutes <= maximum:
            raise ValueError("Latent minutes must lie within their declared support")
        if maximum == 0:
            return 0.0
        if latent_minutes >= (last + maximum) / 2:
            return float(maximum)
        return float(min(last, math.floor(latent_minutes / self.step + 0.5) * self.step))

    def sample(self, latent_minutes: float, maximum: float, rng=None) -> float:
        return self.record(latent_minutes, maximum)

    def cell(self, recorded: float, maximum: float) -> tuple[float, float] | None:
        """Return the latent interval corresponding to a recorded grid point."""
        last = self._last_grid(maximum)
        if not math.isfinite(recorded) or recorded < 0 or recorded > maximum:
            return None
        if maximum == 0:
            return (0.0, 0.0) if recorded == 0 else None
        if math.isclose(recorded, maximum, rel_tol=0, abs_tol=1e-10):
            previous = last if last < maximum - 1e-10 else max(0.0, last - self.step)
            return ((previous + maximum) / 2, maximum)
        index = round(recorded / self.step)
        value = index * self.step
        if index < 0 or not math.isclose(recorded, value, rel_tol=0, abs_tol=1e-10):
            return None
        if value > last:
            return None
        previous = max(0.0, value - self.step)
        following = min(maximum, value + self.step)
        return (0.0 if value == 0 else (previous + value) / 2,
                (value + following) / 2)

    def probability(self, recorded: float, latent_minutes: float, maximum: float) -> float:
        if self.cell(recorded, maximum) is None:
            return 0.0
        return float(math.isclose(recorded, self.record(latent_minutes, maximum),
                                  rel_tol=0, abs_tol=1e-10))


@dataclass(frozen=True)
class SummedRoundingSensitivityKernel:
    """Registered sensitivity probe for NCAA provenance (D-026).

    NCAA season minutes are sums of per-game minutes rounded individually, so
    the recording error is Irwin–Hall-like with variance $G/12$, not a single
    ±½ cell. This kernel widens the once-rounded cell to ±max(step/2,
    c·sqrt(G/12)) around the recorded total (clipped to [0, maximum]) and is
    scored alongside the once-rounded kernel; the gap between the two
    log scores bounds what the working approximation can be hiding. It is
    a probe, not a validated provider kernel: an indicator over ±1 SD
    stands in for the smooth Irwin–Hall density.
    """
    step: float = 1.0
    sd_multiple: float = 1.0
    registration_id: str = "ncaa-summed-rounding-sensitivity-v1"
    games_aware: bool = True

    def half_width(self, games: int) -> float:
        return max(self.step / 2.0, self.sd_multiple * math.sqrt(max(int(games), 1) / 12.0))

    def cell(self, recorded: float, maximum: float, games: int = 1) -> tuple[float, float] | None:
        base = OnceRoundedMinutesKernel(self.step).cell(recorded, maximum)
        if base is None:
            return None
        if maximum == 0:
            return base
        h = self.half_width(games)
        lo, hi = max(0.0, recorded - h), min(maximum, recorded + h)
        return (min(lo, base[0]), max(hi, base[1]))

    def probability(self, recorded: float, latent_minutes: float, maximum: float, games: int = 1) -> float:
        c = self.cell(recorded, maximum, games)
        if c is None:
            return 0.0
        return float(c[0] <= latent_minutes <= c[1])


@dataclass(frozen=True)
class NumericalAccuracy:
    overtime_log_tolerance: float = 5e-5
    quadrature_log_tolerance: float = 5e-5
    absolute_tail_tolerance: float = 1e-8
    max_overtime_terms: int = 512
    quadrature_limit: int = 200
    max_quadrature_refinements: int = 4

    def __post_init__(self):
        for value in (self.overtime_log_tolerance, self.quadrature_log_tolerance,
                      self.absolute_tail_tolerance):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Numerical tolerances must be positive and finite")
        if self.absolute_tail_tolerance >= 1:
            raise ValueError("Absolute Poisson-tail tolerance must be below one")
        for value in (self.max_overtime_terms, self.quadrature_limit,
                      self.max_quadrature_refinements):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Numerical resource limits must be positive integers")


@dataclass(frozen=True)
class IntegrationResult:
    log_probability: float
    quadrature_log_discrepancy: float = 0.0
    evaluations: int = 0
    used_quadrature: bool = False


@dataclass(frozen=True)
class ObservationDiagnostics:
    overtime_terms: int
    poisson_log_tail_bound: float
    overtime_log_error_bound: float
    quadrature_log_discrepancy: float
    quadrature_evaluations: int
    used_quadrature: bool
    total_error_certified: bool = False
    accuracy_status: str = "empirical quadrature convergence; no certified total-error guarantee"


@dataclass(frozen=True)
class ObservationResult:
    log_probability: float
    diagnostics: ObservationDiagnostics


def _beta_interval_probability(lo: float, hi: float, alpha: float, beta: float) -> float:
    # Subtract lower CDFs in the lower tail and survival functions in the
    # upper tail, avoiding catastrophic cancellation near one.
    if lo == 0:
        return float(betainc(alpha, beta, hi))
    if hi == 1:
        return float(betaincc(alpha, beta, lo))
    lower = float(betainc(alpha, beta, lo))
    if lower < 0.5:
        return max(0.0, float(betainc(alpha, beta, hi)) - lower)
    return max(0.0, float(betaincc(alpha, beta, lo) - betaincc(alpha, beta, hi)))


def _checked_production(log_prob: Callable[[float], float], latent_minutes: float) -> float:
    result = float(log_prob(latent_minutes))
    if math.isnan(result) or result > 1e-10:
        raise ValueError("Production callback must return a discrete log probability <= 0")
    return result


def _production_integral(lo: float, hi: float, maximum: float, alpha: float,
                         beta: float, production_log_prob: Callable[[float], float],
                         accuracy: NumericalAccuracy) -> IntegrationResult:
    """Integrate in a rescaled cell with algebraic beta endpoints factored out.

    QUADPACK's algebraic-weight rule handles beta endpoint singularities.
    Log scaling preserves rare production probabilities.  Refinement is an
    empirical convergence check, explicitly not a rigorous error certificate.
    """
    width = hi - lo
    left_power = alpha - 1 if lo == 0 else 0.0
    right_power = beta - 1 if hi == 1 else 0.0
    log_constant = ((1 + left_power + right_power) * math.log(width)
                    - betaln(alpha, beta))
    evaluations = 0

    def smooth_log(t):
        nonlocal evaluations
        evaluations += 1
        u = lo + width * t
        logp = _checked_production(production_log_prob, maximum * u)
        if lo != 0:
            logp += (alpha - 1) * math.log(u)
        if hi != 1:
            logp += (beta - 1) * math.log1p(-u)
        return logp

    probes = [smooth_log(t) for t in np.linspace(0, 1, 33)]
    optimum = minimize_scalar(lambda t: -smooth_log(t), bounds=(1e-12, 1 - 1e-12),
                              method="bounded")
    scale = max(max(probes), smooth_log(float(optimum.x)))
    if scale == -math.inf:
        return IntegrationResult(-math.inf, evaluations=evaluations, used_quadrature=True)

    def scaled(t):
        value = smooth_log(t) - scale
        return 0.0 if value == -math.inf else math.exp(value)

    previous = None
    for refinement in range(accuracy.max_quadrature_refinements + 1):
        tolerance = 10.0 ** (-7 - 2 * refinement)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            integral, error = quad(scaled, 0, 1, weight="alg",
                                   wvar=(left_power, right_power),
                                   epsabs=tolerance, epsrel=tolerance,
                                   limit=accuracy.quadrature_limit)
        if integral <= 0 or not math.isfinite(integral):
            raise NumericalAccuracyError("Quadrature returned no positive finite mass")
        current = math.log(integral) + scale + log_constant
        discrepancy = math.inf if previous is None else abs(current - previous)
        if (previous is not None and discrepancy <= accuracy.quadrature_log_tolerance
                and error / integral <= accuracy.quadrature_log_tolerance and not caught):
            return IntegrationResult(current, discrepancy, evaluations, True)
        previous = current
    raise NumericalAccuracyError("Quadrature failed its registered convergence budget")


def conditional_minutes_log_prob(recorded: float, maximum: float, alpha: float,
                                  beta: float, atom_probability: float,
                                  kernel: OnceRoundedMinutesKernel,
                                  production_log_prob: Callable[[float], float] | None = None,
                                  accuracy: NumericalAccuracy | None = None) -> IntegrationResult:
    """Score one overtime component, with the latent endpoint atom added exactly.

    Omit production to obtain the marginal recorded-minute pmf from exact
    beta-CDF interval probabilities; no density-times-width approximation.
    """
    accuracy = accuracy or NumericalAccuracy()
    if not all(math.isfinite(x) and x > 0 for x in (alpha, beta)):
        raise ValueError("Beta shapes must be positive and finite")
    if not math.isfinite(atom_probability) or not 0 <= atom_probability <= 1:
        raise ValueError("Endpoint atom probability must lie in [0,1]")
    cell = kernel.cell(recorded, maximum)
    if cell is None:
        return IntegrationResult(-math.inf)
    if maximum == 0:
        logp = 0.0 if production_log_prob is None else _checked_production(production_log_prob, 0.0)
        return IntegrationResult(logp)
    lo, hi = cell[0] / maximum, cell[1] / maximum
    continuous = IntegrationResult(-math.inf)
    if atom_probability < 1:
        if production_log_prob is None:
            mass = _beta_interval_probability(lo, hi, alpha, beta)
            if mass > 0:
                continuous = IntegrationResult(math.log(mass))
            else:
                continuous = _production_integral(lo, hi, maximum, alpha, beta,
                                                  lambda _: 0.0, accuracy)
        else:
            continuous = _production_integral(lo, hi, maximum, alpha, beta,
                                              production_log_prob, accuracy)
    continuous_log = (math.log1p(-atom_probability) + continuous.log_probability
                      if atom_probability < 1 else -math.inf)
    atom_log = -math.inf
    if atom_probability > 0 and kernel.probability(recorded, maximum, maximum):
        atom_log = math.log(atom_probability)
        if production_log_prob is not None:
            atom_log += _checked_production(production_log_prob, maximum)
    return IntegrationResult(float(np.logaddexp(continuous_log, atom_log)),
                             continuous.quadrature_log_discrepancy,
                             continuous.evaluations, continuous.used_quadrature)


def poisson_log_tail_upper_bound(last: int, mean: float) -> float:
    """Upper bound Pr(O > last), preserving tails below floating-point range."""
    if mean == 0:
        return -math.inf
    first = last + 1
    if first + 1 > mean:
        first_log_mass = first * math.log(mean) - mean - gammaln(first + 1)
        return min(0.0, float(first_log_mass - math.log1p(-mean / (first + 1))))
    return 0.0


def marginalize_overtime(recorded: float, games: int, alpha: float, beta: float,
                         atom_probability: float, overtime_rate_per_game: float,
                         kernel: OnceRoundedMinutesKernel,
                         production_log_prob: Callable[[float], float] | None = None,
                         regulation_minutes: float = 40.0, overtime_minutes: float = 5.0,
                         accuracy: NumericalAccuracy | None = None) -> ObservationResult:
    """Unnormalized partial Poisson sums, stopped by a relative log bound.

    The bound is mathematical for exact partial sums.  With quadrature it
    uses the converged numerical partial sum and is labeled uncertified.
    No retained overtime probabilities are ever renormalized.
    """
    accuracy = accuracy or NumericalAccuracy()
    if (isinstance(games, bool) or int(games) != games or games < 1 or
            not math.isfinite(overtime_rate_per_game) or overtime_rate_per_game < 0):
        raise ValueError("Positive integer games and nonnegative finite overtime rate required")
    if not all(math.isfinite(x) and x > 0 for x in (regulation_minutes, overtime_minutes)):
        raise ValueError("Game lengths must be positive and finite")
    mean = games * overtime_rate_per_game
    if not math.isfinite(mean):
        raise ValueError("Overtime mean must be finite")
    if atom_probability == 1:
        possible_o = (recorded - regulation_minutes * games) / overtime_minutes
        if (not math.isfinite(possible_o) or possible_o < 0 or
                not math.isclose(possible_o, round(possible_o), rel_tol=0, abs_tol=1e-10)):
            return ObservationResult(-math.inf, ObservationDiagnostics(
                0, 0.0, 0.0, 0.0, 0, False,
                accuracy_status="structural zero: only exact maximum-minute endpoints have mass"))
    log_total = -math.inf
    discrepancy, evaluations, used_quadrature = 0.0, 0, False
    for overtime in range(accuracy.max_overtime_terms):
        maximum = regulation_minutes * games + overtime_minutes * overtime
        component = conditional_minutes_log_prob(recorded, maximum, alpha, beta,
                                                  atom_probability, kernel,
                                                  production_log_prob, accuracy)
        log_weight = float(poisson.logpmf(overtime, mean))
        log_total = float(np.logaddexp(log_total, log_weight + component.log_probability))
        discrepancy = max(discrepancy, component.quadrature_log_discrepancy)
        evaluations += component.evaluations
        used_quadrature |= component.used_quadrature
        log_tail = poisson_log_tail_upper_bound(overtime, mean)
        if mean == 0 and log_total == -math.inf:
            error_bound = 0.0
        elif log_total == -math.inf:
            error_bound = math.inf
        else:
            error_bound = float(np.logaddexp(0.0, log_tail - log_total))
        if (log_tail <= math.log(accuracy.absolute_tail_tolerance)
                and error_bound <= accuracy.overtime_log_tolerance):
            return ObservationResult(log_total, ObservationDiagnostics(
                overtime + 1, log_tail, error_bound, discrepancy, evaluations, used_quadrature))
    raise NumericalAccuracyError(
        f"Overtime failed its registered budget after {accuracy.max_overtime_terms} terms; "
        f"last log-score bound={error_bound:g}. No truncated result was returned.")
