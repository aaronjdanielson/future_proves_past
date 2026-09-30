"""Intercept-only parameter recovery on simulated observations — milestone 0.2, track 3.

Simulate seasons from known parameters through the reference sampler and the
registered synthetic recording kernel; fit the differentiable intercept-only
model from a perturbed start by L-BFGS on the observed-data negative log
likelihood; report estimates with standard errors from the observed
information; flag weakly identified parameters explicitly; and compare the
fitted and true predictive distributions on fresh simulated data.

Identification. Overtime is never observed, so the rate ν is informed only
by recorded minutes that exceed the regulation capacity; the full-minutes
atom π_M only by recorded values that hit the maximum. Both are weak at the
sample sizes of this project, so the study runs a second configuration with
those parameters held at registered values, and reports the observed
information in both. Weakness is measured, not assumed away: a parameter is
flagged when its standard error exceeds its scale or the information matrix
is near-singular in its direction.

This is a synthetic check of the estimator, not a fitted model of any data.
"""
from __future__ import annotations

import math
import platform
import time

import numpy as np

from ..model.distributions import SeasonParameters
from ..model.likelihood import SeasonModel
from ..model.observation import OnceRoundedMinutesKernel
from ..model.production import COUNT_NAMES, points
from ..serialization import to_jsonable

WEAK_PARAMETERS = ("log_nu", "logit_pi_m")


def simulate(params: SeasonParameters, n: int, schedule: int, rng):
    ref = SeasonModel(params, OnceRoundedMinutesKernel())
    return [ref.sample(schedule, rng) for _ in range(n)]


def _perturbed(model, rng, scale: float):
    """Start away from the truth: multiplicative noise on positives, additive on logits."""
    import torch
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.add_(torch.tensor(rng.normal(0.0, scale, size=p.shape), dtype=p.dtype))


def fixed_overtime_terms(likelihood, batch, margin: int = 4):
    """Per-row overtime term counts: one adaptive pass at the current parameters plus a margin.

    Fixing the counts makes the objective smooth in the parameters (the
    adaptive stop is a control decision); the margin covers movement of the
    parameters during optimization and the counts are re-verified at the end.
    """
    import torch
    with torch.no_grad():
        _, diag = likelihood.log_prob(batch)
    return torch.clamp(diag.overtime_terms_per_row + margin, min=1) * (batch.games > 0).long()


def verify_overtime_terms(likelihood, batch, terms) -> dict:
    """At the optimum, confirm the adaptive rule needs no more terms than were fixed."""
    import torch
    with torch.no_grad():
        _, diag = likelihood.log_prob(batch)
    needed = diag.overtime_terms_per_row
    positive = batch.games > 0
    short = int(((needed > terms) & positive).sum())
    return {"rows_needing_more_terms": short, "max_needed": int(needed.max()),
            "max_fixed": int(terms.max()), "verified": short == 0}


def fit(likelihood, batch, frozen=(), max_iter: int = 200, overtime_terms=None,
        tolerance: float = 1e-9) -> dict:
    """L-BFGS on the summed NLL with per-row fixed overtime term counts."""
    import torch
    model = likelihood.model
    if overtime_terms is None:
        overtime_terms = fixed_overtime_terms(likelihood, batch)
    free = [p for n, p in model.named_parameters() if n not in frozen]
    for n, p in model.named_parameters():
        p.requires_grad_(n not in frozen)
    optimizer = torch.optim.LBFGS(free, lr=1.0, max_iter=max_iter, tolerance_grad=tolerance,
                                  tolerance_change=tolerance, history_size=50,
                                  line_search_fn="strong_wolfe")
    history = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        lp, _ = likelihood.log_prob(batch, overtime_terms=overtime_terms)
        if not bool(torch.isfinite(lp).all()):
            raise ValueError("Fitting requires valid observations with finite likelihood")
        loss = -lp.sum()
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    start = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        final, _ = likelihood.log_prob(batch, overtime_terms=overtime_terms)
    for p in model.parameters():
        p.requires_grad_(True)
    terms_info = (verify_overtime_terms(likelihood, batch, overtime_terms)
                  if isinstance(overtime_terms, torch.Tensor) else {"fixed_int": int(overtime_terms)})
    return {"nll_start": history[0] if history else None, "nll_end": float(-final.sum()),
            "evaluations": len(history), "seconds": elapsed,
            "overtime_terms": ("per_row" if isinstance(overtime_terms, torch.Tensor) else int(overtime_terms)),
            "overtime_terms_check": terms_info, "frozen": list(frozen)}


def observed_information(likelihood, batch, overtime_terms=None, eps: float = 1e-4) -> dict:
    """Hessian of the NLL by central differences of autodiff gradients.

    Second-order autodiff through the quadrature is possible but slow; a
    symmetric difference of first-order gradients is adequate for standard
    errors and is checked for symmetry.
    """
    import torch
    model = likelihood.model
    if overtime_terms is None:
        overtime_terms = fixed_overtime_terms(likelihood, batch)
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    sizes = [p.numel() for p in params]
    flat = torch.cat([p.detach().flatten() for p in params]).clone()

    def gradient(vec):
        with torch.no_grad():
            offset = 0
            for p, n in zip(params, sizes):
                p.copy_(vec[offset:offset + n].view_as(p))
                offset += n
        model.zero_grad(set_to_none=True)
        lp, _ = likelihood.log_prob(batch, overtime_terms=overtime_terms)
        (-lp.sum()).backward()
        return torch.cat([p.grad.detach().flatten() for p in params]).clone()

    d = flat.numel()
    H = torch.zeros(d, d, dtype=flat.dtype)
    for i in range(d):
        plus, minus = flat.clone(), flat.clone()
        plus[i] += eps
        minus[i] -= eps
        H[i] = (gradient(plus) - gradient(minus)) / (2 * eps)
    gradient(flat)  # restore parameters
    asym = float((H - H.T).abs().max() / H.abs().max().clamp(min=1e-12))
    H = 0.5 * (H + H.T)
    labels = []
    for name, n in zip(names, sizes):
        labels += [name if n == 1 else f"{name}[{k}]" for k in range(n)]
    return {"hessian": H, "labels": labels, "asymmetry": asym}


def standard_errors(info: dict, free_labels) -> dict:
    """Invert the information restricted to free parameters; report conditioning."""
    import torch
    H, labels = info["hessian"], info["labels"]
    idx = [i for i, lab in enumerate(labels) if lab.split("[")[0] in free_labels]
    sub = H[idx][:, idx]
    eig = torch.linalg.eigvalsh(sub)
    result = {"min_eigenvalue": float(eig.min()), "max_eigenvalue": float(eig.max()),
              "condition_number": float(eig.max() / eig.min()) if float(eig.min()) > 0 else None,
              "positive_definite": bool(float(eig.min()) > 0)}
    if result["positive_definite"]:
        cov = torch.linalg.inv(sub)
        se = torch.sqrt(torch.clamp(torch.diagonal(cov), min=0.0))
        result["se"] = {labels[i]: float(s) for i, s in zip(idx, se)}
    else:
        result["se"] = {labels[i]: None for i in idx}   # not invertible: no standard errors
    return result


WEAK_SE_ABSOLUTE = 0.5      # log/logit scale: a factor of e^0.5 ≈ 1.6 in the natural parameter
WEAK_SE_RELATIVE = 5.0      # times the median standard error of the free parameters


def parameter_table(model, truth_model, se: dict, frozen) -> list[dict]:
    import torch
    usable_ses = [s for s in se.values() if s is not None and math.isfinite(s) and s > 0]
    median_se = float(np.median(usable_ses)) if usable_ses else float("inf")
    rows = []
    for (name, p), (_, t) in zip(model.named_parameters(), truth_model.named_parameters()):
        for k in range(p.numel()):
            label = name if p.numel() == 1 else f"{name}[{k}]"
            est, true = float(p.detach().flatten()[k]), float(t.detach().flatten()[k])
            s = se.get(label) if name not in frozen else None
            usable = s is not None and math.isfinite(s) and s > 0
            z = (est - true) / s if usable else None
            # Weak: no usable SE, or an SE that is large in absolute terms or
            # relative to the other parameters. Recovery within the SE does
            # not make a parameter well identified.
            weak = ((name in frozen) or (not usable)
                    or (s > WEAK_SE_ABSOLUTE) or (s > WEAK_SE_RELATIVE * median_se))
            rows.append({"parameter": label, "true": true, "estimate": est,
                         "se": s if usable else None, "z": z,
                         "frozen": name in frozen,
                         "identification": "constrained" if name in frozen else ("weak" if weak else "identified")})
    return rows


def predictive_comparison(fitted: SeasonParameters, truth: SeasonParameters, n: int, schedule: int, rng) -> dict:
    """Fresh data from the truth: NLL under fitted vs true parameters, plus marginal moments."""
    from ..model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch
    import torch
    test = simulate(truth, n, schedule, rng)
    batch = SeasonBatch.from_outcomes(test, schedule)
    out = {}
    for label, params in (("true", truth), ("fitted", fitted)):
        lik = DifferentiableSeasonLikelihood(InterceptOnlySeasonModel(params))
        with torch.no_grad():
            lp, _ = lik.log_prob(batch)
        out[f"mean_nll_{label}"] = float(-lp.mean())
    out["excess_nll_fitted_minus_true"] = out["mean_nll_fitted"] - out["mean_nll_true"]
    draws = simulate(fitted, n, schedule, rng)

    def moments(items):
        g = np.array([y.games for y in items], dtype=float)
        m = np.array([y.recorded_minutes for y in items], dtype=float)
        p = np.array([points(y.counts) for y in items], dtype=float)
        return {"zero_game_fraction": float((g == 0).mean()), "mean_games": float(g.mean()),
                "mean_recorded_minutes": float(m.mean()), "sd_recorded_minutes": float(m.std()),
                "mean_points": float(p.mean()), "sd_points": float(p.std()),
                "full_minutes_fraction": float(np.mean([y.games > 0 and y.recorded_minutes >= 40 * y.games for y in items])),
                "overtime_fraction": float(np.mean([y.recorded_minutes > 40 * y.games for y in items]))}
    out["moments_true_data"] = moments(test)
    out["moments_fitted_draws"] = moments(draws)
    out["n_test"] = n
    return out


def run_recovery(seed: int = 20260901, n_train: int = 400, n_test: int = 300, schedule: int = 32,
                 nodes: int = 32, start_scale: float = 0.3, max_iter: int = 200,
                 params: SeasonParameters | None = None) -> dict:
    import torch
    from ..model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch
    truth = params or SeasonParameters()
    rng = np.random.default_rng(seed)
    train = simulate(truth, n_train, schedule, rng)
    batch = SeasonBatch.from_outcomes(train, schedule)
    report = {"kind": "intercept_only_recovery_study", "seed": seed, "n_train": n_train,
              "n_test": n_test, "schedule": schedule, "quadrature_nodes": nodes,
              "start_scale": start_scale, "empirical_transfer_gain": None, "trained_model": False,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "torch": torch.__version__},
              "kernel_provenance": to_jsonable(OnceRoundedMinutesKernel().provenance),
              "data_summary": {
                  "zero_game_fraction": float(np.mean([y.games == 0 for y in train])),
                  "full_minutes_rows": int(sum(y.games > 0 and y.recorded_minutes >= 40 * y.games for y in train)),
                  "overtime_rows": int(sum(y.recorded_minutes > 40 * y.games for y in train))},
              "configurations": {}}
    truth_model = InterceptOnlySeasonModel(truth)
    for label, frozen in (("all_free", ()), ("weak_constrained", WEAK_PARAMETERS)):
        model = InterceptOnlySeasonModel(truth)
        _perturbed(model, np.random.default_rng(seed + 1), start_scale)
        if frozen:  # constrained parameters are held at their registered (true) values
            with torch.no_grad():
                for (name, p), (_, t) in zip(model.named_parameters(), truth_model.named_parameters()):
                    if name in frozen:
                        p.copy_(t)
        lik = DifferentiableSeasonLikelihood(model, quadrature_nodes=nodes)
        fit_result = fit(lik, batch, frozen=frozen, max_iter=max_iter)
        info = observed_information(lik, batch)
        free = [n for n, _ in model.named_parameters() if n not in frozen]
        ses = standard_errors(info, free)
        table = parameter_table(model, truth_model, ses["se"], frozen)
        with torch.no_grad():
            nll_truth, _ = lik.__class__(truth_model, quadrature_nodes=nodes).log_prob(batch)
        predictive = predictive_comparison(model.reference(), truth, n_test, schedule, np.random.default_rng(seed + 2))
        report["configurations"][label] = {
            "fit": fit_result, "train_nll_at_truth": float(-nll_truth.sum()),
            "information": {k: v for k, v in ses.items() if k != "se"},
            "hessian_asymmetry": info["asymmetry"],
            "parameters": table,
            "n_weak": sum(r["identification"] == "weak" for r in table),
            "max_abs_z_identified": max((abs(r["z"]) for r in table if r["identification"] == "identified"
                                         and r["z"] is not None), default=None),
            "predictive": predictive}
    return report
