"""Fold report for a finished arm run (D-028).

Reads a run directory (``run.json``, ``per_unit.csv``, ``tower_state.pt``),
rebuilds the fold's evaluation examples from the verified tables, and
reports what a model comparison needs:

* population — training counts by task and the held-out cohorts;
* the primary paired comparison — per-player NLL difference of a second
  run on the same fold (e.g. the aggregated baseline) minus this run, with
  standard errors by cohort;
* the forecast definition — the realized-schedule scenario label carried
  in the run configuration;
* likelihood decomposition — participation hurdle, starts, and the rest,
  with and without the provider-consistent zero-appearance labels;
* supporting performance — games, minutes, and points absolute errors of
  the predictive mean, 90% interval coverage, and participation calibration
  (reliability by decile, Brier score), from draws of the fitted season
  distribution;
* numerical validation — SciPy reference rescoring of random units and of
  the largest disagreements, plus the quadrature and kernel-sensitivity
  diagnostics recorded by the run;
* reproducibility — table digests, configuration, checkpoint hash, seed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.folds import Fold, player_index
from ..data.manifests import file_sha256, load_manifest, write_manifest
from ..data.tables import read_tables
from ..model import features as F
from ..model.baseline import AggregateBaseline
from ..model.dataset import FoldDataset, cohort_masks
from ..model.distributions import SeasonParameters
from ..model.likelihood import SeasonModel, SeasonOutcome
from ..model.production import COUNT_NAMES, MAKE_ATTEMPTS, RATE_NAMES, ProductionParameters
from ..model.torch_likelihood import DifferentiableSeasonLikelihood
from ..model.tower import ReferenceTower
from .train import STRATA, _batches, _score, season_batch

ASSUMED_ZERO = FoldDataset.ASSUMED_ZERO


def source_digests() -> dict:
    """sha256 of every package source file, for the reproducibility block (the project is not a git repository)."""
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root.parent)): file_sha256(p) for p in sorted(root.rglob("*.py"))}


def row_parameters(params: dict, i: int) -> SeasonParameters:
    """SeasonParameters for row i of a per-row parameter dict (for the SciPy reference and sampler)."""
    g = lambda k: float(params[k][i])  # noqa: E731
    prod = ProductionParameters(
        rates=dict(zip(RATE_NAMES, params["rates"][i].tolist())),
        dispersions=dict(zip(RATE_NAMES, params["dispersions"][i].tolist())),
        makes_alpha=dict(zip(MAKE_ATTEMPTS, params["makes_alpha"][i].tolist())),
        makes_beta=dict(zip(MAKE_ATTEMPTS, params["makes_beta"][i].tolist())))
    return SeasonParameters(participation_probability=g("pi_g"), games_alpha=g("alpha_g"), games_beta=g("beta_g"),
                            starts_alpha=g("alpha_j"), starts_beta=g("beta_j"), overtime_rate_per_game=g("nu"),
                            minutes_alpha=g("alpha_m"), minutes_beta=g("beta_m"), minutes_max_probability=g("pi_m"),
                            production=prod)


def reliability(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> dict:
    """Calibration of a probability forecast: mean predicted vs observed rate by quantile bin, and the Brier score."""
    order = np.argsort(prob)
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            rows.append({"n": int(len(chunk)), "predicted": float(prob[chunk].mean()), "observed": float(outcome[chunk].mean())})
    return {"bins": rows, "brier": float(np.mean((prob - outcome) ** 2)), "base_rate": float(outcome.mean())}


def interval_coverage(lo: np.ndarray, hi: np.ndarray, observed: np.ndarray) -> float:
    return float(np.mean((observed >= lo) & (observed <= hi)))


def _paired(delta: np.ndarray, masks: dict) -> dict:
    out = {}
    for name in ["all"] + STRATA:
        m = np.ones(len(delta), dtype=bool) if name == "all" else masks[name]
        m = m & np.isfinite(delta)
        n = int(m.sum())
        out[name] = {"n": n, "mean": float(delta[m].mean()) if n else None,
                     "se": float(delta[m].std(ddof=1) / math.sqrt(n)) if n > 1 else None,
                     "share_improved": float((delta[m] > 0).mean()) if n else None}
    return out


SHOOTING = (("K2", "A2"), ("K3", "A3"), ("Kf", "Af"))
RATE_FAMILIES = {"attempts": ("A2", "A3", "Af"), "rebounds": ("ORB", "DRB"), "assists": ("AST",), "turnovers": ("TOV",),
                 "steals_blocks": ("STL", "BLK"), "fouls": ("PF",)}


def component_diagnostics(lik, examples, model, params_all, units, masks, *, batch_size: int, min_minutes: float = 40.0,
                          cohorts_only: bool = False) -> dict:
    """Declared component diagnostics: exact opportunity / production decomposition of the marginal
    score, makes-given-attempts terms, shooting proportions with attempts as evidence, per-40-minute
    rates (per-minute denominators; possessions and available rebounds are not observed per player),
    assists and turnovers individually and as a ratio, and the conditional expectation of counts at the
    observed exposure (a labeled diagnostic, never a forecast input)."""
    from ..model.torch_likelihood import beta_binomial_logpmf
    from scipy.stats import betabinom
    idx = {n: i for i, n in enumerate(COUNT_NAMES)}
    total, opp, makes, latent = [], [], [], []
    with torch.no_grad():
        for group in _batches(examples, batch_size, None):
            batch = F.collate(group)
            sb = season_batch(batch)
            sb_m = sb.__class__(sb.games, sb.starts, sb.recorded_minutes, sb.counts, sb.schedule, None, None)
            p = model(batch)
            t, _ = lik.log_prob(sb_m, params=p)
            o = lik.log_prob_opportunity(sb_m, params=p)
            m = torch.zeros_like(t)
            for j, (make, attempt) in enumerate(SHOOTING):
                m = m + beta_binomial_logpmf(sb.counts[:, idx[make]], sb.counts[:, idx[attempt]], p["makes_alpha"][:, j], p["makes_beta"][:, j])
            total.append(t); opp.append(o); makes.append(m)
            latent.append(lik.latent_minutes_given_opportunity(sb_m, params=p))
    total, opp, makes, latent = (torch.cat(x).numpy() for x in (total, opp, makes, latent))
    finite = np.isfinite(total) & np.isfinite(opp)
    production_given = total - opp
    volume_other = production_given - makes            # attempts and non-shooting counts given opportunity
    by_cohort = {}
    for name in ["all"] + STRATA:
        mk = finite & (np.ones(len(total), bool) if name == "all" else masks[name])
        n = int(mk.sum())
        entry = {"n": n, "nll_total": float(-total[mk].mean()) if n else None,
                 "nll_opportunity": float(-opp[mk].mean()) if n else None,
                 "nll_production_given_opportunity": float(-production_given[mk].mean()) if n else None,
                 "nll_makes_given_attempts": float(-makes[mk].mean()) if n else None,
                 "nll_attempts_and_other_given_opportunity": float(-volume_other[mk].mean()) if n else None}
        if n:
            entry["sum_check_abs"] = abs(entry["nll_opportunity"] + entry["nll_production_given_opportunity"] - entry["nll_total"])
        by_cohort[name] = entry
    if cohorts_only:
        return {"by_cohort": by_cohort, "per_unit": {"total": total, "opportunity": opp, "production_given_opportunity": production_given}}
    if params_all is None:
        raise ValueError("Full diagnostics need the per-row parameters")
    # Shooting: proportions weighted by attempts, and PIT of makes under the fitted beta-binomial.
    counts = units[F.TARGET_COUNTS].values.astype(float)
    shooting = {}
    rng = np.random.default_rng(0)
    for j, (make, attempt) in enumerate(SHOOTING):
        k, a = counts[:, idx[make]], counts[:, idx[attempt]]
        alpha, beta = params_all["makes_alpha"][:, j], params_all["makes_beta"][:, j]
        q = alpha / (alpha + beta)
        has = a > 0
        pit = np.full(len(k), np.nan)
        lo = betabinom.cdf(k[has] - 1, a[has], alpha[has], beta[has])
        hi = betabinom.cdf(k[has], a[has], alpha[has], beta[has])
        pit[has] = lo + rng.random(int(has.sum())) * (hi - lo)               # randomized PIT for a discrete outcome
        for name in ("all", "intl_entrant"):
            mk = has & (np.ones(len(k), bool) if name == "all" else masks[name])
            if not mk.any():
                continue
            w = a[mk]
            shooting[f"{make}/{attempt}:{name}"] = {
                "units_with_attempts": int(mk.sum()), "attempts": float(w.sum()),
                "observed_pct": float(k[mk].sum() / w.sum()), "predicted_pct_attempt_weighted": float((q[mk] * w).sum() / w.sum()),
                "mae_pct_attempt_weighted": float((np.abs(k[mk] / w - q[mk]) * w).sum() / w.sum()),
                "pit_mean": float(pit[mk].mean()), "pit_share_below_0.05": float((pit[mk] < 0.05).mean()),
                "pit_share_above_0.95": float((pit[mk] > 0.95).mean())}
    # Per-40-minute rates on units with enough exposure, and assists/turnovers individually and as a ratio.
    minutes = units["minutes"].values.astype(float)
    exposure = minutes / 40.0                    # recorded minutes: the plug-in exposure (approximate)
    exact_exposure = latent / 40.0               # E[M* | G, J, M, x] / 40: the model's exact conditional exposure
    enough = minutes >= min_minutes
    rate_index = {n: i for i, n in enumerate(RATE_NAMES)}
    rates = {}
    for fam, names in RATE_FAMILIES.items():
        obs = sum(counts[:, idx[n]] for n in names)
        pred = sum(params_all["rates"][:, rate_index[n]] for n in names)
        for name in ("all", "intl_entrant"):
            mk = enough & (np.ones(len(obs), bool) if name == "all" else masks[name])
            if not mk.any():
                continue
            rates[f"{fam}:{name}"] = {"n": int(mk.sum()), "denominator": "per 40 recorded minutes",
                                      "observed_per40_mean": float((obs[mk] / exposure[mk]).mean()),
                                      "predicted_per40_mean": float(pred[mk].mean()),
                                      "mae_per40": float(np.abs(obs[mk] / exposure[mk] - pred[mk]).mean()),
                                      "conditional_count_mae_exact": float(np.abs(obs[mk] - pred[mk] * exact_exposure[mk]).mean()),
                                      "conditional_count_mae_plugin_recorded_minutes": float(np.abs(obs[mk] - pred[mk] * exposure[mk]).mean())}
    ast, tov = counts[:, idx["AST"]], counts[:, idx["TOV"]]
    p_ast, p_tov = params_all["rates"][:, rate_index["AST"]] * exact_exposure, params_all["rates"][:, rate_index["TOV"]] * exact_exposure
    ratio = {}
    for name in ("all", "intl_entrant"):
        mk = enough & (np.ones(len(ast), bool) if name == "all" else masks[name])
        if not mk.any():
            continue
        zero_to = mk & (tov == 0)
        pos = mk & (tov > 0)
        ratio[name] = {"n": int(mk.sum()), "zero_turnover_units": int(zero_to.sum()),
                       "observed_ast_to_ratio_mean_tov_gt0": float((ast[pos] / tov[pos]).mean()) if pos.any() else None,
                       "predicted_ast_to_ratio_mean_tov_gt0": float((p_ast[pos] / p_tov[pos]).mean()) if pos.any() else None,
                       "ratio_definition": "AST/TOV on units with TOV > 0; predicted ratio from expected counts at observed exposure"}
    return {"by_cohort": by_cohort, "shooting": shooting, "rates_per40": rates, "assists_turnovers": ratio,
            "per_unit": {"total": total, "opportunity": opp, "production_given_opportunity": production_given},
            "notes": ["opportunity = participation + games|participation + starts + recorded minutes|games,overtime with production integrated out",
                      "production|opportunity = total − opportunity (exact; respects the recording cell)",
                      "makes|attempts terms are minute-independent and exact; the remainder is attempts and other counts given opportunity",
                      "conditional_count_mae_exact uses E[M*|G,J,M,x] (latent minutes integrated over the recording cell and overtime, production excluded from the weights); the plug-in version with recorded minutes is approximate; per-40 observed rates use recorded minutes",
                      "both terms of the opportunity/production identity use the once-rounded kernel and marginalized overtime"],
            "expected_latent_minutes_mean": float(latent[np.isfinite(latent)].mean())}


def _rebuild(run_payload: dict, tables_dir: Path):
    tab = read_tables(tables_dir)
    units = tab["units"]
    protocol_test = int(units["season"].max())
    test_players = frozenset(units.loc[(units["season"] == protocol_test) & units["intl_entrant"], "player_id"])
    dobs = F.player_dobs(tab)
    cfg = run_payload["config"]
    fs = F.schema_from_config(cfg, units)                        # the run's feature schema (D-065)
    store = F.GameStore.from_frame(tab["games"], dobs=dobs, schema=fs)
    fold_season = int(run_payload["fold"])
    fold = Fold(fold_season, "development" if fold_season <= 2023 else "selection")
    rw = cfg.get("reconstruction_windows") or {}
    ds = FoldDataset.build(tab, fold, store, test_cohort_players=test_players, index=player_index(tab["games"]),
                           seed=int(run_payload["seed"]), exclude_assumed_zeros=bool(cfg.get("exclude_assumed_zeros", False)),
                           k_values=tuple(rw.get("k_values", (1, 2, 3))), max_years=rw.get("max_years", 4),
                           exclude_post_first_season=str(cfg.get("post_first_season_logs", "included")).startswith("excluded"),
                           label_scope=cfg.get("label_scope", "all"),
                           weighting=cfg.get("weighting", "player_balanced"), first_year_share=cfg.get("first_year_share"))
    vocab_sizes = [len(store.vocab[c]) for c in F.CATEGORICAL]
    static_sizes = [len(v) for v in ds.static_vocab.values()]
    kind = run_payload.get("model", "tower")
    pretrained = run_payload.get("config", {}).get("pretraining") is not None
    from ..model.pool import CONTEXT_FIELDS
    widths = {"n_fields": len(fs.field_names), "n_static": len(fs.static_continuous)}
    model = (ReferenceTower(vocab_sizes, static_sizes, SeasonParameters(), n_context=len(CONTEXT_FIELDS) if pretrained else 0, **widths)
             if kind == "tower" else AggregateBaseline(vocab_sizes, static_sizes, SeasonParameters(), **widths))
    return tab, ds, model


def report_run(*, run: Path, tables: Path, compare: Path | None = None, draws: int = 100, n_rescore: int = 25,
               batch_size: int = 256, seed: int = 20260901) -> dict:
    run = Path(run)
    payload = load_manifest(run / "run.json")
    per_unit = pd.read_csv(run / "per_unit.csv")
    tab, ds, model = _rebuild(payload, tables)
    state = torch.load(run / "tower_state.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    examples = ds.evaluation_examples()
    units = ds.eval_units.reset_index(drop=True)
    assert list(units["unit_id"]) == list(per_unit["unit_id"]), "run per_unit.csv does not match the rebuilt evaluation set"
    masks = cohort_masks(units)
    lik = DifferentiableSeasonLikelihood(None, quadrature_nodes=int(payload["numerical"]["eval_quadrature"]["nodes"])
                                         if "numerical" in payload else 32)
    # Per-row parameters and likelihood components.
    PARAM_KEYS = ("pi_g", "alpha_g", "beta_g", "alpha_j", "beta_j", "nu", "alpha_m", "beta_m", "pi_m",
                  "rates", "dispersions", "makes_alpha", "makes_beta")
    COMP_KEYS = ("total", "participation", "games_given_participation", "starts", "rest")
    params_all = {k: [] for k in PARAM_KEYS}
    comps = {k: [] for k in COMP_KEYS}
    from .train import marginal_batch
    conditioned_total = []
    with torch.no_grad():
        for group in _batches(examples, batch_size, None):
            batch = F.collate(group)
            p = model(batch)
            for k in params_all:
                params_all[k].append(p[k].detach())
            sb = season_batch(batch)
            # Every block of the report uses the reported convention: overtime marginalized (D-029, D-034).
            c = lik.log_prob_components(marginal_batch(sb), params=p)
            for k in comps:
                comps[k].append(c[k])
            conditioned_total.append(lik.log_prob(sb, params=p)[0])
    params_all = {k: torch.cat(v).numpy() for k, v in params_all.items()}
    comps = {k: torch.cat(v).numpy() for k, v in comps.items()}
    nll = -comps["total"]                                   # marginal, identical to _score(..., marginal=True)
    nll_conditioned = -torch.cat(conditioned_total).numpy()   # training convention, reported only for reference
    finite = np.isfinite(nll)
    base_params = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in payload["baseline"]["parameters"].items()}
    nll_baseline = _score(lik, examples, batch_size=batch_size, params=base_params, marginal=True)
    # The run scored these units with the same checkpoint; the rebuilt scores must reproduce them. Runs that
    # predate the marginal score definition stored the conditioned score, so the check is against that.
    run_nll = per_unit["nll_tower"].values
    run_was_conditioned = "nll_tower_conditioned_overtime" not in per_unit
    rebuild_gap = float(np.abs((nll_conditioned if run_was_conditioned else nll)[finite] - run_nll[finite]).max())
    parts_sum = comps["participation"] + comps["games_given_participation"] + comps["starts"] + comps["rest"]
    sum_check = float(np.abs(parts_sum[finite] - comps["total"][finite]).max())
    assumed = (units["evidence"].astype("string") == ASSUMED_ZERO).to_numpy(dtype=bool, na_value=False)
    decomposition = {"convention": "overtime marginalized (same as every other block)", "sum_check_max_abs": sum_check,
                     "rebuild_vs_run_max_abs": rebuild_gap}
    for label, m in (("all", finite), ("without_assumed_zeros", finite & ~assumed), ("assumed_zeros_only", finite & assumed),
                     ("intl_entrant", finite & masks["intl_entrant"])):
        decomposition[label] = {"n": int(m.sum()), **{k: float((-comps[k][m]).mean()) if m.any() else None for k in COMP_KEYS}}
    # Component diagnostics (D-032), declared before any B/C result: exact decomposition of the marginal score
    # into opportunity (participation, games given participation, starts, recorded minutes given games and
    # overtime with production integrated out) and the counts given the observed opportunity; the
    # makes-given-attempts terms separately (they do not depend on minutes); per-40-minute rate and shooting
    # summaries; and the conditional expectation of counts at the observed exposure.
    compare_components = None
    other_params = None
    components = component_diagnostics(lik, examples, model, params_all, units, masks, batch_size=batch_size)
    per_unit_components = components.pop("per_unit")
    # Paired comparison against a second run on the same fold: its checkpoint is rescored on the same
    # examples with the same (marginal) definition, so both sides are like for like.
    paired = None
    delta_disagreement = None
    nll_compare = None
    if compare is not None:
        compare = Path(compare)
        other_payload = load_manifest(compare / "run.json")
        if int(other_payload["fold"]) != int(payload["fold"]) or int(other_payload["seed"]) != int(payload["seed"]):
            raise ValueError("The comparison run must share the fold and seed (same evaluation examples and normalizer)")
        _, ds_other, other_model = _rebuild(other_payload, tables)
        # The comparison checkpoint is scored on inputs built with ITS OWN normalizer and vocabularies
        # (preprocessing is part of the model), on the same units, aligned by unit id (D-034 addendum).
        other_examples = ds_other.evaluation_examples()
        if list(ds_other.eval_units["unit_id"]) != list(units["unit_id"]):
            raise ValueError("The comparison run's evaluation units differ from this run's")
        other_model.load_state_dict(torch.load(compare / "tower_state.pt", weights_only=True))
        other_model.eval()
        nll_compare = _score(lik, other_examples, batch_size=batch_size, tower=other_model, marginal=True)
        delta = nll_compare - nll
        paired = {"compare_run": str(compare), "compare_model": other_payload.get("model", "tower"), "compare_arm": other_payload.get("arm"),
                  "definition": "marginal nll(compare) - marginal nll(this run) per player-season; positive favours this run",
                  **_paired(delta, masks)}
        delta_disagreement = np.abs(delta)
        compare_components = component_diagnostics(lik, other_examples, other_model, None, units, masks, batch_size=batch_size, cohorts_only=True)
        other_params = {k: [] for k in PARAM_KEYS}
        with torch.no_grad():
            for group in _batches(other_examples, batch_size, None):
                p = other_model(F.collate(group))
                for k in other_params:
                    other_params[k].append(p[k].detach())
        other_params = {k: torch.cat(v).numpy() for k, v in other_params.items()}
    baseline_paired = _paired(nll_baseline - nll, masks)
    # Paired component differences against the comparison model (compare − this run), per cohort.
    component_paired = None
    if compare_components is not None:
        other = compare_components.pop("per_unit")
        component_paired = {}
        for key in ("total", "opportunity", "production_given_opportunity"):
            d = other[key] - per_unit_components[key]          # log-probability difference; positive = this run better
            component_paired[key] = _paired(-d, masks)          # in NLL terms: nll(compare) − nll(this)
        components["compare_by_cohort"] = compare_components["by_cohort"]
        components["paired_compare_minus_this"] = component_paired
    # Reconciliation (D-034): on the rows where both scores are finite, the paired mean equals the
    # difference of the two displayed means, and each model's opportunity + production sums to its total.
    reconciliation = {}
    for name in ["all"] + STRATA:
        m = finite & (np.ones(len(nll), bool) if name == "all" else masks[name])
        if nll_compare is not None:
            m = m & np.isfinite(nll_compare)
        entry = {"n": int(m.sum()), "mean_this_run": float(nll[m].mean()) if m.any() else None}
        if nll_compare is not None and m.any():
            entry.update({"mean_compare": float(nll_compare[m].mean()), "paired_mean_compare_minus_this": float((nll_compare - nll)[m].mean()),
                          "difference_of_means": float(nll_compare[m].mean() - nll[m].mean())})
            entry["reconciles"] = abs(entry["paired_mean_compare_minus_this"] - entry["difference_of_means"]) < 1e-9
        reconciliation[name] = entry
    # Participation calibration and predictive summaries from the fitted season distribution.
    rng = np.random.default_rng(seed)
    observed_zero = (units["games_played"].values == 0).astype(float)
    p_zero = 1.0 - params_all["pi_g"]
    calibration = reliability(p_zero, observed_zero)
    schedules = units["scheduled_games"].values.astype(int)

    def predictive(params, seed_offset):
        rr = np.random.default_rng(seed + seed_offset)
        pred = {"games": [], "minutes": [], "points": []}
        lo = {k: [] for k in pred}
        hi = {k: [] for k in pred}
        for i in range(len(units)):
            sm = SeasonModel(row_parameters(params, i))
            samples = [sm.sample(int(schedules[i]), rr) for _ in range(draws)]
            for k, values in (("games", [s.games for s in samples]), ("minutes", [s.recorded_minutes for s in samples]),
                              ("points", [s.points for s in samples])):
                arr = np.asarray(values, dtype=float)
                pred[k].append(arr.mean())
                lo[k].append(np.quantile(arr, 0.05))
                hi[k].append(np.quantile(arr, 0.95))
        return pred, lo, hi

    pred, lo, hi = predictive(params_all, 0)
    observed = {"games": units["games_played"].values.astype(float), "minutes": units["minutes"].values.astype(float),
                "points": units["points"].values.astype(float)}
    # Held-out player predictions side by side (D-035): observed outcome, this model, and the comparison model.
    table = pd.DataFrame({"unit_id": units["unit_id"].values, "player_id": units["player_id"].values,
                          **{k: v for k, v in masks.items()},
                          "obs_games": observed["games"], "obs_minutes": observed["minutes"], "obs_points": observed["points"],
                          "nll_this": nll})
    counts_obs = units[F.TARGET_COUNTS].values.astype(float)
    ci = {n: i for i, n in enumerate(COUNT_NAMES)}
    with np.errstate(divide="ignore", invalid="ignore"):
        table["obs_2p_pct"] = counts_obs[:, ci["K2"]] / counts_obs[:, ci["A2"]]
        table["obs_3p_pct"] = counts_obs[:, ci["K3"]] / counts_obs[:, ci["A3"]]
        table["obs_ast_per40"] = counts_obs[:, ci["AST"]] / (observed["minutes"] / 40.0)
        table["obs_tov_per40"] = counts_obs[:, ci["TOV"]] / (observed["minutes"] / 40.0)
    ri = {n: i for i, n in enumerate(RATE_NAMES)}

    def attach(tag, params, p_, l_, h_):
        for k in p_:
            table[f"{tag}_{k}_mean"] = p_[k]
            table[f"{tag}_{k}_lo90"] = l_[k]
            table[f"{tag}_{k}_hi90"] = h_[k]
        table[f"{tag}_2p_pct"] = params["makes_alpha"][:, 0] / (params["makes_alpha"][:, 0] + params["makes_beta"][:, 0])
        table[f"{tag}_3p_pct"] = params["makes_alpha"][:, 1] / (params["makes_alpha"][:, 1] + params["makes_beta"][:, 1])
        table[f"{tag}_ast_per40"] = params["rates"][:, ri["AST"]]
        table[f"{tag}_tov_per40"] = params["rates"][:, ri["TOV"]]
        table[f"{tag}_p_zero_games"] = 1.0 - params["pi_g"]

    attach("this", params_all, pred, lo, hi)
    if other_params is not None:
        pred_c, lo_c, hi_c = predictive(other_params, 1)
        attach("compare", other_params, pred_c, lo_c, hi_c)
        table["nll_compare"] = nll_compare
    table.to_csv(run / "per_player_predictions.csv", index=False)
    supporting = {}
    for k in pred:
        p_, l_, h_ = map(np.asarray, (pred[k], lo[k], hi[k]))
        supporting[k] = {"mae": float(np.abs(p_ - observed[k]).mean()),
                         "mae_intl_entrant": float(np.abs(p_ - observed[k])[masks["intl_entrant"]].mean()) if masks["intl_entrant"].any() else None,
                         "coverage_90": interval_coverage(l_, h_, observed[k]),
                         "coverage_90_intl_entrant": interval_coverage(l_[masks["intl_entrant"]], h_[masks["intl_entrant"]], observed[k][masks["intl_entrant"]]) if masks["intl_entrant"].any() else None}
    # Reference rescoring: random units and the largest disagreements. The SciPy reference marginalizes
    # overtime, so both sides are compared in marginalized mode at identical per-row parameters, kernel,
    # and realized schedule; the run's conditioned score (observed overtime) is reported alongside.
    order = np.argsort(-(delta_disagreement if delta_disagreement is not None else np.abs(nll_baseline - nll)))
    chosen = sorted(set(rng.choice(np.flatnonzero(finite), min(n_rescore, int(finite.sum())), replace=False).tolist()) | set(order[:n_rescore].tolist()))
    sub_examples = [examples[i] for i in chosen]
    sub_batch = F.collate(sub_examples)
    sb = season_batch(sub_batch)
    sb_marginal = sb.__class__(sb.games, sb.starts, sb.recorded_minutes, sb.counts, sb.schedule, None, None)
    sub_params = {k: torch.as_tensor(params_all[k][chosen], dtype=torch.float64) for k in PARAM_KEYS}
    with torch.no_grad():
        torch_marginal, _ = lik.log_prob(sb_marginal, params=sub_params)
    rescored = []
    for j, i in enumerate(chosen):
        u = units.iloc[i]
        outcome = SeasonOutcome(int(u["games_played"]), int(u["starts"]), float(round(u["minutes"])),
                                {n: int(u[c]) for n, c in zip(COUNT_NAMES, F.TARGET_COUNTS)})
        ref = SeasonModel(row_parameters(params_all, i), lik.kernel, lik.accuracy).log_prob(outcome, int(schedules[i]))
        tm = float(torch_marginal[j])
        rescored.append({"unit_id": u["unit_id"], "torch_marginalized": tm, "reference_marginalized": float(ref),
                         "torch_conditioned_on_observed_overtime": float(comps["total"][i]), "reported_marginal_nll": float(nll[i]),
                         "overtime_known": bool(units.iloc[i]["overtime_known"]),
                         "abs_diff": float(abs(ref - tm)) if np.isfinite(ref) and np.isfinite(tm) else None})
    diffs = [r["abs_diff"] for r in rescored if r["abs_diff"] is not None]
    numerical = {"reference_rescoring": {"n": len(rescored), "comparison": "torch vs SciPy, both marginalizing overtime, same per-row parameters, once-rounded kernel, realized S",
                                         "max_abs_diff_nats": max(diffs) if diffs else None,
                                         "mean_abs_diff_nats": float(np.mean(diffs)) if diffs else None, "rows": rescored},
                 "components_sum_check_max_abs": sum_check, "rebuild_vs_run_max_abs": rebuild_gap,
                 "from_run": payload.get("numerical")}
    counts = payload["counts"]
    population = {"training": {"forward_units": counts["train_forward"], "reconstruction_windows": counts["train_reconstruction"],
                               "stop_units": counts["stop_units"]},
                  "evaluation": {name: int(masks[name].sum()) for name in STRATA} | {"all": int(len(units))}}
    games_cols = set(tab["games"].columns)
    reproducibility = {"tables": str(tables), "tables_manifest_sha256": tab["manifest"].get("payload_sha256"),
                       "table_digests": {k: v["sha256"] for k, v in tab["manifest"]["tables"].items()},
                       "checkpoint_sha256": file_sha256(run / "tower_state.pt"), "config": payload["config"],
                       "seed": payload["seed"], "arm": payload["arm"], "model": payload.get("model", "tower"),
                       "parameters": payload.get("parameters"),
                       "active_exclusions": {
                           "fold_cohort_and_test_cohort_players_held_out": True,
                           "exclude_assumed_zeros_in_training": bool(payload["config"].get("exclude_assumed_zeros", False)),
                           "unresolved_season_period_games_excluded": "season_guard" in games_cols,
                           "note": ("tables carry no season_guard column: the D-028 unresolved-period exclusion is inert for this run"
                                    if "season_guard" not in games_cols else "season_guard present: D-028 exclusion active")},
                       "code": {"source_digests_now": source_digests(),
                                "note": ("Run processes loaded the package at launch; changes made after launch (D-028: "
                                         "season_guard/season_verified columns, unresolved-period exclusion, exclude_assumed_zeros "
                                         "option, log_prob_components, tables_manifest config field) did not affect training or "
                                         "scoring. Decisions active at launch: D-024 to D-027.")}}
    summary = {"fold": payload["fold"], "arm": payload["arm"], "model": payload.get("model", "tower"),
               "forecast_definition": payload["config"].get("schedule", "realized_S_scenario"),
               "score_definition": "marginal NLL of (G, J, M, B), overtime summed out",
               "nll_mean": float(nll[finite].mean()), "nll_baseline_mean": float(nll_baseline[finite].mean()),
               "nll_conditioned_mean": float(nll_conditioned[finite].mean()),
               "nll_compare_mean": float(nll_compare[finite].mean()) if nll_compare is not None else None,
               "paired_vs_compare": {k: v for k, v in paired.items() if k in ("all", "intl_entrant", "first_year", "later_year_intl")} if paired else None,
               "paired_vs_baseline": {k: baseline_paired[k] for k in ("all", "intl_entrant", "first_year", "later_year_intl")},
               "participation_brier": calibration["brier"], "reference_max_abs_diff": numerical["reference_rescoring"]["max_abs_diff_nats"]}
    report = {"kind": "fold_report", "summary": summary, "population": population, "paired_comparison": paired,
              "paired_vs_baseline": baseline_paired, "components": components, "reconciliation": reconciliation,
              "conventions": {"all_blocks": "overtime marginalized; once-rounded kernel; realized schedule",
                              "nll_conditioned_mean": "training convention (observed overtime), reference only",
                              "preprocessing": "each checkpoint scored on inputs built with its own run's normalizer and vocabularies; units aligned by id"},
              "decomposition": decomposition, "assumed_zero_units_in_evaluation": int(assumed.sum()),
              "participation_calibration": calibration, "supporting_performance": supporting, "draws": draws,
              "numerical": numerical, "reproducibility": reproducibility, "evaluation_by_cohort": payload["evaluation"]}
    write_manifest(run / "report.json", report)
    return report
