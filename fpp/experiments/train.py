"""Train and score one arm on one rolling fold (milestone 0.3).

Fold $v$: units of seasons $\\le v-2$ train, units of season $v-1$ decide
early stopping (forward NLL, never the scored season), units of season $v$
are scored once with the selected epoch. Arms available here:

* ``A`` — forward supervision only;
* ``D`` — forward plus reconstruction windows of training units (seasons
  $\\le v-2$, so no early-stopping label is reconstructed), weighted per
  direction by the paper's player-balanced rule or, under
  ``weighting="season_balanced"`` with a ``first_year_share`` (D-062), by
  unit with first seasons carrying that share; the early-stopping score is
  then the same first-season-weighted mean over the stopping season.

Pretraining-pool arms (B, C, C_shuf) need the pool objective and come with
milestone 0.4. The intercept-only model fitted by L-BFGS on the training
units is scored alongside as the baseline every arm must beat.
"""
from __future__ import annotations

import dataclasses

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.folds import Fold, player_index
from ..data.manifests import file_sha256, load_manifest, write_manifest
from ..data.tables import read_tables
from ..model import features as F
from ..model.dataset import FoldDataset, cohort_masks, first_year_flags, future_targets, stratum_weights
from ..model.distributions import SeasonParameters
from ..model.baseline import AggregateBaseline
from ..model.observation import SummedRoundingSensitivityKernel
from ..model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch
from ..model.tower import ARCHITECTURE_DEFAULTS, PARAMETER_FLOORS, ReferenceTower
from .recovery import fit as lbfgs_fit

STRATA = ["intl_entrant", "first_year", "later_year_intl", "no_tracked_history", "club_50plus"]


def season_batch(batch: dict) -> SeasonBatch:
    return SeasonBatch(batch["games"], batch["starts"], batch["minutes"], batch["counts"], batch["schedule"],
                       batch["overtime"].long(), batch["overtime_known"])


def _subset(ds: FoldDataset, unit_mask: np.ndarray, window_mask: np.ndarray | None = None) -> FoldDataset:
    units = ds.train_units[unit_mask]
    windows = ds.windows[window_mask] if window_mask is not None and len(ds.windows) else ds.windows.iloc[0:0]
    # dataclasses.replace keeps every other field (the D-041 exclusion mask included); positional rebuilds dropped it.
    return dataclasses.replace(ds, train_units=units, windows=windows)


CROSS_STOP_SALT = 20260930      # D-067: fixed, seed-independent, so the two halves of a pair are exact complements


def cross_stop_halves(players) -> tuple[set, set]:
    """Split the validation season's players into two balanced halves by a fixed permutation (D-067)."""
    p = np.array(sorted(int(x) for x in players), dtype=np.int64)
    perm = np.random.default_rng(CROSS_STOP_SALT).permutation(len(p))
    cut = (len(p) + 1) // 2
    return set(p[perm[:cut]].tolist()), set(p[perm[cut:]].tolist())


def season_masks(tu: pd.DataFrame, *, train_max_season: int, stop_season: int, stop_players=None):
    """Training and early-stopping unit masks. Without ``stop_players`` the whole validation season stops training
    (registered rule); with them (D-067 cross-stopping) only those players' validation-season units stop it, and the
    other half's validation-season units join the training labels."""
    season = tu["season"].to_numpy()
    if stop_players is None:
        return season <= train_max_season, season == stop_season
    in_stop = tu["player_id"].isin(stop_players).to_numpy()
    return (season <= train_max_season) | ((season == stop_season) & ~in_stop), (season == stop_season) & in_stop


def _examples_by_season(ds: FoldDataset, *, train_max_season: int, stop_season: int, use_windows: bool, stop_players=None):
    tu = ds.train_units
    train_mask, stop_mask = season_masks(tu, train_max_season=train_max_season, stop_season=stop_season, stop_players=stop_players)
    win_mask = None
    if use_windows and len(ds.windows):
        keep = set(tu.loc[train_mask, "unit_id"])
        win_mask = ds.windows["unit_id"].isin(keep).values
    sub = _subset(ds, train_mask, win_mask)
    if (ds.excluded_rows is None) != (sub.excluded_rows is None):
        raise RuntimeError("Dataset subset lost the D-041 exclusion mask")
    train = sub.training_examples()
    stop = [e for e in _subset(ds, stop_mask).training_examples() if e.direction == "-"]
    # Stopping weights: one per unit (a single season, so player- and season-balancing coincide), rebalanced to the
    # dataset's first-season share when one is set (D-062); otherwise every stop unit weighs one, as registered.
    stop_w = stratum_weights(np.ones(len(stop)), first_year_flags(tu[stop_mask]), ds.first_year_share)
    for e, w in zip(stop, stop_w):
        e.weight = float(w)
    return train, stop


def shuffle_reconstruction_targets(examples, units: pd.DataFrame, rng: np.random.Generator) -> dict:
    """$C^{\\rm shuf}$: permute complete NCAA outcome bundles (outcome, schedule, overtime) among the
    *target units* of reconstruction windows within strata of target season × first-year status
    (D-032, D-033). Every nested window of a unit receives the same donor bundle, so no unit mixes
    donors and every bundle keeps its own support metadata. Forward examples are untouched."""
    info = units.set_index("unit_id")[["season", "first_year"]]
    recon = [i for i, e in enumerate(examples) if e.direction == "+"]
    by_unit = {}
    for i in recon:
        by_unit.setdefault(examples[i].unit_id, []).append(i)
    strata = {}
    for uid in by_unit:
        key = (int(info.loc[uid, "season"]), bool(info.loc[uid, "first_year"]))
        strata.setdefault(key, []).append(uid)
    moved_units = 0
    for key, uids in strata.items():
        if len(uids) < 2:
            continue
        perm = rng.permutation(len(uids))
        donors = [examples[by_unit[uids[j]][0]].targets for j in perm]
        for uid, bundle in zip(uids, donors):
            for i in by_unit[uid]:
                examples[i].targets = bundle
        moved_units += len(uids)
    return {"strata": len(strata), "units_shuffled": moved_units, "units_total": len(by_unit), "windows_total": len(recon)}


PRETRAIN_ARMS = {"B", "C", "C_shuf"}
RECON_ARMS = {"C", "C_shuf", "AR"}       # AR = A plus reconstruction, no pretraining; D (C + forward fine-tuning) is not implemented
ARMS = {"A", "AR", "B", "C", "C_shuf"}


def pretrain(tower, *, store, normalizer, cutoff_days: float, heldout, seed: int, epochs: int, examples_per_epoch: int,
             batch_size: int, lr: float, nodes: int, out_path: Path, excluded_rows=None) -> dict:
    """The registered pool recipe (D-032): next-game prediction on the fold's pool, one checkpoint."""
    from ..model.observation import OnceRoundedMinutesKernel
    from ..model.pool import MINUTES_STEP, PoolSampler, collate_pool
    from ..model.torch_likelihood import beta_binomial_logpmf
    sampler = PoolSampler(store, normalizer, cutoff_days=cutoff_days, heldout_players=heldout, seed=seed, excluded_rows=excluded_rows)
    lik = DifferentiableSeasonLikelihood(None, kernel=OnceRoundedMinutesKernel(step=MINUTES_STEP), quadrature_nodes=nodes)
    optimizer = torch.optim.Adam(tower.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    curve = []
    t0 = time.time()
    for epoch in range(epochs):
        tower.train()
        examples = sampler.sample(examples_per_epoch)
        losses, invalid = [], 0
        for group in _batches(examples, batch_size, rng):
            batch = collate_pool(group, len(tower.translator.embeddings), n_fields=len(store.schema.field_names))
            params = tower.pretrain_forward(batch)
            sb = SeasonBatch(batch["games"], batch["starts"], batch["minutes"], batch["counts"], batch["schedule"],
                             batch["overtime"].long(), batch["overtime_known"])
            lp, _ = lik.log_prob(sb, params=params)
            # Objective log p(J, M, B | G = 1, S = 1, x): with S = 1 the games factor is exactly log pi_g,
            # which is removed so the shared encoder receives no attendance gradient (D-033).
            lp = lp - torch.log(params["pi_g"])
            starts_term = beta_binomial_logpmf(sb.starts, sb.games, params["alpha_j"], params["beta_j"])
            lp = torch.where(batch["starts_known"], lp, lp - starts_term)      # unknown starter: no starts evidence
            finite = torch.isfinite(lp)
            invalid += int((~finite).sum())
            if not bool(finite.any()):
                continue
            loss = (-lp[finite]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Pretraining loss became non-finite; the pool objective diverged")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tower.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        curve.append({"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else None, "invalid_rows": invalid,
                      "examples": len(examples), "seconds": time.time() - t0})
        print(json.dumps({"pretrain": curve[-1]}), file=sys.stderr, flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tower.state_dict(), out_path)
    info = {"path": str(out_path), "sha256": file_sha256(out_path), "pool_targets": len(sampler), "pool_players": sampler.n_players,
            "epochs": epochs, "examples_per_epoch": examples_per_epoch, "batch_size": batch_size, "lr": lr, "nodes": nodes,
            "minutes_step": MINUTES_STEP, "curve": curve, "seconds": time.time() - t0, "seed": seed}
    write_manifest(out_path.with_suffix(".json"), info)
    return info


def split_reconstruction_holdout(train: list, share: float, rng: np.random.Generator):
    """D-042: hold out a share of the reconstruction players entirely — their forward and reconstruction
    examples leave training — so their reconstruction windows test whether later international and
    national-team play translates into the known NCAA season for players the model has never seen."""
    player_of = lambda e: int(str(e.unit_id).split(":")[0])  # noqa: E731
    players = np.array(sorted({player_of(e) for e in train if e.direction == "+"}), dtype=np.int64)
    n = int(round(share * len(players)))
    held = set(int(p) for p in rng.choice(players, size=n, replace=False)) if n else set()
    kept = [e for e in train if player_of(e) not in held]
    test = [e for e in train if e.direction == "+" and player_of(e) in held]
    return kept, test, held


def _batches(examples, size: int, rng: np.random.Generator | None):
    order = np.arange(len(examples)) if rng is None else rng.permutation(len(examples))
    for i in range(0, len(order), size):
        yield [examples[j] for j in order[i:i + size]]


def _decay_lr(optimizer, factor: float) -> float:
    """D-074: multiply every parameter group's learning rate by ``factor``; returns the first group's new rate."""
    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"]) * float(factor)
    return float(optimizer.param_groups[0]["lr"])


def marginal_batch(sb: SeasonBatch) -> SeasonBatch:
    """The same observations with overtime summed out: the paper's forecast target is
    $\\mathbf Y=(G,J,M,\\mathbf B)$, so scores marginalize $O$ even when training conditioned on it."""
    return SeasonBatch(sb.games, sb.starts, sb.recorded_minutes, sb.counts, sb.schedule, None, None)


def _score(likelihood, examples, *, batch_size: int, tower=None, params=None, marginal: bool = True):
    """Per-example NLL (float64) under the tower's heads or fixed params; -inf rows become inf NLL.

    ``marginal=True`` (scoring) sums overtime out; ``marginal=False`` keeps the observed-overtime
    conditioning used by the training objective.
    """
    out = np.empty(len(examples))
    i = 0
    with torch.no_grad():
        for group in _batches(examples, batch_size, None):
            batch = F.collate(group)
            sb = season_batch(batch)
            if marginal:
                sb = marginal_batch(sb)
            p = tower(batch) if tower is not None else params
            lp, _ = likelihood.log_prob(sb, params=p)
            out[i:i + len(group)] = (-lp).numpy()
            i += len(group)
    return out


def score_checked(examples, *, batch_size: int, nodes: int, tolerance: float, kernel=None, tower=None, params=None,
                  max_nodes: int = 256, marginal: bool = True) -> tuple[np.ndarray, dict]:
    """Score at ``nodes`` and verify against twice the nodes; double until the registered
    quadrature tolerance holds or the node budget is exhausted (then fail loudly)."""
    n = nodes
    while True:
        lik_a = DifferentiableSeasonLikelihood(None, kernel=kernel, quadrature_nodes=n)
        lik_b = DifferentiableSeasonLikelihood(None, kernel=kernel, quadrature_nodes=2 * n)
        a = _score(lik_a, examples, batch_size=batch_size, tower=tower, params=params, marginal=marginal)
        b = _score(lik_b, examples, batch_size=batch_size, tower=tower, params=params, marginal=marginal)
        finite = np.isfinite(a) & np.isfinite(b)
        gap = float(np.abs(a[finite] - b[finite]).max()) if finite.any() else 0.0
        if gap <= tolerance:
            return b, {"nodes": 2 * n, "checked_against": n, "max_abs_gap": gap, "within_tolerance": True,
                       "overtime": "marginalized" if marginal else "conditioned_on_observed"}
        if 2 * n >= max_nodes:
            raise RuntimeError(f"Quadrature gap {gap:.3g} exceeds tolerance {tolerance:g} at {2 * n} nodes")
        n *= 2


def fit_baseline(likelihood, examples, *, batch_limit: int, seed: int, max_iter: int) -> dict:
    """Intercept-only parameters fitted on (a subsample of) the training units."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(examples))[:batch_limit]
    batch = F.collate([examples[i] for i in idx])
    sb = season_batch(batch)
    valid = likelihood._valid(sb)
    sb = sb.subset(valid.nonzero(as_tuple=False).squeeze(1))
    result = lbfgs_fit(likelihood, sb, max_iter=max_iter)
    result["rows"] = int(len(sb))
    result["invalid_dropped"] = int((~valid).sum())
    return result


def run_arm(*, tables: Path, fold_season: int, arm: str, seed: int, out: Path, epochs: int = 20, batch_size: int = 256,
            lr: float = 1e-3, patience: int = 3, nodes_train: int = 24, nodes_eval: int = 32,
            max_train_units: int | None = None, baseline_rows: int = 5000, baseline_iter: int = 60,
            first_development: int = 2020, threads: int = 8, model_kind: str = "tower", allow_test: bool = False,
            quadrature_tolerance: float = 5e-5, exclude_assumed_zeros: bool = False, lambda_plus: float = 1.0,
            pretrain_epochs: int = 3, pretrain_examples: int = 60000, pretrained_dir: Path = Path("cache/pretrained"),
            recon_max_years: float | None = 4, recon_k: tuple = (1, 2, 3), resume: bool = False,
            exclude_post_first_season_logs: bool = False, recon_holdout_share: float = 0.0,
            test_registration: Path | None = None, deploy: bool = False, label_scope: str = "all",
            weighting: str = "player_balanced", first_year_share: float | None = None,
            feature_schema: str = "auto", feature_blocks=None, cross_stop: int | None = None,
            width: int = 32, hidden: int = 64, direction_film: bool = False, future_weight: float = 0.0,
            game_dropout: float = 0.0, average_last: int = 1,
            head_hidden: int = 0, attention_pool: bool = False, lr_decay: float = 1.0) -> dict:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}; D (C plus forward fine-tuning) is not implemented")
    if cross_stop not in (None, 0, 1):
        raise ValueError("cross_stop must be 0, 1 or None")
    if model_kind not in {"tower", "aggregate"}:
        raise ValueError("model_kind must be 'tower' (reference) or 'aggregate' (feature baseline)")
    if arm in PRETRAIN_ARMS and model_kind != "tower":
        raise ValueError("Pretraining arms are defined for the reference tower")
    out = Path(out)
    # The run directory exists from launch (status.json, epoch checkpoints); run.json marks completion (D-040).
    if (out / "run.json").exists():
        raise FileExistsError(f"Refusing to overwrite the finished run {out}")
    checkpoint_path = out / "checkpoint_last.pt"
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"{out} holds an unfinished run; pass resume=True (--resume) to continue it")
    out.mkdir(parents=True, exist_ok=True)

    def _status(stage: str, **extra):
        (out / "status.json").write_text(json.dumps({"stage": stage, "arm": arm, "model": model_kind, "fold": fold_season, "seed": seed,
                                                     "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}, indent=1))

    _status("data")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    tab = read_tables(tables)
    units = tab["units"]
    protocol_test = int(units["season"].max())
    if deploy:
        # Deployment (D-051): train through protocol_test - 1, early-stop on protocol_test, no evaluation season. Only
        # after the one-shot test is complete; nothing is scored, so nothing needs protecting.
        if fold_season != protocol_test + 1:
            raise ValueError(f"Deployment trains for season {protocol_test + 1}; got {fold_season}")
        if test_registration is None or not Path(test_registration).is_file():
            raise ValueError("Deployment requires the completed test's registration file")
    elif fold_season >= protocol_test and allow_test:
        if test_registration is None or not Path(test_registration).is_file():
            raise ValueError("The one-shot test needs an existing registration file")
    if fold_season >= protocol_test and not allow_test and not deploy:
        raise ValueError(f"Season {protocol_test} is the one-shot test; scoring it requires the registered "
                         "one-shot decision and allow_test=True")
    test_players = (frozenset() if deploy else
                    frozenset(units.loc[(units["season"] == protocol_test) & units["intl_entrant"], "player_id"]))
    dobs = F.player_dobs(tab)
    # Feature schema (D-065): "auto" reproduces the pre-D-065 rule (v8 iff the tables carry recruit_status); v9 adds blocks.
    fs = F.resolve_schema(feature_schema, feature_blocks, units)
    store = F.GameStore.from_frame(tab["games"], dobs=dobs, schema=fs)
    n_fields, n_static = len(fs.field_names), len(fs.static_continuous)
    fold = Fold(fold_season, "deployment" if deploy else "development" if fold_season <= 2023 else "selection" if fold_season < protocol_test else "test")
    ds = FoldDataset.build(tab, fold, store, test_cohort_players=test_players, index=player_index(tab["games"]), seed=seed,
                           exclude_assumed_zeros=exclude_assumed_zeros, k_values=tuple(recon_k), max_years=recon_max_years,
                           exclude_post_first_season=exclude_post_first_season_logs, label_scope=label_scope,
                           weighting=weighting, first_year_share=first_year_share)
    if max_train_units is not None:
        keep = ds.train_units.sample(n=min(max_train_units, len(ds.train_units)), random_state=seed)
        ds = dataclasses.replace(ds, train_units=keep.reset_index(drop=True),
                                 windows=ds.windows[ds.windows["unit_id"].isin(set(keep["unit_id"]))])
    # Players of the stopping season are held out of the pool and of reconstruction windows (D-033): nothing
    # dated after their validation cutoff may shape the representation that scores them. Their earlier
    # units remain forward training examples (earlier labels, inputs before those units' own cutoffs).
    season_players = set(ds.train_units.loc[ds.train_units["season"] == fold_season - 1, "player_id"])
    # D-067 cross-stopping: only the stopping half is held out; the other half's validation-season units are training labels.
    stop_players = season_players if cross_stop is None else cross_stop_halves(season_players)[cross_stop]
    cross_stop_info = None if cross_stop is None else {
        "stopping_half": cross_stop, "salt": CROSS_STOP_SALT, "validation_season_players": len(season_players),
        "stop_players": len(stop_players), "players_trained_from_validation_season": len(season_players - stop_players)}
    if len(ds.windows):
        ds = dataclasses.replace(ds, windows=ds.windows[~ds.windows["player_id"].isin(stop_players)], heldout=ds.heldout | stop_players)
    if exclude_post_first_season_logs:
        # D-041 guard: the exclusion must survive every dataset rebuild above (a dropped mask reproduces arm A exactly).
        if ds.excluded_rows is None or int(ds.excluded_rows.sum()) == 0:
            raise RuntimeError("Strict control requested but no game rows are excluded")
    if future_weight > 0:
        # D-070: the auxiliary "future" target (later professional production) for training units; never for held-out players.
        ds = dataclasses.replace(ds, future=future_targets(tab, fold, ds.train_units, ds.heldout | stop_players))
    train, stop = _examples_by_season(ds, train_max_season=fold_season - 2, stop_season=fold_season - 1, use_windows=(arm in RECON_ARMS),
                                      stop_players=None if cross_stop is None else stop_players)
    recon_holdout, recon_test = None, []
    if recon_holdout_share > 0 and arm in RECON_ARMS:
        train, recon_test, held_players = split_reconstruction_holdout(train, recon_holdout_share, np.random.default_rng(seed + 13))
        recon_holdout = {"share": recon_holdout_share, "players": len(held_players), "windows": len(recon_test),
                         "units": len({e.unit_id for e in recon_test}),
                         "definition": "reconstruction players held out entirely (forward and reconstruction examples); their windows "
                                       "are scored after training: later international/national play -> known NCAA season"}
    shuffle_info = shuffle_reconstruction_targets(train, ds.train_units, np.random.default_rng(seed + 7)) if arm == "C_shuf" else None
    forward_examples = [e for e in train if e.direction == "-"]
    recon_examples = [e for e in train if e.direction == "+"]
    # Realized first-season mass share per direction (D-062 bookkeeping; the registered rule gives 1/2 forward by construction).
    fy_of_unit = dict(zip(ds.train_units["unit_id"], first_year_flags(ds.train_units)))
    def _fy_share(group):
        tot = sum(e.weight for e in group)
        return float(sum(e.weight for e in group if fy_of_unit.get(e.unit_id, False)) / tot) if tot > 0 else None
    mass_shares = {"forward_first_year_mass_share": _fy_share(forward_examples),
                   "reconstruction_first_year_mass_share": _fy_share(recon_examples),
                   "stop_first_year_mass_share": _fy_share(stop)}
    stop_weights = np.array([e.weight for e in stop], dtype=float)
    stop_criterion = "first_year_share_weighted_mean" if first_year_share is not None else "unweighted_mean"
    evaluate = ds.evaluation_examples()
    t_data = time.time() - t0
    n_fwd = sum(e.direction == "-" for e in train)
    n_rec = len(train) - n_fwd
    lambda_plus = float(lambda_plus) if arm in RECON_ARMS else 0.0
    reference = SeasonParameters()
    lik_train = DifferentiableSeasonLikelihood(None, quadrature_nodes=nodes_train)
    lik_eval = DifferentiableSeasonLikelihood(None, quadrature_nodes=nodes_eval)
    # Baseline: intercept-only, fitted on training forward units.
    t1 = time.time()
    base_model = InterceptOnlySeasonModel(reference)
    base_lik = DifferentiableSeasonLikelihood(base_model, quadrature_nodes=nodes_train)
    baseline_fit = fit_baseline(base_lik, [e for e in train if e.direction == "-"], batch_limit=baseline_rows, seed=seed, max_iter=baseline_iter)
    base_params = {k: v.detach() for k, v in base_model.constrained().items()}
    t_base = time.time() - t1
    # Tower, with heads initialized at the fitted intercepts so training starts from the baseline it must beat.
    vocab_sizes = [len(store.vocab[c]) for c in F.CATEGORICAL]
    static_sizes = [len(v) for v in ds.static_vocab.values()]
    pretrained_info = None
    if arm in PRETRAIN_ARMS:
        # One seed-matched pool checkpoint per fold, built once and reused by B, C, and C_shuf (D-032).
        from ..model.pool import CONTEXT_FIELDS
        from ..model.tower import OutcomeHeads
        tower = ReferenceTower(vocab_sizes, static_sizes, base_model.reference(), n_context=len(CONTEXT_FIELDS),
                               n_fields=n_fields, n_static=n_static, width=width, hidden=hidden, direction_film=direction_film,
                               future_head=future_weight > 0, head_hidden=head_hidden, attention_pool=attention_pool)
        ckpt = Path(pretrained_dir) / f"pool_{fold_season}_{seed}.pt"
        if ckpt.exists():
            pretrained_info = load_manifest(ckpt.with_suffix(".json"))
            if pretrained_info["sha256"] != file_sha256(ckpt):
                raise ValueError("Pretrained checkpoint does not match its manifest")
        else:
            pretrained_info = pretrain(tower, store=store, normalizer=ds.normalizer, cutoff_days=float((fold.training_cutoff - pd.Timestamp("1970-01-01")).days),
                                       heldout=ds.heldout, seed=seed, epochs=pretrain_epochs, examples_per_epoch=pretrain_examples,
                                       batch_size=batch_size, lr=lr, nodes=nodes_train, out_path=ckpt, excluded_rows=ds.excluded_rows)
        tower.load_state_dict(torch.load(ckpt, weights_only=True))
        tower.heads = OutcomeHeads(base_model.reference())     # forward heads start at the fitted intercepts, as in A
    else:
        tower = (ReferenceTower(vocab_sizes, static_sizes, base_model.reference(), n_fields=n_fields, n_static=n_static, width=width, hidden=hidden,
                                direction_film=direction_film, future_head=future_weight > 0, head_hidden=head_hidden,
                                attention_pool=attention_pool) if model_kind == "tower"
                 else AggregateBaseline(vocab_sizes, static_sizes, base_model.reference(), n_fields=n_fields, n_static=n_static))
    optimizer = torch.optim.Adam(tower.parameters(), lr=lr)
    curve, best, best_state, bad_epochs = [], math.inf, None, 0

    def _stop_scores(scores: np.ndarray) -> tuple[float, float]:
        """(unweighted mean, first-season-share-weighted mean) over the stopping units; an infinite row makes both infinite."""
        return float(np.mean(scores)), float((stop_weights * scores).sum() / stop_weights.sum())

    base_stop = _stop_scores(_score(lik_eval, stop, batch_size=batch_size, params=base_params))
    stop_nll_base = base_stop[0] if first_year_share is None else base_stop[1]
    t2 = time.time()
    # Matched optimization (D-033): every arm takes exactly one optimizer step per forward minibatch, and the
    # forward minibatch sequence depends only on the seed, so B and C see identical forward batches; a
    # reconstruction arm adds an independently drawn, separately normalized reconstruction batch to each step.
    forward_rng = np.random.default_rng(seed)
    recon_rng = np.random.default_rng(seed + 11)
    aug_rng = np.random.default_rng(seed + 23)          # D-070 game dropout
    epoch_states = {}                                   # D-070 checkpoint averaging: epoch -> state dict
    updates = {"forward_steps": 0, "forward_examples": 0, "reconstruction_examples": 0, "lambda_plus": lambda_plus}

    def _term(batch, sel_weight):
        sb = season_batch(batch)
        params = tower(batch)
        lp, _ = lik_train.log_prob(sb, params=params)
        finite = torch.isfinite(lp)
        w = batch["weight"]
        term = (w[finite] * (-lp[finite])).sum() / w[finite].sum() if bool(finite.any()) else torch.zeros((), dtype=torch.float64)
        if future_weight > 0 and "future" in params:
            known = batch["future_known"]
            if bool(known.any()):           # D-070: squared error on the standardized later-career production
                err = (params["future"][known].double() - batch["future"][known].double()) ** 2
                term = term + future_weight * err.mean()
        return sel_weight * term, int((~finite).sum())

    # Epoch checkpoints with optimizer and RNG state; a status record; resumption (D-040).
    epoch_start, active = 0, {"wall_seconds": 0.0, "cpu_seconds": 0.0}
    if resume and checkpoint_path.exists():
        ck = torch.load(checkpoint_path, weights_only=False)
        tower.load_state_dict(ck["tower"])
        optimizer.load_state_dict(ck["optimizer"])
        curve, best, best_state, bad_epochs, updates, active = ck["curve"], ck["best"], ck["best_state"], ck["bad_epochs"], ck["updates"], ck["active"]
        forward_rng.bit_generator.state = ck["forward_rng"]
        recon_rng.bit_generator.state = ck["recon_rng"]
        if "aug_rng" in ck:
            aug_rng.bit_generator.state = ck["aug_rng"]
        epoch_states = ck.get("epoch_states", {})
        torch.set_rng_state(ck["torch_rng"])
        epoch_start = int(ck["epoch"])
        print(json.dumps({"fold": fold_season, "arm": arm, "resumed_after_epoch": epoch_start, "best_stop_nll": best}), file=sys.stderr, flush=True)

    def _checkpoint(done: int):
        torch.save({"tower": tower.state_dict(), "optimizer": optimizer.state_dict(), "epoch": done, "curve": curve, "best": best,
                    "best_state": best_state, "bad_epochs": bad_epochs, "updates": updates, "active": active,
                    "forward_rng": forward_rng.bit_generator.state, "recon_rng": recon_rng.bit_generator.state,
                    "aug_rng": aug_rng.bit_generator.state, "epoch_states": epoch_states,
                    "torch_rng": torch.get_rng_state()}, checkpoint_path)
        best_epoch = min(curve, key=lambda c: c["stop_nll"])["epoch"] if curve else None
        _status("training", epochs_completed=done, epochs_max=epochs, patience=patience, best_epoch=best_epoch,
                best_stop_nll=(best if curve else None), bad_epochs=bad_epochs, active_wall_seconds=active["wall_seconds"],
                active_cpu_seconds=active["cpu_seconds"], last_checkpoint=time.strftime("%Y-%m-%dT%H:%M:%S"), forward_steps=updates["forward_steps"])

    for epoch in range(epoch_start, epochs):
        tower.train()
        t_epoch, c_epoch = time.time(), time.process_time()
        losses, invalid = [], 0
        recon_order = recon_rng.permutation(len(recon_examples)) if (lambda_plus > 0 and recon_examples) else None
        recon_pos = 0
        for group in _batches(forward_examples, batch_size, forward_rng):
            if game_dropout > 0:
                group = [F.drop_games(e, game_dropout, aug_rng) for e in group]
            loss, bad = _term(F.collate(group), 1.0)
            invalid += bad
            updates["forward_steps"] += 1
            updates["forward_examples"] += len(group)
            if recon_order is not None:
                if recon_pos + batch_size > len(recon_order):
                    recon_order, recon_pos = recon_rng.permutation(len(recon_examples)), 0
                rgroup = [recon_examples[j] for j in recon_order[recon_pos:recon_pos + batch_size]]
                recon_pos += batch_size
                if game_dropout > 0:
                    rgroup = [F.drop_games(e, game_dropout, aug_rng) for e in rgroup]
                aux, bad = _term(F.collate(rgroup), lambda_plus)
                loss, invalid = loss + aux, invalid + bad
                updates["reconstruction_examples"] += len(rgroup)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tower.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        tower.eval()
        stop_mean, stop_weighted = _stop_scores(_score(lik_eval, stop, batch_size=batch_size, tower=tower))   # marginal, like the reported score
        stop_nll = stop_mean if first_year_share is None else stop_weighted     # the selection criterion (D-062)
        curve.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "stop_nll": stop_nll, "stop_nll_unweighted": stop_mean,
                      "stop_nll_first_year_weighted": stop_weighted, "invalid_rows": invalid,
                      "seconds": time.time() - t2, "wall_seconds": time.time() - t_epoch, "cpu_seconds": time.process_time() - c_epoch})
        active["wall_seconds"] += curve[-1]["wall_seconds"]
        active["cpu_seconds"] += curve[-1]["cpu_seconds"]
        print(json.dumps({"fold": fold_season, "arm": arm, "model": model_kind, **curve[-1]}), file=sys.stderr, flush=True)
        if stop_nll < best - 1e-6:
            best, bad_epochs = stop_nll, 0
            best_state = {k: v.detach().clone() for k, v in tower.state_dict().items()}
        else:
            bad_epochs += 1
            if lr_decay < 1.0:
                _decay_lr(optimizer, lr_decay)        # D-074: a smaller step on a plateau, before patience runs out
        curve[-1]["lr"] = float(optimizer.param_groups[0]["lr"])
        if average_last > 1:
            epoch_states[epoch + 1] = {k: v.detach().clone() for k, v in tower.state_dict().items()}
        _checkpoint(epoch + 1)
        if bad_epochs >= patience:
            break
    _status("scoring", epochs_completed=len(curve), best_stop_nll=best, active_wall_seconds=active["wall_seconds"], active_cpu_seconds=active["cpu_seconds"])
    averaging = None
    if average_last > 1 and curve:
        # D-070: average the weights of the ``average_last`` epochs ending at the best epoch; keep it if the stopping score improves.
        best_epoch = min(curve, key=lambda c: c["stop_nll"])["epoch"]
        window = [e for e in range(best_epoch - average_last + 1, best_epoch + 1) if e in epoch_states]
        if len(window) >= 2:
            avg = {k: (sum(epoch_states[e][k].double() for e in window) / len(window)).to(best_state[k].dtype) for k in best_state}
            tower.load_state_dict(avg)
            tower.eval()
            s_mean, s_w = _stop_scores(_score(lik_eval, stop, batch_size=batch_size, tower=tower))
            avg_stop = s_mean if first_year_share is None else s_w
            averaging = {"window_epochs": window, "stop_nll_best_epoch": best, "stop_nll_averaged": avg_stop, "used": bool(avg_stop < best)}
            if avg_stop < best:
                best_state, best = avg, avg_stop
    tower.load_state_dict(best_state)
    tower.eval()
    # Quadrature verification of the selected model on the stopping set (training nodes vs double).
    _, stop_quadrature = score_checked(stop, batch_size=batch_size, nodes=nodes_train, tolerance=quadrature_tolerance, tower=tower)
    if evaluate:
        # Scoring of the fold season: verified quadrature, plus the kernel sensitivity probe (D-026).
        nll_tower, eval_quadrature = score_checked(evaluate, batch_size=batch_size, nodes=nodes_eval, tolerance=quadrature_tolerance, tower=tower)
        nll_base, base_quadrature = score_checked(evaluate, batch_size=batch_size, nodes=nodes_eval, tolerance=quadrature_tolerance, params=base_params)
        # The training objective's conditioned score is kept for reference; the reported metric is the marginal.
        nll_tower_conditioned = _score(DifferentiableSeasonLikelihood(None, quadrature_nodes=eval_quadrature["nodes"]), evaluate,
                                       batch_size=batch_size, tower=tower, marginal=False)
        probe_kernel = SummedRoundingSensitivityKernel()
        nll_tower_probe = _score(DifferentiableSeasonLikelihood(None, kernel=probe_kernel, quadrature_nodes=eval_quadrature["nodes"]),
                                 evaluate, batch_size=batch_size, tower=tower)
        probe_finite = np.isfinite(nll_tower) & np.isfinite(nll_tower_probe)
        kernel_sensitivity = {"kernel": probe_kernel.registration_id, "sd_multiple": probe_kernel.sd_multiple,
                              "mean_abs_gap": float(np.abs(nll_tower - nll_tower_probe)[probe_finite].mean()),
                              "max_abs_gap": float(np.abs(nll_tower - nll_tower_probe)[probe_finite].max()),
                              "mean_gap_probe_minus_once": float((nll_tower_probe - nll_tower)[probe_finite].mean())}
        masks = cohort_masks(ds.eval_units)
        finite = np.isfinite(nll_tower) & np.isfinite(nll_base)
        summary = {}
        for name in ["all"] + STRATA:
            m = finite if name == "all" else finite & masks[name]
            n = int(m.sum())
            diff = nll_base[m] - nll_tower[m]
            summary[name] = {"n": n, "nll_tower": float(nll_tower[m].mean()) if n else None,
                             "nll_baseline": float(nll_base[m].mean()) if n else None,
                             "gain_vs_baseline": float(diff.mean()) if n else None,
                             "gain_se": float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else None}
        per_unit = pd.DataFrame({"unit_id": ds.eval_units["unit_id"].values, "nll_tower": nll_tower, "nll_baseline": nll_base,
                                 "nll_tower_conditioned_overtime": nll_tower_conditioned, **{k: v for k, v in masks.items()}})
    else:   # deployment: no evaluation season
        eval_quadrature = base_quadrature = kernel_sensitivity = None
        summary = {}
        finite = np.zeros(0, dtype=bool)
        per_unit = pd.DataFrame(columns=["unit_id", "nll_tower", "nll_baseline", "nll_tower_conditioned_overtime"])
    out.mkdir(parents=True, exist_ok=True)
    per_unit.to_csv(out / "per_unit.csv", index=False)
    torch.save(best_state, out / "tower_state.pt")
    if recon_holdout is not None and recon_test:
        # D-042: translation test on unseen players — their NCAA season predicted from their later play alone.
        r_tower = np.asarray(_score(lik_eval, recon_test, batch_size=batch_size, tower=tower), dtype=float)
        r_base = np.asarray(_score(lik_eval, recon_test, batch_size=batch_size, params=base_params), dtype=float)
        info = ds.train_units.drop_duplicates("unit_id").set_index("unit_id")[["player_id", "season", "first_year", "intl_history",
                                                                               "n_intl_games", "n_national_games", "nationality"]]
        rows = pd.DataFrame({"unit_id": [e.unit_id for e in recon_test], "k": [str(e.k) for e in recon_test],
                             "n_games": [int(e.intl.x.shape[0]) for e in recon_test], "nll_tower": r_tower, "nll_baseline": r_base})
        rows = rows.join(info, on="unit_id")
        rows.to_csv(out / "recon_heldout.csv", index=False)
        ok = np.isfinite(r_tower) & np.isfinite(r_base)
        d = r_base - r_tower
        fy = rows["first_year"].to_numpy(dtype=bool, na_value=False) & ok
        recon_holdout.update({"nll_tower": float(r_tower[ok].mean()), "nll_baseline": float(r_base[ok].mean()),
                              "gain_vs_baseline": float(d[ok].mean()), "gain_se": float(d[ok].std(ddof=1) / math.sqrt(ok.sum())) if ok.sum() > 1 else None,
                              "first_year_windows": int(fy.sum()), "first_year_nll_tower": float(r_tower[fy].mean()) if fy.any() else None,
                              "first_year_nll_baseline": float(r_base[fy].mean()) if fy.any() else None,
                              "first_year_gain": float(d[fy].mean()) if fy.any() else None,
                              "first_year_gain_se": float(d[fy].std(ddof=1) / math.sqrt(fy.sum())) if fy.sum() > 1 else None})
    payload = {
        "kind": "arm_run", "arm": arm, "model": model_kind, "fold": fold_season, "seed": seed, "tables": str(tables),
        "parameters": int(sum(p.numel() for p in tower.parameters())),
        "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "patience": patience, "nodes_train": nodes_train,
                   "nodes_eval": nodes_eval, "max_train_units": max_train_units, "half_life_years": F.HALF_LIFE_YEARS,
                   "train_seasons_through": fold_season - 2, "stop_season": fold_season - 1,
                   "schedule": "realized_S_scenario (likelihood condition only; encoder sees prior-season games)",
                   "score_definition": "marginal NLL of (G, J, M, B) with overtime summed out (paper eq. observation); "
                                       "training conditions on observed overtime where certified",
                   "exclude_assumed_zeros": exclude_assumed_zeros, "lambda_plus": lambda_plus, "deploy": deploy, "label_scope": label_scope,
                   "weighting": weighting, "first_year_share": first_year_share, "stop_criterion": stop_criterion,
                   "feature_schema": {**fs.describe(), "missing_game_fields": list(store.missing_fields)},
                   "architecture": getattr(tower, "architecture", dict(ARCHITECTURE_DEFAULTS)),
                   "future_weight": future_weight, "game_dropout": game_dropout, "average_last": average_last, "lr_decay": lr_decay,
                   "post_first_season_logs": ("excluded from every input channel, the pool and reconstruction (D-041 strict control)"
                                              if exclude_post_first_season_logs else "included"),
                   "excluded_game_rows": int(ds.excluded_rows.sum()) if ds.excluded_rows is not None else 0,
                   "normalizer_exclusion": True,     # D-072/D-078: the normalization statistics exclude the same rows as the inputs
                   "reconstruction_windows": {"max_years": recon_max_years, "k_values": [k for k in recon_k],
                                              "horizon": "training cutoff" if recon_max_years is None else f"{recon_max_years} years after the season end",
                                              "windows": int(len(ds.windows)), "units": int(ds.windows["unit_id"].nunique()) if len(ds.windows) else 0},
                   "pretraining": pretrained_info, "shuffle": shuffle_info, "updates": updates,
                   "stop_players_held_out_of_pool_and_reconstruction": len(stop_players), "cross_stop": cross_stop_info,
                   "matching": "one optimizer step per forward minibatch; forward batch order fixed by seed; "
                               "reconstruction batch of the same size added per step, separately normalized",
                   "tables_manifest": {"payload_sha256": tab["manifest"].get("payload_sha256"),
                                       "tables": {k: v["sha256"] for k, v in tab["manifest"]["tables"].items()}},
                   "minutes_kernel": "once-rounded season total (working approximation; see kernel_sensitivity)",
                   "parameter_floors": PARAMETER_FLOORS, "quadrature_tolerance": quadrature_tolerance},
        "numerical": {"stop_set_quadrature": stop_quadrature, "eval_quadrature": eval_quadrature,
                      "baseline_quadrature": base_quadrature, "kernel_sensitivity": kernel_sensitivity},
        "reconstruction_heldout": recon_holdout,
        "test_registration": ({"path": str(test_registration), "sha256": file_sha256(Path(test_registration))} if test_registration else None),
        "counts": {"train_forward": n_fwd, "train_reconstruction": n_rec, "stop_units": len(stop), "eval_units": len(evaluate),
                   "eval_invalid": int((~finite).sum()), **mass_shares,
                   "future_targets": int(sum(bool(e.targets.get("future_known", False)) for e in train))},
        "baseline": {"fit": baseline_fit, "parameters": {k: (v.tolist() if v.dim() else float(v)) for k, v in base_params.items()},
                     "stop_nll": stop_nll_base, "stop_nll_unweighted": base_stop[0], "stop_nll_first_year_weighted": base_stop[1]},
        "training": {"curve": curve, "best_stop_nll": best, "epochs_run": len(curve), "seconds": time.time() - t2, "averaging": averaging},
        "timing": {"data_seconds": t_data, "baseline_seconds": t_base, "total_seconds": time.time() - t0},
        "evaluation": summary,
    }
    write_manifest(out / "run.json", payload)
    _status("done", epochs_completed=len(curve), best_stop_nll=best, active_wall_seconds=active["wall_seconds"],
            active_cpu_seconds=active["cpu_seconds"], total_seconds=time.time() - t0)
    return payload
