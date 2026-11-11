"""Cross-fitted future-informed teachers (D-109, D-110): train q(Y | X, V) — or its no-future twin D⁰ — on the players of
every split but one, inside the fold's training era, then write distillation targets for the held-out split.

Design, as registered in D-110: the teacher is trained from scratch per split (no initialization from a tower that saw the
held-out players' outcomes); all windows and seasons of a player stay in one split (``player_split``, seed-independent);
later careers are the fold's reconstruction windows, so they obey the outer training cutoff; the stopping season
(fold − 1) is nobody's label and enters neither the teacher's training nor its targets; teachers run a fixed number of
epochs (no epoch selection on held-out labels). D⁰ is the same module, split, capacity and procedure with ``use_future``
off, so D⁺ − D⁰ is the value of giving the teacher the later career. Targets are written per (outcome, window) with the
window's registered weight; see ``fpp.model.teacher.TeacherCache``.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.folds import Fold, player_index
from ..data.tables import read_tables
from ..model import features as F
from ..model.artifact import canonical_sha256, normalizer_sha256
from ..model.dataset import FoldDataset, _days
from ..model.distributions import SeasonParameters
from ..model.teacher import (PARAM_KEYS, SPLIT_SALT, TeacherCache, TeacherTower, checkpoint_sha256, collate_pairs, pair_examples,
                             player_split)
from ..model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel
from ..model.tower import ReferenceTower
from .distill import makes_logprob_by_shot
from .train import _batches, _subset, batch_loss, fit_baseline, marginal_batch, protocol_test_season, season_batch


def teacher_masks(train_units: pd.DataFrame, windows: pd.DataFrame, *, fold_season: int, split: dict, k: int):
    """(teacher training units, held-out units, their window masks). Training-era units are those with season ≤ fold − 2;
    the stopping season is excluded from both sides. Units of players in split ``k`` are held out; windows follow units."""
    season = train_units["season"].to_numpy()
    era = season <= fold_season - 2
    held = np.array([split.get(int(p), -1) == k for p in train_units["player_id"]], dtype=bool)
    train_mask, held_mask = era & ~held, era & held

    def win(mask):
        if not len(windows):
            return np.zeros(0, dtype=bool)
        keep = set(train_units.loc[mask, "unit_id"])
        return windows["unit_id"].isin(keep).to_numpy()

    return train_mask, held_mask, win(train_mask), win(held_mask)


def refit_preprocessing(store: F.GameStore, units: pd.DataFrame, *, cutoff_days: float, allowed_players, excluded_rows, seed: int,
                        normalizer_rows: int = 400_000):
    """Cross-fitting covers preprocessing (D-110): the normalizer is fitted on games of the teacher's own training players
    only (released by the training cutoff, the store's row policy, the strict-control exclusion if any) and the static
    vocabulary on those players' units; both are then frozen and applied to the held-out split. A held-out player's games
    — the target season's included — cannot move these statistics."""
    players = np.fromiter((int(p) for p in allowed_players), dtype=np.int64)
    allowed = (store.release <= cutoff_days) & np.isin(store.player_id, players) & store.eligible()
    if excluded_rows is not None:
        allowed &= ~excluded_rows
    rows = np.flatnonzero(allowed)
    rng = np.random.default_rng(seed)
    if len(rows) > normalizer_rows:
        rows = np.sort(rng.choice(rows, normalizer_rows, replace=False))
    own = units[units["player_id"].isin(set(int(p) for p in allowed_players))]
    vocab = F.static_vocab(own, store.schema)
    normalizer = F.Normalizer.fit(store, rows, cutoff_days, F.static_continuous(own, store.schema))
    return normalizer, vocab


def rescale_to_mean_one(pairs: list) -> None:
    """Divide a component's weights by their mean (in place), so the batch mean of w·ℓ is an unbiased estimate of the
    component's globally weighted loss Σ wℓ / Σ w under uniform minibatch sampling (D-110 amendment)."""
    if pairs:
        mean = float(np.mean([p.weight for p in pairs]))
        if mean > 0:
            for p in pairs:
                p.weight = p.weight / mean


def step_loss(teacher: TeacherTower, lik, fwd_group: list, win_group: list, lambda_t: float, normalizer: str = "size") -> torch.Tensor:
    """One step of L_T = Σ w⁻ ℓ / Σ w⁻ + λ_T · Σ w⁺ ℓ / Σ w⁺ (D-110): the missing-view rows and the window rows are estimated
    separately, so a window policy changes the evidence supplied, not the coefficient of future-informed supervision. With
    ``normalizer="size"`` and component weights rescaled to mean one, each term is the batch mean of w·ℓ — unbiased for the
    declared globally weighted loss (a per-batch weight-sum denominator is not). D⁰ takes the same rows and weights."""
    bx, bv, has_future, w, ids = collate_pairs(fwd_group)
    lp, _ = lik.log_prob(season_batch(bx), params=teacher(bx, bv, has_future))
    loss = batch_loss(lp, w, ids, normalizer)
    if win_group and lambda_t > 0:
        bx, bv, has_future, w, ids = collate_pairs(win_group)
        lp, _ = lik.log_prob(season_batch(bx), params=teacher(bx, bv, has_future))
        loss = loss + lambda_t * batch_loss(lp, w, ids, normalizer)
    return loss


def prepare_dataset(tables, fold_season: int, *, feature_schema: str, feature_blocks, row_guard: bool, recon_k, recon_max_years, seed: int,
                    label_scope: str = "all", weighting: str = "player_balanced", first_year_share=None, backward_reader: str = "latest"):
    """The fold's dataset as ``run_arm`` builds it (schema, store, windows, stop-season players' windows removed)."""
    protocol_test = protocol_test_season()
    if fold_season >= protocol_test:
        raise ValueError(f"teachers are trained inside development and selection folds; {fold_season} reaches the test season {protocol_test}")
    tab = read_tables(tables)
    units = tab["units"]
    test_players = frozenset(units.loc[(units["season"] == protocol_test) & units["intl_entrant"], "player_id"])
    dobs = F.player_dobs(tab)
    fs = F.resolve_schema(feature_schema, feature_blocks, units)
    store = F.GameStore.from_frame(tab["games"], dobs=dobs, schema=fs, row_guard=row_guard)
    fold = Fold(fold_season, "development" if fold_season <= 2023 else "selection")
    kwargs = dict(test_cohort_players=test_players, index=player_index(tab["games"]), seed=seed, k_values=tuple(recon_k), max_years=recon_max_years,
                  label_scope=label_scope, weighting=weighting, first_year_share=first_year_share)
    if "backward_reader" in FoldDataset.__dataclass_fields__:
        kwargs["backward_reader"] = backward_reader
    elif backward_reader != "latest":
        raise ValueError("this tree has no backward reader option (D-109 patch not applied); only 'latest' is available")
    ds = FoldDataset.build(tab, fold, store, **kwargs)
    season_players = set(ds.train_units.loc[ds.train_units["season"] == fold_season - 1, "player_id"])
    if len(ds.windows):      # D-033: the stopping season's players hold no reconstruction windows
        ds = dataclasses.replace(ds, windows=ds.windows[~ds.windows["player_id"].isin(season_players)], heldout=ds.heldout | season_players)
    return tab, store, fs, ds


def rows_as_arrays(rows: dict) -> dict:
    """The in-progress row lists as concatenated arrays (what ``TeacherCache.write`` saves), for in-memory summaries."""
    out = {"unit_id": np.array(rows["unit_id"], dtype=str), "weight": np.array(rows["weight"], dtype=np.float64)}
    for key in PARAM_KEYS + ("representation",) + TeacherCache.EXTRA:
        if rows.get(key):
            out[key] = np.concatenate(rows[key])
    return out


def train_teacher(*, tables, fold_season: int, n_splits: int, split: int, use_future: bool, out, seed: int, epochs: int = 20,
                  batch_size: int = 256, lr: float = 1e-3, nodes_train: int = 24, nodes_eval: int = 32, threads: int = 8, baseline_rows: int = 5000,
                  baseline_iter: int = 60, feature_schema: str = "auto", feature_blocks=None, row_guard: bool = True,
                  recon_k=(1, 2, 3, None), recon_max_years=None, backward_reader: str = "latest", width: int = 32, hidden: int = 64,
                  teacher_lambda: float = 1.0) -> dict:
    if not 0 <= split < n_splits:
        raise ValueError(f"split must lie in [0, {n_splits})")
    if teacher_lambda < 0:
        raise ValueError("teacher_lambda must be non-negative")
    out = Path(out)
    if (out / "teacher.json").exists():
        raise FileExistsError(f"Refusing to overwrite the finished teacher {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    t0 = time.time()

    def status(stage, **extra):
        (out / "status.json").write_text(json.dumps({"stage": stage, "fold": fold_season, "split": split, "n_splits": n_splits,
                                                     "use_future": use_future, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}, indent=1))

    status("data")
    tab, store, fs, ds = prepare_dataset(tables, fold_season, feature_schema=feature_schema, feature_blocks=feature_blocks, row_guard=row_guard,
                                         recon_k=recon_k, recon_max_years=recon_max_years, seed=seed, backward_reader=backward_reader)
    assignment = player_split(ds.train_units["player_id"], n_splits)
    train_mask, held_mask, train_win, held_win = teacher_masks(ds.train_units, ds.windows, fold_season=fold_season, split=assignment, k=split)
    # Cross-fitting covers preprocessing: refit on the teacher's own training players, freeze, apply to the held-out split.
    allowed_players = {int(p) for p in ds.train_units.loc[train_mask, "player_id"]}
    normalizer, static_vocab = refit_preprocessing(store, ds.train_units[train_mask], cutoff_days=_days(ds.fold.training_cutoff),
                                                   allowed_players=allowed_players, excluded_rows=ds.excluded_rows, seed=seed)
    train_ds = dataclasses.replace(_subset(ds, train_mask, train_win), normalizer=normalizer, static_vocab=static_vocab)
    held_ds = dataclasses.replace(_subset(ds, held_mask, held_win), normalizer=normalizer, static_vocab=static_vocab)
    train_examples = train_ds.training_examples()
    forward = [e for e in train_examples if e.direction == "-"]
    recon = [e for e in train_examples if e.direction == "+"]
    pairs = pair_examples(forward, recon)
    fwd_pairs = [p for p in pairs if p.v is None]
    win_pairs = [p for p in pairs if p.v is not None]
    rescale_to_mean_one(fwd_pairs)            # each component's batch mean of w·ℓ then estimates its global weighted loss without bias
    rescale_to_mean_one(win_pairs)
    # The held-out split's players never reach the teacher's labels: assert it structurally.
    held_players = {int(p) for p in ds.train_units.loc[held_mask, "player_id"]}
    train_players = {int(p) for p in train_ds.train_units["player_id"]}
    if held_players & train_players:
        raise RuntimeError("a held-out player appears among the teacher's training units")
    t_data = time.time() - t0
    # Reference at the intercept baseline fitted on the teacher's own forward units, as run_arm does for the student.
    lik = DifferentiableSeasonLikelihood(None, quadrature_nodes=nodes_train)
    base_model = InterceptOnlySeasonModel(SeasonParameters())
    base_lik = DifferentiableSeasonLikelihood(base_model, quadrature_nodes=nodes_train)
    baseline = fit_baseline(base_lik, forward, batch_limit=baseline_rows, seed=seed, max_iter=baseline_iter)
    vocab_sizes = [len(store.vocab[c]) for c in F.CATEGORICAL]
    static_sizes = [len(v) for v in train_ds.static_vocab.values()]
    base = ReferenceTower(vocab_sizes, static_sizes, base_model.reference(), n_fields=len(fs.field_names), n_static=len(fs.static_continuous),
                          width=width, hidden=hidden)
    teacher = TeacherTower(base, base_model.reference(), use_future=use_future)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    curve = []
    status("training", epochs_max=epochs, pairs=len(pairs), forward_rows=len(fwd_pairs), window_rows=len(win_pairs))
    win_rng = np.random.default_rng(seed + 11)
    for epoch in range(epochs):
        teacher.train()
        t_epoch, losses = time.time(), []
        win_order, win_pos = (win_rng.permutation(len(win_pairs)) if win_pairs else None), 0
        for group in _batches(fwd_pairs, batch_size, rng):
            wgroup = []
            if win_order is not None:                 # one window batch per forward batch, from its own permutation (as run_arm pairs reconstruction batches)
                if win_pos + batch_size > len(win_order):
                    win_order, win_pos = win_rng.permutation(len(win_pairs)), 0
                wgroup = [win_pairs[j] for j in win_order[win_pos:win_pos + batch_size]]
                win_pos += batch_size
            loss = step_loss(teacher, lik, group, wgroup, teacher_lambda)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        curve.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "seconds": time.time() - t_epoch})
        print(json.dumps({"teacher": {"fold": fold_season, "split": split, "use_future": use_future, **curve[-1]}}), file=sys.stderr, flush=True)
        status("training", epochs_completed=epoch + 1, epochs_max=epochs, train_loss=curve[-1]["train_loss"])
    # Targets for the held-out split: one row per (outcome, window), the no-future teacher reading nothing from V. Each row
    # also carries the teacher's held-out log score of that outcome under the reported convention (overtime summed out,
    # nodes_eval quadrature), the same for its makes | attempts factor per shot type (exact), the unit's player and cohort
    # flags, so D⁺ and D⁰ teachers can be compared on the same held-out labels, jointly and on the factor the shooting pilot
    # distills, with intervals clustered by player.
    status("targets")
    teacher.eval()
    lik_eval = DifferentiableSeasonLikelihood(None, quadrature_nodes=nodes_eval)
    held_examples = held_ds.training_examples()
    held_pairs = [p for p in pair_examples([e for e in held_examples if e.direction == "-"], [e for e in held_examples if e.direction == "+"]) if p.v is not None]
    hu = held_ds.train_units.set_index("unit_id")
    cohort = {"first_year": hu["first_year"].astype(bool).to_dict(), "intl_entrant": hu["intl_entrant"].astype(bool).to_dict(),
              "has_history": ((hu["n_intl_games"].fillna(0) > 0) | (hu["n_national_games"].fillna(0) > 0)).to_dict()}
    unit_player = hu["player_id"].astype(int).to_dict()
    rows = TeacherCache.empty_rows()
    position, seen = [], {}
    for p in held_pairs:
        position.append(seen.get(p.unit_id, 0))
        seen[p.unit_id] = position[-1] + 1
    with torch.no_grad():
        start = 0
        for group in _batches(held_pairs, batch_size, None):
            bx, bv, has_future, w, ids = collate_pairs(group)
            out_params = teacher(bx, bv, has_future)
            lp, _ = lik_eval.log_prob(marginal_batch(season_batch(bx)), params=out_params)
            lp_shots = makes_logprob_by_shot(out_params, bx["counts"])        # [B, 3] log BB(K_h | A_h) per shot type under this window's law
            flags = {k: [cohort[k].get(u, False) for u in ids] for k in ("first_year", "intl_entrant", "has_history")}
            TeacherCache.append(rows, ids, [p.k for p in group], position[start:start + len(group)], w.tolist(), out_params,
                                logscore=lp.numpy(), logscore_makes=lp_shots.sum(1).numpy(), logscore_makes_by_shot=lp_shots.numpy(),
                                player_ids=[unit_player[u] for u in ids], flags=flags)
            start += len(group)
    # Held-out teacher score by cohort: the teacher-advantage diagnostic of D-110 (D⁺ − D⁰ on these units). Per unit the
    # primary statistic is the score of the ω-mixture of its windows' laws (what distillation supervises with); the ω-mean of
    # the window scores is kept beside it. For the joint season score, for the shooting factor as the sum of the three shot
    # types' marginal mixtures (what a factorized student can learn), per shot type, and for the joint shooting law.
    held_cache = TeacherCache(rows_as_arrays(rows), {}) if rows["unit_id"] else None
    def _by_cohort(scores):
        def _mean(sel):
            v = [s for s, fy, ent, hist in scores.values() if sel(fy, ent, hist)]
            return {"n": len(v), "mean_logscore": float(np.mean(v)) if v else None}
        return {"all": _mean(lambda fy, ent, hist: True), "first_year_with_history": _mean(lambda fy, ent, hist: fy and hist),
                "first_year_without_history": _mean(lambda fy, ent, hist: fy and not hist), "intl_entrant": _mean(lambda fy, ent, hist: ent)}
    held_out_score = {score: {stat: (_by_cohort(held_cache.unit_scores(score, stat)) if held_cache is not None else {}) for stat in ("mixture", "mean")}
                      for score in ("joint", "makes_transfer", "makes_K2", "makes_K3", "makes_Kf", "makes_given_attempts")}
    sha = checkpoint_sha256(teacher.state_dict())
    torch.save(teacher.state_dict(), out / "teacher_state.pt")
    # The actual player→split mapping and the frozen preprocessing travel with the checkpoint (D-110): a salt alone does not
    # identify the mapping once the player pool changes, and the student must apply the teacher's own normalizer/vocabulary.
    (out / "split_assignment.json").write_text(json.dumps({"n_splits": n_splits, "split": split, "salt": SPLIT_SALT,
                                                           "players": {str(p): s for p, s in sorted(assignment.items())}}))
    torch.save({"normalizer": {k: np.array(getattr(normalizer, k), copy=True) for k in ("mu", "sigma", "static_mu", "static_sigma")},
                "static_vocab": static_vocab, "vocab": store.vocab}, out / "preprocessing.pt")
    meta = {"kind": "teacher", "fold": fold_season, "n_splits": n_splits, "split": split, "use_future": use_future, "seed": seed, "epochs": epochs,
            "batch_size": batch_size, "lr": lr, "nodes_train": nodes_train, "tables": str(tables), "teacher_lambda": teacher_lambda,
            "objective": "forward (missing-view) term + teacher_lambda × window term; each the batch mean of w·ℓ with the component's "
                         "weights rescaled to mean one (unbiased for the globally weighted loss); fixed epochs, no selection on labels",
            "split_assignment_sha256": canonical_sha256(sorted((int(p), int(s)) for p, s in assignment.items())),
            "normalizer_sha256": normalizer_sha256(normalizer), "static_vocab_sha256": canonical_sha256(static_vocab),
            "vocab_sha256": canonical_sha256(store.vocab),
            "tables_manifest": {"payload_sha256": tab["manifest"].get("payload_sha256")},
            "training_cutoff": str(ds.fold.training_cutoff.date()), "forecast_cutoff": str(ds.fold.forecast_cutoff.date()),
            "feature_schema": fs.describe(), "architecture": teacher.architecture, "backward_reader": backward_reader,
            "reconstruction_windows": {"k_values": [k for k in recon_k], "max_years": recon_max_years},
            "split_salt": 20261110, "checkpoint_sha256": sha,
            "counts": {"teacher_training_units": int(train_mask.sum()), "teacher_training_pairs": len(pairs), "teacher_training_windows": int(train_win.sum()),
                       "held_out_units": int(held_mask.sum()), "held_out_players": len(held_players), "held_out_windows": int(held_win.sum()),
                       "target_rows": len(rows["unit_id"]), "target_units": len(seen)},
            "held_out_score": {**held_out_score,
                               "convention": f"per (outcome, window): 'joint' = marginal log probability of the season, overtime summed out, "
                                             f"{nodes_eval}-node quadrature; 'makes_K2/K3/Kf' = log BB(K_h | A_h) per shot type, exact; "
                                             "'makes_given_attempts' their sum. Per unit: 'mixture' = log Σ_w ω̃_w exp(ℓ_w) (the ω-mixture of "
                                             "window laws, which distillation supervises with), 'mean' = Σ_w ω̃_w ℓ_w, ω̃ the unit's window weights "
                                             "normalized to one; 'makes_transfer' = the sum over shot types of the marginal mixture scores (what a "
                                             "factorized student can learn), 'makes_given_attempts' mixture = the joint shooting law mixed over "
                                             "windows; paired units only"},
            "baseline": {"fit": baseline, "parameters": {k: v.detach().tolist() for k, v in base_model.constrained().items()}},
            "curve": curve, "seconds": {"data": t_data, "total": time.time() - t0}}
    TeacherCache.write(out / "targets", rows, meta)
    (out / "teacher.json").write_text(json.dumps(meta, indent=1, default=str))
    status("done", target_rows=len(rows["unit_id"]))
    return meta
