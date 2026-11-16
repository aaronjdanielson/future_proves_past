"""Training and forward scoring of the compact latent-career model (D-115).

One fit = one fold v, one arm, one seed: NCAA labels through season v−2, checkpoint selection on v−1 (the release's rule),
then forward predictive scores of season v and of season v+1 with the frozen model — the D-114 chronology, so the v+1 scores
are information-matched to the saved forecasts of the released architecture.

Arms share every line of code. ``later``: all admissible spells. ``strict``: the D-041 row exclusion of the release's strict
control removes non-NCAA rows dated after the end of a player's first NCAA season from every observation term, encoder
input and statistic. Information rules enforced here: a prefix example reads only rows released by its unit's forecast
cutoff (NCAA rows only from earlier seasons); a training career reads only rows released by the fold's training cutoff
(NCAA rows through v−2); players of the checkpoint-selection season contribute no career; held-out players have no units.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.folds import Fold, player_index
from ..data.tables import read_tables
from ..model import features as F
from ..model import latent_career as LC
from ..model.dataset import FoldDataset, example_weights, mask_post_first_season_statics
from ..model.latent_career import DT, LatentCareerModel, SPELL_TYPES, draw
from ..model.production import MAKE_ATTEMPTS, RATE_NAMES
from ..model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch
from ..model.distributions import SeasonParameters
from ..data.manifests import file_sha256
from .recovery import fit as lbfgs_fit
from .train import protocol_test_season

ARMS = ("later", "strict")
N_FEATURES = 38
N_COVARIATES = 8
YOUTH_AGES = {"U18", "U19", "U21", "youth", "U14", "U15", "U16", "U17", "U20"}
SCHEMA_NAME, SCHEMA_BLOCKS = "v9", ("prior_league", "destination", "on3")      # the released feature schema's static side


def _days(ts) -> float:
    return float((np.datetime64(pd.Timestamp(ts), "D") - np.datetime64("1970-01-01", "D")) / F.DAY)


def _age_basis(age: np.ndarray) -> np.ndarray:
    known = np.isfinite(age)
    a = np.where(known, np.clip((age - LC.AGE_CENTER) / LC.AGE_SCALE, -2.0, 2.5), 0.0)
    return np.stack([np.ones_like(a), a, a * a, (~known).astype(float)], axis=-1)


class SpellSource:
    """Row arrays aligned with the GameStore order, and spell aggregation per (season, competition)."""

    def __init__(self, games: pd.DataFrame, store: F.GameStore, excluded: np.ndarray | None = None):
        g = games.sort_values(["player_id", "date", "record_id"]).reset_index(drop=True)
        if not np.array_equal(g["record_id"].to_numpy(), store.record_id):
            raise RuntimeError("the game frame does not align with the store")
        self.store = store
        self.counts = np.nan_to_num(g[list(F.BOX_FIELDS)].to_numpy(dtype=float))
        self.minutes = np.nan_to_num(g["minutes"].to_numpy(dtype=float))
        self.appeared = g["appeared"].to_numpy(dtype=bool, na_value=False)
        starter = g["starter"].astype("boolean")
        self.started = starter.fillna(False).to_numpy(dtype=bool)
        self.starter_known = starter.notna().to_numpy()
        self.comp = store.cats[:, F.CATEGORICAL.index("competition")].astype(np.int64)
        source = g["source"].astype(str).to_numpy()
        kind = g["competition_kind"].astype(str).to_numpy()
        age_group = g["age_group"].astype(str).to_numpy()
        typ = np.full(len(g), SPELL_TYPES.index("other"), dtype=np.int64)
        typ[np.isin(kind, ["league", "cup", "qualifier", "super_cup"])] = SPELL_TYPES.index("senior")
        typ[(kind == "youth") | np.isin(age_group, list(YOUTH_AGES))] = SPELL_TYPES.index("youth")
        typ[source == "national"] = SPELL_TYPES.index("national")
        typ[source == "events"] = SPELL_TYPES.index("event")
        typ[source == "ncaa"] = SPELL_TYPES.index("ncaa")
        self.type = typ
        valid = g["counts_valid"].to_numpy(dtype=bool, na_value=False) if "counts_valid" in g else np.ones(len(g), dtype=bool)
        self.usable = store.eligible() & valid
        if excluded is not None:
            self.usable = self.usable & ~excluded
        self.strength = {k: store.strength[k] for k in ("team_adj_o", "team_adj_d", "opp_adj_o", "opp_adj_d")}
        self.strength_mu = np.zeros(4)
        self.strength_sd = np.ones(4)

    def fit_strength(self, allowed: np.ndarray, cutoff_days: float) -> None:
        """Standardization of the four rating fields from the rows a fit may see (its own exclusions applied)."""
        for j, (values, avail) in enumerate(self.strength.values()):
            ok = allowed & self.usable & np.isfinite(values) & (avail <= cutoff_days)
            if ok.sum() > 10:
                self.strength_mu[j], self.strength_sd[j] = float(values[ok].mean()), float(values[ok].std() + 1e-6)

    def spells(self, player_id: int, cutoff_days: float, ncaa_max_season: int):
        """Aggregates of the player's usable rows released by the cutoff (NCAA rows only through ``ncaa_max_season``)."""
        s, e = self.store.index.get(int(player_id), (0, 0))
        if e <= s:
            return None
        rows = np.arange(s, e)
        st = self.store
        keep = self.usable[rows] & (st.release[rows] <= cutoff_days) & (~st.is_ncaa[rows] | (st.season[rows] <= ncaa_max_season))
        rows = rows[keep]
        if not len(rows):
            return None
        season = st.season[rows].astype(np.int64)
        uniq, inv = np.unique(season * 8192 + self.comp[rows], return_inverse=True)
        n = len(uniq)
        add = lambda w: np.bincount(inv, weights=w, minlength=n)           # noqa: E731
        first = np.zeros(n, dtype=np.int64)
        first[inv[::-1]] = np.arange(len(rows))[::-1]
        out = {"season": uniq // 8192, "comp": uniq % 8192, "type": self.type[rows][first],
               "minutes": add(self.minutes[rows]), "games": add(self.appeared[rows].astype(float)),
               "starts": add((self.started[rows] & self.appeared[rows]).astype(float)),
               "starts_known": add((self.starter_known[rows] & self.appeared[rows]).astype(float)),
               "counts": np.stack([add(self.counts[rows, j]) for j in range(self.counts.shape[1])], axis=1)}
        for name, (values, avail) in self.strength.items():
            v = values[rows]
            ok = np.isfinite(v) & (avail[rows] <= cutoff_days)
            out[name] = add(np.where(ok, v, 0.0))
            out[name + "_n"] = add(ok.astype(float))
        return out

    def features(self, sp: dict, age: np.ndarray) -> np.ndarray:
        """[n, N_FEATURES] encoder inputs of spells; ``age`` is the player's age at each spell's period."""
        n = len(sp["season"])
        x = np.zeros((n, N_FEATURES), dtype=np.float32)
        minutes, games, counts = sp["minutes"], sp["games"], sp["counts"]
        has = minutes > 0
        exposure = np.where(has, minutes / 40.0, 1.0)
        x[:, 0] = np.log1p(minutes) / 7.0
        x[:, 1] = np.log1p(games) / 4.0
        x[:, 2] = np.where(games > 0, minutes / np.maximum(games, 1.0), 0.0) / 40.0
        x[:, 3:13] = np.where(has[:, None], np.sqrt(counts[:, LC.RATE_INDEX] / exposure[:, None]), 0.0)
        makes, attempts = counts[:, LC.MAKE_INDEX], counts[:, LC.ATTEMPT_INDEX]
        x[:, 13:16] = (makes + 1.0) / (attempts + 2.0) - 0.5
        x[:, 16:19] = np.log1p(attempts) / 5.0
        known = sp["starts_known"] > 0
        x[:, 19] = np.where(known, sp["starts"] / np.maximum(sp["starts_known"], 1.0), 0.0)
        x[:, 20] = known
        for j, (name, col, flag) in enumerate((("team_adj_o", 21, 23), ("team_adj_d", 22, 23), ("opp_adj_o", 24, 26), ("opp_adj_d", 25, 26))):
            ok = sp[name + "_n"] > 0
            mean = np.where(ok, sp[name] / np.maximum(sp[name + "_n"], 1.0), self.strength_mu[j])
            x[:, col] = (mean - self.strength_mu[j]) / self.strength_sd[j]
            x[:, flag] = ok
        x[:, 27:31] = _age_basis(np.asarray(age, dtype=float))
        x[np.arange(n), 31 + sp["type"]] = 1.0
        x[:, 37] = has
        return x


@dataclass
class UnitArrays:
    """Targets, context and weights of a set of NCAA units, in the frame's order."""
    frame: pd.DataFrame
    static: np.ndarray
    static_mask: np.ndarray
    static_cats: np.ndarray
    hw: np.ndarray                       # [N, 4] standardized height and weight with their presence flags
    targets: dict
    weight: np.ndarray

    @classmethod
    def build(cls, ds: FoldDataset, units: pd.DataFrame, weight: np.ndarray | None = None) -> "UnitArrays":
        units = units.reset_index(drop=True)
        z, mask, cats = ds._unit_static(units)
        names = [n for n, _, _ in ds.store.schema.static_continuous]
        ih, iw = names.index("height_cm"), names.index("weight_kg")
        hw = np.stack([z[:, ih], mask[:, ih], z[:, iw], mask[:, iw]], axis=1).astype(np.float32)
        targets = {"games": units["games_played"].to_numpy(dtype=np.int64), "starts": units["starts"].to_numpy(dtype=np.int64),
                   "minutes": np.round(units["minutes"].to_numpy(dtype=float)),
                   "counts": units[list(F.TARGET_COUNTS)].to_numpy(dtype=np.int64),
                   "overtime": units["overtime_periods"].to_numpy(dtype=float),
                   "overtime_known": units["overtime_known"].to_numpy(dtype=bool),
                   "schedule": units["scheduled_games"].to_numpy(dtype=np.int64)}
        w = np.ones(len(units)) if weight is None else np.asarray(weight, dtype=float)
        return cls(units, z, mask, cats.astype(np.int64), hw, targets, w)

    def batch(self, idx: np.ndarray, repeat: int = 1, marginal: bool = False):
        """(SeasonBatch, static, mask, cats) of the units ``idx``, each repeated ``repeat`` times (draw-major order)."""
        t = lambda k, dt=None: torch.as_tensor(np.tile(self.targets[k][idx], (repeat,) + (1,) * (self.targets[k].ndim - 1)), dtype=dt)   # noqa: E731
        sb = SeasonBatch(t("games"), t("starts"), t("minutes", DT), t("counts"), t("schedule"),
                         None if marginal else t("overtime").long(), None if marginal else t("overtime_known"))
        rep = lambda a: torch.as_tensor(np.tile(a[idx], (repeat, 1)))      # noqa: E731
        return sb, rep(self.static), rep(self.static_mask), rep(self.static_cats)


def prefix_examples(source: SpellSource, ua: UnitArrays) -> list[dict]:
    """One filter sequence per unit: the spells released by the unit's forecast cutoff, NCAA rows from earlier seasons only,
    the unit's own season as the last period."""
    out = []
    fr = ua.frame
    cutoffs = [_days(c) for c in fr["forecast_cutoff"]]
    ages = fr["age_at_cutoff"].to_numpy(dtype=float)
    for i, (pid, t) in enumerate(zip(fr["player_id"].to_numpy(), fr["season"].to_numpy())):
        sp = source.spells(int(pid), cutoffs[i], int(t) - 1)
        out.append(_sequence(source, sp, target_season=int(t), age_at_target=ages[i], hw=ua.hw[i], unit=i))
    return out


def _sequence(source: SpellSource, sp: dict | None, *, target_season: int, age_at_target: float, hw: np.ndarray, unit: int | None = None,
              extra_seasons=()) -> dict:
    spell_season = np.minimum(sp["season"], target_season) if sp is not None else np.zeros(0, dtype=np.int64)
    seasons = np.unique(np.concatenate([spell_season, np.asarray(list(extra_seasons) + [target_season], dtype=np.int64)]))
    age = age_at_target + (seasons - target_season)
    period = np.searchsorted(seasons, spell_season)
    feat = source.features(sp, age[period]) if sp is not None else np.zeros((0, N_FEATURES), dtype=np.float32)
    dt = np.concatenate([[0.0], np.diff(seasons).astype(float)])
    age_prev = np.concatenate([[age[0]], age[:-1]])
    cov = np.concatenate([_age_basis(age[:1])[0], hw]).astype(np.float32)
    return {"unit": unit, "seasons": seasons, "dt": dt, "age_prev": age_prev, "cov": cov, "period": period, "feat": feat,
            "comp": sp["comp"] if sp is not None else np.zeros(0, dtype=np.int64), "spells": sp}


def career_examples(source: SpellSource, ua: UnitArrays, *, cutoff_days: float, ncaa_max_season: int, skip_players: set) -> list[dict]:
    """One smoothing sequence per training player: every admissible spell released by the training cutoff, the player's
    training units as NCAA likelihood terms, the non-NCAA spells as production-given-exposure terms."""
    fr = ua.frame
    out = []
    ages = fr["age_at_cutoff"].to_numpy(dtype=float)
    seasons_all = fr["season"].to_numpy()
    for pid, idx in fr.groupby("player_id", sort=False).indices.items():
        if pid in skip_players:
            continue
        idx = np.asarray(idx)
        last = idx[np.argmax(seasons_all[idx])]
        t = int(seasons_all[idx].max())
        sp = source.spells(int(pid), cutoff_days, ncaa_max_season)
        target = t if sp is None else max(t, int(sp["season"].max()))
        seq = _sequence(source, sp, target_season=target, age_at_target=ages[last] + (target - t), hw=ua.hw[last],
                        extra_seasons=[int(s) for s in seasons_all[idx]])
        seq["units"] = idx
        seq["unit_period"] = np.searchsorted(seq["seasons"], seasons_all[idx])
        if sp is not None:
            counts = sp["counts"]
            ok = (sp["type"] != SPELL_TYPES.index("ncaa")) & np.all(counts[:, LC.MAKE_INDEX] <= counts[:, LC.ATTEMPT_INDEX], axis=1)
            seq["emit"] = np.flatnonzero(ok)
        else:
            seq["emit"] = np.zeros(0, dtype=np.int64)
        out.append(seq)
    return out


def collate(seqs: list[dict]) -> dict:
    B_ = len(seqs)
    T = max(len(s["seasons"]) for s in seqs)
    dt, age_prev = np.zeros((B_, T)), np.full((B_, T), 20.0)
    for i, s in enumerate(seqs):
        n = len(s["seasons"])
        dt[i, :n], age_prev[i, :n] = s["dt"], s["age_prev"]
    obs_seq = np.concatenate([np.full(len(s["period"]), i, dtype=np.int64) for i, s in enumerate(seqs)])
    return {"cov": torch.as_tensor(np.stack([s["cov"] for s in seqs])), "dt": torch.as_tensor(dt), "age_prev": torch.as_tensor(age_prev),
            "obs_seq": torch.as_tensor(obs_seq), "obs_period": torch.as_tensor(np.concatenate([s["period"] for s in seqs]).astype(np.int64)),
            "obs_feat": torch.as_tensor(np.concatenate([s["feat"] for s in seqs]).astype(np.float32)),
            "obs_comp": torch.as_tensor(np.concatenate([s["comp"] for s in seqs]).astype(np.int64))}


def prefix_log_score(model, lik, seqs: list[dict], ua: UnitArrays, *, draws: int, generator=None, marginal: bool, parts: bool = False):
    """log (1/R) Σ_r p(Y | z_r, context) per unit [B] under the prefix filter-predictive state (reparameterized draws).
    With ``parts`` also the opportunity-marginal mixture and the mixture with the makes factors removed."""
    mf, Pf = model.infer(collate(seqs), smooth=False)
    m, P = mf[:, -1], Pf[:, -1]
    B_, d = m.shape
    eps = torch.randn(draws, B_, d, dtype=DT, generator=generator)
    z = draw(m.unsqueeze(0), P.unsqueeze(0), eps).reshape(draws * B_, d)
    idx = np.array([s["unit"] for s in seqs])
    sb, static, mask, cats = ua.batch(idx, repeat=draws, marginal=marginal)
    params = model.decode(z, static, mask, cats)
    lp, _ = lik.log_prob(sb, params=params)
    mix = lambda v: torch.logsumexp(v.view(draws, B_), dim=0) - math.log(draws)      # noqa: E731
    if not parts:
        return mix(lp)
    from ..calibration import MakesFactorLikelihood
    opp = lik.log_prob_opportunity(sb, params=params)
    makes, _ = MakesFactorLikelihood.log_prob(sb, params=params)
    return mix(lp), mix(opp), mix(lp - makes), params, (m, P)


def career_elbo(model, lik, seqs: list[dict], ua: UnitArrays, generator=None):
    """(sum of ELBOs over the careers, number of observation terms, mean KL, mean NCAA term, mean emission term)."""
    cb = collate(seqs)
    ms, Ps, kl = model.infer(cb, smooth=True)
    z = draw(ms, Ps, torch.randn(ms.shape, dtype=DT, generator=generator))
    unit_seq = torch.as_tensor(np.concatenate([np.full(len(s["units"]), i, dtype=np.int64) for i, s in enumerate(seqs)]))
    unit_period = torch.as_tensor(np.concatenate([s["unit_period"] for s in seqs]).astype(np.int64))
    idx = np.concatenate([s["units"] for s in seqs])
    sb, static, mask, cats = ua.batch(idx, marginal=False)
    lp_units, _ = lik.log_prob(sb, params=model.decode(z[unit_seq, unit_period], static, mask, cats))
    total = lp_units.sum() - kl.sum()
    n_emit, lp_emit_sum = 0, torch.zeros((), dtype=DT)
    emit_seq, emit_period, comp, kind, exposure, counts, has = [], [], [], [], [], [], []
    for i, s in enumerate(seqs):
        e = s["emit"]
        if len(e):
            sp = s["spells"]
            emit_seq.append(np.full(len(e), i, dtype=np.int64)); emit_period.append(s["period"][e]); comp.append(sp["comp"][e])
            kind.append(sp["type"][e]); exposure.append(sp["minutes"][e] / 40.0); counts.append(sp["counts"][e]); has.append(sp["minutes"][e] > 0)
    if emit_seq:
        cat = lambda a, dt=None: torch.as_tensor(np.concatenate(a), dtype=dt)       # noqa: E731
        ze = z[cat(emit_seq), cat(emit_period, torch.long)]
        lp_emit = model.emission_log_prob(ze, cat(comp, torch.long), cat(kind, torch.long), cat(exposure, DT), cat(counts, torch.long), cat(has))
        lp_emit_sum, n_emit = lp_emit.sum(), len(lp_emit)
        total = total + lp_emit_sum
    return total, len(idx) + n_emit, float(kl.mean().detach()), float(lp_units.mean().detach()), (float(lp_emit_sum.detach()) / n_emit if n_emit else None)


def score_units(model, lik, seqs: list[dict], ua: UnitArrays, *, draws: int, seed: int, chunk: int = 64, parts: bool = False):
    """Forward predictive NLL per unit (marginal convention), with common random numbers fixed by ``seed``."""
    gen = torch.Generator().manual_seed(seed)
    nll, opp, nomakes = (np.empty(len(seqs)) for _ in range(3))
    with torch.no_grad():
        for s in range(0, len(seqs), chunk):
            group = seqs[s:s + chunk]
            if parts:
                a, b, c, _, _ = prefix_log_score(model, lik, group, ua, draws=draws, generator=gen, marginal=True, parts=True)
                opp[s:s + len(group)], nomakes[s:s + len(group)] = (-b).numpy(), (-c).numpy()
            else:
                a = prefix_log_score(model, lik, group, ua, draws=draws, generator=gen, marginal=True)
            nll[s:s + len(group)] = (-a).numpy()
    return (nll, opp, nomakes) if parts else nll


def cohort_frame(units: pd.DataFrame) -> pd.DataFrame:
    b = lambda s: s.to_numpy(dtype=bool, na_value=False)        # noqa: E731
    tracked = (units["n_intl_games"] + units["n_national_games"] + units["n_event_games"] + units["n_ncaa_prior_games"]).to_numpy()
    hist = (units["n_intl_games"].to_numpy() > 0) | (units["n_national_games"].to_numpy() > 0)
    fy = b(units["first_year"])
    return pd.DataFrame({"unit_id": units["unit_id"].astype(str).to_numpy(), "player_id": units["player_id"].to_numpy(),
                         "team_id": units["team_id"].to_numpy(), "season": units["season"].to_numpy(), "first_year": fy,
                         "intl_entrant": b(units["intl_entrant"]), "first_year_with_history": fy & hist,
                         "first_year_without_history": fy & ~hist, "no_tracked_history": fy & (tracked == 0)})


def build_data(tables: Path, fold_season: int, arm: str, seed: int, *, test_season: int | None = None):
    """Tables, store, fold dataset, spell source and the unit sets of one fit (training ≤ v−2, selection v−1)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    tab = read_tables(tables)
    units = tab["units"]
    protocol_test = protocol_test_season() if test_season is None else int(test_season)
    if fold_season + 1 >= protocol_test:
        raise ValueError("the latent-career study scores development seasons only; the protocol's test season is out of reach")
    test_players = frozenset(units.loc[(units["season"] == protocol_test) & units["intl_entrant"], "player_id"])
    fs = F.resolve_schema(SCHEMA_NAME, SCHEMA_BLOCKS, units)
    store = F.GameStore.from_frame(tab["games"], dobs=F.player_dobs(tab), schema=fs, row_guard=True)
    fold = Fold(fold_season, "development")
    ds = FoldDataset.build(tab, fold, store, test_cohort_players=test_players, index=player_index(tab["games"]), seed=seed,
                           exclude_post_first_season=(arm == "strict"))
    if arm == "strict" and (ds.excluded_rows is None or int(ds.excluded_rows.sum()) == 0):
        raise RuntimeError("strict arm requested but no game rows are excluded")
    source = SpellSource(tab["games"], store, ds.excluded_rows)
    cutoff = _days(fold.training_cutoff)
    allowed = (store.release <= cutoff) & ~np.isin(store.player_id, np.fromiter(ds.heldout, dtype=np.int64))
    source.fit_strength(allowed, cutoff)
    tu = ds.train_units
    w_all, _ = example_weights(tu[tu["season"] <= fold_season - 2].reset_index(drop=True), None)
    train = UnitArrays.build(ds, tu[tu["season"] <= fold_season - 2], w_all)
    stop = UnitArrays.build(ds, tu[tu["season"] == fold_season - 1])
    return tab, ds, source, train, stop, protocol_test


def evaluation_units(tab: dict, ds: FoldDataset, season: int, arm: str) -> UnitArrays:
    """Likelihood-eligible units of ``season`` with the fit's own static normalization and vocabularies (the convention of
    bayes_export.py for the season after the fold's)."""
    units = tab["units"]
    ev = units[units["likelihood_eligible"] & (units["season"] == season)]
    if arm == "strict":
        ev = mask_post_first_season_statics(ev)
    return UnitArrays.build(ds, ev)


def run_latent(*, tables: Path, fold_season: int, arm: str, seed: int, out: Path, epochs: int = 20, batch_size: int = 256,
               lr: float = 1e-3, nodes_train: int = 24, nodes_eval: int = 32, train_draws: int = 2, career_batch: int = 96,
               stop_draws: int = 16, eval_draws: int = 64, threads: int = 8, max_wall_seconds: float | None = None,
               max_steps: int | None = None, max_train_units: int | None = None, resume: bool = False, patience: int = 5,
               baseline_rows: int = 5000, baseline_iter: int = 60, drift_l2: float = 1e-3, score: bool = True) -> dict:
    out = Path(out)
    if (out / "run.json").exists():
        raise FileExistsError(f"Refusing to overwrite the finished run {out}")
    ckpt_path = out / "checkpoint_last.pt"
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"{out} holds an unfinished run; pass resume=True to continue it")
    out.mkdir(parents=True, exist_ok=True)
    status = lambda stage, **extra: (out / "status.json").write_text(json.dumps(          # noqa: E731
        {"stage": stage, "arm": arm, "model": "latent_career", "fold": fold_season, "seed": seed, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}, indent=1))
    status("data")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    t0 = time.time()
    tab, ds, source, train, stop, protocol_test = build_data(Path(tables), fold_season, arm, seed)
    if max_train_units is not None:                    # throughput checks and tests only
        keep = np.sort(np.random.default_rng(seed).choice(len(train.frame), min(max_train_units, len(train.frame)), replace=False))
        train = UnitArrays.build(ds, train.frame.iloc[keep], train.weight[keep])
    stop_players = set(stop.frame["player_id"])
    train_seqs = prefix_examples(source, train)
    stop_seqs = prefix_examples(source, stop)
    careers = career_examples(source, train, cutoff_days=_days(ds.fold.training_cutoff), ncaa_max_season=fold_season - 2, skip_players=stop_players)
    t_data = time.time() - t0
    lik_train = DifferentiableSeasonLikelihood(None, quadrature_nodes=nodes_train)
    lik_eval = DifferentiableSeasonLikelihood(None, quadrature_nodes=nodes_eval)
    # Intercept-only baseline on the training units: the heads start from it, as in the release.
    base_model = InterceptOnlySeasonModel(SeasonParameters())
    base_lik = DifferentiableSeasonLikelihood(base_model, quadrature_nodes=nodes_train)
    base_idx = np.random.default_rng(seed).permutation(len(train.frame))[:baseline_rows]
    base_sb = train.batch(base_idx, marginal=False)[0]
    base_valid = base_lik._valid(base_sb)
    baseline_fit = lbfgs_fit(base_lik, base_sb.subset(base_valid.nonzero(as_tuple=False).squeeze(1)), max_iter=baseline_iter)
    baseline_fit["rows"], baseline_fit["invalid_dropped"] = int(base_valid.sum()), int((~base_valid).sum())
    static_sizes = [len(v) for v in ds.static_vocab.values()]
    model = LatentCareerModel(n_features=N_FEATURES, n_competitions=len(ds.store.vocab["competition"]), n_static=train.static.shape[1],
                              static_vocab_sizes=static_sizes, n_covariates=N_COVARIATES, reference=base_model.reference())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    forward_rng, career_rng = np.random.default_rng(seed), np.random.default_rng(seed + 11)
    curve, best, best_state, steps, wall = [], math.inf, None, 0, 0.0
    epoch_start, budget_exhausted, bad_epochs = 0, None, 0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        curve, best, best_state, steps, wall, epoch_start = ck["curve"], ck["best"], ck["best_state"], ck["steps"], ck["wall"], ck["epoch"]
        forward_rng.bit_generator.state, career_rng.bit_generator.state = ck["forward_rng"], ck["career_rng"]
        bad_epochs = ck.get("bad_epochs", 0)
        torch.set_rng_state(ck["torch_rng"])
    n_careers = max(len(careers), 1)
    weights = train.weight
    for epoch in range(epoch_start, epochs):
        model.train()
        t_epoch = time.time()
        losses, kls, lp_u, lp_e = [], [], [], []
        order = forward_rng.permutation(len(train_seqs))
        career_order, pos = career_rng.permutation(len(careers)), 0
        for s in range(0, len(order), batch_size):
            group = [train_seqs[j] for j in order[s:s + batch_size]]
            lp = prefix_log_score(model, lik_train, group, train, draws=train_draws, marginal=False)
            if not bool(torch.isfinite(lp).all()):
                bad = [train.frame["unit_id"].iloc[g["unit"]] for g, ok in zip(group, torch.isfinite(lp).tolist()) if not ok]
                raise RuntimeError(f"non-finite prefix score in a training batch; unit ids {bad[:10]}")
            w = torch.as_tensor(weights[[g["unit"] for g in group]], dtype=DT)
            loss = (w * -lp).sum() / w.sum()
            if careers:
                if pos + career_batch > len(career_order):
                    career_order, pos = career_rng.permutation(len(careers)), 0
                cgroup = [careers[j] for j in career_order[pos:pos + career_batch]]
                pos += career_batch
                elbo, n_terms, kl, u_mean, e_mean = career_elbo(model, lik_train, cgroup, train)
                if not bool(torch.isfinite(elbo)):
                    raise RuntimeError("non-finite career ELBO in a training batch")
                loss = loss - elbo / n_terms
                kls.append(kl); lp_u.append(u_mean)
                if e_mean is not None:
                    lp_e.append(e_mean)
            comp_pen, drift_pen = model.penalty()
            loss = loss + comp_pen / n_careers + drift_l2 * drift_pen
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        model.eval()
        stop_nll = float(np.mean(score_units(model, lik_eval, stop_seqs, stop, draws=stop_draws, seed=seed + 101)))
        wall += time.time() - t_epoch
        curve.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "stop_nll": stop_nll, "mean_kl_per_career": float(np.mean(kls)) if kls else None,
                      "mean_ncaa_term": float(np.mean(lp_u)) if lp_u else None, "mean_emission_term": float(np.mean(lp_e)) if lp_e else None,
                      "wall_seconds": time.time() - t_epoch, "steps": steps, **{k: v for k, v in model.state_summary().items() if k in ("A_norm", "B_norm")}})
        print(json.dumps({"fold": fold_season, "arm": arm, "model": "latent_career", **curve[-1]}), file=sys.stderr, flush=True)
        if stop_nll < best - 1e-6:
            best, best_state, bad_epochs = stop_nll, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "curve": curve, "best": best, "best_state": best_state,
                    "steps": steps, "wall": wall, "epoch": epoch + 1, "bad_epochs": bad_epochs, "forward_rng": forward_rng.bit_generator.state,
                    "career_rng": career_rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, ckpt_path)
        status("training", epochs_completed=epoch + 1, epochs_max=epochs, best_stop_nll=best, steps=steps, active_wall_seconds=wall)
        if max_steps is not None and steps >= max_steps:
            budget_exhausted = "max_steps"
            break
        if bad_epochs >= patience:          # registered early stop: no improvement of the selection score for `patience` epochs
            break
        if max_wall_seconds is not None and epoch + 1 < epochs and wall + wall / (epoch + 1) > max_wall_seconds:
            budget_exhausted = "max_wall_seconds"
            break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, out / "model_state.pt")
    artifact = {"architecture": model.architecture, "state_dict": best_state, "arm": arm, "fold": fold_season, "seed": seed,
                "static_vocab": ds.static_vocab, "competition_vocab": ds.store.vocab["competition"],
                "static_mu": ds.normalizer.static_mu, "static_sigma": ds.normalizer.static_sigma,
                "strength_mu": source.strength_mu, "strength_sd": source.strength_sd,
                "feature_schema": {"name": SCHEMA_NAME, "blocks": list(SCHEMA_BLOCKS)}, "n_features": N_FEATURES}
    torch.save(artifact, out / "model_artifact.pt")
    status("scoring", best_stop_nll=best)
    evaluation, timing_scores = {}, {}
    if score:
        for season in (fold_season, fold_season + 1):
            t_s = time.time()
            ev = evaluation_units(tab, ds, season, arm)
            seqs = prefix_examples(source, ev)
            nll, opp, nomakes = score_units(model, lik_eval, seqs, ev, draws=eval_draws, seed=seed + 1000 + season, parts=True)
            frame = cohort_frame(ev.frame)
            frame["nll"], frame["nll_opportunity"], frame["nll_without_makes"] = nll, opp, nomakes
            frame["n_prefix_spells"] = [len(s["period"]) for s in seqs]
            if season == fold_season + 1:
                # The registered evaluation: an independent replicate of the draws (Monte Carlo error), a larger draw count
                # for the entrants, and decoded parameter draws of the first-year units for the calibration diagnostics.
                fy_idx = np.flatnonzero(frame["first_year"].to_numpy())
                fy_seqs = [seqs[i] for i in fy_idx]
                frame["nll_replicate"] = np.nan
                frame.loc[fy_idx, "nll_replicate"] = score_units(model, lik_eval, fy_seqs, ev, draws=eval_draws, seed=seed + 5000 + season)
                ent_idx = np.flatnonzero((frame["first_year"] & frame["intl_entrant"]).to_numpy())
                frame["nll_draws_256"] = np.nan
                frame.loc[ent_idx, "nll_draws_256"] = score_units(model, lik_eval, [seqs[i] for i in ent_idx], ev, draws=256, seed=seed + 7000 + season, chunk=16)
                gen, kept, n_sets = torch.Generator().manual_seed(seed + 9000 + season), {}, 32
                with torch.no_grad():
                    for s0 in range(0, len(fy_seqs), 64):
                        group = fy_seqs[s0:s0 + 64]
                        params = prefix_log_score(model, lik_eval, group, ev, draws=n_sets, generator=gen, marginal=True, parts=True)[3]
                        for k, v in params.items():
                            kept.setdefault(k, []).append(v.view(n_sets, len(group), *v.shape[1:]).to(torch.float32).numpy())
                np.savez_compressed(out / f"parameter_draws_{season}.npz", unit_id=frame["unit_id"].to_numpy()[fy_idx].astype(str),
                                    **{k: np.concatenate(v, axis=1) for k, v in kept.items()})
            frame.to_csv(out / f"per_unit_{season}.csv", index=False)
            fy = frame["first_year"].to_numpy()
            evaluation[str(season)] = {"units": int(len(frame)), "nonfinite": int((~np.isfinite(nll)).sum()), "nll_all": float(nll.mean()),
                                       "nll_first_year": float(nll[fy].mean()), "nll_intl_entrants": float(nll[frame["intl_entrant"].to_numpy()].mean()),
                                       "draws": eval_draws}
            timing_scores[str(season)] = time.time() - t_s
    from ..model.artifact import code_sha256
    from ..model import torch_likelihood as lik_module, tower as tower_module
    from . import latent as this_module
    payload = {"kind": "latent_career_run", "arm": arm, "fold": fold_season, "seed": seed, "tables": str(tables),
               "parameters": int(sum(p.numel() for p in model.parameters())),
               "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "nodes_train": nodes_train, "nodes_eval": nodes_eval,
                          "train_draws": train_draws, "career_batch": career_batch, "stop_draws": stop_draws, "eval_draws": eval_draws,
                          "drift_l2": drift_l2, "patience": patience, "max_wall_seconds": max_wall_seconds, "max_steps": max_steps, "max_train_units": max_train_units,
                          "train_seasons_through": fold_season - 2, "stop_season": fold_season - 1, "scored_seasons": [fold_season, fold_season + 1],
                          "protocol_test_season": protocol_test, "architecture": model.architecture,
                          "post_first_season_logs": "excluded everywhere (D-041)" if arm == "strict" else "included",
                          "excluded_game_rows": int(ds.excluded_rows.sum()) if ds.excluded_rows is not None else 0,
                          "objective": "prefix predictive mixture NLL (release forward minibatch, player-balanced weights) + career ELBO per "
                                       "observation term on an independent career batch + competition prior / careers + drift L2",
                          "score_definition": "forward predictive mixture over prefix filter draws; marginal NLL of (G, J, M, B), overtime summed out",
                          "tables_manifest": {"payload_sha256": tab["manifest"].get("payload_sha256")}},
               "counts": {"train_units": len(train.frame), "stop_units": len(stop.frame), "careers": len(careers),
                          "career_spells": int(sum(len(c["period"]) for c in careers)), "career_emission_terms": int(sum(len(c["emit"]) for c in careers)),
                          "train_prefix_with_spells": int(sum(len(s["period"]) > 0 for s in train_seqs)), "heldout_players": len(ds.heldout),
                          "stop_players_without_career": len(stop_players)},
               "baseline": {"fit": baseline_fit},
               "training": {"curve": curve, "best_stop_nll": best, "epochs_run": len(curve), "steps": steps, "wall_seconds": wall,
                            "budget_exhausted": budget_exhausted},
               "latent": model.state_summary(), "evaluation": evaluation,
               "timing": {"data_seconds": t_data, "score_seconds": timing_scores, "total_seconds": time.time() - t0},
               "artifact": {"path": "model_artifact.pt", "sha256": file_sha256(out / "model_artifact.pt")},
               "code": code_sha256(this_module, LC, tower_module, lik_module, F)}
    (out / "run.json").write_text(json.dumps(payload, indent=1, default=float))
    status("done", best_stop_nll=best, total_seconds=time.time() - t0)
    return payload
