"""The common pretraining pool $\\mathcal P$ and its next-game objective (paper §2.4, App. A; D-032).

Pool records are the fold-admissible non-NCAA games (released by the
training cutoff) of every player who is not held out — international-only
players and NCAA-linked players alike, with their NCAA correspondence,
prior NCAA inputs, destination context, target age/experience, $\\Delta$,
and target-anchored clocks hidden. One example is a (player, next game)
pair: the prefix is every game of that player dated before the next game
and released by its date, encoded exactly as a forward history but with
clocks anchored at the *prediction date* (the next game's date); the
target is the next game's opportunity and production (appearance,
starter where known, recorded minutes to the second, the 13 counts,
observed overtime), scored by the season likelihood with $S = G = 1$ and
a one-second recording grid. The next game's context — competition,
venue, and the strengths available at the prediction date — is the only
"destination" information, supplied to a separate pretraining head.
Targets are restricted to 40-minute competitions so one regulation length
serves the whole pool.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import features as F

CONTEXT_FIELDS = ["regulation", "home", "opp_adj_o", "opp_adj_d", "opp_adj_pace", "team_adj_o", "team_adj_d", "team_adj_pace",
                  "opp_hist_winpct", "opp_hist_placement", "team_hist_winpct", "team_hist_placement",
                  "opp_record_winpct", "opp_record_margin"]
CONTEXT_INDEX = [F.FIELD_NAMES.index(f) for f in CONTEXT_FIELDS]


def context_index(field_names: list[str]) -> list[int]:
    """Positions of the context fields in a schema's field list (the context fields are base fields, D-065)."""
    return [list(field_names).index(f) for f in CONTEXT_FIELDS]
TARGET_REGULATION = 40.0
MINUTES_STEP = 1.0 / 60.0     # international game minutes are recorded to the second
MAX_TARGET_MINUTES = 65.0     # a 40-minute game with up to five overtimes; larger values are data defects
MAX_TARGET_FOULS = 5          # the season likelihood caps fouls at 5 per game (foul_limit_per_game)


@dataclass
class PoolExample:
    player_id: int
    target_row: int
    intl: F.Channel
    context: np.ndarray        # standardized context fields of the next game
    context_mask: np.ndarray
    context_cats: np.ndarray   # category codes of the next game (competition, kind, country, age group, division, source)
    static: np.ndarray
    static_mask: np.ndarray
    time: np.ndarray
    time_mask: np.ndarray
    targets: dict


class PoolSampler:
    """Samples next-game examples from the pool for one fold."""

    def __init__(self, store: F.GameStore, normalizer: F.Normalizer, *, cutoff_days: float, heldout_players,
                 min_prefix: int = 1, seed: int = 0, excluded_rows=None):
        self.store, self.normalizer = store, normalizer
        self.cutoff = float(cutoff_days)
        self.rng = np.random.default_rng(seed)
        self.min_prefix = min_prefix
        col = {name: j for j, name in enumerate(store.schema.field_names)}
        self.context_index = context_index(store.schema.field_names)
        raw = store.raw
        minutes_raw = np.expm1(raw[:, col["minutes"]])                  # fields are stored log1p-transformed
        minutes_ok = ~np.isnan(minutes_raw) & (minutes_raw >= 0) & (minutes_raw <= MAX_TARGET_MINUTES)
        counts_ok = ~np.isnan(raw[:, [col[f] for f in F.TARGET_COUNTS]]).any(axis=1)
        counts_ok &= np.expm1(raw[:, col["pf"]]) <= MAX_TARGET_FOULS + 1e-6
        for make, attempt in (("k2", "a2"), ("k3", "a3"), ("kf", "af")):        # impossible lines are not targets
            counts_ok &= np.expm1(raw[:, col[make]]) <= np.expm1(raw[:, col[attempt]]) + 1e-6
        regulation_ok = raw[:, col["regulation"]] == TARGET_REGULATION
        admissible = (~store.is_ncaa) & (store.release <= self.cutoff) & ~np.isin(store.player_id, np.fromiter(heldout_players, dtype=np.int64))
        if excluded_rows is not None:                 # D-041 strict control: post-first-season logs leave the pool too
            admissible &= ~np.asarray(excluded_rows, dtype=bool)
        self.admissible = admissible
        eligible_target = admissible & minutes_ok & counts_ok & regulation_ok
        # A target needs a nonempty admissible prefix: at least ``min_prefix`` earlier non-NCAA games of the player.
        order = np.zeros(len(store.date), dtype=np.int64)
        for pid, (s, e) in store.index.items():
            rows = np.arange(s, e)
            rows = rows[~store.is_ncaa[rows] & admissible[rows]]
            order[rows] = np.arange(len(rows))
        self.targets = np.flatnonzero(eligible_target & (order >= min_prefix))
        self.n_players = int(len(np.unique(store.player_id[self.targets])))

    def __len__(self):
        return int(len(self.targets))

    def sample(self, n: int) -> list[PoolExample]:
        chosen = self.rng.choice(self.targets, size=min(n, len(self.targets)), replace=False)
        return [self.example(int(r)) for r in chosen]

    def example(self, row: int) -> PoolExample:
        store = self.store
        pid = int(store.player_id[row])
        target_date = float(store.date[row])
        rows = store.rows(pid)
        prefix = rows[~store.is_ncaa[rows] & self.admissible[rows] & (store.date[rows] < target_date) & (store.release[rows] <= target_date)]
        # Prefix clocks and strengths are computed at the prediction date, not at any NCAA cutoff.
        intl = F.build_channel(store, prefix, cutoff_days=target_date, target_days=target_date, normalizer=self.normalizer)
        raw = store.continuous(np.array([row]), target_date)
        z, mask = self.normalizer.games(raw)
        context, context_mask = z[0, self.context_index], mask[0, self.context_index]
        # Static inputs: everything NCAA- or destination-derived is hidden; only age (when known) survives.
        static = np.full(len(store.schema.static_continuous), np.nan)     # age_at_cutoff is index 0 in every schema
        col = {name: j for j, name in enumerate(store.schema.field_names)}
        age = raw[0, col["age_at_game"]]
        static[0] = age
        s_z, s_mask = self.normalizer.static(static[None, :])
        last = float(store.date[prefix[-1]]) if len(prefix) else np.nan
        delta = (last - target_date) / F.YEAR_DAYS if len(prefix) else np.nan
        source_age = age + delta if not np.isnan(age) and not np.isnan(delta) else np.nan
        t_v, t_m = F.time_basis(delta, source_age, age)
        starter = store.raw[row, col["starter"]]
        counts = store.raw[row, [col[f] for f in F.TARGET_COUNTS]]
        overtime = store.raw[row, col["overtime"]]
        minutes_value = float(np.expm1(store.raw[row, col["minutes"]]))     # feature fields are stored log1p-transformed
        targets = {"games": 1, "starts": int(starter) if not np.isnan(starter) else 0, "starts_known": not np.isnan(starter),
                   "minutes": float(round(minutes_value * 60.0) / 60.0),
                   "counts": np.expm1(counts).round().astype(np.int64),
                   "overtime": float(overtime) if not np.isnan(overtime) else 0.0, "overtime_known": not np.isnan(overtime),
                   "schedule": 1}
        targets["minutes"] = min(targets["minutes"], TARGET_REGULATION + 5.0 * targets["overtime"]) if targets["overtime_known"] else targets["minutes"]
        return PoolExample(pid, row, intl, context.astype(np.float32), context_mask, store.cats[row].astype(np.int64),
                           s_z[0], s_mask[0], t_v, t_m, targets)


def collate_pool(examples: list[PoolExample], n_static_categorical: int = len(F.STATIC_CATEGORICAL), n_fields: int | None = None) -> dict:
    empty = F.Channel.empty(len(F.FIELD_NAMES) if n_fields is None else int(n_fields))
    batch = {
        "intl": F._pad_channel([e.intl for e in examples]),
        "ncaa": F._pad_channel([empty for _ in examples]),
        "context": torch.from_numpy(np.stack([e.context for e in examples])),
        "context_mask": torch.from_numpy(np.stack([e.context_mask for e in examples])),
        "context_cats": torch.from_numpy(np.stack([e.context_cats for e in examples])),
        "static": torch.from_numpy(np.stack([e.static for e in examples])),
        "static_mask": torch.from_numpy(np.stack([e.static_mask for e in examples])),
        "static_cats": torch.zeros(len(examples), n_static_categorical, dtype=torch.long),
        "time": torch.from_numpy(np.stack([e.time for e in examples])),
        "time_mask": torch.from_numpy(np.stack([e.time_mask for e in examples])),
        "games": torch.ones(len(examples), dtype=torch.long),
        "starts": torch.tensor([e.targets["starts"] for e in examples], dtype=torch.long),
        "starts_known": torch.tensor([e.targets["starts_known"] for e in examples], dtype=torch.bool),
        "minutes": torch.tensor([e.targets["minutes"] for e in examples], dtype=torch.float64),
        "counts": torch.from_numpy(np.stack([e.targets["counts"] for e in examples])),
        "overtime": torch.tensor([e.targets["overtime"] for e in examples], dtype=torch.float64),
        "overtime_known": torch.tensor([e.targets["overtime_known"] for e in examples], dtype=torch.bool),
        "schedule": torch.ones(len(examples), dtype=torch.long),
        "player_id": [e.player_id for e in examples],
    }
    return batch
