"""Fold datasets: examples for both directions from the assembled tables.

For fold $v$ (``folds.Fold``), training examples are

* forward (``-``): every eligible training unit $(i,t)$, $t<v$, with the
  international channel = non-NCAA games released by $c_t$ and the NCAA
  channel = the player's NCAA games from seasons before $t$ released by
  $c_t$;
* reconstruction (``+``): every nested window from
  ``folds.reconstruction_windows`` (games after the unit's last NCAA game,
  released by $c^{\\rm tr}_v$), with the same NCAA channel as the forward
  example of that unit.

Evaluation examples are the eligible units of season $v$ in the forward
direction with the fold's forecast cutoff. Loss weights follow the paper's
player-balanced rule per direction (equal mass to players, then seasons
within player, then windows within season), rescaled to mean one.

Normalization statistics and category vocabularies are fitted on the
fold's training games only (games released by $c^{\\rm tr}_v$ whose players
are not held out), never on evaluation rows.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.folds import Fold, fold_membership, player_index
from . import features as F


def _days(ts) -> float:
    return float((np.datetime64(pd.Timestamp(ts), "D") - np.datetime64("1970-01-01", "D")) / F.DAY)


@dataclass
class FoldDataset:
    fold: Fold
    store: F.GameStore
    normalizer: F.Normalizer
    static_vocab: dict
    train_units: pd.DataFrame
    eval_units: pd.DataFrame
    windows: pd.DataFrame
    heldout: set
    excluded_rows: np.ndarray | None = None      # D-041 strict control: game rows removed from every input channel

    ASSUMED_ZERO = "NO_SEASON_LINE_AND_TEAM_MINUTE_SUM_CONSISTENT"

    @classmethod
    def build(cls, tables: dict, fold: Fold, store: F.GameStore, *, test_cohort_players=frozenset(),
              index: dict | None = None, normalizer_rows: int = 400_000, seed: int = 0,
              exclude_assumed_zeros: bool = False, k_values=(1, 2, 3), max_years: float | None = 4,
              exclude_post_first_season: bool = False, label_scope: str = "all") -> "FoldDataset":
        m = fold_membership(tables, fold, test_cohort_players=test_cohort_players, index=index,
                            k_values=k_values, max_years=max_years, label_scope=label_scope)
        train, evaluate, windows = m["train_units"], m["evaluation_units"], m["windows"]
        if exclude_assumed_zeros:
            # Sensitivity arm (D-028): zero-appearance labels that rest on provider-consistent absence
            # (no season line, consistent team minutes) are dropped from training; scoring keeps every unit.
            train = train[train["evidence"] != cls.ASSUMED_ZERO]
            windows = windows[windows["unit_id"].isin(set(train["unit_id"]))] if len(windows) else windows
        excluded = post_first_season_mask(store, tables["units"]) if exclude_post_first_season else None
        if excluded is not None and len(windows):
            windows = windows.iloc[0:0]          # every reconstruction window lies after the first season by construction
        cutoff = _days(fold.training_cutoff)
        # Normalizer on training games only: released by the training cutoff, players not held out.
        allowed = (store.release <= cutoff) & ~np.isin(store.player_id, np.fromiter(m["heldout_players"], dtype=np.int64))
        rows = np.flatnonzero(allowed)
        rng = np.random.default_rng(seed)
        if len(rows) > normalizer_rows:
            rows = np.sort(rng.choice(rows, normalizer_rows, replace=False))
        vocab = F.static_vocab(train)
        normalizer = F.Normalizer.fit(store, rows, cutoff, F.static_continuous(train))
        return cls(fold, store, normalizer, vocab, train.reset_index(drop=True), evaluate.reset_index(drop=True),
                   windows, m["heldout_players"], excluded)

    # ------------------------------------------------------------------ rows per unit
    def _channels(self, unit, cutoff_days: float):
        rows = self.store.rows(unit.player_id)
        released = self.store.release[rows] <= cutoff_days
        ncaa = rows[released & self.store.is_ncaa[rows] & (self.store.season[rows] < unit.season)]
        keep = released & ~self.store.is_ncaa[rows]
        if self.excluded_rows is not None:
            keep &= ~self.excluded_rows[rows]
        return rows[keep], ncaa

    def _window_rows(self, unit, window) -> np.ndarray:
        rows = self.store.rows(unit.player_id)
        rows = rows[~self.store.is_ncaa[rows]]
        if self.excluded_rows is not None:
            rows = rows[~self.excluded_rows[rows]]
        wanted = set(window.record_ids)
        keep = np.fromiter((r in wanted for r in self.store.record_id[rows]), dtype=bool, count=len(rows))
        return rows[keep]

    def _unit_static(self, units: pd.DataFrame):
        raw = F.static_continuous(units)
        z, mask = self.normalizer.static(raw)
        return z, mask, F.static_codes(units, self.static_vocab)

    # ------------------------------------------------------------------ examples
    def training_examples(self) -> list[F.Example]:
        units = self.train_units
        z, mask, cats = self._unit_static(units)
        forward, recon = [], []
        # Player-balanced weights, forward: players -> seasons.
        seasons_per_player = units.groupby("player_id")["season"].nunique()
        n_players = len(seasons_per_player)
        unit_pos = {u: i for i, u in enumerate(units["unit_id"])}
        for i, unit in enumerate(units.itertuples(index=False)):
            cutoff = _days(unit.forecast_cutoff)
            intl, ncaa = self._channels(unit, cutoff)
            w = 1.0 / (n_players * seasons_per_player[unit.player_id])
            forward.append(F.build_example(unit, store=self.store, normalizer=self.normalizer, static=z[i], static_mask=mask[i],
                                           static_cats=cats[i], intl_rows=intl, ncaa_rows=ncaa, cutoff_days=cutoff,
                                           direction="-", weight=w))
        if len(self.windows):
            wins = self.windows
            players = wins.groupby("player_id")["season"].nunique()
            per_unit = wins.groupby("unit_id").size()
            cutoff = _days(self.fold.training_cutoff)
            for window in wins.itertuples(index=False):
                i = unit_pos[window.unit_id]
                unit = units.iloc[i]
                _, ncaa = self._channels(unit, _days(unit.forecast_cutoff))
                intl = self._window_rows(unit, window)
                w = 1.0 / (len(players) * players[window.player_id] * per_unit[window.unit_id])
                recon.append(F.build_example(unit, store=self.store, normalizer=self.normalizer, static=z[i], static_mask=mask[i],
                                             static_cats=cats[i], intl_rows=intl, ncaa_rows=ncaa, cutoff_days=cutoff,
                                             direction="+", k=window.k, weight=w))
        for group in (forward, recon):
            total = sum(e.weight for e in group)
            for e in group:
                e.weight = e.weight * len(group) / total if total > 0 else 1.0
        return forward + recon

    def evaluation_examples(self) -> list[F.Example]:
        units = self.eval_units
        z, mask, cats = self._unit_static(units)
        cutoff = _days(self.fold.forecast_cutoff)
        out = []
        for i, unit in enumerate(units.itertuples(index=False)):
            intl, ncaa = self._channels(unit, cutoff)
            out.append(F.build_example(unit, store=self.store, normalizer=self.normalizer, static=z[i], static_mask=mask[i],
                                       static_cats=cats[i], intl_rows=intl, ncaa_rows=ncaa, cutoff_days=cutoff, direction="-"))
        return out


def post_first_season_mask(store: F.GameStore, units: pd.DataFrame) -> np.ndarray:
    """D-041 strict control: non-NCAA game rows dated after the END of the player's first NCAA season (the
    earliest roster season in the tables), i.e. the research definition of "post-freshman" (after the target
    freshman season: later college-summer national-team play and professional careers), and the same boundary
    as the reconstruction windows (``lower = period_end + 1 day``). Games played during the first season stay.
    Excluding the marked rows from every input channel, from the pool and from reconstruction makes the
    control genuinely free of post-freshman logs; freshman evaluation inputs end at the preseason cutoff and
    are unaffected."""
    first = units.sort_values(["player_id", "period_start"]).drop_duplicates("player_id").set_index("player_id")["period_end"]
    first = pd.to_datetime(first)
    end_days = {int(p): _days(d) for p, d in first.items() if pd.notna(d)}
    row_end = np.array([end_days.get(int(p), np.nan) for p in store.player_id], dtype=float)
    with np.errstate(invalid="ignore"):
        return (~store.is_ncaa) & (store.date > row_end)


def cohort_masks(units: pd.DataFrame) -> dict:
    """Evaluation strata: the primary cohort and the information-volume splits (D-023)."""
    b = lambda s: s.to_numpy(dtype=bool, na_value=False)  # noqa: E731  (parquet-backed columns may be Arrow arrays)
    return {
        "intl_entrant": b(units["intl_entrant"]),
        "first_year": b(units["first_year"]),
        "later_year_intl": b(units["intl_history"] & ~units["first_year"]),
        "no_tracked_history": b((units["n_intl_games"] + units["n_national_games"] + units["n_event_games"]) == 0) & b(units["first_year"]),
        "club_50plus": b(units["n_intl_games"] >= 50) & b(units["first_year"]),
    }
