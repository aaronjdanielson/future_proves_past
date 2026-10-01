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
within player, then windows within season), rescaled to mean one; or, under
``weighting="season_balanced"`` (D-062), equal mass to every unit (windows
within unit), with the first-season units of each direction carrying
``first_year_share`` of the direction's mass when that share is given.

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
    weighting: str = "player_balanced"           # D-062: "player_balanced" (registered) or "season_balanced"
    first_year_share: float | None = None        # D-062: mass share of first-season units per direction; None = natural
    future: pd.Series | None = None              # D-070: standardized later-career production per training unit_id

    ASSUMED_ZERO = "NO_SEASON_LINE_AND_TEAM_MINUTE_SUM_CONSISTENT"
    WEIGHTINGS = ("player_balanced", "season_balanced")

    @classmethod
    def build(cls, tables: dict, fold: Fold, store: F.GameStore, *, test_cohort_players=frozenset(),
              index: dict | None = None, normalizer_rows: int = 400_000, seed: int = 0,
              exclude_assumed_zeros: bool = False, k_values=(1, 2, 3), max_years: float | None = 4,
              exclude_post_first_season: bool = False, label_scope: str = "all",
              weighting: str = "player_balanced", first_year_share: float | None = None) -> "FoldDataset":
        validate_weighting(weighting, first_year_share)
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
        if exclude_post_first_season:
            # Static summaries of games (schema v9 blocks) must obey the same exclusion as the game rows (D-065).
            train, evaluate = mask_post_first_season_statics(train), mask_post_first_season_statics(evaluate)
        cutoff = _days(fold.training_cutoff)
        # Normalizer on training games only: released by the training cutoff, players not held out.
        allowed = (store.release <= cutoff) & ~np.isin(store.player_id, np.fromiter(m["heldout_players"], dtype=np.int64))
        if excluded is not None:
            allowed &= ~excluded          # D-072: the strict control's normalization statistics exclude the same rows as its inputs
        rows = np.flatnonzero(allowed)
        rng = np.random.default_rng(seed)
        if len(rows) > normalizer_rows:
            rows = np.sort(rng.choice(rows, normalizer_rows, replace=False))
        vocab = F.static_vocab(train, store.schema)                                  # the store's feature schema (D-065)
        normalizer = F.Normalizer.fit(store, rows, cutoff, F.static_continuous(train, store.schema))
        return cls(fold, store, normalizer, vocab, train.reset_index(drop=True), evaluate.reset_index(drop=True),
                   windows, m["heldout_players"], excluded, weighting, None if first_year_share is None else float(first_year_share))

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
        raw = F.static_continuous(units, self.store.schema)
        z, mask = self.normalizer.static(raw)
        return z, mask, F.static_codes(units, self.static_vocab)

    # ------------------------------------------------------------------ examples
    def training_examples(self) -> list[F.Example]:
        units = self.train_units
        z, mask, cats = self._unit_static(units)
        forward, recon = [], []
        w_forward, w_recon = example_weights(units, self.windows, weighting=self.weighting, first_year_share=self.first_year_share)
        unit_pos = {u: i for i, u in enumerate(units["unit_id"])}
        for i, unit in enumerate(units.itertuples(index=False)):
            cutoff = _days(unit.forecast_cutoff)
            intl, ncaa = self._channels(unit, cutoff)
            forward.append(F.build_example(unit, store=self.store, normalizer=self.normalizer, static=z[i], static_mask=mask[i],
                                           static_cats=cats[i], intl_rows=intl, ncaa_rows=ncaa, cutoff_days=cutoff,
                                           direction="-", weight=float(w_forward[i])))
        if len(self.windows):
            wins = self.windows
            cutoff = _days(self.fold.training_cutoff)
            for j, window in enumerate(wins.itertuples(index=False)):
                i = unit_pos[window.unit_id]
                unit = units.iloc[i]
                _, ncaa = self._channels(unit, _days(unit.forecast_cutoff))
                intl = self._window_rows(unit, window)
                recon.append(F.build_example(unit, store=self.store, normalizer=self.normalizer, static=z[i], static_mask=mask[i],
                                             static_cats=cats[i], intl_rows=intl, ncaa_rows=ncaa, cutoff_days=cutoff,
                                             direction="+", k=window.k, weight=float(w_recon[j])))
        if self.future is not None and len(self.future):
            for e in forward + recon:
                v = self.future.get(e.unit_id, np.nan)
                if np.isfinite(v):
                    e.targets["future"], e.targets["future_known"] = float(v), True
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


def future_targets(tables: dict, fold: Fold, units: pd.DataFrame, heldout_players, *, min_games: int = 10) -> pd.Series:
    """D-070 auxiliary target per training unit: the player's later professional production, points + rebounds +
    assists per 40 minutes over his international club games after his last NCAA season that were released by the
    fold's training cutoff (at least ``min_games``), standardized over the training units that have one. Held-out players
    (the stopping season's, the evaluation cohort's, the test cohort's) get no target, so nothing dated after their
    cutoffs shapes the representation that scores them (D-033). Indexed by unit_id; absent = unknown."""
    last_end = pd.to_datetime(tables["units"].groupby("player_id")["period_end"].max())
    g = tables["games"]
    g = g[(g["source"] == "intl") & g["player_id"].isin(units["player_id"]) & ~g["player_id"].isin(set(int(p) for p in heldout_players))]
    g = g[pd.to_datetime(g["release_date"]) <= fold.training_cutoff]
    g = g.merge(last_end.rename("last_end"), left_on="player_id", right_index=True)
    g = g[pd.to_datetime(g["date"]) > g["last_end"]]
    agg = g.groupby("player_id").agg(games=("date", "size"), minutes=("minutes", "sum"), pts=("pts", "sum"), orb=("orb", "sum"),
                                     drb=("drb", "sum"), ast=("ast", "sum"))
    agg = agg[(agg["games"] >= min_games) & (agg["minutes"] >= 100)]
    pra40 = 40.0 * (agg["pts"] + agg["orb"] + agg["drb"] + agg["ast"]) / agg["minutes"]
    if len(pra40) < 2:
        return pd.Series(dtype=float)
    z = (pra40 - pra40.mean()) / (pra40.std(ddof=0) + 1e-9)
    out = units[["unit_id", "player_id"]].merge(z.rename("future"), left_on="player_id", right_index=True, how="inner")
    return out.set_index("unit_id")["future"]


def validate_weighting(weighting: str, first_year_share: float | None) -> None:
    if weighting not in FoldDataset.WEIGHTINGS:
        raise ValueError(f"weighting must be one of {FoldDataset.WEIGHTINGS}, got {weighting!r}")
    if first_year_share is not None and not (0.0 < float(first_year_share) < 1.0):
        raise ValueError("first_year_share must lie strictly between 0 and 1, or be None for the natural share")


def first_year_flags(units: pd.DataFrame) -> np.ndarray:
    """First-season flag per unit row (False everywhere when the table has no such column, e.g. synthetic fixtures)."""
    if "first_year" not in units:
        return np.zeros(len(units), dtype=bool)
    return units["first_year"].to_numpy(dtype=bool, na_value=False)


def stratum_weights(base, first_year, share: float | None) -> np.ndarray:
    """Rescale ``base`` weights so the first-season rows carry ``share`` of the total mass (D-062).

    ``share`` None keeps the base weights (the natural share). So does a set with a single stratum: nothing
    to rebalance when every row, or no row, is a first season."""
    base = np.asarray(base, dtype=float)
    fy = np.asarray(first_year, dtype=bool)
    if share is None or len(base) == 0:
        return base
    total, fy_mass = base.sum(), base[fy].sum()
    if fy_mass <= 0 or fy_mass >= total:
        return base
    out = base.copy()
    out[fy] *= share / (fy_mass / total)
    out[~fy] *= (1.0 - share) / (1.0 - fy_mass / total)
    return out


def example_weights(units: pd.DataFrame, windows: pd.DataFrame, *, weighting: str = "player_balanced",
                    first_year_share: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Loss weights per forward unit (in ``units`` order) and per reconstruction window (in ``windows`` order).

    ``player_balanced`` (registered rule): players -> seasons within player -> windows within season.
    ``season_balanced`` (D-062): every unit equal, windows within unit. Either base is then rebalanced so the
    first-season units carry ``first_year_share`` of the direction's mass (None: natural share), and each
    direction is rescaled to mean one so ``lambda_plus`` keeps its meaning."""
    validate_weighting(weighting, first_year_share)
    fy_units = first_year_flags(units)
    if weighting == "player_balanced":
        seasons_per_player = units.groupby("player_id")["season"].nunique()
        n_players = len(seasons_per_player)
        base_fwd = np.array([1.0 / (n_players * seasons_per_player[p]) for p in units["player_id"]], dtype=float)
    else:
        base_fwd = np.ones(len(units), dtype=float)
    w_fwd = stratum_weights(base_fwd, fy_units, first_year_share)
    if windows is not None and len(windows):
        per_unit = windows.groupby("unit_id").size()
        if weighting == "player_balanced":
            players = windows.groupby("player_id")["season"].nunique()
            base_rec = np.array([1.0 / (len(players) * players[p] * per_unit[u]) for p, u in zip(windows["player_id"], windows["unit_id"])], dtype=float)
        else:
            base_rec = np.array([1.0 / per_unit[u] for u in windows["unit_id"]], dtype=float)
        pos = pd.Series(np.arange(len(units)), index=units["unit_id"].to_numpy())
        fy_windows = fy_units[pos.loc[windows["unit_id"].to_numpy()].to_numpy()]
        w_rec = stratum_weights(base_rec, fy_windows, first_year_share)
    else:
        w_rec = np.zeros(0, dtype=float)
    out = []
    for w in (w_fwd, w_rec):
        total = w.sum()
        out.append(w * len(w) / total if total > 0 else np.ones(len(w)))
    return out[0], out[1]


POST_FIRST_SEASON_LEVELS = ("intl", "national_team_only")
ANCHOR_COLUMNS = ("intl_rapm_orapm", "intl_rapm_drapm", "intl_rapm_poss", "intl_rapm_season_gap")


def mask_post_first_season_statics(units: pd.DataFrame) -> pd.DataFrame:
    """D-041 strict control under the v9 blocks (D-065): unit summaries that can be computed from games after the
    player's first NCAA season are neutralised, as the game rows themselves are.

    * prior league: for later-year units, the levels read from international or national-team games (``intl``,
      ``national_team_only``) become the pooled no-information level (``hs_or_none``, known 0); a later-year unit
      without a D1 season in t-1 is a returner, and its t-1 games lie after the first season;
    * international RAPM anchor: masked when its season is at or after the first NCAA season (an anchor labelled
      with the first season's year can include games after that season's end).

    First-season units are untouched: their summaries use games before the first season only. Columns absent
    (tables before v11) leave the frame unchanged."""
    if not any(c in units for c in ("prior_league", *ANCHOR_COLUMNS)):
        return units
    u = units.copy()
    later = ~first_year_flags(u)
    if "prior_league" in u:
        hit = later & u["prior_league"].astype(object).isin(POST_FIRST_SEASON_LEVELS).to_numpy()
        u.loc[hit, "prior_league"] = "hs_or_none"
        if "prior_league_known" in u:
            u.loc[hit, "prior_league_known"] = 0
    if "intl_rapm_season_gap" in u and "first_season" in u:
        anchor_season = pd.to_numeric(u["season"], errors="coerce") - pd.to_numeric(u["intl_rapm_season_gap"], errors="coerce")
        hit = (anchor_season >= pd.to_numeric(u["first_season"], errors="coerce")).to_numpy(dtype=bool, na_value=False)
        for c in ANCHOR_COLUMNS:
            if c in u:
                u.loc[hit, c] = np.nan
    return u


def post_first_season_mask(store: F.GameStore, units: pd.DataFrame) -> np.ndarray:
    """D-041 strict control: non-NCAA game rows dated after the END of the player's first NCAA season (the
    earliest roster season in the tables), i.e. the research definition of "post-freshman" (after the target
    freshman season: later college-summer national-team play and professional careers), and the same boundary
    as the reconstruction windows (``lower = period_end + 1 day``). Games played during the first season stay.
    Excluding the marked rows from every input channel, from the pool and from reconstruction makes the
    control genuinely free of post-freshman logs; freshman evaluation inputs end at the preseason cutoff and
    are unaffected."""
    earliest = units.sort_values(["player_id", "period_start"]).drop_duplicates("player_id").set_index("player_id")
    first = pd.to_datetime(earliest["period_end"])
    if "first_season" in earliest and "season" in earliest:
        # D-066: a first season before the tables' range has no unit and no dated period; its conventional end
        # (30 April of that season) is the boundary. With tables before v11 first_season is the earliest unit's
        # season, so this changes nothing there.
        fs = pd.to_numeric(earliest["first_season"], errors="coerce")
        before = (fs < pd.to_numeric(earliest["season"], errors="coerce")).fillna(False)
        if before.any():
            approx = pd.to_datetime(fs[before].astype("int64").astype(str) + "-04-30")
            first = first.where(~before, approx.reindex(first.index))
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
