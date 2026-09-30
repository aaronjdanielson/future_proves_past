"""Rolling folds over the assembled tables: cutoffs, membership, and counts.

Fold $v$ forecasts NCAA season $v$ with $c_v$ = 1 October of $v-1$ and fits
on everything released by $c^{\\rm tr}_v = c_v - 1$ day (D-001, D-016).
Membership is computed from dates, never from season labels:

* forward history of unit $(i,t)$: non-NCAA games released by $c_t$ (for a
  training unit) or by the fold's forecast cutoff (for an evaluation unit);
* reconstruction windows of unit $(i,t)$, $t<v$: non-NCAA games after the
  unit's last NCAA game, within four years, released by $c^{\\rm tr}_v$,
  nested by the first $k\\in\\{1,2,3\\}$ season labels and deduplicated when
  two $k$ select the same games — the semantics of ``windows.build_windows``,
  which remains the oracle in the tests;
* the pretraining pool: every non-NCAA game released by $c^{\\rm tr}_v$ whose
  player is neither in the fold's evaluation cohort nor in the test cohort.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from .calendar import add_years, forecast_cutoff, training_cutoff

K_VALUES = (1, 2, 3)
MAX_YEARS = 4


@dataclass(frozen=True)
class Fold:
    season: int
    kind: str  # development | selection | test

    @property
    def forecast_cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(forecast_cutoff(self.season))

    @property
    def training_cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(training_cutoff(self.season))


def fold_layout(protocol: dict, *, first_development: int = 2020) -> tuple[Fold, ...]:
    exp = protocol["experiment"]
    dev = tuple(Fold(s, "development") for s in range(first_development, exp["development_through"] + 1))
    sel = tuple(Fold(s, "selection") for s in exp["selection_seasons"])
    return dev + sel + (Fold(exp["test_season"], "test"),)


def season_periods(periods: pd.DataFrame) -> pd.DataFrame:
    """Registered per-label windows for reconstruction nesting: the span of
    all non-NCAA competition periods carrying that label."""
    p = periods[periods["source"] != "ncaa"]
    out = p.groupby("season").agg(start=("start", "min"), end=("end", "max")).reset_index()
    return out.sort_values("season").reset_index(drop=True)


UNRESOLVED_GUARD = "overlong_merge_unresolved"


def player_index(games: pd.DataFrame) -> dict:
    """Non-NCAA games per player, date-ordered.

    Games whose inferred season period the guard could not resolve (D-028)
    are dropped here, so they enter neither forward histories nor nested
    reconstruction windows; ``unresolved_period_games`` counts them.
    """
    g = games[games["source"] != "ncaa"]
    if "season_guard" in g:
        g = g[g["season_guard"] != UNRESOLVED_GUARD]
    g = g.sort_values(["player_id", "date", "record_id"])
    return {p: grp for p, grp in g.groupby("player_id", sort=False)}


def unresolved_period_games(games: pd.DataFrame) -> int:
    return int((games.get("season_guard", pd.Series(dtype=str)) == UNRESOLVED_GUARD).sum())


def forward_history(unit, player_games: pd.DataFrame | None, fold: Fold) -> pd.DataFrame | None:
    """Non-NCAA games released by the unit's own cutoff (training) or the fold's (evaluation)."""
    if player_games is None:
        return None
    cutoff = pd.Timestamp(unit.forecast_cutoff) if unit.season < fold.season else fold.forecast_cutoff
    return player_games[player_games["release_date"] <= cutoff]


def reconstruction_windows(unit, player_games: pd.DataFrame | None, fold: Fold, labels: pd.DataFrame,
                           *, k_values=K_VALUES, max_years=MAX_YEARS) -> list[dict]:
    """Nested first-k reconstruction windows of one training unit."""
    if player_games is None or unit.season >= fold.season:
        return []
    cutoff = fold.training_cutoff
    end = pd.Timestamp(unit.period_end)
    lower = end + timedelta(days=1)
    # Horizon: ``max_years`` after the season end (registered default, D-021) or, when None, every game
    # released by the training cutoff (D-038); ``k`` None selects every admissible season.
    upper = pd.Timestamp(add_years(end.date(), int(max_years))) if max_years is not None else pd.Timestamp("9999-12-31")
    g = player_games[(player_games["release_date"] <= cutoff) & (player_games["date"] >= lower)
                     & (player_games["date"] <= min(upper, cutoff))]
    if g.empty:
        return []
    evidence = set(g["season"].unique())
    periods = labels[labels["season"].isin(evidence) & (labels["end"] >= lower) & (labels["start"] <= min(upper, cutoff))]
    periods = periods.sort_values(["start", "season"])
    windows, seen = [], {}
    for k in k_values:
        chosen = periods if k is None else periods.iloc[:k]
        if chosen.empty:
            continue
        start_k = max(lower, chosen["start"].iloc[0])
        end_k = min(upper, chosen["end"].iloc[-1])
        sel = g[g["season"].isin(set(chosen["season"])) & (g["date"] >= start_k) & (g["date"] <= end_k)]
        if sel.empty:
            continue
        identity = tuple(sel["record_id"])
        if identity in seen:
            w = seen[identity]
            w["k"] = w["k"] + (k,)
            w["partial"] = bool(w["partial"] or end_k > cutoff)
            w["intended_end"] = max(w["intended_end"], end_k)
            continue
        w = {"unit_id": unit.unit_id, "player_id": unit.player_id, "season": unit.season, "k": (k,),
             "intended_start": start_k, "intended_end": end_k, "partial": bool(end_k > cutoff),
             "n_games": int(len(sel)), "record_ids": list(sel["record_id"])}
        seen[identity] = w
        windows.append(w)
    return windows


def fold_membership(tables: dict, fold: Fold, *, test_cohort_players=frozenset(), index: dict | None = None,
                    k_values=K_VALUES, max_years=MAX_YEARS, label_scope: str = "all") -> dict:
    """Training/evaluation units, reconstruction windows, and pool size for one fold."""
    units, games = tables["units"], tables["games"]
    labels = season_periods(tables["competition_periods"])
    index = player_index(games) if index is None else index
    eligible = units[units["likelihood_eligible"]]
    # Training labels must be released by the training cutoff (windows.eligible_training_outcomes);
    # season order alone is not the rule, even though NCAA seasons end before the cutoff in practice.
    before = eligible["season"] < fold.season
    released = pd.to_datetime(eligible["release_date"]) <= fold.training_cutoff
    late_released = int((before & ~released).sum())
    train = eligible[before & released]
    evaluate = eligible[eligible["season"] == fold.season]
    cohort_players = set(evaluate.loc[evaluate["intl_entrant"], "player_id"])
    heldout = cohort_players | set(test_cohort_players)
    train = train[~train["player_id"].isin(heldout)]
    if label_scope == "first_year":
        # D-060: only first NCAA (D1 roster) seasons are labels, forward and reconstruction; the evaluation set is unchanged.
        train = train[train["first_year"].astype(bool)]
    elif label_scope != "all":
        raise ValueError(f"label_scope must be 'all' or 'first_year', got {label_scope!r}")
    windows = []
    for unit in train.itertuples(index=False):
        windows.extend(reconstruction_windows(unit, index.get(unit.player_id), fold, labels, k_values=k_values, max_years=max_years))
    pool = games[(games["source"] != "ncaa") & (pd.to_datetime(games["release_date"]) <= fold.training_cutoff)
                 & ~games["player_id"].isin(heldout)]
    return {"fold": fold, "train_units": train, "evaluation_units": evaluate, "windows": pd.DataFrame(windows),
            "heldout_players": heldout, "pool_players": int(pool["player_id"].nunique()), "pool_games": int(len(pool)),
            "late_released_excluded": late_released, "unresolved_period_games": unresolved_period_games(games)}


def fold_counts(tables: dict, folds, *, test_cohort_players=frozenset()) -> pd.DataFrame:
    index = player_index(tables["games"])
    rows = []
    for fold in folds:
        m = fold_membership(tables, fold, test_cohort_players=test_cohort_players, index=index)
        ev, tr, w = m["evaluation_units"], m["train_units"], m["windows"]
        # A unit has three distinct nested windows only when its first three post-NCAA seasons all carry games.
        windows_per_unit = w.groupby("unit_id").size() if len(w) else pd.Series(dtype=int)
        rows.append({
            "fold": fold.season, "kind": fold.kind, "training_cutoff": fold.training_cutoff.date().isoformat(),
            "train_units": int(len(tr)), "train_first_year": int(tr["first_year"].sum()),
            "eval_units": int(len(ev)), "eval_first_year": int(ev["first_year"].sum()),
            "eval_intl_entrants": int(ev["intl_entrant"].sum()), "eval_any_pre_ncaa": int(ev["any_pre_ncaa"].sum()),
            "eval_intl_later_year": int((ev["intl_history"] & ~ev["first_year"]).sum()),
            "recon_windows": int(len(w)), "recon_units": int(w["unit_id"].nunique()) if len(w) else 0,
            "recon_players": int(w["player_id"].nunique()) if len(w) else 0,
            "recon_units_k2": int((windows_per_unit >= 2).sum()), "recon_units_k3": int((windows_per_unit >= 3).sum()),
            "pool_players": m["pool_players"], "pool_games": m["pool_games"], "heldout_players": int(len(m["heldout_players"])),
            "late_released_excluded": m["late_released_excluded"],
        })
    return pd.DataFrame(rows)
