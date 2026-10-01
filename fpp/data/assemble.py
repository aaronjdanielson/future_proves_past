"""Assemble the columnar training tables from the upstream stores (track 1).

Three tables come out, all with actual dates and explicit provenance:

``competition_periods``
    One row per competition-season. International leagues: each league's
    game dates are clustered at off-season gaps and every cluster is
    labelled by the end-year rule applied to its *start*; national-team and
    showcase editions and NCAA seasons are single periods.

``games``
    One row per player appearance from the four sources, with the 13
    counts in ``BOX_FIELDS`` order, starter and home flags, regulation
    length, observed overtime from team minutes, a competition indicator,
    and opponent/team strength with an availability date and provenance
    (D-023). Rows that fail an internal consistency check are kept with
    ``counts_valid = False`` rather than dropped.

``units``
    One row per confirmed roster-season (player, season, team) with the
    full-season outcome $(G, J, M, \\mathbf B, O)$, the team-minute
    completeness certificate, the exclusion reasons of ``OutcomeUnit``,
    attributes and destination context as of the forecast cutoff, the
    evidence profile (how much tracked history of each kind exists before
    the cutoff), and the cohort flags of D-023.

Season resolution for international club games (D-016). The upstream
``Season`` is a June-1 rollover label; RealGM publishes no game-to-season
link, so the label cannot be verified page by page. Where the label agrees
with the league's date-clustered competition-season the row keeps it, with
the canonical ``provider_season_id`` attached (method
``provider_identifier``); where they disagree — June/July games of summer
leagues and June playoff games — the row is relabelled to the clustered
season (method ``competition_period``) and the change is counted in the
build audit. Ratings built upstream per label bucket are joined on the
original label.
"""
from __future__ import annotations

import re
from datetime import timedelta

import numpy as np
import pandas as pd

from .calendar import forecast_cutoff
from .records import BOX_FIELDS, OutcomeUnit
from .stores import STARTER_SOURCE, Stores

SOURCES = ("ncaa", "intl", "national", "events")
STRENGTH = ["adj_o", "adj_d", "adj_pace"]
GAP_DAYS = 60                 # off-season gap that separates two competition-seasons
MAX_SEASON_DAYS = 330         # one competition-season never spans longer; a longer merged period holds two
POINTS_TOLERANCE = 2          # sum of player points vs team points; larger gaps mean extra or missing lines
MINUTE_TOLERANCE = 2.0        # team minutes may fall short of a regulation total by rounding
OVERTIME_TEAM_MINUTES = 25.0  # five players for five minutes
OVERTIME_TOLERANCE = 10.0     # patterns are 25 apart; accumulated upward rounding stays well inside this
RANK_LIMIT = 1500             # recruiting ranks above this are parser leakage upstream
USA_TEAM_CODES = ("United-States",)
FIBA_COUNTS = {"orb": "OREB", "drb": "DREB", "ast": "AST", "stl": "STL", "blk": "BLK", "tov": "TOV", "pf": "PF"}
MAPPING_VERSION = "season-map-v2/cluster-majority-v1"
# D-065 blocks on game rows (tables v11): the team's box totals, the player's usage and role within the team, and the
# latest pre-game rolling international RAPM snapshot. Sources without a value carry NaN (masked on the feature side).
TEAM_TOTAL_COLUMNS = ["team_fga", "team_fta", "team_tov", "team_ast", "team_reb", "team_poss"]
TEAM_TOTAL_RENAMES = {"FGA": "team_fga", "FTA": "team_fta", "TOV": "team_tov", "AST": "team_ast", "REB": "team_reb", "POSS": "team_poss"}
ROLE_COLUMNS = ["usage_game", "pts_share", "fga_share", "ast_share", "reb_share", "tov_share", "min_rank_on_team", "n_team_contributors"]
RAPM_GAME_COLUMNS = ["rapm_orapm", "rapm_drapm", "rapm_off_equiv", "rapm_snapshot_age_days"]
GAME_COLUMNS = [
    "record_id", "source", "player_id", "game_id", "date", "release_date", "season", "season_method",
    "original_label", "provider_season_id", "label_agrees_with_bucket", "mapping_version",
    "competition_id", "competition", "competition_kind", "country",
    "age_group", "division", "team_id", "opp_id", "team_code", "team_is_usa", "home", "starter", "appeared",
    "minutes", *BOX_FIELDS, "pts", "counts_valid", "regulation_minutes", "team_minutes", "overtime_periods",
    "overtime_certified", "coverage_certified", "points_consistent", "team_pts", "opp_pts", "game_type",
    "opp_adj_o", "opp_adj_d", "opp_adj_pace", "opp_strength_available", "opp_strength_source",
    "opp_prior_adj_o", "opp_prior_adj_d", "opp_prior_adj_pace", "opp_prior_available",
    "opp_hist_winpct", "opp_hist_placement", "opp_hist_available",
    "team_adj_o", "team_adj_d", "team_adj_pace", "team_strength_available",
    "team_hist_winpct", "team_hist_placement", "team_hist_available",
    "opp_record_winpct", "opp_record_margin",
    # Rating uncertainty (D-039, tables_v8): same availability dates as the ratings they belong to.
    "opp_adj_o_sd", "opp_adj_d_sd", "opp_prior_adj_o_sd", "opp_prior_adj_d_sd", "team_adj_o_sd", "team_adj_d_sd",
    *TEAM_TOTAL_COLUMNS, *ROLE_COLUMNS, *RAPM_GAME_COLUMNS,                       # D-065 (tables v11)
]
STRENGTH_SD = ["adj_o_sd", "adj_d_sd"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def end_year_label(dates) -> pd.Series:
    """D-016 fallback: the end-year label rolls on 1 August."""
    d = pd.to_datetime(pd.Series(dates))
    return (d.dt.year + ((d.dt.month > 8) | ((d.dt.month == 8) & (d.dt.day >= 1)))).astype("int64")


def regulation_from_minutes(team_minutes: pd.Series) -> int:
    """Modal team minutes snapped to a 40- or 48-minute game; 40 when unclear."""
    mode = team_minutes.dropna().round().mode()
    if mode.empty:
        return 40
    value = float(mode.iloc[0])
    if 230 <= value <= 252:
        return 48
    return 40


def overtime_from_team_minutes(team_minutes: pd.Series, regulation: pd.Series):
    """Observed overtime periods and two per-game certificates from team minutes.

    ``coverage`` holds when team minutes reach $5r - 2$: no player line is
    missing (an excess above a regulation total is accumulated rounding, not
    missing data). ``periods`` is the nearest nonnegative integer $O$ with
    team minutes $\\approx 5r + 25O$ and ``overtime`` certifies it when the
    residual lies in $[-2, +10]$ — patterns are 25 minutes apart, so this is
    unambiguous; otherwise $O$ is unknown.
    """
    base = 5.0 * regulation.astype(float)
    excess = team_minutes.astype(float) - base
    coverage = (excess >= -MINUTE_TOLERANCE).fillna(False).astype(bool)
    periods = np.floor(excess / OVERTIME_TEAM_MINUTES + 0.5).where(coverage)
    residual = excess - periods * OVERTIME_TEAM_MINUTES
    overtime = ((residual >= -MINUTE_TOLERANCE) & (residual <= OVERTIME_TOLERANCE) & periods.notna()).fillna(False).astype(bool)
    periods = periods.where(overtime).clip(lower=0)
    return periods.astype("float64"), overtime, coverage


def cluster_periods(dates, *, gap_days: int = GAP_DAYS) -> pd.DataFrame:
    """Split one competition's distinct game dates at off-season gaps into inferred periods.

    Each cluster is labelled by the *majority* end-year label of its dates
    (a stray July game before a September start cannot flip the season, and
    a summer league that runs past 1 August keeps its year); clusters that
    share a label (a season interrupted for months, a cup with sparse
    rounds) merge. ``majority_share`` records how one-sided the vote was so
    the build audit can flag ambiguous periods. Two seasons of one league
    closer than ``gap_days`` would merge — an audited limitation (D-027).
    """
    days = pd.Series(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()).sort_values().reset_index(drop=True)
    if days.empty:
        return pd.DataFrame(columns=["season", "start", "end", "majority_share", "guard"])
    cluster = (days.diff().dt.days.fillna(0) > gap_days).cumsum()
    frame = pd.DataFrame({"date": days, "cluster": cluster, "label": end_year_label(days)})
    votes = frame.groupby(["cluster", "label"]).size().rename("n").reset_index()
    # Ties go to the earlier label: a cluster that straddles 1 August is the season that began first.
    votes = votes.sort_values(["cluster", "n", "label"], ascending=[True, False, True]).drop_duplicates("cluster")
    totals = frame.groupby("cluster").size()
    votes["majority_share"] = votes["n"].values / totals.loc[votes["cluster"]].values
    spans = frame.groupby("cluster")["date"].agg(["min", "max"]).reset_index()
    votes = votes.merge(spans, on="cluster").sort_values("min").reset_index(drop=True)
    votes["guard"] = ""
    # Overlong guard: a label whose clusters together span more than a season (> MAX_SEASON_DAYS) holds two
    # seasons — typically a COVID-shifted July–September season voted into the following year. The earliest
    # cluster takes the preceding label when that label is free; otherwise the merge stays and is flagged
    # (a split-season competition cannot be represented under an end-year convention).
    for _ in range(3):
        changed = False
        for label, group in votes.groupby("label"):
            if len(group) < 2 or (group["max"].max() - group["min"].min()).days <= MAX_SEASON_DAYS:
                continue
            first = group.index[0]
            if (label - 1) not in set(votes["label"]):
                votes.loc[first, "label"] = label - 1
                votes.loc[first, "guard"] = "relabelled_preceding_year"
                changed = True
            else:
                votes.loc[group.index, "guard"] = "overlong_merge_unresolved"
        if not changed:
            break
    frame = frame.drop(columns=["label"]).merge(votes[["cluster", "label", "majority_share", "guard"]].rename(columns={"label": "season"}), on="cluster")
    out = frame.groupby("season").agg(start=("date", "min"), end=("date", "max"), majority_share=("majority_share", "min"),
                                      guard=("guard", "max")).reset_index()
    return out


def assign_by_periods(dates: pd.Series, keys: pd.Series, periods: pd.DataFrame) -> pd.Series:
    """Season of the period (per key) containing each date; -1 where none does."""
    out = pd.Series(-1, index=dates.index, dtype="int64")
    dates = pd.to_datetime(dates)
    grouped = {k: g for k, g in periods.groupby("competition_id")}
    for key, idx in keys.groupby(keys).groups.items():
        mine = grouped.get(key)
        if mine is None:
            continue
        d = dates.loc[idx].values
        season = np.full(len(idx), -1, dtype="int64")
        for start, end, label in zip(mine["start"].values, mine["end"].values, mine["season"].values):
            season[(d >= start) & (d <= end)] = int(label)
        out.loc[idx] = season
    return out


def fiba_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """RealGM international columns → the 13 BOX_FIELDS counts."""
    out = pd.DataFrame(index=frame.index)
    out["a2"] = frame["FGA"] - frame["FG3A"]
    out["k2"] = frame["FGM"] - frame["FG3M"]
    out["a3"], out["k3"], out["af"], out["kf"] = frame["FG3A"], frame["FG3M"], frame["FTA"], frame["FTM"]
    for name, col in FIBA_COUNTS.items():
        out[name] = frame[col]
    return out


def counts_valid(counts: pd.DataFrame, minutes: pd.Series) -> pd.Series:
    """Nonnegative integer counts with makes ≤ attempts and finite nonnegative minutes."""
    c = counts.astype("float64")
    ok = c.notna().all(axis=1) & (c >= 0).all(axis=1) & (c == c.round()).all(axis=1)
    ok &= (c["k2"] <= c["a2"]) & (c["k3"] <= c["a3"]) & (c["kf"] <= c["af"])
    return ok & minutes.notna() & np.isfinite(minutes.astype("float64")) & (minutes >= 0)


def age_group_of_tournament(slug: str) -> str:
    m = re.search(r"U-?(1[4-9]|20)", slug)
    return f"U{m.group(1)}" if m else "senior"


def division_of_tournament(slug: str) -> str:
    m = re.search(r"-([ABC])$", slug)
    return m.group(1) if m else ""


def age_group_of_league(slug: str, kind: str) -> str:
    if kind != "youth":
        return "senior"
    for key, value in (("ANGT", "U18"), ("Next-Generation", "U18"), ("Espoirs", "U21"), ("Junior", "U19"),
                       ("Youth", "U21"), ("Liga-U", "U22")):
        if key in slug:
            return value
    return "youth"


def _placement_ordinal(value) -> float:
    m = re.match(r"\s*(\d+)", str(value)) if value is not None else None
    return float(m.group(1)) if m else np.nan


def _record_winpct(value) -> float:
    m = re.match(r"\s*(\d+)-(\d+)", str(value)) if value is not None else None
    if not m:
        return np.nan
    w, l = int(m.group(1)), int(m.group(2))
    return w / (w + l) if w + l else np.nan


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def pair_opponents(team_games: pd.DataFrame) -> pd.DataFrame:
    """Attach the other row's team id and points; one row per (game, team).

    When a game carries more than two team ids (an untracked opponent next
    to a resolved one), the partner with a scored line is kept.
    """
    other = team_games[["GameID", "TeamID", "PTS"]].rename(columns={"TeamID": "opp_id", "PTS": "opp_pts"})
    paired = team_games.merge(other, on="GameID", how="left")
    paired = paired[paired["opp_id"] != paired["TeamID"]]
    paired = paired.sort_values("opp_pts", na_position="last").drop_duplicates(["GameID", "TeamID"], keep="first")
    singles = team_games[~team_games["GameID"].isin(paired["GameID"])].copy()
    singles["opp_id"], singles["opp_pts"] = np.nan, np.nan
    return pd.concat([paired, singles], ignore_index=True)


def _finish(frame: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["release_date"] = frame["date"] + pd.Timedelta(days=lag_days)
    frame["record_id"] = (frame["source"] + ":" + frame["game_id"].astype("int64").astype(str)
                          + ":" + frame["player_id"].astype("int64").astype(str))
    counts = frame[list(BOX_FIELDS)]
    frame["counts_valid"] = counts_valid(counts, frame["minutes"])
    # Appearance evidence as in records.classify_participation: positive minutes, nonzero counts, or a
    # provider starter status. A listed line with none of these is not an appearance; it stays a row
    # (unresolved) and is reconciled against the provider season line at the unit level.
    positive = (frame["minutes"].fillna(0) > 0) | (counts.fillna(0) > 0).any(axis=1)
    frame["appeared"] = positive | frame["starter"].fillna(False).astype(bool)
    frame["mapping_version"] = MAPPING_VERSION
    for col in GAME_COLUMNS:
        if col not in frame:
            frame[col] = np.nan
    return frame[GAME_COLUMNS]


# --------------------------------------------------------------------------- #
# International club and youth competitions
# --------------------------------------------------------------------------- #

def build_intl_games(stores: Stores, *, lag_days: int, gap_days: int):
    leagues = stores.intl_leagues()
    team = stores.intl_team_games()
    team["Date"] = pd.to_datetime(team["Date"])
    # Competition periods from clustered dates, labelled at cluster start.
    periods = []
    for league_id, group in team.groupby("LeagueID"):
        p = cluster_periods(group["Date"], gap_days=gap_days)
        p["competition_id"] = league_id
        periods.append(p)
    periods = pd.concat(periods, ignore_index=True)
    periods["source"] = "intl"
    # One resolver (D-016 order): no per-game provider identifier exists, so every row is assigned by the
    # inferred competition period; rows outside every period take the audited fallback. Agreement with
    # the upstream June-1 bucket and the canonical season id are attached as audit columns, not methods.
    clustered = assign_by_periods(team["Date"], team["LeagueID"], periods)
    team["season"] = np.where(clustered >= 0, clustered, end_year_label(team["Date"])).astype("int64")
    team["season_method"] = np.where(clustered < 0, "fallback_rollover", "competition_period")
    team["label_agrees_with_bucket"] = team["season"] == team["Season"]
    canonical = stores.intl_league_seasons().rename(columns={"season_end_year": "season"})
    team = team.merge(canonical[["LeagueID", "season", "provider_season_id"]], on=["LeagueID", "season"], how="left")
    # Regulation and overtime from team minutes.
    regulation = team.groupby("LeagueID")["Minutes"].apply(regulation_from_minutes).rename("regulation_minutes")
    team = team.merge(regulation, left_on="LeagueID", right_index=True, how="left")
    team["overtime_periods"], team["overtime_certified"], team["coverage_certified"] = overtime_from_team_minutes(team["Minutes"], team["regulation_minutes"])
    team = pair_opponents(team)
    # Strength. Ratings and team history were built per label bucket: join on the original label,
    # available one lag after the bucket's last game.
    bucket_end = stores.intl_bucket_periods().rename(columns={"end_date": "bucket_end"})
    bucket_end["bucket_end"] = pd.to_datetime(bucket_end["bucket_end"])
    bucket_end = bucket_end[["LeagueID", "Season", "bucket_end"]]
    team = _attach_intl_strength(team, stores.intl_ratings(), bucket_end, lag_days)
    team = _attach_team_history(team, stores.intl_team_history(), leagues, bucket_end, lag_days)
    team["opp_strength_source"] = np.select(
        [team["opp_adj_o"].notna(), team["opp_prior_adj_o"].notna(), team["opp_hist_winpct"].notna()],
        ["intl_season_rating", "intl_prior_season_rating", "intl_team_history"], "none")
    # Player lines.
    player = stores.intl_player_games()
    status = stores.intl_status().rename(columns={"player_id": "PlayerID", "game_id": "GameID"})
    player = player.merge(status, on=["PlayerID", "GameID"], how="left")
    # Starter flag: intl.db's own column first (parsed from the box-score page, any player), else the status re-parsed
    # upstream for NCAA-linked players (D-053).
    own = player["Starter"].map({1: True, 0: False, 1.0: True, 0.0: False}).astype("boolean") if "Starter" in player else pd.Series(pd.NA, index=player.index, dtype="boolean")
    player["starter"] = own.where(own.notna(), player["status"].map({"Starter": True, "Bench": False}).astype("boolean"))
    team = team.rename(columns={"PTS": "team_pts", "Minutes": "team_minutes", **TEAM_TOTAL_RENAMES}).drop(columns=["Date", "Home"])
    frame = player.merge(team.drop(columns=["Season"]), on=["GameID", "TeamID", "LeagueID"], how="left")
    frame = frame.merge(leagues[["LeagueID", "Slug", "Kind", "Country"]], on="LeagueID", how="left")
    slug, kind = frame["Slug"].fillna(""), frame["Kind"].fillna("")
    out = pd.DataFrame({
        "source": "intl", "player_id": frame["PlayerID"], "game_id": frame["GameID"], "date": frame["Date"],
        "season": frame["season"], "season_method": frame["season_method"], "original_label": frame["Season"],
        "provider_season_id": frame["provider_season_id"], "label_agrees_with_bucket": frame["label_agrees_with_bucket"],
        "competition_id": frame["LeagueID"], "competition": slug, "competition_kind": kind, "country": frame["Country"],
        "age_group": [age_group_of_league(s, k) for s, k in zip(slug, kind)], "division": "",
        "team_id": frame["TeamID"], "opp_id": frame["opp_id"], "team_code": np.nan, "team_is_usa": False,
        "home": frame["Home"], "starter": frame["starter"], "minutes": frame["Min"], "pts": frame["PTS"],
        "regulation_minutes": frame["regulation_minutes"], "team_minutes": frame["team_minutes"],
        "overtime_periods": frame["overtime_periods"], "overtime_certified": frame["overtime_certified"],
        "coverage_certified": frame["coverage_certified"], "team_pts": frame["team_pts"], "opp_pts": frame["opp_pts"],
    })
    for col in [c for c in GAME_COLUMNS if c.startswith(("opp_adj", "opp_prior", "opp_strength", "opp_hist", "team_adj", "team_strength", "team_hist"))]:
        out[col] = frame[col]
    out = pd.concat([out, fiba_counts(frame), _team_totals(frame)], axis=1)
    out = attach_rolling_rapm(attach_role_fields(out), stores.intl_rapm_rolling())      # D-065 usage and RAPM blocks
    games = _finish(out, lag_days)
    # Where each starter flag came from (D-063), for the build audit: page value, listing order, or none.
    games.attrs["starter_source"] = (frame["starter_source"].map(STARTER_SOURCE).value_counts().to_dict()
                                     if "starter_source" in frame else {})
    return games, periods


def _attach_intl_strength(team, ratings, bucket_end, lag_days):
    if ratings is None:
        for col in ("opp_adj_o", "opp_adj_d", "opp_adj_pace", "opp_strength_available", "team_adj_o", "team_adj_d",
                    "team_adj_pace", "team_strength_available", "opp_prior_adj_o", "opp_prior_adj_d", "opp_prior_adj_pace", "opp_prior_available",
                    "opp_adj_o_sd", "opp_adj_d_sd", "opp_prior_adj_o_sd", "opp_prior_adj_d_sd", "team_adj_o_sd", "team_adj_d_sd"):
            team[col] = np.nan
        return team
    ratings = ratings.rename(columns={"league_id": "LeagueID", "season": "Season"}).copy()
    for col in STRENGTH_SD:                       # older rating stores carry no uncertainty columns
        if col not in ratings:
            ratings[col] = np.nan
    ratings = ratings.merge(bucket_end, on=["LeagueID", "Season"], how="left")
    ratings["available"] = ratings["bucket_end"] + pd.Timedelta(days=lag_days)
    fields = [*STRENGTH, *STRENGTH_SD]
    base = ratings[["team_id", "LeagueID", "Season", "available", *fields]]
    opp = base.rename(columns={"team_id": "opp_id", "available": "opp_strength_available", **{s: f"opp_{s}" for s in fields}})
    team = team.merge(opp, on=["opp_id", "LeagueID", "Season"], how="left")
    own = base.rename(columns={"team_id": "TeamID", "available": "team_strength_available", **{s: f"team_{s}" for s in fields}})
    team = team.merge(own, on=["TeamID", "LeagueID", "Season"], how="left")
    prior = base.copy()
    prior["Season"] = prior["Season"] + 1
    prior = prior.rename(columns={"team_id": "opp_id", "available": "opp_prior_available", **{s: f"opp_prior_{s}" for s in fields}})
    return team.merge(prior, on=["opp_id", "LeagueID", "Season"], how="left")


def _attach_team_history(team, history, leagues, bucket_end, lag_days):
    """Record and placement per team-season; the league's own row is preferred, else the fullest."""
    h = history.copy()
    h["Season"] = h["Season"].astype(str).str[-4:].astype("int64", errors="ignore")
    h = h[pd.to_numeric(h["Season"], errors="coerce").notna()]
    h["Season"] = h["Season"].astype("int64")
    h["winpct"] = h["Record"].map(_record_winpct)
    h["placement"] = h["Placement"].map(_placement_ordinal)
    h["games"] = h["Record"].map(lambda v: sum(int(x) for x in re.findall(r"\d+", str(v))[:2]) if re.match(r"\s*\d+-\d+", str(v)) else 0)
    names = leagues.assign(key=leagues["Name"].map(_normalize_name))[["LeagueID", "key"]]
    h["key"] = h["LeagueName"].map(_normalize_name)
    h = h.merge(names, on="key", how="left")
    h = h.sort_values(["TeamID", "Season", "games"], ascending=[True, True, False])
    exact = h.dropna(subset=["LeagueID"]).drop_duplicates(["TeamID", "Season", "LeagueID"])
    fullest = h.drop_duplicates(["TeamID", "Season"])[["TeamID", "Season", "winpct", "placement"]]
    exact = exact[["TeamID", "Season", "LeagueID", "winpct", "placement"]].rename(columns={"winpct": "x_winpct", "placement": "x_placement"})
    exact["LeagueID"] = exact["LeagueID"].astype("int64")

    def attach(frame, id_col, prefix):
        frame = frame.merge(exact.rename(columns={"TeamID": id_col}), on=[id_col, "LeagueID", "Season"], how="left")
        frame = frame.merge(fullest.rename(columns={"TeamID": id_col, "winpct": "f_winpct", "placement": "f_placement"}), on=[id_col, "Season"], how="left")
        frame[f"{prefix}_hist_winpct"] = frame["x_winpct"].where(frame["x_winpct"].notna(), frame["f_winpct"])
        frame[f"{prefix}_hist_placement"] = frame["x_placement"].where(frame["x_placement"].notna(), frame["f_placement"])
        return frame.drop(columns=["x_winpct", "x_placement", "f_winpct", "f_placement"])

    team = attach(team, "opp_id", "opp")
    team = attach(team, "TeamID", "team")
    team = team.merge(bucket_end, on=["LeagueID", "Season"], how="left")
    team["opp_hist_available"] = team["bucket_end"] + pd.Timedelta(days=lag_days)
    team["team_hist_available"] = team["opp_hist_available"]   # both teams' records are season-end facts
    return team.drop(columns=["bucket_end"])


# --------------------------------------------------------------------------- #
# National-team tournaments and HS showcase events
# --------------------------------------------------------------------------- #

def build_tournament_games(stores: Stores, store: str, *, lag_days: int):
    """national.db and events.db share one layout; each edition is one period."""
    editions = stores.tournament_editions(store)
    team = stores.tournament_team_games(store)
    team["Date"] = pd.to_datetime(team["Date"])
    key = ["TournamentID", "YearID"]
    periods = team.groupby(key)["Date"].agg(start="min", end="max").reset_index()
    periods["season"] = end_year_label(periods["start"])
    periods = periods.merge(editions, on=key, how="left")
    periods["competition_id"] = periods["TournamentID"] * 10000 + periods["YearID"]
    periods["source"] = store
    if store == "events":
        regulation = team.groupby("TournamentID")["Minutes"].apply(regulation_from_minutes).rename("regulation_minutes")
        team = team.merge(regulation, left_on="TournamentID", right_index=True, how="left")
    else:
        team["regulation_minutes"] = 40
    team["overtime_periods"], team["overtime_certified"], team["coverage_certified"] = overtime_from_team_minutes(team["Minutes"], team["regulation_minutes"])
    team = pair_opponents(team)
    team = team.merge(periods[key + ["season", "end", "TournamentSlug"]], on=key, how="left")
    # Edition record: the strength proxy for national teams, known at the edition's end.
    team["margin"] = team["PTS"] - team["opp_pts"]
    team["win"] = (team["margin"] > 0).astype(float)
    record = team.groupby(key + ["TeamID"]).agg(opp_record_winpct=("win", "mean"), opp_record_margin=("margin", "mean")).reset_index()
    team = team.merge(record.rename(columns={"TeamID": "opp_id"}), on=key + ["opp_id"], how="left")
    team["opp_strength_available"] = team["end"] + pd.Timedelta(days=lag_days)
    team["opp_strength_source"] = np.where(team["opp_record_winpct"].notna(), f"{store}_edition_record", "none")
    team = team.rename(columns={"PTS": "team_pts", "Minutes": "team_minutes", **TEAM_TOTAL_RENAMES}).drop(columns=["Date", "Home", "TeamCode", "end", "win", "margin"])
    player = stores.tournament_player_games(store)
    frame = player.merge(team, on=["GameID", "TeamID"] + key, how="left")
    slug = frame["TournamentSlug"].fillna("")
    is_usa = frame["TeamCode"].isin(USA_TEAM_CODES) if store == "national" else frame["TeamCode"].str.contains("USA", na=False)
    out = pd.DataFrame({
        "source": store, "player_id": frame["PlayerID"], "game_id": frame["GameID"], "date": frame["Date"],
        "season": frame["season"], "season_method": "competition_period", "original_label": np.nan, "provider_season_id": np.nan,
        "competition_id": frame["TournamentID"] * 10000 + frame["YearID"], "competition": slug,
        "competition_kind": "national" if store == "national" else "event", "country": np.nan,
        "age_group": [("hs" if store == "events" else age_group_of_tournament(s)) for s in slug],
        "division": [division_of_tournament(s) for s in slug],
        "team_id": frame["TeamID"], "opp_id": frame["opp_id"], "team_code": frame["TeamCode"], "team_is_usa": is_usa,
        "home": frame["Home"], "starter": frame["Starter"].map({0: False, 1: True}).astype("boolean"),
        "minutes": frame["Min"], "pts": frame["PTS"],
        "regulation_minutes": frame["regulation_minutes"], "team_minutes": frame["team_minutes"],
        "overtime_periods": frame["overtime_periods"], "overtime_certified": frame["overtime_certified"],
        "coverage_certified": frame["coverage_certified"], "team_pts": frame["team_pts"], "opp_pts": frame["opp_pts"],
        "opp_record_winpct": frame["opp_record_winpct"], "opp_record_margin": frame["opp_record_margin"],
        "opp_strength_available": frame["opp_strength_available"], "opp_strength_source": frame["opp_strength_source"],
    })
    out = pd.concat([out, fiba_counts(frame), _team_totals(frame)], axis=1)
    out = attach_role_fields(out)                                                    # D-065 usage block
    periods = periods.rename(columns={"TournamentSlug": "competition"})[["source", "competition_id", "competition", "season", "start", "end"]]
    return _finish(out, lag_days), periods


# --------------------------------------------------------------------------- #
# NCAA game logs and team-games
# --------------------------------------------------------------------------- #

def build_ncaa_games(stores: Stores, *, lag_days: int):
    logs = stores.ncaa_gamelogs()
    logs["date"] = pd.to_datetime(logs["date"])
    team = stores.ncaa_team_games()
    # Team box totals keep their own names so they never collide with the player lines' columns (D-065 usage block).
    team = team.rename(columns={"fga": "team_fga", "fta": "team_fta", "tov": "team_tov", "ast": "team_ast", "reb": "team_reb", "poss": "team_poss"})
    # Team minutes summed from the player lines certify each team-game (C7/C10).
    sums = logs.groupby(["game_id", "team_id"]).agg(team_minutes=("minutes", "sum"), lines_pts=("pts", "sum")).reset_index()
    team = team.merge(sums, on=["game_id", "team_id"], how="outer")
    team["regulation_minutes"] = 40
    team["overtime_periods"], team["overtime_certified"], team["coverage_certified"] = overtime_from_team_minutes(team["team_minutes"], team["regulation_minutes"])
    # Provider team minutes (D-053): an exact 5r + 25O total certifies overtime directly and is immune to extra or
    # missing player lines; the player-minute sum keeps certifying coverage. Where both exist they are compared.
    if "minutes" not in team:
        team["minutes"] = np.nan
    prov = pd.to_numeric(team["minutes"], errors="coerce")
    excess = prov - 5 * team["regulation_minutes"]
    prov_periods = (excess / 25).round()
    # The same rounding tolerance as the player-sum certificate (MINUTE_TOLERANCE): 201 or 224 team minutes are
    # regulation and one overtime with a one-minute provider rounding, 213 is neither and falls back to the sum path.
    prov_ok = prov.notna() & (prov > 0) & (prov_periods >= 0) & ((excess - 25 * prov_periods).abs() <= MINUTE_TOLERANCE)
    prov_periods = prov_periods.where(prov_ok)
    team["overtime_source"] = np.where(prov_ok, "provider_team_minutes", np.where(team["overtime_certified"], "player_minute_sum", "none"))
    team["overtime_agrees"] = np.where(prov_ok & team["overtime_certified"], prov_periods == team["overtime_periods"], np.nan)
    team["overtime_periods"] = prov_periods.where(prov_ok, team["overtime_periods"])
    team["overtime_certified"] = prov_ok | team["overtime_certified"].astype(bool)
    # Points-consistency certificate (D-029): the player lines must reproduce the team's points. Extra lines
    # (an opponent's players filed under this team, which read as 400 minutes and eight overtimes) and missing
    # lines both fail it; where team points are unknown the minute certificate stands alone.
    diff = team["lines_pts"] - team["pts"]
    team["points_consistent"] = diff.abs() <= POINTS_TOLERANCE
    known = team["pts"].notna() & team["lines_pts"].notna()
    team["coverage_certified"] = team["coverage_certified"] & (~known | team["points_consistent"])
    # Provider-certified overtime does not depend on the player lines, so the points gate applies only to the sum path.
    sum_path = team["overtime_source"] != "provider_team_minutes"
    team["overtime_certified"] = team["overtime_certified"] & (~sum_path | ~known | team["points_consistent"])
    team["overtime_periods"] = team["overtime_periods"].where(team["overtime_certified"])
    team = team.rename(columns={"game_id": "GameID", "team_id": "TeamID", "pts": "PTS"})
    team = pair_opponents(team).rename(columns={"GameID": "game_id", "TeamID": "team_id", "PTS": "team_pts", "opp_id": "opp_from_boxscore"})
    ratings = stores.ncaa_team_ratings()
    season_end = logs.groupby("season")["date"].max().rename("season_end")
    ratings = ratings.merge(season_end, left_on="season", right_index=True, how="left")
    ratings["available"] = ratings["season_end"] + pd.Timedelta(days=lag_days)
    ratings = ratings[["team_id", "season", "available", *STRENGTH]]
    frame = logs.drop(columns=["game_type"]).merge(team.drop(columns=["season", "lines_pts", "minutes", "overtime_agrees"]), on=["game_id", "team_id"], how="left")
    frame["opp_id"] = frame["opp_id"].where(frame["opp_id"].notna(), frame["opp_from_boxscore"])
    opp = ratings.rename(columns={"team_id": "opp_id", "available": "opp_strength_available", **{s: f"opp_{s}" for s in STRENGTH}})
    frame = frame.merge(opp, on=["opp_id", "season"], how="left")
    own = ratings.rename(columns={"available": "team_strength_available", **{s: f"team_{s}" for s in STRENGTH}})
    frame = frame.merge(own, on=["team_id", "season"], how="left")
    prior = ratings.copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns={"team_id": "opp_id", "available": "opp_prior_available", **{s: f"opp_prior_{s}" for s in STRENGTH}})
    frame = frame.merge(prior, on=["opp_id", "season"], how="left")
    frame["opp_strength_source"] = np.select([frame["opp_adj_o"].notna(), frame["opp_prior_adj_o"].notna()],
                                             ["ncaa_season_rating", "ncaa_prior_season_rating"], "none")
    out = pd.DataFrame({
        "source": "ncaa", "player_id": frame["player_id"], "game_id": frame["game_id"], "date": frame["date"],
        "season": frame["season"], "season_method": "provider_identifier", "original_label": frame["season"],
        "provider_season_id": np.nan, "competition_id": 0, "competition": "NCAA-D1", "competition_kind": "ncaa",
        "country": "United States", "age_group": "college", "division": "", "team_id": frame["team_id"],
        "opp_id": frame["opp_id"], "team_code": np.nan, "team_is_usa": False, "home": frame["home"],
        "starter": frame["status"].map({"Starter": True, "Bench": False}).astype("boolean"),
        "minutes": frame["minutes"], "pts": frame["pts"], "regulation_minutes": 40, "team_minutes": frame["team_minutes"],
        "overtime_periods": frame["overtime_periods"], "overtime_certified": frame["overtime_certified"],
        "coverage_certified": frame["coverage_certified"], "points_consistent": frame["points_consistent"],
        "team_pts": frame["team_pts"], "opp_pts": frame["opp_pts"], "game_type": frame["game_type"],
        "a2": frame["fg2a"], "k2": frame["fg2m"], "a3": frame["fg3a"], "k3": frame["fg3m"], "af": frame["fta"], "kf": frame["ftm"],
        **{name: frame[name] for name in ("orb", "drb", "ast", "stl", "blk", "tov", "pf")},
    })
    for col in [c for c in frame.columns if c.startswith(("opp_adj", "opp_prior", "opp_strength", "team_adj", "team_strength"))]:
        out[col] = frame[col]
    out = attach_role_fields(pd.concat([out, _team_totals(frame)], axis=1))          # D-065 usage block
    periods = logs.groupby("season")["date"].agg(start="min", end="max").reset_index()
    periods["source"], periods["competition_id"], periods["competition"] = "ncaa", 0, "NCAA-D1"
    return _finish(out, lag_days), periods, team


# --------------------------------------------------------------------------- #
# D-065 blocks: usage and role within the team, rolling international RAPM (game rows);
# prior league, destination context, international RAPM anchor, On3 fill (units)
# --------------------------------------------------------------------------- #

def _team_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """The player's team's box totals in that game; NaN where the team line lacks them."""
    return pd.DataFrame({c: (pd.to_numeric(frame[c], errors="coerce") if c in frame else np.nan) for c in TEAM_TOTAL_COLUMNS},
                        index=frame.index)


def attach_role_fields(games: pd.DataFrame) -> pd.DataFrame:
    """Per-game usage rate and role within the team (v8 plan item 14) from the team totals and the teammates' lines.

    usage = 100 (FGA + 0.44 FTA + TOV)(team minutes / 5) / (minutes (team FGA + 0.44 team FTA + team TOV));
    shares are the player's fraction of the team's points, field-goal attempts, assists, rebounds and turnovers;
    ``min_rank_on_team`` ranks the player's minutes among the team's lines of that game (1 = most) and
    ``n_team_contributors`` counts lines with minutes. Impossible values (a share above one) are clipped."""
    g = games
    num = lambda c: (pd.to_numeric(g[c], errors="coerce").astype("float64") if c in g else pd.Series(np.nan, index=g.index))  # noqa: E731
    fga, fta, tov, ast, reb = num("a2") + num("a3"), num("af"), num("tov"), num("ast"), num("orb") + num("drb")
    minutes, team_minutes = num("minutes"), num("team_minutes")
    poss_used = fga + 0.44 * fta + tov
    team_used = num("team_fga") + 0.44 * num("team_fta") + num("team_tov")
    ok = (minutes > 0) & (team_used > 0) & (team_minutes > 0)
    usage = (100.0 * poss_used * (team_minutes / 5.0) / (minutes * team_used)).clip(lower=0.0, upper=100.0)
    g["usage_game"] = usage.where(ok)

    def share(numerator, denominator):
        return (numerator / denominator.where(denominator > 0)).clip(lower=0.0, upper=1.0)

    g["pts_share"] = share(num("pts"), num("team_pts"))
    g["fga_share"] = share(fga, num("team_fga"))
    g["ast_share"] = share(ast, num("team_ast"))
    g["reb_share"] = share(reb, num("team_reb"))
    g["tov_share"] = share(tov, num("team_tov"))
    keys = [g[k] for k in ("game_id", "team_id")]
    played = minutes.where(minutes > 0)
    g["min_rank_on_team"] = played.groupby(keys).rank(ascending=False, method="min")
    g["n_team_contributors"] = (minutes > 0).astype("float64").groupby(keys).transform("sum")
    return g


def attach_rolling_rapm(games: pd.DataFrame, rolling: pd.DataFrame | None) -> pd.DataFrame:
    """The latest rolling international RAPM snapshot dated strictly before each game, per (player, upstream
    season label, league) (v8 plan item 3); games without one carry NaN."""
    for c in RAPM_GAME_COLUMNS:
        games[c] = np.nan
    if rolling is None or len(rolling) == 0 or len(games) == 0:
        return games
    r = rolling.rename(columns={"season": "original_label", "league_id": "competition_id"}).copy()
    r["cutoff_date"] = pd.to_datetime(r["cutoff_date"], errors="coerce")
    r = r.dropna(subset=["cutoff_date", "player_id", "original_label", "competition_id"])
    r = r.astype({"player_id": "int64", "original_label": "int64", "competition_id": "int64"})
    left = pd.DataFrame({"row": np.arange(len(games)), "player_id": pd.to_numeric(games["player_id"], errors="coerce"),
                         "original_label": pd.to_numeric(games["original_label"], errors="coerce"),
                         "competition_id": pd.to_numeric(games["competition_id"], errors="coerce"),
                         "date": pd.to_datetime(games["date"], errors="coerce")})
    left = left.dropna().astype({"player_id": "int64", "original_label": "int64", "competition_id": "int64"})
    if left.empty:
        return games
    merged = pd.merge_asof(left.sort_values("date"), r.sort_values("cutoff_date")[["player_id", "original_label", "competition_id", "cutoff_date", "orapm", "drapm", "off_equiv"]],
                           left_on="date", right_on="cutoff_date", by=["player_id", "original_label", "competition_id"],
                           allow_exact_matches=False, direction="backward")
    hit = merged["cutoff_date"].notna()
    rows = merged.loc[hit, "row"].to_numpy()
    games.iloc[rows, games.columns.get_loc("rapm_orapm")] = merged.loc[hit, "orapm"].to_numpy()
    games.iloc[rows, games.columns.get_loc("rapm_drapm")] = merged.loc[hit, "drapm"].to_numpy()
    games.iloc[rows, games.columns.get_loc("rapm_off_equiv")] = merged.loc[hit, "off_equiv"].to_numpy()
    games.iloc[rows, games.columns.get_loc("rapm_snapshot_age_days")] = (merged.loc[hit, "date"] - merged.loc[hit, "cutoff_date"]).dt.days.to_numpy().astype(float)
    return games


# ``national_team_only`` added upstream on 2026-09-30: no club, youth, school or D1 origin, national-team games before t.
PRIOR_LEAGUE_LEVELS = ("ncaa_d1", "hs_or_none", "juco_naia_other", "ncaa_d2", "ncaa_d3", "intl", "national_team_only")
# Units without an upstream row (e.g. the deployment season's freshmen) take the roster role's reading; ``unknown``
# upstream (a scrape gap) is pooled with ``hs_or_none`` and told apart by the known flag.
PRIOR_LEAGUE_BY_ROLE = {"freshman": ("hs_or_none", 0), "transfer_d1": ("ncaa_d1", 1), "returning": ("ncaa_d1", 1),
                        "transfer_nond1": ("hs_or_none", 0)}


def prior_league_table(stores: Stores) -> pd.DataFrame | None:
    t = stores.player_prior_league()
    if t is None:
        return None
    t = t.copy()
    t["prior_league"] = t["prior_league"].fillna("unknown").replace({"unknown": "hs_or_none"})
    t.loc[~t["prior_league"].isin(PRIOR_LEAGUE_LEVELS), "prior_league"] = "hs_or_none"
    t["prior_league_known"] = pd.to_numeric(t["prior_league_known"], errors="coerce").fillna(0).astype(int)
    return t.drop_duplicates(["player_id", "season"])[["player_id", "season", "prior_league", "prior_league_known"]].reset_index(drop=True)


DEST_STYLE = ["fg3_rate", "fta_rate", "to_pct", "oreb_pct", "dreb_pct"]


def destination_context(stores: Stores, seasons: tuple[int, int]) -> pd.DataFrame:
    """Per (team, season): what returns from the prior season, as known at the preseason cutoff (v8 plan item 11).

    Returning players are those on the season's roster with a provider season line on the same team the season
    before; their shares of that team's prior-season minutes, points, rebounds and starts, their minutes-weighted
    prior-season NCAA RAPM, and the team's prior-season style. ``dest_ret_measurable`` is 0 when the team has no
    prior-season lines (a new programme), where every share is NaN. The roster is the undated final roster: a
    departure after the cutoff still counts as returning (known caution)."""
    rosters = stores.rosters()[["team_id", "season", "player_id"]].drop_duplicates()
    rosters = rosters[(rosters["season"] >= seasons[0]) & (rosters["season"] <= seasons[1] + 1)]
    summ = stores.ncaa_summaries().copy()
    for c in ("minutes", "pts", "trb", "gs"):
        summ[c] = pd.to_numeric(summ[c], errors="coerce").fillna(0.0)
    totals = summ.groupby(["team_id", "season"]).agg(total_min=("minutes", "sum"), total_pts=("pts", "sum"),
                                                    total_reb=("trb", "sum"), total_starts=("gs", "sum")).reset_index()
    totals["season"] = totals["season"] + 1
    prev = summ.rename(columns={"season": "prev_season"})
    prev["season"] = prev["prev_season"] + 1
    ret = rosters.merge(prev[["player_id", "prev_season", "season", "team_id", "minutes", "pts", "trb", "gs"]],
                        on=["player_id", "season", "team_id"], how="inner")
    rapm = stores.player_rapm()
    if rapm is not None:
        rapm = rapm.rename(columns={"season": "prev_season"}).drop_duplicates(["player_id", "prev_season"])
        ret = ret.merge(rapm[["player_id", "prev_season", "orapm", "drapm", "rapm"]], on=["player_id", "prev_season"], how="left")
    else:
        ret["orapm"] = ret["drapm"] = ret["rapm"] = np.nan
    has = ret["rapm"].notna()
    w = ret["minutes"].where(has, 0.0)
    ret["w"], ret["has_rapm"] = w, has.astype(int)
    for c in ("rapm", "orapm", "drapm"):
        ret[f"w_{c}"] = w * ret[c].fillna(0.0)
    agg = ret.groupby(["team_id", "season"]).agg(
        ret_min=("minutes", "sum"), ret_pts=("pts", "sum"), ret_reb=("trb", "sum"), ret_starts=("gs", "sum"),
        dest_ret_players=("player_id", "size"), dest_ret_rapm_n=("has_rapm", "sum"), w=("w", "sum"),
        w_rapm=("w_rapm", "sum"), w_orapm=("w_orapm", "sum"), w_drapm=("w_drapm", "sum")).reset_index()
    teams = rosters[["team_id", "season"]].drop_duplicates().merge(totals, on=["team_id", "season"], how="left")
    teams = teams.merge(agg, on=["team_id", "season"], how="left")
    for c in ("ret_min", "ret_pts", "ret_reb", "ret_starts", "dest_ret_players", "dest_ret_rapm_n", "w", "w_rapm", "w_orapm", "w_drapm"):
        teams[c] = teams[c].fillna(0.0)
    measurable = teams["total_min"].fillna(0.0) > 0

    def share(numerator, denominator):
        return (numerator / denominator.where(denominator > 0)).clip(lower=0.0, upper=1.0).where(measurable)

    weighted = lambda c: (teams[f"w_{c}"] / teams["w"].where(teams["w"] > 0)).where(measurable)  # noqa: E731
    out = pd.DataFrame({
        "team_id": teams["team_id"].astype("int64"), "season": teams["season"].astype("int64"),
        "dest_ret_min_share": share(teams["ret_min"], teams["total_min"]), "dest_ret_pts_share": share(teams["ret_pts"], teams["total_pts"]),
        "dest_ret_reb_share": share(teams["ret_reb"], teams["total_reb"]), "dest_ret_starts_share": share(teams["ret_starts"], teams["total_starts"]),
        "dest_ret_players": teams["dest_ret_players"].where(measurable), "dest_ret_rapm_mean": weighted("rapm"),
        "dest_ret_orapm_mean": weighted("orapm"), "dest_ret_drapm_mean": weighted("drapm"),
        "dest_ret_rapm_n": teams["dest_ret_rapm_n"].where(measurable), "dest_ret_measurable": measurable.astype(float),
    })
    style = stores.team_seasons()[["team_id", "season", *DEST_STYLE]].copy()
    style["season"] = style["season"] + 1
    style = style.rename(columns={c: f"dest_{c}" for c in DEST_STYLE}).drop_duplicates(["team_id", "season"])
    out = out.merge(style, on=["team_id", "season"], how="left")
    return out.sort_values(["season", "team_id"]).reset_index(drop=True)


def intl_rapm_anchor_table(stores: Stores) -> pd.DataFrame | None:
    t = stores.player_rapm_intl()
    if t is None:
        return None
    t = t.copy()
    for c in ("orapm", "drapm", "rapm", "n_poss", "n_leagues"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t.dropna(subset=["season"]).drop_duplicates(["player_id", "season"]).sort_values(["player_id", "season"]).reset_index(drop=True)


def on3_table(stores: Stores) -> pd.DataFrame | None:
    t = stores.on3_rankings()
    if t is None:
        return None
    t = t.copy()
    for c in t.columns:
        if c != "player_id":
            t[c] = pd.to_numeric(t[c], errors="coerce")
    out = pd.DataFrame({"player_id": t["player_id"].astype("int64"), "class_year": t["class_year"],
                        "on3_rank": t["consensus_national_rank"].where(t["consensus_national_rank"].notna(), t["on3_national_rank"]),
                        "on3_stars": t["consensus_stars"].where(t["consensus_stars"].notna(), t["on3_stars"]),
                        "on3_rating": t["consensus_rating"].where(t["consensus_rating"].notna(), t["on3_rating"])})
    return out.dropna(subset=["class_year"]).drop_duplicates(["player_id", "class_year"], keep="last").reset_index(drop=True)


def _latest_rows(units, table, *, on, time_col, limit, columns, prefix):
    """Attach the latest row of ``table`` per key whose ``time_col`` ≤ the unit's limit, columns renamed with ``prefix``."""
    table = table.sort_values([on, time_col])
    grouped = {k: g for k, g in table.groupby(on)}
    rows = []
    for key, lim in zip(units[on], limit):
        g = grouped.get(key)
        if g is not None:
            g = g[g[time_col] <= lim]
        rows.append(list(g.iloc[-1][columns]) if g is not None and len(g) else [np.nan] * len(columns))
    names = [f"{prefix}{c}" for c in columns]
    return pd.concat([units, pd.DataFrame(rows, columns=names, index=units.index)], axis=1)


def attach_unit_blocks(units: pd.DataFrame, aux: dict) -> pd.DataFrame:
    """The D-065 unit-level blocks from the auxiliary tables (``player_prior_league``, ``destination_context``,
    ``intl_rapm_anchor``, ``on3_rankings``). A missing table leaves its block's columns absent, which the feature
    side reads as masked. Used by the assembler and by the deployment forecast on the same rule."""
    units = units.copy()
    pl = aux.get("player_prior_league")
    if pl is not None:
        units = units.merge(pl[["player_id", "season", "prior_league", "prior_league_known"]], on=["player_id", "season"], how="left")
        units["prior_league"] = units["prior_league"].astype(object)
        units["prior_league_known"] = pd.to_numeric(units["prior_league_known"], errors="coerce")
        missing = units["prior_league"].isna().to_numpy()
        if missing.any():
            roles = units.loc[missing, "role"].astype(object).tolist()
            fallback = [PRIOR_LEAGUE_BY_ROLE.get(r, (None, 0)) for r in roles]
            units.loc[missing, "prior_league"] = pd.Series([f[0] for f in fallback], index=units.index[missing], dtype=object)
            units.loc[missing, "prior_league_known"] = pd.Series([float(f[1]) for f in fallback], index=units.index[missing])
        units["prior_league_known"] = units["prior_league_known"].fillna(0).astype(int)
    dc = aux.get("destination_context")
    if dc is not None:
        units = units.merge(dc, on=["team_id", "season"], how="left")
        units["dest_ret_measurable"] = units["dest_ret_measurable"].fillna(0.0)
    anchors = aux.get("intl_rapm_anchor")
    if anchors is not None:
        units = _latest_rows(units, anchors, on="player_id", time_col="season", limit=units["season"] - 1,
                             columns=["orapm", "drapm", "n_poss", "season"], prefix="intl_rapm_")
        units["intl_rapm_poss"] = units.pop("intl_rapm_n_poss")
        units["intl_rapm_season_gap"] = units["season"] - units.pop("intl_rapm_season")
    on3 = aux.get("on3_rankings")
    if on3 is not None:
        units = _latest_rows(units, on3, on="player_id", time_col="class_year", limit=units["season"] - 1,
                             columns=["on3_rank", "on3_stars", "on3_rating"], prefix="")
        fill = units["recruit_national_rank"].isna() & units["on3_rank"].notna()
        units["recruit_rank_from_on3"] = fill.astype(float)
        units.loc[fill, "recruit_national_rank"] = units.loc[fill, "on3_rank"]
        stars = units["recruit_star_rating"].isna() & units["on3_stars"].notna()
        units.loc[stars, "recruit_star_rating"] = units.loc[stars, "on3_stars"]
    return units


def block_coverage(units: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Build audit for the D-065 blocks: coverage on eligible first-season units, on international entrants, and on game rows."""
    fy = units["likelihood_eligible"] & units["first_year"]
    ie = fy & units["intl_entrant"]

    def share(mask, col):
        return float(units.loc[mask, col].notna().mean()) if col in units and mask.any() else None

    out = {col: {"first_year": share(fy, col), "intl_entrants": share(ie, col)}
           for col in ("prior_league", "dest_ret_min_share", "dest_fg3_rate", "intl_rapm_orapm")}
    if "prior_league_known" in units:
        out["prior_league_known_share"] = {"first_year": float(units.loc[fy, "prior_league_known"].mean()) if fy.any() else None,
                                           "intl_entrants": float(units.loc[ie, "prior_league_known"].mean()) if ie.any() else None}
        out["prior_league_levels_first_year"] = units.loc[fy, "prior_league"].value_counts().to_dict()
    if "recruit_rank_from_on3" in units:
        out["on3_rank_filled_first_year_units"] = int(units.loc[fy, "recruit_rank_from_on3"].sum())
    if "usage_game" in games:
        out["usage_game_share_by_source"] = games.groupby("source")["usage_game"].apply(lambda s: float(s.notna().mean())).to_dict()
    if "rapm_orapm" in games:
        intl = games["source"] == "intl"
        out["rolling_rapm_share_intl_rows"] = float(games.loc[intl, "rapm_orapm"].notna().mean()) if intl.any() else None
    return out


# --------------------------------------------------------------------------- #
# Outcome units
# --------------------------------------------------------------------------- #

def build_units(stores: Stores, ncaa_games: pd.DataFrame, team_games: pd.DataFrame, games: pd.DataFrame,
                *, lag_days: int, seasons: tuple[int, int], aux: dict | None = None) -> pd.DataFrame:
    all_rosters = stores.rosters()          # every upstream roster season (1997 on): career history (D-066)
    rosters = all_rosters[(all_rosters["season"] >= seasons[0]) & (all_rosters["season"] <= seasons[1])].copy()
    rosters["unit_id"] = (rosters["player_id"].astype(str) + ":" + rosters["season"].astype(str)
                          + ":" + rosters["team_id"].astype(str))
    # Player lines on the roster team; lines without a team go to a unique roster team.
    logs = ncaa_games.copy()
    roster_team = rosters.groupby(["player_id", "season"])["team_id"].agg(["nunique", "first"]).reset_index()
    logs = logs.merge(roster_team, on=["player_id", "season"], how="left")
    missing = logs["team_id"].isna() & (logs["nunique"] == 1)
    logs.loc[missing, "team_id"] = logs.loc[missing, "first"]
    logs = logs.dropna(subset=["team_id"])
    logs["team_id"] = logs["team_id"].astype("int64")
    team_season = team_season_ledger(team_games, logs, ncaa_games)
    # Destination context known at the cutoff: the team's prior-season game count.
    prior_games = team_season[["team_id", "season", "scheduled_games"]].rename(columns={"scheduled_games": "dest_prior_games"})
    prior_games["season"] = prior_games["season"] + 1
    # Rows without appearance evidence are counted separately (unresolved), never as zeros.
    logs["unresolved_int"] = (~logs["appeared"].astype(bool)).astype(int)
    unresolved = logs.groupby(["player_id", "season", "team_id"])["unresolved_int"].sum().rename("unresolved_rows").reset_index()
    appeared = logs[logs["appeared"]].copy()
    # Starts: unknown starter flags stay unknown; the unit's starts are resolved only when every
    # appearance carries a flag (or, below, when the provider season line supplies them).
    appeared["starter_known"] = appeared["starter"].notna().astype(int)
    appeared["starter_int"] = appeared["starter"].fillna(False).astype(int)
    appeared["invalid_int"] = (~appeared["counts_valid"].astype(bool)).astype(int)
    appeared["one"] = 1
    agg = appeared.groupby(["player_id", "season", "team_id"]).agg(
        games_played=("one", "sum"), starts_observed=("starter_int", "sum"), starters_known=("starter_known", "sum"),
        minutes=("minutes", "sum"), points=("pts", "sum"),
        overtime_periods=("overtime_periods", "sum"), overtime_known=("overtime_certified", "all"),
        invalid_rows=("invalid_int", "sum"), **{name: (name, "sum") for name in BOX_FIELDS}).reset_index()
    units = rosters.merge(agg, on=["player_id", "season", "team_id"], how="left")
    units = units.merge(unresolved, on=["player_id", "season", "team_id"], how="left")
    units = units.merge(team_season, on=["team_id", "season"], how="left")
    units = units.merge(prior_games, on=["team_id", "season"], how="left")
    for name in ["games_played", "starts_observed", "starters_known", "minutes", "points", "overtime_periods",
                 "invalid_rows", "unresolved_rows", *BOX_FIELDS]:
        units[name] = units[name].fillna(0)
    units["overtime_known"] = units["overtime_known"].fillna(True).astype(bool)
    units["starts_known"] = units["starters_known"] == units["games_played"]
    units["release_date"] = units["period_end"] + pd.Timedelta(days=lag_days)
    units["horizon"] = "full"
    # Provider season totals reconcile each unit's game lines (the paper's summary coverage test):
    # games equal, minutes within tolerance, and starts equal whenever both sides know them. A published
    # line with no game rows is a coverage gap; no published line plus a consistent team schedule is
    # provider-consistent nonparticipation.
    summary = stores.ncaa_summaries().rename(columns={"gp": "summary_gp", "gs": "summary_gs", "minutes": "summary_minutes"})
    summary = summary.drop_duplicates(["player_id", "season", "team_id"])
    units = units.merge(summary, on=["player_id", "season", "team_id"], how="left")
    games_ok = (units["summary_gp"] == units["games_played"]) & ((units["summary_minutes"] - units["minutes"]).abs() <= MINUTE_TOLERANCE)
    starts_ok = units["summary_gs"].isna() | ~units["starts_known"] | (units["summary_gs"] == units["starts_observed"])
    units["summary_reconciled"] = games_ok & starts_ok
    # Starts resolve from the game lines when every appearance has a flag, else from a reconciled season line.
    from_lines = units["starts_known"]
    from_summary = ~from_lines & games_ok & units["summary_gs"].notna()
    units["starts"] = np.where(from_lines, units["starts_observed"], np.where(from_summary, units["summary_gs"], np.nan))
    units["starts_source"] = np.where(from_lines, "game_lines", np.where(from_summary, "season_line", "unresolved"))
    units["exclusion_reasons"], units["likelihood_eligible"], units["evidence"] = _gate(units)
    # Roster history and attributes as of the forecast cutoff.
    units["forecast_cutoff"] = pd.to_datetime([forecast_cutoff(int(s)) for s in units["season"]])
    units = career_history(all_rosters, units)
    players = stores.players()
    players["dob"] = pd.to_datetime(players["dob"], errors="coerce")
    units = units.merge(players.drop(columns=["name"]), on="player_id", how="left")
    units["age_at_cutoff"] = (units["forecast_cutoff"] - units["dob"]).dt.days / 365.2425
    recruiting = stores.recruiting()
    # Ranks above RANK_LIMIT are parser leakage upstream (D-021; 2026-09-29: 693 such values re-entered roster.db from
    # draft.db). The rank is nulled and the row kept, so the composite score and stars, which are real, survive (D-055).
    leaked = recruiting["national_rank"] > RANK_LIMIT
    recruiting.loc[leaked, "national_rank"] = np.nan
    recruiting.loc[leaked, "pos_rank"] = np.nan
    units = _latest_before(units, recruiting, on="player_id", time_col="class_year", limit=units["season"] - 1,
                           columns=["national_rank", "pos_rank", "composite_score", "star_rating", "class_year"])
    ratings = stores.ncaa_team_ratings().rename(columns={s: f"dest_prior_{s}" for s in STRENGTH})
    ratings["season"] = ratings["season"] + 1
    units = units.merge(ratings[["team_id", "season", *[f"dest_prior_{s}" for s in STRENGTH]]], on=["team_id", "season"], how="left")
    units = units.merge(stores.team_seasons()[["team_id", "season", "conference", "conference_id"]], on=["team_id", "season"], how="left")
    units = _evidence_profile(units, games)
    units = _showcase_profile(units, games)
    # Cohort flags (D-023).
    units["intl_entrant"] = units["first_year"] & ((units["n_intl_games"] > 0) | (units["n_national_nonusa_games"] > 0))
    units["any_pre_ncaa"] = units["first_year"] & ((units["n_intl_games"] + units["n_national_games"] + units["n_event_games"]) > 0)
    units["intl_history"] = (units["n_intl_games"] > 0) | (units["n_national_nonusa_games"] > 0)
    units = attach_unit_blocks(units, aux or {})          # D-065: the On3 fill precedes the recruiting status
    units["recruit_status"] = recruit_status(units)
    return units


RANKINGS_FIRST_CLASS = 2003      # first 247 class in recruiting_rankings


def recruit_status(units: pd.DataFrame) -> pd.Series:
    """Why a unit has or lacks a recruiting rank (D-049), so the model never reads "never evaluated" as "evaluated
    and left out":

    * ``ranked`` — a 247 national rank exists;
    * ``rated_not_ranked`` — a 247 composite or star rating exists but no national rank (some classes publish
      ratings beyond the ranked list, e.g. 2011–2017 and 2023);
    * ``not_rankable_international`` — no 247 row and an international pathway: pre-college club or youth games or
      non-USA national-team games, or a nationality outside the United States and Canada. 247 does not evaluate
      these players systematically, so a missing rank carries no information about quality;
    * ``class_not_covered`` — the player's class predates the rankings in our data;
    * ``unranked_domestic`` — none of the above: a US or Canadian pathway player in a covered class whom the
      services did not rank.
    """
    rank = units["recruit_national_rank"].notna()
    rated = units["recruit_composite_score"].notna() | units["recruit_star_rating"].notna()
    nat = units["nationality"].astype("string").fillna("")
    foreign_only = (nat != "") & ~nat.str.contains("United States|Canada", regex=True)
    intl_path = (units["n_intl_games"] > 0) | (units["n_national_nonusa_games"] > 0) | foreign_only
    first_class = units["season"] - units["ncaa_seasons_completed"].fillna(0) - 1
    status = np.select([rank, rated, intl_path, first_class < RANKINGS_FIRST_CLASS],
                       ["ranked", "rated_not_ranked", "not_rankable_international", "class_not_covered"], "unranked_domestic")
    return pd.Series(status, index=units.index, dtype="string")


def team_season_ledger(team_games: pd.DataFrame, logs: pd.DataFrame, ncaa_games: pd.DataFrame) -> pd.DataFrame:
    """Every team-game known from team lines or any player line, with the coverage certificate.

    A game with no player lines for this team stays in the denominator and
    counts as uncertified: missing logs reduce coverage, never the schedule
    (review finding 1, D-026). Dates come from any log of the game, either
    team; an undated game is kept, uncertified.
    """
    ledger = pd.concat([team_games[["game_id", "team_id", "season", "coverage_certified"]],
                        logs[["game_id", "team_id", "season", "coverage_certified"]]]).drop_duplicates(["game_id", "team_id"])
    own = logs[["game_id", "team_id"]].drop_duplicates().assign(has_logs=True)
    ledger = ledger.merge(own, on=["game_id", "team_id"], how="left")
    ledger["coverage_certified"] = (ledger["coverage_certified"].fillna(False).astype(bool)
                                    & ledger["has_logs"].fillna(False).astype(bool)).astype(int)
    dates = ncaa_games.groupby("game_id")["date"].min().rename("date").reset_index()
    ledger = ledger.merge(dates, on="game_id", how="left")
    fallback = end_year_label(ledger["date"].fillna(pd.Timestamp("1900-01-01")))
    ledger["season"] = ledger["season"].where(ledger["season"].notna(), fallback).astype("int64")
    ledger = ledger[ledger["season"] > 1900]
    out = ledger.groupby(["team_id", "season"]).agg(
        scheduled_games=("game_id", "nunique"), period_start=("date", "min"), period_end=("date", "max"),
        certified_games=("coverage_certified", "sum"), undated_games=("date", lambda d: int(d.isna().sum()))).reset_index()
    out["certified_games"] = out["certified_games"].astype("int64")
    out["coverage_complete"] = out["certified_games"] == out["scheduled_games"]
    return out


def _gate(units: pd.DataFrame):
    """Exclusion reasons through the OutcomeUnit contract, with the evidence used."""
    reasons, eligible, evidence = [], [], []
    for row in units.itertuples(index=False):
        if pd.isna(row.period_start) or pd.isna(row.scheduled_games):
            reasons.append("NO_TEAM_SCHEDULE")
            eligible.append(False)
            evidence.append(None)
            continue
        played = row.games_played > 0
        consistent = bool(row.coverage_complete)
        starts = None if pd.isna(row.starts) else int(row.starts)
        if played and bool(row.summary_reconciled) and starts is not None:
            proof = "SEASON_SUMMARY_RECONCILED"
        elif (not played and consistent and row.unresolved_rows == 0
              and (pd.isna(row.summary_gp) or row.summary_gp == 0)):
            # Provider-consistent nonparticipation: no season line, no listed line, and every team game's
            # minutes reach the regulation total. The minute sum is a consistency check, not proof that a
            # zero-minute appearance is absent (D-027).
            proof = "NO_SEASON_LINE_AND_TEAM_MINUTE_SUM_CONSISTENT"
        else:
            proof = None
        complete = proof is not None and row.invalid_rows == 0
        try:
            unit = OutcomeUnit(
                player_id=str(row.player_id), season=int(row.season), horizon="full",
                period_start=row.period_start.date(), period_end=row.period_end.date(), release_date=row.release_date.date(),
                scheduled_games=int(row.scheduled_games), games_played=int(row.games_played), starts=starts,
                minutes=float(row.minutes), counts=tuple(int(getattr(row, n)) for n in BOX_FIELDS),
                roster_confirmed=True, completeness="COMPLETE" if complete else "INCOMPLETE_COVERAGE",
                participation_verified=complete, no_participation_verified=complete and not played,
                minutes_provenance="realgm_ncaa_integer_minutes_per_game", provenance_verified=True,
                evidence=proof if complete else None)
        except ValueError as exc:
            reasons.append("CONTRACT_VIOLATION:" + str(exc).replace(" ", "_"))
            eligible.append(False)
            evidence.append(None)
            continue
        reasons.append("|".join(unit.exclusion_reasons))
        eligible.append(unit.likelihood_eligible)
        evidence.append(proof if complete else None)
    return reasons, eligible, evidence


FRESHMAN_CLASSES = ("Fr", "RS-Fr", "Fr (Ineligible)")     # upstream class labels of a first-year player


def career_history(all_rosters: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """First D1 roster season, first-season flag and seasons completed per unit, from every upstream roster season.

    Before D-066 these were computed from the tables' season range only (2003 on), so every 2003 unit was a "first
    season" (2,633 eligible returners among them) and seasons completed were undercounted through 2006. Roster rows
    after the tables' range (e.g. next season's roster) cannot change either quantity for earlier units.

    Role (D-074): the upstream rule turns a first-year player into ``transfer_d1`` when any tracked international game
    precedes the season, so the label depends on when the game logs were loaded: the training seasons were classified
    in April 2026 before most logs existed (897 of 1,377 first-year players with club or youth games read ``freshman``,
    918 of 935 with national-team games only), the 2026–27 roster after them (every such player ``transfer_d1``).
    Here the role is a function of NCAA roster history alone: a first D1 season in a freshman class is ``freshman``
    whatever the player did abroad, which the game towers, the evidence vector and the prior-league level carry. The
    upstream reading is kept as ``role_upstream`` for the audit."""
    history = all_rosters[["player_id", "season"]].drop_duplicates()
    first = history.groupby("player_id")["season"].min().rename("first_season")
    out = units.drop(columns=[c for c in ("first_season",) if c in units]).merge(first, left_on="player_id", right_index=True, how="left")
    out["first_year"] = out["season"] == out["first_season"]
    out["ncaa_seasons_completed"] = _prior_seasons(history, out)
    if "role" in out and "class" in out:
        out["role_upstream"] = out["role"]
        hit = (out["first_year"].to_numpy(dtype=bool, na_value=False)
               & out["class"].astype(object).isin(FRESHMAN_CLASSES).to_numpy()
               & (out["role"].astype(object) == "transfer_d1").to_numpy())
        out.loc[hit, "role"] = "freshman"
    return out


def _prior_seasons(prior: pd.DataFrame, units: pd.DataFrame) -> list[int]:
    prior = prior.sort_values(["player_id", "season"])
    grouped = {p: g["season"].values for p, g in prior.groupby("player_id")}
    return [int(np.searchsorted(grouped[p], s)) for p, s in zip(units["player_id"], units["season"])]


def _latest_before(units, table, *, on, time_col, limit, columns):
    """Attach the latest row of ``table`` per key whose ``time_col`` ≤ the unit's limit, as ``recruit_*`` columns."""
    return _latest_rows(units, table, on=on, time_col=time_col, limit=limit, columns=columns, prefix="recruit_")


def _evidence_profile(units: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Tracked history released before each unit's forecast cutoff, by kind (D-023)."""
    strength_cols = ["team_adj_o", "team_adj_d", "team_adj_pace", "team_adj_o_sd", "team_adj_d_sd", "team_strength_available"]
    g = games.copy()
    for col in strength_cols:                     # tables built before D-039 carry no own-team uncertainty
        if col not in g:
            g[col] = np.nan
    g = g[["player_id", "source", "date", "release_date", "minutes", "competition_id", "team_is_usa", "season", *strength_cols]]
    g["team_strength_available"] = pd.to_datetime(g["team_strength_available"])
    g = g.sort_values(["player_id", "release_date"])
    per_player = {p: grp for p, grp in g.groupby("player_id")}
    names = ["n_intl_games", "n_national_games", "n_national_nonusa_games", "n_event_games", "n_ncaa_prior_games",
             "tracked_minutes", "n_competitions", "first_tracked_date", "last_tracked_date", "days_since_last_tracked",
             # D-039: strength of the most recent pre-cutoff international team whose rating is available by the cutoff.
             "last_team_adj_o", "last_team_adj_d", "last_team_adj_pace", "last_team_adj_o_sd", "last_team_adj_d_sd", "last_team_strength_date"]
    empty_strength = [np.nan, np.nan, np.nan, np.nan, np.nan, pd.NaT]
    rows = []
    for player, cutoff, season in zip(units["player_id"], units["forecast_cutoff"], units["season"]):
        grp = per_player.get(player)
        if grp is None:
            rows.append([0, 0, 0, 0, 0, 0.0, 0, pd.NaT, pd.NaT, np.nan, *empty_strength])
            continue
        k = int(np.searchsorted(grp["release_date"].values, np.datetime64(cutoff), side="right"))
        before = grp.iloc[:k]
        src = before["source"].values
        ncaa_prior = int(((src == "ncaa") & (before["season"].values < season)).sum())
        other = before[src != "ncaa"]
        rated = other[(other["team_strength_available"] <= cutoff) & other["team_adj_o"].notna()]
        if len(rated):
            last = rated.loc[rated["date"].idxmax()]
            strength = [float(last["team_adj_o"]), float(last["team_adj_d"]), float(last["team_adj_pace"]),
                        float(last["team_adj_o_sd"]), float(last["team_adj_d_sd"]), last["date"]]
        else:
            strength = empty_strength
        rows.append([int((src == "intl").sum()), int((src == "national").sum()),
                     int(((src == "national") & ~before["team_is_usa"].astype(bool).values).sum()), int((src == "events").sum()),
                     ncaa_prior, float(other["minutes"].fillna(0).sum()), int(other["competition_id"].nunique()),
                     other["date"].min() if len(other) else pd.NaT, other["date"].max() if len(other) else pd.NaT,
                     float((cutoff - other["date"].max()).days) if len(other) else np.nan, *strength])
    return pd.concat([units, pd.DataFrame(rows, columns=names, index=units.index)], axis=1)


SHOWCASE_FLAGS = {
    "showcase_hoop_summit_world": ("nike_hoop_summit", False),
    "showcase_hoop_summit_usa": ("nike_hoop_summit", True),
    "showcase_mcdonalds": ("mcdonalds", None),
    "showcase_jordan_classic": ("jordan_classic", None),
    "showcase_biosteel": ("biosteel_all_canadian", None),
}


def _showcase_profile(units: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """HS-showcase selection per unit (D-039 addendum): which elite events the player was picked for before the
    unit's cutoff, on which side of the Hoop Summit, and the pooled line. Selection is the signal; the game rows
    themselves stay in the channel. Absence means no selection recorded, not unknown."""
    flag_names = list(SHOWCASE_FLAGS)
    names = [*flag_names, "n_showcase_events", "showcase_games", "showcase_starts", "showcase_minutes", "showcase_points"]
    ev = games[games["source"] == "events"]
    if ev.empty:
        out = units.copy()
        for n in flag_names:
            out[n] = False
        for n in names[len(flag_names):]:
            out[n] = 0.0 if n in ("showcase_minutes", "showcase_points") else 0
        return out
    cols = ["player_id", "release_date", "competition", "team_is_usa", "starter", "minutes", "pts"]
    m = units[["unit_id", "player_id", "forecast_cutoff"]].merge(ev[cols], on="player_id", how="inner")
    m = m[pd.to_datetime(m["release_date"]) <= pd.to_datetime(m["forecast_cutoff"])]
    for name, (slug, usa) in SHOWCASE_FLAGS.items():
        hit = m["competition"] == slug
        if usa is not None:
            hit &= m["team_is_usa"].astype(bool) == usa
        m[name] = hit
    agg = m.groupby("unit_id").agg(**{n: (n, "any") for n in flag_names},
                                   n_showcase_events=("competition", "nunique"), showcase_games=("competition", "size"),
                                   showcase_starts=("starter", lambda s: int(s.fillna(False).astype(bool).sum())),
                                   showcase_minutes=("minutes", lambda s: float(s.fillna(0).sum())),
                                   showcase_points=("pts", lambda s: float(s.fillna(0).sum()))).reset_index()
    out = units.merge(agg, on="unit_id", how="left")
    for n in flag_names:
        out[n] = out[n].fillna(False).astype(bool)
    for n in ("n_showcase_events", "showcase_games", "showcase_starts"):
        out[n] = out[n].fillna(0).astype(int)
    for n in ("showcase_minutes", "showcase_points"):
        out[n] = out[n].fillna(0.0).astype(float)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def build_tables(stores: Stores, *, seasons=(2003, 2026), lag_days: int = 1, gap_days: int = GAP_DAYS) -> dict:
    intl, intl_periods = build_intl_games(stores, lag_days=lag_days, gap_days=gap_days)
    national, national_periods = build_tournament_games(stores, "national", lag_days=lag_days)
    events, events_periods = build_tournament_games(stores, "events", lag_days=lag_days)
    ncaa, ncaa_periods, team_games = build_ncaa_games(stores, lag_days=lag_days)
    games = pd.concat([ncaa, intl, national, events], ignore_index=True)
    for col in ("season", "player_id", "game_id", "competition_id"):
        games[col] = games[col].astype("int64")
    slugs = stores.intl_leagues()[["LeagueID", "Slug"]].rename(columns={"LeagueID": "competition_id", "Slug": "competition"})
    periods = pd.concat([ncaa_periods, intl_periods.merge(slugs, on="competition_id", how="left"), national_periods, events_periods],
                        ignore_index=True)
    periods["majority_share"] = periods.get("majority_share", pd.Series(np.nan, index=periods.index)).fillna(1.0)
    periods["guard"] = periods.get("guard", pd.Series("", index=periods.index)).fillna("")
    periods["derivation"] = np.where(periods["source"] == "intl", "date_cluster_gap60_majority_guarded", "edition_or_provider")
    periods = periods[["source", "competition_id", "competition", "season", "start", "end", "majority_share", "guard", "derivation"]]
    # Season provenance per game (D-028): an inferred period is never independently verified evidence,
    # and a period the guard could not resolve is marked so folds can exclude it from nested windows.
    flags = periods[["source", "competition_id", "season", "guard"]].rename(columns={"guard": "season_guard"})
    games = games.merge(flags, on=["source", "competition_id", "season"], how="left")
    games["season_guard"] = games["season_guard"].fillna("")
    games["season_verified"] = games["source"] != "intl"          # NCAA labels and dated editions; inferred periods are not
    # D-065 auxiliary tables (written alongside the tables so the deployment forecast attaches the same blocks).
    aux = {name: table for name, table in (("player_prior_league", prior_league_table(stores)),
                                           ("destination_context", destination_context(stores, seasons)),
                                           ("intl_rapm_anchor", intl_rapm_anchor_table(stores)),
                                           ("on3_rankings", on3_table(stores))) if table is not None}
    units = build_units(stores, ncaa, team_games, games, lag_days=lag_days, seasons=seasons, aux=aux)
    relabelled = games[(games["source"] == "intl") & (games["label_agrees_with_bucket"] == False)]  # noqa: E712
    ambiguous = intl_periods[intl_periods["majority_share"] < 0.9] if "majority_share" in intl_periods else intl_periods.iloc[0:0]
    guarded = intl_periods[intl_periods["guard"] != ""] if "guard" in intl_periods else intl_periods.iloc[0:0]
    span_days = (pd.to_datetime(intl_periods["end"]) - pd.to_datetime(intl_periods["start"])).dt.days
    audit_periods = {
        "intl_periods_guard_relabelled": int((guarded["guard"] == "relabelled_preceding_year").sum()),
        "intl_periods_overlong_unresolved": int((guarded["guard"] == "overlong_merge_unresolved").sum()),
        "intl_periods_over_330_days": int((span_days > MAX_SEASON_DAYS).sum()),
        "intl_periods_guard_examples": guarded.merge(slugs, on="competition_id", how="left").head(20).astype(str).to_dict(orient="records"),
    }
    audit = {
        **audit_periods,
        "games_by_source": games["source"].value_counts().to_dict(),
        "season_method_by_source": {f"{s}:{m}": int(n) for (s, m), n in games.groupby(["source", "season_method"]).size().items()},
        "mapping_version": MAPPING_VERSION,
        "intl_periods_inferred": int(len(intl_periods)),
        "intl_periods_ambiguous_majority_below_0.9": int(len(ambiguous)),
        "intl_periods_ambiguous_examples": ambiguous.head(20).astype(str).to_dict(orient="records"),
        "intl_relabelled_rows": int(len(relabelled)),
        "intl_relabelled_by_competition": relabelled["competition"].value_counts().head(25).to_dict(),
        "intl_relabelled_pairs": {f"{int(a)}->{int(b)}": int(n) for (a, b), n in relabelled.groupby(["original_label", "season"]).size().items()},
        "provider_season_id_attached_share": float(games.loc[games["source"] == "intl", "provider_season_id"].notna().mean()),
        "counts_invalid": int((~games["counts_valid"]).sum()),
        "overtime_certified_share_by_source": games.groupby("source")["overtime_certified"].mean().round(4).to_dict(),
        "ncaa_overtime_source": team_games["overtime_source"].value_counts().to_dict() if "overtime_source" in team_games else {},
        "intl_starter_source": intl.attrs.get("starter_source", {}),
        "ncaa_overtime_provider_vs_sum_agreement": (float(team_games["overtime_agrees"].dropna().astype(float).mean())
                                                    if "overtime_agrees" in team_games and team_games["overtime_agrees"].notna().any() else None),
        "ncaa_team_games_points_inconsistent": int((~ncaa.drop_duplicates(["game_id", "team_id"])["points_consistent"].fillna(True).astype(bool)).sum()),
        "opp_strength_source": {f"{s}:{m}": int(n) for (s, m), n in games.groupby(["source", "opp_strength_source"]).size().items()},
        "units": int(len(units)), "units_eligible": int(units["likelihood_eligible"].sum()),
        "exclusion_reasons": units.loc[~units["likelihood_eligible"], "exclusion_reasons"].value_counts().to_dict(),
        "starts_source": units["starts_source"].value_counts().to_dict(),
        "units_with_unresolved_rows": int((units["unresolved_rows"] > 0).sum()),
        "evidence": units.loc[units["likelihood_eligible"], "evidence"].value_counts().to_dict(),
    }
    # Bios for every player with a game or a unit (D-050): the game store needs a date of birth for each game row.
    bios = stores.players()
    bios["dob"] = pd.to_datetime(bios["dob"], errors="coerce")
    keep = set(games["player_id"].unique()) | set(units["player_id"].unique())
    players_table = bios[bios["player_id"].isin(keep)].drop_duplicates("player_id").reset_index(drop=True)
    nonncaa = games["source"] != "ncaa"
    audit["nonncaa_game_rows_with_dob_share"] = float(games.loc[nonncaa, "player_id"].isin(players_table.loc[players_table["dob"].notna(), "player_id"]).mean())
    audit["recruit_status"] = units.loc[units["likelihood_eligible"] & units["first_year"], "recruit_status"].value_counts().to_dict()
    if "role_upstream" in units:
        # D-074: first-year freshman-class units whose upstream role was transfer_d1 (tracked international history), by season.
        changed = units["role"].astype(object).to_numpy() != units["role_upstream"].astype(object).to_numpy()
        audit["role_normalized_to_freshman"] = {int(s): int(n) for s, n in units.loc[changed].groupby("season").size().items()}
    audit["feature_blocks"] = block_coverage(units, games)
    return {"games": games, "competition_periods": periods, "units": units, "players": players_table, "audit": audit, **aux}
