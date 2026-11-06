"""High school strength from earlier alumni, adjusted for the strength of the NCAA teams they joined (D-091).

Every NCAA freshman season (a player's first rostered season) is ranked within its season and its destination's
strength decile (adjusted offense minus adjusted defense) on two measures: minutes share = minutes / (40 x team
games), and production = Hollinger game score per 40 minutes (seasons with at least 100 minutes). The talent index
is the mean of the two percentiles (minutes alone when production is undefined), so a freshman who cracks the
rotation of a top team and one who produces at a weak team both rank high in their own decile.

A player whose first NCAA season is s and whose high school is h gets, from h's alumni with first season < s (every
one of their freshman seasons ended before the forecast date of season s):
    hs_alumni_n          number of such alumni with a scored freshman season
    hs_talent            their mean talent, shrunk toward the mean of earlier alumni from the same state or
                         country with k pseudo-alumni: (sum + k * prior) / (n + k)
    hs_alumni_min_share  their mean raw minutes share (NaN without alumni)
    hs_top2_share        share of them who joined a top-two-decile team (NaN without alumni)
The school comes from RealGM player pages (table realgm_bios in ncaa_roster_prediction/prep.db); our player_id is
RealGM's id. Both databases are opened read-only.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

K_SHRINK = 6.0
HALF_LIFE = None        # years; None = every earlier alumnus counts equally (D-091); D-093 sets a half-life
HS_COLUMNS = ["player_id", "entry_season", "hs_id", "hs_region", "hs_alumni_n", "hs_talent", "hs_alumni_min_share", "hs_top2_share",
              "hs_alumni_level"]


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)


def freshman_talent(roster_db: Path) -> pd.DataFrame:
    """One row per player with a scored first NCAA season: season, team strength decile, shares and talent."""
    con = _ro(roster_db)
    d = pd.read_sql("""
        WITH first AS (SELECT player_id, MIN(season) fs FROM team_rosters GROUP BY player_id)
        SELECT s.player_id, s.season, s.team_id, s.gp, s.min, s.fg2m + s.fg3m AS fgm, s.fg2a + s.fg3a AS fga, s.ftm, s.fta,
               s.orb, s.drb, s.ast, s.stl, s.blk, s.tov, s.pf, s.pts,
               r.adj_o - r.adj_d AS team_em, t.games_played AS team_games
        FROM ncaa_summaries s JOIN first f ON f.player_id = s.player_id AND f.fs = s.season
        LEFT JOIN ncaa_team_ratings r ON r.team_id = s.team_id AND r.season = s.season
        LEFT JOIN team_seasons t ON t.team_id = s.team_id AND t.season = s.season""", con)
    con.close()
    d = d.sort_values("min", ascending=False)
    box = ["gp", "min", "fgm", "fga", "ftm", "fta", "orb", "drb", "ast", "stl", "blk", "tov", "pf", "pts"]
    sums = d.groupby("player_id")[box].sum()
    main = d.drop_duplicates("player_id").set_index("player_id")[["season", "team_id", "team_em", "team_games"]]  # main team
    d = main.join(sums).reset_index()
    d = d[d["team_em"].notna() & d["team_games"].notna() & (d["team_games"] > 0)].copy()
    d["min"] = d["min"].fillna(0.0)
    d["min_share"] = (d["min"] / (40.0 * d["team_games"])).clip(upper=1.0)   # a few teams carry miscounted games
    gmsc = (d.pts + 0.4 * d.fgm - 0.7 * d.fga - 0.4 * (d.fta - d.ftm) + 0.7 * d.orb + 0.3 * d.drb + d.stl + 0.7 * d.ast
            + 0.7 * d.blk - 0.4 * d.pf - d.tov)
    d["prod40"] = np.where(d["min"] >= 100, 40.0 * gmsc / d["min"].clip(lower=1.0), np.nan)
    d["decile"] = d.groupby("season")["team_em"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    g = d.groupby(["season", "decile"])
    d["talent"] = pd.concat([g["min_share"].rank(pct=True), g["prod40"].rank(pct=True)], axis=1).mean(axis=1)
    return d[["player_id", "season", "decile", "min_share", "prod40", "talent"]]


def entries(roster_db: Path, prep_db: Path) -> pd.DataFrame:
    """Every rostered player's first NCAA season with his RealGM high school (rows without a school are dropped)."""
    con = _ro(roster_db)
    first = pd.read_sql("SELECT player_id, MIN(season) AS entry_season FROM team_rosters GROUP BY player_id", con)
    con.close()
    con = _ro(prep_db)
    bios = pd.read_sql("SELECT player_id, hs_id, hs_region FROM realgm_bios WHERE hs_id IS NOT NULL", con)
    con.close()
    bios["hs_region"] = bios["hs_region"].fillna("unknown")
    return first.merge(bios, on="player_id", how="inner")


def _prior_cumulative(alumni: pd.DataFrame, key: str, values: list[str]) -> pd.DataFrame:
    """Per (key, season): sums of `values` and counts over alumni with season strictly before that season."""
    per = alumni.groupby([key, "season"])[values].sum()
    per["n"] = alumni.groupby([key, "season"]).size()
    per = per.sort_index()
    cum = per.groupby(level=0).cumsum()
    return cum.reset_index().rename(columns={"season": "through"})       # sums over seasons <= through


def school_scores(talent: pd.DataFrame, ent: pd.DataFrame, k: float = K_SHRINK) -> pd.DataFrame:
    """hs_* columns for every entry, from alumni whose first season precedes the entry season."""
    al = talent.merge(ent[["player_id", "hs_id", "hs_region"]], on="player_id", how="inner")
    al["top2"] = (al["decile"] >= 8).astype(float)
    al["min_share_c"] = al["min_share"].fillna(0.0)
    vals = ["talent", "min_share_c", "top2"]
    out = ent.copy()
    out["lookup"] = out["entry_season"] - 1                                 # alumni seasons <= entry - 1

    def asof(cum: pd.DataFrame, key: str) -> pd.DataFrame:
        left = out[[key, "lookup"]].reset_index().sort_values("lookup")
        right = cum.sort_values("through")
        m = pd.merge_asof(left, right, left_on="lookup", right_on="through", by=key, direction="backward")
        return m.set_index("index").reindex(out.index)

    sch = asof(_prior_cumulative(al, "hs_id", vals), "hs_id")
    reg = asof(_prior_cumulative(al, "hs_region", vals), "hs_region")
    glob_cum = al.groupby("season")[["talent"]].sum().assign(n=al.groupby("season").size()).sort_index().cumsum().reset_index()
    glob = pd.merge_asof(out[["lookup"]].reset_index().sort_values("lookup"), glob_cum.rename(columns={"season": "through"}),
                         left_on="lookup", right_on="through", direction="backward").set_index("index").reindex(out.index)
    n = sch["n"].fillna(0.0)
    prior = (reg["talent"] / reg["n"]).where(reg["n"] > 0, glob["talent"] / glob["n"]).fillna(0.5)
    out["hs_alumni_n"] = n
    out["hs_talent"] = (sch["talent"].fillna(0.0) + k * prior) / (n + k)
    out["hs_alumni_min_share"] = (sch["min_share_c"] / n).where(n > 0)
    out["hs_top2_share"] = (sch["top2"] / n).where(n > 0)
    out["hs_alumni_level"] = np.nan                                      # D-091 tables carry no alumni level
    out["hs_region"] = out["hs_region"].astype(object)
    return out[HS_COLUMNS].reset_index(drop=True)


def build_hs_school(roster_db: Path, prep_db: Path, k: float = K_SHRINK, half_life: float | None = None,
                    level: bool = False) -> tuple[pd.DataFrame, dict]:
    """half_life None and level False reproduce the D-091 table (tables_v12); D-093 (tables_v13): half_life 8, level."""
    talent = freshman_talent(roster_db)
    ent = entries(roster_db, prep_db)
    if half_life is None and not level:
        table = school_scores(talent, ent, k)
    else:
        table = school_scores_decay(talent, ent, k, half_life, alumni_ncaa_seasons(roster_db, ent) if level else None)
    con = _ro(prep_db)
    bios_digest = hashlib.sha256(pd.read_sql("SELECT * FROM realgm_bios ORDER BY player_id", con)
                                 .to_csv(index=False).encode()).hexdigest()
    con.close()
    info = {"k_shrink": k, "half_life": half_life, "alumni_level": level, "entries": int(len(table)), "scored_freshman_seasons": int(len(talent)),
            "distinct_schools": int(table["hs_id"].nunique()), "realgm_bios_sha256": bios_digest,
            "with_alumni_share": float((table["hs_alumni_n"] > 0).mean())}
    return table, info

def school_scores_decay(talent: pd.DataFrame, ent: pd.DataFrame, k: float = K_SHRINK, half_life: float | None = HALF_LIFE,
                        alumni_seasons: pd.DataFrame | None = None) -> pd.DataFrame:
    """D-093: as school_scores, but an alumnus whose freshman season was t years before the entry season weighs
    0.5 ** ((t - 1) / half_life) (the most recent possible alumnus weighs 1), because a school's identity drifts
    (coaches, enrolment, programme strength). Counts become effective counts; shrinkage toward the earlier
    state/country mean (decayed the same way) uses the effective count. half_life None reproduces school_scores."""
    al = talent.merge(ent[["player_id", "hs_id", "hs_region"]], on="player_id", how="inner")
    al["top2"] = (al["decile"] >= 8).astype(float)
    al["min_share_c"] = al["min_share"].fillna(0.0)
    out = ent.copy().reset_index(drop=True)
    w = (lambda gap: np.power(0.5, (gap - 1.0) / half_life)) if half_life else (lambda gap: np.ones_like(gap, dtype=float))

    def weighted(key: str) -> pd.DataFrame:
        per = al.groupby([key, "season"]).agg(n=("player_id", "size"), talent=("talent", "sum"),
                                              min_share=("min_share_c", "sum"), top2=("top2", "sum")).reset_index()
        m = out[[key, "entry_season"]].reset_index().merge(per, on=key, how="inner")
        m = m[m["season"] < m["entry_season"]]
        wt = w((m["entry_season"] - m["season"]).to_numpy(float))
        agg = pd.DataFrame({"index": m["index"], "n": wt * m["n"], "talent": wt * m["talent"],
                            "min_share": wt * m["min_share"], "top2": wt * m["top2"]}).groupby("index").sum()
        return agg.reindex(out.index)

    sch, reg = weighted("hs_id"), weighted("hs_region")
    raw_n = (out[["hs_id", "entry_season"]].reset_index().merge(al[["hs_id", "season"]], on="hs_id", how="inner")
             .query("season < entry_season").groupby("index").size().reindex(out.index).fillna(0.0))
    per_g = al.groupby("season").agg(n=("player_id", "size"), talent=("talent", "sum")).reset_index()
    glob = []
    for e in out["entry_season"].to_numpy():
        g = per_g[per_g["season"] < e]
        wt = w((e - g["season"]).to_numpy(float))
        glob.append((wt * g["talent"]).sum() / max((wt * g["n"]).sum(), 1e-9) if len(g) else np.nan)
    prior = (reg["talent"] / reg["n"]).where(reg["n"] > 0, pd.Series(glob, index=out.index)).fillna(0.5)
    n = sch["n"].fillna(0.0)                                            # effective (recency-weighted) count, for shrinkage
    out["hs_alumni_n"] = raw_n                                          # plain count of earlier alumni (user, D-093)
    out["hs_talent"] = (sch["talent"].fillna(0.0) + k * prior) / (n + k)
    out["hs_alumni_min_share"] = (sch["min_share"] / n).where(n > 0)
    out["hs_top2_share"] = (sch["top2"] / n).where(n > 0)
    out["hs_alumni_level"] = alumni_level(out, alumni_seasons, w) if alumni_seasons is not None else np.nan
    out["hs_region"] = out["hs_region"].astype(object)
    return out[HS_COLUMNS]


def alumni_level(out: pd.DataFrame, seasons: pd.DataFrame, w) -> pd.Series:
    """Typical level at which earlier alumni played (user, D-093): the minutes- and recency-weighted mean adjusted
    efficiency margin (adj_o - adj_d) of the NCAA team-seasons of the school's alumni, over seasons before entry.
    `seasons`: player_id, hs_id, season, min, team_em (every NCAA season of every alumnus)."""
    per = (seasons.assign(mem=seasons["min"] * seasons["team_em"]).groupby(["hs_id", "season"])
           .agg(m=("min", "sum"), mem=("mem", "sum")).reset_index())
    m = out[["hs_id", "entry_season"]].reset_index().merge(per, on="hs_id", how="inner")
    m = m[m["season"] < m["entry_season"]]
    wt = w((m["entry_season"] - m["season"]).to_numpy(float))
    agg = pd.DataFrame({"index": m["index"], "m": wt * m["m"], "mem": wt * m["mem"]}).groupby("index").sum()
    return (agg["mem"] / agg["m"]).where(agg["m"] > 0).reindex(out.index)


def alumni_ncaa_seasons(roster_db: Path, ent: pd.DataFrame) -> pd.DataFrame:
    """Every NCAA season (minutes, destination strength) of every player with a known school."""
    con = _ro(roster_db)
    d = pd.read_sql("""SELECT s.player_id, s.season, SUM(s.min) AS min, MAX(r.adj_o - r.adj_d) AS team_em
                       FROM ncaa_summaries s LEFT JOIN ncaa_team_ratings r ON r.team_id = s.team_id AND r.season = s.season
                       GROUP BY s.player_id, s.season, s.team_id""", con)
    con.close()
    d = d[d["team_em"].notna() & (d["min"] > 0)]
    return d.merge(ent[["player_id", "hs_id"]], on="player_id", how="inner")
