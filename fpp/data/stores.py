"""Read-only access to the upstream stores that feed the training tables.

Four SQLite files supply everything (D-023): ``roster.db`` (rosters, NCAA
game logs and team boxscores, bios, recruiting ranks, NCAA team ratings),
``kenpom/intl.db`` (dated international club and youth games with the
league catalog), ``kenpom/national.db`` (dated national-team tournaments),
and ``ncaa_roster_prediction/events.db`` (dated HS showcase games). The
international team-rating outputs are a fifth, optional store. Every
connection is opened ``mode=ro`` with ``query_only``; nothing here writes.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
DEFAULT_PATHS = {
    "roster": HOME / "Dropbox/ncaa_roster_prediction/roster.db",
    "intl": HOME / "Dropbox/kenpom/intl.db",
    "national": HOME / "Dropbox/kenpom/national.db",
    "events": HOME / "Dropbox/ncaa_roster_prediction/events.db",
    "intl_ratings": HOME / "Dropbox/kenpom/output/intl_team_ratings.db",
    "intl_rapm_rolling": HOME / "Dropbox/intl_approx_rapm/data/intl_rapm_rolling.db",     # D-065: dated RAPM snapshots
}
OPTIONAL_STORES = ("intl_ratings", "intl_rapm_rolling")

BOX_COLUMNS_NCAA = ["fg2a", "fg2m", "fg3a", "fg3m", "fta", "ftm", "orb", "drb", "ast", "stl", "blk", "tov", "pf"]
TEAM_TOTALS = ["FGA", "FTA", "TOV", "AST", "REB", "POSS"]                                  # D-065 usage block


STARTER_SOURCE = {0: "none", 1: "page", 2: "listing_order"}


def resolve_intl_starter(frame: pd.DataFrame) -> pd.DataFrame:
    """One ``Starter`` column (1/0/NaN) from the page value and the listing-order value (D-063).

    The page value wins where it exists, except in a completely parsed team-game whose page marks other than five
    starters (a page error upstream): those rows fall back to the listing order. ``starter_source`` (int8, see
    ``STARTER_SOURCE``) records which value each row carries, for the build audit."""
    n = len(frame)
    page = pd.to_numeric(frame["Starter"], errors="coerce") if "Starter" in frame else pd.Series(np.nan, index=frame.index)
    derived = pd.to_numeric(frame["StarterDerived"], errors="coerce") if "StarterDerived" in frame else pd.Series(np.nan, index=frame.index)
    use_page = page.notna()
    if n and use_page.any():
        key = [frame["GameID"].to_numpy(), frame["TeamID"].to_numpy()]
        complete = page.notna().groupby(key).transform("all")
        marked = page.fillna(0.0).groupby(key).transform("sum")
        use_page &= ~(complete & (marked != 5))
    out = frame.drop(columns=[c for c in ("Starter", "StarterDerived") if c in frame])
    out["Starter"] = np.where(use_page, page, derived)
    out["starter_source"] = np.select([use_page.to_numpy(), derived.notna().to_numpy()], [1, 2], default=0).astype(np.int8)
    return out


def _connect(path: Path) -> sqlite3.Connection:
    path = Path(path).expanduser().resolve(strict=True)
    con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def _fingerprint(path: Path) -> dict:
    stat = Path(path).stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@dataclass
class Stores:
    """Handles to the upstream stores; ``fingerprints`` go into the manifest."""
    paths: dict = field(default_factory=lambda: dict(DEFAULT_PATHS))
    connections: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        for name, path in self.paths.items():
            if name not in OPTIONAL_STORES and not Path(path).is_file():
                raise FileNotFoundError(f"Missing upstream store {name!r}: {path}")

    def connection(self, name: str) -> sqlite3.Connection:
        if name not in self.connections:
            self.connections[name] = _connect(self.paths[name])
        return self.connections[name]

    def close(self):
        for con in self.connections.values():
            con.close()
        self.connections.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    PROBE_TABLES = {
        "roster": ("team_rosters", "ncaa_gamelogs", "ncaa_team_boxscores", "ncaa_summaries", "players",
                   "recruiting_rankings", "ncaa_team_ratings", "team_seasons", "intl_gamelogs",
                   "player_prior_league", "player_rapm", "player_rapm_intl", "on3_industry_rankings"),
        "intl": ("boxscores", "player_boxscores", "leagues", "league_seasons", "competition_periods", "team_history", "teams_years"),
        "national": ("player_boxscores", "boxscores", "tournament_years"),
        "events": ("player_boxscores", "boxscores", "tournament_years"),
        "intl_ratings": ("team_ratings",),
        "intl_rapm_rolling": ("rapm_rolling",),
    }

    def available(self, name: str) -> bool:
        return name in self.paths and Path(self.paths[name]).is_file()

    def has_table(self, store: str, table: str) -> bool:
        if not self.available(store):
            return False
        row = self.connection(store).execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (table,)).fetchone()
        return row is not None

    def content_probe(self, name: str) -> dict:
        """Row counts and maximum rowids of the tables this build reads.

        A SQLite WAL update changes query results without touching the main
        file's size or mtime, so file stats alone cannot identify the source;
        this probe sees such changes. It is a probe of the tables used, not a
        content hash of the database — a frozen fold should build from a
        `VACUUM INTO` snapshot when exact reproducibility is required.
        """
        con = self.connection(name)
        out = {}
        for table in self.PROBE_TABLES.get(name, ()):
            try:
                count, max_rowid = con.execute(f'SELECT COUNT(*), MAX(rowid) FROM "{table}"').fetchone()
            except sqlite3.Error:
                continue
            out[table] = {"rows": int(count), "max_rowid": None if max_rowid is None else int(max_rowid)}
        return out

    def fingerprints(self) -> dict:
        out = {}
        for name, path in self.paths.items():
            path = Path(path)
            if not path.is_file():
                continue
            entry = _fingerprint(path)
            for suffix in ("-wal", "-shm"):
                side = path.with_name(path.name + suffix)
                if side.is_file():
                    entry[suffix.strip("-")] = _fingerprint(side)
            entry["content_probe"] = self.content_probe(name)
            out[name] = entry
        digest = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
        return {"stores": out, "sha256": digest, "identity": "file stats + WAL/SHM stats + table row-count/rowid probe"}

    def frame(self, store: str, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.connection(store), params=params)

    # ------------------------------------------------------------------ roster.db
    def rosters(self) -> pd.DataFrame:
        return self.frame("roster", """
            SELECT team_id, season, player_id, role, class, is_redshirt, prior_team_id
            FROM team_rosters""")

    def players(self) -> pd.DataFrame:
        return self.frame("roster", "SELECT player_id, name, dob, nationality, height_cm, weight_kg, position FROM players")

    def recruiting(self) -> pd.DataFrame:
        return self.frame("roster", """
            SELECT player_id, class_year, national_rank, pos_rank, composite_score, star_rating, height_cm, weight_kg
            FROM recruiting_rankings WHERE player_id IS NOT NULL""")

    def ncaa_team_ratings(self) -> pd.DataFrame:
        return self.frame("roster", "SELECT team_id, season, adj_o, adj_d, adj_pace, adj_o_sd, adj_d_sd FROM ncaa_team_ratings")

    def team_seasons(self) -> pd.DataFrame:
        # Style columns (fg3_rate .. dreb_pct) feed the destination block (D-065) as prior-season context.
        return self.frame("roster", """
            SELECT team_id, season, conference, conference_id, games_played, wins, losses,
                   fg3_rate, fta_rate, to_pct, oreb_pct, dreb_pct
            FROM team_seasons""")

    # ------------------------------------------------------------------ D-065 blocks (roster.db)
    def player_prior_league(self) -> pd.DataFrame | None:
        """Origin league per roster row (player model's adopted feature; `unknown` marks scrape gaps)."""
        if not self.has_table("roster", "player_prior_league"):
            return None
        return self.frame("roster", "SELECT player_id, season, prior_league, prior_league_known FROM player_prior_league")

    def player_rapm(self) -> pd.DataFrame | None:
        """Season-end NCAA RAPM per player-season; only seasons before the unit's may be read (D-065)."""
        if not self.has_table("roster", "player_rapm"):
            return None
        return self.frame("roster", "SELECT player_id, season, orapm, drapm, rapm, n_poss FROM player_rapm")

    def player_rapm_intl(self) -> pd.DataFrame | None:
        """Season anchors of the joint cross-league international RAPM solve (one row per player-season)."""
        if not self.has_table("roster", "player_rapm_intl"):
            return None
        return self.frame("roster", "SELECT player_id, season, orapm, drapm, rapm, n_poss, n_leagues FROM player_rapm_intl")

    def on3_rankings(self) -> pd.DataFrame | None:
        """On3 industry consensus per linked recruit and class (fills 247's rated-but-unranked units, D-065)."""
        if not self.has_table("roster", "on3_industry_rankings"):
            return None
        return self.frame("roster", """
            SELECT player_id, class_year, consensus_national_rank, consensus_stars, consensus_rating,
                   on3_national_rank, on3_stars, on3_rating
            FROM on3_industry_rankings WHERE player_id IS NOT NULL""")

    def intl_rapm_rolling(self) -> pd.DataFrame | None:
        """Dated rolling international RAPM snapshots per (player, upstream season label, league); optional store."""
        if not self.has_table("intl_rapm_rolling", "rapm_rolling"):
            return None
        return self.frame("intl_rapm_rolling", """
            SELECT player_id, season, league_id, cutoff_date, orapm, drapm, rapm, off_equiv FROM rapm_rolling""")

    def ncaa_team_games(self) -> pd.DataFrame:
        """Team lines; ``team_id`` 0 marks an untracked (non-D1) opponent and is dropped. Provider team minutes
        (``minutes``, added upstream 2026-09-29 from kenpom/ncaa.db) are read when the column exists (D-053)."""
        minutes = "minutes" if self.has_column("roster", "ncaa_team_boxscores", "minutes") else "NULL AS minutes"
        return self.frame("roster", f"""
            SELECT game_id, team_id, season, home, pts, poss, game_type, {minutes},
                   fga, fta, tov, ast, reb FROM ncaa_team_boxscores
            WHERE team_id IS NOT NULL AND team_id > 0""")

    def ncaa_summaries(self) -> pd.DataFrame:
        """Provider season totals per roster unit; NULL gp means no stats line was published. Points and rebounds
        feed the destination block's returning shares (D-065)."""
        return self.frame("roster", "SELECT player_id, season, team_id, gp, gs, min AS minutes, pts, trb FROM ncaa_summaries")

    def ncaa_gamelogs(self) -> pd.DataFrame:
        return self.frame("roster", f"""
            SELECT player_id, game_id, season, date, team_id, opp_id, status, min AS minutes, pts,
                   {", ".join(BOX_COLUMNS_NCAA)}, game_type
            FROM ncaa_gamelogs""")

    def intl_status(self) -> pd.DataFrame:
        """Starter/bench status re-parsed upstream; joins intl.db on (player, game)."""
        return self.frame("roster", """
            SELECT player_id, game_id, status FROM intl_gamelogs
            WHERE status IS NOT NULL AND league NOT IN ('National-Team', 'HS-Showcase')""")

    # ------------------------------------------------------------------ intl.db
    def intl_leagues(self) -> pd.DataFrame:
        return self.frame("intl", "SELECT LeagueID, Slug, Name, Kind, Country, Section FROM leagues")

    def intl_league_seasons(self) -> pd.DataFrame:
        """RealGM's canonical season ids per league (standings-page selector)."""
        return self.frame("intl", "SELECT LeagueID, provider_season_id, season_end_year, span_label FROM league_seasons")

    def intl_bucket_periods(self) -> pd.DataFrame:
        """Dated span of each (league, June-1 label) bucket, as built upstream."""
        return self.frame("intl", "SELECT LeagueID, Season, start_date, end_date, n_games FROM competition_periods")

    def intl_team_history(self) -> pd.DataFrame:
        """Record and placement per team-season from RealGM team pages."""
        return self.frame("intl", "SELECT TeamID, Season, LeagueName, Record, Placement FROM team_history")

    def intl_team_games(self) -> pd.DataFrame:
        return self.frame("intl", """
            SELECT GameID, TeamID, LeagueID, Season, Date, Home, Minutes, PTS, POSS, FGA, FTA, TOV, AST, REB FROM boxscores""")

    def intl_player_games(self) -> pd.DataFrame:
        # Starter flag (D-063): ``Starter`` is the value parsed from the box-score page (filled upstream since 2026-09-29,
        # D-053); ``StarterDerived`` is the listing order (first five players listed per team, 99.96% agreement with the
        # page value where both exist). ``resolve_intl_starter`` combines them.
        present = [c for c in ("Starter", "StarterDerived") if self.has_column("intl", "player_boxscores", c)]
        extra = "".join(f", {c}" for c in present)
        return resolve_intl_starter(self.frame("intl", f"""
            SELECT GameID, TeamID, PlayerID, LeagueID, Season, Date, Home, Pos, Min,
                   FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, AST, PF, STL, TOV, BLK, PTS{extra}
            FROM player_boxscores"""))

    def intl_ratings(self) -> pd.DataFrame | None:
        if not Path(self.paths["intl_ratings"]).is_file():
            return None
        # Rating uncertainty (adj_o_sd, adj_d_sd) travels with the rating (D-039).
        return self.frame("intl_ratings", "SELECT team_id, season, league_id, adj_o, adj_d, adj_pace, adj_o_sd, adj_d_sd FROM team_ratings")

    # ------------------------------------------------------------------ national.db / events.db
    def tournament_editions(self, store: str) -> pd.DataFrame:
        return self.frame(store, "SELECT TournamentID, YearID, TournamentSlug FROM tournament_years")

    def tournament_team_games(self, store: str) -> pd.DataFrame:
        return self.frame(store, """
            SELECT GameID, TeamID, TournamentID, YearID, TeamCode, Date, Home, Minutes, PTS, POSS, FGA, FTA, TOV, AST, REB
            FROM boxscores""")

    def has_column(self, store: str, table: str, column: str) -> bool:
        return any(row[1] == column for row in self.connection(store).execute(f'PRAGMA table_info("{table}")'))

    def tournament_player_games(self, store: str) -> pd.DataFrame:
        # events.db carries Starter; national.db gains it once the upstream starter/FIC backfill lands.
        starter = ", Starter" if self.has_column(store, "player_boxscores", "Starter") else ", NULL AS Starter"
        return self.frame(store, f"""
            SELECT GameID, TeamID, PlayerID, TournamentID, YearID, TeamCode, Date, Home, Pos, Min,
                   FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, AST, PF, STL, TOV, BLK, PTS{starter}
            FROM player_boxscores""")
