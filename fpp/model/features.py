"""Feature construction from the assembled tables (paper App. C.1).

Per game: transformed continuous fields with field masks
($\\tilde x_f = \\mathsf m_f\\{T_f(x_f)-\\mu_f\\}/(\\sigma_f+\\epsilon)$, log
transforms for skewed nonnegative counts, none for signed ratings, attempts
kept next to shooting proportions), categorical descriptors as integer
codes for 8-dimensional embeddings with an unknown entry, and the clocks
$[\\tau_g, \\operatorname{sgn}(\\tau_g)\\log(1+|\\tau_g|), \\log(1+r_g),
\\log(1+\\delta_g), \\text{season boundary}, \\text{first game}]$ computed
before any permutation. Pooling weights are $w_g = n_g + 1$.

Per example: the international channel (a forward history or a
reconstruction window), the prior-NCAA channel, evidence features
$\\mathcal E_i$ per channel, the unit's static inputs $Z_{it}$ and
$C_{jt}^{(c_t)}$, the time basis $v_{\\rm time}$ of eq. timebasis, and the
targets $(G, J, M, \\mathbf B, O, S)$.

Strength fields respect the cutoff: a rating whose availability date is
after the cutoff is replaced by the prior season's when that is available
and masked otherwise, so a reconstruction window clipped mid-season never
sees a season-final rating.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.records import BOX_FIELDS

DAY = np.timedelta64(1, "D")
YEAR_DAYS = 365.2425
EPS = 1e-6
MAX_GAMES = 256           # most recent games kept per channel; truncation is recorded in the evidence
HALF_LIFE_YEARS = 0.5     # reference recency half-life (App. C.1); a learned value is challenger (g)
TIME_BASIS_CENTERS = (0.25, 1.0, 2.0, 4.0)   # years, on log1p scale in the basis
TIME_BASIS_WIDTH = 0.6

# (name, source column, transform). Ratios are built from counts; masks come from NaN.
CONTINUOUS = [
    ("minutes", "minutes", "log1p"), ("regulation", "regulation_minutes", "identity"),
    *[(f, f, "log1p") for f in BOX_FIELDS], ("pts", "pts", "log1p"),
    ("p2", None, "ratio"), ("p3", None, "ratio"), ("pft", None, "ratio"),
    ("home", "home", "identity"), ("starter", "starter", "identity"),
    ("team_pts", "team_pts", "log1p"), ("opp_pts", "opp_pts", "log1p"), ("margin", None, "identity"),
    ("opp_adj_o", None, "identity"), ("opp_adj_d", None, "identity"), ("opp_adj_pace", None, "identity"),
    ("team_adj_o", None, "identity"), ("team_adj_d", None, "identity"), ("team_adj_pace", None, "identity"),
    ("opp_hist_winpct", None, "identity"), ("opp_hist_placement", None, "log1p"),
    ("team_hist_winpct", None, "identity"), ("team_hist_placement", None, "log1p"),
    ("opp_record_winpct", None, "identity"), ("opp_record_margin", None, "identity"),
    ("overtime", "overtime_periods", "identity"), ("coverage", "coverage_certified", "identity"),
    ("age_at_game", None, "identity"),
]
FIELD_NAMES = [name for name, _, _ in CONTINUOUS]
CATEGORICAL = ["competition", "competition_kind", "country", "age_group", "division", "source"]
CLOCK_NAMES = ["tau", "tau_log", "recency_log", "gap_log", "season_boundary", "first_game"]
STATIC_CONTINUOUS = ["age_at_cutoff", "ncaa_seasons_completed", "first_year", "height_cm", "weight_kg",
                     "recruit_rank_log", "recruit_composite", "recruit_stars",
                     "dest_prior_adj_o", "dest_prior_adj_d", "dest_prior_adj_pace", "dest_prior_games_log"]
STATIC_CATEGORICAL = ["position", "conference", "role", "class"]                    # feature schema v7
STATIC_CATEGORICAL_V8 = STATIC_CATEGORICAL + ["recruit_status"]                     # v8 (D-049): why a rank is missing
EVIDENCE_NAMES = ["n_games_log", "minutes_log", "coverage_share", "span_years", "gap_to_target_years",
                  "n_seasons", "starter_known_share", "strength_known_share", "age_known_share", "truncated"]
TARGET_COUNTS = list(BOX_FIELDS)   # A2 K2 A3 K3 Af Kf ORB DRB AST STL BLK TOV PF — the COUNT_NAMES order


# --------------------------------------------------------------------------- #
# Feature schemas (D-065): v7 = the registered fields, v8 = v7 + recruiting status,
# v9 = v8 + optional blocks from the sibling-model inventory (docs/SIBLING_FIELDS_2026_09_30.md)
# --------------------------------------------------------------------------- #

# (name, unit column, transform) in STATIC_CONTINUOUS order; the base static inputs of every schema.
BASE_STATIC_CONTINUOUS = [
    ("age_at_cutoff", "age_at_cutoff", "identity"), ("ncaa_seasons_completed", "ncaa_seasons_completed", "identity"),
    ("first_year", "first_year", "identity"), ("height_cm", "height_cm", "identity"), ("weight_kg", "weight_kg", "identity"),
    ("recruit_rank_log", "recruit_national_rank", "log1p"), ("recruit_composite", "recruit_composite_score", "identity"),
    ("recruit_stars", "recruit_star_rating", "identity"),
    ("dest_prior_adj_o", "dest_prior_adj_o", "identity"), ("dest_prior_adj_d", "dest_prior_adj_d", "identity"),
    ("dest_prior_adj_pace", "dest_prior_adj_pace", "identity"), ("dest_prior_games_log", "dest_prior_games", "log1p"),
]
assert [n for n, _, _ in BASE_STATIC_CONTINUOUS] == STATIC_CONTINUOUS

BLOCKS = ("prior_league", "usage", "destination", "intl_rapm", "on3")
# Per-game fields of the optional blocks (tables v11 and later carry the columns; missing columns read as masked).
BLOCK_GAME_FIELDS = {
    "usage": [("usage_game", "usage_game", "identity"), ("pts_share", "pts_share", "identity"), ("fga_share", "fga_share", "identity"),
              ("ast_share", "ast_share", "identity"), ("reb_share", "reb_share", "identity"), ("tov_share", "tov_share", "identity"),
              ("team_poss", "team_poss", "log1p"), ("min_rank_on_team", "min_rank_on_team", "log1p"),
              ("n_team_contributors", "n_team_contributors", "log1p")],
    "intl_rapm": [("rapm_orapm", "rapm_orapm", "identity"), ("rapm_drapm", "rapm_drapm", "identity"),
                  ("rapm_off_equiv", "rapm_off_equiv", "log1p"), ("rapm_snapshot_age", "rapm_snapshot_age_days", "log1p")],
}
# Static unit fields of the optional blocks.
BLOCK_STATIC_FIELDS = {
    "prior_league": [("prior_league_known", "prior_league_known", "identity")],
    "destination": [("dest_ret_min_share", "dest_ret_min_share", "identity"), ("dest_ret_pts_share", "dest_ret_pts_share", "identity"),
                    ("dest_ret_reb_share", "dest_ret_reb_share", "identity"), ("dest_ret_starts_share", "dest_ret_starts_share", "identity"),
                    ("dest_ret_players_log", "dest_ret_players", "log1p"), ("dest_ret_rapm_mean", "dest_ret_rapm_mean", "identity"),
                    ("dest_ret_orapm_mean", "dest_ret_orapm_mean", "identity"), ("dest_ret_drapm_mean", "dest_ret_drapm_mean", "identity"),
                    ("dest_ret_rapm_n_log", "dest_ret_rapm_n", "log1p"), ("dest_ret_measurable", "dest_ret_measurable", "identity"),
                    ("dest_fg3_rate", "dest_fg3_rate", "identity"), ("dest_fta_rate", "dest_fta_rate", "identity"),
                    ("dest_to_pct", "dest_to_pct", "identity"), ("dest_oreb_pct", "dest_oreb_pct", "identity"),
                    ("dest_dreb_pct", "dest_dreb_pct", "identity")],
    "intl_rapm": [("intl_rapm_orapm", "intl_rapm_orapm", "identity"), ("intl_rapm_drapm", "intl_rapm_drapm", "identity"),
                  ("intl_rapm_poss_log", "intl_rapm_poss", "log1p"), ("intl_rapm_season_gap", "intl_rapm_season_gap", "identity")],
    "on3": [("recruit_rank_from_on3", "recruit_rank_from_on3", "identity")],
}
BLOCK_STATIC_CATEGORICAL = {"prior_league": ["prior_league"]}


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    blocks: tuple
    continuous: tuple            # (name, game column or None, transform), CONTINUOUS first
    static_continuous: tuple     # (name, unit column, transform), BASE_STATIC_CONTINUOUS first
    static_categorical: tuple

    @property
    def field_names(self) -> list[str]:
        return [n for n, _, _ in self.continuous]

    @property
    def static_names(self) -> list[str]:
        return [n for n, _, _ in self.static_continuous]

    def describe(self) -> dict:
        return {"name": self.name, "blocks": list(self.blocks), "game_fields": len(self.continuous),
                "static_continuous": len(self.static_continuous), "static_categorical": list(self.static_categorical)}


def feature_schema(name: str = "v7", blocks=None) -> FeatureSchema:
    """``v7`` and ``v8`` are fixed; ``v9`` takes the optional blocks (default: all of ``BLOCKS``), kept in canonical order."""
    if name in ("v7", "v8"):
        if blocks:
            raise ValueError(f"Feature schema {name} takes no blocks")
        cats = STATIC_CATEGORICAL if name == "v7" else STATIC_CATEGORICAL_V8
        chosen = ()
    elif name == "v9":
        wanted = tuple(BLOCKS) if blocks is None else tuple(blocks)
        unknown = sorted(set(wanted) - set(BLOCKS))
        if unknown:
            raise ValueError(f"Unknown feature blocks {unknown}; known: {BLOCKS}")
        chosen = tuple(b for b in BLOCKS if b in wanted)
        cats = STATIC_CATEGORICAL_V8 + [c for b in chosen for c in BLOCK_STATIC_CATEGORICAL.get(b, [])]
    else:
        raise ValueError(f"Unknown feature schema {name!r}; known: v7, v8, v9")
    continuous = tuple(CONTINUOUS) + tuple(f for b in chosen for f in BLOCK_GAME_FIELDS.get(b, []))
    static = tuple(BASE_STATIC_CONTINUOUS) + tuple(f for b in chosen for f in BLOCK_STATIC_FIELDS.get(b, []))
    return FeatureSchema(name, chosen, continuous, static, tuple(cats))


DEFAULT_SCHEMA = feature_schema("v7")


def schema_from_config(config: dict | None, units: pd.DataFrame | None = None) -> FeatureSchema:
    """The schema a run was trained under: recorded in its config (D-065 runs) or, for earlier runs, implied by the
    tables the way the code decided then (v8 when the units carry ``recruit_status``, else v7)."""
    rec = (config or {}).get("feature_schema")
    if rec:
        return feature_schema(rec["name"], rec.get("blocks"))
    return feature_schema("v8" if units is not None and "recruit_status" in units.columns else "v7")


def resolve_schema(name: str = "auto", blocks=None, units: pd.DataFrame | None = None) -> FeatureSchema:
    """CLI resolution: ``auto`` reproduces the pre-D-065 behaviour (v8 iff the tables carry ``recruit_status``)."""
    if name == "auto":
        if blocks:
            raise ValueError("Feature blocks need --feature-schema v9")
        return schema_from_config(None, units)
    return feature_schema(name, blocks)


def _days(series) -> np.ndarray:
    return (pd.to_datetime(series).values.astype("datetime64[D]") - np.datetime64("1970-01-01", "D")) / DAY


def _transform(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    return values


# --------------------------------------------------------------------------- #
# Game store: arrays per game, indexed by player
# --------------------------------------------------------------------------- #

@dataclass
class GameStore:
    """Columnar arrays for every game row, sorted by (player, date), with a player index."""
    raw: np.ndarray                 # [N, F] float32 raw continuous fields, NaN = missing
    cats: np.ndarray                # [N, C] int32 category codes (0 = unknown)
    player_id: np.ndarray           # [N] int64
    date: np.ndarray                # [N] float64 days
    release: np.ndarray             # [N] float64 days
    season: np.ndarray              # [N] int32
    is_ncaa: np.ndarray             # [N] bool
    minutes: np.ndarray             # [N] float32 (NaN → 0 for weights)
    strength: dict                  # name -> (values [N], available_days [N]) for cutoff-dependent fields
    record_id: np.ndarray           # [N] object
    vocab: dict                     # categorical -> {value: code}
    index: dict = field(default_factory=dict)   # player_id -> (start, stop) slice into the arrays
    schema: FeatureSchema = DEFAULT_SCHEMA      # D-065: which continuous fields ``raw`` holds
    missing_fields: tuple = ()                  # schema fields whose column the tables lack (read as masked)

    @classmethod
    def from_frame(cls, games: pd.DataFrame, dobs: pd.Series | None = None, vocab: dict | None = None,
                   schema: FeatureSchema | None = None) -> "GameStore":
        schema = schema or DEFAULT_SCHEMA
        g = games.sort_values(["player_id", "date", "record_id"]).reset_index(drop=True)
        n = len(g)
        raw = np.full((n, len(schema.continuous)), np.nan, dtype=np.float32)
        counts = g[list(BOX_FIELDS)].astype("float64")
        missing = []
        for j, (name, column, kind) in enumerate(schema.continuous):
            if column is not None:
                if column not in g.columns:
                    missing.append(name)          # tables built before the block existed: the field stays masked
                    continue
                values = pd.to_numeric(g[column], errors="coerce").astype("float64").values
            elif name == "p2":
                values = np.where(counts["a2"] > 0, counts["k2"] / counts["a2"].where(counts["a2"] > 0), np.nan)
            elif name == "p3":
                values = np.where(counts["a3"] > 0, counts["k3"] / counts["a3"].where(counts["a3"] > 0), np.nan)
            elif name == "pft":
                values = np.where(counts["af"] > 0, counts["kf"] / counts["af"].where(counts["af"] > 0), np.nan)
            elif name == "margin":
                values = (g["team_pts"] - g["opp_pts"]).astype("float64").values
            elif name == "age_at_game":
                if dobs is None:
                    values = np.full(n, np.nan)
                else:
                    dob_days = _days(g["player_id"].map(dobs))
                    values = (_days(g["date"]) - dob_days) / YEAR_DAYS
            else:
                continue   # cutoff-dependent strength fields are filled per example
            raw[:, j] = _transform(np.asarray(values, dtype="float64"), kind)
        vocab = vocab or {c: {v: i + 1 for i, v in enumerate(sorted(g[c].dropna().astype(str).unique()))} for c in CATEGORICAL}
        cats = np.zeros((n, len(CATEGORICAL)), dtype=np.int32)
        for j, c in enumerate(CATEGORICAL):
            cats[:, j] = g[c].astype(str).map(vocab[c]).fillna(0).astype(np.int32).values
        strength = {}
        for name, avail in (("opp_adj_o", "opp_strength_available"), ("opp_adj_d", "opp_strength_available"),
                            ("opp_adj_pace", "opp_strength_available"), ("team_adj_o", "team_strength_available"),
                            ("team_adj_d", "team_strength_available"), ("team_adj_pace", "team_strength_available"),
                            ("opp_hist_winpct", "opp_hist_available"), ("opp_hist_placement", "opp_hist_available"),
                            ("team_hist_winpct", "team_hist_available"), ("team_hist_placement", "team_hist_available"),
                            ("opp_record_winpct", "opp_strength_available"), ("opp_record_margin", "opp_strength_available")):
            strength[name] = (pd.to_numeric(g[name], errors="coerce").astype("float32").values, _days(g[avail]))
        for name in ("opp_adj_o", "opp_adj_d", "opp_adj_pace"):
            prior = name.replace("opp_", "opp_prior_")
            strength[prior] = (pd.to_numeric(g[prior], errors="coerce").astype("float32").values, _days(g["opp_prior_available"]))
        store = cls(raw=raw, cats=cats, player_id=g["player_id"].values.astype(np.int64), date=_days(g["date"]),
                    release=_days(g["release_date"]), season=g["season"].values.astype(np.int32),
                    is_ncaa=(g["source"] == "ncaa").values, minutes=np.nan_to_num(g["minutes"].astype("float32").values),
                    strength=strength, record_id=g["record_id"].values, vocab=vocab, schema=schema, missing_fields=tuple(missing))
        starts = np.flatnonzero(np.r_[True, store.player_id[1:] != store.player_id[:-1]])
        stops = np.r_[starts[1:], n]
        store.index = {int(store.player_id[s]): (int(s), int(e)) for s, e in zip(starts, stops)}
        return store

    def rows(self, player_id: int) -> np.ndarray:
        s, e = self.index.get(int(player_id), (0, 0))
        return np.arange(s, e)

    def continuous(self, rows: np.ndarray, cutoff_days: float) -> np.ndarray:
        """Raw fields for the rows with cutoff-aware strength substitution; NaN = missing."""
        x = self.raw[rows].copy()
        col = {name: j for j, name in enumerate(self.schema.field_names)}
        for name in ("opp_adj_o", "opp_adj_d", "opp_adj_pace"):
            values, avail = self.strength[name]
            prior_values, prior_avail = self.strength[name.replace("opp_", "opp_prior_")]
            current = np.where(avail[rows] <= cutoff_days, values[rows], np.nan)
            fallback = np.where(prior_avail[rows] <= cutoff_days, prior_values[rows], np.nan)
            x[:, col[name]] = np.where(np.isnan(current), fallback, current)
        for name in ("team_adj_o", "team_adj_d", "team_adj_pace", "opp_hist_winpct", "opp_hist_placement",
                     "team_hist_winpct", "team_hist_placement", "opp_record_winpct", "opp_record_margin"):
            values, avail = self.strength[name]
            kind = dict((n, k) for n, _, k in CONTINUOUS)[name]
            x[:, col[name]] = _transform(np.where(avail[rows] <= cutoff_days, values[rows], np.nan), kind)
        return x


# --------------------------------------------------------------------------- #
# Fold-specific normalization
# --------------------------------------------------------------------------- #

@dataclass
class Normalizer:
    """Masked means and standard deviations fitted on one fold's training games."""
    mu: np.ndarray
    sigma: np.ndarray
    static_mu: np.ndarray
    static_sigma: np.ndarray

    @classmethod
    def fit(cls, store: GameStore, rows: np.ndarray, cutoff_days: float, static: np.ndarray) -> "Normalizer":
        x = store.continuous(rows, cutoff_days)
        mu = np.nanmean(x, axis=0)
        sigma = np.nanstd(x, axis=0)
        s_mu, s_sigma = np.nanmean(static, axis=0), np.nanstd(static, axis=0)
        return cls(np.nan_to_num(mu), np.nan_to_num(sigma), np.nan_to_num(s_mu), np.nan_to_num(s_sigma))

    def games(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = ~np.isnan(x)
        z = np.where(mask, (x - self.mu) / (self.sigma + EPS), 0.0)
        return z.astype(np.float32), mask

    def static(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = ~np.isnan(s)
        z = np.where(mask, (s - self.static_mu) / (self.static_sigma + EPS), 0.0)
        return z.astype(np.float32), mask


# --------------------------------------------------------------------------- #
# Static inputs per unit
# --------------------------------------------------------------------------- #

def static_continuous(units: pd.DataFrame, schema: FeatureSchema | None = None) -> np.ndarray:
    """Raw $Z_{it}$ and $C_{jt}^{(c_t)}$ fields in the schema's static order; NaN = missing.

    The realized schedule length S is a likelihood condition (labeled scenario), never an input: the encoder sees the
    destination's prior-season game count, known at the cutoff. A unit column the tables lack stays masked."""
    schema = schema or DEFAULT_SCHEMA
    out = np.full((len(units), len(schema.static_continuous)), np.nan)
    for j, (name, column, kind) in enumerate(schema.static_continuous):
        if column not in units.columns:
            continue
        values = pd.to_numeric(units[column], errors="coerce").astype("float64").to_numpy()
        out[:, j] = _transform(values, kind)
    return out


def static_categorical_columns(units: pd.DataFrame, schema: FeatureSchema | None = None) -> list[str]:
    """The static categoricals of the run's feature schema. Without an explicit schema (runs before D-065) the tables
    decide: v8 tables carry `recruit_status`, v7 tables do not, so v7 checkpoints keep their four static embeddings."""
    if schema is not None:
        return list(schema.static_categorical)
    return STATIC_CATEGORICAL_V8 if "recruit_status" in units.columns else STATIC_CATEGORICAL


def static_vocab(units: pd.DataFrame, schema: FeatureSchema | None = None) -> dict:
    cols = static_categorical_columns(units, schema)
    return {c: ({v: i + 1 for i, v in enumerate(sorted(units[c].dropna().astype(str).unique()))} if c in units.columns else {})
            for c in cols}


def player_dobs(tables: dict) -> pd.Series:
    """Date of birth per player for age at each game date (D-050): the units' bios, completed by the players table
    when the tables carry one (v8), so players without an NCAA unit still get ages on their game rows."""
    units = tables["units"]
    dobs = pd.to_datetime(units.drop_duplicates("player_id").set_index("player_id")["dob"], errors="coerce")
    if "players" in tables:
        extra = tables["players"].drop_duplicates("player_id").set_index("player_id")["dob"]
        dobs = dobs.combine_first(pd.to_datetime(extra, errors="coerce"))
    return dobs


def static_codes(units: pd.DataFrame, vocab: dict) -> np.ndarray:
    out = np.zeros((len(units), len(vocab)), dtype=np.int32)
    for j, c in enumerate(vocab):
        if c in units.columns:
            out[:, j] = units[c].astype(str).map(vocab[c]).fillna(0).astype(np.int32).values
    return out


def time_basis(delta_years: float, source_age: float, target_age: float) -> tuple[np.ndarray, np.ndarray]:
    """$v_{\\rm time}$ of eq. timebasis with Gaussian bumps on $\\log(1+|\\Delta|)$, gated by sign."""
    k = len(TIME_BASIS_CENTERS)
    v = np.zeros(3 + 2 * k, dtype=np.float32)
    m = np.zeros(3 + 2 * k, dtype=bool)
    if not np.isnan(source_age):
        v[0], m[0] = source_age, True
    if not np.isnan(target_age):
        v[1], m[1] = target_age, True
    if not np.isnan(delta_years):
        a = abs(delta_years)
        v[2], m[2] = a, True
        bumps = np.exp(-0.5 * ((np.log1p(a) - np.log1p(np.array(TIME_BASIS_CENTERS))) / TIME_BASIS_WIDTH) ** 2)
        if delta_years < 0:
            v[3:3 + k], m[3:3 + k] = bumps, True
        elif delta_years > 0:
            v[3 + k:], m[3 + k:] = bumps, True
    return v, m


# --------------------------------------------------------------------------- #
# Examples
# --------------------------------------------------------------------------- #

@dataclass
class Channel:
    x: np.ndarray          # [T, F] standardized, zero where masked
    mask: np.ndarray       # [T, F] field masks
    cats: np.ndarray       # [T, C] category codes
    clocks: np.ndarray     # [T, 6]
    weight: np.ndarray     # [T] pooling weight n_g + 1
    recency: np.ndarray    # [T] years to the channel's latest game
    season: np.ndarray     # [T] season labels (for contrasts)
    order: np.ndarray      # [T] chronological rank within the channel
    evidence: np.ndarray   # [10]
    present: bool

    @classmethod
    def empty(cls, n_fields: int) -> "Channel":
        z = np.zeros((0, n_fields), dtype=np.float32)
        return cls(z, np.zeros((0, n_fields), dtype=bool), np.zeros((0, len(CATEGORICAL)), dtype=np.int32),
                   np.zeros((0, len(CLOCK_NAMES)), dtype=np.float32), np.zeros(0, dtype=np.float32),
                   np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
                   np.zeros(len(EVIDENCE_NAMES), dtype=np.float32), False)


def build_channel(store: GameStore, rows: np.ndarray, *, cutoff_days: float, target_days: float,
                  normalizer: Normalizer) -> Channel:
    """Encode one game channel; rows are already chronological (store order)."""
    names = store.schema.field_names
    if len(rows) == 0:
        return Channel.empty(len(names))
    truncated = len(rows) > MAX_GAMES
    if truncated:
        rows = rows[-MAX_GAMES:]
    raw = store.continuous(rows, cutoff_days)
    x, mask = normalizer.games(raw)
    dates = store.date[rows]
    anchor = dates.max()
    tau = (dates - target_days) / YEAR_DAYS
    recency = (anchor - dates) / YEAR_DAYS
    gaps = np.r_[np.nan, np.diff(dates)] / YEAR_DAYS
    seasons = store.season[rows]
    boundary = np.r_[True, seasons[1:] != seasons[:-1]]
    clocks = np.stack([tau, np.sign(tau) * np.log1p(np.abs(tau)), np.log1p(recency),
                       np.log1p(np.nan_to_num(gaps)), boundary.astype(float), np.r_[1.0, np.zeros(len(rows) - 1)]], axis=1)
    minutes = store.minutes[rows]
    col = {name: j for j, name in enumerate(names)}
    span = (dates.max() - dates.min()) / YEAR_DAYS
    evidence = np.array([
        np.log1p(len(rows)), np.log1p(float(minutes.sum())), float(np.nan_to_num(raw[:, col["coverage"]]).mean()),
        span, (target_days - anchor) / YEAR_DAYS, float(len(np.unique(seasons))),
        float(mask[:, col["starter"]].mean()), float(mask[:, col["opp_adj_o"]].mean()),
        float(mask[:, col["age_at_game"]].mean()), float(truncated)], dtype=np.float32)
    return Channel(x, mask, store.cats[rows], clocks.astype(np.float32), (minutes + 1.0).astype(np.float32),
                   recency.astype(np.float32), seasons.astype(np.int32), np.arange(len(rows), dtype=np.int32), evidence, True)


@dataclass
class Example:
    unit_id: str
    direction: str                  # "-" forward, "+" reconstruction
    k: tuple
    intl: Channel
    ncaa: Channel
    static: np.ndarray
    static_mask: np.ndarray
    static_cats: np.ndarray
    time: np.ndarray
    time_mask: np.ndarray
    targets: dict
    weight: float = 1.0


def build_example(unit, *, store: GameStore, normalizer: Normalizer, static, static_mask, static_cats,
                  intl_rows: np.ndarray, ncaa_rows: np.ndarray, cutoff_days: float, direction: str,
                  k=(), weight: float = 1.0) -> Example:
    target_days = float(_days(pd.Series([unit.period_start]))[0])
    intl = build_channel(store, intl_rows, cutoff_days=cutoff_days, target_days=target_days, normalizer=normalizer)
    ncaa = build_channel(store, ncaa_rows, cutoff_days=cutoff_days, target_days=target_days, normalizer=normalizer)
    dob = unit.dob if isinstance(unit.dob, pd.Timestamp) and not pd.isna(unit.dob) else None
    target_age = float(unit.age_at_cutoff) if not pd.isna(unit.age_at_cutoff) else np.nan
    if intl.present:
        anchor = float(store.date[intl_rows[-1]])
        delta = (anchor - target_days) / YEAR_DAYS
        source_age = (anchor - float(_days(pd.Series([dob]))[0])) / YEAR_DAYS if dob is not None else np.nan
    else:
        delta, source_age = np.nan, np.nan
    t_v, t_m = time_basis(delta, source_age, target_age)
    minutes = float(unit.minutes)
    targets = {
        "games": int(unit.games_played), "starts": int(unit.starts),
        "minutes": float(np.round(minutes)),      # provider minutes are integers per game; rare fractional rows are rounded
        "counts": np.array([int(getattr(unit, c)) for c in TARGET_COUNTS], dtype=np.int64),
        "overtime": float(unit.overtime_periods), "overtime_known": bool(unit.overtime_known),
        "schedule": int(unit.scheduled_games),
    }
    return Example(unit.unit_id, direction, tuple(k), intl, ncaa, static, static_mask, static_cats, t_v, t_m, targets, weight)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #

def _pad_channel(channels: list[Channel]):
    import torch
    B = len(channels)
    T = max((len(c.weight) for c in channels), default=0)
    T = max(T, 1)
    F = max((c.x.shape[1] for c in channels), default=len(FIELD_NAMES))   # the channels' schema width (D-065)
    C, K = len(CATEGORICAL), len(CLOCK_NAMES)
    x = torch.zeros(B, T, F); m = torch.zeros(B, T, F, dtype=torch.bool); cats = torch.zeros(B, T, C, dtype=torch.long)
    clocks = torch.zeros(B, T, K); w = torch.zeros(B, T); rec = torch.zeros(B, T); valid = torch.zeros(B, T, dtype=torch.bool)
    season = torch.zeros(B, T, dtype=torch.long); order = torch.zeros(B, T, dtype=torch.long)
    evidence = torch.zeros(B, len(EVIDENCE_NAMES)); present = torch.zeros(B, dtype=torch.bool)
    for i, c in enumerate(channels):
        n = len(c.weight)
        if n:
            x[i, :n] = torch.from_numpy(c.x); m[i, :n] = torch.from_numpy(c.mask); cats[i, :n] = torch.from_numpy(c.cats.astype(np.int64))
            clocks[i, :n] = torch.from_numpy(c.clocks); w[i, :n] = torch.from_numpy(c.weight); rec[i, :n] = torch.from_numpy(c.recency)
            season[i, :n] = torch.from_numpy(c.season.astype(np.int64)); order[i, :n] = torch.from_numpy(c.order.astype(np.int64))
            valid[i, :n] = True
        evidence[i] = torch.from_numpy(c.evidence); present[i] = c.present
    return {"x": x, "mask": m, "cats": cats, "clocks": clocks, "weight": w, "recency": rec, "valid": valid,
            "season": season, "order": order, "evidence": evidence, "present": present}


def collate(examples: list[Example]) -> dict:
    import torch
    batch = {
        "intl": _pad_channel([e.intl for e in examples]),
        "ncaa": _pad_channel([e.ncaa for e in examples]),
        "static": torch.from_numpy(np.stack([e.static for e in examples])),
        "static_mask": torch.from_numpy(np.stack([e.static_mask for e in examples])),
        "static_cats": torch.from_numpy(np.stack([e.static_cats for e in examples]).astype(np.int64)),
        "time": torch.from_numpy(np.stack([e.time for e in examples])),
        "time_mask": torch.from_numpy(np.stack([e.time_mask for e in examples])),
        "direction": torch.tensor([1.0 if e.direction == "+" else -1.0 for e in examples]),
        "weight": torch.tensor([e.weight for e in examples], dtype=torch.float64),
        "games": torch.tensor([e.targets["games"] for e in examples], dtype=torch.long),
        "starts": torch.tensor([e.targets["starts"] for e in examples], dtype=torch.long),
        "minutes": torch.tensor([e.targets["minutes"] for e in examples], dtype=torch.float64),
        "counts": torch.from_numpy(np.stack([e.targets["counts"] for e in examples])),
        "overtime": torch.tensor([e.targets["overtime"] for e in examples], dtype=torch.float64),
        "overtime_known": torch.tensor([e.targets["overtime_known"] for e in examples], dtype=torch.bool),
        "schedule": torch.tensor([e.targets["schedule"] for e in examples], dtype=torch.long),
        # D-070 auxiliary target: the player's standardized later professional production (NaN/False when unknown).
        "future": torch.tensor([float(e.targets.get("future", float("nan"))) for e in examples], dtype=torch.float32),
        "future_known": torch.tensor([bool(e.targets.get("future_known", False)) for e in examples], dtype=torch.bool),
        "unit_id": [e.unit_id for e in examples],
    }
    return batch


def drop_games(example: Example, p: float, rng: np.random.Generator) -> Example:
    """Training augmentation (D-070): each game of each channel is dropped with probability ``p`` (at least one row is
    kept), the game-count and minutes evidence are recomputed and the first-game flag moves to the first kept row. The
    clocks, recency, season labels and chronological ranks of the kept rows are unchanged."""
    if p <= 0:
        return example

    def thin(ch: Channel) -> Channel:
        n = len(ch.weight)
        if n <= 1:
            return ch
        keep = rng.random(n) >= p
        if not keep.any():
            keep[rng.integers(n)] = True
        rows = np.flatnonzero(keep)
        clocks = ch.clocks[rows].copy()
        clocks[:, 5] = 0.0
        clocks[0, 5] = 1.0
        evidence = ch.evidence.copy()
        minutes = ch.weight[rows] - 1.0
        evidence[0] = np.log1p(len(rows))
        evidence[1] = np.log1p(float(np.maximum(minutes, 0.0).sum()))
        return Channel(ch.x[rows], ch.mask[rows], ch.cats[rows], clocks, ch.weight[rows], ch.recency[rows], ch.season[rows], ch.order[rows],
                       evidence.astype(np.float32), ch.present)

    return Example(example.unit_id, example.direction, example.k, thin(example.intl), thin(example.ncaa), example.static, example.static_mask,
                   example.static_cats, example.time, example.time_mask, example.targets, example.weight)
