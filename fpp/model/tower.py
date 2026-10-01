"""The reference game tower (paper §3.2–3.4, App. C.1) and its outcome heads.

Games share one 64→32 encoder over transformed fields, field masks, clocks,
and 8-dimensional category embeddings. A channel is summarized without
learned attention: an exposure-weighted mean $h^L$ (weights $n_g+1$), a
recency-weighted mean $h^R$ (half-life 0.5 years), their difference, a
projected within-/between-season contrast, the evidence features, and
presence flags, fused by a 64→32 MLP with LayerNorm. An empty channel
returns a learned vector and zero evidence. The international channel and
the prior-NCAA channel use the same encoder and separate fusions. The
translator (64→32) maps both channel vectors, the static inputs
$Z_{it}, C_{jt}^{(c_t)}$, and the time basis $v_{\\rm time}$ to $u$; the
heads map $u$ to the season-distribution parameters with logistic links
for probabilities and softplus links for positive parameters, initialized
at the intercept-only reference so an untrained tower reproduces it.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as Fn

from .distributions import SeasonParameters
from .production import MAKE_ATTEMPTS, RATE_NAMES
from . import features as F

D = 32
HIDDEN = 64
EMBED = 8
DTYPE64 = torch.float64


def _mlp(inp: int, out: int = D, hidden: int = HIDDEN) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, out), nn.LayerNorm(out))


class GameEncoder(nn.Module):
    def __init__(self, n_fields: int, vocab_sizes: list[int], n_clocks: int = len(F.CLOCK_NAMES)):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n + 1, EMBED) for n in vocab_sizes])
        self.net = _mlp(2 * n_fields + n_clocks + EMBED * len(vocab_sizes))

    def forward(self, ch: dict) -> torch.Tensor:
        parts = [ch["x"], ch["mask"].float(), ch["clocks"]]
        parts += [emb(ch["cats"][..., j]) for j, emb in enumerate(self.embeddings)]
        return self.net(torch.cat(parts, dim=-1))          # [B, T, D]


def masked_mean(e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean over T with weights already zero on invalid games; zero vector when no weight."""
    total = w.sum(dim=1, keepdim=True)
    return (e * w.unsqueeze(-1)).sum(dim=1) / total.clamp(min=1e-12)


class ChannelPool(nn.Module):
    """Fixed-weight summaries, contrasts, evidence, and fusion for one channel."""

    def __init__(self, half_life_years: float = F.HALF_LIFE_YEARS):
        super().__init__()
        self.half_life = half_life_years
        self.season_projection = nn.Linear(2 * D + 2, D)
        n_in = 4 * D + D + 1 + len(F.EVIDENCE_NAMES) + 3   # hL, hR, dRL, dseason, z_sum, flag, evidence, m_hist
        self.fusion = _mlp(n_in)
        self.empty = nn.Parameter(torch.zeros(D))

    def forward(self, e: torch.Tensor, ch: dict) -> torch.Tensor:
        valid = ch["valid"]                                           # [B, T]
        w = ch["weight"] * valid
        h_l = masked_mean(e, w)
        w_r = w * torch.pow(2.0, -ch["recency"] / self.half_life)
        h_r = masked_mean(e, w_r)
        d_rl = h_r - h_l
        d_season, flags = self.contrasts(e, ch, w)
        B = e.shape[0]
        z_sum = torch.zeros(B, D, device=e.device)                     # summary channel: not yet populated
        z_flag = torch.zeros(B, 1, device=e.device)
        present = ch["present"].float().unsqueeze(-1)
        m_hist = torch.cat([present, flags], dim=-1)
        fused = self.fusion(torch.cat([h_l, h_r, d_rl, d_season, z_sum, z_flag, ch["evidence"], m_hist], dim=-1))
        return torch.where(present.bool(), fused, self.empty.expand(B, D))

    def contrasts(self, e, ch, w):
        valid, season, order = ch["valid"], ch["season"], ch["order"]
        neg = torch.full_like(season, -1)
        latest = torch.where(valid, season, neg).max(dim=1).values          # [B]
        in_latest = valid & (season == latest.unsqueeze(1))
        # Rank from the end among latest-season games by stored chronological order, not tensor position,
        # so a permuted batch of complete records gives the same contrast.
        later = in_latest.unsqueeze(1) & (order.unsqueeze(1) > order.unsqueeze(2))   # [B, i, j]: j later than i
        r = 1 + later.sum(dim=2)                                                     # 1 = latest game
        n_latest = in_latest.sum(dim=1)
        last5 = in_latest & (r <= 5)
        prev5 = in_latest & (r > 5) & (r <= 10)
        has_within = (n_latest >= 10).float().unsqueeze(-1)
        within = (masked_mean(e, last5.float()) - masked_mean(e, prev5.float())) * has_within
        prev_season = torch.where(valid & (season < latest.unsqueeze(1)), season, neg).max(dim=1).values
        in_prev = valid & (season == prev_season.unsqueeze(1))
        has_between = ((prev_season >= 0) & (n_latest > 0)).float().unsqueeze(-1)
        between = (masked_mean(e, w * in_latest) - masked_mean(e, w * in_prev)) * has_between
        flags = torch.cat([has_within, has_between], dim=-1)
        return self.season_projection(torch.cat([within, between, flags], dim=-1)), flags


class Translator(nn.Module):
    def __init__(self, static_vocab_sizes: list[int], n_static: int | None = None):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n + 1, EMBED) for n in static_vocab_sizes])
        self.n_static_continuous = len(F.STATIC_CONTINUOUS) if n_static is None else int(n_static)   # schema width (D-065)
        n_static = 2 * self.n_static_continuous + EMBED * len(static_vocab_sizes)
        n_time = 2 * (3 + 2 * len(F.TIME_BASIS_CENTERS))
        self.net = _mlp(2 * D + n_static + n_time)

    def forward(self, h_intl, h_ncaa, batch) -> torch.Tensor:
        parts = [h_intl, h_ncaa, batch["static"], batch["static_mask"].float()]
        parts += [emb(batch["static_cats"][:, j]) for j, emb in enumerate(self.embeddings)]
        parts += [batch["time"], batch["time_mask"].float()]
        return self.net(torch.cat(parts, dim=-1))


def _inverse_softplus(v: float) -> float:
    return math.log(math.expm1(v)) if v < 20 else v


# Registered parameter domain (D-026). Positive parameters are floor + softplus(raw); the floors keep
# every head inside the region where the fixed-node quadrature meets its tolerance (tested against
# the SciPy reference on a shape grid down to these values) and away from degenerate shapes.
PARAMETER_FLOORS = {"alpha_g": 0.05, "beta_g": 0.05, "alpha_j": 0.05, "beta_j": 0.05, "nu": 1e-4,
                    "alpha_m": 0.25, "beta_m": 0.25, "rates": 1e-3, "dispersions": 0.1,
                    "makes_alpha": 0.1, "makes_beta": 0.1}
PROBABILITY_BOUNDS = (1e-4, 1 - 1e-4)


def _inverse_softplus_above(value: float, floor: float) -> float:
    return _inverse_softplus(max(value - floor, 1e-6))


def _logit_within(p: float) -> float:
    """Raw value whose bounded logistic link returns p exactly."""
    lo, hi = PROBABILITY_BOUNDS
    q = min(max((p - lo) / (hi - lo), 1e-9), 1 - 1e-9)
    return _logit(q)


class OutcomeHeads(nn.Module):
    """u → season-distribution parameters, initialized at the reference values.

    This reference family conditions every head on the player representation
    only; the paper also allows dependence on preceding outcomes (games,
    earlier counts), which is a later head extension.
    """
    ORDER = ["pi_g", "alpha_g", "beta_g", "alpha_j", "beta_j", "nu", "alpha_m", "beta_m", "pi_m"]

    def __init__(self, reference: SeasonParameters | None = None, init_scale: float = 0.05):
        super().__init__()
        p = reference or SeasonParameters()
        prod = p.production
        fl = PARAMETER_FLOORS
        self.n_out = 9 + 2 * len(RATE_NAMES) + 2 * len(MAKE_ATTEMPTS)
        self.linear = nn.Linear(D, self.n_out)
        with torch.no_grad():
            self.linear.weight.mul_(init_scale)
            bias = [_logit_within(p.participation_probability), _inverse_softplus_above(p.games_alpha, fl["alpha_g"]),
                    _inverse_softplus_above(p.games_beta, fl["beta_g"]), _inverse_softplus_above(p.starts_alpha, fl["alpha_j"]),
                    _inverse_softplus_above(p.starts_beta, fl["beta_j"]), _inverse_softplus_above(p.overtime_rate_per_game, fl["nu"]),
                    _inverse_softplus_above(p.minutes_alpha, fl["alpha_m"]), _inverse_softplus_above(p.minutes_beta, fl["beta_m"]),
                    _logit_within(p.minutes_max_probability)]
            bias += [_inverse_softplus_above(prod.rates[n], fl["rates"]) for n in RATE_NAMES]
            bias += [_inverse_softplus_above(prod.dispersions[n], fl["dispersions"]) for n in RATE_NAMES]
            bias += [_inverse_softplus_above(prod.makes_alpha[n], fl["makes_alpha"]) for n in MAKE_ATTEMPTS]
            bias += [_inverse_softplus_above(prod.makes_beta[n], fl["makes_beta"]) for n in MAKE_ATTEMPTS]
            self.linear.bias.copy_(torch.tensor(bias, dtype=self.linear.bias.dtype))
        self.regulation_minutes = float(p.regulation_minutes)
        self.overtime_minutes = float(p.overtime_minutes)
        self.foul_limit_per_game = int(prod.foul_limit_per_game)

    def forward(self, u: torch.Tensor) -> dict:
        raw = self.linear(u).to(DTYPE64)
        k, r, m = 9, len(RATE_NAMES), len(MAKE_ATTEMPTS)
        fl = PARAMETER_FLOORS
        lo, hi = PROBABILITY_BOUNDS

        def prob(x):
            return lo + (hi - lo) * torch.sigmoid(x)

        def pos(x, name):
            return fl[name] + Fn.softplus(x)

        out = {"pi_g": prob(raw[:, 0]), "alpha_g": pos(raw[:, 1], "alpha_g"), "beta_g": pos(raw[:, 2], "beta_g"),
               "alpha_j": pos(raw[:, 3], "alpha_j"), "beta_j": pos(raw[:, 4], "beta_j"), "nu": pos(raw[:, 5], "nu"),
               "alpha_m": pos(raw[:, 6], "alpha_m"), "beta_m": pos(raw[:, 7], "beta_m"), "pi_m": prob(raw[:, 8]),
               "rates": pos(raw[:, k:k + r], "rates"), "dispersions": pos(raw[:, k + r:k + 2 * r], "dispersions"),
               "makes_alpha": pos(raw[:, k + 2 * r:k + 2 * r + m], "makes_alpha"),
               "makes_beta": pos(raw[:, k + 2 * r + m:], "makes_beta")}
        return out


def _logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


class PretrainHead(nn.Module):
    """Next-game head for the pool objective (D-032): the shared representation plus the next
    game's context (its competition and venue, and the strengths available at the prediction
    date) → per-game season-distribution parameters through its own OutcomeHeads, so the
    forward heads never see pool targets."""

    def __init__(self, encoder: GameEncoder, n_context: int, reference: SeasonParameters | None = None):
        super().__init__()
        self.encoder_embeddings = encoder.embeddings          # shared category embeddings
        self.net = _mlp(D + 2 * n_context + EMBED * len(encoder.embeddings))
        self.heads = OutcomeHeads(reference)

    def forward(self, u: torch.Tensor, batch: dict) -> dict:
        parts = [u, batch["context"], batch["context_mask"].float()]
        parts += [emb(batch["context_cats"][:, j]) for j, emb in enumerate(self.encoder_embeddings)]
        return self.heads(self.net(torch.cat(parts, dim=-1)))


class ReferenceTower(nn.Module):
    """Encoder → per-channel pooling/fusion → translator → heads (+ the pool's next-game head)."""

    def __init__(self, vocab_sizes: list[int], static_vocab_sizes: list[int], reference: SeasonParameters | None = None,
                 n_context: int = 0, n_fields: int | None = None, n_static: int | None = None):
        super().__init__()
        # Input widths follow the run's feature schema (D-065); the defaults are the registered v7 widths.
        self.encoder = GameEncoder(len(F.FIELD_NAMES) if n_fields is None else int(n_fields), vocab_sizes)
        self.pool_intl = ChannelPool()
        self.pool_ncaa = ChannelPool()
        self.translator = Translator(static_vocab_sizes, n_static)
        self.heads = OutcomeHeads(reference)
        self.pretrain_head = PretrainHead(self.encoder, n_context, reference) if n_context else None

    def represent(self, batch: dict) -> torch.Tensor:
        h_intl = self.pool_intl(self.encoder(batch["intl"]), batch["intl"])
        h_ncaa = self.pool_ncaa(self.encoder(batch["ncaa"]), batch["ncaa"])
        return self.translator(h_intl, h_ncaa, batch)

    def forward(self, batch: dict) -> dict:
        return self.heads(self.represent(batch))

    def pretrain_forward(self, batch: dict) -> dict:
        """Pool objective: the same encoder, pooling, and translator (NCAA channel empty, static and
        time inputs hidden by their masks), then the next-game head."""
        if self.pretrain_head is None:
            raise ValueError("The tower was built without a pretraining head")
        return self.pretrain_head(self.represent(batch), batch)
