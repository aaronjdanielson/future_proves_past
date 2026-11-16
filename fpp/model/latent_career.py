"""Compact latent-career model (D-115): one eight-dimensional state per career period, positive-time linear-Gaussian
transitions, the existing NCAA season likelihood as the NCAA observation model, production-given-exposure factors for
non-NCAA spells, and a structured Gaussian variational family (encoder potentials times the linear-Gaussian prior, computed
exactly by Kalman filtering and Rauch–Tung–Striebel smoothing).

Smoothing conditions a training career's states on all of its admissible spells; the deployable forecast uses the FILTER over
the spells released by the target's cutoff, propagated to the target period, and is trained explicitly. The approximation is
variational in the states only — nothing here is a posterior over network weights, and neither the latent axes nor the
competition effects are identified skill or league-strength measures.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as Fn

from .production import COUNT_NAMES, MAKE_ATTEMPTS, RATE_NAMES
from .torch_likelihood import beta_binomial_logpmf, negative_binomial_logpmf
from .tower import OutcomeHeads

DT = torch.float64
LATENT = 8
AGE_CENTER, AGE_SCALE = 20.0, 4.0
MAX_DT = 6.0                                    # years; longer gaps propagate as six
SPELL_TYPES = ("ncaa", "senior", "youth", "national", "event", "other")
COMPETITION_PRIOR_SD = 0.5
JITTER = 1e-8
_CI = {n: i for i, n in enumerate(COUNT_NAMES)}
RATE_INDEX = [_CI[n] for n in RATE_NAMES]
MAKE_INDEX = [_CI[m] for m in MAKE_ATTEMPTS]
ATTEMPT_INDEX = [_CI[a] for a in MAKE_ATTEMPTS.values()]


def age_basis(age: torch.Tensor) -> torch.Tensor:
    """[1, a, a², missing] with a = clip((age − 20)/4, −2, 2.5); a missing age contributes the intercept and its flag."""
    known = torch.isfinite(age)
    a = torch.where(known, ((age - AGE_CENTER) / AGE_SCALE).clamp(-2.0, 2.5), torch.zeros_like(age))
    return torch.stack([torch.ones_like(a), a, a * a, (~known).to(a.dtype)], dim=-1)


def kalman_filter(m0, P0, F, b, Q, J, h):
    """Filter a batch of padded sequences. m0 [B,d], P0 [B,d,d]; F [B,T,d,d], b [B,T,d], Q [B,T,d,d] carry the state INTO
    period t (ignored at t = 0; a padded period has F = I, b = 0, Q = 0); J, h [B,T,d] are the periods' potentials
    exp(−½ zᵀ diag(J) z + hᵀ z) (zero for a period without spells). Returns predicted and filtered moments [B,T,…] and the
    log normalizer [B] of prior × potentials."""
    B_, T, d = J.shape
    eye = torch.eye(d, dtype=DT)
    mp_l, Pp_l, mf_l, Pf_l = [], [], [], []
    log_z = torch.zeros(B_, dtype=DT)
    m, P = m0, P0
    for t in range(T):
        if t > 0:
            m = (F[:, t] @ m.unsqueeze(-1)).squeeze(-1) + b[:, t]
            P = F[:, t] @ P @ F[:, t].transpose(-1, -2) + Q[:, t]
        mp, Pp = m, P
        S = eye + Pp * J[:, t].unsqueeze(-2)                                  # I + P diag(J)
        m = torch.linalg.solve(S, (mp + (Pp @ h[:, t].unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
        P = torch.linalg.solve(S, Pp)
        P = 0.5 * (P + P.transpose(-1, -2))
        pinv_mp = torch.cholesky_solve(mp.unsqueeze(-1), torch.linalg.cholesky(Pp + JITTER * eye)).squeeze(-1)
        log_z = log_z - 0.5 * torch.linalg.slogdet(S)[1] + 0.5 * (m * (pinv_mp + h[:, t])).sum(-1) - 0.5 * (mp * pinv_mp).sum(-1)
        mp_l.append(mp); Pp_l.append(Pp); mf_l.append(m); Pf_l.append(P)
    return torch.stack(mp_l, 1), torch.stack(Pp_l, 1), torch.stack(mf_l, 1), torch.stack(Pf_l, 1), log_z


def rts_smoother(mp, Pp, mf, Pf, F):
    """Smoothed marginals [B,T,…] from the filter's moments."""
    T = mf.shape[1]
    eye = torch.eye(mf.shape[-1], dtype=DT)
    ms, Ps = [None] * T, [None] * T
    ms[-1], Ps[-1] = mf[:, -1], Pf[:, -1]
    for t in range(T - 2, -1, -1):
        G = torch.linalg.solve(Pp[:, t + 1] + JITTER * eye, F[:, t + 1] @ Pf[:, t]).transpose(-1, -2)
        ms[t] = mf[:, t] + (G @ (ms[t + 1] - mp[:, t + 1]).unsqueeze(-1)).squeeze(-1)
        P = Pf[:, t] + G @ (Ps[t + 1] - Pp[:, t + 1]) @ G.transpose(-1, -2)
        Ps[t] = 0.5 * (P + P.transpose(-1, -2))
    return torch.stack(ms, 1), torch.stack(Ps, 1)


def draw(m: torch.Tensor, P: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """m + chol(P) eps for moments [...,d] / [...,d,d] and standard normal eps [...,d] (leading draw axes broadcast)."""
    L = torch.linalg.cholesky(P + JITTER * torch.eye(P.shape[-1], dtype=DT))
    return m + (L @ eps.unsqueeze(-1)).squeeze(-1)


class LatentCareerModel(nn.Module):
    def __init__(self, *, n_features: int, n_competitions: int, n_static: int, static_vocab_sizes, n_covariates: int,
                 reference=None, latent: int = LATENT, hidden: int = 64, width: int = 32, embed: int = 8):
        super().__init__()
        self.latent = latent
        self.architecture = {"kind": "latent_career", "latent": latent, "hidden": hidden, "width": width, "embed": embed,
                             "n_features": n_features, "n_competitions": n_competitions, "n_static": n_static,
                             "static_vocab_sizes": [int(n) for n in static_vocab_sizes], "n_covariates": n_covariates,
                             "age_basis": "[1, a, a^2, missing], a = clip((age-20)/4, -2, 2.5)", "time_unit": "years (season labels)",
                             "max_dt": MAX_DT, "spell_types": list(SPELL_TYPES), "competition_prior_sd": COMPETITION_PRIOR_SD}
        # inference encoder: spell features + competition embedding -> Gaussian potential on the period's state
        self.comp_embed = nn.Embedding(n_competitions + 1, embed, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(n_features + embed, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, 2 * latent))
        # initial state and transition
        self.init_mean = nn.Linear(n_covariates, latent)
        self.init_log_sd = nn.Parameter(torch.zeros(latent))
        self.A = nn.Parameter(torch.zeros(latent, latent))
        self.B = nn.Parameter(torch.zeros(latent, 4))
        self.log_sigma = nn.Parameter(torch.full((latent,), math.log(0.3)))
        # NCAA decoder: state + permitted static/destination context -> the existing outcome heads
        self.static_embeds = nn.ModuleList([nn.Embedding(int(n) + 1, 4) for n in static_vocab_sizes])
        self.context = nn.Sequential(nn.Linear(2 * n_static + 4 * len(static_vocab_sizes), hidden), nn.GELU(), nn.Linear(hidden, width))
        self.state_map = nn.Linear(latent, width, bias=False)
        self.heads = OutcomeHeads(reference, width=width)
        # non-NCAA production given recorded exposure
        R, M = len(RATE_NAMES), len(MAKE_ATTEMPTS)
        self.emit = nn.Linear(latent, R + M)
        self.type_effect = nn.Embedding(len(SPELL_TYPES), R + M)
        self.comp_effect = nn.Embedding(n_competitions + 1, R + M, padding_idx=0)     # deviations; index 0 (unseen) stays zero
        self.log_disp = nn.Parameter(torch.zeros(R))
        self.log_conc = nn.Parameter(torch.full((M,), math.log(20.0)))
        with torch.no_grad():
            self.type_effect.weight.zero_()
            self.comp_effect.weight.zero_()
            self.emit.weight.mul_(0.1)
            self.state_map.weight.mul_(0.3)
            if reference is not None:
                prod = reference.production
                bias = [math.log(max(float(prod.rates[n]), 1e-3)) for n in RATE_NAMES]
                bias += [math.log(float(prod.makes_alpha[m]) / float(prod.makes_beta[m])) for m in MAKE_ATTEMPTS]
                self.emit.bias.copy_(torch.tensor(bias, dtype=self.emit.bias.dtype))

    # ------------------------------------------------------------------------------------------------- inference pieces
    def potentials(self, feat: torch.Tensor, comp: torch.Tensor):
        out = self.encoder(torch.cat([feat, self.comp_embed(comp)], dim=-1)).to(DT)
        d = self.latent
        return out[:, :d], Fn.softplus(out[:, d:] - 2.0).clamp(max=1e3)

    def initial(self, cov: torch.Tensor):
        m0 = self.init_mean(cov).to(DT)
        var = torch.exp(2.0 * self.init_log_sd).to(DT)
        return m0, torch.diag_embed(var.expand(len(m0), -1))

    def transitions(self, dt: torch.Tensor, age_prev: torch.Tensor):
        """dt [B,T] years into period t (0 at t = 0 and for padding), age_prev [B,T] the age at the period left behind."""
        d = self.latent
        dtc = dt.clamp(min=0.0, max=MAX_DT).to(DT)
        F = torch.eye(d, dtype=DT) + dtc[..., None, None] * self.A.to(DT)
        b = dtc[..., None] * (age_basis(age_prev.to(DT)) @ self.B.to(DT).T)
        Q = torch.diag_embed(dtc[..., None] * torch.exp(2.0 * self.log_sigma).to(DT))
        return F, b, Q

    def infer(self, seq: dict, smooth: bool):
        """seq: cov [B,C], dt [B,T], age_prev [B,T], obs_seq [n], obs_period [n], obs_feat [n,F], obs_comp [n].
        Filter: (m [B,T,d], P [B,T,d,d]) — the last period holds the target's predictive state, since padded periods carry
        the state unchanged. Smooth: smoothed moments and KL(q || prior) [B]."""
        B_, T = seq["dt"].shape
        d = self.latent
        J = torch.zeros(B_ * T, d, dtype=DT)
        h = torch.zeros(B_ * T, d, dtype=DT)
        if len(seq["obs_seq"]):
            mu, lam = self.potentials(seq["obs_feat"], seq["obs_comp"])
            flat = seq["obs_seq"] * T + seq["obs_period"]
            J = J.index_add(0, flat, lam)
            h = h.index_add(0, flat, lam * mu)
        J, h = J.view(B_, T, d), h.view(B_, T, d)
        m0, P0 = self.initial(seq["cov"])
        F, b, Q = self.transitions(seq["dt"], seq["age_prev"])
        mp, Pp, mf, Pf, log_z = kalman_filter(m0, P0, F, b, Q, J, h)
        if not smooth:
            return mf, Pf
        ms, Ps = rts_smoother(mp, Pp, mf, Pf, F)
        expected_log_potential = (-0.5 * (J * (torch.diagonal(Ps, dim1=-2, dim2=-1) + ms ** 2)).sum((-1, -2)) + (h * ms).sum((-1, -2)))
        return ms, Ps, expected_log_potential - log_z

    # ------------------------------------------------------------------------------------------------ observation models
    def decode(self, z: torch.Tensor, static: torch.Tensor, static_mask: torch.Tensor, static_cats: torch.Tensor) -> dict:
        """Head parameters of the NCAA season likelihood for states z [N,d] and the units' static/destination context."""
        mask = static_mask.to(torch.float32)
        ctx = torch.cat([static.to(torch.float32) * mask, mask] + [emb(static_cats[:, j]) for j, emb in enumerate(self.static_embeds)], dim=-1)
        return self.heads(self.context(ctx) + self.state_map(z.to(torch.float32)))

    def emission_log_prob(self, z, comp, kind, exposure, counts, has_minutes) -> torch.Tensor:
        """log p(spell production | z, recorded exposure) [n]: ten negative-binomial count families with mean
        exposure·exp(η) (skipped without recorded minutes) and three beta-binomial makes-given-attempts families."""
        R = len(RATE_NAMES)
        eta = self.emit(z.to(torch.float32)).to(DT) + self.type_effect(kind).to(DT) + self.comp_effect(comp).to(DT)
        mean = (exposure.to(DT)[:, None] * torch.exp(eta[:, :R].clamp(-8.0, 8.0))).clamp(min=1e-8)
        disp = 0.1 + torch.exp(self.log_disp).to(DT)
        lp_counts = negative_binomial_logpmf(counts[:, RATE_INDEX], mean, disp[None, :]).sum(-1)
        q = torch.sigmoid(eta[:, R:]).clamp(1e-4, 1.0 - 1e-4)
        conc = 1.0 + torch.exp(self.log_conc).to(DT)
        lp_makes = beta_binomial_logpmf(counts[:, MAKE_INDEX], counts[:, ATTEMPT_INDEX], q * conc, (1.0 - q) * conc).sum(-1)
        return torch.where(has_minutes, lp_counts, torch.zeros_like(lp_counts)) + lp_makes

    def penalty(self) -> torch.Tensor:
        """Unnormalized negative log priors: competition deviations N(0, 0.5²); a light L2 on the drift matrices."""
        return (self.comp_effect.weight ** 2).sum() / (2.0 * COMPETITION_PRIOR_SD ** 2), (self.A ** 2).sum() + (self.B ** 2).sum()

    def state_summary(self) -> dict:
        """Transition scales and decoder loadings for the latent-usage diagnostics."""
        with torch.no_grad():
            return {"sigma": torch.exp(self.log_sigma).tolist(), "init_sd": torch.exp(self.init_log_sd).tolist(),
                    "A_norm": float(self.A.norm()), "B_norm": float(self.B.norm()),
                    "state_map_column_norms": self.state_map.weight.norm(dim=0).tolist(),
                    "emit_row_norm": float(self.emit.weight.norm()), "competition_effect_rms": float(self.comp_effect.weight.pow(2).mean().sqrt())}
