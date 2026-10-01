"""Distributional aggregated-feature baseline (milestone 0.3 acceptance comparator).

Same outcome heads and likelihood as the tower, but no game encoder: each
channel is summarized by the masked mean and standard deviation of its
standardized fields, the field-coverage shares, the mean clocks, and the
evidence features; a 64→32 MLP per channel replaces encoder + pooling, and
the same translator and heads follow. The comparison asks whether learning
a game representation adds anything beyond hand-aggregated season lines.
"""
from __future__ import annotations

import torch
from torch import nn

from . import features as F
from .distributions import SeasonParameters
from .tower import D, EMBED, OutcomeHeads, Translator, _mlp


class AggregateChannel(nn.Module):
    """Masked moments of the fields, mean clocks, mean category embeddings, evidence → 32."""

    def __init__(self, n_fields: int, vocab_sizes: list[int], n_clocks: int = len(F.CLOCK_NAMES)):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n + 1, EMBED) for n in vocab_sizes])
        self.net = _mlp(3 * n_fields + n_clocks + EMBED * len(vocab_sizes) + len(F.EVIDENCE_NAMES) + 1)
        self.empty = nn.Parameter(torch.zeros(D))

    def forward(self, ch: dict) -> torch.Tensor:
        valid = ch["valid"].float().unsqueeze(-1)                       # [B, T, 1]
        n_valid = valid.sum(dim=1).clamp(min=1.0)
        m = ch["mask"].float() * valid                                 # field observed and game valid
        denom = m.sum(dim=1).clamp(min=1.0)
        mean = (ch["x"] * m).sum(dim=1) / denom
        var = ((ch["x"] - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
        share = m.sum(dim=1) / n_valid
        clocks = (ch["clocks"] * valid).sum(dim=1) / n_valid
        # Same competition/context information the tower embeds, averaged over the valid games.
        cats = [(emb(ch["cats"][..., j]) * valid).sum(dim=1) / n_valid for j, emb in enumerate(self.embeddings)]
        present = ch["present"].float().unsqueeze(-1)
        h = self.net(torch.cat([mean, var.sqrt(), share, clocks, *cats, ch["evidence"], present], dim=-1))
        return torch.where(present.bool(), h, self.empty.expand(h.shape[0], D))


class AggregateBaseline(nn.Module):
    def __init__(self, vocab_sizes: list[int], static_vocab_sizes: list[int], reference: SeasonParameters | None = None,
                 n_fields: int | None = None, n_static: int | None = None):
        super().__init__()
        n_fields = len(F.FIELD_NAMES) if n_fields is None else int(n_fields)     # schema width (D-065)
        self.intl = AggregateChannel(n_fields, vocab_sizes)
        self.ncaa = AggregateChannel(n_fields, vocab_sizes)
        self.translator = Translator(static_vocab_sizes, n_static)
        self.heads = OutcomeHeads(reference)

    def represent(self, batch: dict) -> torch.Tensor:
        return self.translator(self.intl(batch["intl"]), self.ncaa(batch["ncaa"]), batch)

    def forward(self, batch: dict) -> dict:
        return self.heads(self.represent(batch))
