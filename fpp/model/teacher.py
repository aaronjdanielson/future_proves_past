"""Future-informed teacher for distillation (D-109, D-110).

The deployed student is the reference tower, p_θ(Y | X), reading only pre-season information X. The teacher
q_φ(Y | X, V) additionally reads the player's later career V — a reconstruction window of games after the NCAA season —
through the *same* encoder, pooling and translator: u⁻ = represent(X), u⁺ = represent(V), fused into a target-season
representation u_T = fuse([u⁻, u⁺, 1]) → outcome heads. A unit without an eligible window, and every unit under the
no-future twin D⁰ (``use_future=False``), takes a learned missing-view vector in place of u⁺ with flag 0; D⁰ never reads
the V batch at all, so no future-derived clock or evidence field can reach it (its invariance to V is a test). Teachers
are training-only components; the student never has V.

The cache a teacher writes for its held-out split (``TeacherCache``) is keyed by outcome (unit_id), backward window (k and
its position), split, teacher checkpoint and cutoff; it carries u_T (width from the checkpoint), the teacher's constrained
head parameters (natural scales) and the window's registered weight ω. Several windows of one outcome stay separate rows:
averaging head parameters is not a mixture of their laws.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import features as F
from .distributions import SeasonParameters
from .tower import OutcomeHeads, ReferenceTower, _mlp

PARAM_KEYS = ("pi_g", "alpha_g", "beta_g", "alpha_j", "beta_j", "nu", "alpha_m", "beta_m", "pi_m", "rates", "dispersions", "makes_alpha", "makes_beta")
SPLIT_SALT = 20261110      # fixed, seed-independent: the same player always lands in the same split for a given K


class TeacherTower(nn.Module):
    """q(Y | X, V) on a shared ``ReferenceTower`` backbone; ``use_future=False`` is the matched no-future teacher D⁰."""

    def __init__(self, base: ReferenceTower, reference: SeasonParameters | None = None, use_future: bool = True):
        super().__init__()
        self.base = base
        a = base.architecture
        self.width = int(a["width"])
        self.use_future = bool(use_future)
        self.missing = nn.Parameter(torch.zeros(self.width))       # learned missing-view vector
        self.fuse = _mlp(2 * self.width + 1, self.width, int(a["hidden"]))
        self.heads = OutcomeHeads(reference, width=self.width, head_hidden=int(a["head_hidden"]))
        self.architecture = {**a, "teacher_use_future": self.use_future}

    def represent(self, batch_x: dict, batch_v: dict | None = None, has_future: torch.Tensor | None = None) -> torch.Tensor:
        u_minus = self.base.represent(batch_x)
        B = u_minus.shape[0]
        missing = self.missing.unsqueeze(0).expand(B, -1)
        if self.use_future and batch_v is not None:
            u_plus = self.base.represent(batch_v)
            if has_future is None:
                has_future = torch.ones(B, dtype=torch.bool, device=u_minus.device)
            u_plus = torch.where(has_future.unsqueeze(-1), u_plus, missing)
            flag = has_future.to(u_minus.dtype).unsqueeze(-1)
        else:                                                       # D⁰, or no window batch supplied: V is never read
            u_plus = missing
            flag = torch.zeros(B, 1, dtype=u_minus.dtype, device=u_minus.device)
        return self.fuse(torch.cat([u_minus, u_plus, flag], dim=-1))

    def forward(self, batch_x: dict, batch_v: dict | None = None, has_future: torch.Tensor | None = None) -> dict:
        u = self.represent(batch_x, batch_v, has_future)
        out = self.heads(u)
        out["representation"] = u
        return out


# ------------------------------------------------------------------------------------------------------ pairing & splits
@dataclass
class Pair:
    x: F.Example                 # the unit's forward example (X)
    v: F.Example | None          # one reconstruction window of the same unit (V), or None for the missing view
    weight: float                # the registered example weight (forward weight, or the window's ρ-normalized weight)
    k: tuple
    unit_id: str


def pair_examples(forward: list, recon: list) -> list[Pair]:
    """Every forward example once with the missing view, plus one pair per reconstruction window of a unit that has a
    forward example (windows of units without one are skipped: the teacher predicts a unit's season only from that unit's
    own pre-season inputs plus its later career)."""
    fwd = {e.unit_id: e for e in forward}
    pairs = [Pair(e, None, float(e.weight), (), e.unit_id) for e in forward]
    for r in recon:
        x = fwd.get(r.unit_id)
        if x is not None:
            pairs.append(Pair(x, r, float(r.weight), tuple(r.k), r.unit_id))
    return pairs


def collate_pairs(pairs: list[Pair]):
    """(batch_x, batch_v, has_future [B] bool, weights [B], unit_ids): rows without a window carry their own forward example
    in ``batch_v`` as a placeholder, which ``TeacherTower.represent`` replaces by the missing view."""
    batch_x = F.collate([p.x for p in pairs])
    batch_v = F.collate([p.v if p.v is not None else p.x for p in pairs])
    has_future = torch.tensor([p.v is not None for p in pairs], dtype=torch.bool)
    weights = torch.tensor([p.weight for p in pairs], dtype=torch.float64)
    return batch_x, batch_v, has_future, weights, [p.unit_id for p in pairs]


def player_split(player_ids, n_splits: int, salt: int = SPLIT_SALT) -> dict:
    """Deterministic, balanced assignment of players to ``n_splits`` cross-fitting splits by a fixed permutation (D-110): all
    of a player's units, windows and seasons share the split; independent of the training seed."""
    if n_splits < 2:
        raise ValueError("cross-fitting needs at least two splits")
    p = np.array(sorted(int(x) for x in set(player_ids)), dtype=np.int64)
    perm = np.random.default_rng(salt).permutation(len(p))
    return {int(p[j]): int(i % n_splits) for i, j in enumerate(perm)}


# ------------------------------------------------------------------------------------------------------------ the cache
class TeacherCache:
    """Teacher targets for the held-out split of one teacher: one row per (outcome, window)."""

    def __init__(self, rows: dict, metadata: dict):
        self.rows, self.metadata = rows, metadata

    @staticmethod
    def empty_rows() -> dict:
        return {"unit_id": [], "k": [], "window_index": [], "weight": [], "representation": [], **{k: [] for k in PARAM_KEYS}}

    @staticmethod
    def append(rows: dict, unit_ids: list, ks: list, window_index: list, weights, out: dict) -> None:
        rows["unit_id"].extend(unit_ids)
        rows["k"].extend("all" if k == () else ",".join("all" if x is None else str(x) for x in k) for k in ks)
        rows["window_index"].extend(int(i) for i in window_index)
        rows["weight"].extend(float(w) for w in weights)
        rows["representation"].append(out["representation"].detach().to(torch.float64).numpy())
        for key in PARAM_KEYS:
            rows[key].append(out[key].detach().to(torch.float64).numpy())

    @classmethod
    def write(cls, path: Path, rows: dict, metadata: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"unit_id": np.array(rows["unit_id"], dtype=str), "k": np.array(rows["k"], dtype=str),
                  "window_index": np.array(rows["window_index"], dtype=np.int64), "weight": np.array(rows["weight"], dtype=np.float64)}
        if rows["unit_id"]:
            arrays["representation"] = np.concatenate(rows["representation"])
            for key in PARAM_KEYS:
                arrays[key] = np.concatenate(rows[key])
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        meta = {**metadata, "rows": len(rows["unit_id"]), "width": int(arrays["representation"].shape[1]) if rows["unit_id"] else None,
                "parameter_keys": list(PARAM_KEYS)}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
        return path.with_suffix(".npz")

    @classmethod
    def read(cls, path: Path) -> "TeacherCache":
        path = Path(path)
        z = np.load(path.with_suffix(".npz"), allow_pickle=False)
        rows = {k: z[k] for k in z.files}
        cache = cls(rows, json.loads(path.with_suffix(".json").read_text()))
        cache.check_integrity()
        return cache

    def check_integrity(self) -> None:
        """Row count against the metadata, unique (outcome, window) keys, the recorded width, finite values everywhere."""
        n = len(self.rows["unit_id"])
        if n != int(self.metadata.get("rows", -1)):
            raise ValueError(f"teacher cache holds {n} rows, metadata says {self.metadata.get('rows')}")
        if n == 0:
            return
        keys = list(zip(self.rows["unit_id"].tolist(), self.rows["k"].tolist(), self.rows["window_index"].tolist()))
        if len(set(keys)) != n:
            raise ValueError("teacher cache has duplicate (outcome, window) keys")
        rep = self.rows["representation"]
        if rep.ndim != 2 or rep.shape != (n, int(self.metadata["width"])):
            raise ValueError(f"representation shape {rep.shape} does not match {n} rows × width {self.metadata.get('width')}")
        for key in PARAM_KEYS + ("representation", "weight"):
            a = np.asarray(self.rows[key], dtype=np.float64)
            if a.shape[0] != n or not np.isfinite(a).all():
                raise ValueError(f"teacher cache column {key!r}: wrong length or non-finite values")

    def validate(self, expected: dict) -> None:
        """Refuse a cache whose identity differs from what the student expects: every key given in ``expected`` (fold,
        training cutoff, tables digest, feature schema, split assignment, checkpoint, …) must equal the metadata's,
        compared as canonical JSON so tuples and lists agree."""
        canon = lambda v: json.dumps(v, sort_keys=True, default=str)   # noqa: E731
        bad = [k for k, v in expected.items() if canon(self.metadata.get(k)) != canon(v)]
        if bad:
            raise ValueError("teacher cache does not match the student's expectations on: " + ", ".join(bad))

    def by_unit(self) -> dict:
        """unit_id → list of row indices (its windows), in cache order."""
        out = {}
        for i, u in enumerate(self.rows["unit_id"]):
            out.setdefault(str(u), []).append(i)
        return out


def checkpoint_sha256(state_dict: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
