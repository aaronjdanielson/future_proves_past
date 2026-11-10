"""Student-side distillation from cross-fitted teachers (D-109, D-110).

Three pieces, all training-only:

* ``load_teacher_caches`` reads one ``TeacherCache`` per split, validates each against the student's fold, cutoff, tables and
  feature schema, checks the splits are distinct and complete, and indexes the rows by outcome so a forward minibatch can
  find its targets. Windows of one outcome stay separate rows with their registered weights ω (normalized within the
  outcome at load).
* ``shooting_cross_entropy`` is the exact pilot term: for each shot type h the teacher law q_h = BB(A_h; α_T, β_T) (an
  ω-mixture over the outcome's windows) and the student law p_h = BB(A_h; α_S, β_S) are finite distributions on
  0..A_h, so CE(q_h, p_h) = −Σ_k q_h(k) log p_h(k) is a finite sum; attempts A_h are the outcome's observed attempts, supplied
  to both. The makes factor leaves the quadrature exactly: in the season likelihood the makes | attempts terms do not depend
  on latent minutes, so log p(y) = log p(y without makes) + Σ_h log BB(K_h | A_h); ``student_makes_logprob`` is that second
  term and ``distilled_rows`` assembles ℓ_other + (1 − αR) ℓ_shoot + αR·CE per unit (the anchored blend of D-109 item 5).
* ``Alignments`` holds one training-only linear map per teacher split for representation matching,
  L_repr = Σ_i w_i Σ_w ω_iw ‖A_{k(i)} u_i⁻ − sg(u_T,iw)‖² / Σ_i w_i over outcomes with targets (D-110).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..model.production import COUNT_NAMES, MAKE_ATTEMPTS
from ..model.teacher import PARAM_KEYS, TeacherCache
from ..model.torch_likelihood import beta_binomial_logpmf

DTYPE = torch.float64
SHOTS = tuple(MAKE_ATTEMPTS.items())                 # (("K2", "A2"), ("K3", "A3"), ("Kf", "Af")): column order of makes_alpha / makes_beta
CI = {name: i for i, name in enumerate(COUNT_NAMES)}


@dataclass
class TeacherTargets:
    arrays: dict                 # concatenated cache columns plus "split" [N]
    index: dict                  # unit_id -> np.ndarray of row indices
    omega: np.ndarray            # [N] window weights normalized within each outcome
    width: int
    n_splits: int
    use_future: bool
    metadata: list               # one metadata record per cache, in split order

    @property
    def n_rows(self) -> int:
        return len(self.omega)

    def rows_for(self, unit_ids: list):
        """(row indices, batch positions, ω, split) for every (outcome, window) of the batch's units that has targets."""
        rows, pos = [], []
        for i, u in enumerate(unit_ids):
            r = self.index.get(str(u))
            if r is not None:
                rows.append(r)
                pos.append(np.full(len(r), i, dtype=np.int64))
        if not rows:
            return None
        rows, pos = np.concatenate(rows), np.concatenate(pos)
        return rows, pos, self.omega[rows], self.arrays["split"][rows]

    def describe(self) -> dict:
        return {"rows": int(self.n_rows), "outcomes": len(self.index), "width": self.width, "n_splits": self.n_splits, "use_future": self.use_future,
                "checkpoints": [m.get("checkpoint_sha256") for m in self.metadata], "splits": [m.get("split") for m in self.metadata],
                "teacher_lambda": [m.get("teacher_lambda") for m in self.metadata]}


def assignment_digest(players: dict) -> str:
    """Digest of a player→split mapping, the same expression the teacher trainer records (``split_assignment_sha256``)."""
    from ..model.artifact import canonical_sha256
    return canonical_sha256(sorted((int(p), int(s)) for p, s in players.items()))


def load_teacher_caches(dirs, *, fold_season: int, training_cutoff: str, tables_sha256: str | None, feature_schema: dict,
                        unit_player: dict) -> TeacherTargets:
    """Read and validate one teacher directory per split (each holds ``targets.{npz,json}`` and ``split_assignment.json``).
    The cross-fitting guarantee is enforced, not assumed: every teacher must carry its assignment file, the recomputed digest
    of the common mapping must equal the recorded one, every cached outcome must belong to a known player sitting in the
    split of the teacher that supplied it, and no outcome may come from two teachers."""
    caches, assignments = [], []
    for d in dirs:
        d = Path(d)
        cache = TeacherCache.read(d / "targets")
        cache.validate({"fold": fold_season, "training_cutoff": training_cutoff, "feature_schema": feature_schema})
        if tables_sha256 is not None and cache.metadata.get("tables_manifest", {}).get("payload_sha256") != tables_sha256:
            raise ValueError(f"{d}: teacher tables differ from the student's")
        if not (d / "split_assignment.json").exists():
            raise ValueError(f"{d}: no split_assignment.json; the teacher's player-to-split mapping is required")
        assignment = json.loads((d / "split_assignment.json").read_text())
        if assignment_digest(assignment["players"]) != cache.metadata.get("split_assignment_sha256"):
            raise ValueError(f"{d}: the split assignment file does not match the digest the teacher recorded")
        if int(assignment.get("split", -1)) != int(cache.metadata["split"]):
            raise ValueError(f"{d}: split_assignment.json names split {assignment.get('split')}, the cache split {cache.metadata['split']}")
        caches.append(cache)
        assignments.append(assignment)
    splits = [int(c.metadata["split"]) for c in caches]
    n_splits = {int(c.metadata["n_splits"]) for c in caches}
    futures = {bool(c.metadata["use_future"]) for c in caches}
    widths = {int(c.metadata["width"]) for c in caches if c.metadata.get("width") is not None}
    if len(n_splits) != 1 or len(futures) != 1 or len(widths) > 1:
        raise ValueError("teacher caches disagree on n_splits, use_future or width")
    n_splits = n_splits.pop()
    if sorted(splits) != list(range(n_splits)):
        raise ValueError(f"teacher caches cover splits {sorted(splits)}; need every split 0..{n_splits - 1} exactly once")
    digests = {c.metadata.get("split_assignment_sha256") for c in caches}
    if len(digests) != 1:
        raise ValueError("teacher caches were built from different player-to-split assignments")
    order = np.argsort(splits)
    caches = [caches[i] for i in order]
    assignments = [assignments[i] for i in order]
    arrays = {k: [] for k in PARAM_KEYS + ("representation", "weight", "unit_id", "k", "window_index")}
    split_col = []
    for c in caches:
        if c.metadata["rows"] == 0:
            continue
        for k in arrays:
            arrays[k].append(c.rows[k])
        split_col.append(np.full(c.metadata["rows"], int(c.metadata["split"]), dtype=np.int64))
    if not split_col:
        raise ValueError("teacher caches hold no targets")
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    arrays["split"] = np.concatenate(split_col)
    unit_ids = arrays["unit_id"].astype(str)
    index, omega = {}, np.zeros(len(unit_ids))
    for i, u in enumerate(unit_ids):
        index.setdefault(u, []).append(i)
    for u, r in index.items():
        r = np.asarray(r, dtype=np.int64)
        index[u] = r
        w = arrays["weight"][r].astype(float)
        omega[r] = w / w.sum() if w.sum() > 0 else 1.0 / len(r)
    # Every outcome comes from exactly one teacher, belongs to a known player, and that player sits in the supplying teacher's
    # held-out split (the teacher never saw that player's outcomes).
    supplied = {}
    for u, s in zip(unit_ids, arrays["split"]):
        if supplied.setdefault(u, int(s)) != int(s):
            raise ValueError(f"outcome {u} is supplied by teachers of splits {supplied[u]} and {int(s)}")
    players = assignments[0]["players"]
    for u, s in supplied.items():
        p = unit_player.get(str(u))
        if p is None:
            raise ValueError(f"outcome {u} in the teacher cache is not among the student's training units")
        if players.get(str(int(p))) != s:
            raise ValueError(f"outcome {u}: its player is not in split {s}, yet split {s}'s teacher supplied its targets")
    return TeacherTargets(arrays, index, omega, int(widths.pop()) if widths else int(arrays["representation"].shape[1]), n_splits, futures.pop(),
                          [c.metadata for c in caches])


# ------------------------------------------------------------------------------------------------------- shooting pilot
def _bb_grid(n: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """[R, K] log-pmf over k = 0..max(n) for per-row (n, α, β); −inf beyond each row's n."""
    K = max(int(n.max().item()) + 1, 1)
    ks = torch.arange(K, dtype=DTYPE)[None, :].expand(len(n), K)
    return beta_binomial_logpmf(ks, n.to(DTYPE)[:, None].expand(-1, K), alpha[:, None], beta[:, None])


def student_makes_logprob(params: dict, counts: torch.Tensor) -> torch.Tensor:
    """Σ_h log BB(K_h | A_h; α_S,h, β_S,h) per row: the makes factor of the season likelihood, exact."""
    total = torch.zeros(counts.shape[0], dtype=DTYPE)
    for h, (make, attempt) in enumerate(SHOTS):
        k, a = counts[:, CI[make]].to(DTYPE), counts[:, CI[attempt]].to(DTYPE)
        total = total + beta_binomial_logpmf(k, a, params["makes_alpha"][:, h].to(DTYPE), params["makes_beta"][:, h].to(DTYPE))
    return total


def shooting_cross_entropy(params: dict, counts: torch.Tensor, targets: TeacherTargets, rows: np.ndarray, pos: np.ndarray, omega: np.ndarray) -> torch.Tensor:
    """CE per batch row (zero where the unit has no targets): Σ_h Σ_w ω_w [−Σ_k q_T,hw(k | A_h) log p_S,h(k | A_h)]."""
    B = counts.shape[0]
    ce = torch.zeros(B, dtype=DTYPE)
    pos_t, om = torch.as_tensor(pos), torch.as_tensor(omega, dtype=DTYPE)
    for h, (make, attempt) in enumerate(SHOTS):
        a = counts[pos_t, CI[attempt]].long()
        if int(a.max()) == 0:
            continue
        logp = _bb_grid(a, params["makes_alpha"][pos_t, h].to(DTYPE), params["makes_beta"][pos_t, h].to(DTYPE))              # student, with gradient
        with torch.no_grad():
            logq = _bb_grid(a, torch.as_tensor(targets.arrays["makes_alpha"][rows, h], dtype=DTYPE),
                            torch.as_tensor(targets.arrays["makes_beta"][rows, h], dtype=DTYPE))                              # teacher window
            q = torch.exp(logq)
            q = q / q.sum(1, keepdim=True).clamp(min=1e-300)
        support = torch.isfinite(logp)
        term = -(q * torch.where(support, logp, torch.zeros_like(logp))).sum(1)                                              # q is 0 beyond A
        ce = ce.index_add(0, pos_t, om * term)
    return ce


def distilled_rows(lp_full: torch.Tensor, lp_makes: torch.Tensor, ce: torch.Tensor, has_targets: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-row loss ℓ_other + (1 − αR) ℓ_shoot + αR·CE; with α = 0 or no targets it is the plain negative log-likelihood."""
    r = has_targets.to(DTYPE) * float(alpha)
    other, shoot = -(lp_full - lp_makes), -lp_makes
    return other + (1.0 - r) * shoot + r * ce


# -------------------------------------------------------------------------------------------- representation matching
class Alignments(nn.Module):
    """One training-only linear map per teacher split (D-110 amendment 1): independently fitted teachers do not share hidden
    coordinates, so each split's targets are matched through its own A_k."""

    def __init__(self, n_splits: int, width_student: int, width_teacher: int):
        super().__init__()
        self.maps = nn.ModuleDict({str(k): nn.Linear(width_student, width_teacher).double() for k in range(n_splits)})
        for m in self.maps.values():                      # start at the identity (or zero when widths differ), in the likelihood's float64
            with torch.no_grad():
                m.weight.zero_()
                if width_student == width_teacher:
                    m.weight.copy_(torch.eye(width_student, dtype=DTYPE))
                m.bias.zero_()

    def forward(self, u: torch.Tensor, split: np.ndarray) -> torch.Tensor:
        out = torch.zeros(u.shape[0], self.maps["0"].out_features, dtype=u.dtype)
        for k in np.unique(split):
            sel = torch.as_tensor(split == k)
            out[sel] = self.maps[str(int(k))](u[sel])
        return out


def representation_loss(u: torch.Tensor, align: Alignments, targets: TeacherTargets, rows: np.ndarray, pos: np.ndarray, omega: np.ndarray,
                        split: np.ndarray, weights: torch.Tensor) -> torch.Tensor:
    """Σ_i w_i Σ_w ω_iw ‖A_k u_i − sg(u_T,iw)‖² / Σ_i w_i over the batch's outcomes with targets (teacher side detached)."""
    pos_t = torch.as_tensor(pos)
    a = align(u[pos_t].to(DTYPE), split)
    target = torch.as_tensor(targets.arrays["representation"][rows], dtype=DTYPE)
    per_row = ((a - target) ** 2).sum(1) * torch.as_tensor(omega, dtype=DTYPE)
    per_unit = torch.zeros(u.shape[0], dtype=DTYPE).index_add(0, pos_t, per_row)
    has = torch.zeros(u.shape[0], dtype=torch.bool)
    has[pos_t] = True
    w = weights.to(DTYPE) * has.to(DTYPE)
    return (w * per_unit).sum() / w.sum().clamp(min=1e-12)
