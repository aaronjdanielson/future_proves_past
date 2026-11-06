"""Identity of fitted objects (D-095): content digests, the pretraining-pool contract, and the fitted-model artifact.

A fitted model is more than its weights: the feature schema and order, the category vocabularies, the normalizer, the
row policy, the season roles and the tables it was fitted on are part of it. ``model_artifact.pt`` carries all of them,
so a forecast or report restores the model as trained instead of refitting its preprocessing on whatever tables are
supplied, and a changed table, vocabulary or normalizer is refused before anything is predicted.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import features as F

ARTIFACT_FORMAT = "fpp-fitted-model-v1"


def canonical_sha256(obj) -> str:
    """SHA-256 of bytes, or of canonical JSON (sorted keys) for anything else."""
    data = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()


def normalizer_sha256(n: F.Normalizer) -> str:
    return canonical_sha256(b"".join(np.ascontiguousarray(a, dtype=np.float64).tobytes()
                                     for a in (n.mu, n.sigma, n.static_mu, n.static_sigma)))


def code_sha256(*objects) -> dict:
    """Digests of the source of modules or functions an artifact depends on."""
    return {getattr(o, "__qualname__", getattr(o, "__name__", str(o))): canonical_sha256(inspect.getsource(o).encode())
            for o in objects}


def pool_contract(*, tables_sha256: str | None, schema: dict, store: F.GameStore, normalizer: F.Normalizer, heldout,
                  excluded_rows, architecture: dict, vocab_sizes, static_sizes, n_context: int, settings: dict,
                  code: dict) -> dict:
    """Everything a pretraining-pool checkpoint depends on (D-095). Two runs may share a pool only when their contracts
    are equal; the cache key is the contract's digest, so a stale or foreign checkpoint is never loaded."""
    return {"tables_payload_sha256": tables_sha256, "feature_schema": schema, "vocab_sha256": canonical_sha256(store.vocab),
            "row_guard": bool(store.row_guard), "normalizer_sha256": normalizer_sha256(normalizer),
            "heldout_players": len(heldout), "heldout_players_sha256": canonical_sha256(sorted(int(p) for p in heldout)),
            "excluded_rows_sha256": None if excluded_rows is None else canonical_sha256(np.packbits(np.asarray(excluded_rows, bool)).tobytes()),
            "architecture": dict(architecture), "vocab_sizes": [int(v) for v in vocab_sizes],
            "static_sizes": [int(v) for v in static_sizes], "n_context": int(n_context), "settings": settings, "code": code}


@dataclass
class FittedModel:
    """A fitted model with everything needed to apply it (see the module docstring)."""
    model_kind: str
    constructor: dict               # vocab_sizes, static_sizes, n_context, n_fields, n_static, architecture
    state_dict: dict
    feature_schema: dict            # the run config's feature_schema record (name, blocks, widths)
    vocab: dict                     # game categorical -> {value: code}
    static_vocab: dict              # static categorical -> {value: code}
    normalizer: F.Normalizer
    row_guard: bool
    seasons: dict                   # fold, train_through, stop, protocol_test, deploy
    tables_manifest: dict           # payload_sha256 and per-table digests of the fitting tables
    data_policy: dict               # normalizer_exclusion, post_first_season_logs, label_scope, held-out players digest
    conventions: dict               # score definition, minutes kernel, quadrature nodes, foul cap
    baseline_parameters: dict = field(default_factory=dict)
    code: dict = field(default_factory=dict)
    format: str = ARTIFACT_FORMAT

    # ------------------------------------------------------------------ restore
    def schema(self) -> F.FeatureSchema:
        return F.schema_from_config({"feature_schema": self.feature_schema})

    def build_model(self):
        """The network with its trained weights, in eval mode (strict load: the recorded architecture must fit)."""
        from ..model.baseline import AggregateBaseline
        from ..model.distributions import SeasonParameters
        from ..model.tower import ReferenceTower
        c = self.constructor
        if self.model_kind == "tower":
            model = ReferenceTower(c["vocab_sizes"], c["static_sizes"], SeasonParameters(), n_context=c["n_context"],
                                   n_fields=c["n_fields"], n_static=c["n_static"], **c["architecture"])
        else:
            model = AggregateBaseline(c["vocab_sizes"], c["static_sizes"], SeasonParameters(), n_fields=c["n_fields"], n_static=c["n_static"])
        model.load_state_dict(self.state_dict, strict=True)
        model.eval()
        return model

    def store(self, games: pd.DataFrame, dobs=None) -> F.GameStore:
        """A game store for new inputs indexed with the model's own vocabulary (unseen values -> 0) and row policy."""
        return F.GameStore.from_frame(games, dobs=dobs, vocab=self.vocab, schema=self.schema(), row_guard=self.row_guard)

    def unit_static(self, units: pd.DataFrame):
        """Standardized static inputs, their masks and category codes, with the model's normalizer and vocabulary."""
        z, mask = self.normalizer.static(F.static_continuous(units, self.schema()))
        return z, mask, F.static_codes(units, self.static_vocab)

    # ------------------------------------------------------------------ checks
    def check_tables(self, tab: dict) -> bool:
        """True when ``tab`` is the table set the model was fitted on (payload digest)."""
        return tab["manifest"].get("payload_sha256") == self.tables_manifest.get("payload_sha256")

    def check_preprocessing(self, *, vocab: dict | None = None, static_vocab: dict | None = None,
                            normalizer: F.Normalizer | None = None) -> None:
        """Refuse a rebuilt vocabulary or normalizer that differs from the fitted one (D-095)."""
        problems = []
        if vocab is not None and canonical_sha256(vocab) != canonical_sha256(self.vocab):
            problems.append("game vocabulary")
        if static_vocab is not None and canonical_sha256(static_vocab) != canonical_sha256(self.static_vocab):
            problems.append("static vocabulary")
        if normalizer is not None and normalizer_sha256(normalizer) != normalizer_sha256(self.normalizer):
            problems.append("normalizer")
        if problems:
            raise ValueError("Rebuilt preprocessing differs from the fitted model's: " + ", ".join(problems))


def save_fitted_model(path: Path, fitted: FittedModel) -> str:
    """Write the artifact (torch format, plain containers only) and return its SHA-256."""
    from ..data.manifests import file_sha256
    n = fitted.normalizer
    payload = {"format": fitted.format, "model_kind": fitted.model_kind, "constructor": fitted.constructor,
               "state_dict": {k: v.detach().cpu() for k, v in fitted.state_dict.items()},
               "feature_schema": fitted.feature_schema, "vocab": fitted.vocab, "static_vocab": fitted.static_vocab,
               # exact arrays, dtype kept (game statistics are float32), so a restored model is bit-identical
               "normalizer": {k: np.array(getattr(n, k), copy=True) for k in ("mu", "sigma", "static_mu", "static_sigma")},
               "row_guard": fitted.row_guard, "seasons": fitted.seasons, "tables_manifest": fitted.tables_manifest,
               "data_policy": fitted.data_policy, "conventions": fitted.conventions,
               "baseline_parameters": fitted.baseline_parameters, "code": fitted.code}
    path = Path(path)
    torch.save(payload, path)
    return file_sha256(path)


def load_fitted_model(path: Path, expected_sha256: str | None = None) -> FittedModel:
    """Load and validate an artifact; with ``expected_sha256`` (the run manifest's) the file must match it."""
    from ..data.manifests import file_sha256
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f"{path} does not match the digest its run recorded")
    p = torch.load(path, weights_only=False)
    if p.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"{path}: unknown artifact format {p.get('format')!r}")
    nz = p["normalizer"]
    normalizer = F.Normalizer(*(np.asarray(nz[k]) for k in ("mu", "sigma", "static_mu", "static_sigma")))
    return FittedModel(model_kind=p["model_kind"], constructor=p["constructor"], state_dict=p["state_dict"],
                       feature_schema=p["feature_schema"], vocab=p["vocab"], static_vocab=p["static_vocab"],
                       normalizer=normalizer, row_guard=bool(p["row_guard"]), seasons=p["seasons"],
                       tables_manifest=p["tables_manifest"], data_policy=p["data_policy"], conventions=p["conventions"],
                       baseline_parameters=p.get("baseline_parameters", {}), code=p.get("code", {}), format=p["format"])
