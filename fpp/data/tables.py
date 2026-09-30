"""Persist and reload the assembled tables with a write-once manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..serialization import to_jsonable
from .manifests import file_sha256, load_manifest, write_manifest

TABLE_NAMES = ("units", "competition_periods", "games")  # smallest first: a dtype failure costs no large write
OPTIONAL_TABLE_NAMES = ("players",)                     # v8 and later (D-050): bios for every player in games or units


def write_tables(tables: dict, out_dir: Path, *, config: dict, fingerprints: dict) -> dict:
    """Write parquet files and a manifest into a directory that does not yet exist."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing table directory: {out_dir}")
    out_dir.mkdir(parents=True)
    digests = {}
    for name in TABLE_NAMES + tuple(n for n in OPTIONAL_TABLE_NAMES if n in tables):
        path = out_dir / f"{name}.parquet"
        frame = tables[name].copy()
        if name == "units":
            frame["exclusion_reasons"] = frame["exclusion_reasons"].astype(str)
            frame["evidence"] = frame["evidence"].astype("string")
        for col in frame.columns[frame.dtypes == object]:
            # Mixed None/str columns are written as strings; anything else must be typed upstream.
            if frame[col].map(lambda v: v is None or isinstance(v, str)).all():
                frame[col] = frame[col].astype("string")
        frame.to_parquet(path, index=False)
        digests[name] = {"rows": int(len(frame)), "columns": list(frame.columns), "sha256": file_sha256(path)}
    payload = {"kind": "training_tables", "config": config, "upstream": fingerprints,
               "tables": digests, "audit": to_jsonable(tables["audit"])}
    write_manifest(out_dir / "manifest.json", payload)
    return payload


def read_tables(in_dir: Path, *, verify: bool = True) -> dict:
    """Load the tables; by default the manifest hash and every table digest are verified first."""
    in_dir = Path(in_dir)
    payload = load_manifest(in_dir / "manifest.json")          # rejects an altered payload
    names = TABLE_NAMES + tuple(n for n in OPTIONAL_TABLE_NAMES if n in payload["tables"])
    if verify:
        for name in names:
            expected = payload["tables"][name]["sha256"]
            actual = file_sha256(in_dir / f"{name}.parquet")
            if actual != expected:
                raise ValueError(f"Table {name!r} does not match its manifest digest ({actual[:12]} != {expected[:12]})")
    tables = {name: pd.read_parquet(in_dir / f"{name}.parquet") for name in names}
    for name in names:
        if len(tables[name]) != payload["tables"][name]["rows"]:
            raise ValueError(f"Table {name!r} row count differs from the manifest")
    tables["audit"] = payload["audit"]
    tables["manifest"] = dict(payload)
    envelope = json.loads((in_dir / "manifest.json").read_text(encoding="utf-8"))
    tables["manifest"]["payload_sha256"] = envelope["payload_sha256"]     # the envelope's digest of this payload
    return tables
