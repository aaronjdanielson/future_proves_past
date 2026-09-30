"""Check the immutable scientific input to each software run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_specification(project: str | Path) -> dict:
    """Fail if source, active plan, or active decision pin disagrees."""
    root = Path(project).resolve()
    protocol_path = root / "config/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    spec = protocol["specification"]
    paper_path = (root / spec["source"]).resolve()
    if not paper_path.is_relative_to(root):
        raise ValueError("Specification source must be inside the project")
    actual = sha256_file(paper_path)
    if actual != spec["sha256"]:
        raise ValueError("Paper source hash disagrees with config/protocol.json")
    if actual not in (root / "IMPLEMENTATION_PLAN.md").read_text():
        raise ValueError("Implementation plan does not contain the current source pin")
    decisions = (root / "DECISIONS.md").read_text()
    marker = f"## {spec['decision']} "
    if marker not in decisions:
        raise ValueError("Current specification decision is absent")
    current = decisions.split(marker, 1)[1].split("\n## ", 1)[0]
    if actual not in current:
        raise ValueError("Current decision does not pin the actual paper source")
    return {
        "revision": spec["revision"],
        "paper_sha256": actual,
        "protocol_sha256": sha256_file(protocol_path),
        "decision": spec["decision"],
        "verified": True,
    }
