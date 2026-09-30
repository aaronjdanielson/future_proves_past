"""Write-once, self-checking manifests for frozen folds and source provenance."""
import hashlib
import json
import os
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path, payload):
    """Create once; never replace an existing fold/run manifest.

    The embedded digest detects accidental alteration. It is not a signature
    and does not defend against an adversary who rewrites both data and hash.
    """
    if not isinstance(payload, dict):
        raise TypeError("Manifest payload must be an object")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    envelope = {"manifest_format": 1, "payload_sha256": digest, "payload": payload}
    content = json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return digest


def load_manifest(path, *, expected_config=None, expected_source_sha256=None):
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(envelope) != {"manifest_format", "payload_sha256", "payload"} or envelope["manifest_format"] != 1:
        raise ValueError("Unknown manifest envelope")
    payload = envelope["payload"]
    actual = hashlib.sha256(canonical_json(payload)).hexdigest()
    if actual != envelope["payload_sha256"]:
        raise ValueError("Manifest payload hash mismatch")
    if expected_config is not None and payload.get("config") != expected_config:
        raise ValueError("Frozen configuration mismatch")
    if expected_source_sha256 is not None and payload.get("source_sha256") != expected_source_sha256:
        raise ValueError("Pinned source hash mismatch")
    return payload
