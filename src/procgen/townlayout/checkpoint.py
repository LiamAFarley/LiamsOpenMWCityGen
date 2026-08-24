"""R1 checkpoint persistence and provenance verification.

The checkpoint is self-contained for downstream geometry and records hashes for
large immutable inputs.  Readers verify those hashes before returning data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .validate import TownLayoutError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checkpoint(product: dict[str, Any], path: str | Path) -> None:
    """Write deterministic JSON and verify its declared input identities."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(product, allow_nan=False, indent=2) + "\n",
                      encoding="utf-8")


def read_checkpoint(path: str | Path, expected_stages: tuple[str, ...] = ("r1",)) -> dict[str, Any]:
    """Read a checkpoint and re-hash every declared file identity."""
    target = Path(path)
    try:
        product = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TownLayoutError(f"checkpoint_read: {target}: {exc}") from exc
    if not isinstance(product, dict) or product.get("stage_id") not in expected_stages:
        raise TownLayoutError(f"checkpoint_read: expected stage_id in {expected_stages}")
    for name, record in (product.get("identities") or {}).items():
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            raise TownLayoutError(f"checkpoint_identity: malformed {name}")
        source = Path(record["path"])
        if not source.is_file():
            raise TownLayoutError(f"checkpoint_identity: missing {name}: {source}")
        actual = sha256_file(source)
        if actual != record["sha256"]:
            raise TownLayoutError(f"checkpoint_identity: hash mismatch {name}")
    return product
