"""Content hashes and lightweight stage provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Mapping, TypeAlias

PathLike: TypeAlias = str | Path


def sha256_file(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally without loading it into memory."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"provenance input is not a file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stamp(
    tool: str,
    version: str,
    inputs: Mapping[str, PathLike],
) -> dict[str, object]:
    """Create a provenance object for attaching under ``meta.provenance``.

    Input hashes are calculated in mapping insertion order but represented by
    ordinary JSON-compatible dictionaries; consumers should treat keys as
    names, not positional data.  The timestamp is UTC and ISO-8601 with a
    trailing ``Z``.
    """

    if not tool or not version:
        raise ValueError("tool and version must be non-empty")
    input_records: dict[str, dict[str, str]] = {}
    for name, path in inputs.items():
        if not name:
            raise ValueError("provenance input names must be non-empty")
        resolved = Path(path)
        input_records[str(name)] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    return {
        "tool": tool,
        "version": version,
        "generated_utc": timestamp,
        "inputs": input_records,
    }
