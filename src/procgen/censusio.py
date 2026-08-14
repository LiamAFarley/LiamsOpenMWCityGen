"""Deterministic I/O and statistics foundations for the D-BRIEF census family.

Pipeline position
------------------
This module is the shared bottom layer of the Cityforge T0.5 D-BRIEF census
(dispatch 5): it provides the byte-deterministic JSON serializer, the
hash-pinning helper, and the single quantile definition used by
``citybrief.py`` (stamp/spacing census) and ``regionpalette.py`` (LAND/VTEX
census).  Consumers are the engine modules and the CLI
``tools/cityforge/build_city_brief.py``; tests pin its behavior on fixtures.

Invariants
----------
- ``deterministic_dumps`` is the ONLY serializer used for the four canonical
  outputs.  It emits UTF-8, ``sort_keys=True``, 2-space indent, no BOM, a
  trailing newline, and rounds every JSON float to 6 decimal places
  (``round(value, 6)``).  Integer values stay integers.  Two builds on the
  same inputs therefore produce byte-identical files.
- ``quantiles_linear`` implements the numpy-default *linear interpolation
  between closest ranks* quantile definition with pure Python arithmetic
  (float64), so it is reproducible by an independent test oracle and does not
  depend on numpy being installed.
- ``PinnedFile`` caches SHA-256 per absolute path within one process; all
  provenance records in the census outputs use it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

JSON = Any  # JSON-compatible object graph

#: Canonical float rounding for every number that reaches the output files.
FLOAT_DIGITS = 6


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase hex SHA-256 of one file (streaming)."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"not a file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class PinnedFile:
    """One read-only input file with a cached SHA-256 pin."""

    def __init__(self, path: str | Path, *, alias: str | None = None) -> None:
        self.path = Path(path)
        self.alias = alias
        self._sha256: str | None = None

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def sha256(self) -> str:
        if self._sha256 is None:
            if not self.exists:
                raise FileNotFoundError(f"pinned input missing: {self.path}")
            self._sha256 = sha256_file(self.path)
        return self._sha256

    def pin(self) -> dict[str, object]:
        """Machine-readable provenance record: path + hash + size bytes."""
        return {
            "path": str(self.path),
            "sha256": self.sha256(),
            "size_bytes": self.path.stat().st_size,
        }

    def pin_named(self) -> dict[str, dict[str, object]]:
        key = self.alias if self.alias is not None else str(self.path)
        return {key: self.pin()}


def norm_float(value: float) -> float:
    """Round one float to the canonical census precision."""
    return round(float(value), FLOAT_DIGITS)


def _round_leaves(node: JSON) -> JSON:
    if isinstance(node, float):
        return norm_float(node)
    if isinstance(node, dict):
        return {key: _round_leaves(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_round_leaves(value) for value in node]
    if isinstance(node, tuple):
        return [_round_leaves(value) for value in node]
    return node


def deterministic_dumps(payload: Mapping[str, Any]) -> bytes:
    """Serialize a payload to the canonical census JSON bytes.

    Rules (all four canonical outputs use this exact function):
    - UTF-8, no BOM, ``sort_keys=True`` (keys sorted bytewise), 2-space
      indent, trailing newline.
    - Every float is rounded to ``FLOAT_DIGITS`` decimals first; integers
      and integral-valued floats that appear as JSON numbers stay as-is
      where the assembly already produced ints (census tile counts are
      ints; cell coordinates are ints; measured quantities are floats).
    - ``ensure_ascii=False`` so record ids/paths with non-ASCII characters
      round-trip byte-identically.
    """
    rounded = _round_leaves(dict(payload))
    text = json.dumps(rounded, sort_keys=True, indent=2, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def write_deterministic(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write one canonical output file; return its SHA-256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = deterministic_dumps(payload)
    target.write_bytes(data)
    return sha256_file(target)


def quantiles_linear(
    values: Sequence[float], quantiles: Sequence[float] = (0.1, 0.5, 0.9)
) -> dict[float, float | None]:
    """numpy-default linear-interpolation quantiles over a finite sample.

    Definition (matches numpy ``quantile(..., method='linear')`` and
    R-7): for sorted sample ``v`` of size ``n`` and probability ``q``,
    ``pos = q * (n - 1)``; ``lo = floor(pos)``; ``hi = lo + 1``;
    result = ``v[lo] + (pos - lo) * (v[hi] - v[lo])``.  Single-element and
    empty samples return the element / ``None``.

    The engine and the independent test oracle share this documented
    definition; the oracle re-implements it from the formula so the test
    does not silently reuse engine code.
    """
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    result: dict[float, float | None] = {}
    for q in quantiles:
        if n == 0:
            result[q] = None
        elif n == 1:
            result[q] = ordered[0]
        else:
            pos = q * (n - 1)
            lo = int(math.floor(pos))
            hi = lo + 1
            frac = pos - lo
            result[q] = ordered[lo] * (1.0 - frac) + ordered[hi] * frac
    return result


def quantile_summary(
    values: Sequence[float],
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> dict[str, object]:
    """Deterministic summary record used for every measured distribution."""
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    return {
        "n": n,
        "min": ordered[0] if n else None,
        "max": ordered[-1] if n else None,
        "mean": (sum(ordered) / n) if n else None,
        "method": "linear interpolation between closest ranks (numpy default)",
        "p10": quantiles_linear(ordered, (0.1,))[0.1],
        "p50": quantiles_linear(ordered, (0.5,))[0.5],
        "p90": quantiles_linear(ordered, (0.9,))[0.9],
    }


def stable_json_identity(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical serialization (used for cross-file pins)."""
    return hashlib.sha256(deterministic_dumps(payload)).hexdigest()


def list_of_pairs(iterable: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Deterministic pair list form (sorted, deduplicated)."""
    return sorted([sorted(pair) for pair in set(iterable)])
