"""Stable hierarchical seed derivation.

Python's built-in ``hash()`` is intentionally randomized between processes,
so it is not suitable for generated world content.  ``derive_seed`` instead
uses a versioned, length-delimited SHA-256 stream and returns the first
unsigned 64 bits.  The type tags and lengths prevent scope concatenation
ambiguities (for example ``("ab", "c")`` versus ``("a", "bc")``).
"""

from __future__ import annotations

import hashlib
from numbers import Integral


def _scope_bytes(part: object) -> tuple[bytes, bytes]:
    """Encode one supported scope part with a deterministic type tag."""

    if isinstance(part, bool):
        return b"bool", (b"1" if part else b"0")
    if isinstance(part, Integral):
        return b"int", str(int(part)).encode("ascii")
    if isinstance(part, str):
        return b"str", part.encode("utf-8")
    if isinstance(part, bytes):
        return b"bytes", part
    raise TypeError(
        "scope parts must be str, int, bool, or bytes; "
        f"got {type(part).__name__}"
    )


def derive_seed(master: int, *scope_parts: object) -> int:
    """Derive a deterministic unsigned 64-bit child seed.

    ``master`` is encoded as a signed decimal integer and each scope part is
    encoded with a type tag and byte length.  The result is stable across
    Python versions, operating systems, and processes, and is suitable for
    seeding ``random.Random`` or NumPy generators at a pipeline stage.
    """

    if isinstance(master, bool) or not isinstance(master, Integral):
        raise TypeError("master must be an integer")

    digest = hashlib.sha256()
    digest.update(b"procgen-seed-v1\0")
    master_bytes = str(int(master)).encode("ascii")
    digest.update(len(master_bytes).to_bytes(4, "big"))
    digest.update(master_bytes)
    for part in scope_parts:
        tag, value = _scope_bytes(part)
        digest.update(len(tag).to_bytes(2, "big"))
        digest.update(tag)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return int.from_bytes(digest.digest()[:8], "big", signed=False)
