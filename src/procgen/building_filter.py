"""Deterministic name rules used by exterior building extraction.

The rules in this module are intentionally conservative. A name match does
not delete a reference from the source scan; it marks the reference as a
conditional support candidate. The Karthgad driver may promote that candidate
only after measured bbox contact with a retained shell piece.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from .chunker import ChunkPiece


DEFAULT_MEMBER_DENYLIST: tuple[str, ...] = (
    "ffence",
    "fence",
    "palisade",
    "wagon",
    "cart",
    "stand",
    "well",
    "pole",
    "post",
    "strut",
    "stake",
)

DEFAULT_STAIR_HINTS: tuple[str, ...] = (
    "wdstp",
    "wdstr",
    "stair",
    "step",
    "ladder",
    "ramp",
)


@dataclass(frozen=True)
class NameRuleMatch:
    pattern: str
    normalized_text: str


def normalized_piece_text(piece: ChunkPiece) -> str:
    """Return stable searchable text from MODL and TES3 object id."""

    return " ".join(
        str(value or "")
        for value in (piece.model, piece.object_id)
    ).casefold().replace("\\", " ").replace("/", " ").replace("_", " ").replace("-", " ")


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _matches(text: str, pattern: str) -> bool:
    normalized = " ".join(text.split())
    candidate = " ".join(str(pattern).casefold().replace("_", " ").replace("-", " ").split())
    if not candidate:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    candidate_tokens = tuple(re.findall(r"[a-z0-9]+", candidate))
    if candidate in normalized:
        return True
    if candidate_tokens and all(token in tokens for token in candidate_tokens):
        return True
    return _compact(candidate) in _compact(normalized)


def denylist_match(
    piece: ChunkPiece,
    patterns: Sequence[str] = DEFAULT_MEMBER_DENYLIST,
) -> NameRuleMatch | None:
    """Return the first configured conditional-membership name match."""

    text = normalized_piece_text(piece)
    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip().casefold()
        if pattern and _matches(text, pattern):
            return NameRuleMatch(pattern=pattern, normalized_text=text)
    return None


def is_stair_piece(
    piece: ChunkPiece,
    patterns: Sequence[str] = DEFAULT_STAIR_HINTS,
) -> bool:
    """Return whether a piece name is a plausible access/stair mesh."""

    text = normalized_piece_text(piece)
    return any(_matches(text, str(pattern)) for pattern in patterns if str(pattern).strip())


def first_name_match(text: str, patterns: Iterable[str]) -> str | None:
    """Small string-only helper for ghost-pool diagnostics and tests."""

    normalized = " ".join(str(text).casefold().replace("_", " ").split())
    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip().casefold()
        if pattern and _matches(normalized, pattern):
            return pattern
    return None


__all__ = [
    "DEFAULT_MEMBER_DENYLIST",
    "DEFAULT_STAIR_HINTS",
    "NameRuleMatch",
    "denylist_match",
    "first_name_match",
    "is_stair_piece",
    "normalized_piece_text",
]
