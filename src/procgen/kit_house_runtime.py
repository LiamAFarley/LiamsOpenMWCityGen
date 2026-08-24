"""JSON-backed runtime for procedural modular house kits.

Purpose
-------
Load a stamp library and mined grammar from JSON, then delegate generation to
``kit_house_grammar``.  The runtime deliberately contains no Falkreath,
Markarth, or other kit-specific rules: kit identity and attachment data live
in the JSON inputs.

Inputs
------
* stamp-library JSON containing ``stamps``;
* grammar JSON containing mined ``shells`` and ``stamp_templates``;
* shell id plus deterministic generation options.

Outputs
-------
A normal D-STAMP house dictionary returned by ``kit_house_grammar``.

Invariants
----------
The library and grammar are read-only inputs.  Procedural door selection is
performed by the grammar, and source windows that overlap selected doors are
suppressed by the existing grammar collision rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import kit_house_grammar


def load_json(path: Path) -> dict[str, Any]:
    """Load one kit JSON document and require an object root."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load kit JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"kit JSON root must be an object: {path}")
    return value


def generate_from_json(
    library_path: Path,
    grammar_path: Path,
    *,
    shell_id: str,
    door_slot_ids: Sequence[str] | None = None,
    include_windows: bool = True,
    include_chimney: bool = True,
    stamp_template_id: str | None = None,
    block_pattern_id: str | None = None,
    generated_id: str | None = None,
    seed: int = 0,
    access_policy: Mapping[str, Any] | None = None,
    window_facade_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate a house using only the supplied kit JSON documents."""
    library = load_json(library_path)
    grammar = load_json(grammar_path)
    return kit_house_grammar.generate_house(
        grammar,
        library,
        shell_id=shell_id,
        door_slot_ids=door_slot_ids,
        include_windows=include_windows,
        include_chimney=include_chimney,
        stamp_template_id=stamp_template_id,
        block_pattern_id=block_pattern_id,
        generated_id=generated_id,
        seed=seed,
        access_policy=access_policy,
        window_facade_ids=window_facade_ids,
    )


__all__ = ["generate_from_json", "load_json"]
