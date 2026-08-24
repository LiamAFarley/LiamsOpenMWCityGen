"""Expand an accepted town layout into world object placements.

Purpose
-------
This is the direct bridge from the authoritative R13 ``city_layout.json`` to
the object-placement boundary consumed by later terrain and ESP stages.  It
does not seat objects on LAND, realize roads, or serialize TES3 records.

Inputs
------
An accepted R13 city layout and the pinned D-STAMP v2 libraries.  Placement
anchors are the layout's 2D seed-door coordinates; ``anchor_z_gu`` is supplied
by the caller and defaults to zero because terrain seating is a later stage.

Outputs
-------
A deterministic JSON-ready document containing one expanded object per stamp
member, stable ``parcel_id:source_id`` identities, preserved source transforms,
and mathematical exterior-cell buckets.

Invariants
----------
Every occupied layout placement resolves to exactly one stamp.  Every stamp
member is emitted exactly once.  Plan yaw and member transforms are delegated
to :func:`procgen.cityplace_transform.place_stamp_members`; no rotation math
is duplicated here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..cityplace_transform import place_stamp_members
from ..townlayout.stamp_index import DEFAULT_LIBRARIES


STAGE_ID = "townlayout_stamp_objects_v1"


class StampObjectError(ValueError):
    """Raised when layout-to-stamp expansion cannot preserve the contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StampObjectError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise StampObjectError(f"{label} is not finite")
    return result


def _anchor(anchor: Sequence[Any], anchor_z_gu: float, label: str) -> list[float]:
    if len(anchor) != 2:
        raise StampObjectError(f"{label} must contain two coordinates")
    return [_number(anchor[0], f"{label}[0]"), _number(anchor[1], f"{label}[1]"),
            _number(anchor_z_gu, "anchor_z_gu")]


def _load_layout(path: Path) -> dict[str, Any]:
    try:
        layout = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StampObjectError(f"cannot read layout {path}: {exc}") from exc
    if not isinstance(layout, dict):
        raise StampObjectError("layout root must be an object")
    if layout.get("stage_id") != "r13_city_layout":
        raise StampObjectError(
            f"expected r13_city_layout, got {layout.get('stage_id')!r}")
    if not isinstance(layout.get("placements"), list):
        raise StampObjectError("layout has no placements list")
    return layout


def realize_stamp_objects(
    layout: Mapping[str, Any],
    libraries: Mapping[str, Mapping[str, Any]],
    *,
    anchor_z_gu: float = 0.0,
    library_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Expand every occupied layout placement into world object rows."""

    if layout.get("stage_id") != "r13_city_layout":
        raise StampObjectError("stamp expansion requires r13_city_layout")
    stamp_by_id: dict[str, Mapping[str, Any]] = {}
    library_by_stamp: dict[str, str] = {}
    for library_id, library in libraries.items():
        for stamp in library.get("stamps", []):
            stamp_id = stamp.get("stamp_id")
            if not isinstance(stamp_id, str) or not stamp_id:
                raise StampObjectError(f"library {library_id} contains a stamp without id")
            if stamp_id in stamp_by_id:
                raise StampObjectError(f"duplicate stamp id {stamp_id}")
            stamp_by_id[stamp_id] = stamp
            library_by_stamp[stamp_id] = str(library_id)

    placements: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    seen_references: set[str] = set()
    for placement in layout.get("placements", []):
        if not isinstance(placement, Mapping):
            raise StampObjectError("placement is not an object")
        parcel_id = placement.get("parcel_id")
        stamp_id = placement.get("stamp_id")
        if not isinstance(parcel_id, str) or not parcel_id:
            raise StampObjectError("placement has no parcel_id")
        if stamp_id is None:
            continue
        if not isinstance(stamp_id, str) or stamp_id not in stamp_by_id:
            raise StampObjectError(f"{parcel_id} references missing stamp {stamp_id!r}")
        anchor = _anchor(placement.get("anchor"), anchor_z_gu, f"{parcel_id}.anchor")
        yaw_deg = _number(placement.get("yaw_deg"), f"{parcel_id}.yaw_deg")
        stamp = stamp_by_id[stamp_id]
        placed = place_stamp_members(
            stamp, anchor_world_gu=anchor, yaw_deg=yaw_deg, include_render_euler=False)
        placement_objects: list[str] = []
        for member in placed:
            reference_id = f"{parcel_id}:{member.source_id}"
            if reference_id in seen_references:
                raise StampObjectError(f"duplicate reference identity {reference_id}")
            seen_references.add(reference_id)
            row = member.to_dict(include_render_euler=False)
            row.update({
                "reference_id": reference_id,
                "placement_id": parcel_id,
                "stamp_id": stamp_id,
                "library_id": library_by_stamp[stamp_id],
                "plan_anchor_gu": [anchor[0], anchor[1]],
                "plan_yaw_deg": yaw_deg,
            })
            objects.append(row)
            placement_objects.append(reference_id)
        placements.append({
            "placement_id": parcel_id,
            "stamp_id": stamp_id,
            "library_id": library_by_stamp[stamp_id],
            "anchor_world_gu": anchor,
            "plan_yaw_deg": yaw_deg,
            "object_ids": placement_objects,
        })

    objects.sort(key=lambda row: row["reference_id"])
    placements.sort(key=lambda row: row["placement_id"])
    cells: dict[str, list[str]] = {}
    for row in objects:
        cell = ":".join(str(value) for value in row["world_cell"])
        cells.setdefault(cell, []).append(row["reference_id"])
    for refs in cells.values():
        refs.sort()
    return {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "source": {
            "layout_stage_id": layout.get("stage_id"),
            "layout_id": layout.get("layout_id") or layout.get("candidate_id"),
            "placement_count": len(placements),
            "library_sha256": {
                str(path): _sha256(Path(path)) for path in library_paths
            },
        },
        "anchor_z_gu": _number(anchor_z_gu, "anchor_z_gu"),
        "placements": placements,
        "objects": objects,
        "exterior_cells": [
            {"cell": [int(x) for x in cell.split(":")], "object_ids": refs}
            for cell, refs in sorted(cells.items())
        ],
        "metrics": {
            "placement_count": len(placements),
            "object_count": len(objects),
            "door_count": sum(1 for row in objects if row["is_door"]),
            "stat_count": sum(1 for row in objects if row["record_type"] == "STAT"),
            "cell_count": len(cells),
        },
    }


def realize_from_paths(
    layout_path: Path,
    library_paths: Sequence[Path] = DEFAULT_LIBRARIES,
    *,
    anchor_z_gu: float = 0.0,
) -> dict[str, Any]:
    """Read canonical inputs and build the stamp-object product."""

    layout = _load_layout(layout_path)
    libraries: dict[str, dict[str, Any]] = {}
    for path in library_paths:
        try:
            library = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StampObjectError(f"cannot read stamp library {path}: {exc}") from exc
        library_id = library.get("library_id")
        if not isinstance(library_id, str) or not library_id:
            raise StampObjectError(f"stamp library {path} has no library_id")
        if library_id in libraries:
            raise StampObjectError(f"duplicate library id {library_id}")
        libraries[library_id] = library
    generated = layout.get("generated_stamps") or {}
    if generated:
        generated_library_id = "generated_fk_house_v1"
        if generated_library_id in libraries:
            raise StampObjectError(f"duplicate library id {generated_library_id}")
        libraries[generated_library_id] = {
            "library_id": generated_library_id,
            "stamps": [generated[stamp_id]
                       for stamp_id in sorted(generated)
                       if isinstance(generated[stamp_id], Mapping)],
        }
    product = realize_stamp_objects(
        layout, libraries, anchor_z_gu=anchor_z_gu, library_paths=library_paths)
    if generated:
        product["generated_stamps"] = generated
    return product
