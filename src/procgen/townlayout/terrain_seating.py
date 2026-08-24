"""Door-first terrain seating for expanded townlayout stamp objects.

Purpose
-------
Seat the stage-1 stamp-object product against the authoritative Falkreath
height field.  The primary door is the controlling datum: its measured D-STAMP
door-step height is preserved above the local terrain, and the whole stamp is
translated vertically by that amount.

Inputs
-------
``townlayout_stamp_objects_v1`` JSON, the D-STAMP v2 libraries, and the
accepted survey-backed ``TerrainField``.

Outputs
-------
The same object product with seated Z coordinates, per-placement terrain
evidence, and explicit local terrain-pad requests for LAND authoring.

Invariants
----------
Every primary door has non-negative terrain clearance. The LAND stage receives
lower-only footprint deformation capped at the primary door's source terrain
height; no building is raised by this stage. Plan XY, yaw, source rotations,
and identities are unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..cityplace_contracts import TerrainField
from ..cityplace_geometry import transform_hull
from ..cityplace_transform import place_stamp_members


MAX_BOTTOM_PROTRUSION_GU = 128.0
FK_MAX_BOTTOM_PROTRUSION_GU = 256.0
DOOR_CLEARANCE_EPSILON_GU = 0.0


class TerrainSeatingError(ValueError):
    """Raised when door-first seating cannot satisfy the stage contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerrainSeatingError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TerrainSeatingError(f"{label} must be a JSON object")
    return value


def _stamp_map(
    library_paths: tuple[Path, ...],
    generated_stamps: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Mapping[str, Any]]:
    stamps: dict[str, Mapping[str, Any]] = {}
    for path in library_paths:
        library = _load_json(path, "stamp library")
        for stamp in library.get("stamps", []):
            stamp_id = stamp.get("stamp_id")
            if not isinstance(stamp_id, str) or stamp_id in stamps:
                raise TerrainSeatingError(f"invalid or duplicate stamp id in {path}")
            stamps[stamp_id] = stamp
    for stamp_id in sorted(generated_stamps or {}):
        stamp = generated_stamps[stamp_id]
        if not isinstance(stamp, Mapping) or stamp.get("stamp_id") != stamp_id:
            raise TerrainSeatingError("invalid generated stamp table")
        if stamp_id in stamps:
            raise TerrainSeatingError(f"duplicate generated stamp id {stamp_id}")
        stamps[stamp_id] = stamp
    return stamps


def seat_stamp_objects(
    product: Mapping[str, Any],
    stamps: Mapping[str, Mapping[str, Any]],
    field: TerrainField,
) -> dict[str, Any]:
    """Apply measured door-step seating to every expanded placement."""

    if product.get("stage_id") != "townlayout_stamp_objects_v1":
        raise TerrainSeatingError("terrain seating requires townlayout_stamp_objects_v1")
    objects_by_placement: dict[str, list[dict[str, Any]]] = {}
    for object_row in product.get("objects", []):
        objects_by_placement.setdefault(str(object_row["placement_id"]), []).append(object_row)

    seated = [dict(row) for row in product.get("objects", [])]
    seated_by_id = {row["reference_id"]: row for row in seated}
    placements: list[dict[str, Any]] = []
    for placement in product.get("placements", []):
        placement_id = str(placement["placement_id"])
        stamp_id = str(placement["stamp_id"])
        stamp = stamps.get(stamp_id)
        if stamp is None:
            raise TerrainSeatingError(f"missing stamp {stamp_id} for {placement_id}")
        anchor = placement.get("anchor_world_gu")
        if not isinstance(anchor, list) or len(anchor) != 3:
            raise TerrainSeatingError(f"invalid anchor for {placement_id}")
        anchor_xy = [float(anchor[0]), float(anchor[1])]
        yaw_deg = float(placement["plan_yaw_deg"])
        members = place_stamp_members(
            stamp, anchor_world_gu=[anchor_xy[0], anchor_xy[1], 0.0],
            yaw_deg=yaw_deg, include_render_euler=False)
        doors = [member for member in members if member.is_door]
        if not doors:
            raise TerrainSeatingError(f"stamp {stamp_id} has no door for {placement_id}")
        seed_id = ((stamp.get("source") or {}).get("seed_door"))
        door = next((member for member in doors if member.source_id == seed_id), doors[0])
        door_sample = field.sample(anchor_xy[0], anchor_xy[1])
        steps = (stamp.get("terrain_envelope") or {}).get("door_step_heights_gu") or []
        door_index = next((i for i, member in enumerate(doors) if member.source_id == door.source_id), 0)
        if door_index >= len(steps):
            raise TerrainSeatingError(f"door step missing for {stamp_id}/{door.source_id}")
        step_gu = float(steps[door_index])
        if not math.isfinite(step_gu) or step_gu < 0.0:
            raise TerrainSeatingError(f"invalid door step for {stamp_id}/{door.source_id}")
        seated_anchor_z = door_sample.height_gu + step_gu - door.source_offset_gu[2]
        hull = transform_hull(
            stamp["footprint"]["hull_xy_rel"],
            anchor_xy_plan_gu=anchor_xy,
            yaw_deg=yaw_deg)
        terrain_heights = [field.sample(x, y).height_gu for x, y in hull]
        min_terrain = min(terrain_heights)
        max_terrain = max(terrain_heights)
        bottom_z = seated_anchor_z + float(stamp["bounds_rel_gu"]["min"][2])
        bottom_protrusion = max(0.0, bottom_z - min_terrain)
        door_clearance = seated_anchor_z + door.source_offset_gu[2] - door_sample.height_gu
        if door_clearance < DOOR_CLEARANCE_EPSILON_GU:
            raise TerrainSeatingError(f"buried primary door {placement_id}: {door_clearance}")
        bottom_limit = (FK_MAX_BOTTOM_PROTRUSION_GU
                        if str(stamp_id).startswith("fkgen__")
                        else MAX_BOTTOM_PROTRUSION_GU)
        if bottom_protrusion > bottom_limit:
            raise TerrainSeatingError(
                f"bottom protrudes too far for {placement_id}: {bottom_protrusion:.2f} GU")
        for row in objects_by_placement.get(placement_id, []):
            target = seated_by_id[row["reference_id"]]
            target["world_position_gu"] = [
                float(target["world_position_gu"][0]),
                float(target["world_position_gu"][1]),
                float(target["world_position_gu"][2]) + seated_anchor_z,
            ]
            target["seated_anchor_z_gu"] = seated_anchor_z
        placements.append({
            **placement,
            "anchor_world_gu": [anchor_xy[0], anchor_xy[1], seated_anchor_z],
            "terrain_seating": {
                "primary_door_source_id": door.source_id,
                "door_terrain_height_gu": door_sample.height_gu,
                "door_step_height_gu": step_gu,
                "door_clearance_gu": door_clearance,
                "footprint_terrain_min_gu": min_terrain,
                "footprint_terrain_max_gu": max_terrain,
                "bottom_z_gu": bottom_z,
                "bottom_protrusion_gu": bottom_protrusion,
                "terrain_deformation": {
                    "polygon": [[[float(x), float(y)] for x, y in hull]],
                    "mode": "lower_only_to_door_terrain",
                    "ceiling_height_gu": door_sample.height_gu,
                    "blend_margin_gu": 1024.0,
                },
                "terrain_sample_count": len(terrain_heights),
            },
        })
    placements.sort(key=lambda row: row["placement_id"])
    seated.sort(key=lambda row: row["reference_id"])
    out = dict(product)
    out.update({
        "stage_id": "townlayout_stamp_objects_seated_v1",
        "terrain_field": field.contract_dict(),
        "seating_policy": {
            "mode": "primary_door_first",
            "max_bottom_protrusion_gu": MAX_BOTTOM_PROTRUSION_GU,
            "door_clearance_epsilon_gu": DOOR_CLEARANCE_EPSILON_GU,
        },
        "placements": placements,
        "objects": seated,
        "metrics": {
            **dict(product.get("metrics") or {}),
            "seated_placement_count": len(placements),
            "buried_primary_door_count": 0,
            "max_bottom_protrusion_gu": max(
                row["terrain_seating"]["bottom_protrusion_gu"] for row in placements),
            "max_door_clearance_gu": max(
                row["terrain_seating"]["door_clearance_gu"] for row in placements),
            "terrain_deformation_count": len(placements),
        },
    })
    return out


def seat_from_paths(
    product_path: Path,
    survey_path: Path,
    field_path: Path,
    library_paths: tuple[Path, ...],
) -> dict[str, Any]:
    product = _load_json(product_path, "stamp-object product")
    survey = _load_json(survey_path, "site survey")
    field = TerrainField.from_npz(field_path, survey=survey, field_pass="planned")
    return seat_stamp_objects(
        product, _stamp_map(library_paths, product.get("generated_stamps")), field)
