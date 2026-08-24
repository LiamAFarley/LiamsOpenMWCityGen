"""Cheap stamp capability index for V2 townlayout (Phase 15).

Purpose
-------
Index every kit-brief-eligible D-STAMP v2 stamp (not Castle Barracks)
by type, size class, frontage-width band, depth band, and door count.
``compatible_stamp_count`` is an OBB/frontage prefilter only; exact hull
placement is Phase 18.

Inputs
------
``kit_brief.json`` plus one or more D-STAMP v2 libraries.

Outputs
-------
A dict with ``stamps`` (one row per eligible stamp) and ``index`` buckets.
Values trace to the source stamp JSON (hull, doors, terrain envelope).

Pipeline position
-----------------
V2 townlayout Phase 15 stamp index; no parcels/placement/VTEX.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from shapely.geometry import Polygon

from .constants import (
    PACK_SLACK_OUTSKIRTS,
    PACK_SLACK_STONE,
    PACK_SLACK_WOOD,
    STAMP_FILL_MAX,
)
from .validate import TownLayoutError

CASTLE_BARRACKS_STAMP_ID = "markarth_side_v1__u114_castle_barracks"
DEFAULT_LIBRARIES = (
    Path("output/cityforge/stamps/karthgad_nord_v2.json"),
    Path("output/cityforge/stamps/markarth_side_stone_v2.json"),
)

# Locked cheap ward → type filter. Role constraints may narrow further.
WARD_BUILDING_TYPES = {
    "market": ("shop", "tavern", "hall", "guild", "house"),
    "craft": ("smith", "shop", "guild", "house"),
    "residential": ("house", "manor", "tavern"),
    "outskirts": ("farm", "stable", "mill", "house"),
    "keep": ("keep",),
}

CORE_WARDS = ("residential", "craft", "market")
DEPTH_BAND_GU = (400.0, 800.0, 1600.0)


def is_outskirts_only(row: Mapping[str, Any]) -> bool:
    """Windmills, named huts, and farms stay off the dense core."""
    sid = str(row.get("stamp_id") or "").lower()
    btype = str(row.get("building_type") or "")
    if btype in ("farm", "stable", "mill"):
        return True
    if "windmill" in sid or "farmhouse" in sid:
        return True
    parts = sid.replace("-", "_").split("_")
    if "hut" in parts:
        return True
    return False


def kit_family(row: Mapping[str, Any]) -> str:
    """Markarth = dense stone; Karthgad = wood. Name tags beat library_id."""
    if is_outskirts_only(row):
        return "outskirts"
    blob = f"{row.get('library_id') or ''} {row.get('stamp_id') or ''}".lower()
    if "karthgad" in blob:
        return "wood"
    if "markarth" in blob:
        return "stone"
    return "wood"


def pack_slack(row: Mapping[str, Any]) -> float:
    fam = kit_family(row)
    if fam == "stone":
        return PACK_SLACK_STONE
    if fam == "outskirts":
        return PACK_SLACK_OUTSKIRTS
    return PACK_SLACK_WOOD


def _band(value: float, cuts: Sequence[float]) -> str:
    if value < cuts[0]:
        return f"0-{int(cuts[0])}"
    for lo, hi in zip(cuts, cuts[1:]):
        if value < hi:
            return f"{int(lo)}-{int(hi)}"
    return f"{int(cuts[-1])}+"


def _frontage_band(width: float) -> str:
    if width < 400.0:
        return "0-400"
    if width < 800.0:
        return "400-800"
    return "800+"


def _hull_area(stamp: Mapping[str, Any]) -> float:
    hull = stamp.get("footprint", {}).get("hull_xy_rel")
    if not isinstance(hull, list) or len(hull) < 3:
        raise TownLayoutError(
            f"invalid_polygon: stamp {stamp.get('stamp_id')} has no hull")
    poly = Polygon([(float(p[0]), float(p[1])) for p in hull])
    if poly.area <= 0:
        raise TownLayoutError(
            f"invalid_polygon: stamp {stamp.get('stamp_id')} hull area <= 0")
    return float(poly.area)


def _obb_width_depth(hull: Sequence[Sequence[float]],
                     heading_deg: float) -> tuple[float, float]:
    """Width along the door-frontage tangent; depth along outward heading."""
    rad = math.radians(heading_deg)
    nx, ny = math.cos(rad), math.sin(rad)
    tx, ty = -ny, nx
    n_vals = []
    t_vals = []
    for pt in hull:
        x, y = float(pt[0]), float(pt[1])
        n_vals.append(x * nx + y * ny)
        t_vals.append(x * tx + y * ty)
    depth = max(n_vals) - min(n_vals)
    width = max(t_vals) - min(t_vals)
    return float(width), float(depth)


def _doors(stamp: Mapping[str, Any]) -> list[dict]:
    hull = stamp.get("footprint", {}).get("hull_xy_rel") or []
    rows = []
    for member in stamp.get("members") or []:
        if not isinstance(member, Mapping) or not member.get("is_door"):
            continue
        heading = member.get("outward_heading_deg")
        if heading is None:
            continue
        heading = float(heading) % 360.0
        width, depth = _obb_width_depth(hull, heading)
        offset = member.get("offset_gu") or [0.0, 0.0, 0.0]
        rows.append({
            "door_id": member.get("source_id"),
            "outward_heading_deg": heading,
            "frontage_width_gu": width,
            "offset_gu": [float(offset[0]), float(offset[1])],
            "depth_gu": depth,
        })
    rows.sort(key=lambda d: (d["door_id"] or "", d["outward_heading_deg"]))
    return rows


def _summarize_stamp(stamp: Mapping[str, Any], kit_row: Mapping[str, Any]) -> dict:
    hull = stamp.get("footprint", {}).get("hull_xy_rel") or []
    doors = _doors(stamp)
    if not doors:
        raise TownLayoutError(
            f"missing_required: stamp {stamp.get('stamp_id')} has no doors")
    primary = doors[0]
    hull_area = _hull_area(stamp)
    terrain = stamp.get("terrain_envelope") or kit_row.get("terrain") or {}
    return {
        "stamp_id": stamp["stamp_id"],
        "building_type": stamp.get("building_type") or kit_row.get("building_type"),
        "size_class": stamp.get("size_class") or kit_row.get("size_class"),
        "door_count": int(stamp.get("door_count") or len(doors)),
        "hull_area_gu2": hull_area,
        "obb_width_gu": primary["frontage_width_gu"],
        "obb_depth_gu": primary["depth_gu"],
        "frontage_band": _frontage_band(primary["frontage_width_gu"]),
        "depth_band": _band(primary["depth_gu"], DEPTH_BAND_GU),
        "multi_shell": bool(stamp.get("multi_shell") or kit_row.get("multi_shell")),
        "terrain_envelope": {
            "burial_depth_gu": terrain.get("burial_depth_gu"),
            "footprint_slope_deg": terrain.get("footprint_slope_deg"),
            "footprint_relief_gu": terrain.get("footprint_relief_gu"),
        },
        "doors": doors,
        "library_id": stamp.get("library_id") or kit_row.get("library_id"),
        "source_stamp_id": stamp["stamp_id"],
    }


def load_stamp_libraries(paths: Sequence[Path]) -> dict[str, dict]:
    by_id = {}
    for path in paths:
        lib = json.loads(Path(path).read_text(encoding="utf-8"))
        library_id = lib.get("library_id")
        for stamp in lib.get("stamps") or []:
            sid = stamp.get("stamp_id")
            if not sid:
                continue
            row = dict(stamp)
            row["library_id"] = library_id
            by_id[sid] = row
    return by_id


def build_stamp_index(
    kit_brief: Mapping[str, Any],
    libraries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Index every eligible kit stamp. Castle Barracks is excluded."""
    eligible = list(kit_brief.get("stamps") or [])
    stamps = []
    seen: set[str] = set()
    for kit_row in eligible:
        sid = kit_row.get("stamp_id")
        if not sid or sid == CASTLE_BARRACKS_STAMP_ID:
            continue
        if sid in seen:
            raise TownLayoutError(f"duplicate_id: stamp {sid} listed twice")
        source = libraries.get(sid)
        if source is None:
            raise TownLayoutError(
                f"missing_required: kit stamp {sid} not in D-STAMP v2 libraries")
        stamps.append(_summarize_stamp(source, kit_row))
        seen.add(sid)
    stamps.sort(key=lambda s: s["stamp_id"])
    index: dict[str, list[str]] = {}

    def _add(key: str, stamp_id: str) -> None:
        index.setdefault(key, []).append(stamp_id)

    for row in stamps:
        _add(f"type:{row['building_type']}", row["stamp_id"])
        _add(f"size:{row['size_class']}", row["stamp_id"])
        _add(f"frontage:{row['frontage_band']}", row["stamp_id"])
        _add(f"depth:{row['depth_band']}", row["stamp_id"])
        _add(f"doors:{row['door_count']}", row["stamp_id"])
    for key in index:
        index[key].sort()
    return {
        "stamp_count": len(stamps),
        "stamps": stamps,
        "index": index,
        "excluded": [CASTLE_BARRACKS_STAMP_ID],
    }


def compatible_stamp_count(
    index: Mapping[str, Any],
    parcel_shape_summary: Mapping[str, Any],
    ward_type: str,
    role_constraints: Optional[Sequence[str]] = None,
) -> int:
    """Cheap OBB/frontage test. Exact hull seating is Phase 18."""
    return len(list_compatible_stamps(
        index, parcel_shape_summary, ward_type, role_constraints))


def list_compatible_stamps(
    index: Mapping[str, Any],
    parcel_shape_summary: Mapping[str, Any],
    ward_type: str,
    role_constraints: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Stamps that pass the cheap OBB/frontage test, sorted by stamp_id."""
    allowed = set(role_constraints) if role_constraints else set(
        WARD_BUILDING_TYPES.get(ward_type, ()))
    area = float(parcel_shape_summary.get("area_gu2") or 0.0)
    frontage = float(parcel_shape_summary.get("frontage_gu") or 0.0)
    depth = float(parcel_shape_summary.get("depth_gu") or 0.0)
    rows = []
    for row in index.get("stamps") or []:
        if row["building_type"] not in allowed:
            continue
        if is_outskirts_only(row) and ward_type != "outskirts":
            continue
        if row["hull_area_gu2"] > area * STAMP_FILL_MAX:
            continue
        if row["obb_width_gu"] > frontage * 0.98:
            continue
        if row["obb_depth_gu"] > depth * 0.98:
            continue
        rows.append(row)
    rows.sort(key=lambda r: r["stamp_id"])
    return rows
