"""Strict TownBrief / CityLayout structural specs for V2 townlayout.

Purpose
-------
Fail-closed recursive schema walk for ``town_brief.json`` and
``city_layout.json`` (schema_version 1).  Unknown keys, wrong types, and
non-finite numbers are errors.  Semantic / cross-reference rules live in
``validate.py``.

Inputs
------
A loaded JSON document (dict) plus a spec dict produced by ``_spec``.

Outputs
-------
A list of issue dicts ``{severity, code, path, message}``.  The walker
never mutates the document.

Pipeline position
-----------------
V2 townlayout Phase 1 contracts; no generation.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

_NUMERIC = ("num", "int")

MORPHOLOGY = ("organic", "loose_grid", "radial", "meandering", "mixed")
FORTIFICATION_MODES = ("palisade", "none")
ANCHOR_PRESENCE = ("required", "optional", "absent")
EDGE_ROLES = ("block", "wall", "fringe")
BOUNDARY_ROAD_CLASSES = (
    "none", "regional_approach", "arterial", "street", "lane", "alley",
)
NODE_KINDS = ("junction", "gate", "source_approach", "plaza", "anchor")
ROAD_HIERARCHY = (
    "regional_approach", "arterial", "street", "lane", "alley",
)
PAINT_SURFACES = ("road", "settlement_dirt", "none")
WARD_TYPES = ("market", "craft", "residential", "outskirts", "keep")
DPLAN_KINDS = (
    "core", "residential", "market", "docks", "farms", "temple", "keep",
    "craft", "outskirts",
)
ANCHOR_KINDS = ("market", "keep", "temple")
OPEN_SPACE_KINDS = ("plaza", "court", "park", "verge")
FRONTAGE_TARGET_TYPES = ("street", "alley", "court", "plaza")
REPORT_STATUSES = ("ok", "repaired", "rejected")

YAW_CONVENTION = (
    "+x east +y north; yaw degrees CCW from +x about door anchor"
)
COORDINATE_SPACE = "site_survey_plan_gu"
STAMP_STATS_SOURCE = "dbrief_census_global_hull_area_gu2"
CANDIDATE_ID_RE = re.compile(r"^c[0-9]{2}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _spec(*, type: str, required: bool = False, nonempty: bool = False,
          values: Optional[tuple] = None, min_points: int = 0,
          min_items: int = 0, keys: Optional[dict] = None,
          item: Any = None, nullable: bool = False,
           pattern: Optional[re.Pattern] = None, dynamic: bool = False) -> dict:
    out: dict[str, Any] = {"type": type, "required": required}
    if nonempty:
        out["nonempty"] = True
    if values is not None:
        out["values"] = values
    if min_points:
        out["min_points"] = min_points
    if min_items:
        out["min_items"] = min_items
    if keys is not None:
        out["keys"] = keys
    if item is not None:
        out["item"] = item
    if nullable:
        out["nullable"] = True
    if pattern is not None:
        out["pattern"] = pattern
    if dynamic:
        out["dynamic"] = True
    return out


def issue(code: str, path: str, message: str,
          severity: str = "error") -> dict:
    return {
        "severity": severity,
        "code": code,
        "path": path,
        "message": message,
    }


def json_path(base: str, *parts: Any) -> str:
    out = base
    for part in parts:
        out = f"{out}{part}" if str(part).startswith("[") else f"{out}.{part}"
    return out


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_number(value: Any, path: str, issues: list, *, int_only: bool) -> None:
    if int_only:
        if not _is_int(value):
            issues.append(issue("wrong_type", path,
                                f"expected integer, got {type(value).__name__}"))
            return
    elif not _is_num(value):
        issues.append(issue("wrong_type", path,
                            f"expected number, got {type(value).__name__}"))
        return
    if isinstance(value, float) and not math.isfinite(value):
        issues.append(issue("non_finite_number", path,
                            "number is NaN or infinite"))


def _resolve_item_spec(tag: Any) -> dict:
    if isinstance(tag, str):
        keys = ITEM_SPEC.get(tag)
        if isinstance(keys, dict) and "type" in keys:
            return keys
        if isinstance(keys, dict):
            return {"type": "object", "keys": keys}
        return _spec(type="str")
    if isinstance(tag, dict):
        # Inline item maps (for example the CityLayout ports array) use the
        # same shorthand as ITEM_SPEC entries and still need an object shell
        # when emitting JSON Schema.
        return tag if "type" in tag else {"type": "object", "keys": tag}
    return _spec(type="str")


def check_structure(node: Any, spec: dict, path: str, issues: list) -> None:
    """Strict structural gate.  Appends issues; does not mutate ``node``."""
    if spec.get("nullable") and node is None:
        return
    kind = spec["type"]
    if kind == "object":
        if not isinstance(node, dict):
            issues.append(issue("wrong_type", path,
                                f"expected object, got {type(node).__name__}"))
            return
        allowed = spec.get("keys", {})
        for key in node:
            if key not in allowed and not spec.get("dynamic"):
                issues.append(issue("unknown_key", json_path(path, key),
                                    "unknown key (strict schema)"))
        for key, child_spec in allowed.items():
            if key not in node:
                if child_spec.get("required"):
                    issues.append(issue("missing_required",
                                        json_path(path, key),
                                        "missing required field"))
                continue
            check_structure(node[key], child_spec, json_path(path, key), issues)
    elif kind == "array":
        if not isinstance(node, list):
            issues.append(issue("wrong_type", path,
                                f"expected array, got {type(node).__name__}"))
            return
        if spec.get("min_items") and len(node) < spec["min_items"]:
            issues.append(issue("too_few_items", path,
                                f"expected at least {spec['min_items']} items, "
                                f"got {len(node)}"))
        item_spec = _resolve_item_spec(spec.get("item"))
        for idx, item in enumerate(node):
            check_structure(item, item_spec, json_path(path, f"[{idx}]"),
                            issues)
    elif kind == "str":
        if not isinstance(node, str):
            issues.append(issue("wrong_type", path,
                                f"expected string, got {type(node).__name__}"))
            return
        if spec.get("nonempty") and not node.strip():
            issues.append(issue("empty_id", path, "string must be non-empty"))
        if spec.get("values") and node not in spec["values"]:
            issues.append(issue("bad_enum", path,
                                f"value {node!r} not in {list(spec['values'])}"))
        pattern = spec.get("pattern")
        if pattern is not None and not pattern.match(node):
            issues.append(issue("bad_pattern", path,
                                f"value {node!r} does not match {pattern.pattern}"))
    elif kind in _NUMERIC:
        _check_number(node, path, issues, int_only=(kind == "int"))
    elif kind == "bool":
        if not isinstance(node, bool):
            issues.append(issue("wrong_type", path,
                                f"expected boolean, got {type(node).__name__}"))
    elif kind == "point":
        if not isinstance(node, list) or len(node) != 2:
            issues.append(issue("wrong_type", path,
                                "expected [x, y] pair of numbers"))
            return
        for idx, val in enumerate(node):
            _check_number(val, json_path(path, f"[{idx}]"), issues,
                          int_only=False)
    elif kind == "points":
        if not isinstance(node, list):
            issues.append(issue("wrong_type", path,
                                f"expected list of [x, y] pairs, got "
                                f"{type(node).__name__}"))
            return
        if spec.get("min_points") and len(node) < spec["min_points"]:
            issues.append(issue("too_few_points", path,
                                f"expected at least {spec['min_points']} points, "
                                f"got {len(node)}"))
        for idx, pt in enumerate(node):
            if not isinstance(pt, list) or len(pt) != 2:
                issues.append(issue("wrong_type", json_path(path, f"[{idx}]"),
                                    "expected [x, y] pair of numbers"))
                continue
            for sub, val in enumerate(pt):
                _check_number(val, json_path(path, f"[{idx}][{sub}]"),
                              issues, int_only=False)
    else:  # pragma: no cover - schema bug guard
        raise RuntimeError(f"internal schema error: unknown type {kind!r}")


# ---------------------------------------------------------------------------
# TownBrief
# ---------------------------------------------------------------------------

TOWN_BRIEF_KEYS = {
    "schema_version": _spec(type="int", required=True),
    "town_id": _spec(type="str", required=True, nonempty=True),
    "master_seed": _spec(type="int", required=True),
    "target_buildings": _spec(type="object", required=True, keys={
        "min": _spec(type="int", required=True),
        "preferred": _spec(type="int", required=True),
        "max": _spec(type="int", required=True),
    }),
    "fortification": _spec(type="object", required=True, keys={
        "mode": _spec(type="str", required=True, nonempty=True),
        "wall_profile": _spec(type="str", required=False, nonempty=True),
    }),
    "morphology": _spec(type="str", required=True, values=MORPHOLOGY),
    "anchors": _spec(type="object", required=True, keys={
        "market": _spec(type="str", required=True, values=ANCHOR_PRESENCE),
        "keep": _spec(type="str", required=True, values=ANCHOR_PRESENCE),
        "temple": _spec(type="str", required=True, values=ANCHOR_PRESENCE),
    }),
    "ward_mix": _spec(type="object", required=True, keys={
        "market": _spec(type="num", required=True),
        "craft": _spec(type="num", required=True),
        "residential": _spec(type="num", required=True),
        "outskirts": _spec(type="num", required=True),
    }),
    "design_constraints": _spec(type="object", required=True, keys={}),
    # Optional per-zone library/density policy.  When omitted, stages use
    # their built-in defaults (inner: markarth/dense, outer: karthgad/low).
    "development_policy": _spec(type="object", required=False, keys={
        "inner": _spec(type="object", required=True, keys={
            "preferred_library": _spec(type="str", required=True, values=("markarth", "karthgad")),
            "density": _spec(type="str", required=True, values=("dense", "low")),
            "house_generator": _spec(type="str", required=False, values=("fk_house",)),
        }),
        "outer": _spec(type="object", required=True, keys={
            "preferred_library": _spec(type="str", required=True, values=("markarth", "karthgad")),
            "density": _spec(type="str", required=True, values=("dense", "low")),
            "house_generator": _spec(type="str", required=False, values=("fk_house",)),
        }),
    }),
    # Optional plan-GU pin.  When omitted, Phase 3 picks max suitability.
    "pin_plan_gu": _spec(type="point", required=False),
    # Optional toggles (when omitted, defaults apply: inner wall follows
    # fortification.mode != "none", outskirts defaults true).
    "has_inner_wall": _spec(type="bool", required=False),
    "has_outskirts": _spec(type="bool", required=False),
}

TOWN_BRIEF_SPEC = {"type": "object", "keys": TOWN_BRIEF_KEYS}

# ---------------------------------------------------------------------------
# CityLayout nested items
# ---------------------------------------------------------------------------

_TERRAIN_SUMMARY = {
    "mean_slope_deg": _spec(type="num", required=True),
    "water": _spec(type="bool", required=True),
}

_SUITABILITY_GRID = {
    "origin_plan_gu": _spec(type="point", required=True),
    "spacing_gu": _spec(type="num", required=True),
    "nx": _spec(type="int", required=True),
    "ny": _spec(type="int", required=True),
    "values": _spec(type="array", required=True, item=_spec(type="num")),
}

_ESTIMATED_URBAN = {
    "min": _spec(type="num", required=True),
    "preferred": _spec(type="num", required=True),
    "max": _spec(type="num", required=True),
}

_STAMP_FOOTPRINT_STATS = {
    "source": _spec(type="str", required=True, values=(STAMP_STATS_SOURCE,)),
    "p10": _spec(type="num", required=True),
    "p50": _spec(type="num", required=True),
    "p90": _spec(type="num", required=True),
    "parcel_yard_factor": _spec(type="num", required=True),
    "urban_space_factor": _spec(type="num", required=True),
}

_FRAME = {
    "origin_gu": _spec(type="point", required=True),
    "units": _spec(type="str", required=True, values=("game_units",)),
    "yaw_convention": _spec(type="str", required=True, values=(YAW_CONVENTION,)),
    "site_survey_sha256": _spec(type="str", required=True, nonempty=True,
                                pattern=SHA256_RE),
    "coordinate_space": _spec(type="str", required=True,
                              values=(COORDINATE_SPACE,)),
}

_SITE_CONTEXT = {
    "site_id": _spec(type="str", required=True, nonempty=True),
    "span_gu": _spec(type="point", required=True),
    "hard_exclusion_polygons": _spec(
        type="array", required=True,
        item=_spec(type="points", min_points=3),
    ),
    # Optional for compatibility with pre-Stage-01 layout fixtures; Stage 01
    # SiteContext products always emit this explicit water geometry.
    "water_polygons": _spec(
        type="array", required=False,
        item=_spec(type="points", min_points=3),
    ),
    "suitability_grid": _spec(type="object", required=True,
                              keys=_SUITABILITY_GRID),
    "estimated_urban_area_gu2": _spec(type="object", required=True,
                                      keys=_ESTIMATED_URBAN),
    "candidate_centers": _spec(type="array", required=True,
                               item=_spec(type="point")),
    "stamp_footprint_stats": _spec(type="object", required=True,
                                   keys=_STAMP_FOOTPRINT_STATS),
}

_WALL = {
    "kind": _spec(type="str", required=True, values=("palisade",)),
    "planning_polygon": _spec(type="points", required=True, min_points=4),
    "source_perimeter": _spec(type="points", required=True, min_points=4),
    "segments": _spec(type="array", required=True, min_items=1, item="wall_segment"),
    "strips": _spec(type="array", required=True, min_items=1, item="wall_strip"),
}

_WALL_SEGMENT = {
    "wall_segment_id": _spec(type="str", required=True, nonempty=True),
    "ring": _spec(type="points", required=True, min_points=2),
    "start_gate_id": _spec(type="str", required=True, nonempty=True, nullable=True),
    "end_gate_id": _spec(type="str", required=True, nonempty=True, nullable=True),
}

_WALL_STRIP = {
    "strip_id": _spec(type="str", required=True, nonempty=True),
    "wall_segment_id": _spec(type="str", required=True, nonempty=True),
    "mode": _spec(type="str", required=True, values=("backs_to_wall", "wall_lane")),
    "polygon": _spec(type="points", required=True, min_points=3),
    "road_ids": _spec(type="array", required=True, item=_spec(type="str", nonempty=True)),
    "declared_depth_gu": _spec(type="num", required=True),
    "depth_supported": _spec(type="bool", required=True),
    "arterial_occupancy_beyond_gate": _spec(type="bool", required=True),
    "short_run": _spec(type="bool", required=True),
}

_PORT = {
    "port_id": _spec(type="str", required=True, nonempty=True),
    "approach_id": _spec(type="str", required=True, nonempty=True),
    "source_edge_id": _spec(type="str", required=True, nonempty=True),
    "position": _spec(type="point", required=True),
    "inward_tangent": _spec(type="point", required=True),
    "protected": _spec(type="bool", required=True),
}

_REWRITE_DOMAIN = {
    "polygon": _spec(type="points", required=True, min_points=3),
    "role": _spec(type="str", required=False),
    "search_clearance_gu": _spec(type="num", required=False),
}

_FRONTAGE = {
    "target_id": _spec(type="str", required=True, nonempty=True),
    "target_type": _spec(type="str", required=True,
                         values=FRONTAGE_TARGET_TYPES),
    "target_arc_start_gu": _spec(type="num", required=True),
    "target_arc_end_gu": _spec(type="num", required=True),
    "frontage_length_gu": _spec(type="num", required=True),
}

ITEM_SPEC = {
    "patch": {
        "patch_id": _spec(type="str", required=True, nonempty=True),
        "polygon": _spec(type="points", required=True, min_points=3),
        "neighbour_patch_ids": _spec(type="array", required=True,
                                     item=_spec(type="str", nonempty=True)),
        "inside_city": _spec(type="bool", required=True),
        "inside_wall": _spec(type="bool", required=True),
        "morphology_region": _spec(type="str", required=True, nonempty=True),
        "terrain_summary": _spec(type="object", required=True,
                                 keys=_TERRAIN_SUMMARY),
    },
    "boundary_edge": {
        "edge_id": _spec(type="str", required=True, nonempty=True),
        "a_node": _spec(type="str", required=True, nonempty=True),
        "b_node": _spec(type="str", required=True, nonempty=True),
        "geometry": _spec(type="points", required=True, min_points=2),
        "patch_left": _spec(type="str", required=True, nonempty=True,
                            nullable=True),
        "patch_right": _spec(type="str", required=True, nonempty=True,
                             nullable=True),
        "edge_role": _spec(type="str", required=True, values=EDGE_ROLES),
        "road_class": _spec(type="str", required=True,
                            values=BOUNDARY_ROAD_CLASSES),
    },
    "road_node": {
        "node_id": _spec(type="str", required=True, nonempty=True),
        "position": _spec(type="point", required=True),
        "kind": _spec(type="str", required=True, values=NODE_KINDS),
    },
    "road_edge": {
        "road_id": _spec(type="str", required=True, nonempty=True),
        "node_a": _spec(type="str", required=True, nonempty=True),
        "node_b": _spec(type="str", required=True, nonempty=True),
        "polyline": _spec(type="points", required=True, min_points=2),
        "hierarchy": _spec(type="str", required=True, values=ROAD_HIERARCHY),
        "clear_width_gu": _spec(type="num", required=True),
        "paint_surface": _spec(type="str", required=True,
                               values=PAINT_SURFACES),
        "source_edge_ids": _spec(type="array", required=True,
                                 item=_spec(type="str", nonempty=True)),
        "boundary_edge_ids": _spec(type="array", required=True,
                                     item=_spec(type="str", nonempty=True)),
        "routing_edge_ids": _spec(type="array", required=False,
                                   item=_spec(type="str", nonempty=True)),
        "source_approach_ids": _spec(type="array", required=False,
                                      item=_spec(type="str", nonempty=True)),
        "tangent_handle_gu": _spec(type="num", required=False),
        "tangent_residual_deg": _spec(type="num", required=False),
        "max_curvature_deg_per_256gu": _spec(type="num", required=False),
    },
    "routing_edge": {
        "routing_edge_id": _spec(type="str", required=True, nonempty=True),
        "a_node": _spec(type="str", required=True, nonempty=True),
        "b_node": _spec(type="str", required=True, nonempty=True),
        "geometry": _spec(type="points", required=True, min_points=2),
        "kind": _spec(type="str", required=True,
                       values=("port_handle", "interior", "boundary")),
        "boundary_edge_ids": _spec(type="array", required=True,
                                    item=_spec(type="str", nonempty=True)),
        "source_patch_ids": _spec(type="array", required=True,
                                   item=_spec(type="str", nonempty=True)),
        "cost": _spec(type="num", required=False),
    },
    "ward": {
        "ward_id": _spec(type="str", required=True, nonempty=True),
        "ward_type": _spec(type="str", required=True, values=WARD_TYPES),
        "patch_ids": _spec(type="array", required=True, min_items=1,
                           item=_spec(type="str", nonempty=True)),
        "dplan_kind": _spec(type="str", required=True, values=DPLAN_KINDS),
        "score_evidence": _spec(type="object", required=True, keys={}),
    },
    "anchor": {
        "anchor_id": _spec(type="str", required=True, nonempty=True),
        "kind": _spec(type="str", required=True, values=ANCHOR_KINDS),
        "patch_id": _spec(type="str", required=True, nonempty=True),
        "polygon": _spec(type="points", required=True, min_points=3),
    },
    "gate": {
        "gate_id": _spec(type="str", required=True, nonempty=True),
        "position": _spec(type="point", required=True),
        "approach_id": _spec(type="str", required=True, nonempty=True),
        "outward_tangent": _spec(type="point", required=True),
        "road_node_id": _spec(type="str", required=True, nonempty=True),
    },
    "source_approach": {
        "approach_id": _spec(type="str", required=True, nonempty=True),
        "source_edge_id": _spec(type="str", required=True, nonempty=True),
        "crossing_plan_gu": _spec(type="point", required=True),
        "inward_tangent": _spec(type="point", required=True),
        "mandatory": _spec(type="bool", required=True),
        "outside_polyline_plan_gu": _spec(type="points", required=True,
                                           min_points=2),
        "inside_polyline_plan_gu": _spec(type="points", required=False,
                                          min_points=2),
        "transition_stub_plan_gu": _spec(type="points", required=True,
                                         min_points=2),
    },
    "open_space": {
        "space_id": _spec(type="str", required=True, nonempty=True),
        "kind": _spec(type="str", required=True, values=OPEN_SPACE_KINDS),
        "polygon": _spec(type="points", required=True, min_points=3),
    },
    "frontage": _FRONTAGE,
    "wall_segment": {"type": "object", "keys": _WALL_SEGMENT},
    "wall_strip": {"type": "object", "keys": _WALL_STRIP},
    "parcel": {
        "parcel_id": _spec(type="str", required=True, nonempty=True),
        "ward_id": _spec(type="str", required=True, nonempty=True),
        "polygon": _spec(type="points", required=True, min_points=3),
        "frontages": _spec(type="array", required=True, item="frontage"),
        "required_occupancy": _spec(type="bool", required=True),
        "allowed_roles": _spec(type="array", required=True,
                               item=_spec(type="str", nonempty=True)),
    },
    "placement": {
        "parcel_id": _spec(type="str", required=True, nonempty=True),
        "stamp_id": _spec(type="str", required=True, nonempty=True,
                          nullable=True),
        "anchor": _spec(type="point", required=True, nullable=True),
        "yaw_deg": _spec(type="num", required=True, nullable=True),
        # Stage 07 extensions are optional here so the pre-Stage-07 CityLayout
        # contract remains readable; validate_stage07 requires them.
        "block_id": _spec(type="str", required=False, nonempty=True),
        "frontage_road_id": _spec(type="str", required=False, nonempty=True),
        "side": _spec(type="str", required=False, values=("left", "right")),
        "mode": _spec(type="str", required=False, values=("frontage", "depth", "rear", "backs_to_wall")),
        "setback_gu": _spec(type="num", required=False),
        "family": _spec(type="str", required=False, values=("stone", "wood", "outskirts")),
        "size_class": _spec(type="str", required=False, values=("small", "medium", "large")),
        "hull": _spec(type="points", required=False, min_points=3),
        "door_world": _spec(type="point", required=False, nullable=True),
        "outward_tick": _spec(type="point", required=False, nullable=True),
        "terrain_evidence": _spec(type="object", required=False, keys={}, dynamic=True),
        "paired_front_id": _spec(type="str", required=False, nonempty=True),
        "paired_gap_gu": _spec(type="num", required=False),
        "wall_segment_id": _spec(type="str", required=False, nonempty=True),
    },
    "report": {
        "stage": _spec(type="str", required=True, nonempty=True),
        "status": _spec(type="str", required=True, values=REPORT_STATUSES),
        "message": _spec(type="str", required=True),
    },
}

# Stage 07 is a compact product (not a CityLayout): these explicit item
# contracts prevent the tempting private top-level evidence dictionaries.
STAGE07_ITEM_SPEC = {
    "frontage_inventory": {"road_id": _spec(type="str", required=True, nonempty=True), "side": _spec(type="str", required=True, values=("left", "right")), "block_id": _spec(type="str", required=True, nonempty=True), "required": _spec(type="bool", required=True), "usable_length_gu": _spec(type="num", required=True), "covered_length_gu": _spec(type="num", required=True), "arc_start_gu": _spec(type="num", required=True), "arc_end_gu": _spec(type="num", required=True), "hierarchy": _spec(type="str", required=True), "ward_type": _spec(type="str", required=True)},
    "provisional_parcel": {"parcel_id": _spec(type="str", required=True, nonempty=True), "block_id": _spec(type="str", required=True, nonempty=True), "frontage_arc": _spec(type="object", required=True, keys={"road_id": _spec(type="str", required=True), "side": _spec(type="str", required=True, values=("left", "right")), "start": _spec(type="num", required=True), "end": _spec(type="num", required=True)}), "intended_family": _spec(type="str", required=True), "placed_stamp_id": _spec(type="str", required=True), "alternate_stamp_ids": _spec(type="array", required=True, item=_spec(type="str")), "polygon": _spec(type="points", required=True, min_points=3)},
    "placement_neighbourhood": {"edges": _spec(type="array", required=True, item={"a": _spec(type="str", required=True), "b": _spec(type="str", required=True), "kind": _spec(type="str", required=True, values=("consecutive_frontage", "front_rear", "hull_near"))})},
    "population_metrics": {"population": _spec(type="int", required=True), "parcel_count": _spec(type="int", required=True), "required_arterial_sides": _spec(type="int", required=True), "covered_arterial_sides": _spec(type="int", required=True), "usable_frontage_gu": _spec(type="num", required=True), "covered_frontage_gu": _spec(type="num", required=True), "required_coverage_pct": _spec(type="num", required=True), "front_count": _spec(type="int", required=True), "paired_rear_count": _spec(type="int", required=True), "wall_count": _spec(type="int", required=True), "gate_blockage_count": _spec(type="int", required=True), "collision_count": _spec(type="int", required=True), "water_overlap_count": _spec(type="int", required=True), "rejections": _spec(type="object", required=True, keys={}, dynamic=True), "deterministic_seed": _spec(type="int", required=True)},
}

CITY_LAYOUT_KEYS = {
    "schema_version": _spec(type="int", required=True),
    "brief_provenance": _spec(type="object", required=False, keys={
        "town_id": _spec(type="str", required=True, nonempty=True),
        "target_buildings": _spec(type="object", required=True, keys={
            "min": _spec(type="int", required=True),
            "preferred": _spec(type="int", required=True),
            "max": _spec(type="int", required=True),
        }),
        "sha256": _spec(type="str", required=True, nonempty=True),
    }),
    "layout_id": _spec(type="str", required=True, nonempty=True),
    "candidate_id": _spec(type="str", required=True, nonempty=True,
                          pattern=CANDIDATE_ID_RE),
    "frame": _spec(type="object", required=True, keys=_FRAME),
    "town_brief": _spec(type="object", required=True, keys=TOWN_BRIEF_KEYS),
    "site_context": _spec(type="object", required=True, keys=_SITE_CONTEXT),
    "patches": _spec(type="array", required=True, item="patch"),
    "boundary_edges": _spec(type="array", required=True, item="boundary_edge"),
    "nodes": _spec(type="array", required=True, item="road_node"),
    "routing_edges": _spec(type="array", required=False, item="routing_edge"),
    "roads": _spec(type="array", required=True, item="road_edge"),
    "wards": _spec(type="array", required=True, item="ward"),
    "anchors": _spec(type="array", required=True, item="anchor"),
    "wall": _spec(type="object", required=True, nullable=True, keys=_WALL),
    "gates": _spec(type="array", required=True, item="gate"),
    "ports": _spec(type="array", required=False, item=_PORT),
    "provisional_ring": _spec(type="points", required=False, min_points=3),
    "approaches": _spec(type="array", required=True, item="source_approach"),
    "rewrite_domain": _spec(type="object", required=True,
                            keys=_REWRITE_DOMAIN),
    "open_spaces": _spec(type="array", required=True, item="open_space"),
    "parcels": _spec(type="array", required=True, item="parcel"),
    "placements": _spec(type="array", required=True, item="placement"),
    "reports": _spec(type="array", required=True, item="report"),
}

CITY_LAYOUT_SPEC = {"type": "object", "keys": CITY_LAYOUT_KEYS}
SITE_CONTEXT_SPEC = {"type": "object", "keys": _SITE_CONTEXT}


def _schema_type(spec: dict) -> dict:
    if spec.get("nullable"):
        base = dict(spec)
        base["nullable"] = False
        return {"anyOf": [_schema_type(base), {"type": "null"}]}
    kind = spec["type"]
    if kind == "object":
        return {
            "type": "object",
            "additionalProperties": bool(spec.get("dynamic")),
            "required": sorted(k for k, s in spec.get("keys", {}).items()
                               if s.get("required")),
            "properties": {k: _schema_type(s)
                           for k, s in spec.get("keys", {}).items()},
        }
    if kind == "array":
        out: dict[str, Any] = {"type": "array"}
        if spec.get("min_items"):
            out["minItems"] = spec["min_items"]
        out["items"] = _schema_type(_resolve_item_spec(spec.get("item")))
        return out
    if kind == "str":
        out = {"type": "string"}
        if spec.get("values"):
            out["enum"] = list(spec["values"])
        if spec.get("nonempty"):
            out["minLength"] = 1
        pattern = spec.get("pattern")
        if pattern is not None:
            out["pattern"] = pattern.pattern
        return out
    if kind == "int":
        return {"type": "integer"}
    if kind == "num":
        return {"type": "number"}
    if kind == "bool":
        return {"type": "boolean"}
    if kind == "point":
        return {
            "type": "array",
            "prefixItems": [{"type": "number"}, {"type": "number"}],
            "items": False, "minItems": 2, "maxItems": 2,
        }
    if kind == "points":
        out = {
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [{"type": "number"}, {"type": "number"}],
                "items": False, "minItems": 2, "maxItems": 2,
            },
        }
        if spec.get("min_points"):
            out["minItems"] = spec["min_points"]
        return out
    raise RuntimeError(f"internal schema error: unknown type {kind!r}")


def emit_json_schema() -> dict:
    """Simplified JSON Schema (draft 2020-12) for TownBrief and CityLayout."""
    defs = {
        "TownBrief": _schema_type(TOWN_BRIEF_SPEC),
        "CityLayout": _schema_type(CITY_LAYOUT_SPEC),
    }
    for name, keys in ITEM_SPEC.items():
        if isinstance(keys, dict) and "type" not in keys:
            defs[name] = _schema_type({"type": "object", "keys": keys})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "cityforge/townlayout/town_layout_schema_v1.json",
        "title": "Cityforge V2 townlayout TownBrief + CityLayout schema v1",
        "description": (
            "Strict machine-readable contract for town_brief.json and "
            "city_layout.json. additionalProperties:false applies recursively; "
            "semantic gates are enforced by src/procgen/townlayout/validate.py. "
            "V2 townlayout Phase 1 contracts; no generation."
        ),
        "oneOf": [
            {"$ref": "#/$defs/TownBrief"},
            {"$ref": "#/$defs/CityLayout"},
        ],
        "$defs": defs,
    }
