"""Semantic and cross-reference validation for V2 townlayout documents.

Purpose
-------
Validate a loaded TownBrief or CityLayout dict against Phase 1 contracts:
structural walk (via ``schema.py``), polygon winding, ID uniqueness,
paint-surface / ward mapping locks, and reference integrity.

Inputs
------
A JSON-loaded dict (the caller owns the object).  This module does not
rewrite coordinates, fill defaults, or reorder keys.

Outputs
-------
``(document, issues)`` where ``issues`` is a list of
``{severity, code, path, message}``.  ``document`` is the same object
passed in.

Pipeline position
-----------------
V2 townlayout Phase 1 contracts; no generation.
"""

from __future__ import annotations

import math
from typing import Any
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from procgen.cityplan import close_ring, point_in_ring, ring_area

from .schema import (
    CITY_LAYOUT_SPEC,
    COORDINATE_SPACE,
    FORTIFICATION_MODES,
    SITE_CONTEXT_SPEC,
    STAGE07_ITEM_SPEC,
    ITEM_SPEC,
    _WALL,
    TOWN_BRIEF_SPEC,
    check_structure,
    issue,
    json_path,
)

# Re-export the allowed cityplan predicates so polygon checks stay in one place.
_ = point_in_ring

WARD_DPLAN_KIND = {
    "market": "market",
    "craft": "craft",
    "residential": "residential",
    "outskirts": "outskirts",
    "keep": "keep",
}

PAINT_BY_HIERARCHY = {
    "regional_approach": "road",
    "arterial": "road",
    "street": "road",
    "lane": "settlement_dirt",
    "alley": "none",
}

UNAVAILABLE_FORTIFICATIONS = ("wall", "stone_wall", "fence")
AREA_EPS = 1e-6
VERTEX_EPS = 1e-6
WARD_MIX_EPS = 1e-6
SCHEMA_VERSION = 1


class TownLayoutError(ValueError):
    """Fail-closed contract error. str(exc) is a single-line reason."""


def _open_ring(points: list) -> list[tuple[float, float]]:
    closed = close_ring(points)
    if len(closed) >= 2:
        return closed[:-1]
    return closed


def _same_point(a: tuple[float, float], b: tuple[float, float],
                eps: float = VERTEX_EPS) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def _check_polygon(points: Any, path: str, issues: list, *,
                   min_points: int) -> None:
    if not isinstance(points, list) or len(points) < min_points:
        issues.append(issue("polygon_invalid", path,
                            f"polygon needs >= {min_points} points"))
        return
    for idx, pt in enumerate(points):
        if (not isinstance(pt, list) or len(pt) != 2
                or not isinstance(pt[0], (int, float))
                or not isinstance(pt[1], (int, float))
                or isinstance(pt[0], bool) or isinstance(pt[1], bool)):
            issues.append(issue("polygon_invalid", json_path(path, f"[{idx}]"),
                                "polygon vertex is not a finite [x, y]"))
            return
        if isinstance(pt[0], float) and not math.isfinite(pt[0]):
            issues.append(issue("non_finite_number",
                                json_path(path, f"[{idx}][0]"),
                                "number is NaN or infinite"))
            return
        if isinstance(pt[1], float) and not math.isfinite(pt[1]):
            issues.append(issue("non_finite_number",
                                json_path(path, f"[{idx}][1]"),
                                "number is NaN or infinite"))
            return
    ring = _open_ring(points)
    if len(ring) < min_points:
        issues.append(issue("polygon_invalid", path,
                            f"polygon needs >= {min_points} open-ring vertices"))
        return
    for i in range(len(ring) - 1):
        if _same_point(ring[i], ring[i + 1]):
            issues.append(issue("polygon_invalid", path,
                                "consecutive duplicate vertices"))
            return
    area = ring_area(points)
    if area <= AREA_EPS:
        issues.append(issue("polygon_invalid", path,
                            f"polygon must be CCW with area > {AREA_EPS}, "
                            f"got {area}"))


def _check_schema_version(document: dict, path: str, issues: list) -> None:
    version = document.get("schema_version")
    if isinstance(version, int) and not isinstance(version, bool):
        if version != SCHEMA_VERSION:
            issues.append(issue("bad_schema_version",
                                json_path(path, "schema_version"),
                                f"schema_version must be {SCHEMA_VERSION}"))


def _check_town_brief_semantics(brief: dict, path: str, issues: list) -> None:
    _check_schema_version(brief, path, issues)
    targets = brief.get("target_buildings")
    if isinstance(targets, dict):
        lo, pref, hi = targets.get("min"), targets.get("preferred"), targets.get("max")
        if all(isinstance(v, int) and not isinstance(v, bool)
               for v in (lo, pref, hi)):
            if lo < 1:
                issues.append(issue("target_buildings_order",
                                    json_path(path, "target_buildings", "min"),
                                    "target_buildings.min must be >= 1"))
            if not (lo <= pref <= hi):
                issues.append(issue("target_buildings_order",
                                    json_path(path, "target_buildings"),
                                    "require min <= preferred <= max"))
    fort = brief.get("fortification")
    if isinstance(fort, dict):
        mode = fort.get("mode")
        if isinstance(mode, str) and mode not in FORTIFICATION_MODES:
            code = ("fortification_mode_unavailable"
                    if mode in UNAVAILABLE_FORTIFICATIONS else "bad_enum")
            issues.append(issue(code, json_path(path, "fortification", "mode"),
                                f"fortification.mode {mode!r} is not available"))
    mix = brief.get("ward_mix")
    if isinstance(mix, dict):
        keys = ("market", "craft", "residential", "outskirts")
        values = []
        ok = True
        for key in keys:
            val = mix.get(key)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                ok = False
                continue
            if isinstance(val, float) and not math.isfinite(val):
                ok = False
                continue
            if val < 0.0 or val > 1.0:
                issues.append(issue("ward_mix_sum",
                                    json_path(path, "ward_mix", key),
                                    "ward_mix weights must be in [0, 1]"))
            values.append(float(val))
        if ok and len(values) == 4:
            total = sum(values)
            if abs(total - 1.0) > WARD_MIX_EPS:
                issues.append(issue("ward_mix_sum",
                                    json_path(path, "ward_mix"),
                                    f"ward_mix must sum to 1.0, got {total}"))


def _unique_ids(items: list, id_key: str, path: str, issues: list) -> set[str]:
    seen: set[str] = set()
    if not isinstance(items, list):
        return seen
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        ident = item.get(id_key)
        if not isinstance(ident, str) or not ident:
            continue
        if ident in seen:
            issues.append(issue("duplicate_id",
                                json_path(path, f"[{idx}]", id_key),
                                f"duplicate {id_key} {ident!r}"))
        else:
            seen.add(ident)
    return seen


def _require_ref(ident: Any, known: set[str], path: str, issues: list,
                 kind: str) -> None:
    if not isinstance(ident, str):
        return
    if ident not in known:
        issues.append(issue("missing_ref", path,
                            f"{kind} {ident!r} is not defined"))


def _nonzero_vec(vec: Any, path: str, issues: list) -> None:
    if not isinstance(vec, list) or len(vec) != 2:
        return
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in vec):
        return
    if any(isinstance(v, float) and not math.isfinite(v) for v in vec):
        return
    if vec[0] == 0 and vec[1] == 0:
        issues.append(issue("zero_tangent", path,
                            "tangent must not be both components 0"))


def _check_city_layout_semantics(document: dict, issues: list) -> None:
    _check_schema_version(document, "$", issues)
    brief = document.get("town_brief")
    if isinstance(brief, dict):
        _check_town_brief_semantics(brief, "$.town_brief", issues)

    frame = document.get("frame")
    if isinstance(frame, dict):
        space = frame.get("coordinate_space")
        if isinstance(space, str) and space != COORDINATE_SPACE:
            issues.append(issue("bad_enum", "$.frame.coordinate_space",
                                f"coordinate_space must be {COORDINATE_SPACE!r}"))

    site = document.get("site_context")
    if isinstance(site, dict):
        grid = site.get("suitability_grid")
        if isinstance(grid, dict):
            nx, ny = grid.get("nx"), grid.get("ny")
            values = grid.get("values")
            spacing = grid.get("spacing_gu")
            if isinstance(spacing, (int, float)) and not isinstance(spacing, bool):
                if spacing != 128.0:
                    issues.append(issue("bad_value",
                                        "$.site_context.suitability_grid.spacing_gu",
                                        "spacing_gu must be 128.0"))
            if isinstance(nx, int) and not isinstance(nx, bool) and nx < 1:
                issues.append(issue("bad_value",
                                    "$.site_context.suitability_grid.nx",
                                    "nx must be >= 1"))
            if isinstance(ny, int) and not isinstance(ny, bool) and ny < 1:
                issues.append(issue("bad_value",
                                    "$.site_context.suitability_grid.ny",
                                    "ny must be >= 1"))
            if (isinstance(values, list)
                    and isinstance(nx, int) and isinstance(ny, int)
                    and not isinstance(nx, bool) and not isinstance(ny, bool)):
                expected = nx * ny
                if len(values) != expected:
                    issues.append(issue("bad_value",
                                        "$.site_context.suitability_grid.values",
                                        f"values length must be nx*ny={expected}"))
                for idx, val in enumerate(values):
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        if isinstance(val, float) and not math.isfinite(val):
                            continue
                        if val < 0.0 or val > 1.0:
                            issues.append(issue(
                                "bad_value",
                                json_path("$.site_context.suitability_grid.values",
                                          f"[{idx}]"),
                                "suitability values must be in [0, 1]"))
        urban = site.get("estimated_urban_area_gu2")
        if isinstance(urban, dict):
            lo, pref, hi = urban.get("min"), urban.get("preferred"), urban.get("max")
            nums = []
            for key, val in (("min", lo), ("preferred", pref), ("max", hi)):
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if isinstance(val, float) and not math.isfinite(val):
                        continue
                    nums.append(float(val))
                    if key == "min" and val <= 0:
                        issues.append(issue(
                            "target_buildings_order",
                            "$.site_context.estimated_urban_area_gu2.min",
                            "estimated_urban_area_gu2.min must be > 0"))
                else:
                    nums = None
                    break
            if nums is not None and len(nums) == 3 and not (nums[0] <= nums[1] <= nums[2]):
                issues.append(issue(
                    "target_buildings_order",
                    "$.site_context.estimated_urban_area_gu2",
                    "require min <= preferred <= max"))
        stats = site.get("stamp_footprint_stats")
        if isinstance(stats, dict):
            for key, expected in (("parcel_yard_factor", 1.8),
                                  ("urban_space_factor", 1.6)):
                val = stats.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if val != expected:
                        issues.append(issue(
                            "bad_value",
                            json_path("$.site_context.stamp_footprint_stats", key),
                            f"{key} must be {expected}"))
        exclusions = site.get("hard_exclusion_polygons")
        if isinstance(exclusions, list):
            for idx, poly in enumerate(exclusions):
                _check_polygon(
                    poly,
                    json_path("$.site_context.hard_exclusion_polygons",
                              f"[{idx}]"),
                    issues, min_points=3)

    patches = document.get("patches") if isinstance(document.get("patches"), list) else []
    edges = document.get("boundary_edges") if isinstance(document.get("boundary_edges"), list) else []
    routing_edges = document.get("routing_edges") if isinstance(document.get("routing_edges"), list) else []
    nodes = document.get("nodes") if isinstance(document.get("nodes"), list) else []
    roads = document.get("roads") if isinstance(document.get("roads"), list) else []
    wards = document.get("wards") if isinstance(document.get("wards"), list) else []
    anchors = document.get("anchors") if isinstance(document.get("anchors"), list) else []
    gates = document.get("gates") if isinstance(document.get("gates"), list) else []
    approaches = document.get("approaches") if isinstance(document.get("approaches"), list) else []
    parcels = document.get("parcels") if isinstance(document.get("parcels"), list) else []
    placements = document.get("placements") if isinstance(document.get("placements"), list) else []
    open_spaces = document.get("open_spaces") if isinstance(document.get("open_spaces"), list) else []

    patch_ids = _unique_ids(patches, "patch_id", "$.patches", issues)
    edge_ids = _unique_ids(edges, "edge_id", "$.boundary_edges", issues)
    routing_ids = _unique_ids(routing_edges, "routing_edge_id", "$.routing_edges", issues)
    node_ids = _unique_ids(nodes, "node_id", "$.nodes", issues)
    _unique_ids(roads, "road_id", "$.roads", issues)
    _unique_ids(wards, "ward_id", "$.wards", issues)
    _unique_ids(anchors, "anchor_id", "$.anchors", issues)
    _unique_ids(gates, "gate_id", "$.gates", issues)
    approach_ids = _unique_ids(approaches, "approach_id", "$.approaches", issues)
    _unique_ids(parcels, "parcel_id", "$.parcels", issues)
    _unique_ids(open_spaces, "space_id", "$.open_spaces", issues)

    for idx, patch in enumerate(patches):
        if not isinstance(patch, dict):
            continue
        _check_polygon(patch.get("polygon"),
                       json_path("$.patches", f"[{idx}]", "polygon"),
                       issues, min_points=3)
        neighbours = patch.get("neighbour_patch_ids")
        if isinstance(neighbours, list):
            for nidx, nid in enumerate(neighbours):
                _require_ref(nid, patch_ids,
                             json_path("$.patches", f"[{idx}]",
                                       "neighbour_patch_ids", f"[{nidx}]"),
                             issues, "patch_id")

    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue
        _require_ref(edge.get("a_node"), node_ids,
                     json_path("$.boundary_edges", f"[{idx}]", "a_node"),
                     issues, "node_id")
        _require_ref(edge.get("b_node"), node_ids,
                     json_path("$.boundary_edges", f"[{idx}]", "b_node"),
                     issues, "node_id")
        for field in ("patch_left", "patch_right"):
            ref = edge.get(field)
            if ref is None:
                continue
            _require_ref(ref, patch_ids,
                         json_path("$.boundary_edges", f"[{idx}]", field),
                         issues, "patch_id")

    for idx, edge in enumerate(routing_edges):
        if not isinstance(edge, dict):
            continue
        if edge.get("a_node") == edge.get("b_node"):
            issues.append(issue("self_edge",
                                json_path("$.routing_edges", f"[{idx}]"),
                                "routing edges must connect distinct nodes"))
        for field in ("a_node", "b_node"):
            _require_ref(edge.get(field), node_ids,
                         json_path("$.routing_edges", f"[{idx}]", field),
                         issues, "node_id")
        for bidx, bid in enumerate(edge.get("boundary_edge_ids") or []):
            _require_ref(bid, edge_ids,
                         json_path("$.routing_edges", f"[{idx}]",
                                   "boundary_edge_ids", f"[{bidx}]"),
                         issues, "edge_id")

    for idx, road in enumerate(roads):
        if not isinstance(road, dict):
            continue
        _require_ref(road.get("node_a"), node_ids,
                     json_path("$.roads", f"[{idx}]", "node_a"),
                     issues, "node_id")
        _require_ref(road.get("node_b"), node_ids,
                     json_path("$.roads", f"[{idx}]", "node_b"),
                     issues, "node_id")
        hierarchy = road.get("hierarchy")
        paint = road.get("paint_surface")
        expected = PAINT_BY_HIERARCHY.get(hierarchy)
        if expected is not None and isinstance(paint, str) and paint != expected:
            issues.append(issue("paint_surface_mismatch",
                                json_path("$.roads", f"[{idx}]", "paint_surface"),
                                f"{hierarchy} requires paint_surface {expected!r}"))
        width = road.get("clear_width_gu")
        if isinstance(width, (int, float)) and not isinstance(width, bool):
            if not (isinstance(width, float) and not math.isfinite(width)):
                if width <= 0:
                    issues.append(issue("bad_value",
                                        json_path("$.roads", f"[{idx}]",
                                                  "clear_width_gu"),
                                        "clear_width_gu must be > 0"))
        bids = road.get("boundary_edge_ids")
        if isinstance(bids, list):
            for bidx, bid in enumerate(bids):
                _require_ref(bid, edge_ids,
                             json_path("$.roads", f"[{idx}]",
                                       "boundary_edge_ids", f"[{bidx}]"),
                              issues, "edge_id")
        rids = road.get("routing_edge_ids")
        if isinstance(rids, list):
            for ridx, rid in enumerate(rids):
                _require_ref(rid, routing_ids,
                             json_path("$.roads", f"[{idx}]",
                                       "routing_edge_ids", f"[{ridx}]"),
                             issues, "routing_edge_id")
        source_ids = road.get("source_approach_ids")
        if isinstance(source_ids, list):
            for sidx, sid in enumerate(source_ids):
                _require_ref(sid, approach_ids,
                             json_path("$.roads", f"[{idx}]",
                                       "source_approach_ids", f"[{sidx}]"),
                             issues, "approach_id")
        residual = road.get("tangent_residual_deg")
        if isinstance(residual, (int, float)) and residual > 5.0:
            issues.append(issue("tangent_mismatch",
                                json_path("$.roads", f"[{idx}]",
                                          "tangent_residual_deg"),
                                "road tangent residual exceeds 5 degrees"))
        curvature = road.get("max_curvature_deg_per_256gu")
        if isinstance(curvature, (int, float)) and curvature > 15.0:
            issues.append(issue("curvature_cap",
                                json_path("$.roads", f"[{idx}]",
                                          "max_curvature_deg_per_256gu"),
                                "road curvature exceeds 15 degrees per 256 GU"))

    for idx, ward in enumerate(wards):
        if not isinstance(ward, dict):
            continue
        pids = ward.get("patch_ids")
        if isinstance(pids, list):
            for pidx, pid in enumerate(pids):
                _require_ref(pid, patch_ids,
                             json_path("$.wards", f"[{idx}]",
                                       "patch_ids", f"[{pidx}]"),
                             issues, "patch_id")
        wtype = ward.get("ward_type")
        dkind = ward.get("dplan_kind")
        expected_kind = WARD_DPLAN_KIND.get(wtype)
        if expected_kind is not None and isinstance(dkind, str) and dkind != expected_kind:
            issues.append(issue("ward_dplan_kind_mismatch",
                                json_path("$.wards", f"[{idx}]", "dplan_kind"),
                                f"ward_type {wtype!r} requires dplan_kind "
                                f"{expected_kind!r}"))

    for idx, anchor in enumerate(anchors):
        if not isinstance(anchor, dict):
            continue
        _require_ref(anchor.get("patch_id"), patch_ids,
                     json_path("$.anchors", f"[{idx}]", "patch_id"),
                     issues, "patch_id")
        _check_polygon(anchor.get("polygon"),
                       json_path("$.anchors", f"[{idx}]", "polygon"),
                       issues, min_points=3)

    for idx, gate in enumerate(gates):
        if not isinstance(gate, dict):
            continue
        _require_ref(gate.get("approach_id"), approach_ids,
                     json_path("$.gates", f"[{idx}]", "approach_id"),
                     issues, "approach_id")
        _nonzero_vec(gate.get("outward_tangent"),
                     json_path("$.gates", f"[{idx}]", "outward_tangent"),
                     issues)

    for idx, approach in enumerate(approaches):
        if not isinstance(approach, dict):
            continue
        _nonzero_vec(approach.get("inward_tangent"),
                     json_path("$.approaches", f"[{idx}]", "inward_tangent"),
                     issues)

    rewrite = document.get("rewrite_domain")
    if isinstance(rewrite, dict):
        _check_polygon(rewrite.get("polygon"),
                       "$.rewrite_domain.polygon", issues, min_points=3)

    wall = document.get("wall")
    mode = None
    if isinstance(brief, dict) and isinstance(brief.get("fortification"), dict):
        mode = brief["fortification"].get("mode")
    if mode == "none" and wall is not None:
        issues.append(issue("wall_mode_mismatch", "$.wall",
                            "fortification.mode none requires wall null"))
    if isinstance(wall, dict):
        _check_polygon(wall.get("planning_polygon"),
                       "$.wall.planning_polygon", issues, min_points=4)
        _check_polygon(wall.get("source_perimeter"),
                       "$.wall.source_perimeter", issues, min_points=4)

    for idx, space in enumerate(open_spaces):
        if not isinstance(space, dict):
            continue
        _check_polygon(space.get("polygon"),
                       json_path("$.open_spaces", f"[{idx}]", "polygon"),
                       issues, min_points=3)

    for idx, parcel in enumerate(parcels):
        if not isinstance(parcel, dict):
            continue
        _check_polygon(parcel.get("polygon"),
                       json_path("$.parcels", f"[{idx}]", "polygon"),
                       issues, min_points=3)
        frontages = parcel.get("frontages")
        if isinstance(frontages, list):
            for fidx, frontage in enumerate(frontages):
                if not isinstance(frontage, dict):
                    continue
                start = frontage.get("target_arc_start_gu")
                end = frontage.get("target_arc_end_gu")
                if (isinstance(start, (int, float)) and isinstance(end, (int, float))
                        and not isinstance(start, bool) and not isinstance(end, bool)):
                    if end < start:
                        issues.append(issue(
                            "bad_value",
                            json_path("$.parcels", f"[{idx}]", "frontages",
                                      f"[{fidx}]", "target_arc_end_gu"),
                            "target_arc_end_gu must be >= target_arc_start_gu"))
                length = frontage.get("frontage_length_gu")
                if isinstance(length, (int, float)) and not isinstance(length, bool):
                    if length < 0:
                        issues.append(issue(
                            "bad_value",
                            json_path("$.parcels", f"[{idx}]", "frontages",
                                      f"[{fidx}]", "frontage_length_gu"),
                            "frontage_length_gu must be >= 0"))

    for idx, placement in enumerate(placements):
        if not isinstance(placement, dict):
            continue
        stamp = placement.get("stamp_id")
        anchor = placement.get("anchor")
        yaw = placement.get("yaw_deg")
        if stamp is None:
            if anchor is not None or yaw is not None:
                issues.append(issue("empty_placement_fields",
                                    json_path("$.placements", f"[{idx}]"),
                                    "null stamp_id requires null anchor and yaw_deg"))
        else:
            if anchor is None or yaw is None:
                issues.append(issue("empty_placement_fields",
                                    json_path("$.placements", f"[{idx}]"),
                                    "occupied placement requires anchor and yaw_deg"))


def validate_town_brief(document: Any) -> tuple[Any, list]:
    """Return ``(document, issues)`` for a TownBrief dict.  Document unchanged."""
    issues: list = []
    if not isinstance(document, dict):
        issues.append(issue("wrong_type", "$",
                            f"expected object, got {type(document).__name__}"))
        return document, issues
    check_structure(document, TOWN_BRIEF_SPEC, "$", issues)
    _check_town_brief_semantics(document, "$", issues)
    return document, issues


def validate_site_context(document: Any) -> tuple[Any, list]:
    """Return ``(document, issues)`` for a Phase 1 ``site_context`` object."""
    issues: list = []
    if not isinstance(document, dict):
        issues.append(issue("wrong_type", "$",
                            f"expected object, got {type(document).__name__}"))
        return document, issues
    check_structure(document, SITE_CONTEXT_SPEC, "$", issues)
    return document, issues


def validate_city_layout(document: Any) -> tuple[Any, list]:
    """Return ``(document, issues)`` for a CityLayout dict.  Document unchanged."""
    issues: list = []
    if not isinstance(document, dict):
        issues.append(issue("wrong_type", "$",
                            f"expected object, got {type(document).__name__}"))
        return document, issues
    check_structure(document, CITY_LAYOUT_SPEC, "$", issues)
    _check_city_layout_semantics(document, issues)
    return document, issues


def validate_fortification_product(document: Any) -> tuple[Any, list]:
    """Validate the compact Stage 06 product emitted from frozen Stage 05 JSON."""
    issues: list = []
    if not isinstance(document, dict):
        return document, [issue("wrong_type", "$", "expected object")]
    wall = document.get("wall")
    if not isinstance(wall, dict):
        if wall is None and document.get("gates") == []:
            return document, issues
        issues.append(issue("wall_invalid", "$.wall", "Stage 06 requires a wall object"))
        return document, issues
    check_structure(wall, {"type": "object", "keys": _WALL}, "$.wall", issues)
    check_structure(document.get("gates"), {"type": "array", "item": {"type": "object", "keys": ITEM_SPEC["gate"]}},
                    "$.gates", issues)
    nodes = {n.get("node_id"): n for n in document.get("nodes", []) if isinstance(n, dict)}
    ring = wall.get("planning_polygon") or []
    boundary = LineString(ring + [ring[0]]) if len(ring) >= 3 else None
    for idx, gate in enumerate(document.get("gates", [])):
        node = nodes.get(gate.get("road_node_id"))
        if node is None:
            issues.append(issue("missing_ref", f"$.gates[{idx}].road_node_id", "road node missing"))
        elif gate.get("position") != node.get("position"):
            issues.append(issue("crossing_node_mismatch", f"$.gates[{idx}].position", "gate is not the exact road node"))
        if boundary is not None and boundary.distance(Point(gate.get("position", [0, 0]))) > 1.0:
            issues.append(issue("gate_off_ring", f"$.gates[{idx}].position", "gate is off ring"))
    segments = wall.get("segments") or []
    strips = wall.get("strips") or []
    segment_ids = [s.get("wall_segment_id") for s in segments]
    strip_segment_ids = [s.get("wall_segment_id") for s in strips]
    if len(set(segment_ids)) != len(segment_ids) or len(set(strip_segment_ids)) != len(strip_segment_ids):
        issues.append(issue("duplicate_id", "$.wall.segments", "wall segment and strip IDs must be unique"))
    if set(segment_ids) != set(strip_segment_ids) or len(strips) != len(segments):
        issues.append(issue("unassigned_wall_arc", "$.wall.strips", "every wall segment needs one strip"))
    if boundary is not None:
        water_overlap = sum(
            boundary_poly.intersection(Polygon(w)).area
            for w in (document.get("water_polygons") or [])
            for boundary_poly in [Polygon(ring)]
        )
        if water_overlap > 128.0 ** 2:
            issues.append(issue("wall_water_intersection", "$.wall.planning_polygon", "wall intersects water"))
        roads = {r.get("road_id"): r for r in document.get("roads", []) if isinstance(r, dict)}
        strip_by_segment = {s.get("wall_segment_id"): s for s in strips if isinstance(s, dict)}
        segment_by_id = {s.get("wall_segment_id"): s for s in segments if isinstance(s, dict)}
        gate_by_id = {g.get("gate_id"): g for g in document.get("gates", []) if isinstance(g, dict)}
        strip_polys = {}
        for idx, strip in enumerate(strips):
            try:
                strip_polys[strip.get("strip_id")] = Polygon(strip.get("polygon") or [])
            except (TypeError, ValueError):
                continue
        strip_items = list(strip_polys.items())
        for i, (sid_a, poly_a) in enumerate(strip_items):
            for sid_b, poly_b in strip_items[i + 1:]:
                if poly_a.is_valid and poly_b.is_valid and poly_a.symmetric_difference(poly_b).area <= 1.0:
                    issues.append(issue("duplicate_wall_strip_geometry", "$.wall.strips", "different strips share identical polygon geometry"))
        for idx, strip in enumerate(strips):
            if strip.get("mode") != "backs_to_wall":
                continue
            segment = segment_by_id.get(strip.get("wall_segment_id")) or {}
            buffers = []
            for gid in (segment.get("start_gate_id"), segment.get("end_gate_id")):
                gate = gate_by_id.get(gid)
                if gate:
                    buffers.append(Point(gate["position"]).buffer(256.0))
            junctions = unary_union(buffers) if buffers else Polygon()
            strip_poly = strip_polys.get(strip.get("strip_id"))
            if strip_poly is None:
                continue
            for road in document.get("roads", []):
                if road.get("hierarchy") != "arterial" or road.get("road_id") not in (strip.get("road_ids") or []):
                    continue
                contact = LineString(road.get("polyline") or []).intersection(strip_poly)
                beyond = contact.difference(junctions) if not junctions.is_empty else contact
                if not beyond.is_empty and beyond.length > 1.0:
                    issues.append(issue("backs_to_wall_arterial_occupancy", f"$.wall.strips[{idx}]", "backs_to_wall strip has arterial occupancy outside gate junction buffers"))
        cursor = None
        initial_cursor = None
        total_reference = 0.0
        for idx, segment in enumerate(segments):
            points = segment.get("ring") or []
            if len(points) < 2:
                continue
            line = LineString(points)
            if any(boundary.distance(Point(p)) > 1.0 for p in points):
                issues.append(issue("segment_off_ring", f"$.wall.segments[{idx}].ring", "segment geometry is off ring"))
                continue
            start = boundary.project(Point(points[0]))
            end = boundary.project(Point(points[-1]))
            if cursor is None:
                cursor = start
                initial_cursor = start
            elif min(abs(start - cursor), boundary.length - abs(start - cursor)) > 1.0:
                issues.append(issue("wall_arc_gap_or_overlap", f"$.wall.segments[{idx}].ring", "ordered segment start is not the previous end"))
            delta = (end - start) % boundary.length
            if delta <= 1e-9:
                delta = boundary.length if len(segments) == 1 else 0.0
            if abs(line.length - delta) > 1.0:
                issues.append(issue("wall_arc_geometry_mismatch", f"$.wall.segments[{idx}].ring", "segment length does not match its ring reference arc"))
            total_reference += delta
            cursor = end
            strip = strip_by_segment.get(segment.get("wall_segment_id"))
            if strip:
                mode = strip.get("mode")
                depth = strip.get("declared_depth_gu")
                if not isinstance(depth, (int, float)) or depth <= 0:
                    issues.append(issue("wall_mode_evidence", f"$.wall.strips[{idx}]", "declared_depth_gu must be positive"))
                if mode == "wall_lane" and depth != 256.0:
                    issues.append(issue("wall_mode_evidence", f"$.wall.strips[{idx}]", "wall_lane must declare a 256 GU strip"))
                if mode == "backs_to_wall" and (strip.get("depth_supported") is not True or strip.get("arterial_occupancy_beyond_gate") is not False):
                    issues.append(issue("wall_mode_evidence", f"$.wall.strips[{idx}]", "backs_to_wall requires supported depth and no non-gate arterial occupancy"))
                for rid in strip.get("road_ids", []):
                    if rid not in roads:
                        issues.append(issue("missing_ref", f"$.wall.strips[{idx}].road_ids", f"road {rid!r} is not defined"))
        if abs(total_reference - boundary.length) > 1.0:
            issues.append(issue("wall_arc_coverage", "$.wall.segments", "ordered segments do not partition the planning ring exactly once"))
        if cursor is not None and initial_cursor is not None and min(abs(cursor - initial_cursor), boundary.length - abs(cursor - initial_cursor)) > 1.0:
            # The final modulo endpoint must return to the initial linear reference.
            issues.append(issue("wall_arc_gap_or_overlap", "$.wall.segments", "segment chain does not close"))
        # Every non-short lane run must expose a gate or a local non-arterial
        # road at each endpoint.  Local modes are allowed to change within a
        # gate-to-gate arc; only connectivity is a contract requirement.
        local_roads = {r.get("road_id"): r for r in document.get("roads", []) if isinstance(r, dict)}
        lane = [strip_by_segment.get(s.get("wall_segment_id")) for s in segments]
        i = 0
        while i < len(lane):
            if not lane[i] or lane[i].get("mode") != "wall_lane":
                i += 1
                continue
            j = i
            while j + 1 < len(lane) and lane[j + 1] and lane[j + 1].get("mode") == "wall_lane":
                j += 1
            if not all(lane[k].get("short_run") is True for k in range(i, j + 1)):
                left_segment, right_segment = segments[i], segments[j]
                left_gate = left_segment.get("start_gate_id")
                right_gate = right_segment.get("end_gate_id")
                left_street = any(local_roads.get(rid, {}).get("hierarchy") in ("regional_approach", "street", "lane", "alley")
                                  for rid in lane[i].get("road_ids", []))
                right_street = any(local_roads.get(rid, {}).get("hierarchy") in ("regional_approach", "street", "lane", "alley")
                                   for rid in lane[j].get("road_ids", []))
                if not (left_gate or left_street):
                    issues.append(issue("wall_lane_disconnected", f"$.wall.strips[{i}]", "lane run lacks a gate/street connection at its start"))
                if not (right_gate or right_street):
                    issues.append(issue("wall_lane_disconnected", f"$.wall.strips[{j}]", "lane run lacks a gate/street connection at its end"))
            i = j + 1
    return document, issues


def validate_stage07_product(document: Any) -> tuple[Any, list]:
    """Strict structural and geometric checks for the Stage 07 checkpoint."""
    issues: list = []
    if not isinstance(document, dict):
        return document, [issue("wrong_type", "$", "expected object")]
    for key, spec in STAGE07_ITEM_SPEC.items():
        value = document.get(key)
        if key == "frontage_inventory":
            if not isinstance(value, list):
                issues.append(issue("missing_required", "$.frontage_inventory", "required Stage 07 inventory")); continue
            for i, row in enumerate(value): check_structure(row, {"type":"object","keys":spec}, f"$.frontage_inventory[{i}]", issues)
        elif key == "provisional_parcel":
            value = document.get("provisional_parcels")
            if not isinstance(value, list):
                issues.append(issue("missing_required", "$.provisional_parcels", "required Stage 07 parcels")); continue
            for i, row in enumerate(value):
                check_structure(row, {"type":"object","keys":spec}, f"$.provisional_parcels[{i}]", issues)
                if isinstance(row, dict): _check_polygon(row.get("polygon"), f"$.provisional_parcels[{i}].polygon", issues, min_points=3)
        else:
            check_structure(document.get(key), {"type":"object","keys":spec}, f"$.{key}", issues)
    placements = document.get("placements") or []
    hulls = []
    for i, p in enumerate(placements):
        check_structure(p, {"type":"object","keys":ITEM_SPEC["placement"]}, f"$.placements[{i}]", issues)
        h = p.get("hull") if isinstance(p, dict) else None
        if h:
            _check_polygon(h, f"$.placements[{i}].hull", issues, min_points=3)
            hulls.append(Polygon(h))
    for i, a in enumerate(hulls):
        for b in hulls[i+1:]:
            if a.intersection(b).area > 1.0:
                issues.append(issue("hull_collision", "$.placements", "accepted hulls overlap"))
    metrics = document.get("population_metrics") or {}
    if isinstance(metrics, dict):
        if metrics.get("population") != len(placements): issues.append(issue("metric_mismatch", "$.population_metrics.population", "population does not equal placements"))
        if metrics.get("parcel_count") != len(document.get("provisional_parcels") or []): issues.append(issue("metric_mismatch", "$.population_metrics.parcel_count", "parcel count does not equal provisional parcels"))
        # These counters describe defects in accepted geometry only.  Rejected
        # attempts belong solely to the rejection histogram.
        accepted_collision = sum(1 for i, a in enumerate(hulls)
                                 for b in hulls[i + 1:]
                                 if a.intersection(b).area > 1.0)
        if metrics.get("collision_count") != accepted_collision:
            issues.append(issue("metric_mismatch", "$.population_metrics.collision_count", "collision metric is not the accepted-hull count"))
        water = [Polygon(w) for w in document.get("water_polygons") or []]
        accepted_water = sum(1 for h in hulls if any(h.intersection(w).area > 1.0 for w in water))
        if metrics.get("water_overlap_count") != accepted_water:
            issues.append(issue("metric_mismatch", "$.population_metrics.water_overlap_count", "water metric is not the accepted-hull count"))
    parcels = document.get("provisional_parcels") or []
    parcel_polys = []
    for i, parcel in enumerate(parcels):
        try:
            poly = Polygon(parcel["polygon"]); parcel_polys.append(poly)
            for prior in parcel_polys[:-1]:
                if poly.intersection(prior).area > 1e-6:
                    issues.append(issue("parcel_overlap", f"$.provisional_parcels[{i}].polygon", "provisional parcels overlap"))
        except (KeyError, TypeError, ValueError):
            continue
    parcel_ids = {p.get("parcel_id") for p in parcels if isinstance(p, dict)}
    placement_by_id = {p.get("parcel_id"): p for p in placements if isinstance(p, dict)}
    for i, placement in enumerate(placements):
        if placement.get("parcel_id") not in parcel_ids:
            issues.append(issue("orphan_placement", f"$.placements[{i}].parcel_id", "accepted placement has no provisional parcel"))
        if not isinstance(placement.get("terrain_evidence"), dict) or not placement["terrain_evidence"].get("available"):
            issues.append(issue("missing_terrain_evidence", f"$.placements[{i}]", "accepted hull lacks independent terrain evidence"))
        parcel = next((q for q in parcels if q.get("parcel_id") == placement.get("parcel_id")), None)
        if parcel is not None:
            try:
                if not Polygon(parcel["polygon"]).buffer(0.01).covers(Polygon(placement["hull"])):
                    issues.append(issue("parcel_not_covering_hull", f"$.placements[{i}]", "parcel does not cover its accepted hull"))
            except (KeyError, TypeError, ValueError):
                pass
        block = next((b for b in document.get("buildable_blocks", [])
                      if b.get("block_id") == placement.get("block_id")), None)
        patch_id = block.get("patch_id") if block else None
        ward = next((w.get("ward_type") for w in document.get("wards", [])
                     if patch_id in (w.get("patch_ids") or [])), None)
        sid = str(placement.get("stamp_id") or "").lower()
        if ward not in (None, "outskirts") and (placement.get("family") == "outskirts" or "hut" in sid):
            issues.append(issue("zoning_violation", f"$.placements[{i}]", "outskirts/hut stamp placed in core ward"))
    # Recompute the local two-hop repeat rule from the emitted graph.
    graph = {}
    for edge in document.get("placement_neighbourhood", {}).get("edges", []):
        graph.setdefault(edge.get("a"), set()).add(edge.get("b"))
        graph.setdefault(edge.get("b"), set()).add(edge.get("a"))
    for pid, placement in placement_by_id.items():
        one = graph.get(pid, set()); near = one | {n for q in one for n in graph.get(q, set())}
        for other in near:
            if other != pid and placement.get("stamp_id") == placement_by_id.get(other, {}).get("stamp_id"):
                issues.append(issue("two_hop_repeat", f"$.placements[{pid}]", "stamp repeats within two graph hops")); break
    # Required means per inventory record, not a global percentage.
    for i, row in enumerate(document.get("frontage_inventory") or []):
        if row.get("required") and float(row.get("covered_length_gu", 0.0)) + 1e-6 < .8 * float(row.get("usable_length_gu", 0.0)):
            issues.append(issue("required_side_coverage", f"$.frontage_inventory[{i}]", "required frontage side is below 80% coverage"))
    # If the producer included frozen-array witnesses, compare them independently
    # rather than trusting a copied boolean or population metric.
    for stage in ("stage05", "stage06"):
        witness = document.get(f"{stage}_frozen_evidence")
        if isinstance(witness, dict):
            for key, expected in witness.get("expected", {}).items():
                if document.get(key) != expected:
                    issues.append(issue("frozen_array_mismatch", f"$.{key}", f"{stage} frozen array differs from evidence"))
    return document, issues
