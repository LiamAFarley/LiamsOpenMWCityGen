"""Cityforge D-PLAN v1: strict city-plan schema, semantic and geometric
validation for ``city_plan.json`` documents (T1.1).

Pipeline position
------------------
Cityforge chain: site survey (T0.2) -> stamp libraries (T0.3) -> kit brief /
region palette (T0.5) -> **this module (T1.1: plan contract + validator)** ->
placement solver (T1.2) -> landscape editor (T1.3) -> authoring (T1.4) ->
review renderer (T1.5) -> first real Falkreath plan (T1.6, user gate).

This module implements the D-PLAN schema v1 contract only.  It never authors
a plan, never invents city layouts, and never performs placement: given one
declarative ``city_plan.json`` plus the accepted planner-input bundle
(site survey, kit brief, region palette, D-STAMP libraries, corrected road
centerlines) it returns a deterministic structured issue list
(``error``/``warning``, code, JSON path, message, measured/limit) plus a
summary and input hashes.  A plan with zero errors may be rendered as an
overlay; a plan with any error is returned to the planner with the full
issue list.  No trusted validated-plan artifact is produced for invalid
plans.

Invariants (binding)
--------------------
- Strict unknown-key rejection at every level (recursive, no exceptions).
- All numbers finite (NaN/Infinity rejected); bool is never a number.
- The plan frame is pinned to the accepted site survey: origin, units,
  yaw-convention text and the exact survey file SHA-256 must match; the
  settlement block must match the survey's seed settlement.
- All plan coordinates are plan-frame GU offsets east/north of the frame
  origin, in [0, 57344) per axis (site span, min inclusive / max exclusive,
  same convention as the survey's tile bounds).
- Building vocabulary comes from the accepted kit brief only: 54 eligible
  stamps, ``building_type_enum``, size classes, capability gaps
  (``lodge`` fails closed).  Surface names come only from
  ``region_palette.semantic_surfaces.surfaces[*].surface`` (closed
  vocabulary).  Stamp footprint geometry is read from the hash-pinned
  D-STAMP libraries, never duplicated as constants.
- Road external connections refer to real aligned-centerline edge/node
  ids (``tamriel_aligned_centerlines_v1.json`` membership) or measured
  map-edge exits computed from the aligned centerlines
  (``exit_<side>_<edge_id>`` ids, measured here by clipping each aligned
  smooth polyline to the site rect).  The aligned product is consumed only
  through ``procgen.aligned_roads`` (the one supported road entry point);
  the source-space bundle and the XCF/BMP are never consumed.  Old
  ``roads_graph_clean.json`` / raw-78-only ``land_roads.json`` geometry is
  never consumed.
- Explicit stamp requests get exact transformed-footprint checks (scope,
  buildable/water coverage, pairwise overlap).  Non-explicit requests are
  resolved by the documented shared deterministic selector below; the
  selected stamp's footprint is then checked exactly and the resolution
  mode is reported per lot.  Geometry is never claimed checked for a stamp
  that was not resolved.
- Boundaries fail closed for every capability gap in the kit brief:
  ``stone_wall`` (unavailable) and ``fence`` (no measured spacing rule) are
  hard errors; measured ``palisade`` may validate.
- Docks are the only water-position exception (feature kind ``dock``).
- Dispatch-5 spacing is measured guidance, never a hard minimum; collision
  geometry (strict polygon overlap) is the only hard spacing rule
  (hard minimum 0.0 GU; touching within the 0.25 GU contact epsilon is
  reported as a warning, matching the extraction contact graph).

Shared deterministic selector (v1, documented)
-----------------------------------------------
For a non-explicit lot request the candidate set is every eligible kit-brief
stamp matching ``building_type``, plus ``size_class`` and ``multi_shell``
when the request constrains them.  The selector picks the candidate with
the smallest ``footprint_hull_area_gu2`` (best fit to a compact lot), ties
broken by sorted ``stamp_id`` -- the same sorted-``stamp_id`` tie-break
rule D-PLACE declares, without D-PLACE's per-lot seeded ranking (that
ranking is solver-stage T1.2 and is not pre-empted here).  The choice is
reported per lot as ``resolution: "selector"`` and is a validator-side
selection only; the planner remains free to make it explicit.

Geometry conventions
--------------------
- Yaw: degrees CCW from +x (east), +y north, applied about the lot's door
  anchor.  Hull point world position = anchor + Rz(yaw) . hull_xy_rel with
  Rz the standard 2D CCW rotation matrix (identical to the D-STAMP/D-PLACE
  member-transform composition for XY; Z is not part of plan geometry).
- Polygons: ``[[x, y], ...]`` plan-frame GU.  District/feature/edit/hint
  polygons are auto-closed (first != last accepted and closed for
  geometry); boundary rings must be explicitly closed (first == last) and
  are validated as simple rings (>= 3 distinct vertices, nonzero signed
  area, no self-intersection).
- Masks: 112x112 uint8 tile grids (512 GU tiles, row-major [y, x], SW
  frame origin) decoded from the site survey; a plan GU point maps to tile
  ``(floor(x/512), floor(y/512))``.  Field spacing is 128 GU
  (``frame.field_spacing_gu``); field indices ``floor(gu/128)`` are
  reported per lot anchor in the summary.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from . import aligned_roads

# ---------------------------------------------------------------------------
# Contract constants (all consumed from the accepted bundle at runtime;
# the values below are structural vocabulary of the schema itself).
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

#: Plan-frame GU span of the accepted Falkreath 7x7 site (survey
#: ``frame.site_span_gu``); min inclusive, max exclusive.
SITE_SPAN_GU = 57344.0
#: Survey tile-grid side (tiles per axis) and tile size in GU.
TILE_SIDE = 112
TILE_SIZE_GU = 512
#: Survey LAND field spacing in GU.
FIELD_SPACING_GU = 128
#: Contact epsilon (GU) from the extraction contact graph / D-PLACE.
CONTACT_EPSILON_GU = 0.25

DISTRICT_KINDS = ("core", "residential", "market", "docks", "farms",
                  "temple", "keep", "craft", "outskirts")
ROAD_CLASSES = ("street", "approach", "path", "dock_lane")
ROAD_GRADE_POLICIES = ("conform", "regrade")
LOT_SIZE_CLASSES = ("small", "medium", "large")
TERRAIN_POLICY_MODES = ("conform", "flatten_pad")
BOUNDARY_KINDS = ("palisade", "fence", "stone_wall")
FEATURE_KINDS = ("well", "statue", "market_stalls", "dock", "boat",
                 "signpost", "keep_trees")
TERRAIN_EDIT_KINDS = ("flatten_shelf", "mound", "terrace", "cut")
WILDERNESS_HINTS = ("dense_forest", "cleared", "meadow", "leave_wild")

#: Capability gaps that fail closed (kit brief ``capability_gaps``).
UNAVAILABLE_BUILDING_TYPES = ("lodge",)
UNAVAILABLE_BOUNDARY_KINDS = ("stone_wall",)
UNAVAILABLE_BOUNDARY_SPACING = ("fence",)

#: Closed surface vocabulary must be loaded from the palette at runtime;
#: this tuple is only the schema's static cross-check (see
#: ``validate_palette_closure`` which always re-derives from the palette).
PALETTE_SURFACES_STATIC = ("base", "settlement_dirt", "settlement_grass_dirt",
                           "settlement_cobble", "road", "water_edge_sand")

#: Soft-diagnostic thresholds (survey ``constraints`` / kit brief priors;
#: loaded from the bundle at runtime, defaults only for synthetic bundles
#: that do not carry them).
DEFAULT_CONSTRAINTS = {
    "door_road_max_gu": 1500.0,
    "conform_max_slope_deg": 15.0,
    "steep_bank_slope_deg": 25.0,
    "flatten_pad_max_cut_fill_gu": 400.0,
    "min_building_gap_gu": 200.0,
}
#: Hard per-vertex terrain delta encoding bound (D-PLAN section 10).
MAX_EDIT_DELTA_GU = 1016.0

# ---------------------------------------------------------------------------
# Structural schema (strict; the JSON Schema under
# ``src/procgen/schemas/city_plan_schema_v1.json`` is kept in agreement by
# ``tests/test_cityplan.py``).
# ---------------------------------------------------------------------------

#: Leaf type tags: str / nonempty str / num / int / bool / point (2 nums) /
#: int2 (2 ints) / points (list of points) / object / array.
_NUMERIC = ("num", "int")


def _spec(*, type: str, required: bool = False, nonempty: bool = False,
          values: Optional[tuple] = None, min_points: int = 0,
          min_items: int = 0, keys: Optional[dict] = None,
          item: Optional[str] = None) -> dict:
    out = {"type": type, "required": required}
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
    return out


_POINT = _spec(type="point", min_points=2)
_POINTS3 = _spec(type="points", min_points=3)

#: Nested sub-object specs referenced by the item specs below (defined
#: first so ``ITEM_SPEC`` can reference them without self-reference).
_LOT_REQUEST_SPEC = {
    "building_type": _spec(type="str", required=True, nonempty=True),
    "size_class": _spec(type="str", values=LOT_SIZE_CLASSES),
    "stamp_id": _spec(type="str", nonempty=True),
    "multi_shell": _spec(type="bool"),
}
_TERRAIN_POLICY_SPEC = {
    "mode": _spec(type="str", required=True, values=TERRAIN_POLICY_MODES),
    "max_cut_fill_gu": _spec(type="num"),
}
_ACCESS_SPEC = {
    "face_road": _spec(type="str", required=True, nonempty=True),
}
_ZONE_CLASS_SPEC = {
    "texture": _spec(type="str", required=True, nonempty=True),
    "weight": _spec(type="num", required=True),
}

#: Allowed keys per item kind (used by the structural checker).
ITEM_SPEC: dict[str, dict[str, dict]] = {
    "settlement": {
        "name": _spec(type="str", required=True, nonempty=True),
        "seed_marker": _spec(type="str", required=True, nonempty=True),
        "anchor_cell": _spec(type="int2", required=True),
        "target_cells": _spec(type="object", required=True, keys={
            "min_x": _spec(type="int", required=True),
            "max_x": _spec(type="int", required=True),
            "min_y": _spec(type="int", required=True),
            "max_y": _spec(type="int", required=True),
        }),
    },
    "frame": {
        "origin_gu": _spec(type="point", required=True),
        "units": _spec(type="str", required=True, nonempty=True),
        "yaw_convention": _spec(type="str", required=True, nonempty=True),
        "site_survey_sha256": _spec(type="str", required=True, nonempty=True),
    },
    "district": {
        "district_id": _spec(type="str", required=True, nonempty=True),
        "kind": _spec(type="str", required=True, values=DISTRICT_KINDS),
        "polygon": _spec(type="points", required=True, min_points=3),
        "texture_zone": _spec(type="str", required=True, nonempty=True),
        "notes": _spec(type="str"),
    },
    "road": {
        "road_id": _spec(type="str", required=True, nonempty=True),
        "class": _spec(type="str", required=True, values=ROAD_CLASSES),
        "polyline": _spec(type="points", required=True, min_points=2),
        "width_gu": _spec(type="num", required=True),
        "surface": _spec(type="str", required=True, nonempty=True),
        "connects": _spec(type="array", required=True, item="conn_ref",
                          min_items=1),
        "grade_policy": _spec(type="str", required=True,
                              values=ROAD_GRADE_POLICIES),
    },
    "conn_ref": _spec(type="str", required=True, nonempty=True),
    "lot_request": _LOT_REQUEST_SPEC,
    "terrain_policy": _TERRAIN_POLICY_SPEC,
    "access": _ACCESS_SPEC,
    "lot": {
        "lot_id": _spec(type="str", required=True, nonempty=True),
        "district": _spec(type="str", required=True, nonempty=True),
        "position": _spec(type="point", required=True),
        "yaw_deg": _spec(type="num", required=True),
        "request": _spec(type="object", required=True, keys=_LOT_REQUEST_SPEC),
        "terrain_policy": _spec(type="object", required=True,
                                keys=_TERRAIN_POLICY_SPEC),
        "access": _spec(type="object", keys=_ACCESS_SPEC),
        "notes": _spec(type="str"),
    },
    "gate": {
        "gate_id": _spec(type="str", required=True, nonempty=True),
        "position": _spec(type="point", required=True),
        "heading_deg": _spec(type="num", required=True),
        "on_road": _spec(type="str", nonempty=True),
    },
    "boundary": {
        "boundary_id": _spec(type="str", required=True, nonempty=True),
        "kind": _spec(type="str", required=True, values=BOUNDARY_KINDS),
        "polygon": _spec(type="points", required=True, min_points=4),
        "gates": _spec(type="array", item="gate"),
    },
    "feature": {
        "feature_id": _spec(type="str", required=True, nonempty=True),
        "kind": _spec(type="str", required=True, values=FEATURE_KINDS),
        "position": _spec(type="point", required=True),
        "yaw_deg": _spec(type="num", required=True),
        "on_road": _spec(type="str", nonempty=True),
        "notes": _spec(type="str"),
    },
    "terrain_edit": {
        "edit_id": _spec(type="str", required=True, nonempty=True),
        "kind": _spec(type="str", required=True, values=TERRAIN_EDIT_KINDS),
        "polygon": _spec(type="points", required=True, min_points=3),
        "target_height_gu": _spec(type="num", required=True),
        "falloff_gu": _spec(type="num", required=True),
        "linked_to": _spec(type="array", required=True, item="link_ref",
                           min_items=1),
    },
    "link_ref": _spec(type="str", required=True, nonempty=True),
    "zone_class": _ZONE_CLASS_SPEC,
    "texture_zone": {
        "zone_id": _spec(type="str", required=True, nonempty=True),
        "classes": _spec(type="array", required=True, item="zone_class",
                         min_items=1),
    },
    "wilderness_hint": {
        "hint": _spec(type="str", required=True, values=WILDERNESS_HINTS),
        "polygon": _spec(type="points", required=True, min_points=3),
        "density": _spec(type="num", required=True),
    },
}

#: Top-level document shape.
DOCUMENT_SPEC: dict[str, dict] = {
    "schema_version": _spec(type="int", required=True),
    "plan_id": _spec(type="str", required=True, nonempty=True),
    "settlement": _spec(type="object", required=True,
                        keys=ITEM_SPEC["settlement"]),
    "frame": _spec(type="object", required=True, keys=ITEM_SPEC["frame"]),
    "design_notes": _spec(type="str", required=True),
    "districts": _spec(type="array", required=True, item="district"),
    "roads": _spec(type="array", required=True, item="road"),
    "lots": _spec(type="array", required=True, item="lot", min_items=1),
    "boundaries": _spec(type="array", required=True, item="boundary"),
    "features": _spec(type="array", required=True, item="feature"),
    "terrain_edits": _spec(type="array", required=True, item="terrain_edit"),
    "texture_zones": _spec(type="array", required=True, item="texture_zone"),
    "wilderness_hints": _spec(type="array", required=True,
                              item="wilderness_hint"),
}

#: Section name -> item-id field used for uniqueness checks.
SECTION_ITEM_ID_FIELD: dict[str, str] = {
    "districts": "district_id",
    "roads": "road_id",
    "lots": "lot_id",
    "boundaries": "boundary_id",
    "features": "feature_id",
    "terrain_edits": "edit_id",
    "texture_zones": "zone_id",
    "wilderness_hints": "hint_id",  # wilderness hints are unnamed in v1
}


# ---------------------------------------------------------------------------
# Issue model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Issue:
    """One deterministic validation finding (error or warning)."""

    severity: str          # "error" | "warning"
    code: str              # stable machine code
    path: str              # JSON-ish path, e.g. "lots[3].request.building_type"
    message: str
    measured: Any = None
    limit: Any = None

    def to_dict(self) -> dict:
        out = {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }
        if self.measured is not None:
            out["measured"] = self.measured
        if self.limit is not None:
            out["limit"] = self.limit
        return out


def _err(code: str, path: str, message: str, measured=None, limit=None) -> Issue:
    return Issue("error", code, path, message, measured, limit)


def _warn(code: str, path: str, message: str, measured=None, limit=None) -> Issue:
    return Issue("warning", code, path, message, measured, limit)


# ---------------------------------------------------------------------------
# Geometry helpers (pure, deterministic, no third-party deps)
# ---------------------------------------------------------------------------

def rot2d_ccw(x: float, y: float, theta_deg: float) -> tuple[float, float]:
    """Rotate (x, y) by theta degrees counter-clockwise (plan-frame +y up)."""
    rad = math.radians(theta_deg)
    c, s = math.cos(rad), math.sin(rad)
    return (x * c - y * s, x * s + y * c)


def yaw_hull(hull: list[list[float]], yaw_deg: float,
             anchor: tuple[float, float]) -> list[tuple[float, float]]:
    """Transform a stamp hull (relative to its door anchor) into plan-frame
    world points: ``anchor + Rz(yaw) . point`` -- the same composition
    D-STAMP section 5.2 defines for member transforms (XY only; Z is not
    part of plan geometry)."""
    return [(anchor[0] + dx, anchor[1] + dy)
            for dx, dy in (rot2d_ccw(p[0], p[1], yaw_deg) for p in hull)]


def close_ring(points: list[list[float]]) -> list[tuple[float, float]]:
    """Return the ring as a list of points with an explicit closing vertex
    (appends the first vertex when the caller left it open)."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def ring_area(points: list[list[float]]) -> float:
    """Signed shoelace area of a ring (auto-closed)."""
    pts = close_ring(points)
    return 0.5 * sum(pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]
                     for i in range(len(pts) - 1))


def ring_is_simple(points: list[list[float]]) -> bool:
    """True when the auto-closed ring is a simple polygon: >= 3 distinct
    vertices, nonzero area, and no non-adjacent segment intersection."""
    pts = close_ring(points)
    if len(set(pts)) < 3 or abs(ring_area(pts)) < 1e-9:
        return False
    n = len(pts) - 1  # ring edge count (closing edge is pts[n-1]->pts[0])
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        a, b = edges[i]
        for j in range(i + 1, n):
            c, d = edges[j]
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # adjacent edges share an endpoint
            if segments_intersect_strict(a, b, c, d):
                return False
    return True


def orient(a: tuple, b: tuple, c: tuple) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: tuple, b: tuple, p: tuple, eps: float = 1e-9) -> bool:
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def segments_intersect_strict(a: tuple, b: tuple, c: tuple, d: tuple) -> bool:
    """True when segments ab and cd properly intersect (interior crossing or
    collinear overlap of positive length).  Touching at endpoints is not a
    proper intersection."""
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and \
       ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)):
        return True
    # Collinear overlap with positive length.
    if abs(o1) < 1e-9 and abs(o2) < 1e-9:
        if on_segment(a, b, c) and (a != c and b != c):
            return True
        if on_segment(a, b, d) and (a != d and b != d):
            return True
        if on_segment(c, d, a) and (c != a and d != a):
            return True
    return False


def point_in_ring(p: tuple, ring: list[tuple]) -> bool:
    """Ray-casting point-in-polygon (boundary points count as inside)."""
    pts = close_ring(ring) if not ring or ring[0] != ring[-1] else ring
    n = len(pts) - 1
    inside = False
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if on_segment(a, b, p):
            return True
        if (a[1] > p[1]) != (b[1] > p[1]):
            x_cross = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x_cross > p[0]:
                inside = not inside
    return inside


def _seg_seg_distance(a: tuple, b: tuple, c: tuple, d: tuple) -> float:
    """Minimum distance between two segments (0 when they intersect)."""
    best = min(point_seg_distance(p, c, d) for p in (a, b))
    best = min(best, *(point_seg_distance(p, a, b) for p in (c, d)))
    return best


def point_seg_distance(p: tuple, a: tuple, b: tuple) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0 and dy == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def point_polyline_distance(p: tuple, polyline: list) -> float:
    pts = [(float(q[0]), float(q[1])) for q in polyline]
    return min(point_seg_distance(p, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def ring_min_distance(a: list, b: list) -> float:
    """Minimum distance between two rings (auto-closed)."""
    ra, rb = close_ring(a), close_ring(b)
    na, nb = len(ra) - 1, len(rb) - 1
    best = float("inf")
    for i in range(na):
        for j in range(nb):
            best = min(best, _seg_seg_distance(ra[i], ra[(i + 1) % na],
                                               rb[j], rb[(j + 1) % nb]))
    return best


def rings_overlap_exact(a: list, b: list) -> bool:
    """Strict polygon overlap: proper edge crossing, or a vertex of either
    ring strictly inside the other.  Touching boundaries are not overlap."""
    ra, rb = close_ring(a), close_ring(b)
    na, nb = len(ra) - 1, len(rb) - 1
    for i in range(na):
        for j in range(nb):
            if segments_intersect_strict(ra[i], ra[(i + 1) % na],
                                         rb[j], rb[(j + 1) % nb]):
                return True
    for p in ra[:-1]:
        if point_in_ring(p, rb) and not on_segment(rb[0], rb[1], p) and \
                not any(on_segment(rb[j], rb[(j + 1) % nb], p) for j in range(nb)):
            return True
    for p in rb[:-1]:
        if point_in_ring(p, ra) and not any(on_segment(ra[j], ra[(j + 1) % na], p) for j in range(na)):
            return True
    return False


def ring_pair_status(a: list, b: list) -> tuple[str, float]:
    """Classify a ring pair: ('overlap', 0.0) for strict overlap,
    ('touch', min_distance) for near contact within CONTACT_EPSILON_GU,
    ('clear', min_distance) otherwise."""
    if rings_overlap_exact(a, b):
        return ("overlap", 0.0)
    dist = ring_min_distance(a, b)
    if dist < CONTACT_EPSILON_GU:
        return ("touch", dist)
    return ("clear", dist)


def polyline_self_intersects(polyline: list) -> bool:
    """True when any two non-adjacent segments of the polyline intersect."""
    pts = [(float(p[0]), float(p[1])) for p in polyline]
    for i in range(len(pts) - 1):
        for j in range(i + 1, len(pts) - 1):
            if abs(i - j) == 1:
                continue
            if segments_intersect_strict(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                return True
    return False


def polygon_centroid(points: list) -> tuple[float, float]:
    """Area-weighted centroid of an auto-closed ring."""
    pts = close_ring(points)
    n = len(pts) - 1
    a = ring_area(pts)
    if abs(a) < 1e-9:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx = sum((pts[i][0] + pts[(i + 1) % n][0]) *
             (pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1])
             for i in range(n)) / (6 * a)
    cy = sum((pts[i][1] + pts[(i + 1) % n][1]) *
             (pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1])
             for i in range(n)) / (6 * a)
    return (cx, cy)


def tiles_covered_by_ring(ring: list, tile_size: int = TILE_SIZE_GU,
                          side: int = TILE_SIDE) -> list[tuple[int, int]]:
    """Tile indices (tx, ty) whose centers lie inside the ring.  Tile
    (tx, ty) covers plan GU [tx*tile_size, (tx+1)*tile_size) x
    [ty*tile_size, (ty+1)*tile_size)."""
    pts = close_ring(ring)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    t0x = max(0, int(math.floor(min(xs) / tile_size)))
    t1x = min(side - 1, int(math.floor(max(xs) / tile_size)))
    t0y = max(0, int(math.floor(min(ys) / tile_size)))
    t1y = min(side - 1, int(math.floor(max(ys) / tile_size)))
    out = []
    for ty in range(t0y, t1y + 1):
        for tx in range(t0x, t1x + 1):
            center = (tx * tile_size + tile_size / 2.0, ty * tile_size + tile_size / 2.0)
            if point_in_ring(center, pts):
                out.append((tx, ty))
    return out


def gu_to_tile(x: float, y: float) -> tuple[int, int]:
    """Plan-frame GU -> 512-GU tile index (row-major [y, x] convention)."""
    return (int(math.floor(x / TILE_SIZE_GU)), int(math.floor(y / TILE_SIZE_GU)))


def gu_to_field(x: float, y: float) -> tuple[int, int]:
    """Plan-frame GU -> 128-GU LAND field index (survey field spacing)."""
    return (int(math.floor(x / FIELD_SPACING_GU)), int(math.floor(y / FIELD_SPACING_GU)))


def in_scope(x: float, y: float) -> bool:
    """Plan-frame GU inside the site [0, span) x [0, span)."""
    return 0.0 <= x < SITE_SPAN_GU and 0.0 <= y < SITE_SPAN_GU


# ---------------------------------------------------------------------------
# Bundle: the accepted planner inputs, loaded and pinned at runtime
# ---------------------------------------------------------------------------

def decode_tile_mask(base64_text: str, side: int = TILE_SIDE) -> list[list[int]]:
    """Decode a survey tile mask (uint8, row-major [y, x], SW origin) into a
    side x side nested list of ints."""
    raw = base64.b64decode(base64_text)
    if len(raw) != side * side:
        raise ValueError(f"mask byte length {len(raw)} != {side * side}")
    return [[raw[y * side + x] for x in range(side)] for y in range(side)]


@dataclass
class Bundle:
    """Accepted planner-input bundle, hash-pinned, with derived indexes.

    Everything the validator consumes is loaded here from the canonical
    files -- the validator never duplicates accepted constants (frame,
    enums, stamps, surfaces, road ids) in code.
    """

    plan_path: Optional[Path] = None
    site_survey: dict = field(default_factory=dict)
    kit_brief: dict = field(default_factory=dict)
    region_palette: dict = field(default_factory=dict)
    libraries: dict = field(default_factory=dict)          # lib_id -> {stamps: [...]}
    centerlines: dict = field(default_factory=dict)
    aligned_network: object = None   # procgen.aligned_roads.AlignedNetwork
    site_survey_path: Optional[Path] = None
    kit_brief_path: Optional[Path] = None
    region_palette_path: Optional[Path] = None
    library_paths: dict = field(default_factory=dict)      # lib_id -> Path
    centerlines_path: Optional[Path] = None
    hashes: dict = field(default_factory=dict)             # input name -> sha256
    water_mask: list = field(default_factory=list)         # 112x112 [y][x]
    buildable_mask: list = field(default_factory=list)     # 112x112 [y][x]
    eligible_stamps: list = field(default_factory=list)    # kit-brief stamps
    stamp_geometry: dict = field(default_factory=dict)     # stamp_id -> library stamp
    surfaces: list = field(default_factory=list)           # closed vocabulary
    surface_set: set = field(default_factory=set)
    constraints: dict = field(default_factory=dict)
    edge_ids: set = field(default_factory=set)
    node_ids: set = field(default_factory=set)
    edge_by_id: dict = field(default_factory=dict)
    node_by_id: dict = field(default_factory=dict)
    map_exits: dict = field(default_factory=dict)          # exit_id -> {side, edge_id, points}

    @property
    def survey_frame(self) -> dict:
        return self.site_survey["frame"]

    @property
    def survey_sha256(self) -> str:
        return self.hashes.get("site_survey", "")

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_paths(cls, *, site_survey: Path | str, kit_brief: Path | str,
                   region_palette: Path | str,
                   stamp_libraries: Iterable[Path | str],
                   centerlines: Path | str) -> "Bundle":
        bundle = cls()
        bundle.site_survey_path = Path(site_survey)
        bundle.kit_brief_path = Path(kit_brief)
        bundle.region_palette_path = Path(region_palette)
        bundle.centerlines_path = Path(centerlines)
        for lib in stamp_libraries:
            lib = Path(lib)
            data = _load_json(lib, "stamp library")
            bundle.libraries[data["library_id"]] = data
            bundle.library_paths[data["library_id"]] = lib
        bundle.site_survey = _load_json(bundle.site_survey_path, "site survey")
        bundle.kit_brief = _load_json(bundle.kit_brief_path, "kit brief")
        bundle.region_palette = _load_json(bundle.region_palette_path, "region palette")
        # Road geometry is consumed exclusively through the aligned consumer
        # product via procgen.aligned_roads.  The loader fails closed on the
        # source-space bundle, hash drift, translation/topology drift, and
        # coordinate invariants; the recorded product hash is the input hash.
        bundle.aligned_network = aligned_roads.load_aligned_network(
            bundle.centerlines_path)
        bundle.hashes["centerlines"] = bundle.aligned_network.product_sha256
        for name, path in (
                ("site_survey", bundle.site_survey_path),
                ("kit_brief", bundle.kit_brief_path),
                ("region_palette", bundle.region_palette_path)):
            bundle.hashes[name] = _sha256_file(path)
        for lib_id, path in bundle.library_paths.items():
            bundle.hashes[f"stamp_library_{lib_id}"] = _sha256_file(path)
        bundle._derive()
        return bundle

    def _derive(self) -> None:
        tg = self.site_survey["tile_grids"]
        if tg["side"] != TILE_SIDE or tg["tile_size_gu"] != TILE_SIZE_GU:
            raise BundleError(
                f"survey tile grid {tg['side']}x{tg['tile_size_gu']} does not match "
                f"the T1.1 contract {TILE_SIDE}x{TILE_SIZE_GU}")
        self.water_mask = decode_tile_mask(tg["water_mask"])
        self.buildable_mask = decode_tile_mask(tg["buildable_mask"])
        # Eligible stamps: kit brief is the accepted eligible list.  Each
        # stamp's geometry comes from the hash-pinned library only.
        self.eligible_stamps = list(self.kit_brief["stamps"])
        self.stamp_geometry = {}
        for stamp in self.eligible_stamps:
            lib = self.libraries.get(stamp["library_id"])
            if lib is None:
                raise BundleError(
                    f"kit-brief stamp {stamp['stamp_id']} references missing "
                    f"library {stamp['library_id']}")
            lib_stamp = next((s for s in lib["stamps"]
                              if s["stamp_id"] == stamp["stamp_id"]), None)
            if lib_stamp is None:
                raise BundleError(
                    f"kit-brief stamp {stamp['stamp_id']} missing from library "
                    f"{stamp['library_id']}")
            self.stamp_geometry[stamp["stamp_id"]] = lib_stamp
        # Closed surface vocabulary.
        self.surfaces = [s["surface"]
                         for s in self.region_palette["semantic_surfaces"]["surfaces"]]
        self.surface_set = set(self.surfaces)
        # Survey constraints (soft diagnostics).
        self.constraints = dict(DEFAULT_CONSTRAINTS)
        self.constraints.update(self.site_survey.get("constraints", {}))
        # Aligned road network (aligned consumer product; the one supported
        # road entry point).  Synthetic bundles may still inject a plain dict
        # for unit fixtures; the aligned product path is the canonical one.
        if self.aligned_network is not None:
            network = self.aligned_network
            for edge in network.edges.values():
                self.edge_ids.add(edge.id)
                self.edge_by_id[edge.id] = edge
            for node in network.nodes.values():
                self.node_ids.add(node.id)
                self.node_by_id[node.id] = node
        else:
            for edge in self.centerlines["edges"]:
                self.edge_ids.add(edge["id"])
                self.edge_by_id[edge["id"]] = edge
            for node in self.centerlines["nodes"]:
                self.node_ids.add(node["id"])
                self.node_by_id[node["id"]] = node
        self.map_exits = measure_map_exits(
            self.aligned_network if self.aligned_network is not None
            else self.centerlines,
            tuple(self.survey_frame["origin_gu"]))

    # -- mask helpers -------------------------------------------------------

    def tile_value(self, mask: list, tx: int, ty: int) -> Optional[int]:
        if 0 <= tx < TILE_SIDE and 0 <= ty < TILE_SIDE:
            return mask[ty][tx]
        return None

    def tile_buildable(self, tx: int, ty: int) -> bool:
        return self.tile_value(self.buildable_mask, tx, ty) == 1

    def tile_water(self, tx: int, ty: int) -> bool:
        return self.tile_value(self.water_mask, tx, ty) == 1

    def door_anchor_state(self, x: float, y: float) -> dict:
        """Tile/buildable/water state of a plan GU door anchor."""
        if not in_scope(x, y):
            return {"in_scope": False, "tx": None, "ty": None,
                    "buildable": False, "water": False}
        tx, ty = gu_to_tile(x, y)
        return {"in_scope": True, "tx": tx, "ty": ty,
                "buildable": self.tile_buildable(tx, ty),
                "water": self.tile_water(tx, ty)}


class BundleError(Exception):
    """Fatal bundle problem (missing input, contract mismatch, corrupt
    accepted file).  Reported by the CLI as a hard configuration failure,
    not as a plan issue."""


def _load_json(path: Path, label: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise BundleError(f"cannot load {label} from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BundleError(f"{label} {path} is not a JSON object")
    return data


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Map-edge exits: measured from the aligned centerlines, plan-frame GU
# ---------------------------------------------------------------------------

def _edge_smooth_chain(edge: Any) -> list:
    """Aligned smooth world-GU chain of one edge record.

    Accepts an ``aligned_roads.AlignedEdge`` (canonical aligned product) or a
    plain dict (synthetic unit fixtures).  Consumers never see source-space
    coordinates here: the aligned product is the only real bundle path.
    """
    if isinstance(edge, dict):
        return edge["smooth_gu_polyline"]
    return edge.smooth_gu_polyline


def _edge_id(edge: Any) -> str:
    return edge["id"] if isinstance(edge, dict) else edge.id


def measure_map_exits(centerlines: Any, plan_origin: tuple) -> dict[str, dict]:
    """Measure the site's external road exits from the aligned centerline
    product: every aligned smooth polyline segment that crosses the site
    rectangle [0, span) x [0, span) in plan-frame GU produces an exit id
    ``exit_<side>_<edge_id>`` (one id per side/edge; a corner crossing
    yields both sides) with the measured crossing points in plan-frame GU.

    ``plan_origin`` is the accepted site survey's frame origin (plan GU 0,0
    in absolute world GU); the aligned polylines are absolute world GU and
    are shifted into the plan frame before clipping.

    This is the only map-edge-exit measurement used by T1.1; the survey's
    raw-78 continuation spans are informational and never drive geometry.
    """
    exits: dict[str, dict] = {}
    edges = (centerlines.edges.values() if hasattr(centerlines, "edges")
             else centerlines["edges"])
    for edge in edges:
        chain = _edge_smooth_chain(edge)
        edge_id = _edge_id(edge)
        pts = [(q[0] - plan_origin[0], q[1] - plan_origin[1]) for q in chain]
        found: dict[str, list] = {}
        for a, b in zip(pts, pts[1:]):
            for side, line in (("south", 0.0), ("west", 0.0),
                               ("north", SITE_SPAN_GU), ("east", SITE_SPAN_GU)):
                if side in ("south", "north"):
                    if (a[1] - line) * (b[1] - line) > 0:
                        continue
                    if a[1] == b[1]:
                        continue
                    t = (line - a[1]) / (b[1] - a[1])
                    if not (0.0 <= t <= 1.0):
                        continue
                    px = a[0] + t * (b[0] - a[0])
                    if not (-1e-6 <= px <= SITE_SPAN_GU + 1e-6):
                        continue
                    p = (round(min(max(px, 0.0), SITE_SPAN_GU), 1), line)
                else:
                    if (a[0] - line) * (b[0] - line) > 0:
                        continue
                    if a[0] == b[0]:
                        continue
                    t = (line - a[0]) / (b[0] - a[0])
                    if not (0.0 <= t <= 1.0):
                        continue
                    py = a[1] + t * (b[1] - a[1])
                    if not (-1e-6 <= py <= SITE_SPAN_GU + 1e-6):
                        continue
                    p = (line, round(min(max(py, 0.0), SITE_SPAN_GU), 1))
                found.setdefault(side, []).append(p)
        for side, points in found.items():
            exit_id = f"exit_{side}_{edge_id}"
            exits[exit_id] = {"side": side, "edge_id": edge_id,
                              "points": sorted(set(points))}
    return dict(sorted(exits.items()))


# ---------------------------------------------------------------------------
# Structural validation (strict recursive unknown-key rejection)
# ---------------------------------------------------------------------------

def _path(base: str, *parts: Any) -> str:
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
            issues.append(_err("wrong_type", path,
                               f"expected integer, got {type(value).__name__}"))
            return
    elif not _is_num(value):
        issues.append(_err("wrong_type", path,
                           f"expected number, got {type(value).__name__}"))
        return
    if isinstance(value, float) and not math.isfinite(value):
        issues.append(_err("not_finite", path, "number is NaN or infinite"))


def _resolve_item_spec(tag: Any) -> dict:
    """Resolve an array ``item`` tag to a full spec dict: named object item
    tags (ITEM_SPEC keys maps) are wrapped as strict objects, string tags
    (conn_ref / link_ref) resolve to their string spec, and inline dict
    specs pass through."""
    if isinstance(tag, str):
        keys = ITEM_SPEC.get(tag)
        if isinstance(keys, dict) and "type" in keys:
            return keys
        if isinstance(keys, dict):
            return {"type": "object", "keys": keys}
        return _spec(type="str")
    if isinstance(tag, dict):
        return tag
    return _spec(type="str")


def _check_structure(node: Any, spec: dict, path: str, issues: list) -> None:
    kind = spec["type"]
    if kind == "object":
        if not isinstance(node, dict):
            issues.append(_err("wrong_type", path,
                               f"expected object, got {type(node).__name__}"))
            return
        allowed = spec.get("keys", {})
        for key in node:
            if key not in allowed:
                issues.append(_err("unknown_key", _path(path, key),
                                   f"unknown key (strict schema)"))
        for key, child_spec in allowed.items():
            if key not in node:
                if child_spec.get("required"):
                    issues.append(_err("missing_required", _path(path, key),
                                       "missing required field"))
                continue
            _check_structure(node[key], child_spec, _path(path, key), issues)
    elif kind == "array":
        if not isinstance(node, list):
            issues.append(_err("wrong_type", path,
                               f"expected array, got {type(node).__name__}"))
            return
        if spec.get("min_items") and len(node) < spec["min_items"]:
            issues.append(_err("too_few_items", path,
                               f"expected at least {spec['min_items']} items, "
                               f"got {len(node)}"))
        item_tag = spec.get("item")
        item_spec = _resolve_item_spec(item_tag)
        for idx, item in enumerate(node):
            _check_structure(item, item_spec, _path(path, f"[{idx}]"), issues)
    elif kind == "str":
        if not isinstance(node, str):
            issues.append(_err("wrong_type", path,
                               f"expected string, got {type(node).__name__}"))
            return
        if spec.get("nonempty") and not node.strip():
            issues.append(_err("empty_id", path, "string must be non-empty"))
        if spec.get("values") and node not in spec["values"]:
            issues.append(_err("bad_enum", path,
                               f"value {node!r} not in {list(spec['values'])}"))
    elif kind in _NUMERIC:
        _check_number(node, path, issues, int_only=(kind == "int"))
    elif kind == "bool":
        if not isinstance(node, bool):
            issues.append(_err("wrong_type", path,
                               f"expected boolean, got {type(node).__name__}"))
    elif kind == "point":
        if not isinstance(node, list) or len(node) != 2:
            issues.append(_err("wrong_type", path,
                               "expected [x, y] pair of numbers"))
            return
        for idx, val in enumerate(node):
            _check_number(val, _path(path, f"[{idx}]"), issues, int_only=False)
    elif kind == "int2":
        if not isinstance(node, list) or len(node) != 2:
            issues.append(_err("wrong_type", path,
                               "expected [x, y] pair of integers"))
            return
        for idx, val in enumerate(node):
            _check_number(val, _path(path, f"[{idx}]"), issues, int_only=True)
    elif kind == "points":
        if not isinstance(node, list):
            issues.append(_err("wrong_type", path,
                               f"expected list of [x, y] pairs, got {type(node).__name__}"))
            return
        if spec.get("min_points") and len(node) < spec["min_points"]:
            issues.append(_err("too_few_points", path,
                               f"expected at least {spec['min_points']} points, "
                               f"got {len(node)}"))
        for idx, pt in enumerate(node):
            if not isinstance(pt, list) or len(pt) != 2:
                issues.append(_err("wrong_type", _path(path, f"[{idx}]"),
                                   "expected [x, y] pair of numbers"))
                continue
            for sub, val in enumerate(pt):
                _check_number(val, _path(path, f"[{idx}][{sub}]"), issues,
                              int_only=False)
    else:  # pragma: no cover - schema bug guard
        raise BundleError(f"internal schema error: unknown type {kind!r}")


def check_structure(plan: dict) -> list:
    """Strict structural gate: types, enums, required fields, recursive
    unknown-key rejection, finite numbers.  Returns issues (no early exit)."""
    issues: list = []
    _check_structure(plan, {"type": "object", "keys": DOCUMENT_SPEC},
                     "$", issues)
    return issues


# ---------------------------------------------------------------------------
# Semantic + geometric validation
# ---------------------------------------------------------------------------

def _finite_checked(value: Any, default: float) -> float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) else default


def _is_finite_pt(p: Any) -> bool:
    """True for a [x, y] pair of finite numbers."""
    return isinstance(p, (list, tuple)) and len(p) == 2 and \
        all(isinstance(v, (int, float)) and not isinstance(v, bool) and
            math.isfinite(v) for v in p)


def validate_plan(plan: dict, bundle: Bundle) -> dict:
    """Validate one city plan against the accepted bundle.

    Returns a deterministic dict::

        {
          "valid": bool,                 # no errors
          "plan_id": str,
          "schema_version": int,
          "issue_count": int, "error_count": int, "warning_count": int,
          "issues": [Issue.to_dict(), ...],          # sorted
          "summary": {...},                          # counts + per-lot resolution
          "input_hashes": {name: sha256, ...},
        }

    No early exit: every check runs; the issue list is complete.  Invalid
    plans produce no trusted validated-plan artifact (callers must not
    treat this dict as one).
    """
    issues: list[Issue] = []
    issues.extend(check_structure(plan))

    if not isinstance(plan, dict):
        # Non-object plans: the structural gate already reported wrong_type;
        # there is nothing to validate semantically.
        errors = [i for i in issues if i.severity == "error"]
        return {
            "valid": not errors,
            "plan_id": "",
            "schema_version": None,
            "issue_count": len(issues),
            "error_count": len(errors),
            "warning_count": len(issues) - len(errors),
            "issues": [i.to_dict() for i in issues],
            "summary": {"plan_id": "", "schema_version": None,
                        "issue_count": len(issues),
                        "error_count": len(errors),
                        "warning_count": len(issues) - len(errors),
                        "sections": {}, "lot_resolution": [],
                        "lot_warnings": {}, "external_references": {},
                        "frame": {}, "warning_codes": []},
            "input_hashes": dict(sorted(bundle.hashes.items())),
        }

    # -- top level ----------------------------------------------------------
    plan_id = plan.get("plan_id", "")
    if isinstance(plan_id, str) and not plan_id.strip():
        issues.append(_err("empty_id", "$.plan_id", "plan_id must be non-empty"))

    # -- frame pin (site survey is the actual authority) --------------------
    frame = plan.get("frame")
    if isinstance(frame, dict):
        origin = frame.get("origin_gu")
        survey_origin = bundle.survey_frame["origin_gu"]
        if origin != survey_origin:
            issues.append(_err(
                "frame_origin_mismatch", "$.frame.origin_gu",
                f"frame origin {origin} does not match the accepted site "
                f"survey origin {survey_origin}", origin, survey_origin))
        if frame.get("units") not in (None, "game_units"):
            issues.append(_err("frame_units_mismatch", "$.frame.units",
                               f"units {frame.get('units')!r}; the D-PLAN v1 "
                               f"contract is 'game_units'"))
        survey_axis = bundle.survey_frame.get("axis_convention", "")
        if frame.get("yaw_convention") != survey_axis:
            issues.append(_err(
                "frame_yaw_convention_mismatch", "$.frame.yaw_convention",
                f"yaw convention {frame.get('yaw_convention')!r} does not "
                f"match the accepted survey {survey_axis!r}"))
        if frame.get("site_survey_sha256") != bundle.survey_sha256:
            issues.append(_err(
                "frame_sha_mismatch", "$.frame.site_survey_sha256",
                "site-survey SHA-256 does not match the pinned accepted "
                "site_survey.json", frame.get("site_survey_sha256"),
                bundle.survey_sha256))

    # -- settlement pin -----------------------------------------------------
    settlement = plan.get("settlement")
    seed = bundle.site_survey.get("seed_settlement", {})
    target = bundle.site_survey.get("target_cells", {})
    if isinstance(settlement, dict):
        if settlement.get("name") != seed.get("name"):
            issues.append(_err("settlement_mismatch", "$.settlement.name",
                               f"settlement name {settlement.get('name')!r} "
                               f"!= accepted {seed.get('name')!r}"))
        if settlement.get("seed_marker") != seed.get("marker_id"):
            issues.append(_err("settlement_mismatch", "$.settlement.seed_marker",
                               f"seed marker {settlement.get('seed_marker')!r} "
                               f"!= accepted {seed.get('marker_id')!r}"))
        if settlement.get("anchor_cell") != seed.get("anchor_cell"):
            issues.append(_err("settlement_mismatch", "$.settlement.anchor_cell",
                               f"anchor cell {settlement.get('anchor_cell')} "
                               f"!= accepted {seed.get('anchor_cell')}"))
        tc = settlement.get("target_cells")
        if isinstance(tc, dict) and dict(tc) != dict(target):
            issues.append(_err("settlement_mismatch", "$.settlement.target_cells",
                               f"target cells {dict(tc)} != accepted {dict(target)}"))

    # -- id namespaces ------------------------------------------------------
    ids: dict[str, set] = {}
    for section, id_field in SECTION_ITEM_ID_FIELD.items():
        if id_field == "hint_id":
            continue
        items = plan.get(section)
        if not isinstance(items, list):
            continue
        seen: set = set()
        for idx, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get(id_field), str):
                continue
            value = item[id_field]
            if value in seen:
                issues.append(_err("duplicate_id",
                                   _path("$", section, f"[{idx}].{id_field}"),
                                   f"duplicate {id_field} {value!r}"))
            seen.add(value)
            ids.setdefault(section, set()).add(value)
        ids.setdefault(section, seen)

    all_ids: set = set()
    for section, section_ids in ids.items():
        all_ids |= section_ids
    gate_ids: set = set()
    boundaries = plan.get("boundaries")
    if isinstance(boundaries, list):
        for bidx, boundary in enumerate(boundaries):
            if not isinstance(boundary, dict):
                continue
            gates = boundary.get("gates")
            if not isinstance(gates, list):
                continue
            seen: set = set()
            for gidx, gate in enumerate(gates):
                if isinstance(gate, dict) and isinstance(gate.get("gate_id"), str):
                    gate_id = gate["gate_id"]
                    if gate_id in seen:
                        issues.append(_err("duplicate_id",
                                           _path("$", "boundaries", f"[{bidx}].gates",
                                                 f"[{gidx}].gate_id"),
                                           f"duplicate gate_id {gate_id!r}"))
                    seen.add(gate_id)
                    gate_ids.add(gate_id)

    # -- districts ----------------------------------------------------------
    districts = plan.get("districts") or []
    zone_ids = ids.get("texture_zones", set())
    district_ids = ids.get("districts", set())
    for idx, district in enumerate(districts):
        if not isinstance(district, dict):
            continue
        base = _path("$", "districts", f"[{idx}]")
        poly = district.get("polygon")
        if isinstance(poly, list) and len(poly) >= 3:
            if not ring_is_simple(poly):
                issues.append(_err("polygon_invalid", _path(base, "polygon"),
                                   "polygon is degenerate, self-intersecting, "
                                   "or has fewer than 3 distinct vertices"))
        if district.get("texture_zone") not in zone_ids:
            issues.append(_err("district_texture_zone_unknown",
                               _path(base, "texture_zone"),
                               f"texture_zone {district.get('texture_zone')!r} "
                               f"does not reference a declared texture zone"))

    # -- texture zones (closed vocabulary + road identity protection) -------
    zones = plan.get("texture_zones") or []
    for idx, zone in enumerate(zones):
        if not isinstance(zone, dict):
            continue
        base = _path("$", "texture_zones", f"[{idx}]")
        classes = zone.get("classes")
        if isinstance(classes, list) and classes:
            total = 0.0
            has_road = False
            for cidx, cls in enumerate(classes):
                if not isinstance(cls, dict):
                    continue
                cbase = _path(base, "classes", f"[{cidx}]")
                tex = cls.get("texture")
                if tex not in bundle.surface_set:
                    issues.append(_err("zone_texture_unknown",
                                       _path(cbase, "texture"),
                                       f"texture {tex!r} is not in the region "
                                       f"palette closed surface vocabulary"))
                if tex == "road":
                    has_road = True
                weight = cls.get("weight")
                if isinstance(weight, (int, float)) and not isinstance(weight, bool):
                    total += weight
            if total < 0.5 or total > 1.5:
                issues.append(_err("zone_weights_sum", base,
                                   f"class weights sum to {total:.4f}; expected ~1.0",
                                   round(total, 6), 1.0))
            elif abs(total - 1.0) > 0.01:
                issues.append(_warn("zone_weights_sum", base,
                                    f"class weights sum to {total:.4f} (within "
                                    f"the hard band but off 1.0)",
                                    round(total, 6), 1.0))
            if has_road:
                issues.append(_err(
                    "zone_references_protected_road", base,
                    "texture zones must not paint the protected raw-78 road "
                    "surface; road identity is authored by the road stage"))

    # -- roads --------------------------------------------------------------
    roads = plan.get("roads") or []
    road_ids = ids.get("roads", set())
    road_by_id: dict[str, dict] = {r.get("road_id"): r for r in roads
                                   if isinstance(r, dict) and isinstance(r.get("road_id"), str)}
    # resolve external references once per road
    road_external: dict[str, list] = {}
    for idx, road in enumerate(roads):
        if not isinstance(road, dict):
            continue
        base = _path("$", "roads", f"[{idx}]")
        road_id = road.get("road_id")
        if not isinstance(road_id, str):
            continue
        polyline = road.get("polyline")
        if isinstance(polyline, list) and len(polyline) >= 2:
            if polyline_self_intersects(polyline):
                issues.append(_err("road_self_intersection", _path(base, "polyline"),
                                   "road polyline self-intersects"))
            pts = [(p[0], p[1]) for p in polyline
                   if isinstance(p, list) and len(p) == 2]
            if all(p == pts[0] for p in pts):
                issues.append(_err("road_degenerate", _path(base, "polyline"),
                                   "road polyline has zero extent"))
        width = road.get("width_gu")
        if isinstance(width, (int, float)) and not isinstance(width, bool) and width <= 0:
            issues.append(_err("road_width_invalid", _path(base, "width_gu"),
                               f"road width must be > 0, got {width}", width, 0))
        surface = road.get("surface")
        if surface not in bundle.surface_set:
            issues.append(_err("road_surface_invalid", _path(base, "surface"),
                               f"surface {surface!r} is not in the region palette "
                               f"closed surface vocabulary"))
        elif road.get("class") in ("street", "approach") and surface != "road":
            issues.append(_err(
                "road_surface_not_road", _path(base, "surface"),
                f"class {road.get('class')!r} must keep the protected raw-78 "
                f"road identity (surface 'road'); scatter/groundcover gates "
                f"key on it"))
        connects = road.get("connects")
        ext_refs: list = []
        if isinstance(connects, list):
            for cidx, ref in enumerate(connects):
                if not isinstance(ref, str) or not ref:
                    continue
                cpath = _path(base, "connects", f"[{cidx}]")
                if ref == road_id:
                    issues.append(_err("road_connects_self", cpath,
                                       "road connects to itself"))
                elif ref in road_ids and ref != road_id:
                    continue  # plan-internal connection
                elif ref in gate_ids:
                    continue  # gate connection (internal)
                elif ref in bundle.edge_ids:
                    ext_refs.append(ref)
                elif ref in bundle.node_ids:
                    ext_refs.append(ref)
                elif ref in bundle.map_exits:
                    ext_refs.append(ref)
                else:
                    issues.append(_err("road_connect_unknown", cpath,
                                       f"connects reference {ref!r} is not a "
                                       f"plan road, gate, aligned-centerline "
                                       f"edge/node id, or measured map-edge exit"))
        road_external[road_id] = ext_refs
        # soft: external element should be geometrically near the road
        for ref in ext_refs:
            anchor_pt = _external_anchor_point(bundle, ref)
            if anchor_pt is None:
                continue
            if not (isinstance(polyline, list) and len(polyline) >= 2 and
                    all(_is_finite_pt(p) for p in polyline)):
                continue
            dist = point_polyline_distance(anchor_pt, polyline)
            if dist > 8192.0:
                issues.append(_warn(
                    "road_external_ref_distance", _path(base, "connects"),
                    f"external reference {ref!r} lies {dist:.0f} GU from the "
                    f"road polyline (soft sanity diagnostic)",
                    round(dist, 1), 8192.0))
    # external reachability: every road component must reach an external ref
    orphan_components = _orphan_road_components(roads, road_external)
    for comp in orphan_components:
        issues.append(_err("road_orphan_component", _path("$", "roads"),
                           f"road component {sorted(comp)} does not reach any "
                           f"external network element (aligned-centerline "
                           f"edge/node id or measured map-edge exit)"))

    # -- lots ---------------------------------------------------------------
    lots = plan.get("lots") or []
    lot_positions: list[tuple[str, tuple, float]] = []
    resolved_hulls: list[tuple[str, list]] = []
    resolution: dict[str, dict] = {}
    lot_warnings: dict[str, list] = {}
    for idx, lot in enumerate(lots):
        if not isinstance(lot, dict):
            continue
        base = _path("$", "lots", f"[{idx}]")
        lot_id = lot.get("lot_id")
        if not isinstance(lot_id, str):
            continue
        position = lot.get("position")
        yaw = lot.get("yaw_deg")
        if not (isinstance(position, list) and len(position) == 2):
            continue
        if not (isinstance(yaw, (int, float)) and not isinstance(yaw, bool) and
                math.isfinite(yaw)):
            yaw = 0.0
        x, y = float(position[0]), float(position[1])
        pos_finite = math.isfinite(x) and math.isfinite(y)
        if not pos_finite:
            # NaN/Infinity already reported by the structural gate; skip all
            # geometric handling for this lot (floor/ring math must not run).
            resolution[lot_id] = {"lot_id": lot_id, "mode": "none",
                                  "stamp_id": None, "geometry_checked": False}
            continue
        if not in_scope(x, y):
            issues.append(_err("out_of_scope", _path(base, "position"),
                               f"lot anchor ({x:.1f}, {y:.1f}) is outside the "
                               f"plan frame [0, {SITE_SPAN_GU:.0f}) x [0, "
                               f"{SITE_SPAN_GU:.0f})", [round(x, 1), round(y, 1)],
                               SITE_SPAN_GU))
        else:
            state = bundle.door_anchor_state(x, y)
            if not state["buildable"]:
                issues.append(_err("door_unbuildable", _path(base, "position"),
                                   f"door anchor tile ({state['tx']}, {state['ty']}) "
                                   f"is not in the site buildable mask",
                                   [state["tx"], state["ty"]], 1))
            if state["water"]:
                issues.append(_err("door_in_water", _path(base, "position"),
                                   f"door anchor tile ({state['tx']}, {state['ty']}) "
                                   f"is in the water mask (docks are the only "
                                   f"water-position exception)",
                                   [state["tx"], state["ty"]], 0))
        # district reference
        if lot.get("district") not in district_ids:
            issues.append(_err("district_unknown", _path(base, "district"),
                               f"district {lot.get('district')!r} does not "
                               f"reference a declared district"))
        # request vocabulary
        request = lot.get("request")
        stamp_id: Optional[str] = None
        if isinstance(request, dict):
            rbase = _path(base, "request")
            btype = request.get("building_type")
            if btype in UNAVAILABLE_BUILDING_TYPES:
                issues.append(_err("building_type_unavailable", _path(rbase, "building_type"),
                                   f"building_type {btype!r} is a kit-brief "
                                   f"capability gap and fails closed"))
            elif not isinstance(btype, str) or \
                    btype not in (bundle.kit_brief.get("building_type_enum") or ()):
                issues.append(_err("building_type_unknown", _path(rbase, "building_type"),
                                   f"building_type {btype!r} is not in the "
                                   f"accepted kit-brief enum"))
            stamp_id = request.get("stamp_id")
            if stamp_id is not None:
                if stamp_id not in bundle.stamp_geometry:
                    issues.append(_err("stamp_not_eligible", _path(rbase, "stamp_id"),
                                       f"stamp_id {stamp_id!r} is not one of the "
                                       f"{len(bundle.eligible_stamps)} eligible "
                                       f"kit-brief stamps"))
                elif btype and isinstance(btype, str) and \
                        bundle.stamp_geometry[stamp_id].get("building_type") != btype:
                    issues.append(_err(
                        "stamp_type_mismatch", _path(rbase, "stamp_id"),
                        f"stamp {stamp_id} is building_type "
                        f"{bundle.stamp_geometry[stamp_id].get('building_type')!r}, "
                        f"requested {btype!r}"))
            size_class = request.get("size_class")
            if size_class is not None and stamp_id is not None and \
                    stamp_id in bundle.stamp_geometry and \
                    bundle.stamp_geometry[stamp_id].get("size_class") != size_class:
                issues.append(_err("stamp_size_mismatch", _path(rbase, "size_class"),
                                   f"stamp {stamp_id} is size_class "
                                   f"{bundle.stamp_geometry[stamp_id].get('size_class')!r}, "
                                   f"requested {size_class!r}"))
            multi = request.get("multi_shell")
            if multi is not None and stamp_id is not None and \
                    stamp_id in bundle.stamp_geometry and \
                    bool(bundle.stamp_geometry[stamp_id].get("multi_shell")) != bool(multi):
                issues.append(_err("stamp_multi_shell_mismatch", _path(rbase, "multi_shell"),
                                   f"stamp {stamp_id} multi_shell="
                                   f"{bool(bundle.stamp_geometry[stamp_id].get('multi_shell'))}, "
                                   f"requested {bool(multi)}"))
        # terrain policy
        policy = lot.get("terrain_policy")
        if isinstance(policy, dict):
            pbase = _path(base, "terrain_policy")
            mode = policy.get("mode")
            max_cut = policy.get("max_cut_fill_gu")
            if mode == "flatten_pad":
                if not isinstance(max_cut, (int, float)) or \
                        isinstance(max_cut, bool) or not math.isfinite(max_cut):
                    issues.append(_err("pad_limit_missing", _path(pbase, "max_cut_fill_gu"),
                                       "flatten_pad lots must declare a finite "
                                       "max_cut_fill_gu"))
                else:
                    limit = bundle.constraints.get("flatten_pad_max_cut_fill_gu", 400.0)
                    if max_cut > MAX_EDIT_DELTA_GU:
                        issues.append(_err(
                            "pad_exceeds_encoding_bound", _path(pbase, "max_cut_fill_gu"),
                            f"max_cut_fill_gu {max_cut} exceeds the ±{MAX_EDIT_DELTA_GU:.0f} "
                            f"GU per-vertex terrain delta encoding bound",
                            max_cut, MAX_EDIT_DELTA_GU))
                    elif max_cut > limit:
                        issues.append(_warn(
                            "pad_exceeds_site_constraint", _path(pbase, "max_cut_fill_gu"),
                            f"max_cut_fill_gu {max_cut} exceeds the surveyed site "
                            f"constraint {limit:.0f} GU (soft; solver-enforced "
                            f"at placement)",
                            max_cut, limit))
        # access
        access = lot.get("access")
        if isinstance(access, dict):
            face_road = access.get("face_road")
            if isinstance(face_road, str) and face_road not in road_ids and \
                    face_road not in bundle.edge_ids and face_road not in bundle.node_ids:
                issues.append(_err("access_face_road_unknown",
                                   _path(base, "access", "face_road"),
                                   f"face_road {face_road!r} does not reference "
                                   f"a plan road or aligned-centerline "
                                   f"edge/node id"))

        # -- resolution (explicit or shared deterministic selector) ---------
        if isinstance(request, dict) and isinstance(request.get("building_type"), str):
            btype = request["building_type"]
            mode: str
            if stamp_id is not None and stamp_id in bundle.stamp_geometry:
                mode = "explicit"
                selected = stamp_id
            else:
                candidates = _candidate_stamps(bundle, request)
                if not candidates:
                    issues.append(_err(
                        "no_compatible_stamp", _path(base, "request"),
                        f"no eligible stamp matches building_type "
                        f"{btype!r} size_class {request.get('size_class')!r} "
                        f"multi_shell {request.get('multi_shell')!r}"))
                    mode, selected = "none", None
                else:
                    mode = "selector"
                    selected = _select_stamp(bundle, candidates)
            if selected is not None:
                geometry = bundle.stamp_geometry[selected]
                hull = geometry.get("footprint", {}).get("hull_xy_rel")
                if not isinstance(hull, list) or len(hull) < 3:
                    issues.append(_err("stamp_hull_missing",
                                       _path(base, "request", "stamp_id"),
                                       f"stamp {selected} has no usable "
                                       f"footprint hull in the pinned library"))
                else:
                    world = yaw_hull(hull, yaw, (x, y))
                    _check_footprint(world, base, lot_id, issues, bundle)
                    resolved_hulls.append((lot_id, world))
            resolution[lot_id] = {
                "lot_id": lot_id,
                "mode": mode,
                "stamp_id": selected,
                "geometry_checked": selected is not None,
            }
        lot_positions.append((lot_id, (x, y), yaw))

    # -- pairwise footprint overlap (hard) and spacing diagnostics (soft) ---
    for i in range(len(resolved_hulls)):
        for j in range(i + 1, len(resolved_hulls)):
            id_a, hull_a = resolved_hulls[i]
            id_b, hull_b = resolved_hulls[j]
            status, dist = ring_pair_status(hull_a, hull_b)
            path = _path("$", "lots")
            if status == "overlap":
                issues.append(_err("footprint_overlap", path,
                                   f"lot {id_a} and lot {id_b} footprints "
                                   f"strictly overlap",
                                   [id_a, id_b], "0.0 GU"))
            elif status == "touch":
                issue = _warn("footprint_touch", path,
                              f"lot {id_a} and lot {id_b} footprints "
                              f"touch at {dist:.2f} GU (within the "
                              f"{CONTACT_EPSILON_GU} GU contact epsilon; "
                              f"0.0 GU hard minimum permits touching)",
                              round(dist, 3), CONTACT_EPSILON_GU)
                issues.append(issue)
                lot_warnings.setdefault(id_a, []).append(issue)
                lot_warnings.setdefault(id_b, []).append(issue)
            else:
                # measured spacing guidance is not a hard minimum (D-BRIEF)
                bps = bundle.kit_brief.get("spacing_priors", {})
                gap = bps.get("inter_building_gap_gu", {})
                p10 = _finite_checked(gap.get("p10"), 20.0)
                if dist < p10:
                    issue = _warn("spacing_below_measured_p10", path,
                                  f"lot {id_a}/{id_b} boundary gap "
                                  f"{dist:.1f} GU is below the measured "
                                  f"inter-building gap p10 {p10:.1f} GU "
                                  f"(measured guidance, not a hard minimum)",
                                  round(dist, 1), round(p10, 1))
                    issues.append(issue)
                    lot_warnings.setdefault(id_a, []).append(issue)
                    lot_warnings.setdefault(id_b, []).append(issue)
                elif dist < _finite_checked(
                        bundle.constraints.get("min_building_gap_gu"), 200.0):
                    issue = _warn(
                        "spacing_below_survey_constraint", path,
                        f"lot {id_a}/{id_b} boundary gap {dist:.1f} GU is below "
                        f"the surveyed min_building_gap_gu "
                        f"{bundle.constraints.get('min_building_gap_gu')} GU "
                        f"(soft; D-BRIEF marks spacing as guidance)",
                        round(dist, 1),
                        bundle.constraints.get("min_building_gap_gu"))
                    issues.append(issue)
                    lot_warnings.setdefault(id_a, []).append(issue)
                    lot_warnings.setdefault(id_b, []).append(issue)

    # -- soft lot diagnostics ----------------------------------------------
    roads_pts: dict[str, list] = {}
    for road in roads:
        if isinstance(road, dict) and isinstance(road.get("road_id"), str):
            polyline = road.get("polyline")
            if isinstance(polyline, list) and len(polyline) >= 2:
                roads_pts[road["road_id"]] = [p for p in
                                              ((p[0], p[1]) for p in polyline
                                               if isinstance(p, list) and len(p) == 2)
                                              if _is_finite_pt(p)]
    cells_by_gu = _cells_lookup(bundle.site_survey)
    origin_gu = tuple(bundle.survey_frame["origin_gu"])
    for lot_id, (x, y), yaw in lot_positions:
        base = _path("$", "lots")
        # door-to-road distance (soft)
        best = float("inf")
        for rpts in roads_pts.values():
            best = min(best, point_polyline_distance((x, y), rpts))
        limit = bundle.constraints.get("door_road_max_gu", 1500.0)
        if best > limit:
            issue = _warn("door_road_distance", base,
                          f"lot {lot_id} door anchor is {best:.0f} GU from "
                          f"the nearest planned road (soft)",
                          round(best, 1), limit)
            issues.append(issue)
            lot_warnings.setdefault(lot_id, []).append(issue)
        # access heading deviation (soft)
        access = next((l for l in lots if isinstance(l, dict) and l.get("lot_id") == lot_id), {})
        if isinstance(access, dict):
            face_road = (access.get("access") or {}).get("face_road")
            if isinstance(face_road, str) and face_road in roads_pts:
                rpts = roads_pts[face_road]
                nearest = _nearest_point_on_polyline((x, y), rpts)
                to_road = math.degrees(math.atan2(nearest[1] - y, nearest[0] - x))
                stamp_geom = None
                res = resolution.get(lot_id)
                if res and res.get("stamp_id"):
                    stamp_geom = bundle.stamp_geometry[res["stamp_id"]]
                heading = yaw + (math.degrees(stamp_geom["access_heading_rad"])
                                 if stamp_geom else 0.0)
                dev = abs((heading - to_road + 180.0) % 360.0 - 180.0)
                if dev > 90.0:
                    issue = _warn("door_heading_deviation", base,
                                  f"lot {lot_id} access heading "
                                  f"{heading:.1f} deg deviates {dev:.1f} "
                                  f"deg from its face_road direction "
                                  f"{to_road:.1f} deg (soft)",
                                  round(dev, 1), 90.0)
                    issues.append(issue)
                    lot_warnings.setdefault(lot_id, []).append(issue)
        # slope / terrain-envelope risk (soft)
        cell = cells_by_gu.get((int(math.floor((origin_gu[0] + x) / 8192.0)),
                                int(math.floor((origin_gu[1] + y) / 8192.0))))
        policy = access.get("terrain_policy") if isinstance(access, dict) else None
        mode = (policy or {}).get("mode") if isinstance(policy, dict) else None
        if cell is not None and isinstance(cell, dict):
            slope = cell.get("slope_mean_deg")
            if isinstance(slope, (int, float)):
                limit_slope = bundle.constraints.get("conform_max_slope_deg", 15.0)
                steep = bundle.constraints.get("steep_bank_slope_deg", 25.0)
                if slope >= steep:
                    issue = _warn("slope_risk", base,
                                  f"lot {lot_id} sits on a cell with mean "
                                  f"slope {slope:.1f} deg (>= steep-bank "
                                  f"{steep:.0f} deg) without a flatten_pad "
                                  f"guarantee",
                                  round(slope, 1), steep)
                    issues.append(issue)
                    lot_warnings.setdefault(lot_id, []).append(issue)
                elif mode != "flatten_pad" and slope > limit_slope:
                    issue = _warn("slope_risk", base,
                                  f"lot {lot_id} sits on a cell with mean "
                                  f"slope {slope:.1f} deg above the "
                                  f"conform limit {limit_slope:.0f} deg "
                                  f"and the lot is not flatten_pad",
                                  round(slope, 1), limit_slope)
                    issues.append(issue)
                    lot_warnings.setdefault(lot_id, []).append(issue)

    # -- boundaries ---------------------------------------------------------
    for bidx, boundary in enumerate(boundaries):
        if not isinstance(boundary, dict):
            continue
        base = _path("$", "boundaries", f"[{bidx}]")
        kind = boundary.get("kind")
        if kind in UNAVAILABLE_BOUNDARY_KINDS:
            issues.append(_err("boundary_capability_unavailable", _path(base, "kind"),
                               f"boundary kind {kind!r} is a kit-brief capability "
                               f"gap (no measured census) and fails closed"))
        if kind in UNAVAILABLE_BOUNDARY_SPACING:
            issues.append(_err("boundary_fence_spacing_unavailable", _path(base, "kind"),
                               f"boundary kind {kind!r} has no measured spacing "
                               f"rule (capability gap fence_spacing) and fails "
                               f"closed"))
        poly = boundary.get("polygon")
        if isinstance(poly, list) and len(poly) >= 4:
            if poly[0] != poly[-1]:
                issues.append(_err("ring_not_closed", _path(base, "polygon"),
                                   "boundary polygon must be an explicitly "
                                   "closed ring (first == last vertex)"))
            ring = close_ring(poly)
            if not ring_is_simple(poly):
                issues.append(_err("polygon_invalid", _path(base, "polygon"),
                                   "boundary ring is degenerate or "
                                   "self-intersecting"))
            else:
                for gidx, gate in enumerate(boundary.get("gates") or []):
                    if not isinstance(gate, dict):
                        continue
                    gbase = _path(base, "gates", f"[{gidx}]")
                    gpos = gate.get("position")
                    if isinstance(gpos, list) and len(gpos) == 2 and \
                            _is_finite_pt(gpos):
                        gx, gy = float(gpos[0]), float(gpos[1])
                        dist = point_polyline_distance((gx, gy), poly)
                        if dist > 128.0:
                            issues.append(_err("gate_off_ring", _path(gbase, "position"),
                                               f"gate ({gx:.1f}, {gy:.1f}) is "
                                               f"{dist:.0f} GU from the boundary "
                                               f"ring (hard tolerance 128 GU)",
                                               round(dist, 1), 128.0))
                        near = min(point_polyline_distance((gx, gy), pts)
                                   for pts in roads_pts.values()) if roads_pts else float("inf")
                        if near > 512.0:
                            issues.append(_err("gate_no_road_nearby", gbase,
                                               f"gate {gate.get('gate_id')!r} has "
                                               f"no planned road within 512 GU "
                                               f"(nearest {near:.0f} GU)",
                                               round(near, 1), 512.0))
                    heading = gate.get("heading_deg")
                    if isinstance(heading, (int, float)) and not isinstance(heading, bool):
                        if not math.isfinite(heading):
                            issues.append(_err("not_finite", _path(gbase, "heading_deg"),
                                               "gate heading is NaN or infinite"))
                    else:
                        issues.append(_err("wrong_type", _path(gbase, "heading_deg"),
                                           "gate heading_deg must be a number"))

    # -- features -----------------------------------------------------------
    features = plan.get("features") or []
    for fidx, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        base = _path("$", "features", f"[{fidx}]")
        fpos = feature.get("position")
        if isinstance(fpos, list) and len(fpos) == 2 and _is_finite_pt(fpos):
            fx, fy = float(fpos[0]), float(fpos[1])
            if not in_scope(fx, fy):
                issues.append(_err("out_of_scope", _path(base, "position"),
                                   f"feature anchor ({fx:.1f}, {fy:.1f}) is "
                                   f"outside the plan frame"))
            elif feature.get("kind") == "dock":
                state = bundle.door_anchor_state(fx, fy)
                if not state["water"]:
                    near = _nearest_water_distance(bundle, fx, fy)
                    if near > 1024.0:
                        issues.append(_warn(
                            "dock_water_distance", _path(base, "position"),
                            f"dock is {near:.0f} GU from the nearest water "
                            f"tile (docks may sit in water; placement depth "
                            f"envelope is solver-domain)",
                            round(near, 1), 1024.0))
        on_road = feature.get("on_road")
        if isinstance(on_road, str) and on_road not in road_ids and \
                on_road not in bundle.edge_ids and on_road not in bundle.node_ids:
            issues.append(_err("feature_on_road_unknown", _path(base, "on_road"),
                               f"on_road {on_road!r} does not reference a plan "
                               f"road or aligned-centerline edge/node id"))

    # -- terrain edits ------------------------------------------------------
    for eidx, edit in enumerate(plan.get("terrain_edits") or []):
        if not isinstance(edit, dict):
            continue
        base = _path("$", "terrain_edits", f"[{eidx}]")
        poly = edit.get("polygon")
        if isinstance(poly, list) and len(poly) >= 3:
            if not ring_is_simple(poly):
                issues.append(_err("polygon_invalid", _path(base, "polygon"),
                                   "edit polygon is degenerate or self-intersecting"))
            else:
                for pt in poly:
                    if not in_scope(pt[0], pt[1]):
                        issues.append(_err("edit_out_of_scope", _path(base, "polygon"),
                                           f"edit vertex {pt} is outside the "
                                           f"target cells / plan frame"))
                        break
        linked = edit.get("linked_to")
        if isinstance(linked, list):
            if not linked:
                issues.append(_err("edit_unlinked", _path(base, "linked_to"),
                                   "terrain edits must be linked to at least "
                                   "one plan element (no orphan terraforming)"))
            for ref in linked:
                if isinstance(ref, str) and ref not in all_ids:
                    issues.append(_err("edit_link_unknown", _path(base, "linked_to"),
                                       f"linked_to {ref!r} does not reference "
                                       f"any lot/feature/road/boundary id"))
        else:
            issues.append(_err("wrong_type", _path(base, "linked_to"),
                               "linked_to must be an array of plan ids"))
        # delta proxy: absolute target vs surveyed cell medians (soft)
        target = edit.get("target_height_gu")
        falloff = edit.get("falloff_gu")
        if isinstance(target, (int, float)) and not isinstance(target, bool) and \
                math.isfinite(target) and isinstance(poly, list) and \
                all(_is_finite_pt(p) for p in poly):
            covered = _cells_covered(poly, cells_by_gu,
                                     bundle.survey_frame["origin_gu"])
            if covered:
                med = [c["elev_med_gu"] for c in covered
                       if isinstance(c.get("elev_med_gu"), (int, float))]
                if med:
                    worst = max(abs(target - m) for m in med)
                    if worst > MAX_EDIT_DELTA_GU:
                        issues.append(_warn(
                            "edit_delta_proxy_exceeds_encoding_bound",
                            _path(base, "target_height_gu"),
                            f"|target - cell median| reaches {worst:.0f} GU "
                            f"vs the ±{MAX_EDIT_DELTA_GU:.0f} GU per-vertex "
                            f"encoding bound (proxy on surveyed cell medians; "
                            f"exact delta is computed at placement)",
                            round(worst, 1), MAX_EDIT_DELTA_GU))

    # -- wilderness hints ---------------------------------------------------
    for hidx, hint in enumerate(plan.get("wilderness_hints") or []):
        if not isinstance(hint, dict):
            continue
        base = _path("$", "wilderness_hints", f"[{hidx}]")
        density = hint.get("density")
        if isinstance(density, (int, float)) and not isinstance(density, bool) and \
                density < 0.0:
            issues.append(_err("hint_density_negative", _path(base, "density"),
                               f"density {density} must be >= 0", density, 0))

    # -- deterministic ordering + summary -----------------------------------
    issues.sort(key=lambda i: (i.path, i.code, i.message))
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    summary = {
        "plan_id": plan_id,
        "schema_version": plan.get("schema_version"),
        "issue_count": len(issues),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "sections": {
            "districts": len(districts),
            "roads": len(roads),
            "lots": len(lots),
            "boundaries": len(boundaries),
            "features": len(features),
            "terrain_edits": len(plan.get("terrain_edits") or []),
            "texture_zones": len(zones),
            "wilderness_hints": len(plan.get("wilderness_hints") or []),
        },
        "lot_resolution": [resolution[l.get("lot_id")] for l in lots
                           if isinstance(l, dict) and l.get("lot_id") in resolution],
        "lot_warnings": {lot_id: [i.to_dict() for i in sorted(issues,
                                                              key=lambda i: (i.path, i.code))]
                         for lot_id, issues in sorted(lot_warnings.items())},
        "external_references": {
            "aligned_centerline_edge_ids": len(bundle.edge_ids),
            "aligned_centerline_node_ids": len(bundle.node_ids),
            "measured_map_edge_exits": len(bundle.map_exits),
            "aligned_road_product_sha256": getattr(
                getattr(bundle, "aligned_network", None), "product_sha256", None),
        },
        "frame": {
            "origin_gu": bundle.survey_frame["origin_gu"],
            "site_span_gu": bundle.survey_frame["site_span_gu"],
            "field_spacing_gu": bundle.survey_frame["field_spacing_gu"],
            "tile_size_gu": TILE_SIZE_GU,
        },
        "warning_codes": sorted({i.code for i in warnings}),
    }
    return {
        "valid": not errors,
        "plan_id": plan_id,
        "schema_version": plan.get("schema_version"),
        "issue_count": len(issues),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "issues": [i.to_dict() for i in issues],
        "summary": summary,
        "input_hashes": dict(sorted(bundle.hashes.items())),
    }


def _candidate_stamps(bundle: Bundle, request: dict) -> list[str]:
    """Complete eligible candidate set for a non-explicit request:
    building_type, plus size_class and multi_shell when constrained."""
    btype = request.get("building_type")
    size = request.get("size_class")
    multi = request.get("multi_shell")
    out = []
    for stamp in bundle.eligible_stamps:
        if stamp.get("building_type") != btype:
            continue
        if size is not None and stamp.get("size_class") != size:
            continue
        if multi is not None and bool(stamp.get("multi_shell")) != bool(multi):
            continue
        out.append(stamp["stamp_id"])
    return sorted(out)


def _select_stamp(bundle: Bundle, candidates: list[str]) -> str:
    """Shared deterministic selector (v1): smallest footprint hull area,
    ties broken by sorted stamp_id -- the D-PLACE sorted-``stamp_id`` tie
    break, without D-PLACE's per-lot seeded ranking (solver-stage T1.2).

    Areas come from the accepted kit-brief stamp records (the same source
    the planner sees); the hull geometry itself comes from the pinned
    D-STAMP library."""
    brief_area = {s["stamp_id"]: s.get("footprint_hull_area_gu2", 0.0)
                  for s in bundle.eligible_stamps}

    def area_key(stamp_id: str) -> tuple[float, str]:
        return (float(brief_area.get(stamp_id, 0.0)), stamp_id)

    return min(candidates, key=area_key)


def _check_footprint(world: list, base: str, lot_id: str, issues: list,
                     bundle: Bundle) -> None:
    """Hard exact-footprint checks for a resolved stamp (explicit or
    selector): scope, buildable/water coverage of covered tiles."""
    for pt in world:
        if not in_scope(pt[0], pt[1]):
            issues.append(_err("footprint_out_of_scope", _path(base, "position"),
                               f"lot {lot_id} yawed footprint vertex {pt} "
                               f"escapes the plan frame"))
            return
    covered = tiles_covered_by_ring(world)
    water = [t for t in covered if bundle.tile_water(*t)]
    if water:
        issues.append(_err("footprint_in_water", _path(base, "position"),
                           f"lot {lot_id} footprint covers water tile(s) "
                           f"{water[:4]}{'...' if len(water) > 4 else ''}",
                           [list(t) for t in water[:4]], 0))
    unbuildable = [t for t in covered if not bundle.tile_buildable(*t)]
    if unbuildable:
        issues.append(_err("footprint_unbuildable", _path(base, "position"),
                           f"lot {lot_id} footprint covers non-buildable "
                           f"tile(s) {unbuildable[:4]}"
                           f"{'...' if len(unbuildable) > 4 else ''}",
                           [list(t) for t in unbuildable[:4]], 1))


def _external_anchor_point(bundle: Bundle, ref: str) -> Optional[tuple]:
    """Plan-frame anchor point of an external reference (for the soft
    proximity diagnostic): edge -> its aligned polyline start, node ->
    aligned position, exit -> first crossing point."""
    plan_origin = tuple(bundle.survey_frame["origin_gu"])
    if ref in bundle.map_exits:
        pts = bundle.map_exits[ref]["points"]
        return tuple(pts[0]) if pts else None
    if ref in bundle.edge_by_id:
        edge = bundle.edge_by_id[ref]
        start = _edge_smooth_chain(edge)[0]
        return (start[0] - plan_origin[0], start[1] - plan_origin[1])
    if ref in bundle.node_by_id:
        node = bundle.node_by_id[ref]
        pos = node.position_gu if not isinstance(node, dict) else node["position_gu"]
        return (pos[0] - plan_origin[0], pos[1] - plan_origin[1])
    return None


def _orphan_road_components(roads: list, road_external: dict) -> list[list[str]]:
    """Connected components of the road graph (plan-internal connections
    only) that contain no external reference -> orphan components."""
    road_ids = {r.get("road_id") for r in roads if isinstance(r, dict)
                and isinstance(r.get("road_id"), str)}
    adj: dict[str, set] = {rid: set() for rid in road_ids}
    for road in roads:
        if not isinstance(road, dict) or not isinstance(road.get("road_id"), str):
            continue
        rid = road["road_id"]
        for ref in road.get("connects") or []:
            if isinstance(ref, str) and ref in road_ids and ref != rid:
                adj[rid].add(ref)
                adj[ref].add(rid)
    seen: set = set()
    components: list[list[str]] = []
    for rid in sorted(road_ids):
        if rid in seen:
            continue
        stack = [rid]
        comp: list[str] = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adj[cur] - seen)
        if not any(road_external.get(c) for c in comp):
            components.append(sorted(comp))
    return sorted(components)


def _nearest_point_on_polyline(p: tuple, polyline: list) -> tuple:
    best, best_t = None, float("inf")
    for a, b in zip(polyline, polyline[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if dx == 0 and dy == 0:
            d = math.hypot(p[0] - a[0], p[1] - a[1])
            if d < best_t:
                best_t, best = d, a
            continue
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) /
                         (dx * dx + dy * dy)))
        q = (a[0] + t * dx, a[1] + t * dy)
        d = math.hypot(p[0] - q[0], p[1] - q[1])
        if d < best_t:
            best_t, best = d, q
    return best


def _cells_lookup(survey: dict) -> dict:
    """Grid (cell_x, cell_y) -> cell record from the site survey."""
    out = {}
    for cell in survey.get("cells", []):
        grid = cell.get("grid")
        if isinstance(grid, list) and len(grid) == 2:
            out[(int(grid[0]), int(grid[1]))] = cell
    return out


def _cells_covered(polygon: list, cells_by_gu: dict, origin_gu: list) -> list:
    """Survey cell records whose rectangle intersects the polygon bbox.
    Cells are computed from plan-frame GU via the accepted survey origin
    (cell = floor((origin + gu) / 8192))."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_cx = int(math.floor((origin_gu[0] + min(xs)) / 8192.0))
    max_cx = int(math.floor((origin_gu[0] + max(xs)) / 8192.0))
    min_cy = int(math.floor((origin_gu[1] + min(ys)) / 8192.0))
    max_cy = int(math.floor((origin_gu[1] + max(ys)) / 8192.0))
    out = []
    for cx in range(min_cx, max_cx + 1):
        for cy in range(min_cy, max_cy + 1):
            cell = cells_by_gu.get((cx, cy))
            if cell is not None:
                out.append(cell)
    return out


def _nearest_water_distance(bundle: Bundle, x: float, y: float) -> float:
    tx, ty = gu_to_tile(x, y)
    best = float("inf")
    for wy in range(max(0, ty - 3), min(TILE_SIDE, ty + 4)):
        for wx in range(max(0, tx - 3), min(TILE_SIDE, tx + 4)):
            if bundle.tile_water(wx, wy):
                cx = wx * TILE_SIZE_GU + TILE_SIZE_GU / 2.0
                cy = wy * TILE_SIZE_GU + TILE_SIZE_GU / 2.0
                best = min(best, math.hypot(cx - x, cy - y))
    if best == float("inf"):
        # no water within the probe window: scan the full mask (rare)
        for wy in range(TILE_SIDE):
            for wx in range(TILE_SIDE):
                if bundle.tile_water(wx, wy):
                    cx = wx * TILE_SIZE_GU + TILE_SIZE_GU / 2.0
                    cy = wy * TILE_SIZE_GU + TILE_SIZE_GU / 2.0
                    best = min(best, math.hypot(cx - x, cy - y))
    return best


def validate_plan_file(plan_path: Path | str, bundle: Bundle) -> dict:
    """Load one plan file and validate it against the bundle."""
    plan = _load_json(Path(plan_path), "city plan")
    result = validate_plan(plan, bundle)
    result["plan_file_sha256"] = _sha256_file(Path(plan_path))
    result["input_hashes"]["plan"] = result["plan_file_sha256"]
    return result


# ---------------------------------------------------------------------------
# Deterministic JSON Schema emission (kept in agreement with DOCUMENT_SPEC;
# see tests/test_cityplan.py)
# ---------------------------------------------------------------------------

def _schema_type(spec: dict) -> dict:
    kind = spec["type"]
    if kind == "object":
        return {"type": "object", "additionalProperties": False,
                "required": sorted(k for k, s in spec.get("keys", {}).items()
                                   if s.get("required")),
                "properties": {k: _schema_type(s) for k, s in spec.get("keys", {}).items()}}
    if kind == "array":
        out = {"type": "array"}
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
        return out
    if kind == "int":
        return {"type": "integer"}
    if kind == "num":
        return {"type": "number"}
    if kind == "bool":
        return {"type": "boolean"}
    if kind == "point":
        return {"type": "array", "prefixItems": [{"type": "number"},
                                                 {"type": "number"}],
                "items": False, "minItems": 2, "maxItems": 2}
    if kind == "int2":
        return {"type": "array", "prefixItems": [{"type": "integer"},
                                                 {"type": "integer"}],
                "items": False, "minItems": 2, "maxItems": 2}
    if kind == "points":
        out = {"type": "array",
               "items": {"type": "array", "prefixItems": [{"type": "number"},
                                                          {"type": "number"}],
                         "items": False, "minItems": 2, "maxItems": 2}}
        if spec.get("min_points"):
            out["minItems"] = spec["min_points"]
        return out
    raise BundleError(f"internal schema error: unknown type {kind!r}")


def emit_json_schema() -> dict:
    """Emit the machine-readable JSON Schema (draft 2020-12) for the D-PLAN
    v1 document, derived from DOCUMENT_SPEC so the two cannot drift."""
    defs = {}
    for name, spec in ITEM_SPEC.items():
        resolved = _resolve_item_spec(spec if not isinstance(spec, dict) or
                                      "type" in spec else name)
        if resolved.get("type") == "object":
            defs[name] = resolved
    # sections whose items are free strings (conn_ref / link_ref) are
    # handled inline by DOCUMENT_SPEC's item tag; keep only object items.
    defs = {k: v for k, v in defs.items()
            if k not in ("conn_ref", "link_ref")}
    doc = _schema_type({"type": "object", "keys": DOCUMENT_SPEC})
    doc["properties"]["districts"]["items"] = {"$ref": "#/$defs/district"}
    doc["properties"]["roads"]["items"] = {"$ref": "#/$defs/road"}
    doc["properties"]["lots"]["items"] = {"$ref": "#/$defs/lot"}
    doc["properties"]["boundaries"]["items"] = {"$ref": "#/$defs/boundary"}
    doc["properties"]["features"]["items"] = {"$ref": "#/$defs/feature"}
    doc["properties"]["terrain_edits"]["items"] = {"$ref": "#/$defs/terrain_edit"}
    doc["properties"]["texture_zones"]["items"] = {"$ref": "#/$defs/texture_zone"}
    doc["properties"]["wilderness_hints"]["items"] = {"$ref": "#/$defs/wilderness_hint"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "cityforge/dplan/city_plan_schema_v1.json",
        "title": "Cityforge D-PLAN city_plan.json strict schema v1",
        "description": "Strict machine-readable contract for city_plan.json "
                       "(T1.1). additionalProperties:false applies recursively; "
                       "semantic gates (frame pin, id namespaces, references, "
                       "geometry) are enforced by src/procgen/cityplan.py.",
        "type": "object",
        "additionalProperties": False,
        **{k: v for k, v in doc.items() if k != "additionalProperties"},
        "$defs": defs,
    }
