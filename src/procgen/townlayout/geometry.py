"""Planar construction helpers for V2 townlayout (Shapely, no silent repair).

Purpose
-------
Normalize open CCW rings, build Shapely polygons, and run logged
destructive difference.  Invalid geometry raises ``TownLayoutError``
whose message starts with ``invalid_polygon:``.  Silent validity
repair is forbidden.

Inputs
------
Rings as ``list[list[float]]`` of ``[x, y]`` in plan GU.

Outputs
-------
``normalize_ring`` / ``destructive_difference`` result dicts, or a
Shapely ``Polygon``.

Pipeline position
-----------------
V2 townlayout Phase 2 geometry/RNG; no generation.
"""

from __future__ import annotations

import math
from typing import Any

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate

from procgen.cityplan import ring_area

from .constants import VERTEX_EPS_GU
from .validate import TownLayoutError

__all__ = [
    "VERTEX_EPS_GU",
    "normalize_ring",
    "polygon_from_ring",
    "destructive_difference",
    "simple_polygon_parts",
]


def _finite_xy(x: float, y: float, path: str) -> tuple[float, float]:
    if not math.isfinite(x) or not math.isfinite(y):
        raise TownLayoutError(f"non_finite_number: {path}")
    return float(x), float(y)


def _same(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return (abs(a[0] - b[0]) <= VERTEX_EPS_GU
            and abs(a[1] - b[1]) <= VERTEX_EPS_GU)


def _ring_to_coords(ring: list) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for idx, pt in enumerate(ring):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise TownLayoutError("invalid_polygon: vertex is not [x, y]")
        x, y = pt[0], pt[1]
        if isinstance(x, bool) or isinstance(y, bool):
            raise TownLayoutError("invalid_polygon: vertex is not [x, y]")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TownLayoutError("invalid_polygon: vertex is not [x, y]")
        coords.append(_finite_xy(float(x), float(y), f"[{idx}]"))
    return coords


def normalize_ring(ring: list) -> dict[str, Any]:
    """Return an open CCW ring with consecutive duplicates removed."""
    coords = _ring_to_coords(ring)
    if len(coords) >= 2 and _same(coords[0], coords[-1]):
        coords = coords[:-1]
    open_before_dedup = len(coords)
    deduped: list[tuple[float, float]] = []
    for pt in coords:
        if deduped and _same(deduped[-1], pt):
            continue
        deduped.append(pt)
    dropped_vertices = open_before_dedup - len(deduped)
    if len(deduped) < 3:
        raise TownLayoutError("invalid_polygon: fewer than 3 vertices")
    as_lists = [[p[0], p[1]] for p in deduped]
    area = ring_area(as_lists)
    reversed_flag = False
    if abs(area) <= 1e-6:
        raise TownLayoutError("invalid_polygon: zero area")
    if area < 0:
        as_lists.reverse()
        reversed_flag = True
        area = ring_area(as_lists)
        if area <= 1e-6:
            raise TownLayoutError("invalid_polygon: zero area")
    return {
        "ring": as_lists,
        "reversed": reversed_flag,
        "dropped_vertices": dropped_vertices,
        "area": area,
    }


def polygon_from_ring(ring: list) -> Polygon:
    """Build a valid Shapely polygon from a ring.  Does not buffer."""
    result = normalize_ring(ring)
    polygon = Polygon(result["ring"])
    if (not polygon.is_valid) or polygon.area <= 0:
        raise TownLayoutError("invalid_polygon: shapely invalid")
    return polygon


def simple_polygon_parts(geometry, *, area_tolerance: float = 1e-5) -> list[Polygon]:
    """Deterministically decompose polygonal geometry into hole-free parts.

    Water clipping can leave an internal lake as a hole.  Rings in the
    townlayout contract have no interior-ring field, so dropping the hole (or
    taking only the largest exterior) would put geometry back in the lake.
    Triangulation followed by intersection is a deterministic equivalent of
    cutting every hole to a visible exterior vertex: every emitted piece has
    an exterior ring only and the union has the original area.
    """
    if geometry.is_empty:
        return []
    raw = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    result: list[Polygon] = []
    for poly in raw:
        if not isinstance(poly, Polygon) or not poly.is_valid:
            raise TownLayoutError("invalid_polygon: non-simple clipping result")
        pieces = [poly] if not poly.interiors else [
            clipped
            for tri in triangulate(poly)
            for clipped in ([tri.intersection(poly)]
                            if isinstance(tri.intersection(poly), Polygon)
                            else list(tri.intersection(poly).geoms))
            if isinstance(clipped, Polygon) and not clipped.is_empty
        ]
        area = sum(float(p.area) for p in pieces)
        if abs(area - float(poly.area)) > area_tolerance:
            raise TownLayoutError("invalid_polygon: hole decomposition area loss")
        result.extend(p for p in pieces if p.area > 0.0 and not p.interiors)
    result.sort(key=lambda p: (-float(p.area), float(p.centroid.x), float(p.centroid.y)))
    return result


def _exterior_ring(polygon: Polygon) -> list[list[float]]:
    coords = list(polygon.exterior.coords)
    return normalize_ring([[c[0], c[1]] for c in coords])["ring"]


def destructive_difference(subject_ring: list, clip_rings: list) -> dict[str, Any]:
    """Subtract clip polygons from subject.  Holes are forbidden in v1."""
    subject = polygon_from_ring(subject_ring)
    area_before = float(subject.area)
    for clip in clip_rings:
        subject = subject.difference(polygon_from_ring(clip))
    parts: list[Polygon] = []
    if subject.is_empty:
        return {
            "rings": [],
            "topology_changed": True,
            "discarded_slivers": 0,
            "area_before": area_before,
            "area_after": 0.0,
        }
    if isinstance(subject, Polygon):
        geoms = [subject]
    elif isinstance(subject, MultiPolygon):
        geoms = list(subject.geoms)
    else:
        raise TownLayoutError("invalid_polygon: non-polygonal difference result")
    discarded_slivers = 0
    kept: list[Polygon] = []
    for geom in geoms:
        if len(geom.interiors) > 0:
            raise TownLayoutError("invalid_polygon: holes not supported in v1")
        if geom.area < 1.0:
            discarded_slivers += 1
            continue
        kept.append(geom)
    rings = [_exterior_ring(geom) for geom in kept]
    area_after = float(sum(geom.area for geom in kept))
    topology_changed = (
        discarded_slivers > 0
        or len(kept) != 1
        or abs(area_after - area_before) > 1e-3
    )
    return {
        "rings": rings,
        "topology_changed": topology_changed,
        "discarded_slivers": discarded_slivers,
        "area_before": area_before,
        "area_after": area_after,
    }
