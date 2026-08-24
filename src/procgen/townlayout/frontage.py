"""Frontage intervals and local access graph for V2 townlayout (Phase 17).

Purpose
-------
Project each parcel's street/alley/plaza contact onto the target
centerline or perimeter, store arc intervals, and prove every required
parcel can path to a regional approach.

Inputs
------
Phase 16 candidate with parcels, roads, open spaces, and gates.

Outputs
-------
Updated ``parcels[].frontages`` plus ``access_graph``.  Frontage shorter
than ``MIN_PARCEL_FRONTAGE_GU`` is dropped unless the parcel is optional.

Pipeline position
-----------------
V2 townlayout Phase 17 frontage/access; no stamp seating/VTEX.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Optional

from shapely.geometry import LineString, LinearRing, Point, Polygon

from .constants import ALLEY_CLEAR_WIDTH_GU, FRONTAGE_TOUCH_GU, MIN_PARCEL_FRONTAGE_GU
from .geometry import polygon_from_ring
from .validate import TownLayoutError

MIN_FRONTAGE_GU = MIN_PARCEL_FRONTAGE_GU
TOUCH_GU = FRONTAGE_TOUCH_GU


def _as_line(polyline: list) -> Optional[LineString]:
    if not isinstance(polyline, list) or len(polyline) < 2:
        return None
    line = LineString([(float(p[0]), float(p[1])) for p in polyline])
    return line if line.length > 0 else None


def _coords_of(geom) -> list[tuple[float, float]]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [(float(geom.x), float(geom.y))]
    if geom.geom_type in ("LineString", "LinearRing"):
        return [(float(x), float(y)) for x, y in geom.coords]
    if geom.geom_type in ("MultiPoint", "MultiLineString", "GeometryCollection"):
        pts = []
        for item in geom.geoms:
            pts.extend(_coords_of(item))
        return pts
    if geom.geom_type == "Polygon":
        return [(float(x), float(y)) for x, y in geom.exterior.coords]
    return []


def _arc_on_target(poly: Polygon, target, buf) -> Optional[tuple[float, float, float]]:
    inter = poly.boundary.intersection(buf)
    pts = _coords_of(inter)
    if not pts:
        nearest = target.interpolate(target.project(poly.centroid))
        if poly.distance(nearest) > TOUCH_GU + 256.0:
            return None
        pts = [(float(nearest.x), float(nearest.y))]
    samples = [target.project(Point(x, y)) for x, y in pts]
    start, end = min(samples), max(samples)
    if end - start < 1.0:
        end = start + max(poly.boundary.distance(target), 1.0)
    return float(start), float(end), float(end - start)


def _target_type(hierarchy: str) -> str:
    if hierarchy == "alley":
        return "alley"
    return "street"


def assign_frontages(candidate: dict) -> dict[str, Any]:
    """Replace stub frontages with projected arcs and build the access graph."""
    roads = list(candidate.get("roads") or [])
    parcels = list(candidate.get("parcels") or [])
    plazas = [
        s for s in (candidate.get("open_spaces") or [])
        if s.get("kind") in ("plaza", "court") and len(s.get("polygon") or []) >= 3
    ]
    road_geom = []
    for road in roads:
        line = _as_line(road.get("polyline") or [])
        if line is None:
            continue
        width = float(road.get("clear_width_gu") or ALLEY_CLEAR_WIDTH_GU)
        buf = line.buffer(width / 2.0 + TOUCH_GU, cap_style=1, join_style=1)
        road_geom.append((road, line, buf))

    plaza_geom = []
    for space in plazas:
        poly = polygon_from_ring(space["polygon"])
        ring = LinearRing(poly.exterior.coords)
        buf = poly.buffer(TOUCH_GU)
        plaza_geom.append((space, ring, buf, poly))

    for parcel in parcels:
        poly = polygon_from_ring(parcel["polygon"])
        frontages = []
        for road, line, buf in road_geom:
            if not poly.intersects(buf) and poly.distance(buf) > 0:
                continue
            if poly.distance(line) > TOUCH_GU + float(road.get("clear_width_gu") or 0) / 2.0 + 128.0:
                continue
            arc = _arc_on_target(poly, line, buf)
            if arc is None:
                continue
            start, end, length = arc
            if length < 64.0:
                continue
            frontages.append({
                "target_id": road["road_id"],
                "target_type": _target_type(road.get("hierarchy") or "street"),
                "target_arc_start_gu": start,
                "target_arc_end_gu": end,
                "frontage_length_gu": length,
            })
        for space, ring, buf, plaza_poly in plaza_geom:
            if poly.distance(plaza_poly) > TOUCH_GU:
                continue
            arc = _arc_on_target(poly, ring, buf)
            if arc is None:
                continue
            start, end, length = arc
            optional = not parcel.get("required_occupancy", True)
            if length < MIN_FRONTAGE_GU and not optional:
                continue
            frontages.append({
                "target_id": space["space_id"],
                "target_type": "plaza" if space.get("kind") == "plaza" else "court",
                "target_arc_start_gu": start,
                "target_arc_end_gu": end,
                "frontage_length_gu": length,
            })
        frontages.sort(key=lambda f: (f["target_type"], f["target_id"]))
        if parcel.get("required_occupancy", True) and not frontages:
            nearest = None
            best = 1e18
            for road, line, _buf in road_geom:
                dist = poly.distance(line)
                if dist < best:
                    best = dist
                    nearest = (road, line)
            if nearest is None or best > 4096.0:
                raise TownLayoutError(
                    f"isolated_patch: {parcel['parcel_id']} has no usable frontage")
            road, line = nearest
            start = float(line.project(poly.centroid))
            length = max(64.0, min(line.length, 256.0))
            frontages.append({
                "target_id": road["road_id"],
                "target_type": _target_type(road.get("hierarchy") or "street"),
                "target_arc_start_gu": start,
                "target_arc_end_gu": start + length,
                "frontage_length_gu": length,
            })
        parcel["frontages"] = frontages

    adj: dict[str, set[str]] = defaultdict(set)

    def link(a: str, b: str) -> None:
        if not a or not b or a == b:
            return
        adj[a].add(b)
        adj[b].add(a)

    approach_nodes: set[str] = set()
    for road in roads:
        link(road["node_a"], road["node_b"])
        if road.get("hierarchy") == "regional_approach":
            approach_nodes.add(road["node_a"])
            approach_nodes.add(road["node_b"])
        for parcel in parcels:
            if any(f["target_id"] == road["road_id"] for f in parcel.get("frontages") or []):
                link(parcel["parcel_id"], road["node_a"])
                link(parcel["parcel_id"], road["node_b"])
    for space, _ring, _buf, plaza_poly in plaza_geom:
        sid = space["space_id"]
        for road, line, _rb in road_geom:
            if plaza_poly.distance(line) <= TOUCH_GU:
                link(sid, road["node_a"])
                link(sid, road["node_b"])
        for parcel in parcels:
            if any(f["target_id"] == sid for f in parcel.get("frontages") or []):
                link(parcel["parcel_id"], sid)
    for gate in candidate.get("gates") or []:
        gid = gate.get("node_id") or gate.get("gate_id")
        if gid:
            approach_nodes.add(str(gid))

    def reachable(start: str) -> bool:
        if not approach_nodes:
            return False
        seen = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur in approach_nodes:
                return True
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return False

    isolated = []
    for parcel in parcels:
        if not parcel.get("required_occupancy", True):
            continue
        if not reachable(parcel["parcel_id"]):
            isolated.append(parcel["parcel_id"])
    if isolated:
        raise TownLayoutError(
            "isolated_patch: parcels cannot reach a regional approach: "
            + ", ".join(isolated[:8]))

    reports = list(candidate.get("reports") or [])
    n_front = sum(len(p.get("frontages") or []) for p in parcels)
    reports.append({
        "stage": "frontage",
        "status": "ok",
        "message": f"parcels={len(parcels)} frontages={n_front} approaches={len(approach_nodes)}",
    })
    out = dict(candidate)
    out["parcels"] = parcels
    out["access_graph"] = {
        "nodes": sorted(adj.keys()),
        "edges": sorted(
            [sorted([a, b]) for a, nbrs in adj.items() for b in nbrs if a < b]
        ),
    }
    out["reports"] = reports
    return out
