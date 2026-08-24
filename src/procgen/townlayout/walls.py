"""Abstract palisade and approach-driven gates for V2 townlayout (Phase 8).

Purpose
-------
Union inner-core patches into a palisade planning polygon, simplify, and
place one gate per mandatory approach where the inward ray meets the wall.

Inputs
------
Phase 7 candidate (inner flags, anchors), SiteContext, TownBrief, approaches.

Outputs
-------
``wall`` (palisade or null), ``gates``, ``inside_wall`` on core patches.

Pipeline position
-----------------
V2 townlayout Phase 8 walls/gates; no meshes, parcels, or VTEX.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .constants import JUNCTION_MERGE_GU, VERTEX_EPS_GU
from .geometry import normalize_ring, polygon_from_ring
from .site_context import SiteContext, _plan_to_px, diagnostic_view
from .validate import TownLayoutError

SIMPLIFY_GU = 64.0
SMOOTH_MAX_MOVE_GU = 32.0
GATE_MAX_DIST_GU = 4096.0
GATE_MIN_ARC_GU = 800.0
GATE_NUDGE_GU = 200.0
RAY_T_GU = 16384.0
WALL_STRIP_DEPTH_GU = 256.0


def _core_patches(patches: list[dict]) -> list[dict]:
    return [
        p for p in patches
        if p.get("inside_city") and p.get("morphology_region") != "outskirts"
    ]


def _drop_thin_concavities(poly: Polygon) -> Polygon:
    """Morphological opening; skip if the result is unusable."""
    half = JUNCTION_MERGE_GU / 2.0
    opened = poly.buffer(-half).buffer(half)
    if opened.is_empty or opened.area <= 0:
        return poly
    if opened.geom_type == "MultiPolygon":
        opened = max(opened.geoms, key=lambda g: g.area)
    if opened.geom_type != "Polygon" or not opened.is_valid:
        return poly
    if opened.area < 0.5 * poly.area:
        return poly
    return opened


def _smooth_ring(ring: list[list[float]], ctx: SiteContext) -> list[list[float]]:
    n = len(ring)
    if n < 4:
        return ring
    out = []
    for i, pt in enumerate(ring):
        prev = ring[(i - 1) % n]
        nxt = ring[(i + 1) % n]
        mx = (prev[0] + pt[0] + nxt[0]) / 3.0
        my = (prev[1] + pt[1] + nxt[1]) / 3.0
        dx, dy = mx - pt[0], my - pt[1]
        dist = math.hypot(dx, dy)
        if dist > SMOOTH_MAX_MOVE_GU or dist <= VERTEX_EPS_GU:
            out.append([float(pt[0]), float(pt[1])])
            continue
        if not ctx.sample(mx, my).get("buildable", False):
            out.append([float(pt[0]), float(pt[1])])
            continue
        out.append([mx, my])
    return normalize_ring(out)["ring"]


def _build_palisade(core: list[dict], ctx: SiteContext) -> dict:
    union = unary_union([polygon_from_ring(p["polygon"]) for p in core])
    if union.geom_type == "MultiPolygon":
        union = max(union.geoms, key=lambda g: g.area)
    if union.geom_type != "Polygon" or union.area <= 0:
        raise TownLayoutError("invalid_polygon: palisade union is not a polygon")
    union = _drop_thin_concavities(union)
    source = normalize_ring([[c[0], c[1]] for c in union.exterior.coords])["ring"]
    simplified = union.simplify(SIMPLIFY_GU, preserve_topology=True)
    if simplified.geom_type != "Polygon" or not simplified.is_valid or simplified.area <= 0:
        simplified = polygon_from_ring(source)
    planning = normalize_ring(
        [[c[0], c[1]] for c in simplified.exterior.coords])["ring"]
    planning = _smooth_ring(planning, ctx)
    return {
        "kind": "palisade",
        "source_perimeter": source,
        "planning_polygon": planning,
    }


def _as_lines(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        out = []
        for part in geom.geoms:
            out.extend(_as_lines(part))
        return out
    return []


def _ray_hit(crossing: list[float], tangent: list[float],
             wall: Polygon) -> Optional[tuple[float, float]]:
    x0, y0 = float(crossing[0]), float(crossing[1])
    origin = Point(x0, y0)
    nearest = wall.boundary.interpolate(wall.boundary.project(origin))
    near_pt = (float(nearest.x), float(nearest.y))
    tx, ty = float(tangent[0]), float(tangent[1])
    n = math.hypot(tx, ty)
    if n <= VERTEX_EPS_GU:
        tx, ty = wall.centroid.x - x0, wall.centroid.y - y0
        n = math.hypot(tx, ty)
        if n <= VERTEX_EPS_GU:
            return None
    tx, ty = tx / n, ty / n
    stub_ref = (x0 + tx * 512.0, y0 + ty * 512.0)

    def in_range(pt: tuple[float, float]) -> bool:
        return math.hypot(pt[0] - stub_ref[0], pt[1] - stub_ref[1]) <= GATE_MAX_DIST_GU

    if origin.distance(wall.boundary) <= SIMPLIFY_GU and in_range(near_pt):
        return near_pt
    candidates = [(tx, ty)]
    cx, cy = float(wall.centroid.x), float(wall.centroid.y)
    ctx, cty = cx - x0, cy - y0
    cn = math.hypot(ctx, cty)
    if cn > VERTEX_EPS_GU:
        candidates.append((ctx / cn, cty / cn))
    for dx, dy in candidates:
        ray = LineString([(x0, y0), (x0 + dx * RAY_T_GU, y0 + dy * RAY_T_GU)])
        hits = []
        inter = ray.intersection(wall.boundary)
        if inter.is_empty:
            continue
        if inter.geom_type == "Point":
            hits.append((float(inter.x), float(inter.y)))
        elif inter.geom_type == "MultiPoint":
            hits.extend((float(p.x), float(p.y)) for p in inter.geoms)
        else:
            for line in _as_lines(inter):
                for hx, hy, *_ in line.coords:
                    hits.append((float(hx), float(hy)))
        hits = [h for h in hits if in_range(h)]
        if not hits:
            continue
        hits.sort(key=lambda p: math.hypot(p[0] - stub_ref[0], p[1] - stub_ref[1]))
        return hits[0]
    if in_range(near_pt):
        return near_pt
    return None


def _arc_sep(ring: LineString, a: tuple[float, float], b: tuple[float, float]) -> float:
    d1 = ring.project(Point(a))
    d2 = ring.project(Point(b))
    delta = abs(d1 - d2)
    return min(delta, ring.length - delta)


def _point_on_ring(ring: LineString, dist: float) -> tuple[float, float]:
    length = ring.length
    d = dist % length
    if d < 0:
        d += length
    p = ring.interpolate(d)
    return (float(p.x), float(p.y))


def _place_gates(wall_poly: Polygon, approaches: list[dict],
                 candidate_id: str) -> list[dict]:
    mandatory = [a for a in approaches if a.get("mandatory", True)]
    if not mandatory:
        return []
    ring = LineString(wall_poly.exterior.coords)
    raw = []
    for approach in mandatory:
        hit = _ray_hit(
            approach["crossing_plan_gu"],
            approach["inward_tangent"],
            wall_poly,
        )
        if hit is None:
            raise TownLayoutError(
                f"gate_unreachable: {approach['approach_id']}")
        raw.append({
            "approach": approach,
            "position": hit,
        })
    # Merge same-edge gates that are closer than GATE_MIN_ARC_GU.
    kept: list[dict] = []
    used = set()
    for i, item in enumerate(raw):
        if i in used:
            continue
        group = [item]
        edge = item["approach"]["source_edge_id"]
        for j, other in enumerate(raw):
            if j <= i or j in used:
                continue
            if other["approach"]["source_edge_id"] != edge:
                continue
            if _arc_sep(ring, item["position"], other["position"]) < GATE_MIN_ARC_GU:
                group.append(other)
                used.add(j)
        sx = sum(g["position"][0] for g in group) / len(group)
        sy = sum(g["position"][1] for g in group) / len(group)
        proj = ring.interpolate(ring.project(Point(sx, sy)))
        kept.append({
            "approach": item["approach"],
            "position": (float(proj.x), float(proj.y)),
            "merged": [g["approach"]["approach_id"] for g in group],
        })
        used.add(i)

    # Nudge remaining close pairs (different source edges).
    kept.sort(key=lambda g: g["approach"]["approach_id"])
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            a, b = kept[i], kept[j]
            sep = _arc_sep(ring, a["position"], b["position"])
            if sep >= GATE_MIN_ARC_GU:
                continue
            da = ring.project(Point(b["position"]))
            db = ring.project(Point(a["position"]))
            # Move later approach_id away from the earlier one.
            direction = 1.0 if da >= db else -1.0
            new_d = da + direction * min(GATE_NUDGE_GU, GATE_MIN_ARC_GU - sep + 1.0)
            nudged = _point_on_ring(ring, new_d)
            if _arc_sep(ring, a["position"], nudged) < GATE_MIN_ARC_GU:
                raise TownLayoutError(
                    f"gate_too_close: {a['approach']['approach_id']} "
                    f"{b['approach']['approach_id']}")
            b["position"] = nudged

    gates = []
    for i, item in enumerate(kept):
        approach = item["approach"]
        tx, ty = approach["inward_tangent"]
        n = math.hypot(tx, ty) or 1.0
        gates.append({
            "gate_id": f"gate_{candidate_id}_{i:04d}",
            "position": [item["position"][0], item["position"][1]],
            "approach_id": approach["approach_id"],
            "outward_tangent": [-tx / n, -ty / n],
        })
    return gates


def _road_gate_node(product: dict, approach_id: str) -> tuple[list[float], str]:
    """Return the frozen Stage 05 arterial entry node for an approach.

    Stage 06 deliberately consumes, rather than recomputes, this node.  This
    is what makes a gate an exact road/ring node and prevents fortification
    from silently editing the accepted road geometry.
    """
    roads = [r for r in product.get("roads", [])
             if approach_id in (r.get("source_approach_ids") or [])
             and r.get("hierarchy") == "arterial"]
    if len(roads) != 1:
        raise TownLayoutError(f"gate_arterial_ambiguity: {approach_id}")
    road = roads[0]
    node_id = road["node_a"]
    node = next((n for n in product.get("nodes", []) if n.get("node_id") == node_id), None)
    if node is None:
        raise TownLayoutError(f"gate_node_missing: {approach_id}")
    return [float(node["position"][0]), float(node["position"][1])], node_id


def _wall_segments(ring: list[list[float]], gates: list[dict], roads: list[dict],
                   wall_poly: Polygon, ctx: Optional[SiteContext] = None) -> tuple[list[dict], list[dict]]:
    """Split gate arcs while retaining exact authoritative ring geometry."""
    boundary = LineString(ring + [ring[0]])
    cuts = sorted((boundary.project(Point(g["position"])), g["gate_id"]) for g in gates)
    if not cuts:
        raise TownLayoutError("unassigned_wall_arc: ring has no segments")
    vertex_dist = [boundary.project(Point(p)) for p in ring]
    stats = (ctx.stamp_footprint_stats if ctx is not None else {}) or {}
    required_depth = math.sqrt(float(stats.get("p10", 0.0))) + 128.0

    def exact_arc(start, end):
        interior = []
        for d in vertex_dist:
            for candidate in (d, d + boundary.length):
                if start < candidate < end:
                    interior.append(candidate)
        ds = [start] + sorted(set(interior)) + [end]
        return [[float(boundary.interpolate(d % boundary.length).x),
                 float(boundary.interpolate(d % boundary.length).y)] for d in ds]

    def segment_evidence(coords, start_gid, end_gid):
        line = LineString(coords)
        strip = wall_poly.intersection(line.buffer(WALL_STRIP_DEPTH_GU, cap_style=2, join_style=2))
        if strip.geom_type == "MultiPolygon":
            strip = max(strip.geoms, key=lambda g: g.area)
        if strip.geom_type == "GeometryCollection":
            polys = [g for g in strip.geoms if g.geom_type == "Polygon" and g.area > 0]
            strip = max(polys, key=lambda g: g.area) if polys else Polygon()
        if strip.geom_type != "Polygon" or strip.area <= 0:
            raise TownLayoutError("unassigned_wall_arc: strip is not polygonal")
        road_ids = sorted(r["road_id"] for r in roads
                          if LineString(r.get("polyline") or []).intersection(strip).length > 0)
        gate_buffers = []
        for gid, endpoint in ((start_gid, coords[0]), (end_gid, coords[-1])):
            if gid:
                gate_buffers.append(Point(endpoint).buffer(256.0))
        gate_junctions = unary_union(gate_buffers) if gate_buffers else Polygon()
        arterial_beyond_gate = False
        for road in roads:
            if road.get("hierarchy") != "arterial" or road.get("road_id") not in road_ids:
                continue
            contact = LineString(road.get("polyline") or []).intersection(strip)
            beyond = contact.difference(gate_junctions) if not gate_junctions.is_empty else contact
            if not beyond.is_empty and beyond.length > 1.0:
                arterial_beyond_gate = True
                break
        supports_depth = False
        if ctx is not None and required_depth > 0:
            supports_depth = True
            for fraction in (0.2, 0.5, 0.8):
                p = line.interpolate(line.length * fraction)
                q = line.interpolate(min(line.length, line.length * fraction + 1.0))
                tx, ty = q.x - p.x, q.y - p.y
                n = math.hypot(tx, ty) or 1.0
                nx, ny = -ty / n, tx / n  # left side of the CCW planning ring
                for depth in (required_depth * .5, required_depth, required_depth + 64.0):
                    probe = Point(p.x + nx * depth, p.y + ny * depth)
                    if (not wall_poly.covers(probe)
                            or not ctx.sample(probe.x, probe.y).get("buildable", False)):
                        supports_depth = False
                        break
                if not supports_depth:
                    break
        return {"line": line, "strip": strip, "road_ids": road_ids,
                "arterial_beyond_gate": arterial_beyond_gate,
                "supports_depth": supports_depth,
                "short": line.length < required_depth}

    segments, strips = [], []
    for arc_no, ((start, start_gid), (end, end_gid)) in enumerate(zip(cuts, cuts[1:] + cuts[:1])):
        if arc_no == len(cuts) - 1:
            end += boundary.length
        raw = exact_arc(start, end)
        simplified = LineString(raw).simplify(SIMPLIFY_GU, preserve_topology=True)
        breakpoints = [0]
        for point in list(simplified.coords)[1:-1]:
            idx = min(range(1, len(raw) - 1), key=lambda i: math.hypot(raw[i][0] - point[0], raw[i][1] - point[1]))
            if idx > breakpoints[-1]:
                breakpoints.append(idx)
        breakpoints.append(len(raw) - 1)
        local_evidence = []
        for edge_no, (ia, ib) in enumerate(zip(breakpoints, breakpoints[1:])):
            coords = raw[ia:ib + 1]
            local_evidence.append(segment_evidence(coords,
                                                   start_gid if edge_no == 0 else None,
                                                   end_gid if edge_no == len(breakpoints) - 2 else None))
        modes = ["wall_lane" if (not e["supports_depth"] or e["arterial_beyond_gate"])
                 else "backs_to_wall" for e in local_evidence]
        short_run = [False] * len(modes)
        stamp_width = math.sqrt(float(stats.get("p10", 0.0)))

        def has_local_connection(e):
            return any(roads_by_id[rid].get("hierarchy") in
                       ("regional_approach", "street", "lane", "alley")
                       for rid in e["road_ids"] if rid in roads_by_id)

        # Extend each lane run only to the nearest passing segment that has a
        # street/road connection (or to the gate at the arc boundary).  This
        # deliberately operates on the local evidence list, never on the
        # whole gate-to-gate arc.
        roads_by_id = {r["road_id"]: r for r in roads}
        run_start = 0
        while run_start < len(modes):
            if modes[run_start] != "wall_lane":
                run_start += 1
                continue
            run_end = run_start
            while run_end + 1 < len(modes) and modes[run_end + 1] == "wall_lane":
                run_end += 1
            run_length = sum(local_evidence[i]["line"].length for i in range(run_start, run_end + 1))
            if run_length >= stamp_width:
                left = run_start - 1
                if left < 0 and not start_gid:
                    left = -1
                while left >= 0 and modes[left] == "backs_to_wall":
                    modes[left] = "wall_lane"
                    if has_local_connection(local_evidence[left]):
                        break
                    left -= 1
                right = run_end + 1
                while right < len(modes) and modes[right] == "backs_to_wall":
                    modes[right] = "wall_lane"
                    if has_local_connection(local_evidence[right]):
                        break
                    right += 1
                if run_start == 0 and not start_gid:
                    raise TownLayoutError("wall_lane_disconnected: arc lacks gate/street connection")
            else:
                for i in range(run_start, run_end + 1):
                    short_run[i] = True
                if run_end == len(modes) - 1 and not end_gid:
                    raise TownLayoutError("wall_lane_disconnected: arc lacks gate/street connection")
            run_start = run_end + 1
        for edge_no, (ia, ib) in enumerate(zip(breakpoints, breakpoints[1:])):
            coords = raw[ia:ib + 1]
            sid = f"wall_segment_{arc_no:02d}_{edge_no:02d}"
            segments.append({"wall_segment_id": sid, "ring": coords,
                             "start_gate_id": start_gid if edge_no == 0 else None,
                             "end_gate_id": end_gid if edge_no == len(breakpoints) - 2 else None})
            mode = modes[edge_no]
            strips.append({"strip_id": f"strip_{sid}", "wall_segment_id": sid,
                           "mode": mode,
                           "declared_depth_gu": required_depth if mode == "backs_to_wall" else WALL_STRIP_DEPTH_GU,
                           "depth_supported": local_evidence[edge_no]["supports_depth"],
                           "arterial_occupancy_beyond_gate": local_evidence[edge_no]["arterial_beyond_gate"],
                           "short_run": short_run[edge_no],
                           "polygon": normalize_ring([[x, y] for x, y in local_evidence[edge_no]["strip"].exterior.coords])["ring"],
                           "road_ids": local_evidence[edge_no]["road_ids"]})
    if not segments or len(strips) != len(segments):
        raise TownLayoutError("unassigned_wall_arc: segment/strip mismatch")
    return segments, strips


def provisional_ring_and_ports(ctx: SiteContext, candidate: dict,
                               approaches: list[dict], *, candidate_id="c00") -> dict:
    """Stage 05 interface: one ring and protected source-road ports.

    Ports are road nodes only; Stage 06 turns the authored arterial/ring
    crossings into final gates without rerouting them.
    """
    selected = [p for p in candidate["patches"] if p.get("inside_city")]
    if not selected:
        raise TownLayoutError("invalid_polygon: no selected patches for provisional ring")
    union = unary_union([polygon_from_ring(p["polygon"]) for p in selected])
    if union.geom_type == "MultiPolygon":
        union = max(union.geoms, key=lambda g: g.area)
    if union.geom_type != "Polygon":
        raise TownLayoutError("invalid_polygon: provisional ring is not a polygon")
    # Stage 05 owns a provisional interface around the selected domain.  The
    # old pre-Stage-06 wall object encloses only non-outskirts patches and is
    # not authoritative for ports or arterial routing.
    ring_poly = union
    ring = LineString(ring_poly.exterior.coords)
    ports = []

    def intersection_points(geometry):
        points = []
        geometries = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
        for item in geometries:
            if item.is_empty:
                continue
            if item.geom_type == "Point":
                points.append(item)
            elif item.geom_type == "LineString":
                points.extend(Point(coord) for coord in
                              (item.coords[0], item.coords[-1]))
        return points

    for approach in sorted((a for a in approaches if a.get("mandatory", True)),
                           key=lambda a: a["approach_id"]):
        crossing = approach["crossing_plan_gu"]
        source_inside = approach.get("inside_polyline_plan_gu") or []
        viable = []
        if len(source_inside) >= 2:
            continuation = LineString(source_inside)
            for point in intersection_points(ring.intersection(continuation)):
                along = continuation.project(point)
                ahead = continuation.interpolate(min(continuation.length, along + 1.0))
                tx, ty = ahead.x - point.x, ahead.y - point.y
                tangent_length = math.hypot(tx, ty)
                if tangent_length <= VERTEX_EPS_GU:
                    continue
                tx, ty = tx / tangent_length, ty / tangent_length
                handle = Point(point.x + tx * 512.0, point.y + ty * 512.0)
                if ring_poly.covers(handle):
                    viable.append((along, point, tx, ty))
        if not viable:
            tx, ty = map(float, approach["inward_tangent"])
            tangent_length = math.hypot(tx, ty) or 1.0
            tx, ty = tx / tangent_length, ty / tangent_length
            ray_length = max(RAY_T_GU, math.hypot(*map(float, ctx.span_gu)) * 2.0)
            ray = LineString([
                [float(crossing[0]), float(crossing[1])],
                [float(crossing[0]) + tx * ray_length,
                 float(crossing[1]) + ty * ray_length],
            ])
            for point in intersection_points(ring.intersection(ray)):
                along = ((point.x - float(crossing[0])) * tx +
                         (point.y - float(crossing[1])) * ty)
                handle = Point(point.x + tx * 512.0, point.y + ty * 512.0)
                if along >= -VERTEX_EPS_GU and ring_poly.covers(handle):
                    viable.append((along, point, tx, ty))
        if not viable:
            raise TownLayoutError(
                f"port_no_inward_intersection: {approach['approach_id']}")
        _along, hit, tx, ty = min(
            viable, key=lambda item: (item[0], item[1].x, item[1].y))
        approach_probe = Point(hit.x + tx * 512.0, hit.y + ty * 512.0)
        if not ctx.sample(approach_probe.x, approach_probe.y).get("buildable", False):
            raise TownLayoutError(f"port_unbuildable: {approach['approach_id']}")
        ports.append({
            "port_id": f"port_{candidate_id}_{len(ports):04d}",
            "approach_id": approach["approach_id"],
            "source_edge_id": approach["source_edge_id"],
            "position": [float(hit.x), float(hit.y)],
            "inward_tangent": [tx, ty],
            "protected": True,
        })
    return {"polygon": normalize_ring([[c[0], c[1]] for c in ring_poly.exterior.coords])["ring"],
            "ports": ports}


def build_walls_and_gates(
    ctx: SiteContext,
    candidate: dict,
    town_brief: dict,
    *,
    approaches: Optional[list] = None,
    candidate_id: str = "c00",
) -> dict[str, Any]:
    """Attach palisade (or null) and approach-driven gates."""
    core = _core_patches(candidate["patches"])
    if not core:
        raise TownLayoutError("invalid_polygon: no inner patches for wall")
    mode = town_brief["fortification"]["mode"]
    approaches = list(approaches or candidate.get("approaches") or [])
    reports = list(candidate.get("reports") or [])

    if mode == "palisade":
        # The Stage 05 provisional ring is authoritative.  The old patch union
        # remains available only as a fallback for pre-Stage-05 callers.
        frozen_ring = candidate.get("provisional_ring") if candidate.get("roads") and candidate.get("nodes") else None
        if frozen_ring:
            ring = normalize_ring(frozen_ring)["ring"]
            wall_poly = polygon_from_ring(ring)
            wall = {"kind": "palisade", "source_perimeter": ring,
                    "planning_polygon": ring}
        else:
            wall = _build_palisade(core, ctx)
            wall_poly = polygon_from_ring(wall["planning_polygon"])
    elif mode == "none":
        wall = None
        union = unary_union([polygon_from_ring(p["polygon"]) for p in core])
        if union.geom_type == "MultiPolygon":
            union = max(union.geoms, key=lambda g: g.area)
        wall_poly = union
    else:
        raise TownLayoutError(f"unavailable_fortification: {mode}")

    if mode == "palisade":
        gates = []
        mandatory = sorted((a for a in approaches if a.get("mandatory", True)),
                           key=lambda a: a["approach_id"])
        if candidate.get("roads") and candidate.get("nodes"):
            for i, approach in enumerate(mandatory):
                pos, node_id = _road_gate_node(candidate, approach["approach_id"])
                if wall_poly.boundary.distance(Point(pos)) > 1.0:
                    raise TownLayoutError(f"gate_ring_node_mismatch: {approach['approach_id']}")
                tx, ty = map(float, approach["inward_tangent"])
                norm = math.hypot(tx, ty) or 1.0
                gates.append({"gate_id": f"gate_{candidate_id}_{i:04d}",
                              "position": pos, "approach_id": approach["approach_id"],
                              "outward_tangent": [-tx / norm, -ty / norm],
                              "road_node_id": node_id})
        else:
            gates = _place_gates(wall_poly, approaches, candidate_id)
            for gate in gates:
                gate["road_node_id"] = "synthetic_" + gate["approach_id"]
        ring_line = LineString(wall["planning_polygon"] + [wall["planning_polygon"][0]])
        for i, a in enumerate(gates):
            for b in gates[i + 1:]:
                if _arc_sep(ring_line, tuple(a["position"]), tuple(b["position"])) < GATE_MIN_ARC_GU:
                    raise TownLayoutError(f"gate_too_close: {a['approach_id']} {b['approach_id']}")
        if candidate.get("roads") and candidate.get("nodes"):
            segments, strips = _wall_segments(wall["planning_polygon"], gates,
                                               candidate.get("roads", []), wall_poly, ctx)
            wall["segments"], wall["strips"] = segments, strips
    else:
        gates = []
    provisional = provisional_ring_and_ports(ctx, candidate, approaches,
                                              candidate_id=candidate_id)
    for gate in gates:
        if not ctx.sample(gate["position"][0], gate["position"][1]).get("buildable", False):
            raise TownLayoutError(f"gate_excluded: {gate['gate_id']}")

    new_patches = []
    core_ids = {p["patch_id"] for p in core}
    for patch in candidate["patches"]:
        item = dict(patch)
        item["inside_wall"] = patch["patch_id"] in core_ids and mode == "palisade"
        new_patches.append(item)

    reports.append({
        "stage": "walls",
        "status": "ok",
        "message": f"mode={mode} gates={len(gates)} core_patches={len(core)}",
    })
    out = dict(candidate)
    out["patches"] = new_patches
    out["wall"] = wall
    out["gates"] = gates
    out["provisional_ring"] = provisional["polygon"]
    out["ports"] = provisional["ports"]
    out["approaches"] = approaches
    out["reports"] = reports
    return out


def write_walls_diagnostic(
    ctx: SiteContext,
    product: dict,
    *,
    topdown_path: Path,
    survey: dict,
    out_png: Path,
    full_site: bool = False,
) -> None:
    from PIL import Image, ImageDraw

    image, mapping = diagnostic_view({"_diagnostic_bounds": [(product.get("wall") or {}).get("planning_polygon") or []]}, topdown_path, survey, full_site=full_site)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    def to_px(pt):
        return _plan_to_px(float(pt[0]), float(pt[1]), mapping)

    for patch in product.get("patches", []):
        ring = patch.get("polygon") or []
        if len(ring) < 3:
            continue
        pts = [to_px(p) for p in ring]
        if patch.get("inside_wall"):
            draw.polygon(pts, fill=(40, 160, 80, 40), outline=(20, 80, 40, 160))
        elif patch.get("inside_city"):
            draw.polygon(pts, outline=(180, 140, 40, 200))
    for road in product.get("roads", []):
        geom = road.get("polyline") or []
        if len(geom) >= 2:
            color = (220, 40, 40, 230) if road.get("hierarchy") == "arterial" else (230, 180, 60, 180)
            draw.line([to_px(p) for p in geom], fill=color,
                      width=4 if road.get("hierarchy") == "arterial" else 2)
    for strip in (product.get("wall") or {}).get("strips", []):
        poly = strip.get("polygon") or []
        if len(poly) >= 3:
            color = (30, 150, 220, 100) if strip.get("mode") == "wall_lane" else (180, 100, 40, 90)
            draw.polygon([to_px(p) for p in poly], fill=color, outline=color)
    wall = product.get("wall")
    if wall:
        ring = wall.get("planning_polygon") or []
        if len(ring) >= 3:
            pts = [to_px(p) for p in ring] + [to_px(ring[0])]
            draw.line(pts, fill=(90, 50, 20, 255), width=3)
    for gate in product.get("gates", []):
        px, py = to_px(gate["position"])
        r = 8
        draw.rectangle([px - r, py - r, px + r, py + r], fill=(255, 220, 0, 255),
                       outline=(80, 40, 0, 255))
        tx, ty = gate.get("outward_tangent", [0.0, 0.0])
        draw.line([px, py, px + tx * 80.0, py + ty * 80.0], fill=(255, 0, 255, 255), width=3)
    Image.alpha_composite(image, overlay).save(out_png)
