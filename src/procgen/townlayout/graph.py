"""Patch-boundary topology graph and A* routing for V2 townlayout (Phase 9).

Purpose
-------
Build a traversable graph from ``boundary_edges``, insert gate nodes by
splitting the nearest edge, pick the market access vertex already on the
plaza patch, and A* every gate to that access node.

Inputs
------
Phase 8 candidate (patches, edges, nodes, wall, gates, anchors) plus
SiteContext.

Outputs
-------
The candidate with annotated ``boundary_edges`` / ``nodes`` and
``graph_paths`` (gate → market) used by Phase 10.

Pipeline position
-----------------
V2 townlayout Phase 9 topology graph; no VTEX.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Optional

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .constants import VERTEX_EPS_GU
from .geometry import polygon_from_ring
from .site_context import SiteContext
from .validate import TownLayoutError

WALL_SNAP_GU = 32.0
TURN_PENALTY = 20.0
ARTERIAL_HALF_WIDTH_GU = 256.0
SETBACK_GU = 128.0


def _length(geom: list) -> float:
    if len(geom) < 2:
        return 0.0
    total = 0.0
    for i in range(len(geom) - 1):
        total += math.hypot(geom[i + 1][0] - geom[i][0],
                            geom[i + 1][1] - geom[i][1])
    return total


def _sample_edge(ctx: SiteContext, geom: list) -> tuple[float, float, bool]:
    if not geom:
        return 0.0, 0.0, False
    pts = [geom[0], geom[len(geom) // 2], geom[-1]]
    slopes, suits, excluded = [], [], 0
    for x, y in pts:
        s = ctx.sample(float(x), float(y))
        slopes.append(float(s.get("slope_cost", 0.0)))
        suits.append(float(s.get("suitability", 0.0)))
        if not s.get("buildable", False):
            excluded += 1
    return (
        sum(slopes) / len(slopes),
        sum(suits) / len(suits),
        excluded == len(pts),
    )


def _edge_cost(length: float, slope_cost: float, suitability: float) -> float:
    return (
        length
        + 2.0 * length * slope_cost
        + 1.0 * length * (1.0 - suitability)
    )


def _on_wall(geom: list, wall_poly: Optional[Polygon]) -> bool:
    if wall_poly is None or len(geom) < 2:
        return False
    a, b = Point(geom[0][0], geom[0][1]), Point(geom[-1][0], geom[-1][1])
    return (wall_poly.boundary.distance(a) <= WALL_SNAP_GU
            and wall_poly.boundary.distance(b) <= WALL_SNAP_GU)


def _split_geom(geom: list, pt: tuple[float, float]) -> tuple[list, list]:
    line = LineString([(float(p[0]), float(p[1])) for p in geom])
    dist = line.project(Point(pt))
    a = [(float(x), float(y)) for x, y, *_ in line.interpolate(dist).coords]
    # Walk original vertices plus the split point.
    left: list[list[float]] = []
    acc = 0.0
    left.append([float(geom[0][0]), float(geom[0][1])])
    for i in range(len(geom) - 1):
        seg = math.hypot(geom[i + 1][0] - geom[i][0], geom[i + 1][1] - geom[i][1])
        if acc + seg >= dist - VERTEX_EPS_GU:
            left.append([pt[0], pt[1]])
            right = [[pt[0], pt[1]]] + [
                [float(p[0]), float(p[1])] for p in geom[i + 1:]]
            if len(left) < 2:
                left = [left[0], [pt[0], pt[1]]]
            if len(right) < 2:
                right = [[pt[0], pt[1]], right[-1]]
            return left, right
        acc += seg
        left.append([float(geom[i + 1][0]), float(geom[i + 1][1])])
    left.append([pt[0], pt[1]])
    return left, [[pt[0], pt[1]], [float(geom[-1][0]), float(geom[-1][1])]]


def _node_key(pt) -> tuple[int, int]:
    return (int(round(float(pt[0]) * 100)), int(round(float(pt[1]) * 100)))


def _ensure_ring_edges(candidate: dict, candidate_id: str) -> None:
    """Add missing consecutive-ring edges so each inner patch boundary is a cycle."""
    nodes = list(candidate["nodes"])
    edges = list(candidate["boundary_edges"])
    by_pos = {_node_key(n["position"]): n["node_id"] for n in nodes}
    node_seq = max((int(n["node_id"].split("_")[-1]) for n in nodes), default=-1) + 1
    edge_seq = max((int(e["edge_id"].split("_")[-1]) for e in edges), default=-1) + 1
    existing = set()
    for edge in edges:
        existing.add(tuple(sorted((edge["a_node"], edge["b_node"]))))

    def node_for(pt) -> str:
        nonlocal node_seq, nodes, by_pos
        key = _node_key(pt)
        if key in by_pos:
            return by_pos[key]
        nid = f"node_{candidate_id}_{node_seq:04d}"
        node_seq += 1
        nodes.append({"node_id": nid, "position": [float(pt[0]), float(pt[1])],
                      "kind": "junction"})
        by_pos[key] = nid
        return nid

    inner = [
        p for p in candidate["patches"]
        if p.get("inside_city") and p.get("morphology_region") != "outskirts"
    ]
    for patch in inner:
        ring = patch["polygon"]
        n = len(ring)
        for i in range(n):
            a = ring[i]
            b = ring[(i + 1) % n]
            na, nb = node_for(a), node_for(b)
            pair = tuple(sorted((na, nb)))
            if pair in existing or na == nb:
                continue
            existing.add(pair)
            eid = f"edge_{candidate_id}_{edge_seq:04d}"
            edge_seq += 1
            edges.append({
                "edge_id": eid,
                "a_node": na,
                "b_node": nb,
                "geometry": [[float(a[0]), float(a[1])],
                             [float(b[0]), float(b[1])]],
                "patch_left": patch["patch_id"],
                "patch_right": None,
                "edge_role": "fringe",
                "road_class": "none",
            })
    candidate["nodes"] = nodes
    candidate["boundary_edges"] = edges


def _insert_gate_nodes(candidate: dict, wall_poly: Optional[Polygon],
                       candidate_id: str, records: Optional[list] = None) -> dict[str, str]:
    """Split nearest boundary edges at gate positions. Return gate_id → node_id."""
    nodes = list(candidate["nodes"])
    edges = list(candidate["boundary_edges"])
    mapping: dict[str, str] = {}
    node_seq = max((int(n["node_id"].split("_")[-1]) for n in nodes), default=-1) + 1
    edge_seq = max((int(e["edge_id"].split("_")[-1]) for e in edges), default=-1) + 1
    by_id = {n["node_id"]: n for n in nodes}

    for gate in (records if records is not None else candidate.get("gates") or []):
        gx, gy = float(gate["position"][0]), float(gate["position"][1])
        best_i = None
        best_d = None
        for i, edge in enumerate(edges):
            line = LineString([(float(p[0]), float(p[1])) for p in edge["geometry"]])
            d = line.distance(Point(gx, gy))
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        if best_i is None:
            raise TownLayoutError(f"gate_unreachable: {gate['gate_id']} no edge")
        edge = edges[best_i]
        nid = f"node_{candidate_id}_{node_seq:04d}"
        node_seq += 1
        nodes.append({
            "node_id": nid,
            "position": [gx, gy],
            "kind": "gate",
        })
        by_id[nid] = nodes[-1]
        mapping[gate.get("gate_id", gate.get("port_id"))] = nid
        left, right = _split_geom(edge["geometry"], (gx, gy))
        e_left = dict(edge)
        e_left["edge_id"] = f"edge_{candidate_id}_{edge_seq:04d}"
        edge_seq += 1
        e_left["b_node"] = nid
        e_left["geometry"] = left
        e_right = dict(edge)
        e_right["edge_id"] = f"edge_{candidate_id}_{edge_seq:04d}"
        edge_seq += 1
        e_right["a_node"] = nid
        e_right["geometry"] = right
        edges.pop(best_i)
        edges.extend([e_left, e_right])

    candidate["nodes"] = nodes
    candidate["boundary_edges"] = edges
    return mapping


def _market_access_node(ctx: SiteContext, candidate: dict) -> str:
    market = next((a for a in candidate.get("anchors") or []
                   if a.get("kind") == "market"), None)
    if market is None:
        raise TownLayoutError("missing_anchor: market required for graph")
    poly = polygon_from_ring(market["polygon"])
    c = poly.centroid
    scored = []
    for node in candidate["nodes"]:
        x, y = float(node["position"][0]), float(node["position"][1])
        if poly.boundary.distance(Point(x, y)) > 64.0:
            continue
        if not ctx.sample(x, y).get("buildable", False):
            continue
        d = math.hypot(x - c.x, y - c.y)
        scored.append((d, node["node_id"]))
    if not scored:
        for node in candidate["nodes"]:
            x, y = float(node["position"][0]), float(node["position"][1])
            d = math.hypot(x - c.x, y - c.y)
            if ctx.sample(x, y).get("buildable", False):
                scored.append((d, node["node_id"]))
    if not scored:
        raise TownLayoutError("missing_ref: no market access node")
    scored.sort()
    best = scored[0][1]
    for node in candidate["nodes"]:
        if node["node_id"] == best and node.get("kind") == "junction":
            node["kind"] = "plaza"
    return best


def _annotate_edges(ctx: SiteContext, candidate: dict,
                    wall_poly: Optional[Polygon]) -> None:
    gate_nodes = {n["node_id"] for n in candidate["nodes"] if n["kind"] == "gate"}
    patch_wall = {
        p["patch_id"]: bool(p.get("inside_wall"))
        for p in candidate["patches"]
    }
    for edge in candidate["boundary_edges"]:
        geom = edge["geometry"]
        length = _length(geom)
        slope, suit, water_bar = _sample_edge(ctx, geom)
        on_wall = _on_wall(geom, wall_poly)
        left, right = edge.get("patch_left"), edge.get("patch_right")
        if on_wall and (
                bool(patch_wall.get(left)) != bool(patch_wall.get(right))
                or (left and not right) or (right and not left)):
            edge["edge_role"] = "wall"
        blocked = bool(water_bar)
        edge["_length"] = length
        edge["_slope_cost"] = slope
        edge["_suitability"] = suit
        edge["_blocked"] = blocked
        edge["_cost"] = _edge_cost(length, slope, suit)
        edge["_on_wall"] = on_wall


def _routing_overlay(ctx: SiteContext, candidate: dict, candidate_id: str) -> None:
    """Build the deterministic, derived interior visibility overlay.

    ``boundary_edges`` remains the frozen Stage-04 provenance graph.  This
    separate collection is the geometry authority for interior arterials.
    Every candidate is rejected using the complete arterial corridor and the
    two-sided frontage depth, rather than merely testing its centre line.
    """
    inner = sorted((p for p in candidate.get("patches", [])
                    if p.get("inside_city")),
                   key=lambda p: p["patch_id"])
    land = unary_union([Polygon(p["polygon"]) for p in inner]) if inner else Polygon()
    # Patch vertices are quantized independently; a tiny planning tolerance
    # closes only those seams and is far below the 512-GU arterial width.
    land_for_routing = land.buffer(128.0)
    waters = [Polygon(w) for w in candidate.get("water_polygons") or []]
    protected = [Polygon(s["polygon"]) for s in candidate.get("open_spaces") or []
                 if len(s.get("polygon") or []) >= 3]
    water = unary_union(waters) if waters else Polygon()
    protected_u = unary_union(protected) if protected else Polygon()
    ring_pts = candidate.get("provisional_ring") or candidate.get("city_domain") or []
    ring = Polygon(ring_pts) if len(ring_pts) >= 3 else land
    stats = ctx.stamp_footprint_stats or candidate.get("stamp_footprint_stats") or {}
    depth = math.sqrt(float(stats.get("p10", 0.0))) + SETBACK_GU

    nodes = list(candidate["nodes"])
    existing = {n["node_id"] for n in nodes}
    # Stable representative points are preferred; the midpoint fallback is
    # deliberately bounded and points inward toward the representative.
    patch_nodes = []
    seq = 0
    for patch in inner:
        poly = Polygon(patch["polygon"])
        rp = poly.representative_point()
        point = (float(rp.x), float(rp.y))
        nid = f"node_{candidate_id}_routing_{seq:04d}"
        seq += 1
        nodes.append({"node_id": nid, "position": [point[0], point[1]], "kind": "junction"})
        patch_nodes.append((nid, patch["patch_id"], point, poly))

    # Deterministic clearance waypoints let visibility routing go around a
    # protected plaza instead of treating its boundary as a road destination.
    # They are derived points, never edits to the Stage-04 patch graph.
    waypoint_nodes = []
    for si, protected_poly in enumerate(protected):
        cx, cy = protected_poly.centroid.x, protected_poly.centroid.y
        for vi, (vx, vy) in enumerate(list(protected_poly.exterior.coords)[:-1]):
            dx, dy = vx - cx, vy - cy
            norm = math.hypot(dx, dy) or 1.0
            point = (float(vx + 512.0 * dx / norm),
                     float(vy + 512.0 * dy / norm))
            p = Point(point)
            if not ring.covers(p) or protected_u.covers(p) or p.intersects(water):
                continue
            nid = f"node_{candidate_id}_routing_waypoint_{si:02d}_{vi:03d}"
            nodes.append({"node_id": nid, "position": [point[0], point[1]], "kind": "junction"})
            waypoint_nodes.append((nid, [], point, None))

    market = next((a for a in candidate.get("anchors") or [] if a.get("kind") == "market"), None)
    market_poly = Polygon(market["polygon"]) if market else None
    market_node = None
    if (market_poly is not None and
            land.centroid.distance(market_poly.centroid) > VERTEX_EPS_GU):
        boundary = market_poly.exterior.interpolate(
            market_poly.exterior.project(land.centroid))
        dx = boundary.x - market_poly.centroid.x
        dy = boundary.y - market_poly.centroid.y
        norm = math.hypot(dx, dy) or 1.0
        access = [float(boundary.x + dx / norm * (ARTERIAL_HALF_WIDTH_GU + 64.0)),
                  float(boundary.y + dy / norm * (ARTERIAL_HALF_WIDTH_GU + 64.0))]
        if ring.covers(Point(access)) and land_for_routing.covers(Point(access)):
            market_node = f"node_{candidate_id}_market_access"
            nodes.append({"node_id": market_node, "position": access,
                          "kind": "junction"})
    node_pos = {n["node_id"]: n["position"] for n in nodes}
    routing = []
    diagnostics = {
        "candidate_count": 0, "accepted_overlay_edges": 0,
        "rejected_overlay_edges": 0, "rejected_boundary_edges": 0,
        "rejection_counts": {}, "port_handles": [],
    }

    def feasible(a, b, handle_span=False):
        """Apply the same corridor and frontage-capacity gate as streets.py.

        This returns measured side ratios for routing cost.  The 80% threshold
        applies to the complete authored arterial, not every constituent edge:
        enforcing it here disconnects otherwise valid port transitions.
        """
        line = LineString([a, b])
        if line.length <= VERTEX_EPS_GU:
            return False, {"reason": "zero_length"}
        corridor = line.buffer(ARTERIAL_HALF_WIDTH_GU, cap_style=2)
        corridor_land = land_for_routing.intersection(corridor)
        if not handle_span and not ring.covers(corridor):
            return False, {"reason": "outside_ring"}
        if not handle_span and corridor_land.area / (corridor.area or 1.0) < 0.999:
            return False, {"reason": "corridor_land"}
        # The immutable port handle is the sole permitted span across the
        # port-side threshold; all subsequent interior edges retain protected
        # containment.  Water is never exempted.
        if corridor.intersects(water) or (not handle_span and corridor.intersects(protected_u)):
            return False, {"reason": "water_or_protected"}
        # The full depth is a frontage-capacity test: both normal strips must
        # be substantially developed.  This rejects slivers while allowing a
        # corridor to pass a natural patch seam (the final report remains the
        # binding authoring gate).
        side = {"left": {"eligible_samples": 0, "pass_samples": 0},
                "right": {"eligible_samples": 0, "pass_samples": 0}}
        sample_distances = []
        sample_distance = 128.0
        while sample_distance < line.length:
            sample_distances.append(sample_distance)
            sample_distance += 256.0
        if not sample_distances:
            sample_distances = [line.length / 2.0]
        for sample_distance in sample_distances:
            sample = line.interpolate(sample_distance)
            x, y = sample.x, sample.y
            dx, dy = b[0] - a[0], b[1] - a[1]
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm, dx / norm
            for side_name, sign in (("left", 1.0), ("right", -1.0)):
                probe = LineString([
                    (x + nx * 256 * sign, y + ny * 256 * sign),
                    (x + nx * (256 + depth) * sign,
                     y + ny * (256 + depth) * sign),
                ])
                developed = land.difference(water).difference(protected_u)
                coverage = developed.intersection(probe).length / depth
                item = side[side_name]
                item["eligible_samples"] += 1
                if coverage >= 0.90:
                    item["pass_samples"] += 1
        ratios = {name: (v["pass_samples"] / v["eligible_samples"]
                         if v["eligible_samples"] else 0.0)
                  for name, v in side.items()}
        detail = {"left_ratio": ratios["left"], "right_ratio": ratios["right"],
                  "probe_depth_gu": depth}
        if not handle_span and min(ratios.values()) < 0.80:
            detail["reason"] = "side_depth"
            return False, detail
        return True, detail

    def frontage_cost_multiplier(detail):
        """Strongly prefer depth-rich spans without replacing the route gate."""
        deficit = (max(0.0, 0.80 - float(detail.get("left_ratio", 0.0))) +
                   max(0.0, 0.80 - float(detail.get("right_ratio", 0.0))))
        return 1.0 + 50.0 * deficit

    def add(aid, bid, geom, kind, patches):
        diagnostics["candidate_count"] += 1
        if aid == bid:
            diagnostics["rejection_counts"]["self_edge"] = diagnostics["rejection_counts"].get("self_edge", 0) + 1
            return False
        line = LineString(geom)
        # Only the immutable port_handle record bypasses ring/corridor
        # containment.  Every derived continuation touching its endpoint is a
        # normal candidate and must pass the full side-depth gate.
        ok, detail = feasible(geom[0], geom[-1], False)
        if not ok:
            reason = detail.get("reason", "unknown")
            diagnostics["rejected_overlay_edges"] += 1
            diagnostics["rejection_counts"][reason] = diagnostics["rejection_counts"].get(reason, 0) + 1
            return False
        slope, suit, blocked = _sample_edge(ctx, geom)
        if blocked:
            return False
        base_cost = _edge_cost(line.length, slope, suit)
        routed_cost = base_cost * frontage_cost_multiplier(detail)
        routing.append({
            "routing_edge_id": f"routing_{candidate_id}_{len(routing):04d}",
            "a_node": aid, "b_node": bid,
            "geometry": [[float(x), float(y)] for x, y in geom],
            "kind": kind, "boundary_edge_ids": [],
            "source_patch_ids": sorted(set(patches)),
            "cost": routed_cost,
            "_length": line.length, "_cost": routed_cost,
            "probe": detail,
        })
        diagnostics["accepted_overlay_edges"] += 1
        return True

    # Ports are protected nodes already inserted by _insert_gate_nodes.  The
    # first span is authored exactly on the stored inward tangent.
    port_nodes = []
    for port in sorted(candidate.get("ports") or [], key=lambda p: p["port_id"]):
        matching = [n for n in nodes if n["position"] == port["position"]]
        gate_match = next((n for n in matching if n.get("kind") == "gate"), None)
        nid = (gate_match or (matching[0] if matching else None))
        nid = nid["node_id"] if nid else None
        if nid is None:
            continue
        t = port.get("inward_tangent") or [0.0, 0.0]
        norm = math.hypot(float(t[0]), float(t[1])) or 1.0
        handle = [float(port["position"][0]) + 512.0 * float(t[0]) / norm,
                  float(port["position"][1]) + 512.0 * float(t[1]) / norm]
        # The fixed handle is retained as a port edge; its endpoint is then
        # connected by the same constrained visibility rule as all other edges.
        handle_inside = ring.covers(Point(handle))
        handle_water = Point(handle).intersects(water)
        diagnostics["port_handles"].append({
            "port_id": port["port_id"], "position": list(port["position"]),
            "handle": handle, "inside_ring": handle_inside,
            "in_water": handle_water,
        })
        if handle_inside and not handle_water:
            routing.append({"routing_edge_id": f"routing_{candidate_id}_{len(routing):04d}",
                            "a_node": nid, "b_node": nid + "_handle",
                            "geometry": [list(port["position"]), handle],
                            "kind": "port_handle", "boundary_edge_ids": [],
                            "source_patch_ids": [], "cost": 512.0, "_length": 512.0,
                            "_cost": 512.0})
            nodes.append({"node_id": nid + "_handle", "position": handle, "kind": "junction"})
            node_pos[nid + "_handle"] = handle
            port_nodes.append((nid + "_handle", []))
        else:
            port_nodes.append((nid, []))

    # Give each immutable handle a short, deterministic continuation so a
    # single long visibility span cannot hide an otherwise feasible route.
    handle_waypoints = []
    for handle_id, _ in port_nodes:
        if not handle_id.endswith("_handle") or handle_id not in node_pos:
            continue
        port_pos = node_pos[handle_id[:-7]]
        hx, hy = node_pos[handle_id]
        dx, dy = hx - port_pos[0], hy - port_pos[1]
        norm = math.hypot(dx, dy) or 1.0
        for wi, distance in enumerate((1024.0, 1536.0)):
            point = [port_pos[0] + distance * dx / norm,
                     port_pos[1] + distance * dy / norm]
            p = Point(point)
            if not ring.covers(p) or p.intersects(water):
                continue
            nid = f"{handle_id}_continuation_{wi:02d}"
            nodes.append({"node_id": nid, "position": point, "kind": "junction"})
            node_pos[nid] = point
            handle_waypoints.append((nid, [], point, handle_id))

    targets = ([(n, [pid]) for n, pid, point, _ in patch_nodes
                if not protected_u.covers(Point(point))] +
               [(n, pids) for n, pids, point, _ in waypoint_nodes] +
               [(n, pids) for n, pids, point, _ in handle_waypoints])
    targets += port_nodes
    if market_node is not None:
        targets.append((market_node, []))
    # Complete visibility among the small deterministic node set.
    for i, (aid, ap) in enumerate(targets):
        for bid, bp in targets[i + 1:]:
            if aid == bid or aid not in node_pos or bid not in node_pos:
                continue
            add(aid, bid, [node_pos[aid], node_pos[bid]], "interior", ap + bp)
    # The explicit market node is retained as the A* goal.  If no boundary
    # node is suitable, the market patch representative is the deterministic goal.
    if market_node is None and patch_nodes:
        market_candidates = [(math.hypot(point[0] - market_poly.centroid.x,
                                         point[1] - market_poly.centroid.y), nid)
                             for nid, pid, point, _ in patch_nodes
                             if market_poly and pid == market.get("patch_id")]
        if market_candidates:
            market_node = min(market_candidates)[1]
    candidate["nodes"] = nodes
    candidate["routing_edges"] = routing
    candidate["market_access_node"] = market_node
    # Only boundary edges that pass this same full candidate gate may enter
    # A*. They are retained for exact provenance/joins but receive a stable
    # penalty so a feasible interior route remains authoritative.
    valid_boundary = []
    for edge in sorted(candidate["boundary_edges"], key=lambda e: e["edge_id"]):
        ok, detail = feasible(edge["geometry"][0], edge["geometry"][-1])
        if ok and not edge.get("_blocked"):
            edge["_routing_probe"] = detail
            edge["_routing_cost_multiplier"] = frontage_cost_multiplier(detail)
            valid_boundary.append(edge["edge_id"])
        else:
            diagnostics["rejected_boundary_edges"] += 1
    candidate["_routing_boundary_edge_ids"] = valid_boundary
    diagnostics["boundary_edges_available"] = len(valid_boundary)
    candidate["routing_overlay_diagnostics"] = diagnostics


def _adjacency(candidate: dict) -> dict[str, list[tuple[str, dict]]]:
    adj: dict[str, list[tuple[str, dict]]] = {}
    allowed_boundary = set(candidate.get("_routing_boundary_edge_ids") or [])
    boundary = [e for e in candidate["boundary_edges"] if e.get("edge_id") in allowed_boundary]
    for edge in boundary + list(candidate.get("routing_edges") or []):
        if edge.get("_blocked") or edge.get("_route_blocked"):
            continue
        edge.setdefault("_length", _length(edge.get("geometry") or []))
        edge.setdefault("_cost", edge.get("_length", 0.0))
        routed = "routing_edge_id" in edge
        if not routed:
            edge = dict(edge)
            edge["_cost"] = (float(edge["_cost"]) * 8.0 *
                             float(edge.get("_routing_cost_multiplier", 1.0)))
        if edge["a_node"] == edge["b_node"]:
            continue
        adj.setdefault(edge["a_node"], []).append((edge["b_node"], edge))
        adj.setdefault(edge["b_node"], []).append((edge["a_node"], edge))
    for items in adj.values():
        items.sort(key=lambda item: (item[0], item[1].get("routing_edge_id", item[1].get("edge_id", ""))))
    return adj


def _dir_of(a: list[float], b: list[float]) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    return (dx / n, dy / n)


def astar_route(candidate: dict, start: str, goal: str) -> Optional[list[str]]:
    """Return the deterministic validated routing-edge path to the goal.

    Turn cost makes arrival direction part of the search state.  Shared trunk
    discounts also make a Euclidean A* heuristic inadmissible, so this is a
    deterministic Dijkstra search over ``(node, previous_node)`` states.
    """
    adj = _adjacency(candidate)
    nodes = {n["node_id"]: n["position"] for n in candidate["nodes"]}
    if start not in nodes or goal not in nodes:
        return None
    if start == goal:
        return []
    start_state = (start, "")
    heap = [(0.0, start, "")]
    best_g = {start_state: 0.0}
    parent: dict[tuple[str, str], tuple[Optional[tuple[str, str]], Optional[str]]] = {
        start_state: (None, None)
    }
    goal_state = None

    while heap:
        g, nid, previous = heapq.heappop(heap)
        state = (nid, previous)
        if g > best_g.get(state, math.inf) + 1e-9:
            continue
        if nid == goal:
            goal_state = state
            break
        pos = nodes[nid]
        in_dir = _dir_of(nodes[previous], pos) if previous else None
        for nbr, edge in adj.get(nid, []):
            if nbr == previous:
                continue
            npos = nodes[nbr]
            out_dir = _dir_of(pos, npos)
            extra = 0.0
            if in_dir is not None:
                cos_t = max(-1.0, min(1.0, in_dir[0] * out_dir[0] + in_dir[1] * out_dir[1]))
                if cos_t < math.cos(math.radians(120.0)):
                    continue
                extra = TURN_PENALTY * float(edge["_length"]) * (1.0 - max(0.0, cos_t))
            ng = g + float(edge["_cost"]) + extra
            next_state = (nbr, nid)
            if ng + 1e-9 < best_g.get(next_state, math.inf):
                best_g[next_state] = ng
                parent[next_state] = (
                    state, edge.get("edge_id", edge.get("routing_edge_id")))
                heapq.heappush(heap, (ng, nbr, nid))
    if goal_state is None:
        return None
    path = []
    state = goal_state
    while state != start_state:
        previous_state, eid = parent[state]
        if previous_state is None or eid is None:
            return None
        path.append(eid)
        state = previous_state
    path.reverse()
    return path


def build_topology_graph(
    ctx: SiteContext,
    candidate: dict,
    *,
    candidate_id: str = "c00",
) -> dict[str, Any]:
    """Insert gates, annotate costs, route every gate to the market access node."""
    wall = candidate.get("wall")
    wall_poly = None
    if wall and wall.get("planning_polygon"):
        wall_poly = polygon_from_ring(wall["planning_polygon"])
    _ensure_ring_edges(candidate, candidate_id)
    ports = list(candidate.get("ports") or [])
    routing_ports = [{**p, "gate_id": p["port_id"]} for p in ports]
    gate_nodes = _insert_gate_nodes(candidate, wall_poly, candidate_id,
                                    routing_ports if routing_ports else None)
    market_node = _market_access_node(ctx, candidate)
    # Keep the explicit access decision stable; overlay node creation must not
    # silently replace it with a representative merely because it is nearer.
    candidate["market_access_node"] = market_node
    _annotate_edges(ctx, candidate, wall_poly)
    _routing_overlay(ctx, candidate, candidate_id)
    market_node = candidate.get("market_access_node") or market_node
    paths = []
    selected_route_ids: set[str] = set()
    selected_route_edges: list[dict] = []
    reports = list(candidate.get("reports") or [])
    route_records = routing_ports if routing_ports else list(candidate.get("gates") or [])
    market_position = next(
        n["position"] for n in candidate["nodes"] if n["node_id"] == market_node)

    def source_route_order(record):
        port = next((item for item in candidate.get("ports") or []
                     if item.get("port_id") == record.get("port_id")), record)
        position = port.get("position") or market_position
        tangent = port.get("inward_tangent") or [0.0, 0.0]
        toward_market = _dir_of(position, market_position)
        alignment = (float(tangent[0]) * toward_market[0] +
                     float(tangent[1]) * toward_market[1])
        return (record.get("source_edge_id") or "", -alignment,
                record.get("approach_id") or record.get("gate_id") or "")

    route_records = sorted(route_records, key=source_route_order)
    for gate in route_records:
        route_key = gate.get("port_id", gate.get("gate_id"))
        nid = gate_nodes[route_key]
        route = astar_route(candidate, nid, market_node)
        if route is None:
            diag = candidate.get("routing_overlay_diagnostics") or {}
            adjacency = _adjacency(candidate)
            handle_id = nid + "_handle"
            raise TownLayoutError(
                f"route_failed: {route_key} to market {market_node}; "
                f"start_degree={len(adjacency.get(nid, []))} "
                f"handle_degree={len(adjacency.get(handle_id, []))} "
                f"market_degree={len(adjacency.get(market_node, []))} "
                f"port_handles={diag.get('port_handles', [])} "
                f"accepted_overlay_edges={diag.get('accepted_overlay_edges', 0)} "
                f"rejected_overlay_edges={diag.get('rejected_overlay_edges', 0)} "
                f"rejected_boundary_edges={diag.get('rejected_boundary_edges', 0)} "
                f"rejection_counts={diag.get('rejection_counts', {})}")
        boundary_route = [eid for eid in route
                          if eid in {e["edge_id"] for e in candidate["boundary_edges"]}]
        paths.append({
            "route_id": f"route_{candidate_id}_{route_key}",
            "from_node": nid,
            "to_node": market_node,
            "edge_ids": boundary_route,
            "routing_edge_ids": route,
            "approach_id": gate.get("approach_id"),
            "source_edge_id": gate.get("source_edge_id"),
        })
        all_edges = {e.get("edge_id", e.get("routing_edge_id")): e
                     for e in (list(candidate["boundary_edges"]) +
                               list(candidate.get("routing_edges") or []))}
        for edge_id in route:
            edge = all_edges[edge_id]
            selected_route_ids.add(edge_id)
            if edge not in selected_route_edges:
                selected_route_edges.append(edge)
            # Shared trunks are preferable to parallel duplicate approaches.
            edge["_cost"] = float(edge.get("_cost", edge.get("_length", 0.0))) * 0.10
        for edge_id, edge in all_edges.items():
            if edge_id in selected_route_ids or edge.get("_route_blocked"):
                continue
            line = LineString(edge.get("geometry") or [])
            edge_nodes = {edge.get("a_node"), edge.get("b_node")}
            for selected in selected_route_edges:
                if edge_nodes.intersection({selected.get("a_node"), selected.get("b_node")}):
                    continue
                if line.crosses(LineString(selected.get("geometry") or [])):
                    edge["_route_blocked"] = True
                    break
    reports.append({
        "stage": "graph",
        "status": "ok",
        "message": f"routes={len(paths)} market_node={market_node}",
    })
    out = dict(candidate)
    out["graph_paths"] = paths
    out["market_access_node"] = market_node
    out["reports"] = reports
    return out
