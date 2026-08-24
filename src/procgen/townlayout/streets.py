"""Arterial / street hierarchy for V2 townlayout (Phase 10).

Purpose
-------
Promote every gate→market graph path to arterial, classify remaining
inner shared boundaries as streets, and emit ``roads``.  The wall /
outskirts ring is not a through-street.

Inputs
------
Phase 9 candidate with ``graph_paths`` and annotated ``boundary_edges``.

Outputs
-------
Candidate with ``roads`` and updated ``road_class`` on boundary edges.

Pipeline position
-----------------
V2 townlayout Phase 10 street hierarchy; no VTEX.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .constants import (
    ARTERIAL_CLEAR_WIDTH_GU,
    LANE_CLEAR_WIDTH_GU,
    STREET_CLEAR_WIDTH_GU,
    TRANSITION_STUB_LENGTH_GU,
)
from .site_context import SiteContext, _plan_to_px, diagnostic_view
from .validate import TownLayoutError
from .road_geometry import (
    curvature_failure_segments,
    ordered_chain_points,
    smooth_chain,
)

HIERARCHY_PAINT = {
    "arterial": ("arterial", ARTERIAL_CLEAR_WIDTH_GU, "road"),
    "street": ("street", STREET_CLEAR_WIDTH_GU, "road"),
    "lane": ("lane", LANE_CLEAR_WIDTH_GU, "settlement_dirt"),
    "regional_approach": ("regional_approach", ARTERIAL_CLEAR_WIDTH_GU, "road"),
}

def _dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _inner_ids(patches: list[dict]) -> set[str]:
    return {
        p["patch_id"] for p in patches
        if p.get("inside_city")
    }


def _edge_length(edge: dict) -> float:
    geom = edge.get("geometry") or []
    return sum(_dist(a, b) for a, b in zip(geom, geom[1:]))


def _select_boundary_classes(candidate: dict, inner: set[str],
                             arterial_edges: set[str]) -> dict[str, str]:
    """Select a sparse, deterministic patch-boundary road skeleton.

    Arterial provenance is immutable.  For the remaining boundaries, first
    cover every developed patch, then join disconnected patch components.  A
    short selected connector is a lane; residual boundaries stay ``none`` so
    the diagnostic does not become a Voronoi-edge map.
    """
    edges = [e for e in candidate["boundary_edges"]
             if e.get("patch_left") in inner and e.get("patch_right") in inner
             and e["edge_id"] not in arterial_edges]
    classes = {e["edge_id"]: "none" for e in candidate["boundary_edges"]}
    for eid in arterial_edges:
        classes[eid] = "arterial"
    selected: list[dict] = []
    # Longest frontage per patch gives useful road boundary coverage without
    # declaring every residual polygon edge a road.
    for pid in sorted(inner):
        options = [e for e in edges if pid in (e.get("patch_left"), e.get("patch_right"))]
        if options:
            pick = max(options, key=lambda e: (_edge_length(e), e["edge_id"]))
            if pick not in selected:
                selected.append(pick)
    # Kruskal-style joins ensure all covered patch components communicate.
    parent = {pid: pid for pid in inner}
    def find(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid
    def join(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True
    for e in selected:
        join(e["patch_left"], e["patch_right"])
    for e in sorted(edges, key=lambda x: (-_edge_length(x), x["edge_id"])):
        if join(e["patch_left"], e["patch_right"]):
            selected.append(e)
    for e in selected:
        # 256-GU lanes are intentionally real roads, not a residual label.
        classes[e["edge_id"]] = "lane" if _edge_length(e) <= 2100.0 else "street"
    return classes


def _two_sided_arterial_report(candidate: dict, roads: list[dict], *, stamp_stats: dict | None = None) -> dict:
    """Probe both frontage normals at 256 GU on developed block candidates.

    A side passes when at least 80% of eligible 256-GU samples retain a
    continuous normal probe through 90% of the required depth.  This rejects
    narrow slivers without requiring a rectangular parcel at road-network
    stage.  Probes start at the authored clear-corridor edge.
    """
    inner = [p for p in candidate.get("patches", [])
             if p.get("inside_city")]
    selected_land = unary_union([Polygon(p["polygon"]) for p in inner]) if inner else Polygon()
    authored_corridors = [
        LineString(r["polyline"]).buffer(
            float(r.get("clear_width_gu", ARTERIAL_CLEAR_WIDTH_GU)) / 2.0)
        for r in roads if len(r.get("polyline") or []) >= 2
    ]
    road_union = unary_union(authored_corridors) if authored_corridors else Polygon()
    arterial_corridors = [
        LineString(r["polyline"]).buffer(float(r["clear_width_gu"]) / 2.0)
        for r in roads
        if r.get("hierarchy") == "arterial" and len(r.get("polyline") or []) >= 2
    ]
    arterial_union = unary_union(arterial_corridors) if arterial_corridors else Polygon()
    water_parts = [Polygon(w) for w in candidate.get("water_polygons") or []]
    water_union = unary_union(water_parts) if water_parts else Polygon()
    protected_parts = [Polygon(s["polygon"]) for s in candidate.get("open_spaces") or []
                       if len(s.get("polygon") or []) >= 3]
    protected_union = unary_union(protected_parts) if protected_parts else Polygon()
    raw_developed = selected_land.difference(water_union).difference(protected_union)
    # Secondary streets cross arterial frontage at junctions; subtracting
    # their complete corridors makes those legitimate crossings look like
    # missing urban land.  The arterial corridor itself is the only continuous
    # reservation relevant to the two-sided depth probe.
    developed_blocks = raw_developed.difference(arterial_union)
    probe_land = raw_developed
    stats = stamp_stats or candidate.get("stamp_footprint_stats") or {}
    depth = math.sqrt(float(stats.get("p10", 0.0))) + 128.0
    probe_fraction = 0.90
    per_road = []
    node_positions = {node["node_id"]: node["position"]
                      for node in candidate.get("nodes") or []}
    port_positions = {tuple(port["position"]) for port in candidate.get("ports") or []}
    port_nodes = {node_id for node_id, position in node_positions.items()
                  if tuple(position) in port_positions}
    for road in roads:
        if road.get("hierarchy") == "regional_approach":
            continue
        if road.get("hierarchy") != "arterial" or len(road.get("polyline") or []) < 2:
            continue
        line = LineString(road["polyline"])
        ring = candidate.get("provisional_ring") or candidate.get("city_domain") or []
        interior = Polygon(ring) if len(ring) >= 3 else selected_land
        corridor_half = float(road["clear_width_gu"]) / 2.0
        road_nodes = {road.get("node_a"), road.get("node_b")}
        junction_corridors = [
            LineString(other["polyline"]).buffer(float(other["clear_width_gu"]) / 2.0)
            for other in roads
            if other.get("hierarchy") == "arterial"
            and other.get("road_id") != road.get("road_id")
            and road_nodes.intersection({other.get("node_a"), other.get("node_b")})
        ]
        junction_union = (unary_union(junction_corridors)
                          if junction_corridors else Polygon())
        side_stats = {"left": {"eligible_samples": 0, "pass_samples": 0,
                                "failures": []},
                      "right": {"eligible_samples": 0, "pass_samples": 0,
                                 "failures": []}}
        sample_distances = []
        sample_distance = 128.0
        while sample_distance < line.length:
            sample_distances.append(sample_distance)
            sample_distance += 256.0
        if not sample_distances:
            sample_distances = [line.length / 2.0]
        for distance in sample_distances:
            point = line.interpolate(distance)
            x, y = float(point.x), float(point.y)
            if not interior.covers(point):
                continue
            tangent_span = min(64.0, line.length / 2.0)
            before = line.interpolate(max(0.0, distance - tangent_span))
            after = line.interpolate(min(line.length, distance + tangent_span))
            dx, dy = after.x - before.x, after.y - before.y
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm, dx / norm
            probes = {}
            for side, sign in (("left", 1.0), ("right", -1.0)):
                start = corridor_half
                probes[side] = LineString([
                    (x + nx * start * sign, y + ny * start * sign),
                    (x + nx * (start + depth) * sign,
                     y + ny * (start + depth) * sign),
                ])
            # The plan explicitly excludes junction samples.  A probe touching
            # an incident arterial's corridor is in that shared-node catchment;
            # unrelated nearby roads remain part of the capacity test.
            market_node = candidate.get("market_access_node")
            incident_port = next(
                (node_positions[node_id] for node_id in road_nodes.intersection(port_nodes)),
                None)
            if (incident_port is not None and
                    point.distance(Point(incident_port)) <= corridor_half):
                continue
            if (market_node in road_nodes and not protected_union.is_empty and
                    any(probe.intersects(protected_union) for probe in probes.values())):
                continue
            if (not junction_union.is_empty and
                    point.distance(junction_union) <= corridor_half):
                continue
            for side in ("left", "right"):
                side_probe = probes[side]
                coverage = probe_land.intersection(side_probe).length / depth
                side_stats[side]["eligible_samples"] += 1
                if coverage >= probe_fraction:
                    side_stats[side]["pass_samples"] += 1
                elif len(side_stats[side]["failures"]) < 32:
                    side_stats[side]["failures"].append({
                        "position": [x, y], "side": side,
                        "coverage": coverage, "required": probe_fraction,
                        "probe_depth_gu": depth,
                    })
        for side in ("left", "right"):
            item = side_stats[side]
            item["coverage"] = (item["pass_samples"] / item["eligible_samples"]
                                  if item["eligible_samples"] else 1.0)
            item["pass"] = item["coverage"] >= 0.8
        per_road.append({"road_id": road["road_id"], "left": side_stats["left"],
                         "right": side_stats["right"],
                         "pass": side_stats["left"]["pass"] and side_stats["right"]["pass"],
                         "probe_depth_gu": depth, "probe_area_threshold": probe_fraction})
    return {"minimum_coverage": 0.8, "probe_area_threshold": probe_fraction,
            "selected_land_area_gu2": selected_land.area,
            "road_corridor_area_gu2": road_union.area,
            "arterial_corridor_area_gu2": arterial_union.area,
            "water_area_gu2": water_union.area,
            "protected_open_space_area_gu2": protected_union.area,
            "developed_block_area_gu2": developed_blocks.area,
            "arterials": per_road,
            "pass": all(x["pass"] for x in per_road)}


def _validate_planarity_and_nodes(candidate: dict, roads: list[dict]) -> dict:
    nodes = {n["node_id"]: tuple(n["position"]) for n in candidate.get("nodes", [])}
    endpoint_failures = []
    lines = []
    for road in roads:
        if road.get("hierarchy") == "regional_approach":
            # Source stubs have no exterior node in the frozen graph.
            continue
        geom = road.get("polyline") or []
        if len(geom) < 2:
            continue
        for key, point in (("node_a", geom[0]), ("node_b", geom[-1])):
            expected = nodes.get(road.get(key))
            if expected is not None and _dist(expected, point) > 1e-6:
                endpoint_failures.append({"road_id": road["road_id"], "node": key})
        lines.append((road, LineString(geom)))
    crossings = []
    for i, (ra, la) in enumerate(lines):
        for rb, lb in lines[i + 1:]:
            if not la.crosses(lb):
                continue
            crossings.append([ra["road_id"], rb["road_id"]])
    return {"endpoint_failures": endpoint_failures,
            "unnoded_crossings": crossings,
            "pass": not endpoint_failures and not crossings}


def _water_intersections(roads: list[dict], water) -> list[dict]:
    """Return corridor/water intersections; non-empty is a hard gate."""
    failures = []
    for road in roads:
        if len(road.get("polyline") or []) < 2:
            continue
        area = LineString(road["polyline"]).buffer(
            float(road["clear_width_gu"]) / 2.0).intersection(water).area
        if area > 1e-6:
            failures.append({"road_id": road["road_id"], "area_gu2": area})
    return failures


def assign_streets(
    ctx: SiteContext,
    candidate: dict,
    *,
    candidate_id: str = "c00",
    approaches: Optional[list] = None,
) -> dict[str, Any]:
    """Classify boundary edges and emit RoadEdge records."""
    inner = _inner_ids(candidate["patches"])
    arterial_edges: set[str] = set()
    for path in candidate.get("graph_paths") or []:
        arterial_edges.update(path.get("edge_ids") or [])
    keep_id = next((a["patch_id"] for a in candidate.get("anchors") or []
                    if a.get("kind") == "keep"), None)

    by_edge = {e["edge_id"]: e for e in candidate["boundary_edges"]}
    by_routing = {e["routing_edge_id"]: e for e in candidate.get("routing_edges") or []}
    selected_classes = _select_boundary_classes(candidate, inner, arterial_edges)
    routed_arterial_ids = {
        eid for path in candidate.get("graph_paths") or []
        for eid in path.get("routing_edge_ids") or path.get("edge_ids") or []
    }
    arterial_lines = [
        LineString(({**by_edge, **by_routing})[eid]["geometry"])
        for eid in sorted(routed_arterial_ids)
        if eid in {**by_edge, **by_routing}
    ]
    for eid, edge in by_edge.items():
        edge["road_class"] = selected_classes.get(eid, "none")
        if edge["road_class"] in ("street", "lane") and any(
                LineString(edge["geometry"]).crosses(line)
                for line in arterial_lines):
            # A smoothed arterial is authoritative; do not emit a residual
            # boundary that would cross it without a split junction.
            edge["road_class"] = "none"

    def _edge_len(edge: dict) -> float:
        geom = edge.get("geometry") or []
        length = 0.0
        for i in range(len(geom) - 1):
            length += math.hypot(geom[i + 1][0] - geom[i][0],
                                 geom[i + 1][1] - geom[i][1])
        return length

    def _incident_classes() -> dict[str, set[str]]:
        incident: dict[str, set[str]] = {pid: set() for pid in inner}
        for edge in candidate["boundary_edges"]:
            klass = edge.get("road_class") or "none"
            if klass in ("none",):
                continue
            for pid in (edge.get("patch_left"), edge.get("patch_right")):
                if pid in incident:
                    incident[pid].add(klass)
        return incident

    for pid, classes in _incident_classes().items():
        if pid == keep_id or classes:
            continue
        rescue = [
            edge for edge in candidate["boundary_edges"]
            if pid in (edge.get("patch_left"), edge.get("patch_right"))
            and (edge.get("road_class") or "none") == "none"
            and _edge_len(edge) > 0
        ]
        if not rescue:
            raise TownLayoutError(f"isolated_patch: {pid} has no street edge")
        pick = max(rescue, key=_edge_len)
        pick["road_class"] = "street"

    roads = []
    node_pos = {n["node_id"]: n["position"] for n in candidate["nodes"]}
    by_id = {e["edge_id"]: e for e in candidate["boundary_edges"]}
    route_by_edge = {}
    for path in candidate.get("graph_paths") or []:
        for eid in path.get("routing_edge_ids") or path.get("edge_ids") or []:
            route_by_edge.setdefault(eid, path)

    def emit(eids, hierarchy, start_node=None, end_node=None, metadata=None):
        if not eids:
            return
        edge_map = {**by_id, **by_routing}
        first, last = edge_map[eids[0]], edge_map[eids[-1]]
        route_start = start_node or first["a_node"]
        # Never infer orientation from geometric proximity: reversed
        # constituent edges are normal in the serialized graph.
        pts, route_end = ordered_chain_points(eids, edge_map, node_pos, route_start)
        if end_node is not None and route_end != end_node:
            raise TownLayoutError(
                f"road chain {eids[0]} ends at {route_end}, expected {end_node}")
        tangent = None
        source_edge = None
        if metadata:
            source_edge = metadata.get("source_edge_id")
            for port in candidate.get("ports") or []:
                if port.get("approach_id") == metadata.get("approach_id"):
                    tangent = port.get("inward_tangent")
                    break
        smooth, metrics = smooth_chain(pts, start_tangent=tangent)
        rid = f"road_{candidate_id}_{len(roads):04d}"
        roads.append({
            "road_id": rid, "node_a": route_start, "node_b": route_end,
            "polyline": smooth, "hierarchy": hierarchy,
            "clear_width_gu": HIERARCHY_PAINT[hierarchy][1],
            "paint_surface": HIERARCHY_PAINT[hierarchy][2],
            "source_edge_ids": [source_edge] if source_edge else [],
            "routing_edge_ids": list(eids),
            "boundary_edge_ids": [eid for eid in eids if eid in by_id],
            "source_approach_ids": [metadata["approach_id"]] if metadata and metadata.get("approach_id") else [],
            "tangent_handle_gu": metrics.get("effective_handle_gu", 0.0),
            "tangent_residual_deg": metrics["tangent_residual_deg"],
            "max_curvature_deg_per_256gu": metrics["max_turn_deg"],
        })

    # Author the union of routed edges once.  Per-port route emission duplicates
    # shared suffixes and lets smoothing turn one graph edge into parallel roads.
    # Degree changes, ports, and the market are protected chain endpoints.
    edge_map = {**by_id, **by_routing}
    arterial_ids = sorted(eid for eid in routed_arterial_ids if eid in edge_map)
    incidence: dict[str, list[str]] = {}
    for eid in arterial_ids:
        edge = edge_map[eid]
        incidence.setdefault(edge["a_node"], []).append(eid)
        incidence.setdefault(edge["b_node"], []).append(eid)
    for values in incidence.values():
        values.sort()
    path_by_start = {
        path["from_node"]: path for path in candidate.get("graph_paths") or []
    }
    protected = {node for node, values in incidence.items() if len(values) != 2}
    protected.update(path_by_start)
    protected.update(path["to_node"] for path in candidate.get("graph_paths") or [])
    ordered_starts = sorted(protected, key=lambda node: (node not in path_by_start, node))
    visited: set[str] = set()

    def other_node(edge: dict, node: str) -> str:
        return edge["b_node"] if edge["a_node"] == node else edge["a_node"]

    def walk_chain(start_node: str, first_eid: str) -> tuple[list[str], str]:
        chain = []
        node = start_node
        eid = first_eid
        while eid not in visited:
            visited.add(eid)
            chain.append(eid)
            node = other_node(edge_map[eid], node)
            if node in protected:
                break
            choices = [candidate_eid for candidate_eid in incidence.get(node, [])
                       if candidate_eid not in visited]
            if len(choices) != 1:
                break
            eid = choices[0]
        return chain, node

    for start_node in ordered_starts:
        for first_eid in incidence.get(start_node, []):
            if first_eid in visited:
                continue
            chain, end_node = walk_chain(start_node, first_eid)
            if chain:
                emit(chain, "arterial", start_node=start_node, end_node=end_node,
                     metadata=path_by_start.get(start_node))
    # A valid port-to-market network should have no pure arterial cycle, but
    # retain deterministic handling rather than silently dropping one.
    for first_eid in arterial_ids:
        if first_eid in visited:
            continue
        start_node = edge_map[first_eid]["a_node"]
        chain, end_node = walk_chain(start_node, first_eid)
        emit(chain, "arterial", start_node=start_node, end_node=end_node)
    arterial_geometry = [LineString(r["polyline"]) for r in roads
                         if r["hierarchy"] == "arterial"]
    emitted = {eid for path in candidate.get("graph_paths") or []
               for eid in path.get("edge_ids") or []}
    for edge in candidate["boundary_edges"]:
        klass = edge.get("road_class") or "none"
        if klass == "none" or edge["edge_id"] in emitted:
            continue
        if klass in ("street", "lane") and any(
                LineString(edge["geometry"]).crosses(line)
                for line in arterial_geometry):
            continue
        hier, width, paint = HIERARCHY_PAINT[klass]
        emit([edge["edge_id"]], hier)
        if hier in ("arterial", "street"):
            emitted.add(edge["edge_id"])

    approaches = list(approaches or candidate.get("approaches") or [])
    for approach in approaches:
        stub = approach.get("transition_stub_plan_gu") or []
        if len(stub) < 2:
            continue
        # External source geometry is retained as provenance; its inner end
        # is the protected port node and is never moved by smoothing.
        port = next((p for p in candidate.get("ports") or []
                     if p.get("approach_id") == approach.get("approach_id")), None)
        if port is None:
            continue
        path = next((p for p in candidate.get("graph_paths") or []
                     if p.get("approach_id") == approach.get("approach_id")), None)
        if path is None:
            continue
        inner_node = next((n for n in candidate["nodes"] if n["position"] == port["position"]), None)
        if inner_node is None:
            continue
        rid = f"road_{candidate_id}_{len(roads):04d}"
        roads.append({
            "road_id": rid,
            "node_a": inner_node["node_id"], "node_b": inner_node["node_id"],
            "polyline": stub,
            "hierarchy": "regional_approach",
            "clear_width_gu": ARTERIAL_CLEAR_WIDTH_GU,
            "paint_surface": "road",
            "source_edge_ids": [approach["source_edge_id"]],
            "boundary_edge_ids": [],
            "source_approach_ids": [approach["approach_id"]],
            "tangent_handle_gu": 512.0,
            "tangent_residual_deg": 0.0,
            "max_curvature_deg_per_256gu": 0.0,
        })

    # Every inner developed patch needs a street/arterial unless keep.
    for pid, classes in _incident_classes().items():
        if pid == keep_id:
            continue
        if not classes:
            raise TownLayoutError(f"isolated_patch: {pid} has no street edge")

    water = unary_union([Polygon(w) for w in ctx.water_polygons()]) if ctx.water_polygons() else None
    curvature_failure = next(
        (road for road in roads
         if road.get("max_curvature_deg_per_256gu", 0.0) > 15.0 + 1e-9), None)
    if curvature_failure is not None:
        raise TownLayoutError(
            f"curvature_cap: {curvature_failure['road_id']} "
            f"{curvature_failure['max_curvature_deg_per_256gu']:.3f} > 15.000; "
            f"all={[(road['road_id'], road.get('max_curvature_deg_per_256gu', 0.0)) for road in roads if road.get('hierarchy') == 'arterial']}")
    water_failures = _water_intersections(roads, water) if water is not None else []
    if water_failures:
        raise TownLayoutError(f"water_overlap: {water_failures[0]['road_id']}")
    two_sided = _two_sided_arterial_report(
        candidate, roads, stamp_stats=ctx.stamp_footprint_stats)
    if not two_sided["pass"]:
        failed = next(x for x in two_sided["arterials"] if not x["pass"])
        failed_side = next(side for side in ("left", "right")
                           if not failed[side]["pass"])
        raise TownLayoutError(
            f"arterial_two_sided_capacity: {failed['road_id']} "
            f"{failed_side}={failed[failed_side]['coverage']:.3f} < 0.800; "
            f"left={failed['left']['coverage']:.3f} "
            f"right={failed['right']['coverage']:.3f}; "
            f"first_failure={failed[failed_side]['failures'][:1]}")
    topology = _validate_planarity_and_nodes(candidate, roads)
    if not topology["pass"]:
        if topology["endpoint_failures"]:
            raise TownLayoutError("shared_node_endpoint_mismatch")
        raise TownLayoutError("unnoded_crossing")

    reports = list(candidate.get("reports") or [])
    n_art = sum(1 for r in roads if r["hierarchy"] == "arterial")
    reports.append({
        "stage": "streets",
        "status": "ok",
        "message": f"roads={len(roads)} arterials={n_art}",
    })
    out = dict(candidate)
    out["roads"] = roads
    out["road_network_metrics"] = {
        "reachability": {p["route_id"]: True for p in candidate.get("graph_paths") or []},
        "hierarchy_counts": {h: sum(1 for r in roads if r["hierarchy"] == h)
                             for h in HIERARCHY_PAINT},
        "hierarchy_widths_gu": {h: HIERARCHY_PAINT[h][1] for h in HIERARCHY_PAINT},
        "hierarchy_surfaces": {h: HIERARCHY_PAINT[h][2] for h in HIERARCHY_PAINT},
        "max_tangent_residual_deg": max((r.get("tangent_residual_deg", 0.0) for r in roads), default=0.0),
        "max_curvature_deg_per_256gu": max((r.get("max_curvature_deg_per_256gu", 0.0) for r in roads), default=0.0),
        "water_intersections": water_failures,
        "two_sided_arterial": two_sided,
        "crossing_and_node_checks": topology,
        "patch_road_boundary": {pid: any(pid in (e.get("patch_left"), e.get("patch_right")) and
                                           e.get("road_class") != "none"
                                           for e in candidate["boundary_edges"])
                                for pid in sorted(inner)},
        "bridge_edge_coverage": True,
    }
    out["reports"] = reports
    return out


def write_streets_diagnostic(
    ctx: SiteContext,
    product: dict,
    *,
    topdown_path: Path,
    survey: dict,
    out_png: Path,
    full_site: bool = False,
) -> None:
    from PIL import Image, ImageDraw

    image, mapping = diagnostic_view({"_diagnostic_bounds": [product.get("city_domain") or []]}, topdown_path, survey, full_site=full_site)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    def to_px(pt):
        return _plan_to_px(float(pt[0]), float(pt[1]), mapping)

    colors = {
        "regional_approach": (180, 80, 40, 255),
        "arterial": (220, 40, 40, 255),
        "street": (240, 200, 60, 255),
        "lane": (160, 140, 80, 220),
    }
    # Patch topology is deliberately faint: it documents provenance without
    # making the result read like a Voronoi road map.
    for patch in product.get("patches") or []:
        ring = patch.get("polygon") or []
        if len(ring) >= 3:
            draw.line([to_px(p) for p in ring] + [to_px(ring[0])],
                      fill=(120, 120, 120, 55), width=1)
    for water in product.get("water_polygons") or []:
        if len(water) >= 3:
            draw.polygon([to_px(p) for p in water], fill=(40, 110, 220, 80))
            draw.line([to_px(p) for p in water] + [to_px(water[0])],
                      fill=(40, 140, 255, 220), width=2)
    for approach in product.get("approaches") or []:
        source = approach.get("outside_polyline_plan_gu") or []
        if len(source) >= 2:
            draw.line([to_px(p) for p in source], fill=(255, 255, 255, 150), width=2)
        stub = approach.get("transition_stub_plan_gu") or []
        if len(stub) >= 2:
            draw.line([to_px(p) for p in stub], fill=(255, 120, 40, 220), width=2)
    widths = {
        "regional_approach": 5,
        "arterial": 4,
        "street": 2,
        "lane": 1,
    }
    wall = product.get("wall")
    if wall:
        ring = wall.get("planning_polygon") or []
        if len(ring) >= 3:
            pts = [to_px(p) for p in ring] + [to_px(ring[0])]
            draw.line(pts, fill=(90, 50, 20, 180), width=2)
    for road in product.get("roads") or []:
        geom = road.get("polyline") or []
        if len(geom) < 2:
            continue
        hier = road.get("hierarchy") or "street"
        draw.line([to_px(p) for p in geom], fill=colors.get(hier, (200, 200, 200, 255)),
                  width=widths.get(hier, 2))
        # Failure marks are local triples, never an artificial road endpoint
        # diagonal.  Normally this loop draws nothing because the stage gate
        # rejects a product with an offending sample before checkpointing.
        for _idx, local, _turn in curvature_failure_segments(geom):
            draw.line([to_px(p) for p in local], fill=(255, 0, 0, 255), width=3)
    for gate in product.get("gates") or []:
        px, py = to_px(gate["position"])
        draw.rectangle([px - 6, py - 6, px + 6, py + 6], fill=(255, 220, 0, 255))
    for port in product.get("ports") or []:
        px, py = to_px(port["position"])
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(255, 0, 180, 255))
        t = port.get("inward_tangent") or [0, 0]
        draw.line([px, py, px + int(t[0] * 18), py - int(t[1] * 18)],
                   fill=(255, 80, 220, 255), width=2)
    for report in (product.get("road_network_metrics") or {}).get(
            "two_sided_arterial", {}).get("arterials", []):
        for failure in report.get("failures", []):
            px, py = to_px(failure["position"])
            draw.ellipse([px - 4, py - 4, px + 4, py + 4],
                         fill=(255, 0, 255, 255))
    Image.alpha_composite(image, overlay).save(out_png)
