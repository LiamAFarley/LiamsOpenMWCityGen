"""Grow a connected minor-road network over Stage-B block boundaries.

The accepted arterial corridor is the permanent network root. Candidate roads
are exact Stage-B interior/outer block edges; every selected edge is attached
constructively to the existing network, gives meaningful block frontage, and
is buffered into real reserved road space. Degree-two runs receive bounded
fillets while junction coordinates remain exact.

Inputs: ``r2w_inner_wall`` checkpoint. Outputs: ``r2c_minor_roads`` with
selected edges/strokes, corridor rings, final corridor-subtracted block faces,
frontage, isolation, area accounting, and visual-review metrics.
"""
from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from typing import Any

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .arterial_routes import fillet_arc
from .geometry import normalize_ring, polygon_from_ring
from .validate import TownLayoutError

MIN_FRONTAGE_GU = 512.0
MINOR_HALF_WIDTH_GU = 192.0
MAX_DEGREE = 4
MAX_TURN_DEG = 150.0
NODE_SCALE = 100.0


def _node_key(point) -> tuple[int, int]:
    return int(round(float(point[0]) * NODE_SCALE)), int(round(float(point[1]) * NODE_SCALE))


def _point(key: tuple[int, int]) -> list[float]:
    return [key[0] / NODE_SCALE, key[1] / NODE_SCALE]


def _parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [g for g in geometry.geoms if g.geom_type == "Polygon" and not g.is_empty]


def _ring(poly: Polygon) -> list[list[float]]:
    return normalize_ring([[float(x), float(y)] for x, y in poly.exterior.coords])["ring"]


def _turn(a, b, c) -> float:
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu <= 1e-9 or nv <= 1e-9:
        return 180.0
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))))
    return 180.0 - angle


def _path_to_block(target: str, required: float, candidates: dict[str, dict],
                   adjacency: dict[tuple[int, int], list[str]], network_nodes: set,
                   selected: set[str], degree: dict, frontage: dict[str, float]):
    """Least-cost unused path from the current network to useful frontage."""
    queue = []
    best = {}
    for node in sorted(network_nodes):
        state = (node, None)
        best[state] = 0.0
        heapq.heappush(queue, (0.0, (), node, None, (), 0.0))
    while queue:
        cost, edge_seq, node, incoming, node_seq, gained = heapq.heappop(queue)
        if cost > best.get((node, incoming), math.inf) + 1e-9:
            continue
        if gained >= required and edge_seq:
            return cost, edge_seq
        for edge_id in adjacency.get(node, []):
            if edge_id in selected or edge_id in edge_seq:
                continue
            edge = candidates[edge_id]
            other = edge["b"] if edge["a"] == node else edge["a"]
            if degree.get(node, 0) + 1 > MAX_DEGREE or degree.get(other, 0) + 1 > MAX_DEGREE:
                continue
            turn = 0.0
            if incoming is not None:
                before = candidates[incoming]
                prev = before["a"] if before["b"] == node else before["b"]
                turn = _turn(_point(prev), _point(node), _point(other))
                if turn >= MAX_TURN_DEG:
                    continue
            boundary_factor = 1.25 if edge["edge_class"] == "outer_candidate" else 1.0
            attach_degree = degree.get(node, 0)
            step = edge["length_gu"] * boundary_factor + 384.0 * (turn / 90.0) ** 2
            step += 1024.0 * max(0, attach_degree - 2) ** 2
            new_cost = cost + step
            incident = target in (edge.get("left_block_id"), edge.get("right_block_id"))
            new_gained = gained + (edge["length_gu"] if incident else 0.0)
            state = (other, edge_id)
            # Keep equal-cost alternatives distinguishable by their complete
            # edge sequence, the deterministic tie-break in the phase plan.
            if new_cost <= best.get(state, math.inf) + 1e-9:
                best[state] = new_cost
                heapq.heappush(queue, (new_cost, edge_seq + (edge_id,), other,
                                       edge_id, node_seq + (node,), new_gained))
    return None


def _selected_strokes(candidates: dict[str, dict], selected: set[str], city_land, water):
    adjacency: dict[tuple[int, int], list[str]] = defaultdict(list)
    for edge_id in selected:
        edge = candidates[edge_id]
        adjacency[edge["a"]].append(edge_id)
        adjacency[edge["b"]].append(edge_id)
    visited = set()
    strokes = []

    def walk(start_node, first_edge):
        nodes = [start_node]
        edges = []
        node, edge_id = start_node, first_edge
        while edge_id not in visited:
            visited.add(edge_id)
            edges.append(edge_id)
            edge = candidates[edge_id]
            node = edge["b"] if edge["a"] == node else edge["a"]
            nodes.append(node)
            options = [e for e in adjacency[node] if e not in visited]
            if len(adjacency[node]) != 2 or not options:
                break
            edge_id = options[0]
        return nodes, edges

    starts = sorted(n for n, edges in adjacency.items() if len(edges) != 2)
    for node in starts:
        for edge_id in sorted(adjacency[node]):
            if edge_id not in visited:
                nodes, edges = walk(node, edge_id)
                strokes.append((nodes, edges))
    for edge_id in sorted(selected):
        if edge_id not in visited:
            nodes, edges = walk(candidates[edge_id]["a"], edge_id)
            strokes.append((nodes, edges))

    rows = []
    for index, (nodes, edges) in enumerate(strokes):
        points = [_point(n) for n in nodes]
        out = [points[0]]
        for i in range(1, len(points) - 1):
            node = nodes[i]
            arc = None
            if len(adjacency[node]) == 2:
                l1 = LineString([points[i - 1], points[i]]).length
                l2 = LineString([points[i], points[i + 1]]).length
                cap = min(256.0, 0.25 * min(l1, l2))
                for radius in (256.0, 128.0, 64.0, 0.0):
                    if radius <= 0 or radius > cap:
                        continue
                    candidate_arc = fillet_arc(points[i - 1], points[i], points[i + 1], radius)
                    if not candidate_arc:
                        continue
                    band = LineString(candidate_arc).buffer(MINOR_HALF_WIDTH_GU)
                    if band.difference(city_land).area <= 1.0 and band.intersection(water).area <= 1.0:
                        arc = candidate_arc
                        break
            if arc:
                # Preserve the straight approach from the preceding node to
                # the fillet tangent. Replacing ``out[-1]`` here erased that
                # approach, which could visually sever a rooted street even
                # though the unsmoothed candidate graph remained connected.
                out.extend(arc)
            else:
                out.append(points[i])
        out.append(points[-1])
        rows.append({"stroke_id": f"minor_stroke_{index:03d}",
                     "edge_ids": edges, "geometry": out})
    return rows, adjacency


def build_minor_roads(product: dict[str, Any], has_inner_wall: bool | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    if product.get("stage_id") != "r2w_inner_wall":
        raise TownLayoutError("C input: expected r2w_inner_wall")
    city_land = unary_union([polygon_from_ring(p["polygon"]) for p in product.get("patches") or []
                             if p.get("inside_city")])
    water = unary_union([polygon_from_ring(r) for r in product.get("water_polygons") or []])
    arterial = unary_union([polygon_from_ring(r) for r in (product.get("corridor") or {}).get("rings") or []])
    arterial = arterial.intersection(city_land)
    arterial_centerlines = unary_union([LineString(s["geometry"])
                                        for s in product.get("smoothed_strokes") or []])
    _inner = product.get("inner_wall")
    if _inner and _inner.get("polygon"):
        wall_polygon = polygon_from_ring(_inner["polygon"])
        wall = wall_polygon.boundary
    else:
        from shapely.geometry import Polygon as _Polygon
        wall_polygon = _Polygon()
        wall = wall_polygon.boundary
    gate_reserve_parts = [polygon_from_ring(ring) for ring in
                          (product.get("wall_fit") or {}).get("gate_reserves", [])
                          if len(ring) >= 3]
    gate_reserve = unary_union(gate_reserve_parts) if gate_reserve_parts else None
    mesh_reserve_parts = [polygon_from_ring(ring) for ring in
                          (product.get("wall_fit") or {}).get("mesh_footprints", [])
                          if len(ring) >= 3]
    mesh_geometry = unary_union(mesh_reserve_parts) if mesh_reserve_parts else None
    mesh_clearance = float((product.get("wall_fit") or {}).get(
        "reserve_clearance_gu", 0.0))
    mesh_reserve = (mesh_geometry.buffer(mesh_clearance, cap_style=2, join_style=2)
                    if mesh_geometry is not None and mesh_clearance > 0.0
                    else mesh_geometry)
    blocked_gate_edges = 0
    blocked_wall_edges = 0

    candidates = {}
    graph: dict[tuple[int, int], list[str]] = defaultdict(list)
    for row in product.get("block_edges") or []:
        if row.get("edge_class") not in ("interior_candidate", "outer_candidate"):
            continue
        geometry = row.get("geometry") or []
        if len(geometry) < 2 or row.get("length_gu", 0) <= 0:
            continue
        line = LineString(geometry)
        if has_inner_wall is not False and (line.crosses(wall) or line.intersection(wall).length > 1.0):
            continue
        if gate_reserve is not None and line.intersects(gate_reserve):
            blocked_gate_edges += 1
            continue
        if (mesh_reserve is not None and
                line.buffer(MINOR_HALF_WIDTH_GU, cap_style=2, join_style=2)
                .intersects(mesh_reserve)):
            blocked_wall_edges += 1
            continue
        inside_band = line.buffer(MINOR_HALF_WIDTH_GU).intersection(city_land)
        if inside_band.intersection(water).area > 1.0:
            continue
        edge = dict(row)
        edge["a"], edge["b"] = _node_key(geometry[0]), _node_key(geometry[-1])
        candidates[row["edge_id"]] = edge
        graph[edge["a"]].append(row["edge_id"])
        graph[edge["b"]].append(row["edge_id"])
    for node in graph:
        graph[node].sort()

    root_links = {}
    for node in sorted(graph):
        source = Point(_point(node))
        if source.distance(arterial_centerlines) > 300.0:
            continue
        target = arterial_centerlines.interpolate(arterial_centerlines.project(source))
        link = LineString([source, target])
        if gate_reserve is not None and link.intersects(gate_reserve):
            continue
        if arterial.buffer(1.0).covers(link):
            root_links[node] = link
    network_nodes = set(root_links)
    if not network_nodes:
        raise TownLayoutError("C connectivity: no candidate node reaches arterial corridor")
    selected: set[str] = set()
    degree: dict[tuple[int, int], int] = defaultdict(int)
    frontage = {b["block_id"]: float(b.get("arterial_frontage_gu", 0.0))
                for b in product.get("blocks") or []}
    added_rows = []
    isolated_ids = set()

    while True:
        unserved = sorted(block for block, length in frontage.items()
                          if length + 1e-6 < MIN_FRONTAGE_GU and block not in isolated_ids)
        if not unserved:
            break
        options = []
        for block_id in unserved:
            found = _path_to_block(block_id, MIN_FRONTAGE_GU - frontage[block_id],
                                   candidates, graph, network_nodes, selected, degree, frontage)
            if found is not None:
                options.append((round(found[0], 9), block_id, found[1]))
        if not options:
            isolated_ids.update(unserved)
            break
        _, target, path = min(options, key=lambda row: (row[0], row[1], row[2]))
        for edge_id in path:
            if edge_id in selected:
                continue
            edge = candidates[edge_id]
            selected.add(edge_id)
            degree[edge["a"]] += 1
            degree[edge["b"]] += 1
            network_nodes.update((edge["a"], edge["b"]))
            for block_id in (edge.get("left_block_id"), edge.get("right_block_id")):
                if block_id in frontage:
                    frontage[block_id] += edge["length_gu"]
            added_rows.append({"edge_id": edge_id, "added_step": len(added_rows),
                               "purpose": "frontage", "target_block_id": target})

    # A 512-GU frontage target alone can leave implausible nib-like streets.
    # Extend only short non-arterial dead ends, following the smoothest legal
    # unused continuation. This is bounded and constructive; it does not add a
    # nearest-neighbour bridge or alter the candidate fabric.
    for _ in range(len(candidates)):
        selected_graph: dict[tuple[int, int], list[str]] = defaultdict(list)
        for edge_id in selected:
            edge = candidates[edge_id]
            selected_graph[edge["a"]].append(edge_id)
            selected_graph[edge["b"]].append(edge_id)
        choice = None
        for node in sorted(selected_graph):
            if len(selected_graph[node]) != 1 or node in root_links:
                continue
            incoming_id = selected_graph[node][0]
            incoming = candidates[incoming_id]
            previous = incoming["a"] if incoming["b"] == node else incoming["b"]
            # Measure the complete current branch back to a junction/root.
            branch_length = incoming["length_gu"]
            cursor, prior_edge = previous, incoming_id
            while len(selected_graph.get(cursor, [])) == 2 and cursor not in root_links:
                next_edge = next(e for e in selected_graph[cursor] if e != prior_edge)
                branch_length += candidates[next_edge]["length_gu"]
                next_row = candidates[next_edge]
                cursor = next_row["a"] if next_row["b"] == cursor else next_row["b"]
                prior_edge = next_edge
            if branch_length >= 1536.0:
                continue
            for edge_id in graph.get(node, []):
                if edge_id in selected:
                    continue
                edge = candidates[edge_id]
                other = edge["b"] if edge["a"] == node else edge["a"]
                if degree.get(node, 0) + 1 > MAX_DEGREE or degree.get(other, 0) + 1 > MAX_DEGREE:
                    continue
                turn = _turn(_point(previous), _point(node), _point(other))
                if turn >= MAX_TURN_DEG:
                    continue
                key = (round(turn, 9), round(edge["length_gu"], 6), edge_id, node, other)
                if choice is None or key < choice:
                    choice = key
        if choice is None:
            break
        _, _, edge_id, node, other = choice
        edge = candidates[edge_id]
        selected.add(edge_id)
        degree[edge["a"]] += 1
        degree[edge["b"]] += 1
        network_nodes.update((edge["a"], edge["b"]))
        for block_id in (edge.get("left_block_id"), edge.get("right_block_id")):
            if block_id in frontage:
                frontage[block_id] += edge["length_gu"]
        added_rows.append({"edge_id": edge_id, "added_step": len(added_rows),
                           "purpose": "dead_end_extension", "target_block_id": None})

    # Join nearby rooted street components when one existing candidate edge
    # connects two dead ends. The frontage pass intentionally builds a minimum
    # network and otherwise leaves conspicuous near-misses such as the two
    # eastern Falkreath branches. Keep this bounded to a single short block
    # edge and never add a cycle within an already connected component.
    for _ in range(len(candidates)):
        selected_graph: dict[tuple[int, int], list[str]] = defaultdict(list)
        for edge_id in selected:
            edge = candidates[edge_id]
            selected_graph[edge["a"]].append(edge_id)
            selected_graph[edge["b"]].append(edge_id)
        labels = {}
        component_id = 0
        for seed in sorted(selected_graph):
            if seed in labels:
                continue
            labels[seed] = component_id
            stack = [seed]
            while stack:
                node = stack.pop()
                for edge_id in selected_graph[node]:
                    edge = candidates[edge_id]
                    other = edge["b"] if edge["a"] == node else edge["a"]
                    if other not in labels:
                        labels[other] = component_id
                        stack.append(other)
            component_id += 1
        options = []
        for edge_id, edge in sorted(candidates.items()):
            if edge_id in selected or edge["length_gu"] > 2048.0:
                continue
            a, b = edge["a"], edge["b"]
            if len(selected_graph.get(a, [])) != 1 or len(selected_graph.get(b, [])) != 1:
                continue
            if labels.get(a) == labels.get(b):
                continue
            legal = True
            for node, other in ((a, b), (b, a)):
                incident = candidates[selected_graph[node][0]]
                previous = incident["a"] if incident["b"] == node else incident["b"]
                if _turn(_point(previous), _point(node), _point(other)) >= MAX_TURN_DEG:
                    legal = False
                    break
            if legal:
                options.append((round(edge["length_gu"], 6), edge_id))
        if not options:
            break
        _, edge_id = min(options)
        edge = candidates[edge_id]
        selected.add(edge_id)
        degree[edge["a"]] += 1
        degree[edge["b"]] += 1
        for block_id in (edge.get("left_block_id"), edge.get("right_block_id")):
            if block_id in frontage:
                frontage[block_id] += edge["length_gu"]
        added_rows.append({"edge_id": edge_id, "added_step": len(added_rows),
                           "purpose": "component_merge", "target_block_id": None})

    if any(value > MAX_DEGREE for value in degree.values()):
        raise TownLayoutError("C degree: selected road node exceeds 4")
    strokes, selected_adjacency = _selected_strokes(candidates, selected, city_land, water)
    unseen_nodes = {node for node, edges in selected_adjacency.items() if edges}
    road_components = []
    while unseen_nodes:
        seed = min(unseen_nodes)
        unseen_nodes.remove(seed)
        stack, component = [seed], {seed}
        while stack:
            node = stack.pop()
            for edge_id in selected_adjacency[node]:
                edge = candidates[edge_id]
                other = edge["b"] if edge["a"] == node else edge["a"]
                if other in unseen_nodes:
                    unseen_nodes.remove(other)
                    component.add(other)
                    stack.append(other)
        road_components.append(component)
    unrooted_components = [component for component in road_components
                           if not (component & set(root_links))]
    if unrooted_components:
        raise TownLayoutError(f"C connectivity: {len(unrooted_components)} unrooted road components")
    junction_links = []
    for node in sorted(n for n, edges in selected_adjacency.items() if edges and n in root_links):
        link = root_links[node]
        row = {"link_id": f"junction_link_{len(junction_links):03d}",
               "geometry": [[float(x), float(y)] for x, y in link.coords],
               "length_gu": float(link.length)}
        junction_links.append(row)
        strokes.append({"stroke_id": row["link_id"], "edge_ids": [],
                        "geometry": row["geometry"], "junction_link": True})
    smoothed_minor = unary_union([
        LineString(stroke["geometry"])
        for stroke in strokes if not stroke.get("junction_link")
    ])
    for row in junction_links:
        link = LineString(row["geometry"])
        if Point(link.coords[0]).distance(smoothed_minor) > 1e-6:
            raise TownLayoutError(
                f"C visible_connectivity: {row['link_id']} misses smoothed minor road")
        if Point(link.coords[-1]).distance(arterial_centerlines) > 1e-6:
            raise TownLayoutError(
                f"C visible_connectivity: {row['link_id']} misses arterial centerline")
    minor_lines = [LineString(s["geometry"]) for s in strokes]
    minor_bands = []
    for line in minor_lines:
        band = line.buffer(MINOR_HALF_WIDTH_GU, cap_style="round", join_style="round")
        midpoint = line.interpolate(line.length / 2.0)
        if has_inner_wall is not False and not arterial.covers(midpoint):
            band = (band.intersection(wall_polygon) if wall_polygon.covers(midpoint)
                    else band.difference(wall_polygon))
        minor_bands.append(band)
    minor_corridor = unary_union(minor_bands) if minor_bands else Polygon()
    minor_inside = minor_corridor.intersection(city_land)
    wet = minor_inside.intersection(water).area
    if wet > 1.0:
        raise TownLayoutError(f"C water_overlap {wet:.3f}")
    full_corridor = unary_union([arterial, minor_inside])

    open_landscape_ids = set(isolated_ids)
    final_blocks = []
    final_union_parts = []
    for block in product.get("blocks") or []:
        original = polygon_from_ring(block["polygon"])
        remainder = original.difference(minor_inside)
        parts = _parts(remainder)
        if not parts:
            raise TownLayoutError(f"C final_block erased {block['block_id']}")
        final_union_parts.extend(parts)
        final_blocks.append({
            **block,
            "polygons": [_ring(poly) for poly in sorted(parts, key=lambda p: (-p.area, p.wkt))],
            "final_area_gu2": float(sum(poly.area for poly in parts)),
            "final_frontage_gu": float(frontage[block["block_id"]]),
            "isolated": False,
            "classification": ("open_landscape" if block["block_id"] in open_landscape_ids
                               else "buildable_block"),
            "buildable": block["block_id"] not in open_landscape_ids,
        })

    represented = unary_union([full_corridor] + final_union_parts
                              + [polygon_from_ring(v["polygon"]) for v in product.get("road_verges") or []]
                              + [polygon_from_ring(i["polygon"]) for i in product.get("isolated_areas") or []])
    gap = city_land.difference(represented).area
    if gap > 1.0:
        raise TownLayoutError(f"C area_accounting gap={gap:.3f}")

    minor_edges = []
    by_added = {row["edge_id"]: row for row in added_rows}
    for edge_id in sorted(selected):
        row = candidates[edge_id]
        minor_edges.append({k: v for k, v in row.items() if k not in ("a", "b")}
                           | by_added[edge_id])
    runtime = time.perf_counter() - started
    metrics = {
        "runtime_s": runtime, "candidate_edge_count": len(candidates),
        "selected_minor_edge_count": len(selected), "minor_stroke_count": len(strokes),
        "minor_road_length_gu": float(sum(candidates[e]["length_gu"] for e in selected)),
        "gate_reserve_blocked_edge_count": blocked_gate_edges,
        "wall_mesh_blocked_edge_count": blocked_wall_edges,
        "minor_corridor_area_gu2": float(minor_inside.area),
        "junction_link_count": len(junction_links),
        "minor_component_count": len(road_components),
        "unrooted_component_count": 0,
        "isolated_block_count": 0,
        "open_landscape_count": len(open_landscape_ids),
        "frontage_failure_count": sum(v < MIN_FRONTAGE_GU for k, v in frontage.items() if k not in isolated_ids),
        "max_degree": max(degree.values(), default=0),
        "water_overlap_gu2": float(wet), "unexplained_gap_gu2": float(gap),
    }
    if runtime > 10.0:
        raise TownLayoutError(f"C runtime {runtime:.2f}s")
    result = dict(product)
    result.update({
        "stage_id": "r2c_minor_roads", "preceding_checkpoint": None,
        "minor_road_edges": minor_edges, "minor_strokes": strokes,
        "junction_links": junction_links,
        "minor_corridor": {"half_width_gu": MINOR_HALF_WIDTH_GU,
                           "rings": [_ring(poly) for poly in _parts(minor_inside)]},
        # Keep arterial and minor bands as separate fill rings. Their geometric
        # union can contain a city-block hole once two rooted street components
        # are joined; serializing only that polygon's exterior would fill the
        # enclosed block as road in consumers that use the legacy ring shape.
        "full_road_corridor": {
            "rings": ([_ring(poly) for poly in _parts(arterial)]
                      + [_ring(poly) for poly in _parts(minor_inside)])
        },
        "final_blocks": final_blocks,
        "new_isolated_areas": [],
        "open_landscapes": [{"block_id": b,
                             "reason": "unserved_waterfront_outside_inner_wall"}
                            for b in sorted(open_landscape_ids)],
        "metrics": metrics,
        "reports": list(product.get("reports") or []) + [{
            "stage": "r2c_minor_roads", "status": "ok",
            "message": (f"minor_edges={len(selected)} strokes={len(strokes)} "
                        f"open_landscape={len(open_landscape_ids)}"),
        }],
    })
    return result
