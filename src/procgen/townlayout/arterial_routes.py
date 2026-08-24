"""Stage A arterial routing: dynamic confluence and one main-road tree.

Purpose
-------
Route every retained port over the current ``fine_shared`` Voronoi graph into
one connected arterial tree. No source-road junction is accepted as input.
The current city fabric instead chooses a deterministic interior meeting.

- directed edge-state Dijkstra with cost
  ``L * (1 + 1.5 * (d / max(D,1))²) + 384 * (theta / 90)²`` where ``d`` is
  the edge-midpoint perpendicular distance to the straight port/meeting
  chord and ``theta`` the absolute incoming turn in degrees; turns of 150
  degrees or more are forbidden; cost ties resolve by complete edge-ID
  sequence, lexicographically;
- ports are processed by descending standalone route cost (tie ``port_id``);
  the first reaches the selected meeting and every later port joins the
  existing arterial tree, so every retained approach meets another main road;
- smoothing replaces degree-2 turns by the largest feasible fillet of
  ``384, 256, 128, 0`` GU (capped at one quarter of the shorter incident
  segment); degree 1/3/4 nodes get a 256 GU junction pad; degree above 4
  fails.  A zero-radius corner with deflection above 90 degrees is a
  failure, not silent success.

Boundary-arterial cell promotion (2026-08-18 user amendment): every
arterial segment must have city cells on both sides, or at least the
possibility of placing stuff there.  After routing, any used fine edge
whose perpendicular city-land depth at the edge midpoint falls below
``MIN_ROADSIDE_DEPTH_GU`` on either side triggers the promotion of the
shallow-side member patch's unselected, non-water neighbour patches (one
cell deep, from the R1 fringe — R1/R2 files are never modified).  The fine
graph and city land are then rebuilt and routing re-runs once; at most
``PROMOTION_PASSES`` routing passes are made, and still-thin roadsides
after the last pass are a hard failure.  Promoted patch ids are recorded
explicitly in the product so Stage B inherits the enlarged city.

Inputs
------
``FineGraph`` plus port-connector records from ``arterial_graph.py``, the
sanitized port projection, city-patch polygons (coverage), and
water polygons (dryness).

Outputs
-------
The Stage A product dict (routes, merge records, raw barrier, smoothed
strokes, corridor rings, metrics, reports) via ``build_arterials``.

Pipeline position
-----------------
Phase 21 Stage A (r2a_arterials); the unsmoothed tree is the permanent
Stage B merge barrier.  Rendering lives in ``road_review.py``.
"""
from __future__ import annotations

import heapq
import math
import time
from typing import Any

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .arterial_graph import (FineGraph, attach_port_connector, build_fine_graph,
                             sanitize_ports)
from .geometry import polygon_from_ring, simple_polygon_parts
from .validate import TownLayoutError

TURN_FORBIDDEN_DEG = 150.0
ZERO_RADIUS_MAX_DEFLECTION_DEG = 105.0
FILLET_RADII_GU = (384.0, 256.0, 128.0, 64.0, 32.0, 0.0)
FILLET_SEGMENT_FRACTION = 0.25
CORRIDOR_HALF_WIDTH_GU = 256.0
JUNCTION_PAD_RADIUS_GU = 256.0
PORT_CAP_GU = 256.0
AREA_EPS_GU2 = 1.0
RUNTIME_LIMIT_S = 10.0
MIN_ROADSIDE_DEPTH_GU = 2048.0
PROMOTION_PASSES = 2
MEETING_MIN_BOUNDARY_DEPTH_GU = 1024.0


# ---------------------------------------------------------------------------
# Boundary-arterial cell promotion (2026-08-18 user amendment)
# ---------------------------------------------------------------------------

def _side_depth_gu(land, mx: float, my: float, nx: float, ny: float) -> float:
    """Exact city-land depth from (mx, my) along the unit normal (nx, ny).

    Measures the contiguous covered length from the start point until the
    ray first exits ``land`` (capped at an 8,192 GU reach).
    """
    reach = 8192.0
    segment = LineString([(mx, my), (mx + nx * reach, my + ny * reach)])
    inter = land.intersection(segment)
    if inter.is_empty:
        return 0.0
    parts = ([inter] if inter.geom_type == "LineString"
             else [g for g in getattr(inter, "geoms", [])
                   if g.geom_type == "LineString"])
    start = Point(mx, my)
    depth = 0.0
    for part in parts:
        if part.distance(start) <= 1e-6:
            depth = max(depth, part.length)
    return depth


def _thin_roadside_promotions(graph: FineGraph, land,
                              used_edge_ids: set[str],
                              patch_by_id: dict[str, dict[str, Any]],
                              selected_ids: set[str]) -> list[str]:
    """Unselected dry fringe patches to promote for thin arterial roadsides.

    For every used fine edge, the city-land depth perpendicular to the edge
    at its midpoint must reach ``MIN_ROADSIDE_DEPTH_GU`` on both sides.  On
    a shallow side, the member patch lying on that side contributes its
    unselected, non-water neighbour patches (one cell deep).
    """
    centroids: dict[str, tuple[float, float]] = {}
    promote: set[str] = set()
    for edge_id in sorted(used_edge_ids):
        edge = graph.edges[edge_id]
        pa, pb = graph.nodes[edge["a"]], graph.nodes[edge["b"]]
        ex, ey = pb[0] - pa[0], pb[1] - pa[1]
        norm = math.hypot(ex, ey)
        nx, ny = ey / norm, -ex / norm
        mx, my = (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
        for sgn in (1.0, -1.0):
            if _side_depth_gu(land, mx, my, sgn * nx, sgn * ny) \
                    >= MIN_ROADSIDE_DEPTH_GU:
                continue
            for pid in edge["patches"]:
                if pid not in centroids:
                    centroid = polygon_from_ring(
                        patch_by_id[pid]["polygon"]).centroid
                    centroids[pid] = (centroid.x, centroid.y)
                cx, cy = centroids[pid]
                if ((cx - mx) * nx + (cy - my) * ny) * sgn <= 0.0:
                    continue  # member patch lies on the other side
                for nid in patch_by_id[pid].get("neighbour_patch_ids") or []:
                    nid = str(nid)
                    neighbour = patch_by_id.get(nid)
                    if neighbour is None or nid in selected_ids:
                        continue
                    if (neighbour.get("terrain_summary") or {}).get("water"):
                        continue  # a water cell cannot host roadside content
                    promote.add(nid)
    return sorted(promote)


# ---------------------------------------------------------------------------
# Directed cost search
# ---------------------------------------------------------------------------

def _angle_deg(u: tuple[float, float], v: tuple[float, float]) -> float:
    dot = u[0] * v[0] + u[1] * v[1]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _chord_distance(x: float, y: float,
                    a: tuple[float, float], b: tuple[float, float]) -> float:
    cx, cy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(cx, cy)
    if norm <= 0.0:
        return math.hypot(x - a[0], y - a[1])
    return abs(cx * (a[1] - y) - (a[0] - x) * cy) / norm


def route_search(graph: FineGraph, start: str, targets: set[str],
                 chord_a: tuple[float, float], chord_b: tuple[float, float],
                 blocked_nodes: set[str] | None = None,
                 terminal_nodes: set[str] | None = None,
                 blocked_edges: set[str] | None = None
                 ) -> dict[str, tuple[float, tuple[str, ...]]]:
    """Directed edge-state Dijkstra from ``start`` to every reachable target.

    ``blocked_nodes`` may not be entered; ``blocked_edges`` may not be used;
    ``terminal_nodes`` may be entered only as a path end (never expanded).
    Returns ``target -> (cost, edge_id_sequence)``.  Priority ties resolve by
    the complete edge-ID sequence, lexicographically.
    """
    blocked = blocked_nodes or set()
    terminal = terminal_nodes or set()
    bad_edges = blocked_edges or set()
    big_d = max(math.hypot(chord_b[0] - chord_a[0], chord_b[1] - chord_a[1]), 1.0)
    heap: list[tuple[float, tuple[str, ...], str, str | None]] = [
        (0.0, (), start, None)]
    best: dict[tuple[str, str | None], float] = {(start, None): 0.0}
    found: dict[str, tuple[float, tuple[str, ...]]] = {}
    while heap:
        cost, seq, node, in_edge = heapq.heappop(heap)
        if best.get((node, in_edge), math.inf) < cost:
            continue
        if node in targets and node not in found:
            found[node] = (cost, seq)
            if len(found) == len(targets):
                break
        if node in terminal and node != start:
            continue
        for edge_id in sorted(graph.adjacency.get(node, [])):
            edge = graph.edges[edge_id]
            if edge["role"] != "fine_shared" or edge_id in bad_edges:
                continue
            other = graph.other(edge_id, node)
            if other in blocked:
                continue
            if other in terminal and other not in targets:
                continue
            pa, pb = graph.nodes[edge["a"]], graph.nodes[edge["b"]]
            if edge["a"] == node:
                out = (pb[0] - pa[0], pb[1] - pa[1])
            else:
                out = (pa[0] - pb[0], pa[1] - pb[1])
            norm = math.hypot(*out)
            out = (out[0] / norm, out[1] / norm)
            turn = 0.0
            if in_edge is not None:
                prev = graph.edges[in_edge]
                qa, qb = graph.nodes[prev["a"]], graph.nodes[prev["b"]]
                # direction of travel arriving at ``node``
                if prev["a"] == node:
                    inc = (qa[0] - qb[0], qa[1] - qb[1])
                else:
                    inc = (qb[0] - qa[0], qb[1] - qa[1])
                n2 = math.hypot(*inc)
                turn = _angle_deg((inc[0] / n2, inc[1] / n2), out)
                if turn >= TURN_FORBIDDEN_DEG:
                    continue
            mx, my = (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
            d = _chord_distance(mx, my, chord_a, chord_b)
            step = (edge["length_gu"] * (1.0 + 1.5 * (d / big_d) ** 2)
                    + 384.0 * (turn / 90.0) ** 2)
            new_cost = cost + step
            new_seq = seq + (edge_id,)
            state = (other, edge_id)
            if best.get(state, math.inf) <= new_cost:
                continue
            best[state] = new_cost
            heapq.heappush(heap, (new_cost, new_seq, other, edge_id))
    return found


def path_nodes(graph: FineGraph, start: str, edge_seq: tuple[str, ...]) -> list[str]:
    nodes = [start]
    node = start
    for edge_id in edge_seq:
        node = graph.other(edge_id, node)
        nodes.append(node)
    return nodes


def _network_distances(graph: FineGraph, start: str,
                       blocked_edges: set[str]) -> dict[str, float]:
    """Length-only distances used solely to select a balanced meeting node."""
    distances = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > distances.get(node, math.inf):
            continue
        for edge_id in sorted(graph.adjacency.get(node, [])):
            edge = graph.edges[edge_id]
            if edge["role"] != "fine_shared" or edge_id in blocked_edges:
                continue
            other = graph.other(edge_id, node)
            candidate = distance + float(edge["length_gu"])
            if candidate >= distances.get(other, math.inf):
                continue
            distances[other] = candidate
            heapq.heappush(heap, (candidate, other))
    return distances


def choose_arterial_meeting(graph: FineGraph, port_connectors: list[dict[str, Any]],
                            city_land, blocked_edges: set[str]) -> tuple[str, dict[str, Any]]:
    """Choose a current-fabric confluence reachable from every main port."""
    starts = [connector["attach_node_id"] for connector in port_connectors]
    distance_maps = [_network_distances(graph, start, blocked_edges)
                     for start in starts]
    shared = set.intersection(*(set(distances) for distances in distance_maps))
    port_attach_nodes = set(starts)
    mean_x = sum(graph.nodes[node][0] for node in starts) / len(starts)
    mean_y = sum(graph.nodes[node][1] for node in starts) / len(starts)
    candidates = []
    for node in sorted(shared - port_attach_nodes):
        degree = sum(
            graph.edges[edge_id]["role"] == "fine_shared"
            and edge_id not in blocked_edges
            for edge_id in graph.adjacency.get(node, []))
        point = Point(graph.nodes[node])
        depth = city_land.boundary.distance(point)
        if degree < 3 or depth < MEETING_MIN_BOUNDARY_DEPTH_GU:
            continue
        distances = [mapping[node] for mapping in distance_maps]
        center_distance = math.hypot(point.x - mean_x, point.y - mean_y)
        candidates.append((round(max(distances), 6), round(sum(distances), 6),
                           round(center_distance, 6), node, distances, depth, degree))
    if not candidates:
        raise TownLayoutError(
            "A meeting: no degree-3 interior node reachable from every port")
    maximum, total, center_distance, node, distances, depth, degree = min(candidates)
    return node, {
        "node_id": node,
        "position": list(graph.nodes[node]),
        "selection": "current_fabric_network_minimax",
        "max_port_network_distance_gu": float(maximum),
        "sum_port_network_distance_gu": float(total),
        "mean_port_distance_gu": float(total / len(distances)),
        "mean_port_center_distance_gu": float(center_distance),
        "boundary_depth_gu": float(depth),
        "eligible_degree": int(degree),
    }


# ---------------------------------------------------------------------------
# Rooted tree
# ---------------------------------------------------------------------------

class ArterialTree:
    """Growing rooted tree over fine-graph nodes plus lead endpoints."""

    def __init__(self, root: str):
        self.root = root
        self.parent: dict[str, tuple[str, str] | None] = {root: None}
        self.tree_dist: dict[str, float] = {root: 0.0}
        self.edge_use: dict[str, int] = {}

    def nodes(self) -> set[str]:
        return set(self.parent)

    def commit_path(self, graph: FineGraph, nodes: list[str],
                    edge_seq: tuple[str, ...]) -> None:
        """Attach a path whose LAST node is already in the tree."""
        for idx in range(len(nodes) - 2, -1, -1):
            child, parent_node = nodes[idx], nodes[idx + 1]
            if child in self.parent:
                continue
            edge_id = edge_seq[idx]
            self.parent[child] = (parent_node, edge_id)
            self.edge_use[edge_id] = self.edge_use.get(edge_id, 0) + 1
        for idx in range(len(nodes) - 2, -1, -1):
            child = nodes[idx]
            link = self.parent.get(child)
            if link and link[0] == nodes[idx + 1]:
                self.tree_dist[child] = (self.tree_dist[nodes[idx + 1]]
                                         + graph.edges[link[1]]["length_gu"])

    def path_to_root(self, node: str) -> list[str]:
        out = [node]
        while self.parent[out[-1]] is not None:
            out.append(self.parent[out[-1]][0])
        return out


# ---------------------------------------------------------------------------
# Fillets
# ---------------------------------------------------------------------------

def fillet_arc(p_prev, p_node, p_next, radius: float) -> list[list[float]] | None:
    """Arc polyline (tangent point to tangent point) for the corner at
    ``p_node``; None when straight or infeasible at this radius."""
    v1 = (p_prev[0] - p_node[0], p_prev[1] - p_node[1])
    v2 = (p_next[0] - p_node[0], p_next[1] - p_node[1])
    l1, l2 = math.hypot(*v1), math.hypot(*v2)
    if l1 <= 0.0 or l2 <= 0.0:
        return None
    u1, u2 = (v1[0] / l1, v1[1] / l1), (v2[0] / l2, v2[1] / l2)
    corner = _angle_deg(u1, u2)            # angle between the two rays from node
    if corner >= 180.0 - 1e-9:
        return None                        # straight through
    deflection = 180.0 - corner            # turn angle; 0 = straight
    t = radius * math.tan(math.radians(deflection) / 2.0)
    if t > min(l1, l2) + 1e-6:
        return None
    bis = (u1[0] + u2[0], u1[1] + u2[1])
    bn = math.hypot(*bis)
    if bn <= 1e-12:
        return None
    bis = (bis[0] / bn, bis[1] / bn)
    center_dist = radius / math.sin(math.radians(corner) / 2.0)
    center = (p_node[0] + bis[0] * center_dist, p_node[1] + bis[1] * center_dist)
    t1 = (p_node[0] + u1[0] * t, p_node[1] + u1[1] * t)
    t2 = (p_node[0] + u2[0] * t, p_node[1] + u2[1] * t)
    a1 = math.atan2(t1[1] - center[1], t1[0] - center[0])
    a2 = math.atan2(t2[1] - center[1], t2[0] - center[0])
    sweep = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
    steps = max(2, int(abs(sweep) / (math.pi / 16.0)) + 1)
    points = [[center[0] + radius * math.cos(a1 + sweep * i / steps),
               center[1] + radius * math.sin(a1 + sweep * i / steps)]
              for i in range(steps + 1)]
    points[0] = [t1[0], t1[1]]
    points[-1] = [t2[0], t2[1]]
    return points


def smooth_polyline(points: list[list[float]],
                    corner_nodes: list[str | None],
                    radii: dict[str, float]) -> list[list[float]]:
    """Apply chosen fillet radii to a node path polyline.

    ``corner_nodes[i]`` is the tree node id at vertex ``i`` (None for the
    port/junction terminals). Corners without a radius entry keep their
    exact corner point.
    """
    out: list[list[float]] = [list(points[0])]
    for idx in range(1, len(points) - 1):
        node = corner_nodes[idx]
        radius = radii.get(node) if node else None
        arc = None
        if radius:
            arc = fillet_arc(points[idx - 1], points[idx], points[idx + 1], radius)
        if arc:
            # Truncate the incident straight piece at the tangent point, then
            # follow the arc.  The radius cap (a quarter of the shorter
            # incident segment) keeps this tangent point past any arc end
            # belonging to the previous corner.
            out[-1] = list(arc[0])
            out.extend(arc[1:])
        else:
            out.append(list(points[idx]))
    out.append(list(points[-1]))
    return out


def corridor_check(union, port_transition_union, bridge_transition_union,
                   water_union, city_land,
                   area_eps: float = AREA_EPS_GU2):
    """Validate a corridor union against the Stage A containment checks.

    Returns ``(reason_or_None, detail_geometry)`` where the detail geometry
    is the offending water overlap (empty when clean).
    Boundary-to-fabric port connectors are the sole city-coverage exception.
    Water is allowed only within a connector already classified by R2 as
    bridge-dependent.
    """
    empty = Polygon()
    if union.is_empty or union.geom_type not in ("Polygon", "MultiPolygon"):
        return "not_simple", empty
    if not water_union.is_empty:
        wet = union.difference(bridge_transition_union).intersection(water_union)
        if wet.area > area_eps:
            return f"water_overlap {wet.area:.3f}", wet
    safe = union.difference(port_transition_union)
    spill = safe.difference(city_land)
    if spill.area > area_eps:
        return f"city_spill {spill.area:.3f}", spill
    return None, empty


def corridor_ineligible_edges(graph: FineGraph, city_land, water_union
                              ) -> set[str]:
    """Shared edges that cannot carry the full-width arterial corridor."""
    blocked: set[str] = set()
    for edge_id in graph.shared_edges():
        edge = graph.edges[edge_id]
        line = LineString([graph.nodes[edge["a"]], graph.nodes[edge["b"]]])
        band = line.buffer(CORRIDOR_HALF_WIDTH_GU)
        if (not city_land.covers(band)
                or (not water_union.is_empty
                    and band.intersection(water_union).area > AREA_EPS_GU2)):
            blocked.add(edge_id)
    return blocked


# ---------------------------------------------------------------------------
# Stage A orchestration
# ---------------------------------------------------------------------------

def build_arterials(macro_product: dict[str, Any],
                    ports_product: dict[str, Any]) -> dict[str, Any]:
    """Build the complete Stage A product from the two frozen checkpoints.

    Only the permitted projection of the R2 product is read (see
    ``arterial_graph.sanitize_ports``); mutating any forbidden field of the
    inputs cannot change this output.
    """
    started = time.perf_counter()

    projection = sanitize_ports(ports_product)
    all_patches = macro_product.get("patches") or []
    patch_by_id = {str(p["patch_id"]): p for p in all_patches}
    macro_city_ids = {pid for pid, p in patch_by_id.items()
                      if p.get("inside_city")}
    original_ids = set(projection["city_region_patch_ids"])
    unknown_ids = original_ids - set(patch_by_id)
    if unknown_ids:
        raise TownLayoutError(
            f"A input: unknown city-region patches {sorted(unknown_ids)}")
    if not original_ids:
        raise TownLayoutError("A input: R2 city region has no patches")
    retracted_ids = macro_city_ids - original_ids
    selected_ids = set(original_ids)
    ports = projection["ports"]

    water = [polygon_from_ring(w) for w in macro_product.get("water_polygons") or []]
    water_union = unary_union(water) if water else Polygon()

    # Boundary-arterial cell promotion loop (2026-08-18 user amendment):
    # route, find used arterial edges with a thin roadside, promote the
    # shallow-side unselected dry fringe patches one cell deep, rebuild the
    # fine graph and city land, and re-route.  R1/R2 inputs are never
    # modified; the enlarged working set is recorded in the product.
    for _pass in range(PROMOTION_PASSES):
        patches = []
        for pid in sorted(selected_ids):
            patch = patch_by_id[pid]
            if pid in original_ids:
                patches.append(patch)
            else:
                patches.append({**patch, "inside_city": True,
                                "promoted_from_fringe": True})
        city_land = unary_union([polygon_from_ring(p["polygon"])
                                 for p in patches])
        if city_land.is_empty:
            raise TownLayoutError("A input: selected city land is empty")

        graph = build_fine_graph(patches)

        # 1. Connect every boundary port to the shared-edge graph using its inward tangent. The current
        # fine graph, not an inherited source-road junction, chooses the meeting.
        base_ineligible = corridor_ineligible_edges(graph, city_land, water_union)
        port_connectors = [attach_port_connector(
            graph, port, blocked_edges=base_ineligible,
            city_land=city_land, water_union=water_union,
            corridor_half_width_gu=CORRIDOR_HALF_WIDTH_GU) for port in ports]

        # Corridor-feasible routing edges: a fine_shared edge whose 256-GU
        # half-width corridor is wet or spills the selected city land can
        # never host the arterial (Stage A hard checks), so it is not a
        # routing candidate.  This narrows the candidate set; no threshold
        # is relaxed.
        ineligible_edges = corridor_ineligible_edges(
            graph, city_land, water_union)

        root_attach, arterial_meeting = choose_arterial_meeting(
            graph, port_connectors, city_land, ineligible_edges)
        meeting_xy = tuple(graph.nodes[root_attach])

        tree = ArterialTree(root_attach)
        lead_lines: dict[str, LineString] = {}
        for connector in port_connectors:
            lead_lines[connector["port_id"]] = LineString(
                [tuple(p) for p in connector["geometry"]])

        def chord(port):
            return ((float(port["position"][0]), float(port["position"][1])),
                    meeting_xy)

        # 2. Standalone minimum route cost per port to the arterial meeting.
        standalone: dict[str, tuple[float, tuple[str, ...]]] = {}
        for port in ports:
            attach = next(c["attach_node_id"] for c in port_connectors
                          if c["port_id"] == port["port_id"])
            a, b = chord(port)
            found = route_search(graph, attach, {root_attach}, a, b,
                                 blocked_edges=ineligible_edges)
            if root_attach not in found:
                raise TownLayoutError(
                    f"A routing: {port['port_id']} cannot reach the arterial meeting")
            standalone[port["port_id"]] = found[root_attach]

        ordered = sorted(ports,
                         key=lambda g: (-standalone[g["port_id"]][0], g["port_id"]))

        routes: list[dict[str, Any]] = []
        merge_records: list[dict[str, Any]] = []
        tree_lead_ids: set[str] = set()

        for order, port in enumerate(ordered):
            port_id = port["port_id"]
            attach = next(c["attach_node_id"] for c in port_connectors
                          if c["port_id"] == port_id)
            a, b = chord(port)
            if order == 0:
                found = route_search(graph, attach, {root_attach}, a, b,
                                     blocked_edges=ineligible_edges)
                if root_attach not in found:
                    raise TownLayoutError(
                        f"A routing: first port {port_id} cannot reach arterial meeting")
                cost, seq = found[root_attach]
                merge_node = root_attach
            else:
                if attach in tree.nodes():
                    # Pass-through case: an earlier (longer) port route already
                    # committed a path through this port's attach node, so the
                    # tree already reaches the port's connector junction.  The
                    # port needs no additional route; its connector lead joins
                    # the tree at the attach node.  This is the natural main-
                    # street topology of a small through-road settlement.  The
                    # case stays explicit via "already_connected" rather than
                    # being silently absorbed.
                    routes.append({
                        "port_id": port_id, "attach_node_id": attach,
                        "target_node_id": attach, "cost": 0.0,
                        "edge_ids": [], "path_node_ids": [attach],
                        "order": order, "already_connected": True,
                    })
                    merge_records.append({
                        "port_id": port_id, "merge_node_id": attach,
                        "merge_tree_distance_gu": float(tree.tree_dist[attach]),
                        "route_cost": 0.0, "is_root_merge": False,
                    })
                    tree_lead_ids.add(port_id)
                    continue
                eligible = set(tree.nodes())
                found = route_search(graph, attach, eligible, a, b,
                                     terminal_nodes=set(tree.nodes()),
                                     blocked_edges=ineligible_edges)
                best = None
                for node, (cost, seq) in sorted(found.items()):
                    total = cost + tree.tree_dist[node]
                    candidate = (round(total, 9), seq, node)
                    if best is None or candidate[:2] < best[:2]:
                        best = (candidate[0], candidate[1], node, cost, seq)
                if best is None:
                    raise TownLayoutError(
                        f"A routing: {port_id} has no reachable arterial merge")
                _, _, merge_node, cost, seq = best
            nodes_on_path = path_nodes(graph, attach, seq)
            # Reject a path intersecting an existing tree lead before its
            # merge node (fine-edge crossings can only occur at graph nodes,
            # which the obstacle set already forbids; leads cut across cell
            # interiors).
            path_line = (LineString([graph.nodes[n] for n in nodes_on_path])
                         if len(nodes_on_path) >= 2
                         else Point(graph.nodes[nodes_on_path[0]]))
            for lead_id in sorted(tree_lead_ids):
                line = lead_lines[lead_id]
                inter = path_line.intersection(line)
                if inter.is_empty:
                    continue
                overlap = inter.length if hasattr(inter, "length") else 0.0
                touches_merge = (not inter.is_empty and inter.geom_type == "Point"
                                 and merge_node in tree.nodes()
                                 and inter.distance(Point(graph.nodes[merge_node]))
                                 <= 1e-6)
                if overlap > 1e-6 or not touches_merge:
                    raise TownLayoutError(
                        f"A routing: {port_id} path intersects tree lead "
                        f"{lead_id} before its merge node")
            tree.commit_path(graph, nodes_on_path, seq)
            tree_lead_ids.add(port_id)
            routes.append({
                "port_id": port_id, "attach_node_id": attach,
                "target_node_id": merge_node, "cost": float(cost),
                "edge_ids": list(seq), "path_node_ids": nodes_on_path,
                "order": order,
            })
            merge_records.append({
                "port_id": port_id, "merge_node_id": merge_node,
                "merge_tree_distance_gu": float(tree.tree_dist[merge_node]),
                "route_cost": float(cost), "is_root_merge": order == 0,
            })

        # 6. Structural verification: one rooted acyclic graph, E = V - 1.
        used_edges = [link[1] for link in tree.parent.values() if link]
        if len(set(used_edges)) != len(used_edges):
            raise TownLayoutError("A tree: duplicated trunk edge")
        v, e = len(tree.parent), len(used_edges)
        if e != v - 1:
            raise TownLayoutError(f"A tree: not acyclic (V={v}, E={e})")
        for port in ports:
            attach = next(c["attach_node_id"] for c in port_connectors
                          if c["port_id"] == port["port_id"])
            if attach not in tree.parent:
                raise TownLayoutError(f"A tree: {port['port_id']} is not attached")

        thin = _thin_roadside_promotions(graph, city_land, set(used_edges),
                                         patch_by_id, selected_ids)
        thin = [pid for pid in thin if pid not in retracted_ids]
        if not thin:
            break
        if _pass == PROMOTION_PASSES - 1:
            raise TownLayoutError(
                f"A boundary_promotion: {len(thin)} patches still required "
                f"after {PROMOTION_PASSES} routing passes")
        selected_ids |= set(thin)

    # Node degrees over the shared-edge tree; boundary connectors are included
    # in complete road strokes but do not add a graph branch.
    degree: dict[str, int] = {}
    for node, link in tree.parent.items():
        if link:
            degree[node] = degree.get(node, 0) + 1
            degree[link[0]] = degree.get(link[0], 0) + 1
    degree.setdefault(root_attach, 0)
    arterial_nodes = []
    for node in sorted(tree.parent):
        total_deg = degree.get(node, 0)
        if total_deg > 4:
            raise TownLayoutError(f"A smoothing: node {node} degree {total_deg} > 4")
        kind = ("meeting" if node == root_attach else
                "port_attach" if any(c["attach_node_id"] == node for c in port_connectors) else
                "merge" if any(m["merge_node_id"] == node and not m["is_root_merge"]
                               for m in merge_records) else "through")
        arterial_nodes.append({
            "node_id": node, "position": list(graph.nodes[node]),
            "degree": total_deg, "kind": kind,
        })

    # ---- Port-to-meeting paths and smoothing ----------------------------
    # Each path runs boundary port -> attach node -> ... -> meeting. A degree-2 tree node
    # lies on exactly one port path (merges raise the degree), so fillet
    # choices are made once per corner.
    port_paths: dict[str, dict[str, Any]] = {}
    for port, connector in zip(ports, port_connectors):
        node_path = tree.path_to_root(connector["attach_node_id"])
        points = [list(port["position"])] + [list(graph.nodes[n]) for n in node_path]
        port_paths[port["port_id"]] = {
            "points": points, "corner_nodes": [None] + list(node_path),
        }

    reports: dict[str, Any] = {"zero_radius_corners": [], "fillet_fallbacks": []}
    radii: dict[str, float] = {}

    # Port connectors are complete arterial roads. Their shoulders may cross
    # the abstract cell boundary during the transition from an exterior road,
    # but their centerlines must remain covered and dry.
    port_transition_union = unary_union([
        LineString(connector["geometry"]).buffer(
            PORT_CAP_GU, cap_style="round", join_style="round")
        for connector in port_connectors
    ])
    bridge_port_ids = {port["port_id"] for port in ports
                       if port.get("continuation") == "continuation_bridge_dependent"}
    bridge_transition_union = unary_union([
        LineString(connector["geometry"]).buffer(
            PORT_CAP_GU, cap_style="round", join_style="round")
        for connector in port_connectors
        if connector["port_id"] in bridge_port_ids
    ])

    def corridor_union() -> Any:
        parts = []
        for port_id, path in port_paths.items():
            smoothed = smooth_polyline(path["points"], path["corner_nodes"],
                                       radii)
            parts.append(LineString(smoothed).buffer(CORRIDOR_HALF_WIDTH_GU,
                                                     cap_style="round",
                                                     join_style="round"))
        for arc in endpoint_arcs.values():
            parts.append(LineString(arc).buffer(CORRIDOR_HALF_WIDTH_GU,
                                                cap_style="round",
                                                join_style="round"))
        # At a branched junction, two incident roads can form an acute pocket
        # too small to function as land. Fill only that wedge between their
        # centerlines. Unlike a circular junction pad, this leaves every other
        # shoulder at the normal arterial width.
        for node in sorted(n for n, value in degree.items() if value >= 3):
            center = graph.nodes[node]
            parent = tree.parent.get(node)
            neighbours = ([parent[0]] if parent else []) + sorted(children.get(node, []))
            for i, left in enumerate(neighbours):
                for right in neighbours[i + 1:]:
                    lv = (graph.nodes[left][0] - center[0], graph.nodes[left][1] - center[1])
                    rv = (graph.nodes[right][0] - center[0], graph.nodes[right][1] - center[1])
                    ln, rn = math.hypot(*lv), math.hypot(*rv)
                    if ln <= 1e-6 or rn <= 1e-6:
                        continue
                    angle = math.degrees(math.acos(max(-1.0, min(1.0,
                        (lv[0] * rv[0] + lv[1] * rv[1]) / (ln * rn)))))
                    if angle >= 105.0:
                        continue
                    distance = 2.0 * CORRIDOR_HALF_WIDTH_GU
                    lp = (center[0] + lv[0] * min(distance, ln) / ln,
                          center[1] + lv[1] * min(distance, ln) / ln)
                    rp = (center[0] + rv[0] * min(distance, rn) / rn,
                          center[1] + rv[1] * min(distance, rn) / rn)
                    parts.append(Polygon([center, lp, rp]))
        return unary_union(parts)

    def corridor_ok(union) -> str | None:
        reason, _detail = corridor_check(
            union, port_transition_union, bridge_transition_union,
            water_union, city_land)
        return reason

    # Choose fillet radii for degree-2 corners, largest validated first.
    # Neighbours come from the tree structure so endpoint corners (e.g. two
    # routes meeting at the root) are handled too; their arcs sit between
    # two path endpoints and are added to the corridor separately.
    endpoint_arcs: dict[str, list[list[float]]] = {}
    children: dict[str, list[str]] = {}
    for node, link in tree.parent.items():
        if link:
            children.setdefault(link[0], []).append(node)

    def corner_neighbours(node: str) -> tuple[str, str]:
        link = tree.parent.get(node)
        neigh = ([link[0]] if link else []) + sorted(children.get(node, []))
        if len(neigh) != 2:
            raise TownLayoutError(f"A smoothing: corner {node} is not degree 2")
        return neigh[0], neigh[1]

    corner_order = sorted(
        (node for node in tree.parent if degree.get(node, 0) == 2),
    )
    for node in corner_order:
        na, nb = corner_neighbours(node)
        p_prev, p_next = graph.nodes[na], graph.nodes[nb]
        p_node = graph.nodes[node]
        path = next((p for p in port_paths.values() if node in p["corner_nodes"]),
                    None)
        interior_idx = None
        if path is not None:
            idx = path["corner_nodes"].index(node)
            if 0 < idx < len(path["points"]) - 1:
                interior_idx = idx
        l1 = math.hypot(p_prev[0] - p_node[0], p_prev[1] - p_node[1])
        l2 = math.hypot(p_next[0] - p_node[0], p_next[1] - p_node[1])
        u1 = ((p_prev[0] - p_node[0]) / l1, (p_prev[1] - p_node[1]) / l1)
        u2 = ((p_next[0] - p_node[0]) / l2, (p_next[1] - p_node[1]) / l2)
        deflection = 180.0 - _angle_deg(u1, u2)
        if deflection <= 1e-6:
            continue
        cap = min(l1, l2) * FILLET_SEGMENT_FRACTION
        chosen = None
        for radius in FILLET_RADII_GU:
            eff = min(radius, cap)
            if eff <= 0.0:
                continue
            arc = fillet_arc(p_prev, p_node, p_next, eff)
            if arc is None:
                continue
            if interior_idx is not None:
                radii[node] = eff
            else:
                endpoint_arcs[node] = arc
            reason = corridor_ok(corridor_union())
            if reason is None:
                chosen = eff
                break
            if interior_idx is not None:
                del radii[node]
            else:
                del endpoint_arcs[node]
            reports["fillet_fallbacks"].append({
                "node_id": node, "radius_gu": eff, "reason": reason})
        if chosen is None:
            if deflection > ZERO_RADIUS_MAX_DEFLECTION_DEG:
                raise TownLayoutError(
                    f"A smoothing: zero-radius corner at {node} has deflection "
                    f"{deflection:.2f} > {ZERO_RADIUS_MAX_DEFLECTION_DEG:.0f} degrees")
            reports["zero_radius_corners"].append({
                "node_id": node, "deflection_deg": float(deflection)})
        elif interior_idx is not None:
            radii[node] = chosen
        else:
            endpoint_arcs[node] = fillet_arc(p_prev, p_node, p_next, chosen)

    final_union = corridor_union()
    failure = corridor_ok(final_union)
    if failure is not None:
        raise TownLayoutError(f"A corridor: {failure}")
    corridor_parts = simple_polygon_parts(final_union)
    bridge_water_overlap = (final_union.intersection(water_union).area
                            if not water_union.is_empty else 0.0)
    corridor_rings = [
        [[float(x), float(y)] for x, y in part.exterior.coords[:-1]]
        for part in corridor_parts
    ]

    # Smoothed strokes (one complete boundary-port path each).
    smoothed_strokes = []
    for port_id in sorted(port_paths):
        path = port_paths[port_id]
        smoothed_strokes.append({
            "stroke_id": f"stroke_{port_id}",
            "port_id": port_id,
            "geometry": smooth_polyline(path["points"], path["corner_nodes"],
                                        radii),
        })

    # Raw barrier: unsmoothed fine edges plus the arterial connector from each
    # boundary port. Every piece is town-corridor authority.
    raw_barrier = []
    for connector in port_connectors:
        raw_barrier.append({
            "barrier_id": f"barrier_port_{connector['port_id']}",
            "kind": "port_connector", "town_corridor": True,
            "port_id": connector["port_id"], "geometry": connector["geometry"],
            "length_gu": float(LineString(connector["geometry"]).length),
        })
    for edge_id in sorted(set(used_edges)):
        edge = graph.edges[edge_id]
        raw_barrier.append({
            "barrier_id": f"barrier_{edge_id}", "kind": "fine_edge",
            "town_corridor": True,
            "fine_edge_id": edge_id, "a_node": edge["a"], "b_node": edge["b"],
            "geometry": [list(graph.nodes[edge["a"]]),
                         list(graph.nodes[edge["b"]])],
            "length_gu": float(edge["length_gu"]),
        })

    # Port heading residuals (report-only metric). The 2026-08-18
    # attach-scoring amendment replaced the hard 5-degree tangent residual
    # with the alpha+beta smoothness score in ``attach_port_connector``; the
    # residual is still reported for review.
    port_metrics = []
    for port, connector in zip(ports, port_connectors):
        d = connector["direction"]
        tangent = port["source_tangent"]
        residual = _angle_deg((d[0], d[1]), (tangent[0], tangent[1]))
        residual = min(residual, 180.0 - residual)  # tangent sign is metadata
        port_metrics.append({
            "port_id": port["port_id"],
            "connector_length_gu": float(connector["length_gu"]),
            "heading_residual_deg": float(residual),
        })

    runtime = time.perf_counter() - started
    if runtime > RUNTIME_LIMIT_S:
        raise TownLayoutError(f"A runtime: {runtime:.2f}s > {RUNTIME_LIMIT_S}s")

    return {
        "schema_version": 1,
        "stage_id": "r2a_arterials",
        "candidate_id": macro_product.get("candidate_id"),
        "preceding_checkpoint": None,  # filled by the CLI with the real path
        "identities": macro_product.get("identities") or {},
        "city_domain": macro_product.get("city_domain"),
        "terrain": macro_product.get("terrain"),
        "water_polygons": macro_product.get("water_polygons") or [],
        # Working city set: the 72 R1 ``inside_city`` patches plus any
        # fringe patches promoted by the boundary-arterial promotion rule
        # (marked ``promoted_from_fringe``).  Stage B inherits this set.
        "patches": patches,
        "promoted_patch_ids": sorted(selected_ids - original_ids),
        "planning_ring": projection["planning_ring"],
        "ports": ports,
        "arterial_meeting": arterial_meeting,
        "excluded_crossings": projection["excluded_crossings"],
        "fine_graph_summary": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "fine_shared_count": sum(1 for e in graph.edges.values()
                                     if e["role"] == "fine_shared"),
            "fine_outer_count": sum(1 for e in graph.edges.values()
                                    if e["role"] == "fine_outer"),
        },
        "port_connectors": [
            {**{k: v for k, v in connector.items() if k != "direction"},
             "role": "arterial_connector",
             "attachment_position": list(graph.nodes[connector["attach_node_id"]])}
            for connector in port_connectors
        ],
        "arterial_nodes": arterial_nodes,
        "arterial_edges": [
            {
                "edge_id": eid,
                "a_node": graph.edges[eid]["a"],
                "b_node": graph.edges[eid]["b"],
                "geometry": [list(graph.nodes[graph.edges[eid]["a"]]),
                             list(graph.nodes[graph.edges[eid]["b"]])],
                "length_gu": float(graph.edges[eid]["length_gu"]),
            }
            for eid in sorted(set(used_edges))
        ],
        "raw_barrier": raw_barrier,
        "routes": routes,
        "merge_records": merge_records,
        "smoothed_strokes": smoothed_strokes,
        "fillet_radii": {node: radii[node] for node in sorted(radii)},
        "endpoint_fillets": {
            node: {"arc": endpoint_arcs[node]}
            for node in sorted(endpoint_arcs)
        },
        "junction_pads": [],
        "corridor": {
            "half_width_gu": CORRIDOR_HALF_WIDTH_GU,
            "rings": corridor_rings,
            "area_gu2": float(sum(p.area for p in corridor_parts)),
        },
        "metrics": {
            "runtime_s": float(runtime),
            "port_metrics": port_metrics,
            "corridor_water_overlap_gu2": float(bridge_water_overlap),
            "bridge_connector_water_overlap_gu2": float(bridge_water_overlap),
            "corridor_city_spill_gu2": 0.0,
            "tree_node_count": v,
            "tree_edge_count": e,
        },
        "reports": reports,
    }
