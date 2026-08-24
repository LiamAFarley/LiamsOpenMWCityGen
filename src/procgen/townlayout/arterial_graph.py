"""Stage A arterial graph: sanitized R2 projection and planar fine graph.

Purpose
-------
Rebuild the planar fine-edge overlay from the exact polygons of the
``inside_city`` patches of the frozen R1 macro checkpoint, and attach the
boundary-port connectors defined by the dynamic arterial-confluence plan. Only the strict permitted
projection of the R2 ports checkpoint is read: ``planning_ring``, per-port
``port_id``/``crossing_id``/``position``/``ring_arc_gu``/``source_tangent``/
``continuation``, and the excluded-crossing disposition. Ingress polylines,
aligned roads, approaches, historical source junctions, and all source-chain
vertices are forbidden inputs and are never read here.

Inputs
------
Loaded checkpoint dicts (R1 macro, R2 ports).  Coordinate identity is
centi-GU (``round(x*100), round(y*100)``), matching the R1 topology
serializer; geometry is never snapped or moved by more than 1 GU.

Outputs
-------
``FineGraph`` (nodes/edges tagged ``fine_shared`` | ``fine_outer`` |
``arterial_barrier``), port-connector records, and the sanitized port/exclusion
projection.  All failures raise ``TownLayoutError``.

Pipeline position
-----------------
Phase 21 Stage A (r2a_arterials); consumed by ``arterial_routes.py`` and the
Stage B/C modules.  No rendering here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import LineString, Point

from .geometry import normalize_ring
from .validate import TownLayoutError

CENTI = 100.0
PORT_CONNECTOR_MIN_GU = 256.0
PORT_CONNECTOR_MAX_GU = 6144.0


def node_key(x: float, y: float) -> tuple[int, int]:
    """Centi-GU identity key, matching the R1 topology serializer."""
    return (int(round(x * CENTI)), int(round(y * CENTI)))


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------

def sanitize_ports(ports_product: dict[str, Any]) -> dict[str, Any]:
    """Extract the strict permitted projection of the R2 checkpoint.

    Reads only ``planning_ring``, the permitted per-port fields, and the excluded
    crossing disposition. Port and exclusion
    counts come from the accepted R2 organic ring rather than the old fixed
    four-port circular-domain artifact.
    """
    ring = (ports_product.get("planning_ring") or {}).get("ring") or []
    if len(ring) < 3:
        raise TownLayoutError("A input_projection: planning_ring missing")

    raw_ports = ports_product.get("ports") or []
    # Rimgrad and other hamlets may have 0-1 roads; allow <2 ports and
    # produce a wall-only/isolated street hamlet.
    if len(raw_ports) < 2:
        # Preserve city_region_patch_ids for downstream block/street stages,
        # but record empty port projection.
        city_region_patch_ids = [str(pid) for pid in
                                 ports_product.get("city_region_patch_ids") or []]
        if not city_region_patch_ids:
            raise TownLayoutError("A input_projection: city_region_patch_ids missing")
        return {
            "planning_ring": ring,
            "ports": [],
            "excluded_crossings": list(ports_product.get("excluded_crossings") or []),
            "city_region_patch_ids": city_region_patch_ids,
        }
    city_region_patch_ids = [str(pid) for pid in
                             ports_product.get("city_region_patch_ids") or []]
    if not city_region_patch_ids:
        raise TownLayoutError("A input_projection: city_region_patch_ids missing")
    ports = []
    for port in raw_ports:
        tangent = port.get("source_tangent")
        if (not isinstance(tangent, list) or len(tangent) != 2
                or (float(tangent[0]) == 0.0 and float(tangent[1]) == 0.0)):
            raise TownLayoutError(
                f"A input_projection: {port.get('port_id')} bad source_tangent")
        ports.append({
            "port_id": str(port["port_id"]),
            "crossing_id": str(port["crossing_id"]),
            "position": [float(port["position"][0]), float(port["position"][1])],
            "ring_arc_gu": float(port["ring_arc_gu"]),
            "source_tangent": [float(tangent[0]), float(tangent[1])],
            "continuation": str(port.get("continuation")),
        })

    excluded = []
    for crossing in ports_product.get("source_crossings") or []:
        if crossing.get("status") == "excluded":
            excluded.append({
                "crossing_id": str(crossing.get("crossing_id")),
                "reason": str(crossing.get("reason")),
            })
    excluded = [item for item in excluded if item["reason"] == "no_ring_crossing"]

    ports.sort(key=lambda p: p["port_id"])
    return {
        "planning_ring": ports_product["planning_ring"],
        "city_region_patch_ids": sorted(city_region_patch_ids),
        "ports": ports,
        "excluded_crossings": excluded,
    }


# ---------------------------------------------------------------------------
# Fine graph
# ---------------------------------------------------------------------------

@dataclass
class FineGraph:
    """Planar edge overlay over the selected city patches.

    ``nodes``: node_id -> (x, y).  ``edges``: edge_id -> record with
    ``a``/``b`` node ids, ``length_gu``, ``patches`` (sorted member patch
    ids), and ``role`` (``fine_shared`` | ``fine_outer`` |
    ``arterial_barrier``).  ``key_to_node`` maps centi-GU keys to node ids.
    """
    nodes: dict[str, tuple[float, float]] = field(default_factory=dict)
    key_to_node: dict[tuple[int, int], str] = field(default_factory=dict)
    edges: dict[str, dict[str, Any]] = field(default_factory=dict)
    adjacency: dict[str, list[str]] = field(default_factory=dict)

    def add_node(self, node_id: str, x: float, y: float) -> None:
        key = node_key(x, y)
        existing = self.key_to_node.get(key)
        if existing is not None and existing != node_id:
            raise TownLayoutError(
                f"A fine_graph: coincident nodes {existing} / {node_id}")
        self.nodes[node_id] = (float(x), float(y))
        self.key_to_node[key] = node_id
        self.adjacency.setdefault(node_id, [])

    def add_edge(self, edge_id: str, a: str, b: str,
                 patches: list[str], role: str) -> None:
        pa, pb = self.nodes[a], self.nodes[b]
        length = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        if length <= 0.0:
            raise TownLayoutError(f"A fine_graph: zero-length edge {edge_id}")
        self.edges[edge_id] = {
            "edge_id": edge_id, "a": a, "b": b,
            "length_gu": length, "patches": sorted(patches), "role": role,
        }
        self.adjacency[a].append(edge_id)
        self.adjacency[b].append(edge_id)

    def other(self, edge_id: str, node_id: str) -> str:
        edge = self.edges[edge_id]
        return edge["b"] if edge["a"] == node_id else edge["a"]

    def shared_edges(self) -> list[str]:
        return sorted(eid for eid, e in self.edges.items()
                      if e["role"] == "fine_shared")


def _point_on_segment_interior(px: float, py: float,
                               a: tuple[float, float],
                               b: tuple[float, float]) -> bool:
    """True when (px, py) is collinear with and strictly between a and b."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = px - a[0], py - a[1]
    cross = abx * apy - aby * apx
    if abs(cross) > 0.01 * max(1.0, math.hypot(abx, aby)):
        return False
    dot = apx * abx + apy * aby
    return 1e-4 < dot < abx * abx + aby * aby - 1e-4


def build_fine_graph(patches: list[dict[str, Any]]) -> FineGraph:
    """Rebuild the planar overlay from exact city-patch polygons.

    Every boundary piece between two adjacent exact shared vertices is one
    edge; a piece shared by exactly two selected city patches is
    ``fine_shared``, a piece on exactly one is ``fine_outer``.  Patch
    boundaries are split at any exact shared vertex lying in a segment
    interior (Voronoi output is already noded; this is the contract guard).
    """
    city = [p for p in patches if p.get("inside_city")]
    if not city:
        raise TownLayoutError("A fine_graph: no inside_city patches")
    rings: dict[str, list[list[float]]] = {}
    all_keys: set[tuple[int, int]] = set()
    for patch in city:
        ring = normalize_ring(patch.get("polygon") or [])["ring"]
        rings[str(patch["patch_id"])] = ring
        for pt in ring:
            all_keys.add(node_key(pt[0], pt[1]))

    graph = FineGraph()
    for key in sorted(all_keys):
        graph.add_node(f"fn_{len(graph.nodes):05d}",
                       key[0] / CENTI, key[1] / CENTI)

    # Collect maximal segments per patch, splitting at interior shared
    # vertices, and count city-patch membership per undirected key pair.
    membership: dict[tuple[tuple[int, int], tuple[int, int]], set[str]] = {}
    for patch_id in sorted(rings):
        ring = rings[patch_id]
        coords = [(float(pt[0]), float(pt[1])) for pt in ring]
        count = len(coords)
        for idx in range(count):
            a = coords[idx]
            b = coords[(idx + 1) % count]
            # Split at exact shared vertices lying in this segment's interior.
            cuts = [a, b]
            for key in all_keys:
                pt = (key[0] / CENTI, key[1] / CENTI)
                if _point_on_segment_interior(pt[0], pt[1], a, b):
                    cuts.append(pt)
            if len(cuts) > 2:
                interior = cuts[1:-1]
                interior.sort(key=lambda p: (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2)
                cuts = [a] + interior + [b]
            for i in range(len(cuts) - 1):
                ka, kb = node_key(*cuts[i]), node_key(*cuts[i + 1])
                if ka == kb:
                    continue
                pair = (ka, kb) if ka < kb else (kb, ka)
                membership.setdefault(pair, set()).add(patch_id)

    for pair in sorted(membership):
        members = membership[pair]
        if len(members) > 2:
            raise TownLayoutError(
                f"A fine_graph: non-manifold segment {pair} ({len(members)} patches)")
        a_id = graph.key_to_node[pair[0]]
        b_id = graph.key_to_node[pair[1]]
        role = "fine_shared" if len(members) == 2 else "fine_outer"
        graph.add_edge(f"fe_{len(graph.edges):05d}", a_id, b_id,
                       sorted(members), role)
    return graph


def split_edge(graph: FineGraph, edge_id: str, x: float, y: float,
               node_id: str) -> tuple[str, list[str]]:
    """Split ``edge_id`` at the exact point.

    Returns ``(node_id, new_edge_ids)``.  When the point coincides with an
    existing node (centi-GU identity), that node is reused and the edge is
    not split — this is the coincident split-node contraction required by
    the plan; no node may be moved to make it happen.
    """
    edge = graph.edges.get(edge_id)
    if edge is None:
        raise TownLayoutError(f"A fine_graph: split of unknown edge {edge_id}")
    a, b = edge["a"], edge["b"]
    pa, pb = graph.nodes[a], graph.nodes[b]
    length = edge["length_gu"]
    dist = math.hypot(x - pa[0], y - pa[1]) + math.hypot(x - pb[0], y - pb[1])
    if abs(dist - length) > 1.0:
        raise TownLayoutError(
            f"A fine_graph: split point of {edge_id} is not on the edge")
    existing = graph.key_to_node.get(node_key(x, y))
    if existing is not None:
        return existing, []
    graph.add_node(node_id, x, y)
    for node in (a, b):
        graph.adjacency[node] = [e for e in graph.adjacency[node] if e != edge_id]
    del graph.edges[edge_id]
    id_a, id_b = f"{edge_id}#s0", f"{edge_id}#s1"
    graph.add_edge(id_a, a, node_id, edge["patches"], edge["role"])
    graph.add_edge(id_b, node_id, b, edge["patches"], edge["role"])
    return node_id, [id_a, id_b]


# ---------------------------------------------------------------------------
# Terminal attachments
# ---------------------------------------------------------------------------

def _turn_deg(u: tuple[float, float], v: tuple[float, float]) -> float:
    dot = u[0] * v[0] + u[1] * v[1]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def attach_port_connector(graph: FineGraph, port: dict[str, Any],
                          blocked_edges: set[str] | None = None,
                          city_land=None, water_union=None,
                          corridor_half_width_gu: float = 256.0) -> dict[str, Any]:
    """Choose the smoothest arterial connector from a boundary port.

    Candidates are every existing node incident to a ``fine_shared`` edge
    and the perpendicular foot on every ``fine_shared`` edge, between 256
    and 6144 GU from the port. Each candidate is scored by ``alpha + beta``
    where ``alpha`` is the angle between the lead direction and the
    inward port tangent and ``beta`` is the minimum turn angle
    where the lead would join the street graph at the candidate (both in
    degrees); the lowest score wins, ties resolve by
    ``(score, candidate id, x, y)``.  A winning edge-interior foot is split
    at the exact point (coincident split-node contraction applies); every
    crossed fine edge is recorded (noded) and the whole lead becomes an
    arterial barrier.  No qualifying candidate is a hard failure.
    """
    gx, gy = float(port["position"][0]), float(port["position"][1])
    tx, ty = float(port["source_tangent"][0]), float(port["source_tangent"][1])
    norm = math.hypot(tx, ty)
    tx, ty = tx / norm, ty / norm
    candidates = []  # score, id, x, y, host, geometry, start direction, length
    blocked = blocked_edges or set()

    def consider(cid: str, px: float, py: float, host: str | None,
                 continuations: list[tuple[float, float]]) -> None:
        geometry = [(gx, gy), (px, py)]
        connector = LineString(geometry)
        dist = connector.length
        if not (PORT_CONNECTOR_MIN_GU <= dist <= PORT_CONNECTOR_MAX_GU):
            return
        first = geometry[1]
        first_len = math.hypot(first[0] - gx, first[1] - gy)
        lead_dir = ((first[0] - gx) / first_len, (first[1] - gy) / first_len)
        join_dir = lead_dir
        if city_land is not None:
            if connector.difference(city_land).length > 1.0:
                return
            if (water_union is not None and not water_union.is_empty
                    and port.get("continuation") != "continuation_bridge_dependent"
                    and connector.buffer(
                        corridor_half_width_gu, cap_style="round",
                        join_style="round").intersection(water_union).area > 1.0):
                return
        alpha = _turn_deg(lead_dir, (tx, ty))
        beta = min(_turn_deg(join_dir, c) for c in continuations)
        candidates.append((round(alpha + beta, 9), cid,
                           round(px, 6), round(py, 6), host,
                           [[float(x), float(y)] for x, y in geometry],
                           lead_dir, float(dist)))

    shared_nodes: set[str] = set()
    for edge_id in graph.shared_edges():
        if edge_id in blocked:
            continue
        edge = graph.edges[edge_id]
        a, b = edge["a"], edge["b"]
        shared_nodes.update((a, b))
        pa, pb = graph.nodes[a], graph.nodes[b]
        abx, aby = pb[0] - pa[0], pb[1] - pa[1]
        u = ((gx - pa[0]) * abx + (gy - pa[1]) * aby) / (abx * abx + aby * aby)
        if 1e-6 < u < 1.0 - 1e-6:
            ea = math.hypot(abx, aby)
            consider(edge_id, pa[0] + u * abx, pa[1] + u * aby, edge_id,
                     [(-abx / ea, -aby / ea), (abx / ea, aby / ea)])
    for node_id in sorted(shared_nodes):
        px, py = graph.nodes[node_id]
        continuations = []
        for edge_id in graph.adjacency.get(node_id, []):
            edge = graph.edges[edge_id]
            if edge["role"] != "fine_shared" or edge_id in blocked:
                continue
            ox, oy = graph.nodes[graph.other(edge_id, node_id)]
            norm2 = math.hypot(ox - px, oy - py)
            continuations.append(((ox - px) / norm2, (oy - py) / norm2))
        if continuations:
            consider(node_id, px, py, None, continuations)

    if not candidates:
        raise TownLayoutError(
            f"A terminal_attachment: no fine_shared attach candidate within "
            f"[256, 6144] GU of {port['port_id']}")
    _score, _cid, x, y, host, geometry, lead_dir, connector_length = min(candidates)
    if host is not None:
        node_id, split_ids = split_edge(graph, host, x, y,
                                        f"an_{port['port_id']}")
        split_edge_id: str | None = host
    else:
        node_id, split_ids, split_edge_id = _cid, [], None

    # Record every fine edge crossed by the lead, nearest first.
    lead_seg = LineString(geometry)
    origin = Point(gx, gy)
    hits = []
    for edge_id in sorted(graph.edges):
        edge = graph.edges[edge_id]
        line = LineString([graph.nodes[edge["a"]], graph.nodes[edge["b"]]])
        inter = lead_seg.intersection(line)
        if not inter.is_empty:
            hits.append((round(inter.distance(origin), 6), edge_id))
    hits.sort()
    return {
        "port_id": port["port_id"],
        "port_node_id": f"port_{port['port_id']}",
        "attach_node_id": node_id,
        "split_edge_id": split_edge_id,
        "split_piece_ids": split_ids,
        "crossed_fine_edge_ids": [eid for _t, eid in hits],
        "geometry": geometry,
        "direction": [float(lead_dir[0]), float(lead_dir[1])],
        "length_gu": connector_length,
    }
