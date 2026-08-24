"""Organic macro patch generation for V2 townlayout (Phase 5A).

Purpose
-------
Seed a development/guard point cloud, build a SciPy Voronoi diagram, clip
cells to the rewrite domain, Lloyd-relax developed seeds, merge short
junctions, and emit the canonical patch / boundary / node topology.

Inputs
------
``SiteContext``, rewrite-domain ring, TownBrief, optional approach
crossings, ``master_seed`` / ``candidate_id``.

Outputs
-------
A ``MacroLayoutCandidate`` dict (patches, nodes, boundary_edges, reports)
plus an optional diagnostic PNG.

Pipeline position
-----------------
V2 townlayout Phase 5A organic patches; no walls, parcels, or VTEX.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from procgen.cityplan import ring_area

from .candidate import require_macro_layout
from .constants import ARTERIAL_CLEAR_WIDTH_GU, JUNCTION_MERGE_GU, MIN_PATCH_AREA_GU2, VERTEX_EPS_GU
from .geometry import normalize_ring, polygon_from_ring, simple_polygon_parts
from .rng import stage_rng
from .site_context import SiteContext, _plan_to_px, diagnostic_view
from .validate import TownLayoutError

LLOYD_PASSES = 2
LLOYD_EXCLUDED_FRACTION = 0.30
MAX_SEED_TRIES = 50
N_DEVELOPED_MIN = 12
N_DEVELOPED_MAX = 48
# Single Stage 04 density revision: usable land is divided into 3.5 raw
# p50-stamp footprints.  This replaces preferred/3 because Falkreath's
# preferred/3 floor requested only 12 seeds over ~26 target patch footprints.
RAW_PATCH_TARGET_FOOTPRINTS = 3.5
EXPECTED_ROAD_HALF_WIDTH_GU = ARTERIAL_CLEAR_WIDTH_GU / 2.0
LAND_COVERAGE_SLOP_GU2 = 1e-5


def retain_land_parts(clipped) -> list[Polygon]:
    """Keep every simple land part above the tessellation area floor.

    Suitability/slope is deliberately absent: a representative sample is not
    authoritative geometry and must not create a land hole.
    """
    return [part for part in simple_polygon_parts(clipped)
            if part.area >= MIN_PATCH_AREA_GU2]


def _coalesce_subfloor_cells(rings, patch_ids, kept_from_seed):
    """Absorb sub-floor Voronoi slivers into a shared-boundary land patch."""
    polys = [polygon_from_ring(r) for r in rings]
    alive = list(range(len(polys)))
    for source in sorted(alive, key=lambda i: (polys[i].area, patch_ids[i])):
        if source not in alive or polys[source].area >= MIN_PATCH_AREA_GU2:
            continue
        candidates = [(polys[source].boundary.intersection(polys[target].boundary).length,
                       -polys[target].area, patch_ids[target], target)
                      for target in alive if target != source]
        candidates.sort(reverse=True)
        for _shared, _area, _pid, target in candidates:
            merged = polys[target].union(polys[source])
            if merged.geom_type == "Polygon" and not merged.interiors:
                polys[target] = merged
                alive.remove(source)
                break
    if len(alive) == len(polys):
        return rings, patch_ids, kept_from_seed
    remap = {old: new for new, old in enumerate(alive)}
    new_kept = {seed: [remap[i] for i in indices if i in remap]
                for seed, indices in kept_from_seed.items()}
    return ([_open_ring_from_poly(polys[i]) for i in alive],
            [patch_ids[i] for i in alive], new_kept)


def _n_developed(ctx: SiteContext, domain: Polygon, town_brief: dict) -> int:
    """Return the bounded R1 seed count derived from the brief target.

    Falkreath R1 trials ``preferred`` (amended 2026-08-15 from
    ``preferred / 5``): the 16-seed artifact produced ~5,900 GU walled
    superblocks that failed visual review.  Fine cells of roughly 1-2 stamps
    let R3 produce varied blocks by selective seam promotion (see
    `.opencode/runs/townlayout-phase21-recovery/2026-08-15_r2_visual_correction_design.md`).
    The resulting capacity and patch scale are reported by the checkpoint and
    remain an empirical gate.
    """
    try:
        preferred = float(town_brief["target_buildings"]["preferred"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TownLayoutError("missing_key: target_buildings.preferred") from exc
    return max(48, min(96, int(round(preferred))))


def _open_ring_from_poly(poly: Polygon) -> list[list[float]]:
    return normalize_ring([[c[0], c[1]] for c in poly.exterior.coords])["ring"]


def _finite_voronoi_regions(vor: Voronoi, radius: float):
    """Reconstruct finite 2D Voronoi regions (scipy cookbook, no silent repair)."""
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    all_ridges: dict[int, list] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(int(p1), []).append((int(p2), int(v1), int(v2)))
        all_ridges.setdefault(int(p2), []).append((int(p1), int(v1), int(v2)))
    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[int(region_index)]
        if all(v >= 0 for v in vertices) and len(vertices) >= 3:
            new_regions.append(list(vertices))
            continue
        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            norm = float(np.linalg.norm(t))
            if norm <= VERTEX_EPS_GU:
                continue
            t = t / norm
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        if len(new_region) < 3:
            new_regions.append([])
            continue
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_regions.append([new_region[i] for i in np.argsort(angles)])
    return new_regions, np.asarray(new_vertices)


def _cells_for_points(
    points: np.ndarray, clip: Optional[Polygon],
) -> tuple[list[Optional[Polygon]], list[tuple[int, int]]]:
    if len(points) < 4:
        raise TownLayoutError("invalid_polygon: need at least 4 Voronoi seeds")
    try:
        vor = Voronoi(points)
    except Exception as exc:
        raise TownLayoutError(f"invalid_polygon: Voronoi failed ({exc})") from exc
    bounds = clip.bounds if clip is not None else (
        float(points[:, 0].min()), float(points[:, 1].min()),
        float(points[:, 0].max()), float(points[:, 1].max()))
    radius = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 4.0
    regions, vertices = _finite_voronoi_regions(vor, radius)
    cells: list[Optional[Polygon]] = []
    for region in regions:
        if len(region) < 3:
            cells.append(None)
            continue
        coords = vertices[region]
        poly = Polygon(coords)
        if (not poly.is_valid) or poly.area <= 0:
            cells.append(None)
            continue
        if clip is not None:
            poly = poly.intersection(clip)
        if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
            cells.append(None)
            continue
        cells.append(poly)
    ridges = [(int(a), int(b)) for a, b in vor.ridge_points]
    return cells, ridges


def _accept_seed(ctx: SiteContext, x: float, y: float, rng) -> bool:
    sample = ctx.sample(x, y)
    if not sample.get("buildable", False):
        return False
    p = max(0.0, min(1.0, float(sample["suitability"]) ** 2))
    if p >= 1.0:
        return True
    return rng.random() <= p


def _place_point(rng, bbox, domain: Polygon, ctx: SiteContext,
                 cx: float, cy: float, radius: float, inner: bool,
    attractor: Optional[tuple[float, float]] = None,
                 ) -> Optional[tuple[float, float]]:
    minx, miny, maxx, maxy = bbox
    last = None
    for _ in range(MAX_SEED_TRIES):
        pick = rng.random()
        if inner and pick < 0.35:
            ang = rng.random() * 2.0 * math.pi
            rad = (rng.random() ** 0.7) * radius
            x = cx + math.cos(ang) * rad
            y = cy + math.sin(ang) * rad
        else:
            x = minx + rng.random() * (maxx - minx)
            y = miny + rng.random() * (maxy - miny)
        last = (x, y)
        pt = Point(x, y)
        if inner and not domain.contains(pt):
            continue
        if (not inner) and domain.contains(pt):
            continue
        if _accept_seed(ctx, x, y, rng):
            return (x, y)
    return None


def _far_enough(pt, existing, min_d: float) -> bool:
    return all(math.hypot(pt[0] - q[0], pt[1] - q[1]) >= min_d for q in existing)


def _make_seeds(ctx: SiteContext, domain: Polygon, inner: Polygon,
                rng, n_developed: int,
    attractor: Optional[tuple[float, float]] = None,
                ) -> tuple[list[tuple[float, float]], int]:
    cx, cy = ctx.candidate_centers[0]
    radius = math.sqrt(max(inner.area, 1.0) / math.pi)
    spacing = math.sqrt(max(inner.area / max(n_developed, 1), MIN_PATCH_AREA_GU2))
    min_d = 0.35 * spacing
    developed: list[tuple[float, float]] = []
    bbox_in = inner.bounds
    guard_band = inner.buffer(spacing * 1.25)
    bbox_g = guard_band.bounds
    attempts = 0
    while len(developed) < n_developed and attempts < n_developed * 500:
        attempts += 1
        pt = _place_point(rng, bbox_in, inner, ctx, cx, cy, radius, True)
        if pt is None:
            break
        if not inner.contains(Point(*pt)):
            continue
        if not _far_enough(pt, developed, min_d):
            continue
        developed.append(pt)
    if len(developed) < n_developed:
        raise TownLayoutError(
            f"insufficient_compact_capacity: requested_developed_patches={n_developed} "
            f"admitted={len(developed)}")
    n_guard = max(n_developed, 8)
    guards: list[tuple[float, float]] = []
    attempts = 0
    while len(guards) < n_guard and attempts < n_guard * 80:
        attempts += 1
        pt = _place_point(rng, bbox_g, inner, ctx, cx, cy, radius * 1.6, False)
        if pt is None:
            break
        if inner.contains(Point(*pt)):
            continue
        if not guard_band.contains(Point(*pt)):
            continue
        if not _far_enough(pt, developed + guards, min_d * 0.5):
            continue
        guards.append(pt)
    # Ensure Voronoi stays finite: extra ring on the guard envelope.
    envelope = guard_band.exterior
    n_ring = max(8, n_developed)
    for i in range(n_ring):
        d = envelope.length * (i + 0.5) / n_ring
        p = envelope.interpolate(d)
        guards.append((float(p.x), float(p.y)))
    return developed + guards, len(developed)


def _lloyd(points: np.ndarray, n_developed: int, clip: Polygon,
           ctx: SiteContext) -> np.ndarray:
    pts = np.array(points, dtype=np.float64)
    for _ in range(LLOYD_PASSES):
        cells, _ridges = _cells_for_points(pts, clip)
        for i in range(n_developed):
            cell = cells[i] if i < len(cells) else None
            if cell is None or cell.is_empty:
                continue
            c = cell.centroid
            nx, ny = float(c.x), float(c.y)
            sample = ctx.sample(nx, ny)
            ox, oy = float(pts[i][0]), float(pts[i][1])
            if not sample.get("buildable", False):
                nx = ox + (nx - ox) * LLOYD_EXCLUDED_FRACTION
                ny = oy + (ny - oy) * LLOYD_EXCLUDED_FRACTION
                if not ctx.sample(nx, ny).get("buildable", False):
                    continue
            pts[i][0] = nx
            pts[i][1] = ny
    return pts


def _merge_close_vertices(rings: list[list[list[float]]]) -> tuple[list[list[list[float]]], int]:
    """Merge vertices closer than JUNCTION_MERGE_GU. Skip rings that become invalid."""
    pts = []
    for ring in rings:
        for p in ring:
            pts.append((float(p[0]), float(p[1])))
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < JUNCTION_MERGE_GU:
                union(i, j)
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    merged_coord = {}
    for root, members in clusters.items():
        sx = sum(pts[i][0] for i in members) / len(members)
        sy = sum(pts[i][1] for i in members) / len(members)
        merged_coord[root] = [sx, sy]
    new_rings = []
    skipped = 0
    gi = 0
    for ri, ring in enumerate(rings):
        rebuilt = []
        for _vi, p in enumerate(ring):
            rebuilt.append(merged_coord[find(gi)])
            gi += 1
        try:
            norm = normalize_ring(rebuilt)
            if abs(ring_area(norm["ring"])) < MIN_PATCH_AREA_GU2:
                new_rings.append(ring)
                skipped += 1
                continue
            polygon_from_ring(norm["ring"])
            new_rings.append(norm["ring"])
        except TownLayoutError:
            new_rings.append(ring)
            skipped += 1
    return new_rings, skipped


def _coalesce_short_topology_edges(topo: dict[str, Any]) -> dict[str, int]:
    """Collapse sub-96 GU boundary edges into a shared topology node.

    Water-cropped shoreline polygons legitimately interleave at 128-GU sample
    phase, so a mutual T-junction (each fragment endpoint a vertex of exactly
    one polygon) splits one shared boundary into a long run plus a sub-96 GU
    fragment.  The fragment is a serialization artifact, not a street: merge
    its endpoint nodes and drop the fragment edge.  The cropped patch polygons
    stay authoritative and are not nudged.  A patch pair left with no shared
    edge degenerates to point contact and is demoted from adjacency.
    """
    edges = topo["boundary_edges"]

    def edge_length(edge) -> float:
        coords = edge["geometry"]
        return sum(math.hypot(coords[i + 1][0] - coords[i][0],
                              coords[i + 1][1] - coords[i][1])
                   for i in range(len(coords) - 1))

    parent: dict[str, str] = {}

    def find(nid: str) -> str:
        parent.setdefault(nid, nid)
        while parent[nid] != nid:
            parent[nid] = parent[parent[nid]]
            nid = parent[nid]
        return nid

    for edge in edges:
        if edge_length(edge) < JUNCTION_MERGE_GU:
            ra, rb = find(edge["a_node"]), find(edge["b_node"])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    if not parent:
        return {"fragment_edges": 0, "self_loop_edges": 0,
                "dropped_edges": 0, "demoted_adjacencies": 0}
    nodes_by_id = {n["node_id"]: n for n in topo["nodes"]}
    members: dict[str, list[str]] = {}
    for nid in list(parent):
        members.setdefault(find(nid), []).append(nid)
    position = {}
    for root, group in members.items():
        position[root] = [sum(nodes_by_id[m]["position"][0] for m in group) / len(group),
                          sum(nodes_by_id[m]["position"][1] for m in group) / len(group)]
    kept = []
    dropped_pairs = []
    dropped_self_loops = 0
    dropped_fragments = 0
    for edge in edges:
        ra, rb = find(edge["a_node"]), find(edge["b_node"])
        if ra == rb:
            if edge_length(edge) < JUNCTION_MERGE_GU:
                dropped_fragments += 1
            else:
                dropped_self_loops += 1
            dropped_pairs.append((edge["patch_left"], edge["patch_right"]))
            continue
        edge["a_node"], edge["b_node"] = ra, rb
        coords = edge["geometry"]
        if ra in position and math.hypot(coords[0][0] - position[ra][0],
                                         coords[0][1] - position[ra][1]) < JUNCTION_MERGE_GU:
            coords[0] = list(position[ra])
        if rb in position and math.hypot(coords[-1][0] - position[rb][0],
                                         coords[-1][1] - position[rb][1]) < JUNCTION_MERGE_GU:
            coords[-1] = list(position[rb])
        kept.append(edge)
    for root, pos in position.items():
        nodes_by_id[root]["position"] = pos
    absorbed = {m for root, group in members.items() for m in group if m != root}
    topo["nodes"] = [n for n in topo["nodes"] if n["node_id"] not in absorbed]
    surviving_pairs = {(e["patch_left"], e["patch_right"]) for e in kept}
    demoted = 0
    for pair in dropped_pairs:
        if pair in surviving_pairs or pair[0] is None or pair[1] is None:
            continue
        demoted += 1
        for patch in topo["patches"]:
            if patch["patch_id"] == pair[0] and pair[1] in patch["neighbour_patch_ids"]:
                patch["neighbour_patch_ids"].remove(pair[1])
            if patch["patch_id"] == pair[1] and pair[0] in patch["neighbour_patch_ids"]:
                patch["neighbour_patch_ids"].remove(pair[0])
    topo["boundary_edges"] = kept
    return {"fragment_edges": dropped_fragments,
            "self_loop_edges": dropped_self_loops,
            "demoted_adjacencies": demoted}


def _shared_geoms(a: Polygon, b: Polygon) -> list:
    inter = a.boundary.intersection(b.boundary)
    geoms = []
    if inter.is_empty:
        # Nearness is not adjacency: point contact and gaps are not shared
        # patch boundaries after water cropping.
        return geoms
    if inter.geom_type == "LineString":
        geoms = [inter]
    elif inter.geom_type == "MultiLineString":
        geoms = list(inter.geoms)
    elif inter.geom_type == "GeometryCollection":
        geoms = [g for g in inter.geoms if g.geom_type in ("LineString", "MultiLineString")]
        extra = []
        for g in geoms:
            if g.geom_type == "MultiLineString":
                extra.extend(list(g.geoms))
        geoms = [g for g in geoms if g.geom_type == "LineString"] + extra
    return geoms


def _coalesce_short_shared_edges(
    rings: list[list[list[float]]], patch_ids: list[str],
) -> tuple[list[list[list[float]]], list[str], int]:
    """Merge adjacent polygons whose shared boundary is below 96 GU.

    Only a simple, hole-free union is accepted.  The scan restarts after each
    merge so transitive short junctions are handled deterministically; an
    invalid union is left intact and is reported to the caller.
    """
    merged_count = 0
    while True:
        polygons = [polygon_from_ring(ring) for ring in rings]
        choice = None
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                shared = polygons[i].boundary.intersection(polygons[j].boundary)
                if shared.is_empty or float(shared.length) >= JUNCTION_MERGE_GU:
                    continue
                union = polygons[i].union(polygons[j])
                if union.geom_type != "Polygon" or union.interiors:
                    continue
                if abs(float(union.area) - polygons[i].area - polygons[j].area) > 1.0:
                    continue
                choice = (i, j, union)
                break
            if choice is not None:
                break
        if choice is None:
            break
        i, j, union = choice
        keep_id = min(patch_ids[i], patch_ids[j])
        rings[i] = _open_ring_from_poly(union)
        patch_ids[i] = keep_id
        del rings[j]
        del patch_ids[j]
        merged_count += 1
    return rings, patch_ids, merged_count


def _build_topology(rings: list[list[list[float]]], candidate_id: str,
                    ctx: SiteContext,
                    neighbour_pairs: Optional[set[tuple[int, int]]] = None,
                    patch_ids: Optional[list[str]] = None,
                    ) -> dict[str, Any]:
    patches = []
    nodes_by_key: dict[tuple[int, int], dict] = {}
    node_seq = 0

    def node_id_for(pt: list[float]) -> str:
        nonlocal node_seq
        key = (int(round(pt[0] * 100)), int(round(pt[1] * 100)))
        if key not in nodes_by_key:
            nid = f"node_{candidate_id}_{node_seq:04d}"
            node_seq += 1
            nodes_by_key[key] = {
                "node_id": nid,
                "position": [float(pt[0]), float(pt[1])],
                "kind": "junction",
            }
        return nodes_by_key[key]["node_id"]

    polys = [polygon_from_ring(r) for r in rings]
    ids = patch_ids or [f"patch_{candidate_id}_{i:04d}" for i in range(len(rings))]
    neighbours = [set() for _ in rings]
    edges = []
    edge_seq = 0
    n = len(rings)
    pairs: set[tuple[int, int]] = set()
    # Rebuild adjacency from the cropped rings.  The pre-crop Voronoi ridge
    # list is only a seed hint and is deliberately not topology authority.
    for i in range(n):
        for j in range(i + 1, n):
            pairs.add((i, j))
    for i, j in sorted(pairs):
        geoms = _shared_geoms(polys[i], polys[j])
        wrote = False
        for geom in geoms:
            if geom.length < VERTEX_EPS_GU:
                continue
            coords = [[float(x), float(y)] for x, y, *_ in geom.coords]
            if len(coords) < 2:
                continue
            neighbours[i].add(j)
            neighbours[j].add(i)
            eid = f"edge_{candidate_id}_{edge_seq:04d}"
            edge_seq += 1
            edges.append({
                "edge_id": eid,
                "a_node": node_id_for(coords[0]),
                "b_node": node_id_for(coords[-1]),
                "geometry": coords,
                "patch_left": ids[i],
                "patch_right": ids[j],
                "edge_role": "block",
                "road_class": "none",
            })
            wrote = True
    for i in range(n):
        for vi in range(len(rings[i])):
            node_id_for(rings[i][vi])

    # Isolated islands are valid land components.  Give them an exterior
    # boundary edge so the canonical topology does not discard them.
    for i, poly in enumerate(polys):
        if neighbours[i]:
            continue
        coords = list(poly.exterior.coords)
        a, b = max(zip(coords, coords[1:]), key=lambda pair:
                   (math.hypot(pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]), pair))
        edges.append({
            "edge_id": f"edge_{candidate_id}_{edge_seq:04d}",
            "a_node": node_id_for([a[0], a[1]]),
            "b_node": node_id_for([b[0], b[1]]),
            "geometry": [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]],
            "patch_left": ids[i], "patch_right": None,
            "edge_role": "block", "road_class": "none",
        })
        edge_seq += 1

    for i, ring in enumerate(rings):
        poly = polys[i]
        c = poly.centroid
        sample = ctx.sample(float(c.x), float(c.y))
        mean_slope = float(sample.get("slope_cost", 0.0)) * 25.0
        patches.append({
            "patch_id": ids[i],
            "polygon": ring,
            "neighbour_patch_ids": sorted(
                ids[j] for j in neighbours[i]),
            "inside_city": True,
            "inside_wall": False,
            "morphology_region": "organic",
            "terrain_summary": {
                "mean_slope_deg": mean_slope,
                "water": not sample.get("buildable", True),
            },
        })
    nodes = sorted(nodes_by_key.values(), key=lambda n: n["node_id"])
    return {"patches": patches, "boundary_edges": edges, "nodes": nodes}


def generate_organic_patches(
    ctx: SiteContext,
    domain_ring: list,
    town_brief: dict,
    *,
    master_seed: int,
    candidate_id: str = "c00",
    approaches: Optional[list] = None,
) -> dict[str, Any]:
    """Return a MacroLayoutCandidate dict.  Approach crossings bias density."""
    # Approaches are deliberately not a patch-seed attractor.  Stage 04's
    # domain score is the sole macro-growth authority; road influence belongs
    # to later stages and previously created spokes toward gates.
    domain = polygon_from_ring(domain_ring)
    n_dev = _n_developed(ctx, domain, town_brief)
    rng = stage_rng(int(master_seed), candidate_id, "patches", "")
    # Seed the complete usable land envelope.  Restricting developed seeds to
    # a preferred-area disk was the Stage 03 source of one giant unselected
    # boundary wedge; compact domain growth, not seed omission, chooses the
    # eventual city footprint.
    inner = domain
    seeds, n_developed = _make_seeds(
        ctx, domain, inner, rng, n_dev)
    pts = np.asarray(seeds, dtype=np.float64)
    pts = _lloyd(pts, n_developed, domain, ctx)
    # The finite Voronoi guard points make complete cells available beyond the
    # search ring.  Keep only complete cells contained by that ring; boundary
    # cells are construction support, never clipped city patches.
    cells, ridges = _cells_for_points(pts, None)
    water = unary_union(ctx.water_polygons()) if ctx.water_polygons() else None
    rings = []
    patch_ids: list[str] = []
    reports = []
    kept_from_seed: dict[int, list[int]] = {}
    wet_area = 0.0
    preclip_area = 0.0
    for i, cell in enumerate(cells):
        if cell is None or not domain.covers(cell):
            reports.append({
                "stage": "patches",
                "status": "rejected",
                "message": f"seed {i} discarded as sliver or empty",
            })
            continue
        preclip_area += float(cell.area)
        clipped = cell.difference(water) if water is not None else cell
        wet_area += float(cell.area - clipped.area)
        # Terrain is placement evidence, not a tessellation eraser.  In
        # particular, a steep representative point must not discard an
        # otherwise valid land part and open a hole in the partition.
        parts = retain_land_parts(clipped)
        parts.sort(key=lambda p: (-float(p.area), float(p.centroid.x), float(p.centroid.y)))
        if not parts:
            reports.append({"stage": "patches", "status": "rejected",
                            "message": f"seed {i} discarded after water crop"})
            continue
        kept_from_seed[i] = []
        for part_no, part in enumerate(parts):
            rings.append(_open_ring_from_poly(part))
            patch_ids.append(f"patch_{candidate_id}_{i:04d}" +
                             (f"_part{part_no:02d}" if part_no else ""))
            kept_from_seed[i].append(len(rings) - 1)
    if len(rings) < 3:
        raise TownLayoutError("invalid_polygon: fewer than 3 developed patches")
    # Averaging vertices after clipping can move a coast edge back into water;
    # cropped geometry is already authoritative and must not be nudged.
    rings, skipped = ((rings, 0) if water is not None else _merge_close_vertices(rings))
    if skipped:
        reports.append({
            "stage": "patches",
            "status": "repaired",
            "message": f"skipped {skipped} junction merges that broke simplicity",
        })
    rings, patch_ids, kept_from_seed = _coalesce_subfloor_cells(
        rings, patch_ids, kept_from_seed)
    # Reconstructed finite Voronoi regions can be absent for degenerate guard
    # points.  Close only the measured remainder of the land partition; this is
    # not a terrain/suitability decision and cannot overlap existing patches.
    complete_cells = [cell for cell in cells
                      if cell is not None and cell.geom_type == "Polygon"
                      and domain.covers(cell)]
    if not complete_cells:
        raise TownLayoutError("invalid_polygon: no complete interior Voronoi cells")
    land_for_partition = unary_union(complete_cells)
    if water is not None:
        land_for_partition = land_for_partition.difference(water)
    # The enlarged search envelope can expose detached coastal remnants below
    # the established patch floor.  They are not usable patch geometry and
    # must not become invalid coverage patches merely to close the envelope.
    land_parts = [part for part in simple_polygon_parts(land_for_partition)
                  if part.area >= MIN_PATCH_AREA_GU2]
    effective_land = unary_union(land_parts) if land_parts else land_for_partition
    dropped_land_area = float(land_for_partition.area - effective_land.area)
    if dropped_land_area > LAND_COVERAGE_SLOP_GU2:
        reports.append({"stage": "patches", "status": "ok",
                        "message": f"dropped_subfloor_land_gu2={dropped_land_area:.1f}"})
    remainder = effective_land.difference(unary_union(
        [polygon_from_ring(r) for r in rings]))
    # Remainders are first materialized even below the normal patch floor so
    # they can be absorbed into an adjacent valid patch instead of becoming a
    # coverage hole.
    remainder_parts = [
        part for part in simple_polygon_parts(remainder)
        if part.area >= MIN_PATCH_AREA_GU2
    ]
    subfloor_remainder = [
        part for part in simple_polygon_parts(remainder)
        if LAND_COVERAGE_SLOP_GU2 < part.area < MIN_PATCH_AREA_GU2
    ]
    if subfloor_remainder:
        effective_land = effective_land.difference(unary_union(subfloor_remainder))
        reports.append({"stage": "patches", "status": "ok",
                        "message": f"dropped_subfloor_remainder_gu2={sum(p.area for p in subfloor_remainder):.1f}"})
    for part_no, part in enumerate(remainder_parts):
        rings.append(_open_ring_from_poly(part))
        patch_ids.append(f"patch_{candidate_id}_coverage{part_no:02d}")
    if remainder_parts:
        reports.append({"stage": "patches", "status": "repaired",
                        "message": f"closed {len(remainder_parts)} finite-cell coverage gaps"})
        rings, patch_ids, kept_from_seed = _coalesce_subfloor_cells(
            rings, patch_ids, kept_from_seed)
    rings, patch_ids, short_edge_merges = _coalesce_short_shared_edges(rings, patch_ids)
    if short_edge_merges:
        reports.append({"stage": "patches", "status": "ok",
                        "message": f"coalesced_short_shared_edges={short_edge_merges}"})
    neighbour_pairs: set[tuple[int, int]] = set()
    for a, b in ridges:
        if a in kept_from_seed and b in kept_from_seed:
            for ia in kept_from_seed[a]:
                for ib in kept_from_seed[b]:
                    neighbour_pairs.add((min(ia, ib), max(ia, ib)))
    topo = _build_topology(
        rings, candidate_id, ctx,
        neighbour_pairs if neighbour_pairs else None, patch_ids=patch_ids)
    topo_stats = _coalesce_short_topology_edges(topo)
    if topo_stats["fragment_edges"] or topo_stats["self_loop_edges"]:
        reports.append({"stage": "patches", "status": "ok",
                        "message": f"coalesced_short_topology_edges={topo_stats}"})
    # Hard gate: no canonical edge shorter than JUNCTION_MERGE_GU may be
    # serialized; if no legal collapse existed, R1 fails here (plan_rev3 §5.3).
    for edge in topo["boundary_edges"]:
        coords = edge["geometry"]
        edge_len = sum(math.hypot(coords[i + 1][0] - coords[i][0],
                                  coords[i + 1][1] - coords[i][1])
                       for i in range(len(coords) - 1))
        if edge_len < JUNCTION_MERGE_GU:
            raise TownLayoutError(
                f"R1 junction_merge: {edge['edge_id']} length={edge_len:.1f} GU")
    p50 = float(ctx.stamp_footprint_stats["p50"])
    developed_rings = list(rings)
    raw_equivalent = sorted(float(polygon_from_ring(r).area) / p50 for r in developed_rings)
    # Conservative post-road-loss estimate: erode every actual land patch by
    # half the existing arterial corridor width.  This is not a retention
    # factor; it is the area left after the documented 512 GU corridor rule.
    post_loss_equivalent = sorted(
        max(0.0, float(polygon_from_ring(r).buffer(-EXPECTED_ROAD_HALF_WIDTH_GU).area / p50))
        for r in developed_rings)
    distribution = {
        "raw_p10": float(np.percentile(raw_equivalent, 10)),
        "raw_p50": float(np.percentile(raw_equivalent, 50)),
        "raw_p90": float(np.percentile(raw_equivalent, 90)),
        "post_road_loss_p10": float(np.percentile(post_loss_equivalent, 10)),
        "post_road_loss_p50": float(np.percentile(post_loss_equivalent, 50)),
        "post_road_loss_p90": float(np.percentile(post_loss_equivalent, 90)),
        "count": len(raw_equivalent),
        "requested_developed_count": n_dev,
        "admitted_developed_count": n_developed,
        "expected_road_half_width_gu": EXPECTED_ROAD_HALF_WIDTH_GU,
        "raw_target_footprints": RAW_PATCH_TARGET_FOOTPRINTS,
    }
    reports.append({"stage": "patches", "status": "ok",
                    "message": (f"preclip_water_fraction={wet_area / preclip_area if preclip_area else 0.0:.6f} "
                                 f"requested={n_dev} admitted={n_developed} "
                                 f"raw_p10_p50_p90={distribution['raw_p10']:.3f},{distribution['raw_p50']:.3f},{distribution['raw_p90']:.3f} "
                                 f"post_road_loss_p10_p50_p90="
                                 f"{distribution['post_road_loss_p10']:.3f},{distribution['post_road_loss_p50']:.3f},{distribution['post_road_loss_p90']:.3f}")})
    patch_water_overlap = sum(
        polygon_from_ring(r).intersection(water).area for r in rings
    ) if water is not None else 0.0
    land = domain.difference(water) if water is not None else domain
    patch_polys = [polygon_from_ring(r) for r in rings]
    patch_union = unary_union(patch_polys)
    coverage_gap = float(effective_land.difference(patch_union).area)
    overlap_area = float(sum(patch_polys[i].intersection(patch_polys[j]).area
                             for i in range(len(patch_polys))
                             for j in range(i + 1, len(patch_polys))))
    coverage_slop = LAND_COVERAGE_SLOP_GU2
    if coverage_gap > coverage_slop or overlap_area > coverage_slop:
        raise TownLayoutError(
            f"land_coverage: gap={coverage_gap:.9g} overlap={overlap_area:.9g}")
    product = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "city_domain": domain_ring,
        "morphology_regions": [{"region_id": "organic", "patch_ids": [
            p["patch_id"] for p in topo["patches"]]}],
        "patches": topo["patches"],
        "boundary_edges": topo["boundary_edges"],
        "nodes": topo["nodes"],
        "reports": reports,
        "water_metrics": {"preclip_water_area_gu2": wet_area,
                          "preclip_patch_area_gu2": preclip_area,
                          "preclip_water_fraction": wet_area / preclip_area if preclip_area else 0.0,
                           "patch_water_intersection_gu2": float(patch_water_overlap)},
        "land_coverage_metrics": {
            "rewrite_land_area_gu2": float(land.area),
            "uncovered_land_gu2": coverage_gap,
            "overlap_gu2": overlap_area,
            "patch_water_intersection_gu2": float(patch_water_overlap),
            "slop_gu2": coverage_slop,
        },
        "water_polygons": [normalize_ring([[c[0], c[1]] for c in p.exterior.coords])["ring"]
                           for p in ctx.water_polygons()],
        "patch_area_distribution_footprints": distribution,
    }
    return require_macro_layout(product)


def write_patches_diagnostic(
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

    domain = product.get("city_domain") or []
    if len(domain) >= 3:
        draw.polygon([to_px(p) for p in domain], outline=(0, 220, 255, 220))
    for water in product.get("water_polygons") or []:
        if len(water) >= 3:
            draw.polygon([to_px(p) for p in water], fill=(20, 80, 255, 150), outline=(0, 20, 255, 255))
    palette = [
        (220, 80, 80, 70), (80, 180, 80, 70), (80, 80, 220, 70),
        (220, 180, 60, 70), (180, 80, 180, 70), (60, 200, 200, 70),
        (200, 120, 60, 70), (120, 200, 80, 70), (80, 140, 220, 70),
        (200, 80, 140, 70), (140, 140, 80, 70), (100, 180, 160, 70),
    ]
    for i, patch in enumerate(product.get("patches", [])):
        ring = patch.get("polygon") or []
        if len(ring) < 3:
            continue
        color = palette[i % len(palette)]
        draw.polygon([to_px(p) for p in ring], fill=color, outline=(20, 20, 20, 220))
        if product.get("water_polygons"):
            from shapely.ops import unary_union
            if polygon_from_ring(ring).intersection(unary_union(
                    [polygon_from_ring(w) for w in product["water_polygons"]])).area > 0.0:
                draw.polygon([to_px(p) for p in ring], fill=(255, 0, 0, 230))
    Image.alpha_composite(image, overlay).save(out_png)
