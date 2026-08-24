"""Build arterial-safe town blocks from the accepted Stage-A road tree.

Inputs are the self-contained ``r2a_arterials`` checkpoint.  The module
subtracts the displayed arterial corridor from every selected city cell,
retains every positive-area remainder, and merges only across original
non-arterial shared cell boundaries.  Its output is the Stage-B authority
consumed by minor-road growth; arterial barriers are never crossed or rebuilt.

Invariants: selected land is accounted for once, road verges remain explicit,
blocks are simple hole-free polygons, deterministic ordering is used
throughout, and no merge may exceed the accepted serviced-lot/shape limits.
"""
from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from typing import Any

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .arterial_graph import build_fine_graph
from .constants import SERVICED_LOT_AREA_GU2
from .geometry import normalize_ring, polygon_from_ring
from .validate import TownLayoutError

MIN_VERGE_AREA_GU2 = 65_536.0
MIN_SHARED_GU = 96.0
MAX_LOT_EQ = 6.5
MAX_ASPECT = 4.0
MIN_COMPACTNESS = 0.08
EROSION_GU = 256.0


def _parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [g for g in geometry.geoms if g.geom_type == "Polygon" and not g.is_empty]


def _ring(poly: Polygon) -> list[list[float]]:
    return normalize_ring([[float(x), float(y)] for x, y in poly.exterior.coords])["ring"]


def _compactness(poly: Polygon) -> float:
    return 4.0 * math.pi * poly.area / (poly.length * poly.length) if poly.length else 0.0


def _aspect(poly: Polygon) -> float:
    box = poly.minimum_rotated_rectangle
    coords = list(box.exterior.coords)
    lengths = sorted(LineString([coords[i], coords[i + 1]]).length for i in range(4))
    return lengths[-1] / max(lengths[0], 1e-9)


def _legal(poly: Polygon) -> bool:
    if (poly.geom_type != "Polygon" or poly.is_empty or not poly.is_valid
            or poly.interiors or poly.area > MAX_LOT_EQ * SERVICED_LOT_AREA_GU2 + 1.0
            or _aspect(poly) > MAX_ASPECT or _compactness(poly) < MIN_COMPACTNESS):
        return False
    eroded = poly.buffer(-EROSION_GU)
    return eroded.geom_type == "Polygon" and not eroded.is_empty


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.floor(fraction * (len(ordered) - 1))))]


def _graph_distances(start: str, adjacency: dict[str, dict[str, float]]) -> dict[str, float]:
    distances = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for other, weight in adjacency[node].items():
            candidate = distance + weight
            if candidate < distances.get(other, math.inf):
                distances[other] = candidate
                heapq.heappush(queue, (candidate, other))
    return distances


def _components(ids: list[str], adjacency: dict[str, dict[str, float]]) -> list[list[str]]:
    unseen = set(ids)
    result = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component = []
        unseen.remove(seed)
        while stack:
            node = stack.pop()
            component.append(node)
            for other in sorted(adjacency[node], reverse=True):
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        result.append(sorted(component))
    return result


def _choose_seeds(component: list[str], count: int,
                  adjacency: dict[str, dict[str, float]],
                  faces: dict[str, dict]) -> list[str]:
    # A seed is already a serialized block if growth happens to favour other
    # targets.  Therefore seed only faces that independently meet the final
    # shape contract; awkward atomic remnants remain available for absorption.
    eligible = [face_id for face_id in component if _legal(faces[face_id]["geometry"])]
    if not eligible:
        eligible = list(component)
    seeds = [min(eligible)]
    distance_cache = {seeds[0]: _graph_distances(seeds[0], adjacency)}
    while len(seeds) < min(count, len(eligible)):
        best = None
        for face_id in eligible:
            if face_id in seeds:
                continue
            nearest = min(distance_cache[s].get(face_id, math.inf) for s in seeds)
            key = (nearest, face_id)
            if best is None or key[0] > best[0] or (key[0] == best[0] and key[1] < best[1]):
                best = key
        assert best is not None
        seeds.append(best[1])
        distance_cache[best[1]] = _graph_distances(best[1], adjacency)
    return seeds


def _class_targets(seeds: list[str], faces: dict[str, dict]) -> dict[str, tuple[str, float]]:
    ordered = sorted(seeds, key=lambda face_id: (faces[face_id]["geometry"].area, face_id))
    count = len(ordered)
    if count < 8:
        return {seed: ("standard", 3.0) for seed in ordered}
    islands = int(math.floor(0.15 * count + 0.5))
    standards = int(math.floor(0.70 * count + 0.5))
    standards = min(standards, count - islands)
    result = {}
    for index, seed in enumerate(ordered):
        if index < islands:
            result[seed] = ("island", 1.5)
        elif index < islands + standards:
            result[seed] = ("standard", 3.0)
        else:
            result[seed] = ("run", 5.0)
    return result


def _merge_component(component: list[str], faces: dict[str, dict],
                     adjacency: dict[str, dict[str, float]]) -> tuple[list[dict], list[str]]:
    area = sum(faces[f]["geometry"].area for f in component)
    seed_count = max(1, int(math.floor(area / (3.0 * SERVICED_LOT_AREA_GU2) + 0.5)))
    seeds = _choose_seeds(component, seed_count, adjacency, faces)
    targets = _class_targets(seeds, faces)
    owner = {seed: seed for seed in seeds}
    members = {seed: {seed} for seed in seeds}
    geometry = {seed: faces[seed]["geometry"] for seed in seeds}
    unassigned = set(component) - set(seeds)

    while unassigned:
        options = []
        for seed in sorted(seeds):
            boundary = set()
            for member in members[seed]:
                boundary.update(adjacency[member])
            for face_id in sorted(boundary & unassigned):
                candidate = unary_union([geometry[seed], faces[face_id]["geometry"]])
                if not _legal(candidate):
                    continue
                target = targets[seed][1]
                lot_eq = candidate.area / SERVICED_LOT_AREA_GU2
                score = abs(lot_eq - target) / target + (1.0 - _compactness(candidate))
                options.append((round(score, 12), seed, face_id, candidate))
        if not options:
            # Preserve rather than force malformed geometry.  These explicit
            # exceptions are a visual-review stop condition, never hidden.
            break
        _, seed, face_id, candidate = min(options, key=lambda row: row[:3])
        owner[face_id] = seed
        members[seed].add(face_id)
        geometry[seed] = candidate
        unassigned.remove(face_id)

    rows = []
    for seed in sorted(seeds):
        poly = geometry[seed]
        target_class, target_eq = targets[seed]
        lot_eq = poly.area / SERVICED_LOT_AREA_GU2
        actual = "island" if lot_eq <= 2.25 else "standard" if lot_eq < 4.0 else "run"
        rows.append({
            "seed_face_id": seed,
            "geometry": poly,
            "member_atomic_face_ids": sorted(members[seed]),
            "member_patch_ids": sorted({faces[f]["patch_id"] for f in members[seed]}),
            "target_class": target_class,
            "target_lot_equivalents": target_eq,
            "actual_class": actual,
            "exception_reason": None,
        })
    for face_id in sorted(unassigned):
        rows.append({
            "seed_face_id": face_id,
            "geometry": faces[face_id]["geometry"],
            "member_atomic_face_ids": [face_id],
            "member_patch_ids": [faces[face_id]["patch_id"]],
            "target_class": "exception",
            "target_lot_equivalents": 0.0,
            "actual_class": "exception",
            "exception_reason": "no_legal_merge",
        })
    return rows, sorted(unassigned)


def _line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type in ("Point", "MultiPoint", "Polygon", "MultiPolygon"):
        # Boundary intersections can degenerate to pure touch points (a block
        # or corridor meeting at exactly one vertex); they carry no length and
        # are never edges.
        return []
    return [g for g in geometry.geoms if g.geom_type == "LineString" and g.length > 1e-6]


def build_road_blocks(product: dict[str, Any]) -> dict[str, Any]:
    """Build and validate the Stage-B block checkpoint."""
    started = time.perf_counter()
    if product.get("stage_id") != "r2a_arterials":
        raise TownLayoutError("B input: expected r2a_arterials")
    selected_patches = [p for p in product.get("patches") or [] if p.get("inside_city")]
    if not selected_patches:
        raise TownLayoutError("B input: no selected city patches")
    city_land = unary_union([polygon_from_ring(p["polygon"]) for p in selected_patches])
    corridor = unary_union([polygon_from_ring(r) for r in (product.get("corridor") or {}).get("rings") or []])
    corridor = corridor.intersection(city_land)
    barrier = unary_union([LineString(row["geometry"]) for row in product.get("raw_barrier") or []])

    faces: dict[str, dict] = {}
    verges = []
    for patch in sorted(selected_patches, key=lambda p: p["patch_id"]):
        remainder = polygon_from_ring(patch["polygon"]).difference(corridor)
        for part_index, poly in enumerate(sorted(_parts(remainder), key=lambda p: (-p.area, p.wkt))):
            face_id = f"face_{patch['patch_id']}_{part_index:02d}"
            row = {"face_id": face_id, "patch_id": patch["patch_id"], "geometry": poly}
            if poly.area < MIN_VERGE_AREA_GU2:
                verges.append(row)
            else:
                faces[face_id] = row

    junction_points = [Point(row["position"])
                       for row in product.get("arterial_nodes") or []
                       if row.get("degree", 0) >= 3]
    junction_reach = 3.0 * float((product.get("corridor") or {}).get(
        "half_width_gu", 256.0))
    absorbed_faces = {
        face_id for face_id, row in faces.items()
        if row["geometry"].area <= 0.25 * SERVICED_LOT_AREA_GU2
        and any(row["geometry"].distance(point) <= junction_reach
                for point in junction_points)
    }
    absorbed_verges = {
        row["face_id"] for row in verges
        if any(row["geometry"].distance(point) <= junction_reach
               for point in junction_points)
    }
    junction_fill = ([faces[face_id]["geometry"] for face_id in absorbed_faces]
                     + [row["geometry"] for row in verges
                        if row["face_id"] in absorbed_verges])
    if junction_fill:
        corridor = unary_union([corridor] + junction_fill)
        for face_id in absorbed_faces:
            del faces[face_id]
        verges = [row for row in verges if row["face_id"] not in absorbed_verges]

    adjacency: dict[str, dict[str, float]] = {face_id: {} for face_id in faces}
    face_ids = sorted(faces)
    for index, left in enumerate(face_ids):
        for right in face_ids[index + 1:]:
            shared_geom = faces[left]["geometry"].boundary.intersection(faces[right]["geometry"].boundary)
            shared = shared_geom.length
            if shared < MIN_SHARED_GU or shared_geom.intersection(barrier).length >= MIN_SHARED_GU:
                continue
            ca, cb = faces[left]["geometry"].centroid, faces[right]["geometry"].centroid
            weight = math.hypot(cb.x - ca.x, cb.y - ca.y)
            adjacency[left][right] = weight
            adjacency[right][left] = weight

    arterial_adjacent = {face_id for face_id, row in faces.items()
                         if row["geometry"].boundary.intersection(corridor).length >= MIN_SHARED_GU}
    reachable_components = []
    isolated_components = []
    for component in _components(face_ids, adjacency):
        (reachable_components if set(component) & arterial_adjacent else isolated_components).append(component)

    blocks_internal = []
    exceptions = []
    for component in reachable_components:
        rows, failed = _merge_component(component, faces, adjacency)
        blocks_internal.extend(rows)
        exceptions.extend(failed)

    blocks = []
    for index, row in enumerate(sorted(blocks_internal, key=lambda r: r["seed_face_id"])):
        poly = row.pop("geometry")
        blocks.append({
            "block_id": f"block_{index:03d}",
            "polygon": _ring(poly),
            "usable_area_gu2": float(poly.area),
            "serviced_lot_equivalents": float(poly.area / SERVICED_LOT_AREA_GU2),
            "arterial_frontage_gu": float(poly.boundary.intersection(corridor).length),
            **row,
        })

    isolated = []
    for index, component in enumerate(isolated_components):
        geom = unary_union([faces[f]["geometry"] for f in component])
        for part_index, poly in enumerate(_parts(geom)):
            isolated.append({
                "isolated_id": f"isolated_{index:03d}_{part_index:02d}",
                "reason": "no_arterial_reachable_boundary",
                "member_atomic_face_ids": component,
                "polygon": _ring(poly),
                "area_gu2": float(poly.area),
            })

    block_polys = {b["block_id"]: polygon_from_ring(b["polygon"]) for b in blocks}
    block_edges = []
    seen_pairs = set()
    for block_id in sorted(block_polys):
        poly = block_polys[block_id]
        for other_id in sorted(block_polys):
            if other_id <= block_id:
                continue
            shared = poly.boundary.intersection(block_polys[other_id].boundary)
            for line in _line_parts(shared):
                if line.length < 1e-6:
                    continue
                key = (block_id, other_id, line.wkb_hex)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                block_edges.append({"edge_id": f"be_{len(block_edges):04d}",
                                    "edge_class": "interior_candidate",
                                    "left_block_id": block_id, "right_block_id": other_id,
                                    "geometry": [[float(x), float(y)] for x, y in line.coords],
                                    "length_gu": float(line.length)})
        for line in _line_parts(poly.boundary.intersection(corridor)):
            block_edges.append({"edge_id": f"be_{len(block_edges):04d}",
                                "edge_class": "arterial", "left_block_id": block_id,
                                "right_block_id": None,
                                "geometry": [[float(x), float(y)] for x, y in line.coords],
                                "length_gu": float(line.length)})
        occupied = corridor.boundary.union(unary_union([
            block_polys[o].boundary for o in block_polys if o != block_id]))
        for line in _line_parts(poly.boundary.difference(occupied.buffer(0.01))):
            if line.length >= 1.0:
                block_edges.append({"edge_id": f"be_{len(block_edges):04d}",
                                    "edge_class": "outer_candidate",
                                    "left_block_id": block_id, "right_block_id": None,
                                    "geometry": [[float(x), float(y)] for x, y in line.coords],
                                    "length_gu": float(line.length)})

    represented = unary_union([corridor]
        + [polygon_from_ring(b["polygon"]) for b in blocks]
        + [polygon_from_ring(i["polygon"]) for i in isolated]
        + [v["geometry"] for v in verges])
    gap = city_land.difference(represented).area
    overlap_sum = (corridor.area + sum(b["usable_area_gu2"] for b in blocks)
                   + sum(i["area_gu2"] for i in isolated)
                   + sum(v["geometry"].area for v in verges)) - represented.area
    values = [b["serviced_lot_equivalents"] for b in blocks
              if not b.get("exception_reason")]
    p10, p50, p90 = (_percentile(values, x) for x in (0.10, 0.50, 0.90))
    distribution_pass = 0.75 <= p10 <= 2.25 and 2.0 <= p50 <= 4.0 and 4.0 <= p90 <= 6.5
    invalid = []
    for b in blocks:
        poly = polygon_from_ring(b["polygon"])
        if not _legal(poly):
            invalid.append(
                f"{b['block_id']}(eq={poly.area / SERVICED_LOT_AREA_GU2:.2f},"
                f"aspect={_aspect(poly):.2f},compact={_compactness(poly):.3f},"
                f"erosion={poly.buffer(-EROSION_GU).geom_type},"
                f"exception={b.get('exception_reason')})")
    if gap > 1.0 or overlap_sum > 1.0:
        raise TownLayoutError(f"B area_accounting gap={gap:.3f} overlap={overlap_sum:.3f}")
    if invalid:
        raise TownLayoutError(f"B invalid_blocks {','.join(invalid)}")

    runtime = time.perf_counter() - started
    metrics = {
        "runtime_s": runtime, "atomic_face_count": len(faces),
        "block_count": len(blocks), "road_verge_count": len(verges),
        "isolated_area_count": len(isolated), "exception_count": len(exceptions),
        "p10_lot_equivalents": p10, "p50_lot_equivalents": p50,
        "p90_lot_equivalents": p90, "distribution_pass": distribution_pass,
        "city_land_area_gu2": float(city_land.area),
        "corridor_area_gu2": float(corridor.area),
        "unexplained_gap_gu2": float(gap), "overlap_gu2": float(overlap_sum),
    }
    result = dict(product)
    result.update({
        "stage_id": "r2b_road_blocks", "preceding_checkpoint": None,
        "corridor": {**(product.get("corridor") or {}),
                     "rings": [_ring(poly) for poly in _parts(corridor)]},
        "atomic_faces": [{"face_id": f, "patch_id": row["patch_id"],
                          "polygon": _ring(row["geometry"]),
                          "area_gu2": float(row["geometry"].area)}
                         for f, row in sorted(faces.items())],
        "blocks": blocks,
        "block_edges": sorted(block_edges, key=lambda e: e["edge_id"]),
        "road_verges": [{"verge_id": f"verge_{i:03d}", "patch_id": v["patch_id"],
                          "polygon": _ring(v["geometry"]), "area_gu2": float(v["geometry"].area)}
                         for i, v in enumerate(verges)],
        "isolated_areas": isolated,
        "metrics": metrics,
        "reports": list(product.get("reports") or []) + [{
            "stage": "r2b_road_blocks", "status": "review_required" if exceptions or not distribution_pass else "ok",
            "message": (f"blocks={len(blocks)} p10={p10:.2f} p50={p50:.2f} p90={p90:.2f} "
                        f"exceptions={len(exceptions)} isolated={len(isolated)}"),
        }],
    })
    if runtime > 10.0:
        raise TownLayoutError(f"B runtime {runtime:.2f}s")
    return result
