"""R2 planning-ring and true external source-port construction.

Inputs are the identity-verified R1 macro checkpoint.  This module freezes the
union boundary of the complete city region, distinguishes external road entries from
bounded excursions across a concave outline, and reports only the exterior
continuation of each retained port.  Raw LAND/VTEX-78 tiles are authoritative
for the final port coordinate when a substantial road-footprint overlap is
found; extracted vectors identify and order the approach.  Interior source-road
subchains are guides for later routing, never accepted corridor geometry in
this checkpoint.  No road-network reload or retry search occurs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import substring, unary_union

from .geometry import normalize_ring, polygon_from_ring
from .validate import TownLayoutError

RING_SIMPLIFY_GU = 64.0
RING_AREA_TOLERANCE = 0.005
SAMPLE_GU = 128.0
DRY_CHECK_GU = 4096.0
PORT_MERGE_ARC_GU = 800.0
ENDPOINT_EPS_GU = 1.0
ROAD_TILE_GU = 512.0
TEXTURE_PORT_SEARCH_GU = 4096.0
TEXTURE_PORT_MAX_VECTOR_OFFSET_GU = 512.0
TEXTURE_PORT_MIN_RING_OVERLAP_GU = 1024.0
TEXTURE_PORT_MIN_RELOCATION_GU = 512.0
SHALLOW_INCURSION_MAX_GU = 2048.0
TEXTURE_PORT_PROBE_RADIUS_GU = 2048.0
TEXTURE_PORT_MIN_INTERIOR_DEPTH_GU = 1024.0


def _road_raw_values(product: dict) -> tuple[int, ...]:
    """Read explicit road raw values inherited from the active survey."""

    assignments = product.get("road_assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise TownLayoutError("R2 road texture mask has no explicit road_assignments")
    values = tuple(sorted({int(row["raw_vtex"]) for row in assignments.values()}))
    if not values or any(value <= 0 for value in values):
        raise TownLayoutError("R2 road assignments contain no positive raw VTEX values")
    return values


def _points(geometry) -> list[Point]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type in ("MultiPoint", "GeometryCollection"):
        out = []
        for item in geometry.geoms:
            out.extend(_points(item))
        return out
    return []


def _ring_and_patches(product: dict, selected_ids: set[str] | None = None
                      ) -> tuple[Polygon, dict[str, Polygon]]:
    ids = selected_ids or {str(p["patch_id"]) for p in product.get("patches", [])
                           if p.get("inside_city")}
    patches = {p["patch_id"]: polygon_from_ring(p["polygon"])
               for p in product.get("patches", []) if p.get("patch_id") in ids}
    if not patches:
        raise TownLayoutError("R2 ring: no city-region patches")
    union = unary_union(list(patches.values()))
    if union.geom_type != "Polygon" or union.interiors:
        raise TownLayoutError("R2 ring: city-region union is not a simple polygon")
    return union, patches


def build_planning_ring(product: dict, selected_ids: set[str] | None = None
                        ) -> tuple[list[list[float]], dict[str, Any]]:
    """Freeze the complete city-region exterior without simplification."""
    union, _patches = _ring_and_patches(product, selected_ids)
    original = union
    simplified = union.simplify(RING_SIMPLIFY_GU, preserve_topology=True)
    chosen = original
    shift = 0.0
    area_change = 0.0
    # The optional simplification is deliberately not selected in R2.  The
    # exact union boundary is already within contract and keeps source crossing
    # coordinates stable; later phases consume this frozen geometry verbatim.
    ring = normalize_ring([[x, y] for x, y in chosen.exterior.coords])['ring']
    return ring, {"max_shift_gu": shift, "area_change_pct": area_change,
                  "frozen": True}


def _refine_city_region(product: dict, stroke_data: dict[str, dict[str, Any]]
                        ) -> tuple[set[str], list[dict[str, Any]]]:
    """Retract boundary patches that only nick an external road.

    A patch is retracted when it participates in either (a) a short road
    incursion whose source stroke remains outside at both ends, or (b) a raw
    road-texture contact with no road tile reaching 1,024 GU into the city
    within 2,048 GU of the contact. The ring is rebuilt after each removal so
    the decision is local to the current boundary rather than a fixed town.
    """
    selected = {str(p["patch_id"]) for p in product.get("patches", [])
                if p.get("inside_city")}
    with np.load(Path(product["identities"]["fields"]["path"])) as fields:
        tiles = np.asarray(fields["raw_vtex_tiles"], dtype=np.uint16)
    road_y, road_x = np.where(np.isin(tiles, _road_raw_values(product)))
    road_centers = [Point((float(x) + 0.5) * ROAD_TILE_GU,
                          (float(y) + 0.5) * ROAD_TILE_GU)
                    for y, x in zip(road_y, road_x)]
    records: list[dict[str, Any]] = []

    for _iteration in range(len(selected)):
        ring_points, _meta = build_planning_ring(product, selected)
        ring = polygon_from_ring(ring_points)
        _union, patches = _ring_and_patches(product, selected)
        removal: tuple[str, str, float] | None = None

        # A through-stroke that only enters for a short distance is a clipped
        # exterior road, not two city ports.
        for stroke_id in sorted(stroke_data):
            line = stroke_data[stroke_id]["line"]
            if ring.covers(Point(line.coords[0])) or ring.covers(Point(line.coords[-1])):
                continue
            interior = line.intersection(ring)
            interior_length = float(getattr(interior, "length", 0.0))
            if not (ENDPOINT_EPS_GU < interior_length < SHALLOW_INCURSION_MAX_GU):
                continue
            patch_id = min(patches,
                           key=lambda pid: patches[pid].distance(interior))
            removal = (patch_id, f"short_incursion:{stroke_id}", interior_length)
            break

        # Texture is authoritative when the extracted vector bends away from
        # the visible road. A contact without meaningful interior texture depth
        # retracts its nearest boundary patch and is reconsidered next pass.
        if removal is None:
            for overlap in sorted(_road_texture_ring_overlaps(product, ring),
                                  key=lambda line: ring.boundary.project(
                                      line.interpolate(0.5, normalized=True))):
                if overlap.length < TEXTURE_PORT_MIN_RING_OVERLAP_GU:
                    continue
                contacts = [Point(overlap.coords[0]), Point(overlap.coords[-1])]
                contacts.sort(key=ring.boundary.project)
                for contact in contacts:
                    depths = [ring.boundary.distance(center)
                              for center in road_centers
                              if center.distance(contact) <= TEXTURE_PORT_PROBE_RADIUS_GU
                              and ring.covers(center)]
                    depth = max(depths or [0.0])
                    if depth >= TEXTURE_PORT_MIN_INTERIOR_DEPTH_GU:
                        continue
                    patch_id = min(
                        patches, key=lambda pid: patches[pid].distance(contact))
                    removal = (patch_id, "shallow_texture_contact", float(depth))
                    break
                if removal is not None:
                    break

        if removal is None:
            return selected, records
        patch_id, reason, measure = removal
        trial = selected - {patch_id}
        trial_union, _trial_patches = _ring_and_patches(product, trial)
        if trial_union.geom_type != "Polygon" or trial_union.interiors:
            raise TownLayoutError(
                f"R2 boundary retraction would break city region at {patch_id}")
        selected = trial
        records.append({"patch_id": patch_id, "reason": reason,
                        "measure_gu": float(measure)})
    raise TownLayoutError("R2 boundary retraction did not converge")


def _inside_sample(line: LineString, distance: float, ring: Polygon) -> bool:
    return ring.contains(line.interpolate(max(0.0, min(line.length, distance))))


def _crossing_tangent(line: LineString, point: Point, ring: Polygon) -> tuple[tuple[float, float], bool]:
    distance = line.project(point)
    before = _inside_sample(line, distance - SAMPLE_GU, ring)
    after = _inside_sample(line, distance + SAMPLE_GU, ring)
    if before == after:
        raise ValueError("tangential crossing")
    outside_before = not before
    if before:
        # Storage direction is inside -> outside; reverse it.
        a = line.interpolate(min(line.length, distance + SAMPLE_GU))
        b = line.interpolate(max(0.0, distance - SAMPLE_GU))
    else:
        a = line.interpolate(max(0.0, distance - SAMPLE_GU))
        b = line.interpolate(min(line.length, distance + SAMPLE_GU))
    dx, dy = b.x - a.x, b.y - a.y
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        raise ValueError("zero crossing tangent")
    return (dx / norm, dy / norm), outside_before


def _outside_segment(line: LineString, point: Point, outside_before: bool) -> LineString:
    """Return the source stroke from the port outward, preserving every bend."""
    distance = line.project(point)
    if outside_before:
        segment = substring(line, 0.0, distance)
        return LineString(list(reversed(segment.coords)))
    segment = substring(line, distance, line.length)
    return LineString(segment.coords)


def _line_parts(geometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        return [part for part in geometry.geoms if part.geom_type == "LineString"]
    return []


def _road_texture_ring_overlaps(product: dict, ring: Polygon) -> list[LineString]:
    """Intersect the planning ring with authoritative raw-78 LAND tiles."""
    fields_path = Path(product["identities"]["fields"]["path"])
    with np.load(fields_path) as fields:
        tiles = np.asarray(fields["raw_vtex_tiles"], dtype=np.uint16)
    road_y, road_x = np.where(np.isin(tiles, _road_raw_values(product)))
    if not len(road_x):
        raise TownLayoutError("R2 texture ports: no configured road tiles")
    road_mask = unary_union([
        box(float(x) * ROAD_TILE_GU, float(y) * ROAD_TILE_GU,
            float(x + 1) * ROAD_TILE_GU, float(y + 1) * ROAD_TILE_GU)
        for y, x in zip(road_y, road_x)
    ])
    return _line_parts(ring.boundary.intersection(road_mask))


def _texture_port(crossing: dict, line: LineString, ring: Polygon,
                  overlaps: list[LineString], patches: dict[str, Polygon],
                  water) -> None:
    """Move a vector port to the first substantial road-texture overlap."""
    vector_point = Point(crossing["position"])
    vector_distance = line.project(vector_point)
    outside_before = bool(crossing["_outside_before"])
    candidates = []
    for overlap in overlaps:
        if (overlap.length < TEXTURE_PORT_MIN_RING_OVERLAP_GU
                or overlap.distance(line) > TEXTURE_PORT_MAX_VECTOR_OFFSET_GU):
            continue
        midpoint = overlap.interpolate(0.5, normalized=True)
        midpoint_projected = line.project(midpoint)
        outward_delta = (vector_distance - midpoint_projected if outside_before
                         else midpoint_projected - vector_distance)
        if -ENDPOINT_EPS_GU <= outward_delta <= TEXTURE_PORT_SEARCH_GU:
            endpoints = [Point(overlap.coords[0]), Point(overlap.coords[-1])]
            port_point = min(endpoints, key=line.project) if outside_before \
                else max(endpoints, key=line.project)
            port_projected = line.project(port_point)
            order = port_projected if outside_before else -port_projected
            candidates.append((round(order, 6),
                               round(ring.boundary.project(port_point), 6),
                               port_point, port_projected,
                               midpoint_projected,
                               vector_point.distance(midpoint), overlap))
    if not candidates:
        crossing["port_basis"] = "aligned_vector_crossing_fallback"
        crossing["texture_port_shift_gu"] = 0.0
        return
    (_order, _arc, port_point, projected, _midpoint_projected,
     midpoint_shift, _chosen_overlap) = min(candidates, key=lambda item: item[:2])
    shift = vector_point.distance(port_point)
    if midpoint_shift < TEXTURE_PORT_MIN_RELOCATION_GU:
        crossing["port_basis"] = "aligned_vector_crossing"
        crossing["texture_port_shift_gu"] = 0.0
        return

    projected_point = line.interpolate(projected)
    source_outside = _outside_segment(line, projected_point, outside_before)
    outside_points = [[float(port_point.x), float(port_point.y)]]
    if port_point.distance(projected_point) > ENDPOINT_EPS_GU:
        outside_points.append([float(projected_point.x), float(projected_point.y)])
    outside_points.extend([[float(x), float(y)] for x, y in source_outside.coords[1:]])
    outside = LineString(outside_points)

    crossing["vector_crossing_position"] = list(crossing["position"])
    crossing["position"] = [float(port_point.x), float(port_point.y)]
    crossing["ring_arc_gu"] = float(ring.boundary.project(port_point))
    crossing["perimeter_patch_id"] = min(
        patches, key=lambda patch_id: patches[patch_id].distance(port_point))
    crossing["port_basis"] = "configured_road_vtex_ring_overlap"
    crossing["texture_port_shift_gu"] = float(shift)
    crossing["texture_opening_geometries"] = [
        [[float(x), float(y)] for x, y in overlap.coords]
        for _candidate_order, _candidate_arc, _candidate_port,
            candidate_projected, _candidate_midpoint_projected,
            _candidate_midpoint_shift, overlap in candidates
        if (projected - ENDPOINT_EPS_GU <= candidate_projected
            <= vector_distance + ENDPOINT_EPS_GU)
        if outside_before
    ] if outside_before else [
        [[float(x), float(y)] for x, y in overlap.coords]
        for _candidate_order, _candidate_arc, _candidate_port,
            candidate_projected, _candidate_midpoint_projected,
            _candidate_midpoint_shift, overlap in candidates
        if (vector_distance - ENDPOINT_EPS_GU <= candidate_projected
            <= projected + ENDPOINT_EPS_GU)
    ]
    crossing["_outside"] = outside
    crossing["continuation_checked_length_gu"] = float(outside.length)
    if outside.intersects(water) if water is not None else False:
        crossing["continuation"] = "continuation_bridge_dependent"
    elif outside.length < DRY_CHECK_GU:
        crossing["continuation"] = "short"
    else:
        crossing["continuation"] = "dry"


def _edge_strokes(edges: list[dict]) -> dict[str, str]:
    """Assign stable stroke IDs, joining through degree-2 serialized nodes."""
    by_node: dict[str, list[str]] = {}
    for edge in edges:
        for node in (edge["from_node"], edge["to_node"]):
            by_node.setdefault(node, []).append(edge["id"])
    adjacency = {edge_id: set() for edge_id in (e["id"] for e in edges)}
    for incident in by_node.values():
        if len(incident) == 2:
            a, b = incident
            adjacency[a].add(b)
            adjacency[b].add(a)
    result = {}
    for edge in sorted(edges, key=lambda item: item["id"]):
        if edge["id"] in result:
            continue
        stack = [edge["id"]]
        members = []
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.append(current)
            stack.extend(sorted(adjacency[current] - set(members), reverse=True))
        stroke = "stroke_" + min(members)
        for member in members:
            result[member] = stroke
    return result


def _stroke_polylines(edges: list[dict]) -> dict[str, dict[str, Any]]:
    """Order each stroke's edges into one polyline through degree-2 nodes.

    Returns stroke_id -> {"points": [...], "edge_ids": [...], "line": LineString}.
    Stroke endpoints are serialized nodes whose degree is not 2 (real source
    junctions/termini); ordering starts at the smallest member edge id
    incident to such a node so the walk is deterministic.
    """
    strokes = _edge_strokes(edges)
    by_id = {e["id"]: e for e in edges}
    by_node: dict[str, list[str]] = {}
    for edge in edges:
        by_node.setdefault(edge["from_node"], []).append(edge["id"])
        by_node.setdefault(edge["to_node"], []).append(edge["id"])

    def other_node(edge: dict, node: str) -> str:
        return edge["to_node"] if edge["from_node"] == node else edge["from_node"]

    out: dict[str, dict[str, Any]] = {}
    for stroke_id in sorted(set(strokes.values())):
        members = sorted(eid for eid, sid in strokes.items() if sid == stroke_id)
        start = next((eid for eid in members
                      if len(by_node[by_id[eid]["from_node"]]) != 2
                      or len(by_node[by_id[eid]["to_node"]]) != 2), members[0])
        first = by_id[start]
        if len(by_node[first["from_node"]]) != 2:
            node, chain = first["from_node"], list(first["plan_polyline"])
        elif len(by_node[first["to_node"]]) != 2:
            node, chain = first["to_node"], list(reversed(first["plan_polyline"]))
        else:
            # Pure degree-2 loop: start at the smallest edge's from_node.
            node, chain = first["from_node"], list(first["plan_polyline"])
        order = [start]
        visited = {start}
        while True:
            exit_node = other_node(by_id[order[-1]], node)
            following = [eid for eid in sorted(by_node[exit_node])
                         if eid not in visited and strokes[eid] == stroke_id]
            if not following:
                break
            nxt = following[0]
            edge = by_id[nxt]
            pts = list(edge["plan_polyline"])
            if edge["to_node"] == exit_node and edge["from_node"] != exit_node:
                pts = list(reversed(pts))
            chain.extend(pts[1:])
            order.append(nxt)
            visited.add(nxt)
            node = exit_node
            if len(by_node[exit_node]) != 2:
                break
        out[stroke_id] = {"points": [[float(x), float(y)] for x, y in chain],
                          "edge_ids": order,
                          "line": LineString(chain)}
    return out


def _endpoint_covered(line: LineString, at_start: bool, ring: Polygon) -> bool:
    """Classify an endpoint, resolving boundary-only cases 1 GU inward."""
    distance = 0.0 if at_start else line.length
    point = line.interpolate(distance)
    if ring.boundary.distance(point) > 1e-7:
        return ring.covers(point)
    sample_distance = min(ENDPOINT_EPS_GU, line.length)
    sample = line.interpolate(sample_distance if at_start
                              else max(0.0, line.length - sample_distance))
    if ring.boundary.distance(sample) <= 1e-7:
        side = "start" if at_start else "end"
        raise TownLayoutError(f"R2 unresolved boundary endpoint tangency: {side}")
    return ring.covers(sample)


def build_ports(product: dict) -> dict[str, Any]:
    roads = list(product.get("aligned_roads", {}).get("edges", []))
    strokes = _edge_strokes(roads)
    stroke_data = _stroke_polylines(roads)
    city_region_patch_ids, boundary_retractions = _refine_city_region(
        product, stroke_data)
    ring, simplification = build_planning_ring(product, city_region_patch_ids)
    ring_poly = polygon_from_ring(ring)
    _union, patches = _ring_and_patches(product, city_region_patch_ids)
    water = unary_union([polygon_from_ring(r) for r in product.get("water_polygons", [])]) \
        if product.get("water_polygons") else None
    texture_overlaps = _road_texture_ring_overlaps(product, ring_poly)
    approaches = {item.get("source_edge_id"): item for item in product.get("approaches", [])}
    by_id = {e["id"]: e for e in roads}

    true_crossings: list[dict] = []
    internal_gaps: list[dict] = []

    for stroke_id in sorted(stroke_data):
        data = stroke_data[stroke_id]
        line = data["line"]
        start_covered = _endpoint_covered(line, True, ring_poly)
        end_covered = _endpoint_covered(line, False, ring_poly)
        stroke_crossings = []
        for point in _points(line.intersection(ring_poly.boundary)):
            try:
                tangent, outside_before = _crossing_tangent(line, point, ring_poly)
            except ValueError:
                continue
            outside = _outside_segment(line, point, outside_before)
            if outside.intersects(water) if water is not None else False:
                continuation = "continuation_bridge_dependent"
            elif outside.length < DRY_CHECK_GU:
                continuation = "short"
            else:
                continuation = "dry"
            source_edge = min(data["edge_ids"],
                              key=lambda eid: LineString(by_id[eid]["plan_polyline"]).distance(point))
            patch_id = min(patches, key=lambda pid: patches[pid].distance(point))
            stroke_crossings.append({
                "crossing_id": "",
                "source_edge_ids": data["edge_ids"],
                "source_stroke_id": stroke_id,
                "position": [float(point.x), float(point.y)],
                "ring_arc_gu": float(ring_poly.boundary.project(point)),
                "source_tangent": [float(tangent[0]), float(tangent[1])],
                "perimeter_patch_id": patch_id,
                "status": "valid",
                "continuation": continuation,
                "continuation_checked_length_gu": float(outside.length),
                "_outside": outside,
                "_outside_before": outside_before,
                "_projected_distance": float(line.project(point)),
                "_source_edge": source_edge,
            })

        stroke_crossings.sort(key=lambda item: item["_projected_distance"])
        if not stroke_crossings:
            continue
        for ordinal, crossing in enumerate(stroke_crossings):
            crossing["crossing_id"] = f"crossing_{crossing['_source_edge']}_{ordinal:02d}"

        retained_ids: set[str] = set()
        if not start_covered:
            candidate = next((item for item in stroke_crossings
                              if item["_outside_before"]), None)
            if candidate is None:
                raise TownLayoutError(
                    f"R2 external stroke start has no inward crossing: {stroke_id}")
            retained_ids.add(candidate["crossing_id"])
        if not end_covered:
            candidate = next((item for item in reversed(stroke_crossings)
                              if not item["_outside_before"]), None)
            if candidate is None:
                raise TownLayoutError(
                    f"R2 external stroke end has no inward crossing: {stroke_id}")
            retained_ids.add(candidate["crossing_id"])

        for crossing in stroke_crossings:
            if crossing["crossing_id"] in retained_ids:
                _texture_port(crossing, line, ring_poly, texture_overlaps,
                              patches, water)
                true_crossings.append(crossing)
            else:
                crossing["status"] = "internal_gap_crossing"
                crossing["reason"] = "bounded_outside_excursion"
                internal_gaps.append(crossing)

    # Merge nearby true crossings deterministically, but never merge crossings
    # belonging to the same stroke or diagnostic internal-gap records.
    true_crossings.sort(key=lambda item: (item["ring_arc_gu"],
                                          item["source_stroke_id"],
                                          item["crossing_id"]))
    retained: list[dict] = []
    for crossing in true_crossings:
        nearby = next((item for item in retained
                       if item["source_stroke_id"] != crossing["source_stroke_id"]
                       and abs(item["ring_arc_gu"] - crossing["ring_arc_gu"])
                       <= PORT_MERGE_ARC_GU), None)
        if nearby is None:
            crossing["merged_source_ids"] = []
            retained.append(crossing)
            continue
        width_of = lambda c: float(by_id[c["source_edge_ids"][0]]["width"].get("estimated_width_gu", 0.0))
        if (width_of(crossing), ) > (width_of(nearby), ) or (
                width_of(crossing) == width_of(nearby)
                and crossing["source_stroke_id"] < nearby["source_stroke_id"]):
            crossing["merged_source_ids"] = nearby["source_edge_ids"] + nearby.get("merged_source_ids", [])
            retained[retained.index(nearby)] = crossing
        else:
            nearby.setdefault("merged_source_ids", []).extend(crossing["source_edge_ids"])

    # Approaches with no true crossing stay explicit exclusions.
    excluded = []
    crossed_edges = {eid for c in true_crossings for eid in c["source_edge_ids"]}
    for edge_id, approach in sorted(approaches.items()):
        if edge_id not in crossed_edges:
            excluded.append({"crossing_id": f"crossing_{edge_id}",
                             "source_edge_ids": [edge_id],
                             "status": "excluded", "reason": "no_ring_crossing",
                             "source_stroke_id": strokes.get(edge_id)})

    # Build strict Stage A ports. Interior source subchains are
    # intentionally absent: they are report-only guides, not accepted roads.
    ports = []
    augmented_nodes = list(product.get("nodes", []))
    augmented_edges = list(product.get("boundary_edges", []))
    for index, crossing in enumerate(retained):
        port_id = f"port_{product['candidate_id']}_{index:04d}"
        port_node_id = f"node_{port_id}"
        augmented_nodes.append({"node_id": port_node_id,
                                "position": crossing["position"], "kind": "port"})
        ports.append({"port_id": port_id, "crossing_id": crossing["crossing_id"],
                      "position": crossing["position"], "ring_arc_gu": crossing["ring_arc_gu"],
                      "source_stroke_id": crossing["source_stroke_id"],
                      "source_tangent": crossing["source_tangent"],
                      "continuation": crossing["continuation"],
                      "port_basis": crossing.get("port_basis"),
                      "texture_port_shift_gu": crossing.get("texture_port_shift_gu", 0.0),
                      "vector_crossing_position": crossing.get("vector_crossing_position"),
                      "texture_opening_geometries": crossing.get(
                          "texture_opening_geometries", []),
                      "merged_source_ids": crossing.get("merged_source_ids", []),
                      "perimeter_patch_id": crossing["perimeter_patch_id"]})

    all_crossings = [
        *true_crossings,
        *internal_gaps,
        *excluded,
    ]
    private_keys = {"_outside", "_outside_before", "_projected_distance", "_source_edge"}
    all_crossings = [{k: v for k, v in crossing.items() if k not in private_keys}
                     for crossing in all_crossings]
    all_crossings.sort(key=lambda item: (float(item.get("ring_arc_gu", math.inf)),
                                         str(item.get("source_stroke_id", "")),
                                         str(item.get("crossing_id", ""))))
    regional = []
    retained_by_crossing = {port["crossing_id"]: port for port in ports}
    for crossing in retained:
        port = retained_by_crossing[crossing["crossing_id"]]
        outside = crossing["_outside"]
        status = ("retained" if crossing["continuation"] == "dry"
                  else "retained_bridge_dependent"
                  if crossing["continuation"] == "continuation_bridge_dependent"
                  else "retained_short_continuation")
        regional.append({
            "port_id": port["port_id"],
            "crossing_id": crossing["crossing_id"],
            "source_edge_id": crossing["_source_edge"],
            "source_stroke_id": crossing["source_stroke_id"],
            "polyline": [[float(x), float(y)] for x, y in outside.coords],
            "status": status,
            "reason": None,
        })
    return {"planning_ring": {"ring": ring, "simplification": simplification},
            "city_region_patch_ids": sorted(city_region_patch_ids),
            "boundary_retractions": boundary_retractions,
            "source_crossings": all_crossings, "ports": ports,
            "regional_outside_polylines": regional,
            "nodes": augmented_nodes, "edges": augmented_edges,
            "stage_metrics": {"retained_port_count": len(ports),
                              "internal_gap_crossing_count": len(internal_gaps),
                              "excluded_crossing_count": len(excluded),
                              "bridge_dependent_count": sum(
                                  c.get("continuation") == "continuation_bridge_dependent"
                                  for c in retained),
                              "texture_port_count": sum(
                                   c.get("port_basis") == "configured_road_vtex_ring_overlap"
                                  for c in retained),
                              "source_junction_count": 0,
                              "boundary_retraction_count": len(boundary_retractions),
                              "ring_length_gu": ring_poly.length},
            "reports": [{"stage": "r2", "status": "ok",
                         "message": (f"ports={len(ports)} crossings={len(all_crossings)} "
                                     f"internal_gaps={len(internal_gaps)}")}]}
