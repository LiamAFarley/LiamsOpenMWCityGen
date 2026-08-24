"""Select the central inner city and derive its arterial-only wall gates.

This phase sits between arterial-safe block formation and minor-road growth.
It grows one contiguous set of existing Voronoi patches outward from the
accepted arterial meeting until roughly two thirds of city land is enclosed.
The union boundary is the wall authority. Only exact intersections of the
accepted arterial centerlines with that boundary become gates; later minor
roads may meet but never cross the wall.

Input: ``r2b_road_blocks``. Output: ``r2w_inner_wall`` with selected patch IDs,
wall polygon/centerline, arterial gate records, area fraction, and reports.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .geometry import normalize_ring, polygon_from_ring
from .validate import TownLayoutError
from procgen.wall_compose import WallComposeError, compose_city_wall
from procgen.wall_kit import WallKitError, load_kit, validate_kit

TARGET_FRACTION = 2.0 / 3.0


def _ring(poly) -> list[list[float]]:
    return normalize_ring([[float(x), float(y)] for x, y in poly.exterior.coords])["ring"]


def _points(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type == "MultiPoint":
        return list(geometry.geoms)
    result = []
    for geom in getattr(geometry, "geoms", []):
        result.extend(_points(geom))
    return result


def _first_crossing(stroke: LineString, wall, port_position) -> Point | None:
    hits = _points(stroke.intersection(wall))
    if not hits:
        return None
    port = Point(port_position)
    return min(hits, key=lambda point: (stroke.project(point), point.distance(port), point.x, point.y))


def _workspace_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


def _load_wall_profile(product: dict[str, Any]) -> tuple[dict, dict] | None:
    brief_ref = ((product.get("identities") or {}).get("brief") or {}).get("path")
    if not brief_ref:
        return None
    brief = json.loads(_workspace_path(brief_ref).read_text(encoding="utf-8"))
    profile_ref = (brief.get("fortification") or {}).get("wall_profile")
    if not profile_ref:
        return None
    profile_path = _workspace_path(profile_ref)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    kit_path = _workspace_path(profile["kit_json"])
    kit = load_kit(kit_path)
    rules = dict(kit["rules"])
    if "allowed_fill_piece_ids" in profile:
        rules["allowed_fill_piece_ids"] = list(profile["allowed_fill_piece_ids"])
    kit = dict(kit)
    kit["rules"] = rules
    validate_kit(kit)
    profile = dict(profile)
    profile["profile_path"] = str(profile_path)
    profile["kit_path"] = str(kit_path)
    return profile, kit


def _fit_ring_to_profile(ring: list[list[float]], gates: list[dict], profile: dict, kit: dict):
    """Simplify unprotected gate arcs and prove the selected kit can compose."""
    source = list(ring)
    if source[0] == source[-1]:
        source.pop()
    source_line = LineString(source + [source[0]])
    if not source_line.is_ring:
        raise TownLayoutError("wall fit source ring is not closed")
    gate_arcs = sorted(
        (source_line.project(Point(gate["position"])), gate) for gate in gates
    )
    if not gate_arcs:
        raise TownLayoutError("wall fit requires at least one protected gate")
    tolerance = float(profile["simplify_tolerance_gu"])
    boundaries = [arc for arc, _ in gate_arcs]
    pieces: list[list[float]] = []
    for index, start_arc in enumerate(boundaries):
        end_arc = boundaries[(index + 1) % len(boundaries)]
        if end_arc <= start_arc:
            end_arc += source_line.length
        vertices = []
        for vertex in source[1:] + source[:1]:
            arc = source_line.project(Point(vertex))
            if arc <= start_arc:
                arc += source_line.length
            if start_arc < arc < end_arc:
                vertices.append((arc, Point(vertex)))
        points = [source_line.interpolate(start_arc)]
        points.extend(point for _arc, point in sorted(vertices, key=lambda row: row[0]))
        points.append(source_line.interpolate(end_arc % source_line.length))
        simplified = LineString([(p.x, p.y) for p in points]).simplify(
            tolerance, preserve_topology=False
        )
        coords = [[float(x), float(y)] for x, y in simplified.coords]
        if not pieces:
            pieces.extend(coords)
        else:
            pieces.extend(coords[1:])
    if pieces and pieces[0] == pieces[-1]:
        pieces.pop()
    fitted = normalize_ring(pieces)["ring"]
    polygon = polygon_from_ring(fitted)
    if not polygon.is_valid or polygon.interiors or polygon.area <= 0.0:
        raise TownLayoutError("wall fit produced invalid polygon")
    max_move = max(
        Point(point).distance(source_line) for point in fitted
    )
    if max_move > float(profile["max_vertex_move_gu"]):
        raise TownLayoutError("wall fit exceeded max_vertex_move_gu")
    # normalize_ring returns an open ring. Do not drop the final real vertex;
    # doing so makes the probe wall differ from the wall composed downstream.
    path_points = [(float(x), float(y)) for x, y in fitted]
    compose_gates = [
        {
            "position_xy": list(gate["position"]),
            "heading_deg": math.degrees(math.atan2(
                float(gate["arterial_tangent"][1]),
                float(gate["arterial_tangent"][0]),
            )),
        }
        for gate in gates
    ]
    try:
        probe = compose_city_wall(
            path_points, compose_gates, lambda _x, _y: 0.0, kit,
            stamp_id="wall_fit_probe",
        )
    except (WallComposeError, WallKitError) as exc:
        raise TownLayoutError(f"wall_fit_infeasible: {exc}") from exc
    fill_ids = sorted({
        member["piece_id"] for member in probe["members"]
        if member["structural_role"] == "straight"
    })
    if fill_ids != sorted(profile["allowed_fill_piece_ids"]):
        raise TownLayoutError(f"wall fit selected disallowed fill pieces: {fill_ids}")
    probe_origin = [float(v) for v in probe["origin_gu"]]
    gate_reserves = []
    mesh_footprints = []
    for member in probe["members"]:
        footprint = member.get("footprint_xy_rel")
        if not footprint or len(footprint) < 3:
            continue
        absolute = [
            [probe_origin[0] + float(point[0]), probe_origin[1] + float(point[1])]
            for point in footprint
        ]
        mesh_footprints.append(absolute)
        if member["structural_role"] == "gatehouse" or "gate_side" in member.get("meta", {}):
            gate_reserves.append(absolute)
    return fitted, {
        "profile_version": profile["profile_version"],
        "profile_path": profile["profile_path"],
        "kit_path": profile["kit_path"],
        "centerline": fitted,
        "allowed_fill_piece_ids": list(profile["allowed_fill_piece_ids"]),
        "selected_fill_piece_ids": fill_ids,
        "member_count": len(probe["members"]),
        "tower_count": sum(m["structural_role"] == "tower" for m in probe["members"]),
        "gate_count": sum(m["structural_role"] == "gatehouse" for m in probe["members"]),
        "gate_reserves": gate_reserves,
        "mesh_footprints": mesh_footprints,
        "reserve_clearance_gu": float(profile["reserve_clearance_gu"]),
    }


def build_inner_wall(product: dict[str, Any], target_fraction: float = TARGET_FRACTION) -> dict[str, Any]:
    started = time.perf_counter()
    if product.get("stage_id") != "r2b_road_blocks":
        raise TownLayoutError("W input: expected r2b_road_blocks")
    patches = {p["patch_id"]: p for p in product.get("patches") or [] if p.get("inside_city")}
    if not patches:
        raise TownLayoutError("W input: no city patches")
    polygons = {pid: polygon_from_ring(row["polygon"]) for pid, row in patches.items()}
    city_land = unary_union(list(polygons.values()))
    meeting = Point((product.get("arterial_meeting") or {}).get("position") or [])
    if meeting.is_empty:
        raise TownLayoutError("W input: arterial meeting missing")
    start = next((pid for pid in sorted(patches) if polygons[pid].covers(meeting)), None)
    if start is None:
        start = min(patches, key=lambda pid: (polygons[pid].distance(meeting), pid))

    # Use exact shared-boundary adjacency instead of neighbour metadata so
    # cropped shoreline cells and promoted fringe patches remain consistent.
    adjacency = {pid: set() for pid in patches}
    ids = sorted(patches)
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            if polygons[left].boundary.intersection(polygons[right].boundary).length >= 96.0:
                adjacency[left].add(right)
                adjacency[right].add(left)

    arterial_centerlines = unary_union([LineString(s["geometry"])
                                        for s in product.get("smoothed_strokes") or []])
    arterial_patches = {pid for pid, poly in polygons.items()
                        if poly.intersection(arterial_centerlines).length >= 96.0}

    def components(members: set[str]) -> list[set[str]]:
        unseen = set(members)
        rows = []
        while unseen:
            seed = min(unseen)
            stack, row = [seed], set()
            unseen.remove(seed)
            while stack:
                node = stack.pop()
                row.add(node)
                for other in adjacency[node] & unseen:
                    unseen.remove(other)
                    stack.append(other)
            rows.append(row)
        return rows

    # Erode the outskirts inward from arterial-bearing outer patches. Every
    # excluded component therefore retains a route to a real arterial instead
    # of becoming a marooned lobe behind the wall.
    target_area = city_land.area * float(target_fraction)
    removed: set[str] = set()
    for port in product.get("ports") or []:
        point = Point(port["position"])
        port_patch = min(patches, key=lambda pid: (polygons[pid].distance(point), pid))
        removed.add(port_patch)
    selected = set(patches) - removed
    if start not in selected or len(components(selected)) != 1:
        raise TownLayoutError("W selection: boundary-port seeds disconnect central city")
    selected_union = unary_union([polygons[item] for item in sorted(selected)])
    if selected_union.geom_type != "Polygon" or selected_union.interiors:
        raise TownLayoutError("W selection: boundary-port seeds make invalid wall")
    while selected_union.area > target_area:
        choices = []
        for pid in sorted(selected - {start}):
            initially_outer = polygons[pid].boundary.intersection(city_land.boundary).length >= 96.0
            if not initially_outer and not (adjacency[pid] & removed):
                continue
            trial_selected = selected - {pid}
            trial_removed = removed | {pid}
            if len(components(trial_selected)) != 1:
                continue
            if any(not (component & arterial_patches) for component in components(trial_removed)):
                continue
            candidate = unary_union([polygons[item] for item in sorted(trial_selected)])
            if candidate.geom_type != "Polygon" or candidate.interiors or not candidate.is_valid:
                continue
            remaining_error = abs(candidate.area - target_area) / target_area
            compactness = 4.0 * math.pi * candidate.area / (candidate.length * candidate.length)
            distance = polygons[pid].centroid.distance(meeting)
            # Prefer distant erosion, then compact retained walls, while area
            # error keeps the final step near the requested two-thirds target.
            score = remaining_error + (1.0 - compactness) - distance / max(city_land.length, 1.0)
            choices.append((round(score, 9), pid, candidate))
        if not choices:
            break
        _, pid, candidate = min(choices, key=lambda row: (row[0], row[1]))
        # Do not take a step that moves farther from the requested fraction.
        if abs(candidate.area - target_area) > abs(selected_union.area - target_area):
            break
        selected.remove(pid)
        removed.add(pid)
        selected_union = candidate


    wall = selected_union.boundary
    port_by_id = {p["port_id"]: p for p in product.get("ports") or []}
    gates = []
    for stroke in sorted(product.get("smoothed_strokes") or [], key=lambda s: s["stroke_id"]):
        port_id = stroke.get("port_id")
        if port_id not in port_by_id:
            continue
        line = LineString(stroke["geometry"])
        hit = _first_crossing(line, wall, port_by_id[port_id]["position"])
        if hit is None:
            raise TownLayoutError(f"W gate: arterial {port_id} does not cross inner wall")
        # Tangent is sampled on the accepted centerline, not inferred from the
        # wall or the old source-road vector.
        d = line.project(hit)
        before = line.interpolate(max(0.0, d - 64.0))
        after = line.interpolate(min(line.length, d + 64.0))
        norm = math.hypot(after.x - before.x, after.y - before.y) or 1.0
        tangent = [(after.x - before.x) / norm, (after.y - before.y) / norm]
        gates.append({"gate_id": f"wall_gate_{len(gates):02d}", "port_id": port_id,
                      "position": [float(hit.x), float(hit.y)],
                      "arterial_tangent": tangent, "opening_width_gu": 512.0})
    # Shared arterial trunks may yield coincident crossings. They represent one
    # physical gate with multiple route provenance records.
    physical = []
    for gate in gates:
        existing = next((row for row in physical
                         if Point(row["position"]).distance(Point(gate["position"])) <= 32.0), None)
        if existing:
            existing["port_ids"].append(gate["port_id"])
        else:
            physical.append({"gate_id": f"wall_gate_{len(physical):02d}",
                             "position": gate["position"],
                             "arterial_tangent": gate["arterial_tangent"],
                             "opening_width_gu": 512.0,
                             "port_ids": [gate["port_id"]]})

    wall_fit = None
    fitted_ring = _ring(selected_union)
    profile_kit = _load_wall_profile(product)
    if profile_kit is not None:
        profile, kit = profile_kit
        fitted_ring, wall_fit = _fit_ring_to_profile(fitted_ring, physical, profile, kit)

    runtime = time.perf_counter() - started
    metrics = {"runtime_s": runtime, "target_fraction": float(target_fraction),
               "actual_fraction": float(selected_union.area / city_land.area),
               "selected_patch_count": len(selected), "city_patch_count": len(patches),
               "gate_count": len(physical), "wall_length_gu": float(wall.length)}
    result = dict(product)
    result.update({"stage_id": "r2w_inner_wall", "preceding_checkpoint": None,
                   "inner_wall": {"polygon": fitted_ring,
                                  "centerline": fitted_ring,
                                  "selected_patch_ids": sorted(selected),
                                  "target_fraction": float(target_fraction),
                                  "actual_fraction": metrics["actual_fraction"]},
                   "wall_gates": physical, "wall_fit": wall_fit, "metrics": metrics,
                   "reports": list(product.get("reports") or []) + [{
                       "stage": "r2w_inner_wall", "status": "ok",
                       "message": f"inner={metrics['actual_fraction']:.3f} gates={len(physical)}",
                   }]})
    return result
