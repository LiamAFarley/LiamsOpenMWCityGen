"""Pure Phase 4 roof-patch and observed dormer relation extraction.

This module is the host-side half of the Phase 4 pipeline.  It consumes only
evaluated triangle evidence exported by Blender; Blender is not involved in
selection, fitting, polygon union, or relation eligibility.  All tolerances
are supplied by the caller's JSON measurement mapping.  Shapely is used for
finite triangle unions so holes and disconnected pieces remain explicit.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
from shapely.ops import unary_union


def _v(a, b): return tuple(float(a[i]) - float(b[i]) for i in range(3))
def _add(a, b): return tuple(float(a[i]) + float(b[i]) for i in range(3))
def _mul(a, s): return tuple(float(x) * float(s) for x in a)
def _dot(a, b): return sum(float(a[i]) * float(b[i]) for i in range(3))
def _cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def _norm(a):
    d = math.sqrt(_dot(a, a))
    return _mul(a, 1.0 / d) if d else (0.0, 0.0, 0.0)
def _angle(a, b): return math.degrees(math.acos(max(-1.0, min(1.0, _dot(a, b)))))
def _key(v, weld): return tuple(int(round(float(x) / weld)) for x in v)


def _candidate_triangles(triangles, bounds, cfg):
    lo, hi = bounds["min"][2], bounds["max"][2]
    floor = lo + float(cfg["roof_floor_fraction"]) * (hi - lo)
    nlo, nhi = float(cfg["upward_normal_z_min"]), float(cfg["upward_normal_z_max"])
    return [dict(t) for t in triangles if nlo <= float(t["normal"][2]) <= nhi
            and float(t["centroid"][2]) >= floor
            and float(t["area"]) >= float(cfg["candidate_min_area_gu2"])]


def _supported(candidates, all_triangles, cfg):
    radius = float(cfg["support_radius_gu"]); gap_hi = float(cfg["support_max_vertical_gap_gu"])
    gap_lo = float(cfg["support_min_vertical_gap_gu"])
    result = []
    candidate_signatures = {tuple(tuple(float(x) for x in v) for v in t["verts"]) for t in candidates}
    for t in candidates:
        c = t["centroid"]; supported = 0; area = 0.0; nearest = None
        support_ids = set()
        tx = [p[0] for p in t["verts"]]; ty = [p[1] for p in t["verts"]]
        for lower_index, lower in enumerate(all_triangles):
            if tuple(tuple(float(x) for x in v) for v in lower["verts"]) in candidate_signatures: continue
            lc = lower["centroid"]; dz = float(c[2]) - float(lc[2])
            if dz < gap_lo or dz > gap_hi: continue
            lx = [p[0] for p in lower["verts"]]; ly = [p[1] for p in lower["verts"]]
            dx = max(min(tx)-max(lx), min(lx)-max(tx), min(ty)-max(ly), min(ly)-max(ty), 0.0)
            # The max expression above is an XY AABB separation, not a ground AABB.
            if dx <= radius:
                supported += 1; area += float(lower["area"]); support_ids.add(lower_index)
                nearest = dz if nearest is None else min(nearest, dz)
        result.append((support_ids, area, nearest))
    return result


def _groups(candidates, cfg):
    weld = float(cfg["vertex_weld_gu"]); ang = float(cfg["component_normal_tolerance_deg"])
    by_vertex = {}
    for i, t in enumerate(candidates):
        for v in t["verts"]: by_vertex.setdefault(_key(v, weld), []).append(i)
    adjacent = [set() for _ in candidates]
    for values in by_vertex.values():
        for i in values:
            for j in values:
                if i != j and _angle(candidates[i]["normal"], candidates[j]["normal"]) <= ang:
                    adjacent[i].add(j)
    # A second BFS applies intercept/normal splitting while connected; it does
    # not globally bucket normals and therefore preserves separate wings.
    unvisited = set(range(len(candidates))); groups = []
    off_tol = float(cfg["plane_offset_tolerance_gu"])
    while unvisited:
        seed = min(unvisited); unvisited.remove(seed); q = deque([seed]); group = [seed]
        sn = _norm(candidates[seed]["normal"]); so = _dot(sn, candidates[seed]["centroid"])
        while q:
            i = q.popleft()
            for j in sorted(adjacent[i] & unvisited):
                nj = _norm(candidates[j]["normal"]); oj = _dot(sn, candidates[j]["centroid"])
                if _angle(sn, nj) <= ang and abs(so - oj) <= off_tol:
                    unvisited.remove(j); q.append(j); group.append(j)
        groups.append(group)
    return groups


def _boundary_edges(group, candidates, weld):
    """Return unique component boundary edges in deterministic endpoint order."""

    counts = {}
    representatives = {}
    for index in group:
        verts = candidates[index]["verts"]
        for a, b in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            ka, kb = _key(a, weld), _key(b, weld)
            if ka == kb:
                continue
            edge_key = tuple(sorted((ka, kb)))
            counts[edge_key] = counts.get(edge_key, 0) + 1
            representatives.setdefault(edge_key, (tuple(a), tuple(b)))
    result = []
    for edge_key, count in counts.items():
        if count != 1:
            continue
        a, b = representatives[edge_key]
        if tuple(a) > tuple(b):
            a, b = b, a
        result.append((a, b))
    return sorted(result)


def _ring(points):
    coordinates = list(points.coords)
    return [[round(float(x), 6), round(float(y), 6)] for x, y in coordinates[:-1]]


def _classify_boundaries(polygons, cfg):
    """Label exterior and hole edges without erasing hole topology."""

    near_slope = float(cfg["boundary_near_horizontal_slope"])
    position_tolerance = float(cfg["boundary_position_tolerance_gu"])
    outer_v_values = [float(y) for polygon in polygons for x, y in polygon.exterior.coords[:-1]]
    v_min = min(outer_v_values, default=0.0)
    v_max = max(outer_v_values, default=0.0)
    result = []
    for polygon in polygons:
        loops = [("exterior", polygon.exterior)] + [("hole", ring) for ring in polygon.interiors]
        for loop_kind, ring in loops:
            coords = list(ring.coords)
            for a, b in zip(coords, coords[1:]):
                du = abs(float(b[0]) - float(a[0]))
                dv = abs(float(b[1]) - float(a[1]))
                length = math.hypot(du, dv)
                near_horizontal = dv <= near_slope * max(length, 1.0)
                midpoint_v = (float(a[1]) + float(b[1])) / 2.0
                if loop_kind == "hole":
                    kind = "valley" if near_horizontal else "unresolved"
                elif near_horizontal and abs(midpoint_v - v_min) <= position_tolerance:
                    kind = "eave"
                elif near_horizontal and abs(midpoint_v - v_max) <= position_tolerance:
                    kind = "ridge"
                elif dv > du:
                    kind = "gable"
                else:
                    kind = "unresolved"
                result.append({
                    "loop_kind": loop_kind,
                    "start_uv": [round(float(a[0]), 6), round(float(a[1]), 6)],
                    "end_uv": [round(float(b[0]), 6), round(float(b[1]), 6)],
                    "classification": kind,
                })
    return result


def _frame_and_union(group, candidates, cfg):
    area = sum(float(candidates[i]["area"]) for i in group)
    n = _norm(tuple(sum(float(candidates[i]["normal"][k]) * float(candidates[i]["area"]) for i in group) for k in range(3)))
    if n[2] < 0:
        n = _mul(n, -1.0)
    offset = sum(_dot(n, candidates[i]["centroid"]) * float(candidates[i]["area"]) for i in group) / area
    points = [v for i in group for v in candidates[i]["verts"]]
    residuals = [abs(_dot(n, p) - offset) for p in points]

    edges = _boundary_edges(group, candidates, float(cfg["vertex_weld_gu"]))
    near_slope = float(cfg["boundary_near_horizontal_slope"])
    edge_candidates = []
    for a, b in edges:
        delta = _v(b, a)
        horizontal = math.hypot(delta[0], delta[1])
        length = math.sqrt(_dot(delta, delta))
        if horizontal > 1e-9:
            edge_candidates.append((abs(delta[2]) / horizontal, -length, a, b))
    near_edges = [edge for edge in edge_candidates if edge[0] <= near_slope]
    chosen_pool = near_edges or edge_candidates
    if chosen_pool:
        _, _, a, b = min(chosen_pool)
        u = _norm((_v(b, a)[0], _v(b, a)[1], 0.0))
        eave_selection = "measured_boundary_edge" if near_edges else "unresolved"
    else:
        u = _norm((-n[1], n[0], 0.0))
        eave_selection = "unresolved"
    if abs(u[0]) < 1e-12 and abs(u[1]) < 1e-12:
        u = _norm((-n[1], n[0], 0.0))
        eave_selection = "unresolved"
    v = _cross(n, u)
    if v[2] < 0:
        u, v = _mul(u, -1.0), _mul(v, -1.0)
    origin = _mul(n, offset)

    projected = []
    for i in group:
        polygon = Polygon([(_dot(p, u), _dot(p, v)) for p in candidates[i]["verts"]])
        if not polygon.is_empty and polygon.area > 0:
            projected.append(polygon)
    union = unary_union(projected).buffer(0)
    polygons = list(union.geoms) if isinstance(union, MultiPolygon) else ([union] if not union.is_empty else [])
    polygons.sort(key=lambda p: (-p.area, p.bounds))
    inset = float(cfg["roof_inset_gu"])
    usable = union.buffer(-inset)
    eligible = bool(polygons) and float(union.area) >= float(cfg["min_patch_area_gu2"])
    max_res = max(residuals, default=float("inf"))
    rms = math.sqrt(sum(x * x for x in residuals) / len(residuals)) if residuals else float("inf")
    if max_res > float(cfg["max_plane_fit_residual_gu"]):
        eligible = False
    if usable.is_empty or not usable.is_valid:
        eligible = False
    pieces = [{"outer": _ring(p.exterior), "holes": [_ring(h) for h in p.interiors]} for p in polygons]
    usable_pieces = []
    if not usable.is_empty:
        inset_polygons = list(usable.geoms) if isinstance(usable, MultiPolygon) else [usable]
        usable_pieces = [{"outer": _ring(p.exterior), "holes": [_ring(h) for h in p.interiors]} for p in inset_polygons]
    return {
        "n": list(map(float, n)),
        "u": list(map(float, u)),
        "v": list(map(float, v)),
        "plane_offset_gu": offset,
        "origin_gu": list(origin),
        "eave_selection": eave_selection,
        "polygon_pieces_uv": pieces,
        "usable_region_uv": usable_pieces,
        "boundary_segments": _classify_boundaries(polygons, cfg),
        "triangle_indices": sorted(group),
        "triangle_count": len(group),
        "area_gu2": float(union.area),
        "support": None,
        "fit": {"max_abs_residual_gu": max_res, "rms_abs_residual_gu": rms},
        "status": "eligible" if eligible else "ineligible",
    }


def extract_roof_profile(model_key: str, triangles: Sequence[Mapping[str, Any]], bounds: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract deterministic finite roof patches from one evidence row."""
    candidates = _candidate_triangles(triangles, bounds, config)
    support = _supported(candidates, triangles, config)
    groups = _groups(candidates, config)
    patches = []
    for index, group in enumerate(groups, 1):
        patch = _frame_and_union(group, candidates, config)
        vals = [support[i] for i in group]
        supported_ids = set().union(*(x[0] for x in vals)); supported = len(supported_ids); total = sum(float(triangles[j]["area"]) for j in supported_ids)
        patch["support"] = {"supported_triangle_count": supported, "supported_area_gu2": total,
                             "support_fraction": supported / max(len(group), 1),
                             "nearest_support_gap_gu": min((x[2] for x in vals if x[2] is not None), default=None),
                             "rejected_unsupported_count": sum(not x[0] for x in vals)}
        if patch["support"]["support_fraction"] < float(config["min_component_support_fraction"]): patch["status"] = "ineligible"
        patch["patch_id"] = f"r{index:03d}"
        patches.append(patch)
    patches.sort(key=lambda p: (-p["area_gu2"], p["patch_id"]))
    for i, p in enumerate(patches, 1): p["patch_id"] = f"r{i:03d}"
    return {"model_key": model_key, "candidate_triangle_count": len(candidates), "patch_count": len(patches), "patches": patches,
            "status": "eligible" if any(p["status"] == "eligible" for p in patches) else "ineligible"}


def _matrix(rotation):
    from procgen.engine_transform import tes3_euler_to_matrix
    return np.asarray(tes3_euler_to_matrix(rotation), dtype=float)


def _uv_region(pieces):
    polygons = []
    for piece in pieces:
        outer = piece.get("outer", [])
        holes = piece.get("holes", [])
        if len(outer) >= 3:
            polygons.append(Polygon(outer, holes))
    return unary_union(polygons) if polygons else Polygon()


def build_dormer_relation(site_id: str, stamp: Mapping[str, Any], shell: Mapping[str, Any], dormer: Mapping[str, Any], roof: Mapping[str, Any], dormer_triangles: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Measure one explicit shell-attachment dormer relation; fail closed."""
    edges = stamp.get("shell_attachment_edges", [])
    if not any({e.get("ref_a"), e.get("ref_b")} == {shell.get("source_id"), dormer.get("source_id")} for e in edges):
        raise ValueError("relation_missing_shell_attachment_edge")
    eligible = [p for p in roof.get("patches", []) if p.get("status") == "eligible"]
    if not eligible: raise ValueError("relation_no_eligible_roof_patch")
    rs, rd = _matrix(shell.get("rotation")), _matrix(dormer.get("rotation"))
    shell_scale = float(shell["scale"]); dormer_scale = float(dormer["scale"])
    origin = np.linalg.inv(rs) @ (np.asarray(dormer["offset_gu"], float)-np.asarray(shell["offset_gu"], float)) / shell_scale
    dverts = [(rs.T @ (np.asarray(dormer["offset_gu"],float)+rd @ (dormer_scale*np.asarray(v,float))-np.asarray(shell["offset_gu"],float))/shell_scale) for t in dormer_triangles for v in t["verts"]]
    best = None
    for p in eligible:
        n,u,v = map(np.asarray, (p["n"], p["u"], p["v"]))
        po = float(p["plane_offset_gu"]); uv = [(float(np.dot(x,u)), float(np.dot(x,v))) for x in dverts]
        region = _uv_region(p["polygon_pieces_uv"])
        projected_hull = MultiPoint(uv).convex_hull if uv else Polygon()
        overlap_area = float(region.intersection(projected_hull).area) if not projected_hull.is_empty else 0.0
        overlap = sum(1 for x in uv if region.buffer(float(config["contact_projected_distance_gu"])).contains(Point(x)))
        plane = [float(np.dot(x,n)-po) for x in dverts]; near = min(abs(x) for x in plane) if plane else float("inf")
        footprint_distance = float(region.distance(projected_hull)) if not projected_hull.is_empty else float("inf")
        score = near + footprint_distance
        if (overlap_area > 0.0 or overlap) and near <= float(config["contact_plane_tolerance_gu"]):
            if best is not None and abs(score-best[0]) < 1e-9: raise ValueError("relation_contact_tie")
            best = (score, p, overlap, near, plane, overlap_area, uv)
    if best is None: raise ValueError("relation_no_contact_evidence")
    _, p, overlap, near, plane, overlap_area, uv = best; n,u,v = map(np.asarray, (p["n"],p["u"],p["v"]))
    coords = {"u_along_eave": float(np.dot(origin,u)), "v_up_slope": float(np.dot(origin,v)), "n_sink": float(np.dot(origin,n)-p["plane_offset_gu"])}
    reconstructed = u*coords["u_along_eave"] + v*coords["v_up_slope"] + n*(p["plane_offset_gu"]+coords["n_sink"])
    frame = np.column_stack((u, v, n))
    shell_to_dormer = rs.T @ rd
    roof_to_dormer = frame.T @ shell_to_dormer
    reconstructed_shell_to_dormer = frame @ roof_to_dormer
    rotation_residual = float(np.max(np.abs(reconstructed_shell_to_dormer - shell_to_dormer)))
    position_residual = float(np.max(np.abs(reconstructed-origin)))
    eligible_relation = position_residual <= float(config["position_roundtrip_tolerance_gu"]) and rotation_residual <= float(config["rotation_roundtrip_tolerance"])
    member_by_id = {str(member.get("source_id")): member for member in stamp.get("members", [])}
    child_windows = set()
    for edge in stamp.get("member_contact_edges", []):
        refs = {str(edge.get("ref_a")), str(edge.get("ref_b"))}
        if str(dormer["source_id"]) not in refs:
            continue
        other = next(iter(refs - {str(dormer["source_id"])}), None)
        if other is not None and str(member_by_id.get(other, {}).get("structural_role", "")).casefold() == "window":
            child_windows.add(other)
    usable_region = _uv_region(p.get("usable_region_uv", []))
    projected_points = [Point(x) for x in uv]
    min_usable_distance = min((float(usable_region.distance(point)) for point in projected_points), default=float("inf"))
    usable_count = sum(usable_region.covers(point) for point in projected_points)
    return {
        "site_id": site_id,
        "source_stamp_id": stamp["stamp_id"],
        "shell_member_source_id": shell["source_id"],
        "roof_patch_id": p["patch_id"],
        "dormer_member_source_id": dormer["source_id"],
        "child_window_witness_ids": sorted(child_windows),
        "dormer_model_key": dormer["model_key"],
        "dormer_authored_scale": dormer_scale,
        "roof_frame_coordinates_gu": coords,
        "dormer_origin_shell_frame_gu": origin.tolist(),
        "dormer_orientation_relative_to_roof_frame": roof_to_dormer.tolist(),
        "contact_evidence": {
            "projected_vertex_count": overlap,
            "projected_overlap_area_gu2": overlap_area,
            "nearest_plane_distance_gu": near,
            "signed_plane_distances_gu": plane,
        },
        "clearance_evidence": {
            "projected_overlap_area_gu2": overlap_area,
            "projected_min_distance_to_usable_region_gu": min_usable_distance,
            "projected_vertices_inside_usable_region": usable_count,
            "signed_plane_min_gu": min(plane, default=None),
            "signed_plane_max_gu": max(plane, default=None),
        },
        "roundtrip": {"position_residual_gu": position_residual, "rotation_matrix_residual": rotation_residual},
        "evidence_class": "observed_exact" if eligible_relation else "ineligible",
    }
