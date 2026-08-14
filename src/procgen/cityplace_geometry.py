"""Terrain, footprint, collision, and road-access checks for Cityforge T1.2.

Pipeline position
------------------
The solver orchestrator resolves a T1.1 lot and produces matrix-space member
transforms; this module measures the transformed footprint against the accepted
site masks/terrain and against other emitted footprints.  It deliberately does
not select stamps, compose member rotations, write TES3 records, or move a lot
to improve access.

Geometry domain
---------------
Footprints are the exact D-STAMP convex hulls, transformed with the same
authoritative world-yaw matrix used for member offsets.  Strict overlap *and
contact* are hard conflicts.  Dispatch-5 gap distributions and the survey
``min_building_gap_gu`` value are reported only as guidance warnings.  Source
triangle surfaces and member AABBs are not present in the accepted D-STAMP
libraries, so the module emits an explicit ``fine_collision_deferred`` ledger
entry rather than claiming a fine check.

Terrain measurements use every covered 128-GU field node, hull vertices, edge
midpoints, and member/door XY samples.  Bilinear height, analytic normal, and
slope values come from :class:`procgen.cityplace_contracts.TerrainField`.
Coverage failure is never filled with a nearest or flat value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import cityplan
from .cityplace_contracts import (
    FieldCoverageError,
    PlacementConfig,
    TerrainField,
    TerrainSample,
    WATER_LEVEL_GU,
)
from .cityplace_transform import PlacedMember, world_yaw_matrix


SITE_SPAN_GU = 57344.0
ROAD_HEADING_WARN_DEG = 90.0


def _point_tuple(value: Sequence[float]) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


def transform_hull(
    hull_rel_xy: Sequence[Sequence[float]],
    *,
    anchor_xy_plan_gu: Sequence[float],
    yaw_deg: float,
) -> list[tuple[float, float]]:
    """Transform an exact stamp hull with the production engine yaw matrix."""

    anchor = np.asarray([float(anchor_xy_plan_gu[0]), float(anchor_xy_plan_gu[1]), 0.0])
    matrix = world_yaw_matrix(yaw_deg)
    out: list[tuple[float, float]] = []
    for point in hull_rel_xy:
        relative = np.asarray([float(point[0]), float(point[1]), 0.0])
        transformed = anchor + matrix @ relative
        out.append((float(transformed[0]), float(transformed[1])))
    return out


def _closed(points: Sequence[Sequence[float] | tuple[float, float]]) -> list[tuple[float, float]]:
    result = [_point_tuple(point) for point in points]
    if result and result[0] != result[-1]:
        result.append(result[0])
    return result


def _edge_midpoints(ring: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    closed = _closed(ring)
    return [
        ((closed[i][0] + closed[i + 1][0]) * 0.5,
         (closed[i][1] + closed[i + 1][1]) * 0.5)
        for i in range(len(closed) - 1)
    ]


def _sample_unique(
    field: TerrainField,
    points: Iterable[tuple[float, float, str]],
    *,
    seen_digits: int = 9,
) -> tuple[list[TerrainSample], list[dict[str, Any]]]:
    samples: list[TerrainSample] = []
    missing: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for x, y, kind in points:
        key = (round(float(x), seen_digits), round(float(y), seen_digits))
        if key in seen:
            continue
        seen.add(key)
        try:
            samples.append(field.sample(float(x), float(y)))
        except FieldCoverageError as exc:
            missing.append({
                "position_plan_gu": [float(x), float(y)],
                "kind": kind,
                "reason": str(exc),
            })
    return samples, missing


def sample_footprint(
    field: TerrainField,
    hull_plan_xy: Sequence[Sequence[float] | tuple[float, float]],
    *,
    member_positions_plan: Iterable[tuple[float, float]],
    door_positions_plan: Iterable[tuple[float, float]],
) -> tuple[list[TerrainSample], list[dict[str, Any]]]:
    """Sample full footprint coverage plus doors/members on the field."""

    ring = _closed(hull_plan_xy)
    if len(ring) < 4:
        return [], [{"kind": "footprint", "reason": "hull has fewer than three vertices"}]
    xs = [point[0] for point in ring[:-1]]
    ys = [point[1] for point in ring[:-1]]
    spacing_x, spacing_y = field.spacing_gu
    # The field grid is expressed in absolute frame coordinates whose plan
    # origin is the field origin, so plan sample nodes are i*spacing.
    min_ix = math.floor(min(xs) / spacing_x)
    max_ix = math.ceil(max(xs) / spacing_x)
    min_iy = math.floor(min(ys) / spacing_y)
    max_iy = math.ceil(max(ys) / spacing_y)
    requests: list[tuple[float, float, str]] = []
    requests.extend((x, y, "hull_vertex") for x, y in ring[:-1])
    requests.extend((x, y, "edge_midpoint") for x, y in _edge_midpoints(ring))
    requests.extend((x, y, "member_xy") for x, y in member_positions_plan)
    requests.extend((x, y, "door_xy") for x, y in door_positions_plan)
    for iy in range(min_iy, max_iy + 1):
        for ix in range(min_ix, max_ix + 1):
            x, y = ix * spacing_x, iy * spacing_y
            if cityplan.point_in_ring((x, y), ring):
                requests.append((x, y, "footprint_field_node"))
    return _sample_unique(field, requests)


def _best_fit_slope(samples: Sequence[TerrainSample]) -> float:
    """Fit z=a*x+b*y+c and return ``atan(sqrt(a²+b²))`` in degrees."""

    if len(samples) < 3:
        return 0.0
    matrix = np.asarray([[s.x_plan_gu, s.y_plan_gu, 1.0] for s in samples], dtype=np.float64)
    values = np.asarray([s.height_gu for s in samples], dtype=np.float64)
    try:
        coeff, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    return math.degrees(math.atan(math.hypot(float(coeff[0]), float(coeff[1]))))


def _finite_max(values: Iterable[float], default: float = 0.0) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return max(values, default=default)


def _finite_min(values: Iterable[float], default: float = 0.0) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return min(values, default=default)


def terrain_metrics(
    field: TerrainField,
    hull_plan_xy: Sequence[Sequence[float] | tuple[float, float]],
    *,
    member_positions_plan: Iterable[tuple[float, float]],
    door_positions_plan: Iterable[tuple[float, float]],
    anchor_plan_xy: Sequence[float],
    anchor_z_gu: float,
    placed_members: Sequence[PlacedMember],
    bounds_min_z_gu: float,
) -> dict[str, Any]:
    """Measure all terrain values needed by the T1.2 lot contract."""

    samples, missing = sample_footprint(
        field,
        hull_plan_xy,
        member_positions_plan=member_positions_plan,
        door_positions_plan=door_positions_plan,
    )
    heights = [sample.height_gu for sample in samples]
    slopes = [sample.slope_deg for sample in samples]
    # D-STAMP burial is measured from the transformed member-bounds minimum,
    # not from the minimum of member origins (which can sit well above a mesh
    # base).  The relative bounds are source evidence and are not invented by
    # the solver.
    member_bottom = anchor_z_gu + float(bounds_min_z_gu)
    max_terrain = _finite_max(heights, default=float("nan"))
    min_terrain = _finite_min(heights, default=float("nan"))
    return {
        "field_pass": field.field_pass,
        "sample_count": len(samples),
        "missing_sample_count": len(missing),
        "missing_samples": missing,
        "samples": [sample.to_dict() for sample in samples],
        "height_min_gu": min_terrain,
        "height_max_gu": max_terrain,
        "relief_gu": (max_terrain - min_terrain) if samples else float("nan"),
        "best_fit_slope_deg": _best_fit_slope(samples),
        "max_local_slope_deg": _finite_max(slopes, default=float("nan")),
        "seed_sample": field.sample(float(anchor_plan_xy[0]), float(anchor_plan_xy[1])).to_dict(),
        "member_bottom_z_gu": member_bottom,
        "bottom_clearance_gu": member_bottom - min_terrain if samples else float("nan"),
        "burial_depth_gu": max_terrain - member_bottom if samples else float("nan"),
    }


def _water_tiles_for_hull(hull: Sequence[tuple[float, float]], bundle: Any) -> list[list[int]]:
    return [list(tile) for tile in cityplan.tiles_covered_by_ring([list(p) for p in hull])
            if bundle.tile_water(tile[0], tile[1])]


def scope_and_mask_checks(
    hull: Sequence[tuple[float, float]],
    *,
    anchor_xy: Sequence[float],
    bundle: Any,
) -> dict[str, Any]:
    """Independently recheck exact scope, water, and buildable coverage."""

    out_of_scope = [list(point) for point in hull if not cityplan.in_scope(point[0], point[1])]
    covered = cityplan.tiles_covered_by_ring([list(point) for point in hull]) if not out_of_scope else []
    unbuildable = [list(tile) for tile in covered if not bundle.tile_buildable(tile[0], tile[1])]
    water = _water_tiles_for_hull(hull, bundle) if not out_of_scope else []
    anchor_state = bundle.door_anchor_state(float(anchor_xy[0]), float(anchor_xy[1]))
    return {
        "footprint_vertices_in_scope": not out_of_scope,
        "out_of_scope_vertices": out_of_scope,
        "covered_tiles": [list(tile) for tile in covered],
        "covered_tile_count": len(covered),
        "unbuildable_tiles": unbuildable,
        "water_tiles": water,
        "anchor": anchor_state,
    }


def _angle_deviation_deg(a: float, b: float) -> float:
    delta = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


def _nearest_polyline(
    point: tuple[float, float], polyline: Sequence[Sequence[float]]
) -> tuple[float, tuple[float, float], int, float, tuple[float, float]]:
    """Return distance, nearest point, segment index, t, and segment vector."""

    best = (float("inf"), (float(polyline[0][0]), float(polyline[0][1])), 0, 0.0, (0.0, 0.0))
    for index, (a_raw, b_raw) in enumerate(zip(polyline, polyline[1:])):
        a, b = _point_tuple(a_raw), _point_tuple(b_raw)
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 == 0.0 else ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2
        t = max(0.0, min(1.0, t))
        nearest = (a[0] + t * dx, a[1] + t * dy)
        distance = math.hypot(point[0] - nearest[0], point[1] - nearest[1])
        if distance < best[0]:
            best = (distance, nearest, index, t, (dx, dy))
    return best


def _road_candidates(plan: Mapping[str, Any], bundle: Any) -> dict[str, dict[str, Any]]:
    """Collect planned road polylines, aligned centerlines as fallback."""

    roads: dict[str, dict[str, Any]] = {}
    for road in plan.get("roads", []):
        if isinstance(road, Mapping) and isinstance(road.get("road_id"), str):
            polyline = road.get("polyline")
            if isinstance(polyline, list) and len(polyline) >= 2:
                roads[str(road["road_id"])] = {
                    "road_id": road["road_id"],
                    "class": road.get("class"),
                    "width_gu": float(road.get("width_gu", 0.0)),
                    "polyline": polyline,
                    "source": "planned_road",
                }
    origin = tuple(float(v) for v in bundle.survey_frame["origin_gu"])
    network = getattr(bundle, "aligned_network", None)
    if network is not None:
        edges = sorted(network.edges.values(), key=lambda item: item.id)
        for edge in edges:
            edge_id = edge.id
            chain = edge.smooth_gu_polyline
            if len(chain) < 2:
                continue
            roads.setdefault(
                edge_id,
                {
                    "road_id": edge_id,
                    "class": "source_centerline",
                    "width_gu": 0.0,
                    "polyline": [[float(p[0]) - origin[0], float(p[1]) - origin[1]]
                                 for p in chain],
                    "source": "aligned_centerline_fallback",
                },
            )
        return roads
    for edge in sorted(bundle.centerlines.get("edges", []), key=lambda item: str(item.get("id"))):
        edge_id = edge.get("id")
        chain = edge.get("smooth_gu_polyline")
        if not isinstance(edge_id, str) or not isinstance(chain, list) or len(chain) < 2:
            continue
        roads.setdefault(
            edge_id,
            {
                "road_id": edge_id,
                "class": edge.get("road_class", "source_centerline"),
                "width_gu": 0.0,
                "polyline": [[float(p[0]) - origin[0], float(p[1]) - origin[1]] for p in chain],
                "source": "aligned_centerline_fallback",
            },
        )
    return roads


def road_access_check(
    plan: Mapping[str, Any],
    bundle: Any,
    *,
    lot: Mapping[str, Any],
    door_xy_plan: tuple[float, float],
    source_access_heading_rad: float,
    yaw_deg: float,
    field: TerrainField,
    hull: Sequence[tuple[float, float]],
    config: PlacementConfig,
) -> dict[str, Any]:
    """Measure exact/nearest road access without changing the plan yaw."""

    candidates = _road_candidates(plan, bundle)
    access = lot.get("access") if isinstance(lot.get("access"), Mapping) else {}
    named = access.get("face_road") if isinstance(access, Mapping) else None
    if isinstance(named, str) and named in candidates:
        selected = {named: candidates[named]}
        resolution = "exact_named_planned_road" if candidates[named]["source"] == "planned_road" else "named_source_centerline"
    else:
        selected = candidates
        resolution = "nearest_planned_or_source_centerline"
    if not selected:
        return {
            "status": "deferred",
            "code": "road_geometry_unavailable",
            "resolution": "none",
            "message": "no planned road or aligned centerline is available",
        }
    best: dict[str, Any] | None = None
    for road_id in sorted(selected):
        road = selected[road_id]
        distance, nearest, segment_index, segment_t, vector = _nearest_polyline(door_xy_plan, road["polyline"])
        if best is None or distance < best["distance_gu"]:
            best = {
                "road_id": road_id,
                "road": road,
                "distance_gu": distance,
                "nearest_point_plan_gu": list(nearest),
                "segment_index": segment_index,
                "segment_t": segment_t,
                "segment_vector": list(vector),
            }
    assert best is not None
    nearest = tuple(best["nearest_point_plan_gu"])
    dx, dy = nearest[0] - door_xy_plan[0], nearest[1] - door_xy_plan[1]
    access_heading_world = float(source_access_heading_rad) + math.radians(float(yaw_deg))
    heading_to_road = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1.0e-12 else access_heading_world
    road_heading = math.atan2(best["segment_vector"][1], best["segment_vector"][0])
    # Cross-slope is measured on the shortest door-to-corridor segment.  A
    # zero-length access segment is flat by definition.
    if best["distance_gu"] <= 1.0e-12:
        cross_slope = 0.0
        terrain_delta = 0.0
    else:
        seed = field.sample(*door_xy_plan)
        road_sample = field.sample(*nearest)
        terrain_delta = road_sample.height_gu - seed.height_gu
        cross_slope = math.degrees(math.atan(abs(terrain_delta) / best["distance_gu"]))
    hull_road_distance = min(
        _nearest_polyline(point, selected[best["road_id"]]["polyline"])[0]
        for point in hull
    ) if hull else float("inf")
    result = {
        "status": "measured",
        "resolution": resolution,
        "road_id": best["road_id"],
        "road_source": best["road"]["source"],
        "closest_point_plan_gu": best["nearest_point_plan_gu"],
        "door_to_road_distance_gu": best["distance_gu"],
        # T1.2 has no graph route yet; this is the measured shortest corridor
        # distance, not a fabricated network path length.
        "path_distance_gu": best["distance_gu"],
        "hull_to_road_distance_gu": hull_road_distance,
        "cross_slope_deg": cross_slope,
        "terrain_delta_door_to_road_gu": terrain_delta,
        "resulting_door_access_heading_rad": access_heading_world,
        "door_to_road_heading_rad": heading_to_road,
        "road_segment_heading_rad": road_heading,
        "angular_deviation_deg": _angle_deviation_deg(access_heading_world, heading_to_road),
        "preferred_distance_gu": config.preferred_road_distance_gu,
        "hard_distance_gu": config.hard_road_distance_gu,
        "hard_cross_slope_deg": config.hard_cross_slope_deg,
        "plan_yaw_preserved_deg": float(yaw_deg),
    }
    return result


def road_corridor_conflict(
    hull: Sequence[tuple[float, float]], road: Mapping[str, Any], *, margin_gu: float = 128.0
) -> bool:
    """Conservative corridor/hull conflict check for the houses-only stage."""

    width = max(0.0, float(road.get("width_gu", 0.0)))
    radius = width * 0.5 + float(margin_gu)
    polyline = road.get("polyline", [])
    if not isinstance(polyline, list) or len(polyline) < 2:
        return False
    if any(cityplan.point_polyline_distance(point, polyline) < radius for point in hull):
        return True
    closed = _closed(hull)
    for a, b in zip(closed, closed[1:]):
        if cityplan.point_polyline_distance(a, polyline) < radius or cityplan.point_polyline_distance(b, polyline) < radius:
            return True
        for road_a, road_b in zip(polyline, polyline[1:]):
            if cityplan.segments_intersect_strict(
                a, b, _point_tuple(road_a), _point_tuple(road_b)
            ):
                return True
    if any(cityplan.point_in_ring(_point_tuple(point), _closed(hull)) for point in polyline):
        return True
    return False


def collision_status(
    hull: Sequence[tuple[float, float]],
    others: Iterable[tuple[str, Sequence[tuple[float, float]]]],
    *,
    config: PlacementConfig,
) -> dict[str, Any]:
    """Return exact hard hull conflict details; spacing guidance is separate."""

    conflicts: list[dict[str, Any]] = []
    guidance: list[dict[str, Any]] = []
    for other_id, other_hull in sorted(others, key=lambda item: item[0]):
        status, distance = cityplan.ring_pair_status(
            [list(point) for point in hull], [list(point) for point in other_hull]
        )
        row = {
            "other_lot_id": other_id,
            "status": status,
            "boundary_gap_gu": distance,
            "hard_minimum_gu": 0.0,
            "spacing_guidance_only": True,
        }
        if status in ("overlap", "touch"):
            conflicts.append(row)
        else:
            guidance.append(row)
    return {
        "status": "conflict" if conflicts else "clear",
        "hard_conflicts": conflicts,
        "guidance_measurements": guidance,
        "hard_rule": "exact footprint overlap or contact rejects; no dispatch-5 minimum",
    }


def fine_collision_ledger_entry(lot_id: str, stamp_id: str, member_count: int) -> dict[str, Any]:
    """Explicitly record unavailable triangle/AABB geometry."""

    return {
        "lot_id": lot_id,
        "stamp_id": stamp_id,
        "member_count": member_count,
        "status": "deferred",
        "code": "fine_collision_deferred",
        "checked": False,
        "triangle_intersection_checked": False,
        "member_aabb_checked": False,
        "reason": (
            "accepted D-STAMP libraries provide convex hulls but no source "
            "triangle surfaces or member world AABBs for this stage"
        ),
    }
