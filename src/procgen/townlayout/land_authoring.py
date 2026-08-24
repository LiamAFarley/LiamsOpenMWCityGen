"""Author town-layout LAND records with road, gate, and building grading.

Purpose
-------
Turn circulation paint requests, seated building footprints, and an optional
composed wall document into a masterless tes3conv JSON document containing the
affected LAND records. Source LAND payloads are copied intact; normalized VTEX
tiles and explicitly selected terrain vertices are changed.

Terrain and paint order
-----------------------
1. Building footprint terrain is lowered only where source height exceeds the
   primary door's source terrain height, with a short blended transition ring.
2. Wall and authored-slope footprints are seated to their composer-published
   small burial fraction, with one nearest profile per vertex.
3. Broad road surfaces receive a smoothed longitudinal profile while each road
   cross-section shares one height; transitions blend into source terrain.
4. Gate crossings receive flat, oriented platforms and capped-grade approaches;
   their wall-derived corridors are also repainted with the road assignment.
5. Every modified height is clamped above the configured water-safe floor and
   duplicated LAND-cell border vertices are synchronized before serialization.
6. Existing source-road tiles inside the city domain become the configured base
   grass assignment, erasing the old road texture.
7. Authored broad roads/civic polygons use each request's explicit semantic
   assignment, so each road class remains independently configurable.

Narrow alley/apron requests remain in the circulation product for later
terrain-following geometry and are not rasterized here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box

from .. import espland, tes3json
from ..cityscape_vtex import load_surface_assignments


class LandAuthoringError(ValueError):
    """Raised when LAND source payloads or paint coverage are incomplete."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LandAuthoringError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LandAuthoringError(f"{label} must be a JSON object")
    return value


def _rings_polygon(rings: list[list[list[float]]]) -> Polygon:
    if not rings:
        return Polygon()
    return Polygon(rings[0])


def _tile_center(cell: tuple[int, int], tile_x: int, tile_y: int) -> tuple[float, float]:
    return (
        (cell[0] + 95) * 8192.0 + (tile_x + 0.5) * 512.0,
        (cell[1] + 11) * 8192.0 + (tile_y + 0.5) * 512.0,
    )


def _cells_for_requests(requests: list[Mapping[str, Any]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for request in requests:
        for ring in request.get("polygon", []):
            for point in ring:
                x, y = float(point[0]), float(point[1])
                cells.add((math.floor(x / 8192.0) - 95, math.floor(y / 8192.0) - 11))
    return cells


def _paint_mask(
    requests: list[Mapping[str, Any]], cells: set[tuple[int, int]]
) -> dict[tuple[int, int], np.ndarray]:
    masks = {cell: np.zeros((16, 16), dtype=bool) for cell in cells}
    for request in requests:
        source_grid = request.get("source_tile_grid")
        source_local = request.get("source_tile_local")
        if isinstance(source_grid, list) and isinstance(source_local, list):
            cell = (int(source_grid[0]), int(source_grid[1]))
            if cell in masks:
                masks[cell][int(source_local[1]), int(source_local[0])] = True
            continue
        polygon = _rings_polygon(request.get("polygon") or [])
        if polygon.is_empty:
            continue
        intersects_tiles = request.get("coverage_mode") == "intersects"
        for cell in cells:
            for tile_y in range(16):
                for tile_x in range(16):
                    x, y = _tile_center(cell, tile_x, tile_y)
                    covered = (
                        polygon.intersects(box(x - 256.0, y - 256.0, x + 256.0, y + 256.0))
                        if intersects_tiles
                        else polygon.covers(Point(x, y))
                    )
                    if covered:
                        masks[cell][tile_y, tile_x] = True
    return masks


def _deform_heights(
    source_heights: Any,
    cell: tuple[int, int],
    requests: list[Mapping[str, Any]],
    minimum_height_thu: float,
) -> tuple[np.ndarray, int]:
    """Lower source VHGT only above each building's door-height ceiling."""

    grid = np.asarray(source_heights, dtype=np.float64).reshape(65, 65).copy()
    changed = 0
    for request in requests:
        rings = request.get("polygon") or []
        polygon = _rings_polygon(rings)
        if polygon.is_empty:
            continue
        ceiling = float(request.get("ceiling_height_gu")) / 8.0
        margin = max(1.0, float(request.get("blend_margin_gu", 256.0)))
        outer = polygon.buffer(margin)
        for vertex_y in range(65):
            for vertex_x in range(65):
                x = (cell[0] + 95) * 8192.0 + vertex_x * 128.0
                y = (cell[1] + 11) * 8192.0 + vertex_y * 128.0
                point = Point(x, y)
                if polygon.covers(point):
                    weight = 1.0
                elif outer.covers(point):
                    normalized = max(0.0, min(1.0, point.distance(polygon) / margin))
                    # Smoothstep gives zero slope at both ends of the ramp;
                    # the previous linear falloff made visible terrace edges.
                    weight = 1.0 - (normalized * normalized * (3.0 - 2.0 * normalized))
                else:
                    continue
                current = float(grid[vertex_y, vertex_x])
                if current <= ceiling:
                    continue
                target = current + (ceiling - current) * weight
                rounded = max(round(minimum_height_thu), round(target))
                if rounded < grid[vertex_y, vertex_x]:
                    grid[vertex_y, vertex_x] = rounded
                    changed += 1
    return grid, changed


def _source_height_thu(
    source_land: Mapping[tuple[int, int], espland.LandRecord], x: float, y: float
) -> float:
    """Bilinear source-LAND height at an absolute plan-GU point."""

    cell = (math.floor(x / 8192.0) - 95, math.floor(y / 8192.0) - 11)
    record = source_land.get(cell)
    if record is None or record.heights_thu is None:
        raise LandAuthoringError(f"source LAND missing grading sample at {cell}")
    grid = np.asarray(record.heights_thu, dtype=np.float64).reshape(65, 65)
    local_x = (x - (cell[0] + 95) * 8192.0) / 128.0
    local_y = (y - (cell[1] + 11) * 8192.0) / 128.0
    ix = min(max(int(math.floor(local_x)), 0), 63)
    iy = min(max(int(math.floor(local_y)), 0), 63)
    tx = min(max(local_x - ix, 0.0), 1.0)
    ty = min(max(local_y - iy, 0.0), 1.0)
    return float(
        grid[iy, ix] * (1.0 - tx) * (1.0 - ty)
        + grid[iy, ix + 1] * tx * (1.0 - ty)
        + grid[iy + 1, ix] * (1.0 - tx) * ty
        + grid[iy + 1, ix + 1] * tx * ty
    )


def _smooth_profile(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) < 3:
        return values.copy()
    offsets = np.arange(-radius, radius + 1, dtype=float)
    weights = (radius + 1.0) - np.abs(offsets)
    weights /= float(np.sum(weights))
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, weights, mode="valid")


def _road_grade_specs(
    requests: list[Mapping[str, Any]],
    source_land: Mapping[tuple[int, int], espland.LandRecord],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    config = policy.get("road_grading") or {}
    if not config.get("enabled", False):
        return []
    spacing = float(config["profile_sample_spacing_gu"])
    window = float(config["smoothing_window_gu"])
    cross_extension = float(config.get("cross_section_extension_gu", 0.0))
    if spacing <= 0.0 or window < 0.0 or cross_extension < 0.0:
        raise LandAuthoringError("road grading spacing/window must be nonnegative")
    specs = []
    for request in requests:
        centerline = request.get("centerline") or []
        source_polygon = _rings_polygon(request.get("polygon") or [])
        if len(centerline) < 2 or source_polygon.is_empty:
            continue
        polygon = source_polygon.buffer(cross_extension)
        line = LineString(centerline)
        count = max(2, int(math.ceil(line.length / spacing)) + 1)
        arcs = np.linspace(0.0, float(line.length), count)
        raw = np.asarray(
            [
                _source_height_thu(source_land, float(line.interpolate(arc).x), float(line.interpolate(arc).y))
                for arc in arcs
            ],
            dtype=float,
        )
        sample_step = float(arcs[1] - arcs[0]) if len(arcs) > 1 else spacing
        radius = int(round((window / 2.0) / max(sample_step, 1e-9)))
        specs.append(
            {
                "kind": "road",
                "line": line,
                "polygon": polygon,
                "outer": polygon.buffer(float(config["blend_margin_gu"])),
                "arcs": arcs,
                "profile_thu": _smooth_profile(raw, radius),
                "blend_margin_gu": float(config["blend_margin_gu"]),
                "max_delta_thu": float(config["max_cut_fill_gu"]) / 8.0,
            }
        )
    return specs


def _gate_grade_specs(
    wall: Mapping[str, Any] | None,
    source_land: Mapping[tuple[int, int], espland.LandRecord],
    policy: Mapping[str, Any],
    road_requests: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    config = policy.get("gate_platform") or {}
    if not config.get("enabled", False) or not wall:
        return []
    default_half_width = float(config["half_width_across_road_gu"])
    default_half_length = float(config["half_length_along_road_gu"])
    approach_length = float(config["approach_length_gu"])
    max_road_grade = float(config["max_road_grade"])
    spacing = float(config["sample_spacing_gu"])
    if min(default_half_width, default_half_length, approach_length, spacing, max_road_grade) <= 0.0:
        raise LandAuthoringError("gate platform dimensions must be positive")
    specs = []
    seen: set[str] = set()
    for member in wall.get("members", []):
        if member.get("structural_role") != "gatehouse":
            continue
        meta = member.get("meta") or {}
        half_width = float(meta.get("landing_half_width_across_road_gu", default_half_width))
        half_length = float(meta.get("landing_half_length_along_road_gu", default_half_length))
        gate_id = str(meta.get("gate_id") or member.get("source_id"))
        if gate_id in seen:
            continue
        seen.add(gate_id)
        landing_center = meta.get("landing_center_xy_gu")
        if not isinstance(landing_center, list) or len(landing_center) != 2:
            raise LandAuthoringError(f"gate {gate_id} has no landing_center_xy_gu")
        gatehouse_bottom_z = meta.get("gatehouse_bottom_z_gu")
        if not isinstance(gatehouse_bottom_z, (int, float)):
            raise LandAuthoringError(f"gate {gate_id} has no gatehouse_bottom_z_gu")
        platform_points = meta.get("gatehouse_platform_polygon_xy_gu")
        if not isinstance(platform_points, list) or len(platform_points) < 3:
            raise LandAuthoringError(f"gate {gate_id} has no gatehouse platform polygon")
        rotz = float((member.get("rotation") or [0.0, 0.0, 0.0])[2])
        wall_axis = np.asarray([math.cos(rotz), -math.sin(rotz)], dtype=float)
        road_axis = np.asarray([math.sin(rotz), math.cos(rotz)], dtype=float)
        center = np.asarray(
            [float(landing_center[0]), float(landing_center[1])], dtype=float
        )
        road_matches = []
        for request in road_requests:
            centerline = request.get("centerline") or []
            assignment = request.get("surface_assignment")
            if len(centerline) < 2 or not isinstance(assignment, Mapping):
                continue
            raw_vtex = assignment.get("raw_vtex")
            if not isinstance(raw_vtex, int):
                continue
            line = LineString(centerline)
            road_matches.append((float(line.distance(Point(*center))), raw_vtex))
        if not road_matches:
            raise LandAuthoringError(f"gate {gate_id} has no authored road centerline")
        road_distance, road_raw_vtex = min(road_matches, key=lambda row: row[0])
        if road_distance > max(half_width, 512.0):
            raise LandAuthoringError(
                f"gate {gate_id} is {road_distance:.3f} GU from its nearest road"
            )
        polygon = Polygon(
            [(float(point[0]), float(point[1])) for point in platform_points]
        )
        if polygon.is_empty or not polygon.is_valid:
            raise LandAuthoringError(f"gate {gate_id} has an invalid platform polygon")
        corridor_corners = [
            center + wall_axis * (sx * half_width)
            + road_axis * (sy * (half_length + approach_length))
            for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
        ]
        corridor = Polygon(
            [(float(point[0]), float(point[1])) for point in corridor_corners]
        )
        samples = []
        for across in np.arange(-half_width, half_width + spacing * 0.5, spacing):
            for along in np.arange(-half_length, half_length + spacing * 0.5, spacing):
                point = center + wall_axis * across + road_axis * along
                samples.append(_source_height_thu(source_land, float(point[0]), float(point[1])))
        maximum_approach_rise_thu = approach_length * max_road_grade / 8.0
        approach_end_thu = {}
        for side in (-1.0, 1.0):
            endpoint = center + road_axis * (side * (half_length + approach_length))
            source_endpoint_thu = _source_height_thu(
                source_land, float(endpoint[0]), float(endpoint[1])
            )
            approach_end_thu[side] = float(gatehouse_bottom_z) / 8.0 + max(
                -maximum_approach_rise_thu,
                min(
                    maximum_approach_rise_thu,
                    source_endpoint_thu - float(gatehouse_bottom_z) / 8.0,
                ),
            )
        specs.append(
            {
                "kind": "gate",
                "gate_id": gate_id,
                "polygon": polygon,
                "outer": polygon.union(corridor).buffer(float(config["blend_margin_gu"])),
                "paint_polygon": corridor,
                "target_thu": float(gatehouse_bottom_z) / 8.0,
                "source_min_thu": min(samples),
                "source_max_thu": max(samples),
                "blend_margin_gu": float(config["blend_margin_gu"]),
                "max_delta_thu": float(config["max_cut_fill_gu"]) / 8.0,
                "center": center,
                "wall_axis": wall_axis,
                "road_axis": road_axis,
                "half_width_gu": half_width,
                "half_length_gu": half_length,
                "approach_length_gu": approach_length,
                "max_road_grade": max_road_grade,
                "road_raw_vtex": int(road_raw_vtex),
                "road_centerline_distance_gu": road_distance,
                "approach_end_thu_negative": approach_end_thu[-1.0],
                "approach_end_thu_positive": approach_end_thu[1.0],
            }
        )
    return specs


def _wall_grade_specs(
    wall: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Narrow foundation benches following each solved wall/slope profile."""

    config = policy.get("wall_foundation") or {}
    if not config.get("enabled", False) or not wall:
        return []
    margin = float(config["blend_margin_gu"])
    max_delta = float(config["max_cut_fill_gu"]) / 8.0
    if margin <= 0.0 or max_delta <= 0.0:
        raise LandAuthoringError("wall foundation grading values must be positive")
    origin = np.asarray(wall.get("origin_gu") or [0.0, 0.0], dtype=float)
    specs = []
    for member in wall.get("members", []):
        role = member.get("structural_role")
        meta = member.get("meta") or {}
        if role not in {"straight", "slope", "gate_neck"}:
            continue
        profile = meta.get("terrain_grade_profile")
        footprint = member.get("footprint_xy_rel") or []
        if not isinstance(profile, Mapping) or len(footprint) < 3:
            raise LandAuthoringError(
                f"wall member {member.get('source_id')} has no terrain grade profile"
            )
        polygon = Polygon(
            [
                (float(point[0]) + float(origin[0]), float(point[1]) + float(origin[1]))
                for point in footprint
            ]
        )
        end_a = profile.get("end_a_xy")
        end_b = profile.get("end_b_xy")
        if not isinstance(end_a, list) or not isinstance(end_b, list):
            raise LandAuthoringError("wall terrain grade profile has no endpoints")
        line = LineString([end_a, end_b])
        ground_a = float(profile["ground_z_end_a_gu"])
        ground_b = float(profile["ground_z_end_b_gu"])
        specs.append(
            {
                "kind": "wall",
                "polygon": polygon,
                "outer": polygon.buffer(margin),
                "line": line,
                "arcs": np.asarray([0.0, float(line.length)], dtype=float),
                "profile_thu": np.asarray(
                    [
                        ground_a / 8.0,
                        ground_b / 8.0,
                    ],
                    dtype=float,
                ),
                "minimum_profile_thu": np.asarray(
                    [
                        float(profile["minimum_ground_z_end_a_gu"]) / 8.0,
                        float(profile["minimum_ground_z_end_b_gu"]) / 8.0,
                    ],
                    dtype=float,
                ),
                "maximum_profile_thu": np.asarray(
                    [
                        float(profile["maximum_ground_z_end_a_gu"]) / 8.0,
                        float(profile["maximum_ground_z_end_b_gu"]) / 8.0,
                    ],
                    dtype=float,
                ),
                "blend_margin_gu": margin,
                "max_delta_thu": max_delta,
            }
        )
    return specs


def _apply_grade_specs(
    grid: np.ndarray,
    cell: tuple[int, int],
    specs: list[dict[str, Any]],
    minimum_height_thu: float,
) -> int:
    changed = 0
    cell_min_x = (cell[0] + 95) * 8192.0
    cell_min_y = (cell[1] + 11) * 8192.0
    cell_max_x = cell_min_x + 8192.0
    cell_max_y = cell_min_y + 8192.0
    active_specs = [
        spec
        for spec in specs
        if not (
            spec["outer"].bounds[2] < cell_min_x
            or spec["outer"].bounds[0] > cell_max_x
            or spec["outer"].bounds[3] < cell_min_y
            or spec["outer"].bounds[1] > cell_max_y
        )
    ]
    for spec in active_specs:
        polygon = spec["polygon"]
        outer = spec["outer"]
        margin = max(float(spec["blend_margin_gu"]), 1.0)
        for vertex_y in range(65):
            for vertex_x in range(65):
                x = (cell[0] + 95) * 8192.0 + vertex_x * 128.0
                y = (cell[1] + 11) * 8192.0 + vertex_y * 128.0
                point = Point(x, y)
                current = float(grid[vertex_y, vertex_x])
                if spec["kind"] == "gate":
                    if spec["polygon"].covers(point):
                        rounded = max(
                            round(minimum_height_thu), round(float(spec["target_thu"]))
                        )
                        if rounded != int(grid[vertex_y, vertex_x]):
                            grid[vertex_y, vertex_x] = rounded
                            changed += 1
                        continue
                    relative = np.asarray([x, y], dtype=float) - spec["center"]
                    across = abs(float(relative @ spec["wall_axis"]))
                    along = abs(float(relative @ spec["road_axis"]))
                    longitudinal = max(0.0, along - float(spec["half_length_gu"]))
                    lateral = max(0.0, across - float(spec["half_width_gu"]))
                    if (
                        longitudinal > float(spec["approach_length_gu"])
                        or lateral > margin
                    ):
                        if not outer.covers(point):
                            continue
                        normalized = max(
                            0.0, min(1.0, point.distance(polygon) / margin)
                        )
                        weight = 1.0 - normalized * normalized * (
                            3.0 - 2.0 * normalized
                        )
                        rounded = max(
                            round(minimum_height_thu),
                            round(
                                current
                                + (float(spec["target_thu"]) - current) * weight
                            ),
                        )
                        if rounded != int(grid[vertex_y, vertex_x]):
                            grid[vertex_y, vertex_x] = rounded
                            changed += 1
                        continue
                    target = float(spec["target_thu"])
                    if longitudinal <= 0.0:
                        profile_target = target
                    else:
                        side_key = (
                            "approach_end_thu_positive"
                            if float(relative @ spec["road_axis"]) >= 0.0
                            else "approach_end_thu_negative"
                        )
                        fraction = min(
                            1.0,
                            longitudinal / float(spec["approach_length_gu"]),
                        )
                        profile_target = target + (
                            float(spec[side_key]) - target
                        ) * fraction
                    lateral_fraction = max(0.0, min(1.0, lateral / margin))
                    lateral_weight = 1.0 - lateral_fraction * lateral_fraction * (
                        3.0 - 2.0 * lateral_fraction
                    )
                    rounded = max(
                        round(minimum_height_thu),
                        round(current + (profile_target - current) * lateral_weight),
                    )
                    if rounded != int(grid[vertex_y, vertex_x]):
                        grid[vertex_y, vertex_x] = rounded
                        changed += 1
                    continue
                inside = polygon.covers(point)
                if inside:
                    weight = 1.0
                elif outer.covers(point):
                    normalized = max(0.0, min(1.0, point.distance(polygon) / margin))
                    weight = 1.0 - normalized * normalized * (3.0 - 2.0 * normalized)
                else:
                    continue
                if spec["kind"] == "road":
                    arc = float(spec["line"].project(point))
                    target = float(np.interp(arc, spec["arcs"], spec["profile_thu"]))
                elif spec["kind"] == "wall":
                    arc = float(spec["line"].project(point))
                    target = float(np.interp(arc, spec["arcs"], spec["profile_thu"]))
                else:
                    target = float(spec["target_thu"])
                if spec["kind"] == "wall" and inside:
                    rounded = max(round(minimum_height_thu), round(target))
                else:
                    delta = max(
                        -float(spec["max_delta_thu"]),
                        min(float(spec["max_delta_thu"]), target - current),
                    )
                    rounded = max(
                        round(minimum_height_thu), round(current + delta * weight)
                    )
                if rounded != int(grid[vertex_y, vertex_x]):
                    grid[vertex_y, vertex_x] = rounded
                    changed += 1
    return changed


def _apply_wall_grade_specs(
    grid: np.ndarray,
    cell: tuple[int, int],
    specs: list[dict[str, Any]],
    minimum_height_thu: float,
) -> int:
    """Apply one nearest wall-foundation profile per vertex.

    Sequential overlapping buffers compound deep cuts. Selecting one profile
    makes the result independent of member count while preserving exact core
    seating and a broad transition back to source terrain.
    """

    cell_min_x = (cell[0] + 95) * 8192.0
    cell_min_y = (cell[1] + 11) * 8192.0
    cell_max_x = cell_min_x + 8192.0
    cell_max_y = cell_min_y + 8192.0
    active = [
        spec
        for spec in specs
        if not (
            spec["outer"].bounds[2] < cell_min_x
            or spec["outer"].bounds[0] > cell_max_x
            or spec["outer"].bounds[3] < cell_min_y
            or spec["outer"].bounds[1] > cell_max_y
        )
    ]
    changed = 0
    for vertex_y in range(65):
        for vertex_x in range(65):
            x = cell_min_x + vertex_x * 128.0
            y = cell_min_y + vertex_y * 128.0
            point = Point(x, y)
            candidates = [spec for spec in active if spec["outer"].covers(point)]
            if not candidates:
                continue
            spec = min(candidates, key=lambda row: point.distance(row["polygon"]))
            distance = float(point.distance(spec["polygon"]))
            margin = max(float(spec["blend_margin_gu"]), 1.0)
            normalized = max(0.0, min(1.0, distance / margin))
            weight = 1.0 - normalized * normalized * (3.0 - 2.0 * normalized)
            arc = float(spec["line"].project(point))
            target = float(np.interp(arc, spec["arcs"], spec["profile_thu"]))
            current = float(grid[vertex_y, vertex_x])
            if spec["polygon"].covers(point):
                rounded = max(round(minimum_height_thu), round(target))
            else:
                delta = max(
                    -float(spec["max_delta_thu"]),
                    min(float(spec["max_delta_thu"]), target - current),
                )
                rounded = max(
                    round(minimum_height_thu), round(current + delta * weight)
                )
            if rounded != int(grid[vertex_y, vertex_x]):
                grid[vertex_y, vertex_x] = rounded
                changed += 1
    return changed


def _clamp_wall_coverage_specs(
    grid: np.ndarray,
    cell: tuple[int, int],
    specs: list[dict[str, Any]],
    minimum_height_thu: float,
) -> int:
    """Enforce the final wall burial interval after roads and gates grade LAND."""

    cell_min_x = (cell[0] + 95) * 8192.0
    cell_min_y = (cell[1] + 11) * 8192.0
    cell_max_x = cell_min_x + 8192.0
    cell_max_y = cell_min_y + 8192.0
    active = [
        spec
        for spec in specs
        if not (
            spec["polygon"].bounds[2] < cell_min_x
            or spec["polygon"].bounds[0] > cell_max_x
            or spec["polygon"].bounds[3] < cell_min_y
            or spec["polygon"].bounds[1] > cell_max_y
        )
    ]
    changed = 0
    for vertex_y in range(65):
        for vertex_x in range(65):
            x = cell_min_x + vertex_x * 128.0
            y = cell_min_y + vertex_y * 128.0
            point = Point(x, y)
            candidates = [spec for spec in active if spec["polygon"].covers(point)]
            if not candidates:
                continue
            spec = min(candidates, key=lambda row: point.distance(row["line"]))
            arc = float(spec["line"].project(point))
            minimum = float(
                np.interp(arc, spec["arcs"], spec["minimum_profile_thu"])
            )
            maximum = float(
                np.interp(arc, spec["arcs"], spec["maximum_profile_thu"])
            )
            lower = max(math.ceil(minimum - 1e-9), math.ceil(minimum_height_thu))
            upper = math.floor(maximum + 1e-9)
            if lower > upper:
                raise LandAuthoringError("wall burial interval is narrower than LAND quantization")
            current = int(grid[vertex_y, vertex_x])
            clamped = min(max(current, lower), upper)
            if clamped != current:
                grid[vertex_y, vertex_x] = clamped
                changed += 1
    return changed


def _flags(flags: int) -> str:
    names = []
    if flags & 0x1:
        names.append("USES_VERTEX_HEIGHTS_AND_NORMALS")
    if flags & 0x2:
        names.append("USES_VERTEX_COLORS")
    if flags & 0x4:
        names.append("USES_TEXTURES")
    remaining = flags & ~0x7
    bit = 8
    while remaining:
        if remaining & bit:
            names.append(f"0x{bit:x}")
            remaining &= ~bit
        bit <<= 1
    return " | ".join(names)


def _synchronize_land_borders(
    grids: Mapping[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], int]:
    """Give every duplicated LAND-cell border vertex one identical height.

    TES3 stores the same world-space edge vertex in both adjacent LAND
    records. Independent grading can otherwise leave a visible terrain crack;
    choosing the higher authored value also avoids reopening water holes.
    """

    locations: dict[tuple[int, int], list[tuple[tuple[int, int], int, int]]] = {}
    for cell in grids:
        for vertex_y in range(65):
            for vertex_x in range(65):
                if vertex_x not in {0, 64} and vertex_y not in {0, 64}:
                    continue
                world_key = (cell[0] * 64 + vertex_x, cell[1] * 64 + vertex_y)
                locations.setdefault(world_key, []).append(
                    (cell, vertex_y, vertex_x)
                )
    changed = {cell: 0 for cell in grids}
    for duplicates in locations.values():
        if len(duplicates) < 2:
            continue
        height = max(float(grids[cell][vy, vx]) for cell, vy, vx in duplicates)
        for cell, vertex_y, vertex_x in duplicates:
            if float(grids[cell][vertex_y, vertex_x]) != height:
                grids[cell][vertex_y, vertex_x] = height
                changed[cell] += 1
    return changed


def _validate_gate_platforms(
    deformed_heights: Mapping[tuple[int, int], np.ndarray],
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fail closed unless each complete gatehouse footprint is one LAND plane.

    The left/right split is measured across the road at the same entrance, so
    it catches the exact tilted-under-arch defect that is obvious in frontal
    close-ups. LAND is quantized to 8 GU; the composer publishes an 8-GU plane.
    """

    evidence: list[dict[str, Any]] = []
    for spec in specs:
        target = int(round(float(spec["target_thu"])))
        all_values: list[int] = []
        entrance_left: list[int] = []
        entrance_right: list[int] = []
        for cell, grid in deformed_heights.items():
            cell_min_x = (cell[0] + 95) * 8192.0
            cell_min_y = (cell[1] + 11) * 8192.0
            for vertex_y in range(65):
                for vertex_x in range(65):
                    x = cell_min_x + vertex_x * 128.0
                    y = cell_min_y + vertex_y * 128.0
                    point = Point(x, y)
                    if not spec["polygon"].covers(point):
                        continue
                    value = int(grid[vertex_y, vertex_x])
                    all_values.append(value)
                    relative = np.asarray([x, y], dtype=float) - spec["center"]
                    across = float(relative @ spec["wall_axis"])
                    along = abs(float(relative @ spec["road_axis"]))
                    if along <= float(spec["half_length_gu"]):
                        if across < 0.0:
                            entrance_left.append(value)
                        elif across > 0.0:
                            entrance_right.append(value)
        if not all_values or not entrance_left or not entrance_right:
            raise LandAuthoringError(
                f"gate {spec['gate_id']} platform has insufficient LAND samples"
            )
        if any(value != target for value in all_values):
            raise LandAuthoringError(
                f"gate {spec['gate_id']} platform is not level with gatehouse bottom"
            )
        if set(entrance_left) != {target} or set(entrance_right) != {target}:
            raise LandAuthoringError(
                f"gate {spec['gate_id']} entrance left/right heights differ"
            )
        evidence.append(
            {
                "gate_id": spec["gate_id"],
                "gatehouse_bottom_z_gu": target * 8,
                "platform_vertex_count": len(all_values),
                "platform_min_z_gu": min(all_values) * 8,
                "platform_max_z_gu": max(all_values) * 8,
                "entrance_left_vertex_count": len(entrance_left),
                "entrance_left_z_gu": entrance_left[0] * 8,
                "entrance_right_vertex_count": len(entrance_right),
                "entrance_right_z_gu": entrance_right[0] * 8,
            }
        )
    return evidence


def _validate_gate_road_paint(
    grids: Mapping[tuple[int, int], np.ndarray],
    specs: list[dict[str, Any]],
    road_raw_vtex: int,
) -> list[dict[str, Any]]:
    """Require an unbroken road-tile chain along each measured arch axis."""

    evidence: list[dict[str, Any]] = []
    for spec in specs:
        extent = float(spec["half_length_gu"]) + float(spec["approach_length_gu"])
        sampled_tiles: set[tuple[tuple[int, int], int, int]] = set()
        for along in np.arange(-extent, extent + 128.0, 128.0):
            point = spec["center"] + spec["road_axis"] * along
            x, y = float(point[0]), float(point[1])
            cell = (math.floor(x / 8192.0) - 95, math.floor(y / 8192.0) - 11)
            if cell not in grids:
                raise LandAuthoringError(
                    f"gate {spec['gate_id']} road centerline leaves authored LAND"
                )
            local_x = x - (cell[0] + 95) * 8192.0
            local_y = y - (cell[1] + 11) * 8192.0
            tile_x = min(15, max(0, int(math.floor(local_x / 512.0))))
            tile_y = min(15, max(0, int(math.floor(local_y / 512.0))))
            sampled_tiles.add((cell, tile_x, tile_y))
        wrong = [
            (cell, tile_x, tile_y)
            for cell, tile_x, tile_y in sampled_tiles
            if int(grids[cell][tile_y, tile_x]) != int(road_raw_vtex)
        ]
        if wrong:
            raise LandAuthoringError(
                f"gate {spec['gate_id']} road texture is discontinuous through arch"
            )
        evidence.append(
            {
                "gate_id": spec["gate_id"],
                "road_axis_tile_count": len(sampled_tiles),
                "road_axis_wrong_tile_count": 0,
                "road_raw_vtex": int(road_raw_vtex),
            }
        )
    return evidence


def author_land_records(
    circulation: Mapping[str, Any],
    palette: Mapping[str, Any],
    source_land: Mapping[tuple[int, int], espland.LandRecord],
    *,
    source_plugin: str,
    seated_objects: Mapping[str, Any] | None = None,
    wall_doc: Mapping[str, Any] | None = None,
    grading_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated masterless LAND JSON document."""

    source_erase = list(circulation.get("source_road_erase_requests") or [])
    broad = list(circulation.get("land_paint_requests") or [])
    deformations = []
    for placement in (seated_objects or {}).get("placements", []):
        seating = placement.get("terrain_seating") or {}
        request = seating.get("terrain_deformation")
        if isinstance(request, Mapping):
            deformations.append(request)
    policy = grading_policy or {}
    minimum_height_thu = float(policy.get("minimum_graded_land_height_gu", 0.0)) / 8.0
    wall_grades = _wall_grade_specs(wall_doc, policy)
    road_grades = _road_grade_specs(broad, source_land, policy)
    gate_grades = _gate_grade_specs(wall_doc, source_land, policy, broad)
    grade_specs = road_grades + gate_grades
    grade_requests = [
        {"polygon": [[list(point) for point in spec["outer"].exterior.coords]]}
        for spec in wall_grades + grade_specs
    ]
    all_requests = source_erase + broad + deformations + grade_requests
    cells = _cells_for_requests(all_requests)
    if not cells:
        raise LandAuthoringError("no affected LAND cells")
    missing = sorted(cell for cell in cells if cell not in source_land)
    if missing:
        raise LandAuthoringError(f"source LAND missing affected cells {missing[:8]}")
    source_masks = _paint_mask(source_erase, cells)
    assignments = load_surface_assignments(palette)
    grass = assignments["base"]
    broad_assignments: list[tuple[list[dict[str, Any]], SurfaceAssignment]] = []
    for request in broad:
        assignment_row = request.get("surface_assignment")
        if not isinstance(assignment_row, Mapping):
            raise LandAuthoringError(f"broad paint request {request.get('realization_id')} has no surface assignment")
        raw = assignment_row.get("raw_vtex")
        assignment = next(
            (candidate for candidate in assignments.values() if candidate.raw_vtex == int(raw)),
            None,
        )
        if assignment is None:
            raise LandAuthoringError(
                f"broad paint request {request.get('realization_id')} uses an unclosed raw VTEX {raw}"
            )
        broad_assignments.append(([request], assignment))
    gate_paint_requests = []
    for raw_vtex in sorted({int(spec["road_raw_vtex"]) for spec in gate_grades}):
        assignment = next(
            (candidate for candidate in assignments.values()
             if int(candidate.raw_vtex) == raw_vtex),
            None,
        )
        if assignment is None:
            raise LandAuthoringError(
                f"gate approaches use unclosed road raw VTEX {raw_vtex}"
            )
        requests = [
            {
                "realization_id": f"gate_approach:{spec['gate_id']}",
                "polygon": [[list(point) for point in spec["paint_polygon"].exterior.coords]],
                "coverage_mode": "intersects",
            }
            for spec in gate_grades
            if int(spec["road_raw_vtex"]) == raw_vtex
        ]
        gate_paint_requests.extend(requests)
        broad_assignments.append((requests, assignment))
    grids: dict[tuple[int, int], np.ndarray] = {}
    deformed_heights: dict[tuple[int, int], np.ndarray] = {}
    changed: dict[tuple[int, int], dict[str, int]] = {}
    for cell in sorted(cells, key=lambda item: (item[1], item[0])):
        source = source_land[cell]
        if source.texture_indices is None:
            raise LandAuthoringError(f"source LAND {cell} has no VTEX")
        deformed, deformation_changes = _deform_heights(
            source.heights_thu, cell, deformations, minimum_height_thu)
        wall_grading_changes = _apply_wall_grade_specs(
            deformed, cell, wall_grades, minimum_height_thu
        )
        grading_changes = wall_grading_changes + _apply_grade_specs(
            deformed, cell, grade_specs, minimum_height_thu
        )
        grading_changes += _clamp_wall_coverage_specs(
            deformed, cell, wall_grades, minimum_height_thu
        )
        deformed_heights[cell] = deformed
        grid = np.asarray(source.texture_indices, dtype=np.uint16).reshape(16, 16).copy()
        before = grid.copy()
        grid[source_masks[cell]] = np.uint16(grass.raw_vtex)
        broad_tile_count = 0
        for requests, assignment in broad_assignments:
            mask = _paint_mask(requests, cells)[cell]
            grid[mask] = np.uint16(assignment.raw_vtex)
            broad_tile_count += int(np.count_nonzero(mask))
        grids[cell] = grid
        changed[cell] = {
            "source_road_erase_tiles": int(np.count_nonzero(source_masks[cell])),
            "authored_civic_tiles": broad_tile_count,
            "changed_tiles": int(np.count_nonzero(grid != before)),
            "lowered_height_vertices": int(deformation_changes),
            "graded_height_vertices": int(grading_changes),
        }

    border_changes = _synchronize_land_borders(deformed_heights)
    for cell, count in border_changes.items():
        changed[cell]["graded_height_vertices"] += int(count)
    gate_platform_evidence = _validate_gate_platforms(
        deformed_heights, gate_grades
    )
    gate_road_paint_evidence = []
    for spec in gate_grades:
        gate_road_paint_evidence.extend(
            _validate_gate_road_paint(
                grids, [spec], int(spec["road_raw_vtex"])
            )
        )

    required_raw = sorted({int(value) for grid in grids.values() for value in grid.reshape(-1) if int(value) > 0})
    ltex_by_index: dict[int, dict[str, Any]] = {}
    source_ltex = {row.index: row for row in espland.load_ltex(source_plugin).values()}
    for raw in required_raw:
        index = raw - 1
        if index == grass.ltex_index:
            ltex_by_index[index] = {"record_id": grass.ltex_id, "file_name": grass.file_name}
        elif assignment := next(
            (candidate for candidate in assignments.values() if candidate.ltex_index == index), None
        ):
            ltex_by_index[index] = {"record_id": assignment.ltex_id, "file_name": assignment.file_name}
        elif index in source_ltex:
            ltex_by_index[index] = {
                "record_id": source_ltex[index].record_id,
                "file_name": source_ltex[index].file_name,
            }
        else:
            raise LandAuthoringError(f"no LTEX provenance for raw VTEX {raw}")

    document = tes3json.new_plugin({
        "author": "Procedural Tamriel",
        "description": "Falkreath townlayout circulation LAND preparation",
        "masters": [],
        "num_objects": len(ltex_by_index) + len(cells),
    })
    for index in sorted(ltex_by_index):
        row = ltex_by_index[index]
        document.append(tes3json.build_ltex(row["record_id"], index, row["file_name"]))
    for cell in sorted(cells, key=lambda item: (item[1], item[0])):
        source = source_land[cell]
        if any(value is None for value in (source.heights_thu, source.vertex_normals, source.world_map_data, source.vertex_colors)):
            raise LandAuthoringError(f"source LAND {cell} has incomplete payload")
        serialized = espland.transpose_vtex_openmw_to_serialized(
            tuple(int(value) for value in grids[cell].reshape(-1)))
        document.append(tes3json.build_land(
            cell, tuple(tuple(int(value) for value in row)
                        for row in deformed_heights[cell].tolist()), heights_in_thu=True,
            landscape_flags=_flags(source.flags),
            vertex_normals=source.vertex_normals,
            world_map_data=source.world_map_data,
            vertex_colors=source.vertex_colors,
            texture_indices=serialized,
        ))
    issues = tes3json.validate(document)
    if issues:
        raise LandAuthoringError("tes3json validation failed: " + "; ".join(map(str, issues[:8])))
    raw_counts = {str(raw): int(sum(np.count_nonzero(grid == raw) for grid in grids.values())) for raw in required_raw}
    return {
        "records": document,
        "authoring_evidence": {
            "stage_id": "falkreath_townlayout_land_records_v1",
            "source_plugin": source_plugin,
            "plugin_scope": {"masters": [], "file_type": "Esp"},
            "paint_order": ["lower_building_terrain_to_door_height", "grade_wall_foundations", "smooth_road_cross_sections", "flatten_gate_platforms", "source_road_erase_to_grass", "authored_civic_to_hr_road"],
            "record_counts": {"ltex": len(ltex_by_index), "land": len(cells)},
            "affected_cells": [list(cell) for cell in sorted(cells, key=lambda item: (item[1], item[0]))],
            "source_road_erase_tile_count": int(sum(row["source_road_erase_tiles"] for row in changed.values())),
            "authored_civic_tile_count": int(sum(row["authored_civic_tiles"] for row in changed.values())),
            "changed_tile_count": int(sum(row["changed_tiles"] for row in changed.values())),
            "terrain_deformation_count": len(deformations),
            "lowered_height_vertex_count": int(sum(row["lowered_height_vertices"] for row in changed.values())),
            "graded_height_vertex_count": int(sum(row["graded_height_vertices"] for row in changed.values())),
            "synchronized_border_vertex_count": int(sum(border_changes.values())),
            "road_grade_count": len(road_grades),
            "gate_platform_count": len(gate_grades),
            "gate_platform_evidence": gate_platform_evidence,
            "gate_road_paint_request_count": len(gate_paint_requests),
            "gate_road_paint_evidence": gate_road_paint_evidence,
            "wall_foundation_grade_count": len(wall_grades),
            "raw_counts": raw_counts,
            "per_cell": {f"{cell[0]},{cell[1]}": changed[cell] for cell in sorted(changed)},
            "narrow_geometry_deferred_count": len(circulation.get("terrain_following_requests") or []),
            "tes3json_issue_count": 0,
        },
    }


def author_from_paths(
    circulation_path: Path,
    palette_path: Path,
    source_plugin: Path,
    *,
    source_land_json: Path | None = None,
    output_path: Path,
    seated_objects_path: Path | None = None,
    wall_doc_path: Path | None = None,
    grading_policy_path: Path | None = None,
) -> dict[str, Any]:
    circulation = _load_json(circulation_path, "circulation realization")
    palette = _load_json(palette_path, "region palette")
    source_land = espland.load_land(source_plugin)
    if source_land_json is not None:
        try:
            source_payload = json.loads(source_land_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LandAuthoringError(
                f"cannot read source LAND JSON {source_land_json}: {exc}"
            ) from exc
        if not isinstance(source_payload, (dict, list)):
            raise LandAuthoringError("source LAND JSON must be a tes3conv document")
        source_land = {
            **source_land,
            **tes3json.land_records_from_json(
                source_payload
            ),
        }
    seated_objects = (_load_json(seated_objects_path, "seated objects")
                      if seated_objects_path is not None else None)
    wall_doc = (_load_json(wall_doc_path, "wall document")
                if wall_doc_path is not None else None)
    grading_policy = (_load_json(grading_policy_path, "terrain grading policy")
                      if grading_policy_path is not None else None)
    product = author_land_records(circulation, palette, source_land,
                                  source_plugin=str(source_plugin),
                                  seated_objects=seated_objects,
                                  wall_doc=wall_doc,
                                  grading_policy=grading_policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(product["records"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(product["authoring_evidence"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    product["authoring_manifest_path"] = str(manifest_path)
    return product
