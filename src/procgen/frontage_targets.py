"""Shared circulation-target geometry for Cityforge frontage fitting.

Pipeline position
------------------
This module is the pure plan-stage target boundary shared by ``plan_sketch``
and the v1 frontage fitter.  It converts the accepted aligned-road product and
the authored roads/spaces into the same target records used by the visual
planner.  It contains no rendering, image I/O, Blender, TES3 writing, or
subprocess calls.

The distance rules intentionally mirror the existing visual-planner contract:
authored roads measure reach to their centreline, source roads measure reach
to their practical path edge, and polygon targets measure distance to their
ring (zero while inside).  Keeping these helpers in one small module prevents
the fitter's report and the resolved sketch's ``checks.json`` from disagreeing
about a named target.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .aligned_roads import SOURCE_ROAD_PRACTICAL_PATH_FRACTION
from .cityplan import point_in_ring, point_seg_distance


def nearest_polyline(
    point: Sequence[float],
    polyline: Sequence[Sequence[float]],
) -> tuple[float, tuple[float, float]]:
    """Return the nearest distance and point on a non-empty polyline."""

    best = (float("inf"), (float(polyline[0][0]), float(polyline[0][1])))
    for first, second in zip(polyline, polyline[1:]):
        ax, ay = float(first[0]), float(first[1])
        bx, by = float(second[0]), float(second[1])
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq == 0.0 else max(
            0.0,
            min(
                1.0,
                ((float(point[0]) - ax) * dx + (float(point[1]) - ay) * dy)
                / length_sq,
            ),
        )
        nearest = (ax + t * dx, ay + t * dy)
        distance = math.hypot(float(point[0]) - nearest[0], float(point[1]) - nearest[1])
        if distance < best[0]:
            best = (distance, nearest)
    return best


def nearest_point_on_ring(
    point: Sequence[float],
    ring: Sequence[Sequence[float]],
) -> tuple[float, tuple[float, float]]:
    """Return ring distance and nearest point; points inside have distance 0."""

    point_xy = (float(point[0]), float(point[1]))
    if point_in_ring(point_xy, list(ring)):
        return 0.0, point_xy
    best = (float("inf"), (float(ring[0][0]), float(ring[0][1])))
    closed = list(ring) + [list(ring[0])]
    for first, second in zip(closed, closed[1:]):
        distance = point_seg_distance(point_xy, first, second)
        if distance < best[0]:
            ax, ay = float(first[0]), float(first[1])
            bx, by = float(second[0]), float(second[1])
            dx, dy = bx - ax, by - ay
            length_sq = dx * dx + dy * dy
            t = 0.0 if length_sq == 0.0 else max(
                0.0,
                min(
                    1.0,
                    ((point_xy[0] - ax) * dx + (point_xy[1] - ay) * dy) / length_sq,
                ),
            )
            best = (distance, (ax + t * dx, ay + t * dy))
    return best


def reach_distance(point: Sequence[float], target: Mapping[str, Any]) -> float:
    """Return the plan-stage reach distance for one target."""

    polyline = target.get("polyline")
    if isinstance(polyline, list) and len(polyline) >= 2:
        distance = nearest_polyline(point, polyline)[0]
        if target.get("kind") == "existing_source_road":
            half = float(target.get("width_gu", 0.0)) * SOURCE_ROAD_PRACTICAL_PATH_FRACTION / 2.0
            distance = max(0.0, distance - half)
        return distance
    polygon = target.get("polygon")
    if isinstance(polygon, list) and len(polygon) >= 3:
        return nearest_point_on_ring(point, polygon)[0]
    return float("inf")


def target_nearest_point(
    point: Sequence[float], target: Mapping[str, Any], *, path_edge: bool = False
) -> tuple[float, tuple[float, float] | None]:
    """Return reach distance and the point representing the same target.

    By default the returned point is the established target point: authored
    roads use their centreline, source roads use their practical path edge,
    and polygons use their nearest ring point (or the point itself while
    inside).  ``path_edge=True`` additionally moves authored-road points to
    the declared corridor edge; this is used only by the fitter's safety-gap
    construction.
    """

    polyline = target.get("polyline")
    if isinstance(polyline, list) and len(polyline) >= 2:
        centre_distance, nearest = nearest_polyline(point, polyline)
        if target.get("kind") != "existing_source_road" and not path_edge:
            return centre_distance, nearest
        half = float(target.get("width_gu", 0.0)) / 2.0
        if target.get("kind") == "existing_source_road":
            half *= SOURCE_ROAD_PRACTICAL_PATH_FRACTION
        if centre_distance <= half:
            return 0.0, (float(point[0]), float(point[1]))
        edge_distance = centre_distance - half
        scale = edge_distance / centre_distance if centre_distance > 0.0 else 0.0
        edge = (
            float(point[0]) + (nearest[0] - float(point[0])) * scale,
            float(point[1]) + (nearest[1] - float(point[1])) * scale,
        )
        return edge_distance, edge
    polygon = target.get("polygon")
    if isinstance(polygon, list) and len(polygon) >= 3:
        return nearest_point_on_ring(point, polygon)
    return float("inf"), None


def nearest_target(
    point: Sequence[float], targets: Mapping[str, Mapping[str, Any]]
) -> tuple[float, str, dict[str, Any]]:
    """Return the nearest target with the existing ``(distance, id)`` tie-break."""

    best = (float("inf"), "", {})
    for target_id in sorted(targets):
        target = targets[target_id]
        distance = reach_distance(point, target)
        if distance < best[0]:
            best = (distance, target_id, dict(target))
    return best


def build_target_map(
    site: Mapping[str, Any], origin: Sequence[float], network: Any
) -> dict[str, dict[str, Any]]:
    """Build existing source-road targets using the aligned-road consumer API."""

    targets: dict[str, dict[str, Any]] = {}
    for row in site.get("source_roads", []):
        edge_id = row.get("edge_id")
        if not isinstance(edge_id, str):
            continue
        try:
            edge = network.edge(edge_id)
        except Exception:  # noqa: BLE001 - unresolved edges fail closed upstream
            continue
        targets[edge_id] = {
            "kind": "existing_source_road",
            "polyline": [network.to_site_local(point, origin) for point in edge.smooth_gu_polyline],
            "width_gu": float(edge.estimated_width_gu),
        }
    return targets


def road_target(record: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """Return an authored road/alley target in plan-frame coordinates."""

    return {
        "kind": kind,
        "polyline": record["polyline_plan_gu"],
        "width_gu": float(record["width_gu"]),
    }


def authored_targets(
    sketch: Mapping[str, Any], origin: Sequence[float]
) -> dict[str, dict[str, Any]]:
    """Build authored road, alley, plaza, and court targets exactly once."""

    targets: dict[str, dict[str, Any]] = {}
    for road in sketch.get("roads", []):
        polyline = [
            [float(point[0]) - float(origin[0]), float(point[1]) - float(origin[1])]
            for point in road["points"]
        ]
        record = {"polyline_plan_gu": polyline, "width_gu": road["width_gu"]}
        targets[road["id"]] = road_target(
            record, "authored_road" if road["kind"] == "street" else "alley"
        )
    for space in sketch.get("spaces", []):
        polygon = [
            [float(point[0]) - float(origin[0]), float(point[1]) - float(origin[1])]
            for point in space["polygon"]
        ]
        targets[space["id"]] = {
            "kind": "road_surface_polygon" if space["kind"] == "plaza" else "shared_court",
            "polygon": polygon,
            "width_gu": 0.0,
        }
    return targets


def corridor_rings(target: Mapping[str, Any]) -> list[list[tuple[float, float]]]:
    """Return the established rectangular hard-clearance bands for a road target."""

    polyline = target.get("polyline")
    if not isinstance(polyline, list) or len(polyline) < 2:
        return []
    width = max(0.0, float(target.get("width_gu", 0.0)))
    if target.get("kind") == "existing_source_road":
        width *= SOURCE_ROAD_PRACTICAL_PATH_FRACTION
    if width == 0.0:
        return []
    half = width / 2.0
    rings: list[list[tuple[float, float]]] = []
    for first, second in zip(polyline, polyline[1:]):
        dx, dy = float(second[0]) - float(first[0]), float(second[1]) - float(first[1])
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        nx, ny = -dy / length * half, dx / length * half
        rings.append(
            [
                (float(first[0]) + nx, float(first[1]) + ny),
                (float(second[0]) + nx, float(second[1]) + ny),
                (float(second[0]) - nx, float(second[1]) - ny),
                (float(first[0]) - nx, float(first[1]) - ny),
            ]
        )
    return rings


__all__ = [
    "SOURCE_ROAD_PRACTICAL_PATH_FRACTION",
    "authored_targets",
    "build_target_map",
    "corridor_rings",
    "nearest_point_on_ring",
    "nearest_polyline",
    "nearest_target",
    "reach_distance",
    "road_target",
    "target_nearest_point",
]
