"""Wall-footprint exclusions shared by settlement scatter and rendering.

Pipeline position: after wall-aware city layout and raw scatter generation,
before groundcover and town-plugin assembly. Inputs are the composed wall JSON,
the matching R13 city layout, and seated-object frame origin. Outputs are a
filtered scatter document and exclusion counts. Wall member footprints are
measured z-slice polygons in plan GU; scatter AABBs are global GU and are
converted through the seated terrain frame before intersection.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def _ref_footprint(ref: Mapping[str, Any]):
    aabb = (ref.get("bbox") or {}).get("world_aabb_gu") or {}
    minimum = aabb.get("min")
    maximum = aabb.get("max")
    if (
        isinstance(minimum, list)
        and len(minimum) >= 2
        and isinstance(maximum, list)
        and len(maximum) >= 2
    ):
        return Polygon(
            [
                [float(minimum[0]), float(minimum[1])],
                [float(maximum[0]), float(minimum[1])],
                [float(maximum[0]), float(maximum[1])],
                [float(minimum[0]), float(maximum[1])],
            ]
        )
    position = ref.get("position_gu")
    if isinstance(position, list) and len(position) >= 2:
        return Point(float(position[0]), float(position[1]))
    raise ValueError(
        f"scatter ref has no position or measured world AABB: {ref.get('ref_id')}"
    )


def _wall_geometry(
    wall: Mapping[str, Any], city_layout: Mapping[str, Any]
) -> tuple[Polygon, Polygon]:
    origin = [float(value) for value in wall.get("origin_gu", [0.0, 0.0])]
    parts = []
    for member in wall.get("members", []):
        footprint = member.get("footprint_xy_rel")
        if not isinstance(footprint, list) or len(footprint) < 3:
            raise ValueError(
                f"wall member lacks measured footprint: {member.get('source_id')}"
            )
        parts.append(
            Polygon(
                [
                    [origin[0] + float(point[0]), origin[1] + float(point[1])]
                    for point in footprint
                ]
            )
        )
    if not parts:
        raise ValueError("wall document has no measured members")
    wall_mesh = unary_union(parts)
    polygon = (city_layout.get("inner_wall") or {}).get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("city layout lacks inner_wall.polygon")
    wall_domain = Polygon([[float(point[0]), float(point[1])] for point in polygon])
    if wall_domain.is_empty or not wall_domain.is_valid or wall_domain.area <= 0.0:
        raise ValueError("city layout inner_wall.polygon is invalid")
    return wall_mesh, wall_domain


def filter_scatter_document(
    scatter: Mapping[str, Any],
    wall: Mapping[str, Any],
    city_layout: Mapping[str, Any],
    frame_origin_gu: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject refs touching wall meshes or lying inside the fitted wall."""

    if len(frame_origin_gu) < 2:
        raise ValueError("seated terrain frame origin must contain two values")
    wall_mesh, wall_domain = _wall_geometry(wall, city_layout)
    origin_x, origin_y = float(frame_origin_gu[0]), float(frame_origin_gu[1])
    excluded: dict[str, Any] = {
        "wall_mesh": 0,
        "wall_domain": 0,
        "by_category": {},
    }
    kept_by_category: dict[str, int] = {}
    kept_total = 0
    filtered_cells = []
    density = scatter.get("density")
    if not isinstance(density, Mapping) or not isinstance(density.get("cells"), list):
        raise ValueError("scatter document has no density.cells list")
    for cell in density["cells"]:
        if not isinstance(cell, Mapping):
            continue
        refs = []
        for ref in cell.get("refs", []):
            if not isinstance(ref, Mapping):
                continue
            footprint = _ref_footprint(ref)
            plan_footprint = Polygon(
                [
                    [float(x) - origin_x, float(y) - origin_y]
                    for x, y in footprint.exterior.coords
                ]
            ) if hasattr(footprint, "exterior") else Point(
                float(footprint.x) - origin_x, float(footprint.y) - origin_y
            )
            category = str(ref.get("category", "unknown"))
            reason = None
            if plan_footprint.intersection(wall_mesh).area > 1.0:
                reason = "wall_mesh"
            elif plan_footprint.intersection(wall_domain).area > 1.0:
                reason = "wall_domain"
            if reason is None:
                refs.append(dict(ref))
                kept_total += 1
                kept_by_category[category] = kept_by_category.get(category, 0) + 1
                continue
            excluded[reason] += 1
            by_category = excluded["by_category"]
            by_category.setdefault(category, {})
            by_category[category][reason] = by_category[category].get(reason, 0) + 1
        row = dict(cell)
        row["refs"] = refs
        filtered_cells.append(row)
    filtered_density = dict(density)
    filtered_density["cells"] = filtered_cells
    filtered = dict(scatter)
    filtered["density"] = filtered_density
    placement_stats = filtered.get("placement_stats")
    if isinstance(placement_stats, Mapping):
        placement_stats = dict(placement_stats)
        placement_stats["total_refs"] = kept_total
        placement_stats["by_category"] = dict(sorted(kept_by_category.items()))
        filtered["placement_stats"] = placement_stats
    filtered["wall_exclusion"] = {
        "rule": "reject measured scatter AABBs intersecting wall meshes or fitted inner-wall domain",
        "frame_origin_gu": [origin_x, origin_y],
        "excluded": excluded,
    }
    return filtered, excluded


def wall_surface_exclusions(wall: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return measured wall polygons in plan coordinates for clearing queries."""

    origin = [float(value) for value in wall.get("origin_gu", [0.0, 0.0])]
    result = []
    for member in wall.get("members", []):
        footprint = member.get("footprint_xy_rel")
        if not isinstance(footprint, list) or len(footprint) < 3:
            raise ValueError(f"wall member lacks measured footprint: {member.get('source_id')}")
        result.append(
            {
                "kind": "polygon",
                "rings": [
                    [
                        [origin[0] + float(point[0]), origin[1] + float(point[1])]
                        for point in footprint
                    ]
                ],
                "source_id": f"wall:{member.get('object_id') or member.get('source_id')}",
            }
        )
    return result
