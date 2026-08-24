"""Realize accepted townlayout circulation into terrain-aware surface requests.

Purpose
-------
Convert the accepted R13 roads, civic surfaces, and door aprons into the
intermediate geometry contract used by the future LAND/mesh authoring stage.
Existing source-road tiles inside the city domain are first erased to the
configured base-grass assignment. Roads and authored civic polygons use the
semantic assignments declared by the active palette. Broad civic polygons become LAND paint requests;
narrow alleys and aprons become terrain-following geometry requests.

Inputs
-------
The accepted R13 city layout, the seated stamp-object product, the closed
Falkreath region palette, and the survey-backed terrain field.

Outputs
-------
A deterministic ``r13_circulation_realization_v1`` document containing road
polygons, LAND paint requests, terrain-following polygons with sampled Z
vertices, and complete source-ID coverage evidence. This stage does not write
LAND records, meshes, or an ESP.

Invariants
----------
Every source road, civic surface, and door apron is represented exactly once.
Roads and civic surfaces always use the explicit palette road assignment.
Source semantic classes are retained as provenance and do not select a
different texture until a later civic-surface palette policy is accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from ..cityplace_contracts import TerrainField
from ..cityscape_vtex import SurfaceAssignment, load_surface_assignments
from ..road_semantics import load_road_assignments, road_class_for_plan_road


LAND_AREA_THRESHOLD_GU2 = 750_000.0


class CirculationRealizationError(ValueError):
    """Raised when circulation geometry or palette closure is incomplete."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CirculationRealizationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CirculationRealizationError(f"{label} must be a JSON object")
    return value


def _rings(row: Mapping[str, Any]) -> list[list[list[float]]]:
    polygons = row.get("polygon")
    if not isinstance(polygons, list) or not polygons:
        raise CirculationRealizationError(f"{row.get('surface_id') or row.get('apron_id')} has no polygon")
    result = []
    for ring in polygons:
        if not isinstance(ring, list) or len(ring) < 3:
            raise CirculationRealizationError("surface polygon ring is malformed")
        result.append([[float(point[0]), float(point[1])] for point in ring])
    return result


def _polygon_area(rings: Sequence[Sequence[Sequence[float]]]) -> float:
    return float(sum(Polygon(ring).area for ring in rings))


def _assignment(assignments: Mapping[str, SurfaceAssignment], source_class: str) -> tuple[str, SurfaceAssignment]:
    if source_class == "public_packed_earth":
        canonical = PUBLIC_PACKED_EARTH_ALIAS
    else:
        canonical = source_class
    result = assignments.get(canonical)
    if result is None:
        raise CirculationRealizationError(
            f"surface class {source_class!r} has no closed palette assignment")
    return canonical, result


def _terrain_vertices(
    rings: Sequence[Sequence[Sequence[float]]], field: TerrainField
) -> list[list[float]]:
    vertices: list[list[float]] = []
    for ring in rings:
        for point in ring:
            sample = field.sample(float(point[0]), float(point[1]))
            vertices.append([float(point[0]), float(point[1]), float(sample.height_gu)])
    return vertices


def _assignment_dict(assignment: SurfaceAssignment) -> dict[str, Any]:
    return assignment.to_dict()


def _geometry_rings(geometry) -> list[list[list[float]]]:
    polygons = [geometry] if geometry.geom_type == "Polygon" else [part for part in geometry.geoms if part.geom_type == "Polygon"]
    return [
        [[float(x), float(y)] for x, y in polygon.exterior.coords]
        for polygon in polygons if polygon.area > 0.0
    ]


def realize_circulation(
    layout: Mapping[str, Any],
    seated_objects: Mapping[str, Any],
    palette: Mapping[str, Any],
    field: TerrainField,
    source_roads: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the circulation realization contract from accepted inputs."""

    if layout.get("stage_id") != "r13_city_layout":
        raise CirculationRealizationError("circulation realization requires r13_city_layout")
    if seated_objects.get("stage_id") != "townlayout_stamp_objects_seated_v1":
        raise CirculationRealizationError("circulation realization requires seated stamp objects")
    assignments = load_surface_assignments(palette)
    road_classes = load_road_assignments(palette)
    grass_assignment = assignments.get("base")
    if grass_assignment is None:
        raise CirculationRealizationError("closed palette has no base grass assignment")
    settlement_dirt_assignment = assignments.get("settlement_dirt")
    if settlement_dirt_assignment is None:
        raise CirculationRealizationError("closed palette has no settlement_dirt assignment")

    land_paint: list[dict[str, Any]] = []
    follow_geometry: list[dict[str, Any]] = []
    source_surface_modes: dict[str, str] = {}
    road_rows: list[dict[str, Any]] = []
    source_road_erase: list[dict[str, Any]] = []

    city_domain = Polygon(layout.get("city_domain", []))
    if city_domain.is_empty or not city_domain.is_valid:
        raise CirculationRealizationError("city domain is not a valid polygon")
    for tile in (source_roads or {}).get("road_tiles", []):
        bounds = tile.get("plan_gu_bounds") or {}
        minimum = bounds.get("min")
        maximum = bounds.get("max_exclusive")
        if not isinstance(minimum, list) or not isinstance(maximum, list):
            raise CirculationRealizationError("source road tile has no plan bounds")
        clipped = box(float(minimum[0]), float(minimum[1]),
                      float(maximum[0]), float(maximum[1])).intersection(city_domain)
        rings = _geometry_rings(clipped)
        if not rings:
            continue
        source_id = str(tile.get("tile_id"))
        source_road_erase.append({
            "realization_id": f"source_road_erase:{source_id}",
            "source_id": source_id,
            "source_raw_vtex": int(tile["raw_vtex"]),
            "source_tile_grid": [int(tile["grid"][0]), int(tile["grid"][1])],
            "source_tile_local": [int(tile["cell_local_tile"][0]), int(tile["cell_local_tile"][1])],
            "geometry_kind": "land_paint_polygon",
            "polygon": rings,
            "surface_assignment": _assignment_dict(grass_assignment),
            "terrain_vertices": _terrain_vertices(rings, field),
        })

    for road in layout.get("roads", []):
        road_id = str(road.get("road_id"))
        try:
            road_assignment = road_class_for_plan_road(
                road, road_classes, palette.get("road_class_by_hierarchy")
            )
        except ValueError as exc:
            raise CirculationRealizationError(str(exc)) from exc
        road_class = road_assignment.road_class
        polyline = [[float(point[0]), float(point[1])] for point in road.get("polyline", [])]
        if len(polyline) < 2:
            raise CirculationRealizationError(f"road {road_id} has fewer than two points")
        width = float(road.get("clear_width_gu", 0.0))
        if width <= 0.0:
            raise CirculationRealizationError(f"road {road_id} has no positive clear width")
        polygon = LineString(polyline).buffer(width / 2.0, cap_style=2, join_style=2)
        rings = [[[float(x), float(y)] for x, y in polygon.exterior.coords]]
        row = {
            "realization_id": f"road:{road_id}",
            "source_id": road_id,
            "role": road.get("hierarchy", "road"),
            "geometry_kind": "land_paint_polygon",
            "polygon": rings,
            "centerline": polyline,
            "width_gu": width,
            "road_class": road_class,
            "surface_assignment": _assignment_dict(road_assignment),
            "terrain_vertices": _terrain_vertices(rings, field),
        }
        land_paint.append(row)
        source_surface_modes[road_id] = "land_paint"
        road_rows.append(row)

    for surface in layout.get("surfaces", []):
        surface_id = str(surface.get("surface_id"))
        rings = _rings(surface)
        area = _polygon_area(rings)
        source_class = str(surface.get("surface_class", ""))
        role = str(surface.get("role", "surface"))
        use_land = role in {"plaza", "front_courtyard", "back_court"} or area >= LAND_AREA_THRESHOLD_GU2
        row = {
            "realization_id": f"surface:{surface_id}",
            "source_id": surface_id,
            "role": role,
            "source_surface_class": source_class,
            "canonical_surface_class": source_class,
            "geometry_kind": "land_paint_polygon" if use_land else "terrain_following_polygon",
            "polygon": rings,
            "area_gu2": area,
            "surface_assignment": _assignment_dict(
                assignments.get(source_class, settlement_dirt_assignment)
            ),
            "terrain_vertices": _terrain_vertices(rings, field),
        }
        (land_paint if use_land else follow_geometry).append(row)
        source_surface_modes[surface_id] = row["geometry_kind"]

    apron_ids: set[str] = set()
    for apron in layout.get("door_aprons", []) + layout.get("rear_aprons", []):
        apron_id = str(apron.get("apron_id"))
        if apron_id in apron_ids:
            raise CirculationRealizationError(f"duplicate apron {apron_id}")
        apron_ids.add(apron_id)
        rings = _rings(apron)
        source_class = "settlement_dirt"
        follow_geometry.append({
            "realization_id": f"apron:{apron_id}",
            "source_id": apron_id,
            "placement_id": apron.get("placement_id"),
            "role": "door_apron" if apron_id.startswith("front_apron:") else "rear_apron",
            "source_surface_class": source_class,
            "canonical_surface_class": source_class,
            "geometry_kind": "terrain_following_polygon",
            "polygon": rings,
            "area_gu2": _polygon_area(rings),
            "surface_assignment": _assignment_dict(
                assignments.get(source_class, settlement_dirt_assignment)
            ),
            "terrain_vertices": _terrain_vertices(rings, field),
        })

    source_surface_modes = dict(sorted(source_surface_modes.items()))
    land_paint.sort(key=lambda row: row["realization_id"])
    follow_geometry.sort(key=lambda row: row["realization_id"])
    road_rows.sort(key=lambda row: row["realization_id"])
    source_road_erase.sort(key=lambda row: row["realization_id"])
    expected_surface_ids = {str(row["surface_id"]) for row in layout.get("surfaces", [])}
    if expected_surface_ids != {key for key in source_surface_modes if key not in {str(r["road_id"]) for r in layout.get("roads", [])}}:
        raise CirculationRealizationError("surface coverage ledger does not match layout surfaces")
    if len(apron_ids) != len(layout.get("door_aprons", [])) + len(layout.get("rear_aprons", [])):
        raise CirculationRealizationError("apron coverage ledger does not match layout aprons")

    return {
        "schema_version": 1,
        "stage_id": "r13_circulation_realization_v1",
        "source": {
            "layout_stage_id": layout.get("stage_id"),
            "layout_id": layout.get("layout_id") or layout.get("candidate_id"),
            "seated_object_stage_id": seated_objects.get("stage_id"),
            "terrain_field": field.contract_dict(),
        },
        "policy": {
            "road_surface_classes": {
                name: _assignment_dict(assignments[name])
                for name in road_classes
            },
            "civic_surface_override": _assignment_dict(settlement_dirt_assignment),
            "source_road_erase_surface": _assignment_dict(grass_assignment),
            "land_area_threshold_gu2": LAND_AREA_THRESHOLD_GU2,
            "narrow_surface_mode": "terrain_following_polygon",
            "outside_explicit_circulation": "source_terrain_unchanged",
        },
        "roads": road_rows,
        "source_road_erase_requests": source_road_erase,
        "land_paint_requests": land_paint,
        "terrain_following_requests": follow_geometry,
        "coverage": {
            "road_count": len(road_rows),
            "source_road_erase_count": len(source_road_erase),
            "surface_count": len(layout.get("surfaces", [])),
            "apron_count": len(apron_ids),
            "land_paint_count": len(land_paint),
            "terrain_following_count": len(follow_geometry),
            "surface_modes": source_surface_modes,
        },
        "palette_assignments": {
            name: _assignment_dict(assignment) for name, assignment in sorted(assignments.items())
        },
    }


def realize_from_paths(
    layout_path: Path,
    seated_objects_path: Path,
    palette_path: Path,
    survey_path: Path,
    field_path: Path,
    source_roads_path: Path,
) -> dict[str, Any]:
    layout = _load_json(layout_path, "city layout")
    seated = _load_json(seated_objects_path, "seated stamp objects")
    palette = _load_json(palette_path, "region palette")
    source_roads = _load_json(source_roads_path, "source road mask")
    survey = _load_json(survey_path, "site survey")
    field = TerrainField.from_npz(field_path, survey=survey, field_pass="planned")
    return realize_circulation(layout, seated, palette, field, source_roads)
