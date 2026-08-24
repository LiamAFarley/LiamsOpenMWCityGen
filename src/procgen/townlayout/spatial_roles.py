"""Derive deliberate plaza/court anchors and dense rear-quarter infill domains.

The old R10 implementation classified every residual polygon and thereby turned
unused terrain into dozens of alleged courts. This module instead preserves the
R5 layout, expands its real stamp doors, preserves two bounded civic spaces,
and exposes every substantial inner-block rear pocket to R11.  Residual ground
is never called a courtyard merely because it is empty: ordinary pockets are
``alley_quarter`` domains that must receive connected circulation and housing.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, polylabel, unary_union

from procgen.cityplan import rot2d_ccw
from procgen.frontage_fit import _stamp_doors
from .constants import ROUTE_CONNECTOR_GU, ROUTE_REACH_GU
from .stamp_index import DEFAULT_LIBRARIES, load_stamp_libraries
from .validate import TownLayoutError

BUILDING_CLEARANCE_GU = 16.0
WALL_CLEARANCE_GU = 80.0
ALLEY_CLEAR_WIDTH_GU = 224.0
PLAZA_MOUTH_WIDTH_GU = 384.0
REAR_APRON_WIDTH_GU = 128.0
REAR_APRON_DEPTH_GU = 96.0
MAX_NEW_INNER_PLACEMENTS = 160
MIN_BACKFILL_AREA_GU2 = 2_000_000.0

_FALKREATH_SECTORS = (
    ("east_market", "plaza", "block_024_inner_00", 640.0, 1),
    ("north_court", "front_courtyard", "block_002_inner_00", 384.0, 1),
)

_SPECIAL_BY_BLOCK = {row[2]: row for row in _FALKREATH_SECTORS}


def _parts(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]


def _ring(poly: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in poly.exterior.coords]


def _holes(poly: Polygon) -> list[list[list[float]]]:
    """Serialize interior clearance holes without losing free-ground topology."""
    return [
        [[float(x), float(y)] for x, y in ring.coords]
        for ring in poly.interiors
    ]


def _road_reserve(source: dict[str, Any]):
    return unary_union([
        LineString(row["polyline"]).buffer(float(row["clear_width_gu"]) / 2.0,
                                            cap_style=2, join_style=2)
        for row in source["roads"]
    ])


def _door_rows(source: dict[str, Any]) -> list[dict]:
    libraries = load_stamp_libraries(DEFAULT_LIBRARIES)
    libraries.update(source.get("generated_stamps") or {})
    rows = []
    for placement in source["placements"]:
        stamp = libraries.get(placement["stamp_id"])
        if stamp is None:
            raise TownLayoutError(f"missing stamp library record {placement['stamp_id']}")
        transformed = []
        for door in _stamp_doors(stamp):
            dx, dy = rot2d_ccw(door.offset[0], door.offset[1], placement["yaw_deg"])
            transformed.append({
                "door_id": f"{placement['parcel_id']}:{door.door_id}",
                "placement_id": placement["parcel_id"],
                "source_door_id": door.door_id,
                "position": [placement["anchor"][0] + dx, placement["anchor"][1] + dy],
                "outward_heading_deg": (door.heading_deg + placement["yaw_deg"]) % 360.0,
                "role": "secondary",
                "source": "stamp_library",
            })
        if transformed:
            primary = Point(placement["door_world"])
            chosen = min(transformed, key=lambda row: primary.distance(Point(row["position"])))
            chosen["position"] = list(placement["door_world"])
            chosen["role"] = "primary"
        rows.extend(transformed)
    return sorted(rows, key=lambda row: row["door_id"])


def _mouths_for_block(source: dict[str, Any], block_id: str, free: Polygon,
                       occupied) -> list[dict]:
    """Return clear road-to-interior gaps, at most one ranked mouth per road."""
    arc_roads = sorted({row["road_id"] for row in source["frontage_inventory"]
                        if row["block_id"] == block_id})
    road_by_id = {row["road_id"]: row for row in source["roads"]}
    result = []
    center = polylabel(free, tolerance=32.0)
    safe = free.buffer(-(ALLEY_CLEAR_WIDTH_GU / 2.0 + 4.0))
    safe_parts = ([safe] if safe.geom_type == "Polygon" else
                  [g for g in getattr(safe, "geoms", []) if g.geom_type == "Polygon"])
    for road_id in arc_roads:
        road = road_by_id[road_id]
        line = LineString(road["polyline"])
        candidates = []
        station = 128.0
        while station < line.length - 128.0:
            q = line.interpolate(station)
            inside = nearest_points(q, free)[1]
            connector = LineString([q, inside])
            def connector_ok(point: Point) -> bool:
                for part in safe_parts:
                    if part.distance(point) > ROUTE_REACH_GU:
                        continue
                    anchor = nearest_points(point, part)[1]
                    connector = LineString([point, anchor])
                    if (connector.length <= ROUTE_CONNECTOR_GU and
                            free.buffer(2.0).covers(connector) and
                            connector.buffer(ALLEY_CLEAR_WIDTH_GU / 2.0,
                                             cap_style=2).intersection(occupied).area <= 1.0):
                        return True
                return False

            inside_pt_ok = connector_ok(inside)
            center_pt_ok = connector_ok(center)
            if (q.distance(inside) <= 1400.0 and
                    connector.buffer(ALLEY_CLEAR_WIDTH_GU / 2.0,
                                     cap_style=2).intersection(occupied).area <= 1.0 and
                    inside_pt_ok and center_pt_ok):
                candidates.append((inside.distance(center) + q.distance(inside),
                                   station, q, inside))
            station += 128.0
        if candidates:
            _score, station, q, inside = min(candidates,
                                             key=lambda row: (row[0], row[1]))
            result.append({
                "mouth_id": f"mouth:{block_id}:{road_id}",
                "road_id": road_id,
                "road_point": [float(q.x), float(q.y)],
                "inside_point": [float(inside.x), float(inside.y)],
                "station_gu": float(station),
                "width_gu": ALLEY_CLEAR_WIDTH_GU,
            })
    return result


def derive_spatial_roles(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("stage_id") != "r5_wall_front_rows":
        raise TownLayoutError("spatial roles requires r5_wall_front_rows")
    blocks = {row["block_id"]: row for row in source["buildable_blocks"]}
    occupied = unary_union([Polygon(row["hull"]) for row in source["placements"]])
    road_reserve = _road_reserve(source)
    wall = Polygon(source["wall"]["planning_polygon"])
    wall_band = wall.boundary.buffer(WALL_CLEARANCE_GU)
    water = unary_union([Polygon(ring) for ring in source.get("water_polygons") or []])
    by_block = defaultdict(list)
    for placement in source["placements"]:
        by_block[placement["block_id"]].append(Polygon(placement["hull"]))

    sectors = []
    free_rows = []
    for block_id in sorted(blocks):
        if blocks[block_id].get("development_zone") != "inner":
            continue
        special = _SPECIAL_BY_BLOCK.get(block_id)
        if special:
            sector_id, role, _block_id, kernel_radius, mouth_count = special
        else:
            sector_id = f"quarter_{block_id.removeprefix('block_').removesuffix('_inner_00')}"
            role, kernel_radius, mouth_count = "alley_quarter", 0.0, 2
        block = Polygon(blocks[block_id]["polygon"])
        local_occupied = unary_union(by_block[block_id]) if by_block[block_id] else Polygon()
        unavailable = local_occupied.buffer(BUILDING_CLEARANCE_GU).union(
            road_reserve).union(wall_band).union(water)
        parts = [part for part in _parts(block.difference(unavailable))
                 if part.area >= 512.0 ** 2]
        if not parts:
            if special:
                raise TownLayoutError(f"sector {sector_id} has no usable free ground")
            continue
        free = max(parts, key=lambda part: part.area)
        if not special and free.area < MIN_BACKFILL_AREA_GU2:
            continue
        center = polylabel(free, tolerance=32.0)
        clear_radius = center.distance(free.boundary)
        if kernel_radius:
            radius = min(kernel_radius, clear_radius * 0.62)
            if radius < 320.0:
                raise TownLayoutError(f"sector {sector_id} cannot fit compact kernel")
            kernel = center.buffer(radius, resolution=12)
        else:
            radius = 0.0
            kernel = Polygon()
        mouths = _mouths_for_block(source, block_id, free, occupied)
        if special and len(mouths) < mouth_count:
            raise TownLayoutError(f"sector {sector_id} found {len(mouths)}/{mouth_count} mouths")
        if not mouths:
            continue
        if special or len(mouths) == 1:
            mouths.sort(key=lambda row: (Point(row["inside_point"]).distance(center),
                                         row["road_id"]))
            chosen = mouths[:mouth_count]
        else:
            # A rear quarter should normally be permeable. Choose the two
            # clearest mouths that span the widest part of the pocket rather
            # than creating another road-adjacent dead end.
            pairs = [(Point(a["inside_point"]).distance(Point(b["inside_point"])),
                      a, b)
                     for index, a in enumerate(mouths)
                     for b in mouths[index + 1:]]
            _distance, a, b = max(pairs, key=lambda row: (row[0],
                                                          row[1]["road_id"],
                                                          row[2]["road_id"]))
            chosen = [a, b]
        sector = {
            "sector_id": sector_id,
            "candidate_id": sector_id,
            "role": role,
            "status": "selected",
            "block_id": block_id,
            "free_polygon": _ring(free),
            "free_holes": _holes(free),
            "polygon": _ring(kernel) if not kernel.is_empty else [],
            "center": [float(center.x), float(center.y)],
            "kernel_radius_gu": float(radius),
            "mouths": chosen,
            "free_area_gu2": float(free.area),
            "existing_building_ids": [row["parcel_id"] for row in source["placements"]
                                      if row["block_id"] == block_id],
        }
        sectors.append(sector)
        free_rows.append({"free_id": f"free:{sector_id}", "block_id": block_id,
                          "polygon": _ring(free), "holes": _holes(free),
                          "area_gu2": float(free.area)})

    doors = _door_rows(source)
    out = dict(source)
    out.update({
        "stage_id": "r10_spatial_roles",
        "doors": doors,
        "free_ground": free_rows,
        "gap_records": [mouth for sector in sectors for mouth in sector["mouths"]],
        "spatial_role_candidates": sectors,
        "spatial_roles": sectors,
        "spatial_role_metrics": {
            "sector_count": len(sectors),
            "alley_quarter_count": sum(row["role"] == "alley_quarter"
                                        for row in sectors),
            "plaza_count": sum(row["role"] == "plaza" for row in sectors),
            "front_court_count": sum(row["role"] == "front_courtyard" for row in sectors),
            "door_count": len(doors),
            "primary_door_count": sum(row["role"] == "primary" for row in doors),
            "secondary_door_count": sum(row["role"] == "secondary" for row in doors),
        },
    })
    return out
