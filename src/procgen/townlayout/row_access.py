"""Derive courtyard mouths and access reservations after R5 front rows.

R5 already reaches the town brief's capacity band, so this stage does not add
rear buildings.  It proves which residual block interiors have a clear opening
to an accepted road and reserves those openings before later lane/alley work.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .geometry import normalize_ring, polygon_from_ring
from .validate import TownLayoutError

PEDESTRIAN_HALF_GU = 64.0
MIN_MOUTH_WIDTH_GU = 256.0


def _parts(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [g for g in getattr(geometry, "geoms", [])
            if g.geom_type == "Polygon" and g.area > 1.0]


def _normal(line: LineString, station: float, side: str) -> tuple[float, float]:
    a = line.interpolate(max(0.0, station - 8.0))
    b = line.interpolate(min(line.length, station + 8.0))
    tx, ty = b.x - a.x, b.y - a.y
    length = math.hypot(tx, ty) or 1.0
    left = (-ty / length, tx / length)
    return left if side == "left" else (-left[0], -left[1])


def reserve_row_access(source: dict[str, Any], *, _removed: list[dict] | None = None,
                       _removals_by_block: dict[str, int] | None = None,
                       _forced_court_blocks: set[str] | None = None) -> dict[str, Any]:
    if source.get("stage_id") != "r5_wall_front_rows":
        raise TownLayoutError("row access requires r5_wall_front_rows")
    _removed = list(_removed or [])
    _removals_by_block = dict(_removals_by_block or {})
    _forced_court_blocks = set(_forced_court_blocks or ())
    blocks = {row["block_id"]: row for row in source["buildable_blocks"]}
    roads = {row["road_id"]: row for row in source["roads"]}
    arcs_by_block = defaultdict(list)
    for arc in source.get("frontage_inventory") or []:
        arcs_by_block[arc["block_id"]].append(arc)
    hulls_by_block = defaultdict(list)
    frontages_by_block = defaultdict(set)
    hull_areas = []
    for placement in source.get("placements") or []:
        hull = Polygon(placement["hull"])
        hulls_by_block[placement["block_id"]].append(hull)
        frontages_by_block[placement["block_id"]].add(
            (placement.get("frontage_road_id"), placement.get("side")))
        hull_areas.append(float(hull.area))
    if not hull_areas:
        raise TownLayoutError("row access received no front placements")
    p50_hull_area = statistics.median(hull_areas)

    courtyards = []
    verges = []
    mouths = []
    paths = []
    inaccessible = []
    residual_area = 0.0
    for block_id, block in sorted(blocks.items()):
        block_poly = polygon_from_ring(block["polygon"])
        occupied = unary_union([h.buffer(PEDESTRIAN_HALF_GU)
                                for h in hulls_by_block.get(block_id, [])]) \
            if hulls_by_block.get(block_id) else Polygon()
        residual = block_poly.difference(occupied)
        parts = sorted(_parts(residual), key=lambda g: (-g.area, g.centroid.x, g.centroid.y))
        for part_no, part in enumerate(parts):
            residual_area += float(part.area)
            placement_count = len(hulls_by_block.get(block_id, []))
            is_courtyard_candidate = (
                block.get("development_zone") == "inner"
                and ((placement_count >= 3
                      and len(frontages_by_block.get(block_id, ())) >= 2)
                     or block_id in _forced_court_blocks)
                and 2.0 * p50_hull_area <= part.area <= 10.0 * p50_hull_area
            )
            if not is_courtyard_candidate:
                kind = ("open_landscape" if block.get("development_zone") == "outer"
                        else "development_reserve" if part.area >= 2.0 * p50_hull_area
                        else "verge")
                verges.append({
                    "space_id": f"verge_{block_id}_{part_no:02d}",
                    "block_id": block_id, "kind": kind,
                    "polygon": normalize_ring([[float(x), float(y)]
                                               for x, y in part.exterior.coords])["ring"],
                })
                continue
            safe = part.buffer(-PEDESTRIAN_HALF_GU)
            if safe.is_empty:
                inaccessible.append((block_id, part_no, "no_128_gu_channel"))
                continue
            best = None
            for arc in arcs_by_block.get(block_id, []):
                road = roads[arc["road_id"]]
                line = LineString(road["polyline"])
                lo = float(arc["arc_start_gu"]) + MIN_MOUTH_WIDTH_GU / 2.0
                hi = float(arc["arc_end_gu"]) - MIN_MOUTH_WIDTH_GU / 2.0
                station = lo
                while station <= hi + 1e-6:
                    q = line.interpolate(station)
                    nx, ny = _normal(line, station, arc["side"])
                    curb = float(road["clear_width_gu"]) / 2.0
                    for depth in (128.0, 192.0, 256.0, 384.0, 512.0):
                        inside = Point(q.x + nx * (curb + depth),
                                       q.y + ny * (curb + depth))
                        if safe.covers(inside):
                            score = inside.distance(part.centroid) + depth
                            row = (score, arc, road, station, q, inside)
                            if best is None or row[0] < best[0]:
                                best = row
                            break
                    station += 64.0
            if best is None:
                if not arcs_by_block.get(block_id):
                    verges.append({
                        "space_id": f"landscape_{block_id}_{part_no:02d}",
                        "block_id": block_id, "kind": "open_landscape",
                        "reason": "no_road_frontage",
                        "polygon": normalize_ring([[float(x), float(y)]
                                                   for x, y in part.exterior.coords])["ring"],
                    })
                    continue
                inaccessible.append((block_id, part_no, "no_frontage_mouth"))
                continue
            _score, arc, road, station, q, inside = best
            mouth_id = f"access_mouth_{len(mouths):03d}"
            court_id = f"courtyard_{len(courtyards):03d}"
            path_id = f"court_path_{len(paths):03d}"
            mouths.append({
                "mouth_id": mouth_id, "block_id": block_id,
                "road_id": road["road_id"], "side": arc["side"],
                "station_gu": station, "width_gu": MIN_MOUTH_WIDTH_GU,
                "position": [float(inside.x), float(inside.y)],
            })
            paths.append({
                "path_id": path_id, "block_id": block_id,
                "courtyard_id": court_id, "mouth_id": mouth_id,
                "width_gu": PEDESTRIAN_HALF_GU * 2.0,
                "geometry": [[float(q.x), float(q.y)],
                             [float(inside.x), float(inside.y)]],
            })
            courtyards.append({
                "courtyard_id": court_id, "block_id": block_id,
                "mouth_id": mouth_id, "access_path_id": path_id,
                "polygon": normalize_ring([[float(x), float(y)]
                                           for x, y in part.exterior.coords])["ring"],
                "area_gu2": float(part.area),
            })

    if inaccessible:
        block_id = inaccessible[0][0]
        count = _removals_by_block.get(block_id, 0)
        candidates = [p for p in source.get("placements") or []
                      if p["block_id"] == block_id]
        if count < 2 and candidates:
            blocker = min(candidates, key=lambda p: (
                Polygon(p["hull"]).area, p["parcel_id"]))
            revised = dict(source)
            revised["placements"] = [p for p in source["placements"]
                                      if p["parcel_id"] != blocker["parcel_id"]]
            revised["placement_hulls"] = dict(source.get("placement_hulls") or {})
            revised["placement_hulls"].pop(blocker["parcel_id"], None)
            removed = dict(blocker)
            removed["removal_reason"] = "courtyard_mouth_blocker"
            next_counts = dict(_removals_by_block)
            next_counts[block_id] = count + 1
            return reserve_row_access(
                revised, _removed=_removed + [removed],
                _removals_by_block=next_counts,
                _forced_court_blocks=_forced_court_blocks | {block_id})
        detail = ", ".join(f"{bid}:{part}:{reason}"
                           for bid, part, reason in inaccessible[:8])
        raise TownLayoutError(f"R6 inaccessible courtyard {detail}")
    out = dict(source)
    out.update({
        "stage_id": "r6_rows_access",
        "courtyards": courtyards,
        "verges": verges,
        "access_mouths": mouths,
        "reserved_access_paths": paths,
        "removed_frontage_blockers": _removed,
        "row_access_metrics": {
            "p50_hull_area_gu2": p50_hull_area,
            "courtyard_count": len(courtyards),
            "verge_count": len(verges),
            "mouth_count": len(mouths),
            "access_path_count": len(paths),
            "inaccessible_count": 0,
            "rear_placement_count": 0,
            "removed_blocker_count": len(_removed),
            "final_population": len(source.get("placements") or []),
            "rear_placement_reason": "front rows already occupy the brief capacity band",
            "residual_area_gu2": residual_area,
        },
    })
    return out
