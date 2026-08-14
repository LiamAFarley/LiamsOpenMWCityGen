"""Deterministic T1.3 VTEX painting and local-LTEX closure.

Pipeline position
------------------
This module consumes the effective normalized VTEX grids from the accepted
Falkreath remap, the validated D-PLAN roads/district texture zones, and T1.2
placed lot hulls.  It paints raw VTEX values in a fixed priority order and
returns per-cell 16x16 grids plus the exact masterless-plugin LTEX table for
T1.4.  It never resolves a raw value through a global load-order table.

Priority and semantics
----------------------
1. road corridors -> explicit raw 78;
2. placed lot footprints -> explicit settlement dirt raw 241;
3. declared zone mixes -> deterministic low-frequency plan-hash hashing;
4. margin transitions -> grass-dirt where that class is declared.

Raw 1/lakebed tiles are preserved and are never assigned to roads.  Existing
raw 78 source roads are also protected from district repainting; planned road
corridors are unioned with that source road set.  Raw 92 pine and raw 33 base
grass remain source/effective classes outside paint support.  Every positive raw value in the emitted grids must have one exact
local LTEX record (index = raw - 1) with the palette/remap id/path contract.

Invariants
----------
* Grids outside the union of declared zone, road, and lot support are byte
  identical to the effective source grids.
* No checkerboard/band fallback or RNG global state is used: all choices are
  a pure function of the plan hash, zone id, and tile coordinates.
* Paint counts and realized fractions are recorded against the target weights;
  any assignment/id/index disagreement fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from . import espland, regionpalette
from .cityscape_field import TargetBlock


TILE_SIDE = 16
TILE_SIZE_GU = 512.0
CELL_SIZE_GU = 8192.0


class CityscapeVTEXError(ValueError):
    """Hard texture assignment, support, water, or LTEX closure failure."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_unit(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    points = [(float(p[0]), float(p[1])) for p in polygon]
    if points[0] != points[-1]:
        points.append(points[0])
    inside = False
    for index in range(len(points) - 1):
        a, b = points[index], points[index + 1]
        if (a[1] > point[1]) != (b[1] > point[1]):
            cross = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if cross > point[0]:
                inside = not inside
    return inside


def _point_segment_distance(point: tuple[float, float], a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(point[0] - ax, point[1] - ay)
    t = max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(point[0] - (ax + t * dx), point[1] - (ay + t * dy))


def _polyline_distance(point: tuple[float, float], polyline: Sequence[Sequence[float]]) -> float:
    if len(polyline) < 2:
        raise CityscapeVTEXError("road polyline requires at least two points")
    return min(_point_segment_distance(point, polyline[i], polyline[i + 1]) for i in range(len(polyline) - 1))


def _tile_plan_point(block: TargetBlock, cell: tuple[int, int], tile_x: int, tile_y: int) -> tuple[float, float]:
    min_x = min(grid[0] for grid in block.cells)
    min_y = min(grid[1] for grid in block.cells)
    return (
        (cell[0] - min_x) * CELL_SIZE_GU + (tile_x + 0.5) * TILE_SIZE_GU,
        (cell[1] - min_y) * CELL_SIZE_GU + (tile_y + 0.5) * TILE_SIZE_GU,
    )


def _grid_sha256(grid: np.ndarray) -> str:
    return _sha256_bytes(np.asarray(grid, dtype="<u2").tobytes(order="C"))


@dataclass(frozen=True)
class SurfaceAssignment:
    surface: str
    raw_vtex: int
    ltex_index: int
    ltex_id: str
    file_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "raw_vtex": self.raw_vtex,
            "ltex_index": self.ltex_index,
            "ltex_id": self.ltex_id,
            "file_name": self.file_name,
        }


def load_surface_assignments(palette: Mapping[str, Any]) -> dict[str, SurfaceAssignment]:
    """Load explicit dispatch-5 raw/index/id assignments; never infer from ordinals."""

    surfaces = palette.get("semantic_surfaces", {}).get("surfaces")
    if not isinstance(surfaces, list):
        raise CityscapeVTEXError("region palette has no semantic surface table")
    checks = regionpalette.validate_authoring_assignments(surfaces)
    failed = [check for check in checks if not check.get("passed")]
    if failed:
        raise CityscapeVTEXError(f"palette authoring assignment gate failed: {failed}")
    result: dict[str, SurfaceAssignment] = {}
    for surface in surfaces:
        name = surface.get("surface")
        assignment = surface.get("planned_assignment")
        if not isinstance(name, str) or not isinstance(assignment, Mapping):
            raise CityscapeVTEXError("semantic surface has no explicit planned assignment")
        raw = assignment.get("planned_raw_vtex")
        index = assignment.get("planned_ltex_index")
        ltex_id = assignment.get("planned_ltex_id")
        if not isinstance(raw, int) or not isinstance(index, int) or raw <= 0 or index != raw - 1 or not isinstance(ltex_id, str):
            raise CityscapeVTEXError(f"surface {name!r} has invalid raw/index/id assignment")
        measured = surface.get("measured_identity", {})
        # The output assignment is authoritative for painting; measured
        # identity remains a separate provenance block in the palette.
        path = ""
        if isinstance(measured, Mapping):
            remap = measured.get("remap_identity")
            if isinstance(remap, Mapping) and remap.get("ltex_id") == ltex_id:
                path = str(remap.get("texture_path", ""))
        if not path:
            # Texture paths for the future city plugin are present in the
            # measured census table, but some palette versions only expose the
            # id in planned_assignment.  Use the effective remap table for
            # shared indices and require explicit path evidence for new ids.
            if index in regionpalette.REMAP_LTEX_TABLE and regionpalette.REMAP_LTEX_TABLE[index][0] == ltex_id:
                path = regionpalette.REMAP_LTEX_TABLE[index][1]
            else:
                for row in palette.get("settlement_clearance", {}).get("recomputed_census", {}).get("rows", []):
                    if isinstance(row, Mapping) and row.get("ltex_id") == ltex_id:
                        path = str(row.get("texture_path", ""))
                        break
        if not path:
            # Canonical dispatch-5 names are stable measured identities.  The
            # path fallback is intentionally explicit and limited to those
            # three local city records; arbitrary ids fail closed.
            known_paths = {
                "T_Sky_TerrDirtRE_01": "Tx_Skyrim_rocky_dirt_04.dds",
                "T_Sky_TerrGrassDirtRE_01": "Tx_Skyrim_grass_dirt_03.dds",
                "T_Nor_Set_TxCobbleStone_01": "Tx_Skyrim_Cobblest_AG_01.dds",
            }
            path = known_paths.get(ltex_id, "")
        if not path:
            raise CityscapeVTEXError(f"surface {name!r} lacks a local LTEX texture path for {ltex_id!r}")
        if name in result:
            raise CityscapeVTEXError(f"duplicate semantic surface {name!r}")
        result[name] = SurfaceAssignment(name, raw, index, ltex_id, path)
    required = palette.get("planned_output_plugin", {}).get("required_local_ltex")
    required_indices = sorted(int(row["ltex_index"]) for row in required) if isinstance(required, list) else []
    if required_indices != sorted(assignment.ltex_index for assignment in result.values()):
        raise CityscapeVTEXError("palette required_local_ltex does not equal explicit surface assignments")
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class PaintResult:
    """Deterministic painted grids and audit data."""

    grids: Mapping[tuple[int, int], np.ndarray]
    source_grids: Mapping[tuple[int, int], np.ndarray]
    support_masks: Mapping[tuple[int, int], np.ndarray]
    paint_ledger: Mapping[str, Any]
    local_ltex: tuple[Mapping[str, Any], ...]


def _zone_rows(plan: Mapping[str, Any], assignments: Mapping[str, SurfaceAssignment]) -> list[dict[str, Any]]:
    zones = {str(zone.get("zone_id")): zone for zone in plan.get("texture_zones", []) if isinstance(zone, Mapping)}
    result: list[dict[str, Any]] = []
    for district in plan.get("districts", []):
        if not isinstance(district, Mapping):
            continue
        zone_id = district.get("texture_zone")
        if not isinstance(zone_id, str) or zone_id not in zones:
            continue
        polygon = district.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise CityscapeVTEXError(f"district {district.get('district_id')} has no valid texture-zone polygon")
        classes = zones[zone_id].get("classes")
        if not isinstance(classes, list) or not classes:
            raise CityscapeVTEXError(f"texture zone {zone_id} has no classes")
        normalized: list[tuple[str, float]] = []
        for row in classes:
            if not isinstance(row, Mapping) or not isinstance(row.get("texture"), str):
                raise CityscapeVTEXError(f"texture zone {zone_id} has malformed class")
            texture = str(row["texture"])
            if texture not in assignments:
                raise CityscapeVTEXError(f"texture zone {zone_id} uses unknown surface {texture!r}")
            weight = float(row.get("weight"))
            if not math.isfinite(weight) or weight < 0.0:
                raise CityscapeVTEXError(f"texture zone {zone_id} has invalid weight {weight!r}")
            normalized.append((texture, weight))
        total = sum(weight for _, weight in normalized)
        if total <= 0.0 or abs(total - 1.0) > 1.0e-6:
            raise CityscapeVTEXError(f"texture zone {zone_id} weights sum to {total}, expected 1")
        result.append({
            "zone_id": zone_id,
            "polygon": polygon,
            "classes": normalized,
            "district_id": district.get("district_id"),
        })
    return sorted(result, key=lambda row: (str(row["zone_id"]), str(row.get("district_id"))))


def _road_tiles(
    block: TargetBlock,
    plan: Mapping[str, Any],
    placement: Mapping[str, Any] | None = None,
) -> set[tuple[tuple[int, int], int, int]]:
    result: set[tuple[tuple[int, int], int, int]] = set()
    roads = list(plan.get("roads", []))
    if isinstance(placement, Mapping):
        # D-PLACE may later expose rasterized solver step paths separately
        # from declarative plan roads.  Consume them with the same strict
        # width/surface gate when present; the accepted T1.2 fixture has none.
        roads.extend(placement.get("road_corridors", []))
    for road in roads:
        if not isinstance(road, Mapping):
            continue
        if str(road.get("surface")) != "road":
            continue
        polyline = road.get("polyline")
        width = float(road.get("width_gu", 0.0))
        if not isinstance(polyline, list) or len(polyline) < 2 or not math.isfinite(width) or width <= 0.0:
            raise CityscapeVTEXError(f"road {road.get('road_id')} has invalid painting geometry")
        for cell in block.cells:
            for tile_y in range(TILE_SIDE):
                for tile_x in range(TILE_SIDE):
                    point = _tile_plan_point(block, cell, tile_x, tile_y)
                    # A tile is part of a corridor when its center or a point
                    # within half a tile reaches the declared road width.  This
                    # rasterization is conservative and deterministic.
                    if _polyline_distance(point, polyline) <= width / 2.0 + TILE_SIZE_GU / 2.0:
                        result.add((cell, tile_x, tile_y))
    return result


def _lot_tiles(block: TargetBlock, placement: Mapping[str, Any] | None) -> set[tuple[tuple[int, int], int, int]]:
    result: set[tuple[tuple[int, int], int, int]] = set()
    if not isinstance(placement, Mapping):
        return result
    rows = list(placement.get("placements", [])) + list(placement.get("provisional_pad_lots", []))
    for lot in rows:
        if not isinstance(lot, Mapping):
            continue
        hull = lot.get("footprint_hull_xy_plan_gu")
        if not isinstance(hull, list) or len(hull) < 3:
            continue
        for cell in block.cells:
            for tile_y in range(TILE_SIDE):
                for tile_x in range(TILE_SIDE):
                    if _point_in_polygon(_tile_plan_point(block, cell, tile_x, tile_y), hull):
                        result.add((cell, tile_x, tile_y))
    return result


def _survey_water_tiles(block: TargetBlock, survey: Mapping[str, Any]) -> set[tuple[tuple[int, int], int, int]]:
    """Decode the accepted 112x112 mask and map it to per-cell VTEX tiles."""

    tile_grids = survey.get("tile_grids", {})
    encoded = tile_grids.get("water_mask") if isinstance(tile_grids, Mapping) else None
    result: set[tuple[tuple[int, int], int, int]] = set()
    if not isinstance(encoded, str):
        return result
    try:
        import base64
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise CityscapeVTEXError(f"cannot decode site water mask: {exc}") from exc
    if len(raw) != 112 * 112:
        raise CityscapeVTEXError(f"site water mask has {len(raw)} bytes, expected 12544")
    min_x = min(cell[0] for cell in block.cells)
    min_y = min(cell[1] for cell in block.cells)
    for cell in block.cells:
        for tile_y in range(TILE_SIDE):
            for tile_x in range(TILE_SIDE):
                global_tx = (cell[0] - min_x) * 16 + tile_x
                global_ty = (cell[1] - min_y) * 16 + tile_y
                if raw[global_ty * 112 + global_tx] == 1:
                    result.add((cell, tile_x, tile_y))
    return result


def _authorized_water_tiles(block: TargetBlock, plan: Mapping[str, Any]) -> set[tuple[tuple[int, int], int, int]]:
    """Rasterize explicit dock/basin feature positions as water exceptions."""

    result: set[tuple[tuple[int, int], int, int]] = set()
    min_x = min(cell[0] for cell in block.cells)
    min_y = min(cell[1] for cell in block.cells)
    cell_x_values = {cell[0] for cell in block.cells}
    cell_y_values = {cell[1] for cell in block.cells}
    for feature in plan.get("features", []):
        if not isinstance(feature, Mapping):
            continue
        kind = str(feature.get("kind", ""))
        feature_id = str(feature.get("feature_id", ""))
        if kind != "dock" and "dock" not in feature_id.lower() and "basin" not in feature_id.lower():
            continue
        position = feature.get("position")
        if not isinstance(position, list) or len(position) != 2:
            raise CityscapeVTEXError(f"water feature {feature_id} has no plan position")
        x, y = float(position[0]), float(position[1])
        cell_x = min_x + int(math.floor(x / CELL_SIZE_GU))
        cell_y = min_y + int(math.floor(y / CELL_SIZE_GU))
        tile_x = int(math.floor((x % CELL_SIZE_GU) / TILE_SIZE_GU))
        tile_y = int(math.floor((y % CELL_SIZE_GU) / TILE_SIZE_GU))
        if cell_x in cell_x_values and cell_y in cell_y_values and 0 <= tile_x < TILE_SIDE and 0 <= tile_y < TILE_SIDE:
            result.add(((cell_x, cell_y), tile_x, tile_y))
    return result


def _zone_for_tile(point: tuple[float, float], zones: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    matches = [zone for zone in zones if _point_in_polygon(point, zone["polygon"])]
    return sorted(matches, key=lambda row: str(row["zone_id"]))[0] if matches else None


def _allocate_zone_classes(
    candidates: Sequence[tuple[tuple[int, int], int, int]],
    zone: Mapping[str, Any],
    plan_hash: str,
    assignments: Mapping[str, SurfaceAssignment],
) -> dict[tuple[tuple[int, int], int, int], SurfaceAssignment]:
    classes = list(zone["classes"])
    if not candidates:
        return {}
    # Low-frequency hash: coarse 1024-GU cell and tile-local hash are blended,
    # preventing a repeating checkerboard while keeping every result pure.
    ranked = sorted(
        candidates,
        key=lambda key: (
            0.68 * _hash_unit(plan_hash, zone["zone_id"], key[1] // 2, key[2] // 2)
            + 0.32 * _hash_unit(plan_hash, zone["zone_id"], key[0], key[1], key[2]),
            key[0][1], key[0][0], key[2], key[1],
        ),
    )
    total = len(ranked)
    quotas: list[int] = []
    residuals: list[tuple[float, int]] = []
    used = 0
    for index, (_, weight) in enumerate(classes):
        exact = total * float(weight)
        count = int(math.floor(exact))
        quotas.append(count)
        used += count
        residuals.append((exact - count, index))
    for _, index in sorted(residuals, key=lambda row: (-row[0], row[1]))[: total - used]:
        quotas[index] += 1
    result: dict[tuple[tuple[int, int], int, int], SurfaceAssignment] = {}
    cursor = 0
    for (surface, _), quota in zip(classes, quotas):
        assignment = assignments[surface]
        for key in ranked[cursor : cursor + quota]:
            result[key] = assignment
        cursor += quota
    if cursor != total:
        raise CityscapeVTEXError("zone class quota accounting did not cover candidate tiles")
    return result


def _local_ltex_table(
    block: TargetBlock,
    assignments: Mapping[str, SurfaceAssignment],
    grids: Mapping[tuple[int, int], np.ndarray],
) -> tuple[Mapping[str, Any], ...]:
    raw_values = sorted({int(value) for grid in grids.values() for value in np.asarray(grid).reshape(-1) if int(value) > 0})
    table: list[dict[str, Any]] = []
    for raw in raw_values:
        index = espland.resolve_vtex_to_ltex_index(raw)
        assert index is not None
        assignment = next((item for item in assignments.values() if item.raw_vtex == raw), None)
        if assignment is None:
            source = block.effective_ltex.get(index)
            if source is None:
                raise CityscapeVTEXError(f"output raw {raw} has no effective local LTEX index {index}")
            assignment = SurfaceAssignment("source_raw_" + str(raw), raw, index, source.record_id, source.file_name)
        if assignment.ltex_index != index or assignment.raw_vtex != raw:
            raise CityscapeVTEXError(f"raw {raw} local LTEX assignment disagrees with index {index}")
        table.append({
            "index": index,
            "record_id": assignment.ltex_id,
            "file_name": assignment.file_name,
            "raw_vtex": raw,
            "source_or_planned": "planned_surface" if assignment.surface in assignments else "effective_source",
        })
    indices = [int(row["index"]) for row in table]
    if len(indices) != len(set(indices)):
        raise CityscapeVTEXError("output local LTEX table contains duplicate indices")
    return tuple(table)


def paint_vtex(
    *,
    block: TargetBlock,
    plan: Mapping[str, Any],
    plan_hash: str,
    palette: Mapping[str, Any],
    survey: Mapping[str, Any],
    placement: Mapping[str, Any] | None = None,
) -> PaintResult:
    """Paint effective VTEX grids using dispatch-5 assignments and priorities."""

    assignments = load_surface_assignments(palette)
    zones = _zone_rows(plan, assignments)
    roads = _road_tiles(block, plan, placement)
    lots = _lot_tiles(block, placement)
    water = _survey_water_tiles(block, survey)
    authorized_water = _authorized_water_tiles(block, plan)
    source_grids = {cell: block.effective_texture_grid(cell) for cell in block.cells}
    grids = {cell: np.array(grid, dtype=np.uint16, copy=True) for cell, grid in source_grids.items()}
    supports = {cell: np.zeros((TILE_SIDE, TILE_SIDE), dtype=bool) for cell in block.cells}
    priority_counts = {"road": 0, "lot": 0, "zone": 0, "water_preserved": 0, "source_unchanged": 0}
    zone_candidates: dict[str, list[tuple[tuple[int, int], int, int]]] = {str(zone["zone_id"]): [] for zone in zones}
    zone_for_key: dict[tuple[tuple[int, int], int, int], Mapping[str, Any]] = {}
    water_set = set(water)
    # Raw 1 is the dispatch-5 Sand/lakebed identity.  It remains protected
    # even if an old survey mask omits an individual raw-1 tile.
    raw1_set = {
        (cell, tile_x, tile_y)
        for cell, grid in source_grids.items()
        for tile_y in range(TILE_SIDE)
        for tile_x in range(TILE_SIDE)
        if int(grid[tile_y, tile_x]) == 1
    }
    water_set |= raw1_set
    water_set -= authorized_water
    # Existing effective raw-78 tiles are already surveyed road identity.  A
    # district zone must never reclassify them; planned/solver corridors are
    # unioned with that protected source set before any lower-priority pass.
    source_road_set = {
        (cell, tile_x, tile_y)
        for cell, grid in source_grids.items()
        for tile_y in range(TILE_SIDE)
        for tile_x in range(TILE_SIDE)
        if int(grid[tile_y, tile_x]) == 78
    }
    road_set = set(roads) | source_road_set
    lot_set = set(lots)
    # Discover support first so source-outside-support is a meaningful hard
    # gate even when a higher-priority road or lot later overwrites a zone.
    for cell in block.cells:
        for tile_y in range(TILE_SIDE):
            for tile_x in range(TILE_SIDE):
                key = (cell, tile_x, tile_y)
                point = _tile_plan_point(block, cell, tile_x, tile_y)
                zone = _zone_for_tile(point, zones)
                if zone is not None:
                    zone_for_key[key] = zone
                    zone_candidates[str(zone["zone_id"])].append(key)
                if key in road_set or key in lot_set or zone is not None:
                    supports[cell][tile_y, tile_x] = True
    # Roads have absolute priority, except a water/lakebed tile is preserved.
    road_assignment = assignments.get("road")
    if road_assignment is None:
        raise CityscapeVTEXError("closed palette has no road assignment")
    dirt_assignment = assignments.get("settlement_dirt")
    if dirt_assignment is None:
        raise CityscapeVTEXError("closed palette has no settlement_dirt assignment")
    for key in sorted(road_set, key=lambda item: (item[0][1], item[0][0], item[2], item[1])):
        cell, tile_x, tile_y = key
        source_raw = int(source_grids[cell][tile_y, tile_x])
        if key in water_set or source_raw == assignments.get("water_edge_sand", SurfaceAssignment("", 1, 0, "", "")).raw_vtex:
            priority_counts["water_preserved"] += 1
            continue
        grids[cell][tile_y, tile_x] = road_assignment.raw_vtex
        priority_counts["road"] += 1
    # Exact lot footprint dirt is next, but never overwrites an already-painted
    # road or preserved water tile.
    for key in sorted(lot_set, key=lambda item: (item[0][1], item[0][0], item[2], item[1])):
        cell, tile_x, tile_y = key
        if key in water_set or key in road_set and int(grids[cell][tile_y, tile_x]) == road_assignment.raw_vtex:
            continue
        grids[cell][tile_y, tile_x] = dirt_assignment.raw_vtex
        priority_counts["lot"] += 1
    # Zone classes cover remaining declared support.  A class allocation is
    # measured independently per zone, so target weights are not diluted by
    # roads/lots/water that have higher priority.
    zone_counts: dict[str, dict[str, int]] = {}
    for zone in zones:
        zone_id = str(zone["zone_id"])
        candidates = [
            key for key in zone_candidates[zone_id]
            if key not in road_set and key not in lot_set and key not in water_set and (int(source_grids[key[0]][key[2], key[1]]) not in {1, 92} or key in authorized_water)
        ]
        allocated = _allocate_zone_classes(candidates, zone, plan_hash, assignments)
        target = {surface: float(weight) for surface, weight in zone["classes"]}
        realized = {surface: 0 for surface in target}
        for key, assignment in sorted(allocated.items(), key=lambda item: (item[0][0][1], item[0][0][0], item[0][2], item[0][1])):
            cell, tile_x, tile_y = key
            grids[cell][tile_y, tile_x] = assignment.raw_vtex
            realized[assignment.surface] = realized.get(assignment.surface, 0) + 1
            priority_counts["zone"] += 1
        zone_counts[zone_id] = {
            "candidate_tile_count": len(candidates),
            "target_weights": target,
            "realized_counts": realized,
            "realized_fractions": {surface: (count / len(candidates) if candidates else 0.0) for surface, count in realized.items()},
        }
    # Margin blending is a declared-support diagnostic.  It only changes zone
    # tiles adjacent to an un-zoned tile and never crosses priority classes.
    margin_assignment = assignments.get("settlement_grass_dirt")
    margin_count = 0
    if margin_assignment is not None:
        for key, zone in sorted(zone_for_key.items(), key=lambda item: (item[0][0][1], item[0][0][0], item[0][2], item[0][1])):
            cell, tile_x, tile_y = key
            if key in road_set or key in lot_set or key in water_set:
                continue
            neighbors = []
            boundary_margin = False
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = tile_x + dx, tile_y + dy
                if 0 <= nx < 16 and 0 <= ny < 16:
                    neighbors.append((cell, nx, ny))
                else:
                    # A target-block edge is itself a declared district edge;
                    # its outside neighbour is intentionally absent from the
                    # paint support and therefore receives the same measured
                    # transition treatment as an interior un-zoned neighbour.
                    neighbor_cell = (cell[0] + (1 if nx >= 16 else -1 if nx < 0 else 0), cell[1] + (1 if ny >= 16 else -1 if ny < 0 else 0))
                    local = (nx % 16, ny % 16)
                    if neighbor_cell in block.cells:
                        neighbors.append((neighbor_cell, local[0], local[1]))
                    else:
                        boundary_margin = True
            if boundary_margin or any((neighbor[0], neighbor[1], neighbor[2]) not in zone_for_key for neighbor in neighbors):
                grids[cell][tile_y, tile_x] = margin_assignment.raw_vtex
                margin_count += 1
    # Hard source-support and water/road assertions.
    changed_outside = 0
    water_changed = 0
    road_raw1 = 0
    for cell in block.cells:
        source = source_grids[cell]
        output = grids[cell]
        mask = supports[cell]
        changed_outside += int(np.count_nonzero((output != source) & ~mask))
        water_mask = np.asarray(
            [[(cell, x, y) in water_set for x in range(TILE_SIDE)] for y in range(TILE_SIDE)],
            dtype=bool,
        )
        water_changed += int(np.count_nonzero(output[water_mask] != source[water_mask]))
        for tile_y in range(16):
            for tile_x in range(16):
                if (cell, tile_x, tile_y) in road_set and int(output[tile_y, tile_x]) == 1:
                    road_raw1 += 1
                if int(output[tile_y, tile_x]) == int(source[tile_y, tile_x]) and not mask[tile_y, tile_x]:
                    priority_counts["source_unchanged"] += 1
    if changed_outside:
        raise CityscapeVTEXError(f"VTEX changed {changed_outside} tiles outside declared support")
    if water_changed:
        raise CityscapeVTEXError(f"VTEX changed {water_changed} protected water tiles")
    if road_raw1:
        raise CityscapeVTEXError(f"raw 1 Sand was painted as road on {road_raw1} tiles")
    # Report both the pre-margin quota realization and the final class counts
    # actually handed to T1.4; the latter is the acceptance-facing fraction
    # when a district edge transition overrides a quota tile.
    raw_to_surface = {assignment.raw_vtex: name for name, assignment in assignments.items()}
    for zone in zones:
        zone_id = str(zone["zone_id"])
        candidates = [
            key for key in zone_candidates[zone_id]
            if key not in road_set and key not in lot_set and key not in water_set and (int(source_grids[key[0]][key[2], key[1]]) not in {1, 92} or key in authorized_water)
        ]
        final_counts: dict[str, int] = {surface: 0 for surface, _ in zone["classes"]}
        for cell, tile_x, tile_y in candidates:
            surface = raw_to_surface.get(int(grids[cell][tile_y, tile_x]))
            if surface in final_counts:
                final_counts[surface] += 1
        zone_counts[zone_id]["final_counts_after_margin"] = final_counts
        zone_counts[zone_id]["final_fractions_after_margin"] = {
            surface: (count / len(candidates) if candidates else 0.0)
            for surface, count in final_counts.items()
        }
    local_ltex = _local_ltex_table(block, assignments, grids)
    raw_counts: dict[str, int] = {}
    for grid in grids.values():
        for value in grid.reshape(-1):
            raw_counts[str(int(value))] = raw_counts.get(str(int(value)), 0) + 1
    paint_ledger = {
        "plan_hash": plan_hash,
        "priority": ["road_raw_78", "lot_settlement_dirt_raw_241", "zone_weight_mix", "margin_grass_dirt"],
        "declared_zone_count": len(zones),
        "road_support_tile_count": len(road_set),
        "source_road_tile_count": len(source_road_set),
        "lot_support_tile_count": len(lot_set),
        "water_tile_count": len(water_set),
        "authorized_water_exception_tile_count": len(authorized_water),
        "support_tile_count": int(sum(np.count_nonzero(mask) for mask in supports.values())),
        "priority_counts": {**priority_counts, "margin_blend": margin_count},
        "zone_realization": zone_counts,
        "raw_counts": dict(sorted(raw_counts.items(), key=lambda item: int(item[0]))),
        "source_outside_support_unchanged": changed_outside == 0,
        "water_preserved": water_changed == 0,
        "raw_78_road_gate": road_raw1 == 0,
        "surface_assignments": {name: assignment.to_dict() for name, assignment in assignments.items()},
        "local_ltex_indices": [int(row["index"]) for row in local_ltex],
        "local_ltex_complete": True,
    }
    for cell in block.cells:
        paint_ledger.setdefault("cells", {})[f"{cell[0]},{cell[1]}"] = {
            "source_grid_sha256": _grid_sha256(source_grids[cell]),
            "painted_grid_sha256": _grid_sha256(grids[cell]),
            "changed_tile_count": int(np.count_nonzero(grids[cell] != source_grids[cell])),
            "support_tile_count": int(np.count_nonzero(supports[cell])),
        }
    return PaintResult(
        grids=grids,
        source_grids=source_grids,
        support_masks=supports,
        paint_ledger=paint_ledger,
        local_ltex=local_ltex,
    )


__all__ = [
    "CityscapeVTEXError",
    "PaintResult",
    "SurfaceAssignment",
    "load_surface_assignments",
    "paint_vtex",
]
