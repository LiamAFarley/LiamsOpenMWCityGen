"""Decode and measure road-painted LAND/VTEX tiles for a planner site.

Pipeline position::

    read-only TES3 LAND source
        -> normalized raw-VTEX road mask and tile evidence
        -> Cityforge survey JSON / ``land_roads.json``
        -> exact mask-derived planner annotation

The functions in this module are deliberately independent of the cleaned
world-map road graph.  They consume :class:`procgen.espland.LandRecord`
objects, whose ``texture_indices`` are already in the OpenMW-normalized
16x16 row-major order, and never synthesize a polyline or a junction.  The
target mask is the source occupancy at 512 GU per tile.  A one-cell source
ring is required when boundary evidence is built so an edge tile is reported
as a continuation only when its orthogonally adjacent outside LAND tile has
the same raw VTEX value.

Inputs are a complete target-plus-perimeter LAND selection, inclusive cell
bounds, and one raw VTEX class.  Outputs are a uint8 target mask and a JSON
serializable evidence document containing sorted tile rows, deterministic
8-neighbour components, a 4-neighbour diagnostic, and grouped boundary
continuation spans.  No files are authored here; callers own source hashes
and output paths.  The implementation uses no random state, so identical
LAND payloads produce identical evidence.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .espland import LAND_TEXTURE_SIDE, LandRecord


CELL_SIZE_GU = 8192
TILE_SIZE_GU = 512
SIDE_ORDER = ("west", "east", "south", "north")

Grid = tuple[int, int]
Bounds = tuple[int, int, int, int]
SiteTile = tuple[int, int]


def _validated_bounds(bounds: Sequence[int]) -> Bounds:
    if len(bounds) != 4:
        raise ValueError("bounds must be [min_x, max_x, min_y, max_y]")
    minimum_x, maximum_x, minimum_y, maximum_y = (int(value) for value in bounds)
    if minimum_x > maximum_x or minimum_y > maximum_y:
        raise ValueError("bounds minimum must not exceed maximum")
    return minimum_x, maximum_x, minimum_y, maximum_y


def target_grids(bounds: Sequence[int]) -> tuple[Grid, ...]:
    """Return inclusive target grids in deterministic row-major order."""

    minimum_x, maximum_x, minimum_y, maximum_y = _validated_bounds(bounds)
    return tuple(
        (x, y)
        for y in range(minimum_y, maximum_y + 1)
        for x in range(minimum_x, maximum_x + 1)
    )


def perimeter_grids(bounds: Sequence[int]) -> tuple[Grid, ...]:
    """Return the complete one-cell bounding-box ring around ``bounds``.

    The corners are included intentionally.  They are part of the requested
    x/y perimeter selection even though only orthogonally adjacent tiles can
    prove a target-edge continuation.
    """

    minimum_x, maximum_x, minimum_y, maximum_y = _validated_bounds(bounds)
    return tuple(
        (x, y)
        for y in range(minimum_y - 1, maximum_y + 2)
        for x in range(minimum_x - 1, maximum_x + 2)
        if not (minimum_x <= x <= maximum_x and minimum_y <= y <= maximum_y)
    )


def source_selection_grids(bounds: Sequence[int]) -> tuple[Grid, ...]:
    """Return target plus the full one-cell perimeter selection."""

    return target_grids(bounds) + perimeter_grids(bounds)


def decode_land_road_mask(
    records: Mapping[Grid, LandRecord],
    bounds: Sequence[int],
    *,
    raw_vtex: int,
) -> np.ndarray:
    """Decode one exact target occupancy mask from normalized LAND VTEX.

    The returned array is ``uint8`` with row-major ``[site_tile_y,
    site_tile_x]`` indexing.  A missing VTEX payload is an essential source
    failure rather than an empty/false tile, because treating missing data as
    non-road would silently alter topology.
    """

    minimum_x, maximum_x, minimum_y, maximum_y = _validated_bounds(bounds)
    width = (maximum_x - minimum_x + 1) * LAND_TEXTURE_SIDE
    height = (maximum_y - minimum_y + 1) * LAND_TEXTURE_SIDE
    mask = np.zeros((height, width), dtype=np.uint8)
    for cell_x, cell_y in target_grids(bounds):
        record = records.get((cell_x, cell_y))
        if record is None:
            raise ValueError(f"LAND source is missing target grid {(cell_x, cell_y)}")
        if record.texture_indices is None:
            raise ValueError(f"LAND {record.grid} has no VTEX payload")
        values = np.asarray(record.texture_indices, dtype=np.uint16)
        if values.shape != (LAND_TEXTURE_SIDE * LAND_TEXTURE_SIDE,):
            raise ValueError(f"LAND {record.grid} has an invalid normalized VTEX shape {values.shape}")
        x0 = (cell_x - minimum_x) * LAND_TEXTURE_SIDE
        y0 = (cell_y - minimum_y) * LAND_TEXTURE_SIDE
        mask[y0 : y0 + LAND_TEXTURE_SIDE, x0 : x0 + LAND_TEXTURE_SIDE] = (
            values.reshape(LAND_TEXTURE_SIDE, LAND_TEXTURE_SIDE) == int(raw_vtex)
        ).astype(np.uint8)
    return mask


def connected_components(mask: np.ndarray, *, connectivity: int) -> list[tuple[SiteTile, ...]]:
    """Return sorted connected components of a boolean tile mask.

    ``connectivity=8`` is the authoritative road topology because diagonal
    LAND paint is a plausible continuation at tile resolution.  Four-neighbour
    output is retained for diagnostics and tests only.  Components and their
    member tiles are returned in first-encounter row-major order; each member
    tuple is itself row-major.
    """

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("road mask must be two-dimensional")
    if connectivity not in (4, 8):
        raise ValueError("component connectivity must be 4 or 8")
    height, width = values.shape
    visited = np.zeros(values.shape, dtype=bool)
    orthogonal = ((-1, 0), (0, -1), (0, 1), (1, 0))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    offsets = orthogonal + (diagonal if connectivity == 8 else ())
    result: list[tuple[SiteTile, ...]] = []
    for tile_y in range(height):
        for tile_x in range(width):
            if not values[tile_y, tile_x] or visited[tile_y, tile_x]:
                continue
            stack = [(tile_x, tile_y)]
            visited[tile_y, tile_x] = True
            component: list[SiteTile] = []
            while stack:
                current_x, current_y = stack.pop()
                component.append((current_x, current_y))
                for offset_x, offset_y in offsets:
                    neighbour_x = current_x + offset_x
                    neighbour_y = current_y + offset_y
                    if not (0 <= neighbour_x < width and 0 <= neighbour_y < height):
                        continue
                    if values[neighbour_y, neighbour_x] and not visited[neighbour_y, neighbour_x]:
                        visited[neighbour_y, neighbour_x] = True
                        stack.append((neighbour_x, neighbour_y))
            result.append(tuple(sorted(component, key=lambda item: (item[1], item[0]))))
    return result


def _plan_tile_geometry(
    site_tile: SiteTile,
    *,
    frame_origin_gu: Sequence[float],
) -> dict[str, Any]:
    site_x, site_y = (int(value) for value in site_tile)
    minimum_x = site_x * TILE_SIZE_GU
    minimum_y = site_y * TILE_SIZE_GU
    maximum_x = (site_x + 1) * TILE_SIZE_GU
    maximum_y = (site_y + 1) * TILE_SIZE_GU
    origin_x, origin_y = (float(frame_origin_gu[0]), float(frame_origin_gu[1]))
    return {
        "plan_gu_bounds": {
            "min": [minimum_x, minimum_y],
            "max_exclusive": [maximum_x, maximum_y],
            "convention": "relative to frame SW origin; max is exclusive",
        },
        "plan_gu_center": [minimum_x + TILE_SIZE_GU / 2, minimum_y + TILE_SIZE_GU / 2],
        "absolute_gu_bounds": {
            "min": [origin_x + minimum_x, origin_y + minimum_y],
            "max_exclusive": [origin_x + maximum_x, origin_y + maximum_y],
        },
        "absolute_gu_center": [
            origin_x + minimum_x + TILE_SIZE_GU / 2,
            origin_y + minimum_y + TILE_SIZE_GU / 2,
        ],
    }


def _tile_detail(
    records: Mapping[Grid, LandRecord],
    *,
    grid: Grid,
    cell_tile: tuple[int, int],
    bounds: Bounds,
    frame_origin_gu: Sequence[float],
    raw_vtex: int,
    component_id: str | None = None,
) -> dict[str, Any]:
    cell_x, cell_y = grid
    tile_x, tile_y = (int(value) for value in cell_tile)
    record = records.get(grid)
    if record is None or record.texture_indices is None:
        raise ValueError(f"perimeter LAND source is missing VTEX at {grid}")
    raw_value = int(record.texture_index(tile_x, tile_y))
    site_tile = (
        (cell_x - bounds[0]) * LAND_TEXTURE_SIDE + tile_x,
        (cell_y - bounds[2]) * LAND_TEXTURE_SIDE + tile_y,
    )
    row: dict[str, Any] = {
        "grid": [cell_x, cell_y],
        "cell_local_tile": [tile_x, tile_y],
        "site_tile": [site_tile[0], site_tile[1]],
        "raw_vtex": raw_value,
    }
    row.update(_plan_tile_geometry(site_tile, frame_origin_gu=frame_origin_gu))
    if component_id is not None:
        row["component_id_8"] = component_id
    if raw_value != int(raw_vtex):
        raise ValueError(
            f"road tile detail requested raw VTEX {raw_vtex} but source {grid} {cell_tile} has {raw_value}"
        )
    return row


def _contiguous_runs(values: Sequence[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for value in values:
        value = int(value)
        if not runs or value != runs[-1][1] + 1:
            runs.append((value, value))
        else:
            runs[-1] = (runs[-1][0], value)
    return runs


def _boundary_tile_positions(side: str, along: int, bounds: Bounds) -> tuple[SiteTile, Grid, tuple[int, int], Grid, tuple[int, int]]:
    """Return target/outside site and source coordinates for one edge index."""

    minimum_x, maximum_x, minimum_y, maximum_y = bounds
    width = (maximum_x - minimum_x + 1) * LAND_TEXTURE_SIDE
    height = (maximum_y - minimum_y + 1) * LAND_TEXTURE_SIDE
    if side == "west":
        target_site = (0, along)
        target_grid = (minimum_x, minimum_y + along // LAND_TEXTURE_SIDE)
        target_tile = (0, along % LAND_TEXTURE_SIDE)
        outside_grid = (minimum_x - 1, target_grid[1])
        outside_tile = (LAND_TEXTURE_SIDE - 1, target_tile[1])
    elif side == "east":
        target_site = (width - 1, along)
        target_grid = (maximum_x, minimum_y + along // LAND_TEXTURE_SIDE)
        target_tile = (LAND_TEXTURE_SIDE - 1, along % LAND_TEXTURE_SIDE)
        outside_grid = (maximum_x + 1, target_grid[1])
        outside_tile = (0, target_tile[1])
    elif side == "south":
        target_site = (along, 0)
        target_grid = (minimum_x + along // LAND_TEXTURE_SIDE, minimum_y)
        target_tile = (along % LAND_TEXTURE_SIDE, 0)
        outside_grid = (target_grid[0], minimum_y - 1)
        outside_tile = (target_tile[0], LAND_TEXTURE_SIDE - 1)
    elif side == "north":
        target_site = (along, height - 1)
        target_grid = (minimum_x + along // LAND_TEXTURE_SIDE, maximum_y)
        target_tile = (along % LAND_TEXTURE_SIDE, LAND_TEXTURE_SIDE - 1)
        outside_grid = (target_grid[0], maximum_y + 1)
        outside_tile = (target_tile[0], 0)
    else:
        raise ValueError(f"unknown target boundary side: {side}")
    return target_site, target_grid, target_tile, outside_grid, outside_tile


def _boundary_plan_geometry(side: str, start: int, end: int, bounds: Bounds) -> dict[str, Any]:
    width = (bounds[1] - bounds[0] + 1) * LAND_TEXTURE_SIDE
    height = (bounds[3] - bounds[2] + 1) * LAND_TEXTURE_SIDE
    if side in ("west", "east"):
        axis = "y"
        border = 0 if side == "west" else width * TILE_SIZE_GU
    else:
        axis = "x"
        border = 0 if side == "south" else height * TILE_SIZE_GU
    return {
        "axis": axis,
        "coordinate_gu": border,
        "span_gu": [start * TILE_SIZE_GU, (end + 1) * TILE_SIZE_GU],
        "span_convention": "lower-inclusive, upper-exclusive in plan-frame GU",
    }


def derive_boundary_continuations(
    records: Mapping[Grid, LandRecord],
    target_mask: np.ndarray,
    bounds: Sequence[int],
    *,
    raw_vtex: int,
    frame_origin_gu: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Prove target-edge exits from orthogonally adjacent perimeter VTEX.

    The first return value contains only grouped, confirmed spans.  The second
    contains per-side counts.  The third is an explicit ledger of target-edge
    road tiles that were *not* confirmed outside the target; renderers must
    not turn those rows into exits.
    """

    normalized_bounds = _validated_bounds(bounds)
    values = np.asarray(target_mask, dtype=bool)
    expected_shape = (
        (normalized_bounds[3] - normalized_bounds[2] + 1) * LAND_TEXTURE_SIDE,
        (normalized_bounds[1] - normalized_bounds[0] + 1) * LAND_TEXTURE_SIDE,
    )
    if values.shape != expected_shape:
        raise ValueError(f"target road mask has {values.shape}; expected {expected_shape}")

    spans: list[dict[str, Any]] = []
    unconfirmed: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {
        "edge_road_tiles": {},
        "confirmed_edge_tiles": {},
        "unconfirmed_edge_tiles": {},
        "continuation_spans": {},
    }
    for side in SIDE_ORDER:
        along_limit = expected_shape[0] if side in ("west", "east") else expected_shape[1]
        confirmed_indices: list[int] = []
        edge_indices: list[int] = []
        side_unconfirmed: list[int] = []
        for along in range(along_limit):
            target_site, target_grid, target_tile, outside_grid, outside_tile = _boundary_tile_positions(
                side, along, normalized_bounds
            )
            if not bool(values[target_site[1], target_site[0]]):
                continue
            edge_indices.append(along)
            outside_record = records.get(outside_grid)
            if outside_record is None or outside_record.texture_indices is None:
                raise ValueError(f"perimeter LAND source is missing VTEX at {outside_grid}")
            outside_value = int(outside_record.texture_index(*outside_tile))
            if outside_value == int(raw_vtex):
                confirmed_indices.append(along)
            else:
                side_unconfirmed.append(along)
                target_detail = _tile_detail(
                    records,
                    grid=target_grid,
                    cell_tile=target_tile,
                    bounds=normalized_bounds,
                    frame_origin_gu=frame_origin_gu,
                    raw_vtex=raw_vtex,
                )
                unconfirmed.append(
                    {
                        "side": side,
                        "tile_along_edge": along,
                        "target_tile": target_detail,
                        "matching_outside_tile": {
                            "grid": [outside_grid[0], outside_grid[1]],
                            "cell_local_tile": [outside_tile[0], outside_tile[1]],
                            "raw_vtex": outside_value,
                        },
                        "rule": "not a continuation because the adjacent perimeter raw VTEX differs",
                    }
                )
        runs = _contiguous_runs(confirmed_indices)
        statistics["edge_road_tiles"][side] = len(edge_indices)
        statistics["confirmed_edge_tiles"][side] = len(confirmed_indices)
        statistics["unconfirmed_edge_tiles"][side] = len(side_unconfirmed)
        statistics["continuation_spans"][side] = len(runs)
        for span_index, (start, end) in enumerate(runs, start=1):
            target_rows: list[dict[str, Any]] = []
            outside_rows: list[dict[str, Any]] = []
            for along in range(start, end + 1):
                target_site, target_grid, target_tile, outside_grid, outside_tile = _boundary_tile_positions(
                    side, along, normalized_bounds
                )
                target_rows.append(
                    _tile_detail(
                        records,
                        grid=target_grid,
                        cell_tile=target_tile,
                        bounds=normalized_bounds,
                        frame_origin_gu=frame_origin_gu,
                        raw_vtex=raw_vtex,
                    )
                )
                outside_rows.append(
                    _tile_detail(
                        records,
                        grid=outside_grid,
                        cell_tile=outside_tile,
                        bounds=normalized_bounds,
                        frame_origin_gu=frame_origin_gu,
                        raw_vtex=raw_vtex,
                    )
                )
            spans.append(
                {
                    "continuation_id": f"{side[0].upper()}{span_index:02d}",
                    "side": side,
                    "target_tile_span": {
                        "axis": "y" if side in ("west", "east") else "x",
                        "start": start,
                        "end": end,
                        "end_inclusive": True,
                        "length_tiles": end - start + 1,
                    },
                    "target_tiles": target_rows,
                    "matching_outside_tiles": outside_rows,
                    "plan_gu_border": _boundary_plan_geometry(side, start, end, normalized_bounds),
                    "proof": "target edge raw VTEX equals the orthogonally adjacent perimeter LAND raw VTEX",
                }
            )
    spans.sort(key=lambda row: (SIDE_ORDER.index(str(row["side"])), row["target_tile_span"]["start"]))
    unconfirmed.sort(key=lambda row: (SIDE_ORDER.index(str(row["side"])), int(row["tile_along_edge"])))
    statistics["total_edge_road_tiles"] = sum(statistics["edge_road_tiles"].values())
    statistics["total_confirmed_edge_tiles"] = sum(statistics["confirmed_edge_tiles"].values())
    statistics["total_unconfirmed_edge_tiles"] = sum(statistics["unconfirmed_edge_tiles"].values())
    statistics["total_continuation_spans"] = len(spans)
    return spans, statistics, unconfirmed


def build_land_road_evidence(
    records: Mapping[Grid, LandRecord],
    bounds: Sequence[int],
    *,
    raw_vtex: int,
    frame_origin_gu: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the canonical source-derived road mask and evidence document."""

    normalized_bounds = _validated_bounds(bounds)
    expected_grids = set(source_selection_grids(normalized_bounds))
    missing = sorted(expected_grids - set(records))
    if missing:
        raise ValueError(f"LAND road source is missing target/perimeter grids: {missing}")
    for grid in sorted(expected_grids):
        if records[grid].texture_indices is None:
            raise ValueError(f"LAND road source has no VTEX payload at {grid}")

    target_mask = decode_land_road_mask(records, normalized_bounds, raw_vtex=raw_vtex)
    components_8 = connected_components(target_mask, connectivity=8)
    components_4 = connected_components(target_mask, connectivity=4)
    component_by_tile: dict[SiteTile, str] = {}
    component_rows: list[dict[str, Any]] = []
    for component_index, component in enumerate(components_8, start=1):
        component_id = f"C{component_index:03d}"
        for site_tile in component:
            component_by_tile[site_tile] = component_id
        min_site_x = min(tile[0] for tile in component)
        max_site_x = max(tile[0] for tile in component)
        min_site_y = min(tile[1] for tile in component)
        max_site_y = max(tile[1] for tile in component)
        centres = [
            (tile[0] * TILE_SIZE_GU + TILE_SIZE_GU / 2, tile[1] * TILE_SIZE_GU + TILE_SIZE_GU / 2)
            for tile in component
        ]
        component_rows.append(
            {
                "component_id": component_id,
                "connectivity": 8,
                "tile_count": len(component),
                "site_tile_bounds": {
                    "min": [min_site_x, min_site_y],
                    "max": [max_site_x, max_site_y],
                    "max_exclusive": [(max_site_x + 1), (max_site_y + 1)],
                },
                "plan_gu_bounds": {
                    "min": [min_site_x * TILE_SIZE_GU, min_site_y * TILE_SIZE_GU],
                    "max_exclusive": [
                        (max_site_x + 1) * TILE_SIZE_GU,
                        (max_site_y + 1) * TILE_SIZE_GU,
                    ],
                },
                "plan_gu_centroid": [
                    round(sum(point[0] for point in centres) / len(centres), 3),
                    round(sum(point[1] for point in centres) / len(centres), 3),
                ],
            }
        )

    road_tiles: list[dict[str, Any]] = []
    tile_id_by_site: dict[SiteTile, str] = {}
    tile_index = 0
    for site_y in range(target_mask.shape[0]):
        for site_x in range(target_mask.shape[1]):
            if not target_mask[site_y, site_x]:
                continue
            cell_x = normalized_bounds[0] + site_x // LAND_TEXTURE_SIDE
            cell_y = normalized_bounds[2] + site_y // LAND_TEXTURE_SIDE
            cell_tile = (site_x % LAND_TEXTURE_SIDE, site_y % LAND_TEXTURE_SIDE)
            row = _tile_detail(
                records,
                grid=(cell_x, cell_y),
                cell_tile=cell_tile,
                bounds=normalized_bounds,
                frame_origin_gu=frame_origin_gu,
                raw_vtex=raw_vtex,
                component_id=component_by_tile[(site_x, site_y)],
            )
            tile_index += 1
            row["tile_id"] = f"T{tile_index:04d}"
            tile_id_by_site[(site_x, site_y)] = row["tile_id"]
            road_tiles.append(row)

    # Cross-reference component members after global tile ids are assigned.
    # Components can interleave in row-major space, so deriving ids from the
    # component ordinal would be incorrect even though it looks deterministic.
    for component_row, component in zip(component_rows, components_8):
        component_row["tile_ids"] = [tile_id_by_site[site_tile] for site_tile in component]
        if len(set(component_row["tile_ids"])) != int(component_row["tile_count"]):
            raise AssertionError(f"component {component_row['component_id']} has duplicate tile ids")

    continuations, boundary_statistics, unconfirmed = derive_boundary_continuations(
        records,
        target_mask,
        normalized_bounds,
        raw_vtex=raw_vtex,
        frame_origin_gu=frame_origin_gu,
    )
    mask_bytes = np.ascontiguousarray(target_mask, dtype=np.uint8).tobytes(order="C")
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "source_kind": "LAND/VTEX",
        "source_semantics": "OpenMW-normalized LAND VTEX, row-major [tile_y, tile_x]",
        "target_bounds": {
            "min_x": normalized_bounds[0],
            "max_x": normalized_bounds[1],
            "min_y": normalized_bounds[2],
            "max_y": normalized_bounds[3],
        },
        "perimeter_bounds": {
            "min_x": normalized_bounds[0] - 1,
            "max_x": normalized_bounds[1] + 1,
            "min_y": normalized_bounds[2] - 1,
            "max_y": normalized_bounds[3] + 1,
        },
        "source_record_count": len(expected_grids),
        "source_record_selection": "target 7x7 plus complete one-cell bounding-box ring, including corners",
        "raw_vtex": {
            "value": int(raw_vtex),
            "ltex_index": int(raw_vtex) - 1 if int(raw_vtex) > 0 else None,
            "tile_size_gu": TILE_SIZE_GU,
            "target_mask_shape": [int(target_mask.shape[0]), int(target_mask.shape[1])],
        },
        "frame": {
            "origin_gu": [float(frame_origin_gu[0]), float(frame_origin_gu[1])],
            "plan_axis": "+x east, +y north",
            "tile_bounds_convention": "plan_gu_bounds min inclusive, max_exclusive exclusive",
        },
        "target_mask": {
            "dtype": "uint8",
            "shape": [int(target_mask.shape[0]), int(target_mask.shape[1])],
            "axis_order": "row-major [tile_y, tile_x]",
            "road_tile_count": int(np.count_nonzero(target_mask)),
            "base64": base64.b64encode(mask_bytes).decode("ascii"),
            "raw_bytes_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        },
        "road_tiles": road_tiles,
        "component_connectivity_authority": 8,
        "components": component_rows,
        "component_statistics": {
            "eight_neighbour_count": len(components_8),
            "eight_neighbour_sizes": [len(component) for component in components_8],
            "four_neighbour_count_diagnostic": len(components_4),
            "four_neighbour_sizes_diagnostic": [len(component) for component in components_4],
            "road_tile_count": int(np.count_nonzero(target_mask)),
        },
        "boundary_continuations": continuations,
        "boundary_statistics": boundary_statistics,
        "unconfirmed_target_edge_tiles": unconfirmed,
        "ordering": {
            "road_tiles": "row-major site tile [y,x]",
            "components": "first occupied row-major tile",
            "continuations": "side west,east,south,north, then span start",
        },
    }
    return target_mask, evidence


def evidence_mask(evidence: Mapping[str, Any]) -> np.ndarray:
    """Decode and validate the canonical mask embedded in evidence."""

    target_mask = evidence.get("target_mask")
    if not isinstance(target_mask, Mapping):
        raise ValueError("road evidence has no target_mask object")
    shape = target_mask.get("shape")
    encoded = target_mask.get("base64")
    if not isinstance(shape, list) or len(shape) != 2 or not isinstance(encoded, str):
        raise ValueError("road evidence target_mask is incomplete")
    raw = base64.b64decode(encoded, validate=True)
    expected = int(shape[0]) * int(shape[1])
    if len(raw) != expected:
        raise ValueError(f"road evidence mask has {len(raw)} bytes; expected {expected}")
    values = np.frombuffer(raw, dtype=np.uint8).reshape((int(shape[0]), int(shape[1])))
    if int(np.count_nonzero(values)) != int(target_mask.get("road_tile_count", -1)):
        raise ValueError("road evidence mask count does not match its declared count")
    return values.copy()


__all__ = [
    "CELL_SIZE_GU",
    "SIDE_ORDER",
    "TILE_SIZE_GU",
    "build_land_road_evidence",
    "connected_components",
    "decode_land_road_mask",
    "derive_boundary_continuations",
    "evidence_mask",
    "perimeter_grids",
    "source_selection_grids",
    "target_grids",
]
