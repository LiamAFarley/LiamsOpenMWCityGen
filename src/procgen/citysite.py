"""Build deterministic D-SITE planner surveys from TES3 exterior LAND.

This module is the host-side survey stage of the Cityforge pipeline:

``read-only LAND/metadata -> stitched fields and masks -> site_survey.json``

It deliberately does not author a plugin or alter any source asset.  The
Falkreath builder supplies the remap ESP as the owning LAND/LTEX source and
uses :mod:`procgen.espland` for the engine-faithful VHGT/VTEX interpretation.
The resulting 449 by 449 field is in game units at 128 GU spacing, while the
planner masks are one byte per 512 GU tile.  All ordering is explicit and all
derived collections are sorted so repeated runs with the same inputs produce
the same JSON and NPZ contents (apart from the normal ZIP timestamp metadata
inside NumPy's compressed container).

Inputs are the read-only remap ESP used for real terrain rendering,
``tamriel.esm`` for the authoritative source LAND and a sampled height
cross-check, terrain-cell metadata, the remap count report, the authoritative
settlement marker, and the v6 scatter audit.  Road evidence is decoded from
normalized LAND/VTEX raw value 78 in ``tamriel.esm``; the rejected cleaned
world-map graph is not opened or used.  Outputs are the D-SITE JSON contract,
its dense NPZ sidecar, a canonical ``land_roads.json`` evidence document, and
source/measurement evidence embedded in the JSON for the report and focused
tests.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from .espland import LAND_SIDE, THU_TO_GU, LandRecord, iter_land
from .landroads import (
    build_land_road_evidence,
    source_selection_grids,
)


CELL_SIZE_GU = 8192.0
TILE_SIZE_GU = 512.0
FIELD_SPACING_GU = 128.0
FIELD_VERTICES_PER_CELL = 64
SITE_ID = "falkreath_v1"
TARGET_BOUNDS = (-95, -89, -11, -5)
ANCHOR_GRID = (-92, -10)
REGION_ID = "R072"
REGION_NAME = "KREATHI DALE"
SETTLEMENT_MARKER_ID = "M0400"
SETTLEMENT_NAME = "Falkreath"
ROAD_RAW_VTEX = 78
WATER_LEVEL_GU = 0.0
SLOPE_BUILDABLE_LIMIT_DEG = 15.0
STEEP_BANK_LIMIT_DEG = 25.0
BAND_NAMES = {
    0: "sea",
    1: "coastal",
    2: "lowland",
    3: "highland",
    4: "mountain",
    5: "alpine",
}


@dataclass(frozen=True)
class SiteBounds:
    """Inclusive cell bounds and their absolute frame geometry."""

    min_x: int
    max_x: int
    min_y: int
    max_y: int

    @property
    def width_cells(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height_cells(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def width_gu(self) -> float:
        return self.width_cells * CELL_SIZE_GU

    @property
    def height_gu(self) -> float:
        return self.height_cells * CELL_SIZE_GU

    @property
    def origin_gu(self) -> tuple[float, float]:
        """Return the SW corner of the minimum cell, not the anchor cell."""

        return (self.min_x * CELL_SIZE_GU, self.min_y * CELL_SIZE_GU)

    @property
    def grids(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (x, y)
            for y in range(self.min_y, self.max_y + 1)
            for x in range(self.min_x, self.max_x + 1)
        )


def _bounds(value: Sequence[int]) -> SiteBounds:
    if len(value) != 4:
        raise ValueError("bounds must be [min_x, max_x, min_y, max_y]")
    result = SiteBounds(*(int(item) for item in value))
    if result.min_x > result.max_x or result.min_y > result.max_y:
        raise ValueError("minimum bounds must not exceed maximum bounds")
    return result


def _json_read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON source {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    """Hash a source file without loading large ESM/JSON files into memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash required source {path}: {exc}") from exc
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    return path


def _load_selected_land(path: Path, grids: set[tuple[int, int]]) -> dict[tuple[int, int], LandRecord]:
    """Read only selected LAND records while streaming a potentially large ESM."""

    selected: dict[tuple[int, int], LandRecord] = {}
    for record in iter_land(path):
        if record.grid in grids:
            selected[record.grid] = record
            if len(selected) == len(grids):
                break
    missing = sorted(grids - set(selected))
    if missing:
        raise ValueError(f"{path} is missing required LAND grids: {missing}")
    return selected


def _terrain_rows(path: Path, bounds: SiteBounds) -> dict[tuple[int, int], list[Any]]:
    document = _json_read(path)
    if not isinstance(document, Mapping) or not isinstance(document.get("cells"), list):
        raise ValueError(f"terrain cell source has no cells list: {path}")
    fields = document.get("fields")
    expected_fields = [
        "x",
        "y",
        "e_min_gu",
        "e_med_gu",
        "e_max_gu",
        "slope_mean_deg",
        "water_frac",
        "wdist_gu",
        "band",
        "land",
    ]
    if fields != expected_fields:
        raise ValueError(f"unexpected terrain_cells fields: {fields!r}")
    rows: dict[tuple[int, int], list[Any]] = {}
    for row in document["cells"]:
        if not isinstance(row, list) or len(row) != len(expected_fields):
            continue
        grid = (int(row[0]), int(row[1]))
        if bounds.min_x <= grid[0] <= bounds.max_x and bounds.min_y <= grid[1] <= bounds.max_y:
            if grid in rows:
                raise ValueError(f"duplicate terrain_cells row {grid}")
            rows[grid] = row
    missing = sorted(set(bounds.grids) - set(rows))
    if missing:
        raise ValueError(f"terrain_cells.json is missing target rows: {missing}")
    if len(rows) != len(bounds.grids):
        raise ValueError("terrain_cells target selection did not contain exactly one row per cell")
    return rows


def _stitch_heights(records: Mapping[tuple[int, int], LandRecord], bounds: SiteBounds) -> np.ndarray:
    """Join 65-vertex LAND grids on shared edges into float64 GU heights."""

    side_x = bounds.width_cells * FIELD_VERTICES_PER_CELL + 1
    side_y = bounds.height_cells * FIELD_VERTICES_PER_CELL + 1
    field = np.empty((side_y, side_x), dtype=np.float64)
    for iy in range(side_y):
        cell_y_offset, local_y = divmod(iy, FIELD_VERTICES_PER_CELL)
        cell_y = bounds.min_y + min(cell_y_offset, bounds.height_cells - 1)
        if iy == side_y - 1:
            local_y = FIELD_VERTICES_PER_CELL
        for ix in range(side_x):
            cell_x_offset, local_x = divmod(ix, FIELD_VERTICES_PER_CELL)
            cell_x = bounds.min_x + min(cell_x_offset, bounds.width_cells - 1)
            if ix == side_x - 1:
                local_x = FIELD_VERTICES_PER_CELL
            record = records[(cell_x, cell_y)]
            if record.heights_thu is None:
                raise ValueError(f"target LAND {record.grid} has no VHGT heights")
            field[iy, ix] = record.height_thu(local_x, local_y) * THU_TO_GU

    # Shared edges must agree.  The check catches a source/plugin mismatch
    # before a planner sees a hidden one-vertex seam in the NPZ field.
    for y in range(bounds.min_y, bounds.max_y + 1):
        for x in range(bounds.min_x, bounds.max_x):
            left = records[(x, y)]
            right = records[(x + 1, y)]
            assert left.heights_thu is not None and right.heights_thu is not None
            for local_y in range(LAND_SIDE):
                if left.heights_thu[local_y][64] != right.heights_thu[local_y][0]:
                    raise ValueError(f"LAND x seam mismatch at {(x, y)} local_y={local_y}")
    for y in range(bounds.min_y, bounds.max_y):
        for x in range(bounds.min_x, bounds.max_x + 1):
            south = records[(x, y)]
            north = records[(x, y + 1)]
            assert south.heights_thu is not None and north.heights_thu is not None
            for local_x in range(LAND_SIDE):
                if south.heights_thu[64][local_x] != north.heights_thu[0][local_x]:
                    raise ValueError(f"LAND y seam mismatch at {(x, y)} local_x={local_x}")
    return field


def _slope_field(height_gu: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(height_gu, FIELD_SPACING_GU, FIELD_SPACING_GU)
    return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float64, copy=False)


def _distance_field(water_vertices: np.ndarray) -> np.ndarray:
    """Return exact Euclidean distance to a water vertex on the 128-GU grid."""

    if water_vertices.dtype != np.bool_:
        water_vertices = water_vertices.astype(bool)
    if not bool(np.any(water_vertices)):
        return np.full(water_vertices.shape, np.inf, dtype=np.float64)
    # The inverse mask puts zero-valued sites at water vertices, which is the
    # distance_transform_edt convention required by this calculation.
    return distance_transform_edt(~water_vertices).astype(np.float64) * FIELD_SPACING_GU


def _tile_slices(tile_x: int, tile_y: int) -> tuple[slice, slice]:
    x0 = tile_x * 4
    y0 = tile_y * 4
    return slice(y0, y0 + 5), slice(x0, x0 + 5)


def _raw_vtex_tiles(records: Mapping[tuple[int, int], LandRecord], bounds: SiteBounds) -> np.ndarray:
    tiles = np.zeros((bounds.height_cells * 16, bounds.width_cells * 16), dtype=np.uint16)
    for cell_x, cell_y in bounds.grids:
        record = records[(cell_x, cell_y)]
        if record.texture_indices is None:
            raise ValueError(f"target LAND {record.grid} has no VTEX payload")
        offset_x = (cell_x - bounds.min_x) * 16
        offset_y = (cell_y - bounds.min_y) * 16
        tiles[offset_y : offset_y + 16, offset_x : offset_x + 16] = np.asarray(
            record.texture_indices, dtype=np.uint16
        ).reshape(16, 16)
    return tiles


def _scatter_density(path: Path, bounds: SiteBounds) -> np.ndarray:
    document = _json_read(path)
    density = document.get("density") if isinstance(document, Mapping) else None
    cells = density.get("cells") if isinstance(density, Mapping) else None
    if not isinstance(cells, list):
        raise ValueError(f"scatter audit has no density.cells list: {path}")
    result = np.zeros((bounds.height_cells * 16, bounds.width_cells * 16), dtype=np.float32)
    origin_x, origin_y = bounds.origin_gu
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("refs"), list):
            continue
        for ref in cell["refs"]:
            if not isinstance(ref, Mapping):
                continue
            position = ref.get("position_gu")
            if not isinstance(position, list) or len(position) < 2:
                raise ValueError("scatter density ref has no two-dimensional position_gu")
            local_x = float(position[0]) - origin_x
            local_y = float(position[1]) - origin_y
            tile_x = math.floor(local_x / TILE_SIZE_GU)
            tile_y = math.floor(local_y / TILE_SIZE_GU)
            if 0 <= tile_x < result.shape[1] and 0 <= tile_y < result.shape[0]:
                result[tile_y, tile_x] += 1.0
    return result


def _b64_array(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes(order="C")).decode("ascii")


def _input_reference(path: Path, workspace: Path) -> str:
    try:
        display = path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        display = str(path.resolve()).replace("\\", "/")
    return f"{display} (sha256:{sha256_file(path)})"


def _source_metric_check(
    records: Mapping[tuple[int, int], LandRecord],
    base_records: Mapping[tuple[int, int], LandRecord],
    terrain_rows: Mapping[tuple[int, int], list[Any]],
    bounds: SiteBounds,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    base_vertices = 0
    max_base_delta_thu = 0
    for grid in bounds.grids:
        remap = records[grid]
        base = base_records[grid]
        if remap.heights_thu is None or base.heights_thu is None:
            raise ValueError(f"height cross-check cannot use empty LAND {grid}")
        deltas = [
            remap.heights_thu[y][x] - base.heights_thu[y][x]
            for y in range(LAND_SIDE)
            for x in range(LAND_SIDE)
        ]
        base_vertices += len(deltas)
        max_base_delta_thu = max(max_base_delta_thu, max(abs(item) for item in deltas))
        remap_values_gu = [value * THU_TO_GU for row in remap.heights_thu for value in row]
        row = terrain_rows[grid]
        measured = {
            "elev_min_gu": int(min(remap_values_gu)),
            "elev_med_gu": int(statistics.median(remap_values_gu)),
            "elev_max_gu": int(max(remap_values_gu)),
            "source_water": bool(min(remap_values_gu) <= WATER_LEVEL_GU),
        }
        expected = {
            "elev_min_gu": int(row[2]),
            "elev_med_gu": int(row[3]),
            "elev_max_gu": int(row[4]),
            "source_water": bool(float(row[2]) <= WATER_LEVEL_GU),
        }
        checks.append(
            {
                "grid": [grid[0], grid[1]],
                "terrain_cells": expected,
                "land_measured": measured,
                "delta_gu": {
                    key: measured[key] - expected[key]
                    for key in ("elev_min_gu", "elev_med_gu", "elev_max_gu")
                },
                "base_esm_max_abs_delta_thu": max(abs(item) for item in deltas),
            }
        )
    return checks, {
        "records_compared": len(checks),
        "vertices_compared": base_vertices,
        "max_abs_delta_thu": max_base_delta_thu,
        "max_abs_delta_gu": max_base_delta_thu * THU_TO_GU,
    }


def _water_bodies(water_cells: set[tuple[int, int]], bounds: SiteBounds) -> list[dict[str, Any]]:
    remaining = set(water_cells)
    bodies: list[dict[str, Any]] = []
    while remaining:
        start = min(remaining)
        component = {start}
        queue = [start]
        remaining.remove(start)
        while queue:
            x, y = queue.pop()
            for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        shoreline_edges = 0
        for x, y in component:
            for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbour not in component:
                    shoreline_edges += 1
        bodies.append(
            {
                "kind": "lake",
                "cells": [[x, y] for x, y in sorted(component)],
                "shore_length_gu": shoreline_edges * int(CELL_SIZE_GU),
                "shore_length_definition": "4-neighbour edges inside the surveyed block; block boundary is counted",
            }
        )
    return bodies


def build_site_survey(
    *,
    workspace: Path,
    land_source: Path,
    base_esm: Path,
    terrain_cells_path: Path,
    remap_report_path: Path,
    settlements_path: Path,
    scatter_path: Path,
    output_dir: Path,
    town_grammars_path: Path | None = None,
) -> dict[str, Any]:
    """Build the complete host-side Falkreath survey and patch grammar data."""

    bounds = _bounds(TARGET_BOUNDS)
    required = (
        (land_source, "LAND source"),
        (base_esm, "base ESM"),
        (terrain_cells_path, "terrain cell source"),
        (remap_report_path, "LAND remap report"),
        (settlements_path, "settlement source"),
        (scatter_path, "scatter source"),
    )
    for path, label in required:
        _require_file(path, label)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = set(bounds.grids)
    records = _load_selected_land(land_source, target)
    # The road source selection is deliberately wider than the terrain render
    # selection.  All 81 records are decoded directly from tamriel.esm so an
    # exit is never inferred from a graph line that happens to cross the site
    # frame.
    road_source_grids = set(source_selection_grids(TARGET_BOUNDS))
    base_records = _load_selected_land(base_esm, road_source_grids)
    terrain_rows = _terrain_rows(terrain_cells_path, bounds)
    settlements = _json_read(settlements_path)
    if not isinstance(settlements, Mapping) or not isinstance(settlements.get("settlements"), list):
        raise ValueError("settlements.json has no settlements list")
    marker_rows = [
        item
        for item in settlements["settlements"]
        if isinstance(item, Mapping) and item.get("marker_id") == SETTLEMENT_MARKER_ID
    ]
    if len(marker_rows) != 1:
        raise ValueError(f"expected one authoritative {SETTLEMENT_MARKER_ID} marker, found {len(marker_rows)}")
    marker = marker_rows[0]
    if marker.get("name") != SETTLEMENT_NAME or marker.get("game_cell") != list(ANCHOR_GRID):
        raise ValueError(f"authoritative marker mismatch: {marker}")

    height_gu = _stitch_heights(records, bounds)
    slope_deg = _slope_field(height_gu)
    water_vertices = height_gu <= WATER_LEVEL_GU
    water_distance_gu = _distance_field(water_vertices)
    raw_vtex_tiles = _raw_vtex_tiles(records, bounds)
    authoritative_road_mask, roads = build_land_road_evidence(
        base_records,
        TARGET_BOUNDS,
        raw_vtex=ROAD_RAW_VTEX,
        frame_origin_gu=bounds.origin_gu,
    )
    source_raw_vtex_tiles = _raw_vtex_tiles(base_records, bounds)
    if not np.array_equal(source_raw_vtex_tiles, raw_vtex_tiles):
        mismatch_count = int(np.count_nonzero(source_raw_vtex_tiles != raw_vtex_tiles))
        raise ValueError(
            "render LAND source VTEX differs from authoritative tamriel.esm "
            f"at {mismatch_count} target tiles; refusing source/highlight mismatch"
        )
    scatter_tiles = _scatter_density(scatter_path, bounds)
    tile_side = bounds.width_cells * 16
    water_mask = np.zeros((tile_side, tile_side), dtype=np.uint8)
    steep_bank_mask = np.zeros((tile_side, tile_side), dtype=np.uint8)
    slope_tile_mean = np.zeros((tile_side, tile_side), dtype=np.float32)
    for tile_y in range(tile_side):
        for tile_x in range(tile_side):
            ys, xs = _tile_slices(tile_x, tile_y)
            patch_height = height_gu[ys, xs]
            patch_slope = slope_deg[ys, xs]
            water_mask[tile_y, tile_x] = 1 if bool(np.any(patch_height <= WATER_LEVEL_GU)) else 0
            slope_tile_mean[tile_y, tile_x] = float(np.mean(patch_slope))
    for tile_y in range(tile_side):
        for tile_x in range(tile_side):
            if water_mask[tile_y, tile_x]:
                continue
            neighbour_water = any(
                0 <= nx < tile_side
                and 0 <= ny < tile_side
                and water_mask[ny, nx]
                for nx, ny in (
                    (tile_x - 1, tile_y),
                    (tile_x + 1, tile_y),
                    (tile_x, tile_y - 1),
                    (tile_x, tile_y + 1),
                )
            )
            ys, xs = _tile_slices(tile_x, tile_y)
            if neighbour_water and float(np.max(slope_deg[ys, xs])) >= STEEP_BANK_LIMIT_DEG:
                steep_bank_mask[tile_y, tile_x] = 1
    buildable_mask = (
        (water_mask == 0)
        & (slope_tile_mean < SLOPE_BUILDABLE_LIMIT_DEG)
        & (steep_bank_mask == 0)
    ).astype(np.uint8)
    road_mask = authoritative_road_mask.astype(np.uint8, copy=False)

    metric_checks, base_check = _source_metric_check(records, base_records, terrain_rows, bounds)
    water_cells = {
        grid
        for grid in bounds.grids
        if bool(np.min(height_gu[(grid[1] - bounds.min_y) * 64 : (grid[1] - bounds.min_y + 1) * 64 + 1,
                                  (grid[0] - bounds.min_x) * 64 : (grid[0] - bounds.min_x + 1) * 64 + 1]) <= WATER_LEVEL_GU)
    }
    row_water_cells = {grid for grid in bounds.grids if float(terrain_rows[grid][6]) > 0.0}
    road_tile_count = int(np.count_nonzero(road_mask))
    report_document = _json_read(remap_report_path)
    report_road_count = sum(
        int(item.get("raw_vtex_counts", {}).get(str(ROAD_RAW_VTEX), 0))
        for item in report_document.get("per_cell_counts", [])
        if isinstance(item, Mapping)
        and item.get("grid") in [[x, y] for x, y in bounds.grids]
    )
    if road_tile_count != report_road_count:
        raise ValueError(
            "authoritative tamriel.esm raw VTEX 78 count disagrees with the "
            f"remap report: source={road_tile_count} report={report_road_count}"
        )

    output_ltex = (
        report_document.get("output_ltex", {}).get("records", {}).get(str(ROAD_RAW_VTEX - 1))
        if isinstance(report_document, Mapping)
        else None
    )
    if not isinstance(output_ltex, Mapping):
        raise ValueError("LAND remap report has no output LTEX metadata for raw VTEX 78")
    roads["raw_vtex"].update(
        {
            "output_ltex_index": ROAD_RAW_VTEX - 1,
            "output_ltex_record_id": str(output_ltex.get("record_id", "")),
            "output_texture_path": str(output_ltex.get("file_name", "")),
        }
    )

    roads["inputs"] = {
        "source_plugin": _input_reference(base_esm, workspace),
        "render_land_source": _input_reference(land_source, workspace),
        "remap_report": _input_reference(remap_report_path, workspace),
    }
    roads["source_count_crosscheck"] = {
        "decoded_target_raw_vtex_78": road_tile_count,
        "remap_report_target_raw_vtex_78": report_road_count,
        "counts_match": road_tile_count == report_road_count,
    }
    roads["rejected_vector_graph"] = {
        "path": "output/mapdata/roads_graph_clean.json",
        "used_as_input": False,
        "used_for_geometry": False,
        "status": "removed legacy provenance only; file was not opened",
    }
    roads_path = output_dir / "land_roads.json"
    roads_path.write_text(json.dumps(roads, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    frame_origin = [float(value) for value in bounds.origin_gu]
    site_span_gu = [bounds.width_gu, bounds.height_gu]
    # The fixed 4k canvas leaves the required 1024 GU margin on every side;
    # render_site.py records the same affine map after the Blender camera run.
    render_resolution = 4096
    render_margin_gu = 1024.0
    render_total_gu = bounds.width_gu + 2.0 * render_margin_gu
    px_per_gu = render_resolution / render_total_gu
    margin_px = render_margin_gu * px_per_gu
    render_mapping = {
        "site_topdown.png": {
            "px_per_gu": px_per_gu,
            "origin_px": [margin_px, render_resolution - margin_px],
            "y_down_image": True,
            "origin_semantics": "frame SW corner in image pixels",
            "resolution": [render_resolution, render_resolution],
            "margin_gu": render_margin_gu,
            "transform": "px_x=origin_px[0]+(gu_x-origin_gu[0])*px_per_gu; px_y=origin_px[1]-(gu_y-origin_gu[1])*px_per_gu",
        }
    }

    cells: list[dict[str, Any]] = []
    for grid in bounds.grids:
        row = terrain_rows[grid]
        record = records[grid]
        assert record.texture_indices is not None
        counts: dict[str, int] = {}
        for value in record.texture_indices:
            counts[str(int(value))] = counts.get(str(int(value)), 0) + 1
        cells.append(
            {
                "grid": [grid[0], grid[1]],
                "elev_min_gu": int(row[2]),
                "elev_med_gu": int(row[3]),
                "elev_max_gu": int(row[4]),
                "slope_mean_deg": float(row[5]),
                "water_frac": float(row[6]),
                "water_dist_gu": int(row[7]),
                "band": BAND_NAMES.get(int(row[8]), str(row[8])),
                "vtex_histogram": dict(sorted(counts.items(), key=lambda pair: int(pair[0]))),
                "road_tiles_78": int(counts.get(str(ROAD_RAW_VTEX), 0)),
                "land_flag": int(row[9]),
                "source_land_min_gu_measured": int(metric_checks[len(cells)]["land_measured"]["elev_min_gu"]),
                "source_land_med_gu_measured": int(metric_checks[len(cells)]["land_measured"]["elev_med_gu"]),
                "source_land_max_gu_measured": int(metric_checks[len(cells)]["land_measured"]["elev_max_gu"]),
            }
        )

    fields_path = output_dir / "survey_fields.npz"
    np.savez_compressed(
        fields_path,
        height_gu=height_gu.astype(np.float64),
        slope_deg=slope_deg.astype(np.float64),
        water_distance_gu=water_distance_gu.astype(np.float64),
        water_vertices=water_vertices.astype(np.uint8),
        x_gu=np.arange(height_gu.shape[1], dtype=np.float64) * FIELD_SPACING_GU,
        y_gu=np.arange(height_gu.shape[0], dtype=np.float64) * FIELD_SPACING_GU,
        raw_vtex_tiles=raw_vtex_tiles,
    )

    survey: dict[str, Any] = {
        "schema_version": 1,
        "survey_id": SITE_ID,
        "target_cells": {
            "min_x": bounds.min_x,
            "max_x": bounds.max_x,
            "min_y": bounds.min_y,
            "max_y": bounds.max_y,
        },
        "frame": {
            "origin_gu": frame_origin,
            "cell_size_gu": CELL_SIZE_GU,
            "axis_convention": "+x east, +y north; plan yaw = degrees CCW from +x",
            "render_mapping": render_mapping,
            "site_span_gu": site_span_gu,
            "field_spacing_gu": FIELD_SPACING_GU,
        },
        "inputs": {
            "land_source": _input_reference(land_source, workspace),
            "base_esm": _input_reference(base_esm, workspace),
            "terrain_cells": _input_reference(terrain_cells_path, workspace),
            "land_roads_source": _input_reference(base_esm, workspace),
            "land_remap_report": _input_reference(remap_report_path, workspace),
            "settlements": _input_reference(settlements_path, workspace),
            "existing_scatter": _input_reference(scatter_path, workspace),
        },
        "region": {
            "id": REGION_ID,
            "name": REGION_NAME,
            "cell_set_definition": "ptr_planning_polygon",
        },
        "seed_settlement": {
            "marker_id": SETTLEMENT_MARKER_ID,
            "name": SETTLEMENT_NAME,
            "anchor_cell": list(ANCHOR_GRID),
            "masterlist": {"type": "Town (hold capital)", "status": "planned"},
            "authoritative_source": "output/mapdata/settlements.json",
        },
        "cells": cells,
        "tile_grids": {
            "tile_size_gu": int(TILE_SIZE_GU),
            "side": tile_side,
            "axis_order": "row-major [y,x], SW frame origin",
            "water_mask": _b64_array(water_mask),
            "buildable_mask": _b64_array(buildable_mask),
            "road_mask": _b64_array(road_mask),
            "scatter_density": _b64_array(scatter_tiles.astype(np.float32)),
            "water_mask_dtype": "uint8",
            "buildable_mask_dtype": "uint8",
            "road_mask_dtype": "uint8",
            "scatter_density_dtype": "float32",
            "steep_bank_mask": _b64_array(steep_bank_mask),
            "slope_tile_mean_deg": _b64_array(slope_tile_mean),
        },
        "roads": roads,
        "water": {
            "level_gu": WATER_LEVEL_GU,
            "bodies": _water_bodies(water_cells, bounds),
            "water_cells_measured": [[x, y] for x, y in sorted(water_cells)],
            "mask_definition": "512-GU tile is water when any of its 5x5 128-GU LAND vertices is at or below z=0",
        },
        "constraints": {
            "conform_max_slope_deg": SLOPE_BUILDABLE_LIMIT_DEG,
            "flatten_pad_max_cut_fill_gu": 400,
            "door_road_max_gu": 1500,
            "tree_clearance_shell_gu": 600,
            "min_building_gap_gu": 200,
            "steep_bank_slope_deg": STEEP_BANK_LIMIT_DEG,
            "water_threshold_gu": WATER_LEVEL_GU,
        },
        "stats": {
            "land_cells": sum(int(row[9]) for row in terrain_rows.values()),
            "water_cells": len(water_cells),
            "terrain_above_water_cells": len(bounds.grids) - len(water_cells),
            "elev_range_gu": [int(min(row[2] for row in terrain_rows.values())), int(max(row[4] for row in terrain_rows.values()))],
            "slope_mean_deg": round(float(np.mean([float(row[5]) for row in terrain_rows.values()])), 2),
            "road_tiles_78": road_tile_count,
            "scatter_refs_measured": int(np.sum(scatter_tiles)),
            "buildable_tiles": int(np.count_nonzero(buildable_mask)),
            "water_tiles": int(np.count_nonzero(water_mask)),
            "road_components_8": int(roads["component_statistics"]["eight_neighbour_count"]),
            "road_components_4_diagnostic": int(roads["component_statistics"]["four_neighbour_count_diagnostic"]),
            "road_continuation_spans": int(roads["boundary_statistics"]["total_continuation_spans"]),
        },
        "source_crosscheck": {
            "terrain_rows": len(terrain_rows),
            "land_source_records": len(records),
            "land_roads_source_records": len(base_records),
            "base_esm": base_check,
            "per_cell_height_metrics": metric_checks,
            "water_cells_from_land": [[x, y] for x, y in sorted(water_cells)],
            "water_cells_from_terrain_cells_water_frac": [[x, y] for x, y in sorted(row_water_cells)],
            "water_cell_sets_match": water_cells == row_water_cells,
            "road_tiles_78_measured_from_land": road_tile_count,
            "road_tiles_78_report_crosscheck": report_road_count,
            "road_tile_counts_match_report": road_tile_count == report_road_count,
            "road_mask_source": "tamriel.esm LAND/VTEX raw 78",
            "road_mask_render_source_raw_vtex_match": bool(
                np.array_equal(source_raw_vtex_tiles, raw_vtex_tiles)
            ),
            "rejected_vector_graph_used": False,
            "all_land_flags_one": all(int(row[9]) == 1 for row in terrain_rows.values()),
        },
        "artifacts": {
            "fields_npz": "survey_fields.npz",
            "fields_npz_sha256": sha256_file(fields_path),
            "land_roads": "land_roads.json",
            "land_roads_sha256": sha256_file(roads_path),
            "height_field_shape": list(height_gu.shape),
            "height_field_dtype": str(height_gu.dtype),
            "height_field_spacing_gu": FIELD_SPACING_GU,
        },
        "determinism": {
            "seed": 20260801,
            "ordering": "cell rows, LAND road tiles, components, continuation spans, and masks use explicit sorted row-major order",
        },
    }
    survey_path = output_dir / "site_survey.json"
    survey_path.write_text(json.dumps(survey, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    grammar_evidence: dict[str, Any] | None = None
    if town_grammars_path is not None:
        grammar_evidence = patch_town_grammar(town_grammars_path)
    return {
        "survey": survey,
        "survey_path": survey_path,
        "fields_path": fields_path,
        "land_roads_path": roads_path,
        "grammar_evidence": grammar_evidence,
        "source_hashes": {key: value.split("sha256:", 1)[1].rstrip(")") for key, value in survey["inputs"].items()},
    }


def patch_town_grammar(path: Path) -> dict[str, Any]:
    """Correct only the stale (-92,-10) marker row in town_grammars.json.

    The file is a large pretty-printed JSON document.  A narrowly anchored
    byte replacement preserves every unrelated byte and avoids reserializing
    155 MB of generated data with a different formatter or key order.
    """

    _require_file(path, "town grammar source")
    before_hash = sha256_file(path)
    raw = path.read_bytes()
    pattern = re.compile(
        rb'("grid"\s*:\s*\[\s*-92\s*,\s*-10\s*\].{0,1800}?"name"\s*:\s*)"Farm"(\s*,\s*"settlement_key"\s*:\s*)"marker:0399"',
        re.DOTALL,
    )
    match = pattern.search(raw)
    if match is None:
        already = re.search(
            rb'"grid"\s*:\s*\[\s*-92\s*,\s*-10\s*\].{0,1800}?"name"\s*:\s*"Falkreath".{0,120}?"settlement_key"\s*:\s*"marker:0400"',
            raw,
            re.DOTALL,
        )
        if already is None:
            raise ValueError("could not locate stale Falkreath grammar row")
        return {
            "path": str(path),
            "changed": False,
            "before_sha256": before_hash,
            "after_sha256": before_hash,
            "matched_grid": [-92, -10],
            "before": {"name": "Falkreath", "settlement_key": "marker:0400"},
            "after": {"name": "Falkreath", "settlement_key": "marker:0400"},
            "replacement_count": 0,
            "unrelated_bytes_preserved": True,
        }
    before = {"name": "Farm", "settlement_key": "marker:0399"}
    replacement = match.group(1) + b'"Falkreath"' + match.group(2) + b'"marker:0400"'
    updated = raw[: match.start()] + replacement + raw[match.end() :]
    path.write_bytes(updated)
    after_hash = sha256_file(path)
    return {
        "path": str(path),
        "changed": True,
        "before_sha256": before_hash,
        "after_sha256": after_hash,
        "matched_grid": [-92, -10],
        "before": before,
        "after": {"name": "Falkreath", "settlement_key": "marker:0400"},
        "replacement_count": 1,
        "replacement_byte_delta": len(replacement) - (match.end() - match.start()),
        "prefix_bytes_preserved": raw[: match.start()] == updated[: match.start()],
        "suffix_bytes_preserved": raw[match.end() :] == updated[match.start() + len(replacement) :],
        "unrelated_bytes_preserved": True,
    }


__all__ = [
    "ANCHOR_GRID",
    "CELL_SIZE_GU",
    "FIELD_SPACING_GU",
    "ROAD_RAW_VTEX",
    "SITE_ID",
    "SiteBounds",
    "TARGET_BOUNDS",
    "build_site_survey",
    "patch_town_grammar",
    "sha256_file",
]
