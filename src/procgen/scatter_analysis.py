"""Deterministic, terrain-aware scatter analysis helpers.

This module is the analysis stage for the Vorndgad scatter investigation.  It
does not author TES3 records and it never opens a NIF.  Binary CELL/LAND
parsing stays in :mod:`procgen.espscan` and :mod:`procgen.espland`; this module
joins the two-pass Sky_Main/Tamriel_Data definitions, applies the explicit
screening policy, and aggregates the terrain conditions needed by the later
scatter generator.  LAND VTEX tiles are already in OpenMW-normalized order at
the parser boundary; texture records retain both the raw VTEX audit value and
the resolved owning-plugin LTEX index (or the explicit base-texture sentinel).

All coordinates in this module are TES3 game units unless a field is suffixed
``_thu``.  Reference rotations are TES3 radians.  The analysis intentionally
uses no random numbers: sorting and percentile rules are explicit so repeated
runs over the same inputs produce byte-identical JSON documents.

Raw TES3 reference geometry uses the OpenMW/OpenSceneGraph static-placement
matrix ``Rx(-rx) @ Ry(-ry) @ Rz(-rz)``.  OpenMW 0.51
``components/misc/convert.hpp:50-54`` constructs ``Qz * Qy * Qx`` and
OpenSceneGraph's ``osg::Quat::operator*`` reverses the Hamilton operands.  The
point helper below applies that authoritative matrix directly; it is used by
the analysis bbox gate and by the scatter generator through
``transformed_bbox``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .espland import (
    BASE_LAND_TEXTURE_NAME,
    BASE_LAND_TEXTURE_PATH,
    CELL_SIZE_GAME_UNITS,
    THU_TO_GU,
    LandRecord,
    LandscapeTexture,
    height_at_game_position,
    validate_land_samples,
)
from .espscan import CellReference, CellSummary, ObjectDefinition


GIANT_BBOX_THRESHOLD_GU = 2_500.0
STACKER_Z_OFFSET_THRESHOLD_GU = 300.0
ADJACENCY_RADIUS_GU = 512.0
SLOPE_SAMPLE_SPACING_GU = 128.0
WATER_THRESHOLD_THU = 0.0
WATER_SAMPLE_SPACING_GU = 128.0
WATER_MARGIN_CELLS = 8
PERCENTILES = (0.10, 0.50, 0.90)

ELEVATION_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sea", float("-inf"), 0.0),
    ("coastal", 0.0, 500.0),
    ("lowland", 500.0, 2_000.0),
    ("highland", 2_000.0, 8_000.0),
    ("mountain", 8_000.0, 15_000.0),
    ("alpine", 15_000.0, float("inf")),
)

SCATTER_CATEGORIES = frozenset({"flora", "rocks", "terrain", "terrain-landscape"})
OUTPUT_CATEGORY = {"terrain": "terrain-landscape"}
SCREEN_CLASS_BY_CATEGORY = {
    "exterior": "structures",
    "interior": "interior",
    "door": "structures",
    "clutter": "clutter",
    "other": "other",
}
SLOPE_DIRECTIONS: tuple[tuple[int, int], ...] = tuple(
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dx, dy) != (0, 0)
)

# Rotation is not independent of terrain in the source ESP.  Keep the bins
# deliberately coarse: they are a measurement report and a generation lookup,
# not a new ecological rule.  A mesh with no full-rotation observations in the
# relevant bin must remain Z-only in the generator.
ROTATION_SLOPE_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 8.0),
    (8.0, 16.0),
    (16.0, 24.0),
    (24.0, 32.0),
    (32.0, 45.0),
    (45.0, float("inf")),
)


def _round(value: float | int | None, digits: int = 6) -> float | int | None:
    """Round numeric output while normalizing negative zero."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    rounded = round(float(value), digits)
    return 0.0 if abs(rounded) < 0.5 * (10.0 ** -digits) else rounded


def _round_list(values: Sequence[float | int | None]) -> list[float | int | None]:
    return [_round(value) for value in values]


def _finite_values(values: Iterable[float | int | None]) -> list[float]:
    result = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return result


def distribution(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    """Return a deterministic min/p10/p50/p90/max/mean envelope.

    Percentiles use linear interpolation over ``(n - 1) * q``.  The rule is
    written here rather than delegated to a numerical package so fixture tests
    and production runs share exactly the same behavior.
    """

    ordered = sorted(_finite_values(values))
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "max": None,
            "mean": None,
        }

    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        fraction = index - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    return {
        "count": len(ordered),
        "min": _round(ordered[0]),
        "p10": _round(quantile(0.10)),
        "p50": _round(quantile(0.50)),
        "p90": _round(quantile(0.90)),
        "max": _round(ordered[-1]),
        "mean": _round(sum(ordered) / len(ordered)),
    }


def _sorted_counter(counter: Counter[str | int | None], limit: int | None = None) -> list[dict[str, object]]:
    rows = sorted(counter.items(), key=lambda item: (-item[1], str(item[0]).casefold(), str(item[0])))
    if limit is not None:
        rows = rows[:limit]
    return [{"value": value, "count": count} for value, count in rows]


def _counter_dict(counter: Counter[str | int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda item: (str(item).casefold(), str(item)))}


def elevation_band(elevation_gu: float) -> str:
    """Map a terrain height in GU to the shared terrain_maps band names."""

    for name, lower, upper in ELEVATION_BANDS:
        if elevation_gu > lower and elevation_gu <= upper:
            return name
    # The first band includes every negative value through zero.
    return "sea"


def rotation_mode(rotation: Sequence[float] | None) -> str:
    """Classify a TES3 rotation as z-only or full 3-axis rotation."""

    if rotation is None:
        return "z_only"
    if len(rotation) != 3:
        raise ValueError("reference rotation must contain three radians")
    return "z_only" if abs(float(rotation[0])) <= 1e-6 and abs(float(rotation[1])) <= 1e-6 else "full"


def rotation_slope_dependency(
    rotations: Sequence[Sequence[float]], slopes: Sequence[float]
) -> list[dict[str, Any]]:
    """Summarize per-mesh rotation behavior conditioned on measured slope.

    The analysis historically emitted one rotation distribution per mesh.  The
    ESP refs also contain slope at each placement, so this extension preserves
    the observed relationship without inventing a tilt rule.  Empty bins are
    retained to make the lookup deterministic and explicit.
    """

    if len(rotations) != len(slopes):
        raise ValueError("rotation/slope samples must have equal lengths")
    output: list[dict[str, Any]] = []
    for lower, upper in ROTATION_SLOPE_BINS:
        selected = [
            (tuple(float(value) for value in rotation), float(slope))
            for rotation, slope in zip(rotations, slopes)
            if len(rotation) == 3
            and math.isfinite(float(slope))
            and float(slope) >= lower
            and (float(slope) < upper if math.isfinite(upper) else True)
        ]
        mode_counts = Counter(rotation_mode(rotation) for rotation, _slope in selected)
        output.append(
            {
                "slope_min_deg": _round(lower),
                "slope_max_deg": _round(upper) if math.isfinite(upper) else None,
                "count": len(selected),
                "mode_counts": _counter_dict(mode_counts),
                "x_radians": distribution(rotation[0] for rotation, _slope in selected),
                "y_radians": distribution(rotation[1] for rotation, _slope in selected),
                "z_radians": distribution(rotation[2] for rotation, _slope in selected),
                "z_only_fraction": _round(
                    mode_counts.get("z_only", 0) / len(selected)
                )
                if selected
                else None,
            }
        )
    return output


def normalize_mesh_key(mesh: str) -> str:
    return mesh.replace("/", "\\").strip().casefold()


def merge_object_definitions(
    dependency_definitions: Mapping[str, ObjectDefinition],
    source_definitions: Mapping[str, ObjectDefinition],
) -> dict[str, ObjectDefinition]:
    """Merge master definitions followed by source overrides.

    The dependency (Tamriel_Data.esm) is scanned first.  Sky_Main definitions
    then win on a case-folded object id, matching TES3 load-order semantics.
    """

    merged = {str(key).casefold(): value for key, value in dependency_definitions.items()}
    merged.update({str(key).casefold(): value for key, value in source_definitions.items()})
    return merged


def select_vorndgad_cells(
    cells: Iterable[CellSummary],
    *,
    region_name: str = "Vorndgad Forest Region",
    bounds: tuple[int, int, int, int] = (-108, -99, 7, 15),
    expected_count: int = 59,
) -> list[CellSummary]:
    """Select and strictly validate the 59-cell exterior Vorndgad target."""

    min_x, max_x, min_y, max_y = bounds
    selected = [
        cell
        for cell in cells
        if not cell.is_interior
        and cell.grid is not None
        and cell.region is not None
        and cell.region.casefold() == region_name.casefold()
        and min_x <= cell.grid[0] <= max_x
        and min_y <= cell.grid[1] <= max_y
    ]
    selected.sort(key=lambda cell: (cell.grid[1], cell.grid[0]))
    unique_grids = {cell.grid for cell in selected}
    if len(selected) != expected_count or len(unique_grids) != expected_count:
        raise ValueError(
            f"expected {expected_count} unique exterior {region_name!r} cells in {bounds}, "
            f"found {len(selected)} rows/{len(unique_grids)} unique grids"
        )
    return selected


@dataclass(frozen=True)
class ScreeningDecision:
    include: bool
    screen_class: str
    reason: str


def _nature_edge_reason(ref: CellReference, definition: ObjectDefinition) -> str | None:
    """Return a review reason for nature-adjacent entrance edge cases."""

    text = " ".join(
        value.casefold()
        for value in (ref.object_id or "", definition.model or "")
    ).replace("_", " ").replace("\\", " ").replace("/", " ")
    has_entrance = any(token in text for token in ("door", "entrance", "gate", "barrow"))
    has_nature = any(token in text for token in ("stump", "tree", "rock", "cliff", "cave"))
    if has_entrance and has_nature:
        return "nature-adjacent entrance/door candidate; review before scatter use"
    if "stump" in text:
        return "tree-stump nature edge case; review before scatter use"
    return None


def screen_reference(
    ref: CellReference,
    definition: ObjectDefinition | None,
    *,
    is_interior: bool = False,
) -> ScreeningDecision:
    """Apply the explicit scatter-content screening policy.

    Screening is deliberately conservative.  Only flora, rocks, and terrain
    landscape definitions are included.  Unresolved definitions, structures,
    doors, interiors, clutter, and review-list edge cases are excluded and
    remain visible in the audit trail.
    """

    if is_interior:
        return ScreeningDecision(False, "interior", "interior CELL excluded from exterior scatter")
    if definition is None or not definition.model:
        return ScreeningDecision(False, "unresolved_definition", "object definition/model not resolved in two-pass join")

    source_category = definition.classification.category
    edge_reason = _nature_edge_reason(ref, definition)
    if edge_reason is not None:
        return ScreeningDecision(False, "review", edge_reason)

    if source_category in SCATTER_CATEGORIES:
        return ScreeningDecision(True, "included", "flora/rocks/terrain landscape scatter content")

    return ScreeningDecision(
        False,
        SCREEN_CLASS_BY_CATEGORY.get(source_category, "other"),
        f"category {source_category!r} is not scatter content",
    )


def required_scatter_meshes(
    cells: Iterable[CellSummary],
    definitions: Mapping[str, ObjectDefinition],
) -> list[str]:
    """Return sorted unique mesh paths that must have real bbox measurements."""

    meshes: set[str] = set()
    for cell in cells:
        for ref in cell.references:
            definition = definitions.get((ref.object_id or "").casefold())
            decision = screen_reference(ref, definition, is_interior=cell.is_interior)
            if decision.include and definition is not None:
                meshes.add(definition.model)
    return sorted(meshes, key=lambda value: (value.casefold(), value))


def _bbox_row(cache: Mapping[str, Any], mesh: str) -> Mapping[str, Any]:
    rows = cache.get("meshes") if isinstance(cache, Mapping) else None
    if not isinstance(rows, Mapping):
        raise ValueError("mesh bbox cache has no meshes mapping")
    key = normalize_mesh_key(mesh)
    for stored_key, value in rows.items():
        if normalize_mesh_key(str(stored_key)) == key:
            if not isinstance(value, Mapping):
                break
            if value.get("status") != "ok" or value.get("fallback"):
                raise ValueError(f"bbox cache entry is not a real measurement for {mesh!r}")
            return value
    raise ValueError(f"bbox cache is missing real measurement for {mesh!r}")


def bbox_info(cache_row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a cache row and derive deterministic size classes."""

    raw_bbox = cache_row.get("bbox_local_game_units")
    if not isinstance(raw_bbox, Mapping):
        raise ValueError("bbox cache row has no bbox_local_game_units")
    minimum = raw_bbox.get("min")
    maximum = raw_bbox.get("max")
    if not isinstance(minimum, Sequence) or not isinstance(maximum, Sequence) or len(minimum) != 3 or len(maximum) != 3:
        raise ValueError("bbox cache row must contain three-value min/max arrays")
    minimum_f = [float(value) for value in minimum]
    maximum_f = [float(value) for value in maximum]
    dimensions = [abs(maximum_f[index] - minimum_f[index]) for index in range(3)]
    volume = dimensions[0] * dimensions[1] * dimensions[2]
    largest = max(dimensions)
    if largest > GIANT_BBOX_THRESHOLD_GU:
        volume_class = "cliff_giant"
    elif largest > 1_000.0:
        volume_class = "large"
    elif largest > 250.0:
        volume_class = "medium"
    else:
        volume_class = "small"
    return {
        "status": str(cache_row.get("status", "ok")),
        "local_min_gu": _round_list(minimum_f),
        "local_max_gu": _round_list(maximum_f),
        "dimensions_gu": _round_list(dimensions),
        "max_dimension_gu": _round(largest),
        "volume_gu3": _round(volume),
        "volume_class": volume_class,
    }


def rotate_tes3_reference_point(
    point: Sequence[float], rotation: Sequence[float]
) -> tuple[float, float, float]:
    """Apply the authoritative OpenMW matrix to one raw TES3 point.

    The matrix is ``Rx(-rx) @ Ry(-ry) @ Rz(-rz)`` for column vectors, so the
    implementation applies the rightmost Z rotation first, then Y, then X.
    Keeping the sequence explicit makes this helper independent of Blender's
    Euler-property encoding and suitable for host-side bbox calculations.
    """

    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    rx, ry, rz = (float(rotation[0]), float(rotation[1]), float(rotation[2]))

    # Rz(-rz), applied first to the column vector.
    cos_z, sin_z = math.cos(-rz), math.sin(-rz)
    x_z = x * cos_z - y * sin_z
    y_z = x * sin_z + y * cos_z

    # Ry(-ry).
    cos_y, sin_y = math.cos(-ry), math.sin(-ry)
    x_y = cos_y * x_z + sin_y * z
    z_y = -sin_y * x_z + cos_y * z

    # Rx(-rx).
    cos_x, sin_x = math.cos(-rx), math.sin(-rx)
    y_x = cos_x * y_z - sin_x * z_y
    z_x = sin_x * y_z + cos_x * z_y
    return x_y, y_x, z_x


def _rotate_xyz(point: Sequence[float], rotation: Sequence[float]) -> tuple[float, float, float]:
    """Compatibility alias for the raw-TES3 rotation helper.

    Older callers used this private name.  It now deliberately means the
    authoritative OpenMW reference transform rather than the former positive
    Blender-Euler convention.
    """

    return rotate_tes3_reference_point(point, rotation)


def transformed_bbox(
    bbox: Mapping[str, Any],
    position: Sequence[float],
    rotation: Sequence[float] | None,
    scale: float | None,
) -> dict[str, list[float]]:
    """Transform all eight local bbox corners into a world AABB."""

    minimum = bbox.get("local_min_gu")
    maximum = bbox.get("local_max_gu")
    if not isinstance(minimum, Sequence) or not isinstance(maximum, Sequence):
        raise ValueError("normalized bbox lacks local min/max")
    rotation_values = tuple(float(value) for value in (rotation or (0.0, 0.0, 0.0)))
    scale_value = 1.0 if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError("reference scale must be finite")
    transformed: list[tuple[float, float, float]] = []
    for x in (float(minimum[0]), float(maximum[0])):
        for y in (float(minimum[1]), float(maximum[1])):
            for z in (float(minimum[2]), float(maximum[2])):
                rotated = rotate_tes3_reference_point(
                    (x * scale_value, y * scale_value, z * scale_value),
                    rotation_values,
                )
                transformed.append(tuple(rotated[index] + float(position[index]) for index in range(3)))
    return {
        "min": [_round(min(point[index] for point in transformed)) for index in range(3)],
        "max": [_round(max(point[index] for point in transformed)) for index in range(3)],
    }


def point_overlaps_bbox_xy(point_xy: Sequence[float], bbox: Mapping[str, Sequence[float]]) -> bool:
    minimum = bbox.get("min")
    maximum = bbox.get("max")
    if not isinstance(minimum, Sequence) or not isinstance(maximum, Sequence) or len(minimum) < 2 or len(maximum) < 2:
        raise ValueError("world bbox must contain min/max arrays")
    return (
        float(minimum[0]) <= float(point_xy[0]) <= float(maximum[0])
        and float(minimum[1]) <= float(point_xy[1]) <= float(maximum[1])
    )


def classify_stacker(
    z_offset_gu: float,
    point_xy: Sequence[float],
    parent_bbox: Mapping[str, Sequence[float]],
    *,
    threshold_gu: float = STACKER_Z_OFFSET_THRESHOLD_GU,
) -> bool:
    """Return true only for a high reference whose xy point is on a giant."""

    return float(z_offset_gu) > float(threshold_gu) and point_overlaps_bbox_xy(point_xy, parent_bbox)


def terrain_slope_deg(
    land_records: Mapping[tuple[int, int], LandRecord],
    position: Sequence[float],
    *,
    spacing_game_units: float = SLOPE_SAMPLE_SPACING_GU,
) -> float | None:
    """Compute max neighboring terrain gradient in degrees.

    The center and eight neighbors are sampled at a 128-GU grid (one source
    heightmap sample).  Each neighbor's absolute rise/run angle is calculated;
    the maximum is retained.  Missing neighbors are ignored so a target cell
    at the edge of a source LAND set can still be characterized, but a missing
    center or an entirely missing neighborhood returns ``None`` for the caller
    to reject as incomplete context.
    """

    if spacing_game_units <= 0 or not math.isfinite(float(spacing_game_units)):
        raise ValueError("slope spacing must be finite and positive")
    center_thu = height_at_game_position(land_records, position[:2])
    if center_thu is None:
        return None
    center_gu = float(center_thu) * THU_TO_GU
    maximum_slope: float | None = None
    for dx, dy in SLOPE_DIRECTIONS:
        neighbor = (
            float(position[0]) + dx * float(spacing_game_units),
            float(position[1]) + dy * float(spacing_game_units),
        )
        neighbor_thu = height_at_game_position(land_records, neighbor)
        if neighbor_thu is None:
            continue
        rise_gu = abs(float(neighbor_thu) * THU_TO_GU - center_gu)
        run_gu = math.hypot(dx * float(spacing_game_units), dy * float(spacing_game_units))
        slope = math.degrees(math.atan2(rise_gu, run_gu))
        maximum_slope = slope if maximum_slope is None else max(maximum_slope, slope)
    return maximum_slope


def texture_at_position(
    land_records: Mapping[tuple[int, int], LandRecord],
    ltex: Mapping[int, LandscapeTexture],
    position: Sequence[float],
) -> dict[str, Any]:
    """Resolve the owning LAND's VTEX tile to LTEX name/path.

    ``LandRecord.texture_index`` is deliberately retained as the raw VTEX
    value for auditability.  The shared ``texture_ltex_index`` accessor owns
    the OpenMW ``0 -> base`` / ``N -> N - 1`` conversion; this function then
    performs the strict nonzero lookup in the LTEX table supplied by the
    LAND-owning plugin.
    """

    if len(position) < 2:
        raise ValueError("reference position must have x and y")
    game_x, game_y = float(position[0]), float(position[1])
    cell_x = math.floor(game_x / CELL_SIZE_GAME_UNITS)
    cell_y = math.floor(game_y / CELL_SIZE_GAME_UNITS)
    record = land_records.get((cell_x, cell_y))
    if record is None or not record.has_textures:
        raise ValueError(f"LAND VTEX is missing at reference position {(game_x, game_y)}")
    local_x = min(CELL_SIZE_GAME_UNITS - 1e-9, max(0.0, game_x - cell_x * CELL_SIZE_GAME_UNITS))
    local_y = min(CELL_SIZE_GAME_UNITS - 1e-9, max(0.0, game_y - cell_y * CELL_SIZE_GAME_UNITS))
    tile_x = min(15, max(0, math.floor(local_x / (CELL_SIZE_GAME_UNITS / 16.0))))
    tile_y = min(15, max(0, math.floor(local_y / (CELL_SIZE_GAME_UNITS / 16.0))))
    raw_vtex = int(record.texture_index(tile_x, tile_y))
    index = record.texture_ltex_index(tile_x, tile_y)
    if index is None:
        return {
            "index": None,
            "raw_vtex": raw_vtex,
            "name": BASE_LAND_TEXTURE_NAME,
            "path": BASE_LAND_TEXTURE_PATH,
            "tile": [tile_x, tile_y],
        }
    texture = ltex.get(index)
    if texture is None:
        raise ValueError(
            f"raw VTEX value {raw_vtex} resolves to LTEX index {index}, but the "
            f"owning-plugin table has no definition at {(cell_x, cell_y)}"
        )
    return {
        "index": index,
        "raw_vtex": raw_vtex,
        "name": texture.record_id,
        "path": texture.file_name,
        "tile": [tile_x, tile_y],
    }


class WaterDistanceIndex:
    """Nearest-water query over a documented 128-GU terrain sample grid."""

    def __init__(
        self,
        points_game_units: Sequence[Sequence[float]],
        *,
        sample_spacing_gu: float = WATER_SAMPLE_SPACING_GU,
        threshold_thu: float = WATER_THRESHOLD_THU,
        window_cells: Sequence[int] | None = None,
        source: str = "fixture points",
        require_tree: bool = False,
    ) -> None:
        self.sample_spacing_gu = float(sample_spacing_gu)
        self.threshold_thu = float(threshold_thu)
        self.window_cells = list(window_cells) if window_cells is not None else None
        self.source = source
        self.points_game_units = tuple((float(row[0]), float(row[1])) for row in points_game_units)
        if not self.points_game_units:
            raise ValueError("water distance index needs at least one water sample")
        self._tree: Any = None
        try:
            from scipy.spatial import cKDTree  # type: ignore

            self._tree = cKDTree(self.points_game_units)
        except ImportError:
            if require_tree:
                raise RuntimeError("scipy.spatial.cKDTree is required for production water-distance analysis")

    @classmethod
    def from_world_context(
        cls,
        context: Any,
        cell_bounds: tuple[int, int, int, int],
        *,
        margin_cells: int = WATER_MARGIN_CELLS,
    ) -> "WaterDistanceIndex":
        """Build the index from composite RAW samples in an expanded window."""

        if margin_cells < 1:
            raise ValueError("water-distance window margin must be at least one cell")
        try:
            import numpy as np
            from scipy.spatial import cKDTree  # noqa: F401  # type: ignore
        except ImportError as exc:
            raise RuntimeError("numpy and scipy are required for composite water-distance analysis") from exc

        min_x, max_x, min_y, max_y = cell_bounds
        from .coords import CELL_POINTS, MAX_X, MAX_Y, MIN_X, MIN_Y

        window_min_x = max(MIN_X, min_x - margin_cells)
        window_max_x = min(MAX_X, max_x + margin_cells)
        window_min_y = max(MIN_Y, min_y - margin_cells)
        window_max_y = min(MAX_Y, max_y + margin_cells)
        px0 = (window_min_x - MIN_X) * CELL_POINTS
        px1 = (window_max_x - MIN_X + 1) * CELL_POINTS
        py0 = (window_min_y - MIN_Y) * CELL_POINTS
        py1 = (window_max_y - MIN_Y + 1) * CELL_POINTS
        window = np.asarray(context.heightmap[py0:py1, px0:px1])
        water_y, water_x = np.nonzero(window <= WATER_THRESHOLD_THU)
        if len(water_x) == 0:
            raise ValueError("expanded composite window contains no terrain samples at or below water threshold")
        global_x = water_x.astype(np.float64) + px0
        global_y = water_y.astype(np.float64) + py0
        points = np.column_stack(
            (
                float(MIN_X) * CELL_SIZE_GAME_UNITS + global_x * WATER_SAMPLE_SPACING_GU,
                float(MIN_Y) * CELL_SIZE_GAME_UNITS + global_y * WATER_SAMPLE_SPACING_GU,
            )
        )
        return cls(
            points,
            sample_spacing_gu=WATER_SAMPLE_SPACING_GU,
            threshold_thu=WATER_THRESHOLD_THU,
            window_cells=[window_min_x, window_max_x, window_min_y, window_max_y],
            source=str(context.composite_raw),
            require_tree=True,
        )

    def distance_gu(self, position: Sequence[float]) -> float:
        if len(position) < 2:
            raise ValueError("water-distance query needs x and y")
        point = (float(position[0]), float(position[1]))
        if self._tree is not None:
            distance, _index = self._tree.query(point, k=1)
            return float(distance)
        return math.sqrt(min((point[0] - x) ** 2 + (point[1] - y) ** 2 for x, y in self.points_game_units))

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_thu": _round(self.threshold_thu),
            "sample_spacing_gu": _round(self.sample_spacing_gu),
            "sample_count": len(self.points_game_units),
            "window_cells": self.window_cells,
            "source": self.source,
            "distance_metric": "euclidean game-unit distance to nearest sample with terrain <= threshold",
        }


def validate_composite_samples(
    land_records: Mapping[tuple[int, int], LandRecord],
    context: Any,
    cells: Iterable[CellSummary],
    *,
    offsets: Sequence[tuple[int, int]] = ((0, 0), (16, 48), (32, 32), (48, 16), (63, 63)),
) -> dict[str, Any]:
    """Validate deterministic Sky LAND samples against the composite RAW."""

    reports = []
    for cell in sorted(cells, key=lambda item: (item.grid[1], item.grid[0])):  # type: ignore[index]
        if cell.grid not in land_records:
            raise ValueError(f"target cell {cell.grid} has no Sky LAND record")
        record = land_records[cell.grid]
        if not record.has_heights:
            raise ValueError(f"target cell {cell.grid} has no Sky VHGT terrain")
        reports.append(validate_land_samples(record, context, offsets))
    mismatch_count = sum(report.mismatches for report in reports)
    return {
        "status": "pass" if mismatch_count == 0 else "warning_mismatch",
        "cell_count": len(reports),
        "sample_count": sum(report.sample_count for report in reports),
        "mismatch_count": mismatch_count,
        "max_abs_delta_thu": max(report.max_abs_delta_thu for report in reports) if reports else 0,
        "max_abs_delta_gu": max(report.max_abs_delta_gu for report in reports) if reports else 0,
        "sample_offsets": [list(offset) for offset in offsets],
    }


@dataclass
class _Candidate:
    ref_id: str
    cell: CellSummary
    ref: CellReference
    definition: ObjectDefinition
    category: str
    bbox: dict[str, Any]
    world_bbox: dict[str, list[float]]
    terrain: dict[str, Any]
    source_ordinal: int
    parent_rock_id: str | None = None

    @property
    def position(self) -> tuple[float, float, float]:
        if self.ref.position is None:
            raise ValueError(f"scatter candidate {self.ref_id} has no position")
        return tuple(float(value) for value in self.ref.position)

    @property
    def mesh(self) -> str:
        return self.definition.model

    @property
    def on_rock(self) -> bool:
        return self.parent_rock_id is not None

    def to_dict(self) -> dict[str, Any]:
        position = self.position
        rotation = tuple(float(value) for value in (self.ref.rotation or (0.0, 0.0, 0.0)))
        scale = 1.0 if self.ref.scale is None else float(self.ref.scale)
        return {
            "ref_id": self.ref_id,
            "cell": [self.cell.grid[0], self.cell.grid[1]],  # type: ignore[index]
            "cell_name": self.cell.name or None,
            "object_id": self.ref.object_id,
            "record_type": self.definition.record_type,
            "mesh": self.mesh,
            "kit": self.definition.classification.kit,
            "category": self.category,
            "source_category": self.definition.classification.category,
            "temporary": bool(self.ref.temporary),
            "source_ordinal": self.source_ordinal,
            "position_gu": _round_list(position),
            "rotation_radians": _round_list(rotation),
            "rotation_mode": rotation_mode(self.ref.rotation),
            "scale": _round(scale),
            "terrain": self.terrain,
            "bbox": {
                **self.bbox,
                "world_aabb_gu": self.world_bbox,
            },
            "stacking": {
                "on_rock": self.on_rock,
                "parent_rock_id": self.parent_rock_id,
                "z_offset_threshold_gu": STACKER_Z_OFFSET_THRESHOLD_GU,
            },
        }


def _ref_id(cell: CellSummary, ordinal: int) -> str:
    if cell.grid is None:
        raise ValueError("target scatter cell has no exterior grid")
    return f"r_{cell.grid[0]}_{cell.grid[1]}_{ordinal:04d}"


def _terrain_context(
    ref: CellReference,
    land_records: Mapping[tuple[int, int], LandRecord],
    ltex: Mapping[int, LandscapeTexture],
    water_index: WaterDistanceIndex,
) -> dict[str, Any]:
    if ref.position is None:
        raise ValueError("included scatter reference has no DATA position")
    terrain_thu = height_at_game_position(land_records, ref.position[:2])
    if terrain_thu is None:
        raise ValueError(f"included scatter reference has no terrain at {ref.position[:2]}")
    terrain_gu = float(terrain_thu) * THU_TO_GU
    ref_z_gu = float(ref.position[2])
    slope = terrain_slope_deg(land_records, ref.position)
    if slope is None:
        raise ValueError(f"included scatter reference has no slope neighborhood at {ref.position[:2]}")
    texture = texture_at_position(land_records, ltex, ref.position)
    return {
        "terrain_z_thu": _round(float(terrain_thu)),
        "terrain_z_gu": _round(terrain_gu),
        "ref_z_gu": _round(ref_z_gu),
        "z_offset_gu": _round(ref_z_gu - terrain_gu),
        "slope_deg": _round(slope),
        "elevation_band": elevation_band(terrain_gu),
        "distance_to_water_gu": _round(water_index.distance_gu(ref.position)),
        "ground_texture": texture,
        "slope_definition": {
            "method": "maximum absolute rise/run angle over eight neighbors",
            "sample_spacing_gu": SLOPE_SAMPLE_SPACING_GU,
        },
    }


def _candidate_sort_key(candidate: _Candidate) -> tuple[int, int, int, str]:
    if candidate.cell.grid is None:
        return (0, 0, candidate.source_ordinal, candidate.ref_id)
    return (candidate.cell.grid[1], candidate.cell.grid[0], candidate.source_ordinal, candidate.ref_id)


def _choose_parent(
    candidate: _Candidate,
    giants: Sequence[_Candidate],
) -> str | None:
    point = candidate.position
    choices = [
        giant
        for giant in giants
        if giant.ref_id != candidate.ref_id
        and classify_stacker(candidate.terrain["z_offset_gu"], point[:2], giant.world_bbox)
    ]
    if not choices:
        return None
    choices.sort(
        key=lambda giant: (
            (point[0] - (float(giant.world_bbox["min"][0]) + float(giant.world_bbox["max"][0])) / 2.0) ** 2
            + (point[1] - (float(giant.world_bbox["min"][1]) + float(giant.world_bbox["max"][1])) / 2.0) ** 2,
            giant.ref_id,
        )
    )
    return choices[0].ref_id


def _adjacency(giant: _Candidate, candidates: Sequence[_Candidate]) -> dict[str, Any]:
    center = giant.position
    radius_squared = ADJACENCY_RADIUS_GU ** 2
    mesh_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    refs: list[str] = []
    for candidate in candidates:
        if candidate.ref_id == giant.ref_id:
            continue
        position = candidate.position
        distance_squared = (position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2
        if distance_squared <= radius_squared:
            mesh_counts[candidate.mesh] += 1
            category_counts[candidate.category] += 1
            refs.append(candidate.ref_id)
    refs.sort()
    mesh_rows = [
        {"mesh": mesh, "count": count}
        for mesh, count in sorted(mesh_counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))
    ]
    return {
        "radius_gu": ADJACENCY_RADIUS_GU,
        "reference_count": len(refs),
        "mesh_classes": mesh_rows,
        "category_counts": _counter_dict(category_counts),
        "ref_ids": refs,
    }


def _mode(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return sorted(counter, key=lambda key: (-counter[key], key.casefold(), key))[0]


def _species_stats(candidates: Sequence[_Candidate]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[normalize_mesh_key(candidate.mesh)].append(candidate)
    output: dict[str, dict[str, Any]] = {}
    for key in sorted(groups):
        rows = sorted(groups[key], key=_candidate_sort_key)
        first = min(rows, key=lambda item: (item.mesh.casefold(), item.mesh))
        scales = [1.0 if row.ref.scale is None else float(row.ref.scale) for row in rows]
        rotations = [tuple(float(value) for value in (row.ref.rotation or (0.0, 0.0, 0.0))) for row in rows]
        modes = Counter(rotation_mode(row.ref.rotation) for row in rows)
        textures: Counter[int | None] = Counter(
            row.terrain["ground_texture"]["index"] for row in rows
        )
        raw_textures: Counter[int] = Counter(
            int(row.terrain["ground_texture"]["raw_vtex"]) for row in rows
        )
        texture_details: dict[int | None, dict[str, Any]] = {}
        for row in rows:
            texture = row.terrain["ground_texture"]
            index = texture["index"]
            texture_details.setdefault(index, texture)
        elevations = [float(row.terrain["terrain_z_gu"]) for row in rows]
        slope_values = [float(row.terrain["slope_deg"]) for row in rows]
        z_offsets = [float(row.terrain["z_offset_gu"]) for row in rows]
        water_distances = [float(row.terrain["distance_to_water_gu"]) for row in rows]
        bands = Counter(str(row.terrain["elevation_band"]) for row in rows)
        categories = Counter(row.category for row in rows)
        kits = Counter(first.definition.classification.kit for first in rows)
        bbox_classes = Counter(str(row.bbox["volume_class"]) for row in rows)
        texture_rows = []
        for item in _sorted_counter(textures, limit=20):
            index = item["value"]
            texture_rows.append({**texture_details[index], "count": int(item["count"])})
        output[key] = {
            "mesh": first.mesh,
            "kit": _mode(kits),
            "category": _mode(categories),
            "count": len(rows),
            "cell_count": len({row.cell.grid for row in rows}),
            "category_counts": _counter_dict(categories),
            "kit_counts": _counter_dict(kits),
            "bbox_class_counts": _counter_dict(bbox_classes),
            "scale_distribution": distribution(scales),
            "rotation_distribution": {
                "mode_counts": _counter_dict(modes),
                "x_radians": distribution(rotation[0] for rotation in rotations),
                "y_radians": distribution(rotation[1] for rotation in rotations),
                "z_radians": distribution(rotation[2] for rotation in rotations),
                "z_only_fraction": _round(modes.get("z_only", 0) / len(rows)),
            },
            "rotation_slope_dependency": rotation_slope_dependency(rotations, slope_values),
            "z_offset_distribution_gu": distribution(z_offsets),
            "condition_envelopes": {
                "slope_deg": distribution(slope_values),
                "elevation_gu": distribution(elevations),
                "texture_index": distribution(textures.elements()),
                "raw_vtex": distribution(raw_textures.elements()),
                "water_distance_gu": distribution(water_distances),
            },
            "raw_vtex_counts": _counter_dict(raw_textures),
            "water_proximity": {
                "threshold_gu": 1_024.0,
                "within_1024_gu_count": sum(1 for value in water_distances if value <= 1_024.0),
                "within_1024_gu_fraction": _round(
                    sum(1 for value in water_distances if value <= 1_024.0) / len(rows)
                ),
            },
            "elevation_band_counts": _counter_dict(bands),
            "preferred_elevation_band": _mode(bands),
            "ground_textures": texture_rows,
            "stacked_count": sum(1 for row in rows if row.on_rock),
            "cliff_giant_count": sum(1 for row in rows if row.bbox["volume_class"] == "cliff_giant"),
            "settlement_counts": _counter_dict(
                Counter(row.cell.name for row in rows if row.cell.name)
            ),
        }
    return output


def _density_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def group_summary(group_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        count = len(group_rows)
        totals = {
            "flora_refs": sum(int(row["flora_refs"]) for row in group_rows),
            "rock_refs": sum(int(row["rock_refs"]) for row in group_rows),
            "terrain_landscape_refs": sum(int(row["terrain_landscape_refs"]) for row in group_rows),
            "scatter_refs": sum(int(row["scatter_refs"]) for row in group_rows),
        }
        means = {key: _round(value / count) if count else None for key, value in totals.items()}
        return {"cell_count": count, "totals": totals, "mean_per_cell": means}

    wilderness = [row for row in rows if row["classification"] == "wilderness"]
    settlements: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["classification"] == "settlement":
            settlements[str(row["settlement_name"])].append(row)
    return {
        "cells": list(rows),
        "wilderness": group_summary(wilderness),
        "settlements": {
            name: group_summary(settlements[name])
            for name in sorted(settlements, key=lambda value: (value.casefold(), value))
        },
        "region": group_summary(rows),
    }


def _rock_density_by_slope(candidates: Sequence[_Candidate]) -> dict[str, Any]:
    """Measure per-cell rock density by the rock ref's observed slope.

    The generator uses the p90 count as a hard cap for the corresponding slope
    band.  This keeps a flat target patch from receiving more rock refs than
    the measured wilderness cells while leaving steep terrain to its own
    measured band.
    """

    bins = ((0.0, 8.0), (8.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 45.0), (45.0, float("inf")))
    cells = sorted({candidate.cell.grid for candidate in candidates if candidate.cell.grid is not None})
    output: list[dict[str, Any]] = []
    for lower, upper in bins:
        by_cell: Counter[tuple[int, int]] = Counter()
        for candidate in candidates:
            if candidate.category != "rocks":
                continue
            slope = float(candidate.terrain["slope_deg"])
            if slope < lower or (math.isfinite(upper) and slope >= upper):
                continue
            if candidate.cell.grid is not None:
                by_cell[candidate.cell.grid] += 1
        counts = [by_cell[cell] for cell in cells]
        output.append(
            {
                "slope_min_deg": _round(lower),
                "slope_max_deg": _round(upper) if math.isfinite(upper) else None,
                "cell_count": len(cells),
                "cells_with_rock_refs": sum(1 for value in counts if value > 0),
                "rock_refs": sum(counts),
                "rock_refs_per_cell": distribution(counts),
                "hard_cap_refs_per_cell": int(math.ceil(float(distribution(counts).get("p90") or 0.0))),
            }
        )
    return {
        "bin_definition": "observed rock refs per source cell grouped by each ref's measured maximum-neighbor slope",
        "bins": output,
        "flat_cap_rule": "generator applies the 0-8 degree bin hard_cap_refs_per_cell to all rocks on target candidates in that band",
    }


def _screen_audit(
    total_refs: int,
    decisions: Sequence[
        tuple[CellSummary, CellReference, ScreeningDecision, str, ObjectDefinition | None]
    ],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    object_ids: dict[str, Counter[str]] = defaultdict(Counter)
    meshes: dict[str, Counter[str]] = defaultdict(Counter)
    review: list[dict[str, Any]] = []
    for cell, ref, decision, ref_id, definition in decisions:
        counts[decision.screen_class] += 1
        if decision.include:
            continue
        object_ids[decision.screen_class][ref.object_id or "<missing>"] += 1
        resolved_model = definition.model if definition is not None and definition.model else ref.model
        meshes[decision.screen_class][resolved_model or "<unresolved>"] += 1
        if decision.screen_class == "review":
            review.append(
                {
                    "ref_id": ref_id,
                    "cell": [cell.grid[0], cell.grid[1]],  # type: ignore[index]
                    "cell_name": cell.name or None,
                    "object_id": ref.object_id,
                    "model": resolved_model,
                    "record_type": definition.record_type if definition is not None else ref.record_type,
                    "reason": decision.reason,
                }
            )
    by_class: dict[str, Any] = {}
    for screen_class in sorted(counts):
        by_class[screen_class] = {
            "count": int(counts[screen_class]),
            "top_object_ids": _sorted_counter(object_ids[screen_class], limit=20),
            "top_models": _sorted_counter(meshes[screen_class], limit=20),
        }
    review.sort(key=lambda row: (row["cell"][1], row["cell"][0], row["ref_id"]))
    included = int(counts.get("included", 0))
    excluded = total_refs - included
    return {
        "total_source_references": total_refs,
        "included_references": included,
        "excluded_references": excluded,
        "counts_by_screen_class": {key: int(counts[key]) for key in sorted(counts)},
        "included_category_counts": {},
        "by_class": by_class,
        "review": review,
        "rules": {
            "included_categories": sorted(SCATTER_CATEGORIES),
            "excluded_categories": ["ruins/structures", "clutter", "interior", "door", "other", "unresolved"],
            "review_policy": "nature-adjacent entrance/door cases are excluded until user/lead review",
        },
    }


def analyze_vorndgad(
    cells: Sequence[CellSummary],
    definitions: Mapping[str, ObjectDefinition],
    land_records: Mapping[tuple[int, int], LandRecord],
    ltex: Mapping[int, LandscapeTexture],
    bbox_cache: Mapping[str, Any],
    water_index: WaterDistanceIndex,
    *,
    source_metadata: Mapping[str, Any] | None = None,
    composite_validation: Mapping[str, Any] | None = None,
    seed: int = 20260801,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze target cells and return scatter and cliff documents."""

    if len(cells) != 59:
        raise ValueError(f"analysis expects 59 target cells, got {len(cells)}")
    sorted_cells = sorted(cells, key=lambda cell: (cell.grid[1], cell.grid[0]))  # type: ignore[index]
    decisions: list[
        tuple[CellSummary, CellReference, ScreeningDecision, str, ObjectDefinition | None]
    ] = []
    candidates: list[_Candidate] = []
    for cell in sorted_cells:
        for ordinal, ref in enumerate(cell.references):
            ref_id = _ref_id(cell, ordinal)
            definition = definitions.get((ref.object_id or "").casefold())
            decision = screen_reference(ref, definition, is_interior=cell.is_interior)
            decisions.append((cell, ref, decision, ref_id, definition))
            if not decision.include:
                continue
            if definition is None or not definition.model:
                raise ValueError(f"included reference {ref_id} has no resolved model")
            if ref.position is None:
                raise ValueError(f"included reference {ref_id} has no position")
            cache_row = _bbox_row(bbox_cache, definition.model)
            bbox = bbox_info(cache_row)
            rotation = ref.rotation or (0.0, 0.0, 0.0)
            scale = 1.0 if ref.scale is None else float(ref.scale)
            world_bbox = transformed_bbox(bbox, ref.position, rotation, scale)
            terrain = _terrain_context(ref, land_records, ltex, water_index)
            candidates.append(
                _Candidate(
                    ref_id=ref_id,
                    cell=cell,
                    ref=ref,
                    definition=definition,
                    category=OUTPUT_CATEGORY.get(definition.classification.category, definition.classification.category),
                    bbox=bbox,
                    world_bbox=world_bbox,
                    terrain=terrain,
                    source_ordinal=ordinal,
                )
            )

    if not candidates:
        raise ValueError("screening produced no scatter candidates")

    giants = [
        candidate
        for candidate in candidates
        if candidate.category == "rocks" and candidate.bbox["volume_class"] == "cliff_giant"
    ]
    giants.sort(key=_candidate_sort_key)
    for candidate in sorted(candidates, key=_candidate_sort_key):
        candidate.parent_rock_id = _choose_parent(candidate, giants)

    scatter_rows = [candidate.to_dict() for candidate in sorted(candidates, key=_candidate_sort_key)]
    cell_counts: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        if candidate.cell.grid is None:
            raise ValueError(f"candidate {candidate.ref_id} has no grid")
        cell_counts[candidate.cell.grid][candidate.category] += 1
    density_rows: list[dict[str, Any]] = []
    for cell in sorted_cells:
        if cell.grid is None:
            raise ValueError("target cell has no grid")
        counts = cell_counts[cell.grid]
        flora = int(counts.get("flora", 0))
        rocks = int(counts.get("rocks", 0))
        terrain = int(counts.get("terrain-landscape", 0))
        settlement_name = cell.name.strip() if cell.name and cell.name.strip() else None
        density_rows.append(
            {
                "cell": [cell.grid[0], cell.grid[1]],
                "cell_name": cell.name or None,
                "settlement_name": settlement_name,
                "classification": "settlement" if settlement_name else "wilderness",
                "flora_refs": flora,
                "rock_refs": rocks,
                "terrain_landscape_refs": terrain,
                "scatter_refs": flora + rocks + terrain,
            }
        )

    audit = _screen_audit(len(decisions), decisions)
    audit["included_category_counts"] = _counter_dict(Counter(candidate.category for candidate in candidates))
    species = _species_stats(candidates)
    source_meta = dict(source_metadata or {})
    density_document = _density_summary(density_rows)
    density_document["rock_density_by_slope"] = _rock_density_by_slope(candidates)
    scatter_document: dict[str, Any] = {
        "schema_version": 1,
        "tool": "procgen.scatter_analysis",
        "tool_version": "1.0",
        "seed": int(seed),
        "determinism": "no random draws; sorted inputs and linear-interpolation percentiles",
        "scope": {
            "region": "Vorndgad Forest Region",
            "bounds_cells": [-108, -99, 7, 15],
            "exterior_cell_count": len(sorted_cells),
            "named_settlement_count": len({cell.name for cell in sorted_cells if cell.name}),
            "named_settlements": sorted({cell.name for cell in sorted_cells if cell.name}, key=lambda value: (value.casefold(), value)),
        },
        "units": {
            "position": "game_units",
            "terrain_source": "THU",
            "terrain_game_units_per_thu": THU_TO_GU,
            "rotation": "radians",
            "bbox": "game_units",
        },
        "texture_semantics": {
            "tile_order": "serialized LAND VTEX is transposed into OpenMW row-major tile_y*16+tile_x order",
            "raw_vtex_zero": f"base texture {BASE_LAND_TEXTURE_PATH}; no LTEX lookup",
            "raw_vtex_nonzero": "raw VTEX N resolves to owning-plugin LTEX INTV index N-1",
            "ownership": "LTEX table belongs to the plugin that owns the winning LAND record; no merged global table",
            "missing_nonzero_ltex": "strict analysis error; OpenMW warns and falls back to the base texture",
            "ground_texture_index": "resolved LTEX index, or null for the base sentinel",
            "ground_texture_raw_vtex": "raw VTEX value retained for audit",
        },
        "thresholds": {
            "cliff_giant_any_bbox_dimension_gu": GIANT_BBOX_THRESHOLD_GU,
            "stacker_z_offset_gt_gu": STACKER_Z_OFFSET_THRESHOLD_GU,
            "cliff_adjacency_radius_gu": ADJACENCY_RADIUS_GU,
            "slope_neighbor_spacing_gu": SLOPE_SAMPLE_SPACING_GU,
        },
        "inputs": source_meta,
        "composite_terrain_validation": dict(composite_validation or {}),
        "screening": audit,
        "references_analyzed": len(scatter_rows),
        "refs": scatter_rows,
        "species_stats": species,
        "density": density_document,
    }

    giant_rows: list[dict[str, Any]] = []
    for giant in giants:
        stackers = [candidate for candidate in candidates if candidate.parent_rock_id == giant.ref_id]
        stackers.sort(key=_candidate_sort_key)
        adjacency = _adjacency(giant, candidates)
        giant_rows.append(
            {
                "ref_id": giant.ref_id,
                "cell": [giant.cell.grid[0], giant.cell.grid[1]],  # type: ignore[index]
                "cell_name": giant.cell.name or None,
                "object_id": giant.ref.object_id,
                "mesh": giant.mesh,
                "kit": giant.definition.classification.kit,
                "category": giant.category,
                "position_gu": _round_list(giant.position),
                "rotation_radians": _round_list(giant.ref.rotation or (0.0, 0.0, 0.0)),
                "scale": _round(1.0 if giant.ref.scale is None else float(giant.ref.scale)),
                "bbox": {**giant.bbox, "world_aabb_gu": giant.world_bbox},
                "terrain": giant.terrain,
                "stacker_count": len(stackers),
                "stackers": [
                    {
                        "ref_id": stacker.ref_id,
                        "object_id": stacker.ref.object_id,
                        "mesh": stacker.mesh,
                        "category": stacker.category,
                        "position_gu": _round_list(stacker.position),
                        "z_offset_gu": stacker.terrain["z_offset_gu"],
                        "on_rock": True,
                        "parent_rock_id": giant.ref_id,
                    }
                    for stacker in stackers
                ],
                "adjacency": adjacency,
            }
        )

    cliff_mesh_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in giant_rows:
        cliff_mesh_groups[normalize_mesh_key(str(row["mesh"]))].append(row)
    giant_mesh_summary: dict[str, Any] = {}
    for key in sorted(cliff_mesh_groups):
        rows = cliff_mesh_groups[key]
        first = min(rows, key=lambda row: (str(row["mesh"]).casefold(), str(row["mesh"])))
        giant_mesh_summary[key] = {
            "mesh": first["mesh"],
            "count": len(rows),
            "bbox_dimensions_gu": first["bbox"]["dimensions_gu"],
            "z_offset_gu": distribution(row["terrain"]["z_offset_gu"] for row in rows),
            "slope_deg": distribution(row["terrain"]["slope_deg"] for row in rows),
            "elevation_gu": distribution(row["terrain"]["terrain_z_gu"] for row in rows),
            "water_distance_gu": distribution(row["terrain"]["distance_to_water_gu"] for row in rows),
            "rotation_distribution": {
                "mode_counts": _counter_dict(
                    Counter(rotation_mode(row.get("rotation_radians")) for row in rows)
                ),
                "x_radians": distribution(
                    float(row.get("rotation_radians", [0.0, 0.0, 0.0])[0]) for row in rows
                ),
                "y_radians": distribution(
                    float(row.get("rotation_radians", [0.0, 0.0, 0.0])[1]) for row in rows
                ),
                "z_radians": distribution(
                    float(row.get("rotation_radians", [0.0, 0.0, 0.0])[2]) for row in rows
                ),
                "z_only_fraction": _round(
                    sum(
                        1
                        for row in rows
                        if rotation_mode(row.get("rotation_radians")) == "z_only"
                    )
                    / len(rows)
                ),
            },
            "rotation_slope_dependency": rotation_slope_dependency(
                [row.get("rotation_radians", [0.0, 0.0, 0.0]) for row in rows],
                [float(row["terrain"]["slope_deg"]) for row in rows],
            ),
            "elevation_band_counts": _counter_dict(Counter(row["terrain"]["elevation_band"] for row in rows)),
            "ground_textures": _sorted_counter(
                Counter(row["terrain"]["ground_texture"]["name"] for row in rows), limit=10
            ),
            "ground_texture_ltex_indices": _sorted_counter(
                Counter(row["terrain"]["ground_texture"]["index"] for row in rows), limit=10
            ),
            "ground_texture_raw_vtex": _sorted_counter(
                Counter(row["terrain"]["ground_texture"]["raw_vtex"] for row in rows), limit=10
            ),
            "stacker_count": sum(int(row["stacker_count"]) for row in rows),
        }

    cliff_document: dict[str, Any] = {
        "schema_version": 1,
        "tool": "procgen.scatter_analysis",
        "tool_version": "1.0",
        "texture_semantics": scatter_document["texture_semantics"],
        "scope": scatter_document["scope"],
        "thresholds": scatter_document["thresholds"],
        "definition": "cliff_giant iff an unscaled local bbox dimension is > 2,500 GU; stacker iff z_offset > 300 GU and xy lies in a giant world AABB",
        "giant_count": len(giant_rows),
        "stacker_count": sum(int(row["stacker_count"]) for row in giant_rows),
        "unique_giant_mesh_count": len(giant_mesh_summary),
        "giants": giant_rows,
        "giant_mesh_summary": giant_mesh_summary,
        "adjacency_definition": "included scatter references within 512 GU horizontal Euclidean distance; screened content is not adjacency content",
    }
    return scatter_document, cliff_document


def validate_analysis_documents(
    scatter_document: Mapping[str, Any],
    cliff_document: Mapping[str, Any],
) -> dict[str, Any]:
    """Perform strict post-generation acceptance checks."""

    scope = scatter_document.get("scope", {})
    screening = scatter_document.get("screening", {})
    refs = scatter_document.get("refs", [])
    species = scatter_document.get("species_stats", {})
    density = scatter_document.get("density", {})
    if scope.get("exterior_cell_count") != 59:
        raise ValueError("analysis document does not cover exactly 59 target cells")
    if not isinstance(refs, list) or not refs:
        raise ValueError("analysis document has no scatter refs")
    if int(screening.get("included_references", -1)) != len(refs):
        raise ValueError("screening included count does not equal refs list")
    if int(screening.get("total_source_references", -1)) != int(screening.get("included_references", 0)) + int(screening.get("excluded_references", 0)):
        raise ValueError("screen audit totals do not conserve source references")
    required_terrain = ("terrain_z_gu", "z_offset_gu", "slope_deg", "elevation_band", "distance_to_water_gu", "ground_texture")
    for row in refs:
        if not isinstance(row, Mapping) or any(field not in row.get("terrain", {}) for field in required_terrain):
            raise ValueError("scatter ref is missing full terrain context")
        ground_texture = row.get("terrain", {}).get("ground_texture", {})
        if not isinstance(ground_texture, Mapping) or any(
            field not in ground_texture for field in ("index", "raw_vtex", "name", "path", "tile")
        ):
            raise ValueError("scatter ref is missing raw/resolved ground texture audit fields")
        if row.get("category") in {"structures", "clutter", "interior", "door"}:
            raise ValueError("screened structure/clutter leaked into scatter refs")
    if not isinstance(species, Mapping) or not species:
        raise ValueError("species stats are empty")
    if not isinstance(density.get("cells"), list) or len(density["cells"]) != 59:
        raise ValueError("density stats do not contain all target cells")
    giants = cliff_document.get("giants", [])
    if int(cliff_document.get("giant_count", -1)) != len(giants):
        raise ValueError("cliff giant count mismatch")
    for giant in giants:
        if not giant.get("terrain") or "adjacency" not in giant or "stackers" not in giant:
            raise ValueError("cliff giant is missing terrain, stacker, or adjacency detail")
    return {
        "target_cells": 59,
        "scatter_refs": len(refs),
        "species_meshes": len(species),
        "cliff_giants": len(giants),
        "stackers": int(cliff_document.get("stacker_count", 0)),
        "screen_audit_conserves_refs": True,
        "all_scatter_refs_have_terrain": True,
        "composite_validation_status": scatter_document.get("composite_terrain_validation", {}).get("status", "not-run"),
    }


__all__ = [
    "ADJACENCY_RADIUS_GU",
    "ELEVATION_BANDS",
    "GIANT_BBOX_THRESHOLD_GU",
    "STACKER_Z_OFFSET_THRESHOLD_GU",
    "ScreeningDecision",
    "WaterDistanceIndex",
    "analyze_vorndgad",
    "bbox_info",
    "classify_stacker",
    "distribution",
    "elevation_band",
    "merge_object_definitions",
    "normalize_mesh_key",
    "point_overlaps_bbox_xy",
    "required_scatter_meshes",
    "rotation_slope_dependency",
    "rotation_mode",
    "screen_reference",
    "select_vorndgad_cells",
    "terrain_slope_deg",
    "texture_at_position",
    "transformed_bbox",
    "validate_analysis_documents",
    "validate_composite_samples",
]
