"""Analytic T1.3 terrain edits, ordered composition, and THU legality gates.

Pipeline position
------------------
This module is the height-edit core between the stitched source field and
VNML/VTEX assembly.  It consumes pass-1 ``terrain_edits`` from D-PLAN and
pass-2 ``auto_pad`` / ``road_grade`` requests from T1.2/plan roads.  It emits a
float64 field, edit/provenance ledgers, and a single final THU quantization
result.  It never clips an illegal height delta.

Supported primitives
--------------------
``flatten_shelf`` uses a polygon plateau and smoothstep falloff; ``mound`` is a
center/radius bump; ``terrace`` applies ordered shelf polygons; ``cut`` lowers a
polyline corridor; ``auto_pad`` consumes the exact T1.2 hull/pad request; and
``road_grade`` performs a deterministic bounded least-squares grade fit inside
a road corridor.  All support geometry is checked against the 449x449 target
field and the immutable outer vertex border before any value is changed.

Invariants
----------
* Parameters are finite and links resolve when a link registry is supplied.
* Water-level crossings are rejected as ``unintentional_basin`` unless the
  edit is explicitly linked to an authorized dock/basin feature.
* Values outside declared support remain bit-exact copies of the input field.
* Final values are rounded to signed THU exactly once.  Every TES3 VHGT delta
  (including the row-start deltas used by ``tes3json.build_land``) must fit
  -127..127 THU.  A violation returns structured ``edit_too_steep`` evidence
  with a measured minimum falloff; it is never clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .cityscape_field import FIELD_SIDE, FIELD_SPACING_GU, TargetBlock, outer_border_mask, split_field


TARGET_SPAN_GU = (FIELD_SIDE - 1) * FIELD_SPACING_GU
THU_TO_GU = 8.0
MAX_DELTA_THU = 127
MAX_DELTA_GU = MAX_DELTA_THU * THU_TO_GU


class CityscapeEditError(ValueError):
    """Hard edit validation or terrain encoding failure."""

    def __init__(self, failure: Mapping[str, Any] | str) -> None:
        if isinstance(failure, str):
            payload = {"code": "edit_invalid", "message": failure}
        else:
            payload = dict(failure)
        self.failure = payload
        super().__init__(str(payload.get("message", payload)))


@dataclass(frozen=True)
class EditFailure:
    """Structured failure returned by the public validation helper."""

    code: str
    edit_id: str
    message: str
    path: str = ""
    measured: Any = None
    limit: Any = None
    minimum_required_falloff_gu: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "severity": "error",
            "code": self.code,
            "edit_id": self.edit_id,
            "path": self.path or f"$.terrain_edits[{self.edit_id}]",
            "message": self.message,
        }
        if self.measured is not None:
            result["measured"] = self.measured
        if self.limit is not None:
            result["limit"] = self.limit
        if self.minimum_required_falloff_gu is not None:
            result["minimum_required_falloff_gu"] = self.minimum_required_falloff_gu
        return result


@dataclass(frozen=True)
class EditApplication:
    """Result of one validated analytic edit."""

    edit_id: str
    kind: str
    linked_to: tuple[str, ...]
    values_gu: np.ndarray
    support_mask: np.ndarray
    changed_mask: np.ndarray
    ledger: Mapping[str, Any]


@dataclass(frozen=True)
class TerrainEditResult:
    """Pass output before/after the one final THU quantization."""

    values_gu: np.ndarray
    quantized_values_gu: np.ndarray
    edit_ledger: tuple[Mapping[str, Any], ...]
    vertex_ledger: Mapping[str, Any]
    final_encoding: Mapping[str, Any]
    source_unchanged: Mapping[str, Any]


def _failure(
    edit: Mapping[str, Any],
    code: str,
    message: str,
    *,
    path: str = "",
    measured: Any = None,
    limit: Any = None,
    minimum_required_falloff_gu: float | None = None,
) -> CityscapeEditError:
    return CityscapeEditError(EditFailure(
        code=code,
        edit_id=str(edit.get("edit_id", edit.get("lot_id", "<unnamed>"))),
        message=message,
        path=path,
        measured=measured,
        limit=limit,
        minimum_required_falloff_gu=minimum_required_falloff_gu,
    ).to_dict())


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-number point")
    return (_finite(value[0], f"{label}[0]"), _finite(value[1], f"{label}[1]"))


def _points(value: Any, label: str, *, minimum: int = 2) -> list[tuple[float, float]]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} points")
    result = [_point(item, f"{label}[{index}]") for index, item in enumerate(value)]
    return result


def _closed(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    result = list(points)
    if result[0] != result[-1]:
        result.append(result[0])
    return result


def _orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    def sign(value: float) -> int:
        return 1 if value > 1.0e-9 else -1 if value < -1.0e-9 else 0
    o1, o2, o3, o4 = (_orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b))
    if sign(o1) * sign(o2) < 0 and sign(o3) * sign(o4) < 0:
        return True
    return False


def _simple_polygon(points: Sequence[tuple[float, float]]) -> bool:
    ring = _closed(points)
    if len(set(ring[:-1])) < 3:
        return False
    area = sum(ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1] for i in range(len(ring) - 1))
    if abs(area) <= 1.0e-9:
        return False
    edges = [(ring[i], ring[i + 1]) for i in range(len(ring) - 1)]
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if j <= i or j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _segments_intersect(a, b, c, d):
                return False
    return True


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    ring = _closed(polygon)
    inside = False
    for index in range(len(ring) - 1):
        a, b = ring[index], ring[index + 1]
        if ((a[1] > point[1]) != (b[1] > point[1])):
            crossing = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if crossing > point[0]:
                inside = not inside
    return inside


def _point_segment_distance(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0.0 and dy == 0.0:
        return math.hypot(point[0] - a[0], point[1] - a[1])
    t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy))


def _polygon_boundary_distance(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> float:
    ring = _closed(polygon)
    return min(_point_segment_distance(point, ring[i], ring[i + 1]) for i in range(len(ring) - 1))


def _polyline_distance(point: tuple[float, float], polyline: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Return distance and along-line station for a polyline."""

    best_distance = float("inf")
    best_station = 0.0
    station = 0.0
    for index in range(len(polyline) - 1):
        a, b = polyline[index], polyline[index + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length <= 1.0e-12:
            continue
        t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (length * length)))
        closest = (a[0] + t * dx, a[1] + t * dy)
        distance = math.hypot(point[0] - closest[0], point[1] - closest[1])
        if distance < best_distance:
            best_distance = distance
            best_station = station + t * length
        station += length
    if not math.isfinite(best_distance):
        raise ValueError("polyline contains no nonzero segment")
    return best_distance, best_station


def _smoothstep(value: float) -> float:
    t = max(0.0, min(1.0, float(value)))
    return t * t * (3.0 - 2.0 * t)


def _blend_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]], falloff: float) -> float:
    if _point_in_polygon(point, polygon):
        return 1.0
    distance = _polygon_boundary_distance(point, polygon)
    if falloff <= 0.0 or distance >= falloff:
        return 0.0
    return _smoothstep(1.0 - distance / falloff)


def _blend_radial(point: tuple[float, float], center: tuple[float, float], radius: float, falloff: float) -> float:
    distance = math.hypot(point[0] - center[0], point[1] - center[1])
    if distance <= radius:
        return 1.0
    if falloff <= 0.0 or distance >= radius + falloff:
        return 0.0
    return _smoothstep(1.0 - (distance - radius) / falloff)


def _blend_corridor(point: tuple[float, float], line: Sequence[tuple[float, float]], half_width: float, falloff: float) -> tuple[float, float]:
    distance, station = _polyline_distance(point, line)
    if distance <= half_width:
        return 1.0, station
    if falloff <= 0.0 or distance >= half_width + falloff:
        return 0.0, station
    return _smoothstep(1.0 - (distance - half_width) / falloff), station


def _bounds(points: Sequence[tuple[float, float]], margin: float) -> tuple[float, float, float, float]:
    xs, ys = zip(*points)
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def _validate_support(
    edit: Mapping[str, Any],
    points: Sequence[tuple[float, float]],
    falloff: float,
) -> None:
    if falloff < 0.0 or not math.isfinite(falloff):
        raise _failure(edit, "invalid_falloff", "falloff must be finite and non-negative")
    min_x, min_y, max_x, max_y = _bounds(points, falloff)
    # The outermost target vertex is an immutable seam.  A support interval
    # touching it is rejected, even if the smoothstep would happen to be zero
    # at one particular floating-point sample.
    if min_x <= 0.0 or min_y <= 0.0 or max_x >= TARGET_SPAN_GU or max_y >= TARGET_SPAN_GU:
        raise _failure(
            edit,
            "out_of_bounds",
            "edit shape plus falloff must be strictly inside the target field and immutable border",
            measured=[min_x, min_y, max_x, max_y],
            limit=[0.0, 0.0, TARGET_SPAN_GU, TARGET_SPAN_GU],
        )


def _links(edit: Mapping[str, Any]) -> tuple[str, ...]:
    value = edit.get("linked_to")
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise _failure(edit, "unknown_link", "every edit needs one or more non-empty linked_to ids")
    return tuple(str(item) for item in value)


def _shape_and_blend(edit: Mapping[str, Any]) -> tuple[str, list[tuple[float, float]], float, Any]:
    kind = str(edit.get("kind", ""))
    falloff = _finite(edit.get("falloff_gu", 0.0), f"{kind}.falloff_gu")
    if kind in {"flatten_shelf", "auto_pad"}:
        polygon_value = edit.get("polygon", edit.get("pad_polygon"))
        polygon = _points(polygon_value, f"{kind}.polygon", minimum=3)
        if not _simple_polygon(polygon):
            raise ValueError(f"{kind} polygon is not a simple nonzero-area ring")
        return "polygon", polygon, falloff, lambda point: _blend_polygon(point, polygon, falloff)
    if kind == "terrace":
        shelves = edit.get("shelves")
        if not isinstance(shelves, list) or not shelves:
            raise ValueError("terrace requires a non-empty shelves list")
        shelf_polygons: list[list[tuple[float, float]]] = []
        for shelf_index, shelf in enumerate(shelves):
            if not isinstance(shelf, Mapping):
                raise ValueError(f"terrace shelf {shelf_index} is not an object")
            polygon = _points(shelf.get("polygon"), f"terrace.shelves[{shelf_index}].polygon", minimum=3)
            if not _simple_polygon(polygon):
                raise ValueError(f"terrace shelf {shelf_index} polygon is invalid")
            shelf_polygons.append(polygon)
        all_points = [point for polygon in shelf_polygons for point in polygon]
        return "polygon", all_points, falloff, lambda point: max(
            _blend_polygon(point, polygon, falloff) for polygon in shelf_polygons
        )
    if kind == "mound":
        center = _point(edit.get("center"), "mound.center")
        radius = _finite(edit.get("radius_gu"), "mound.radius_gu")
        if radius <= 0.0:
            raise ValueError("mound radius must be positive")
        bound_points = [
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] - radius),
            (center[0] + radius, center[1] + radius),
            (center[0] - radius, center[1] + radius),
        ]
        return "radial", bound_points, falloff, lambda point: _blend_radial(point, center, radius, falloff)
    if kind == "cut":
        line = _points(edit.get("polyline"), "cut.polyline", minimum=2)
        width = _finite(edit.get("width_gu"), "cut.width_gu")
        if width <= 0.0:
            raise ValueError("cut width must be positive")
        margin = width / 2.0
        expanded_points = [(x - margin, y - margin) for x, y in line] + [(x + margin, y + margin) for x, y in line]
        return "corridor", expanded_points, falloff, lambda point: _blend_corridor(point, line, margin, falloff)[0]
    if kind == "road_grade":
        line = _points(edit.get("polyline"), "road_grade.polyline", minimum=2)
        width = _finite(edit.get("width_gu"), "road_grade.width_gu")
        if width <= 0.0:
            raise ValueError("road_grade width must be positive")
        margin = width / 2.0
        expanded_points = [(x - margin, y - margin) for x, y in line] + [(x + margin, y + margin) for x, y in line]
        return "corridor", expanded_points, falloff, lambda point: _blend_corridor(point, line, margin, falloff)[0]
    raise ValueError(f"unknown terrain edit kind {kind!r}")


def _grid_coordinates() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(FIELD_SIDE, dtype=np.float64) * FIELD_SPACING_GU
    x, y = np.meshgrid(axis, axis)
    return x, y


def _support_mask(
    points: Sequence[tuple[float, float]],
    blend_function: Any,
) -> tuple[np.ndarray, np.ndarray]:
    x_grid, y_grid = _grid_coordinates()
    mask = np.zeros((FIELD_SIDE, FIELD_SIDE), dtype=bool)
    blend = np.zeros((FIELD_SIDE, FIELD_SIDE), dtype=np.float64)
    min_x = max(0, int(math.floor(min(point[0] for point in points) / FIELD_SPACING_GU)) - 2)
    max_x = min(FIELD_SIDE - 1, int(math.ceil(max(point[0] for point in points) / FIELD_SPACING_GU)) + 2)
    min_y = max(0, int(math.floor(min(point[1] for point in points) / FIELD_SPACING_GU)) - 2)
    max_y = min(FIELD_SIDE - 1, int(math.ceil(max(point[1] for point in points) / FIELD_SPACING_GU)) + 2)
    for iy in range(min_y, max_y + 1):
        for ix in range(min_x, max_x + 1):
            value = float(blend_function((float(x_grid[iy, ix]), float(y_grid[iy, ix]))))
            if value > 0.0:
                blend[iy, ix] = value
                mask[iy, ix] = True
    return mask, blend


def _quantize_field(values_gu: np.ndarray) -> np.ndarray:
    values = np.asarray(values_gu, dtype=np.float64)
    if not np.isfinite(values).all():
        raise CityscapeEditError("terrain field contains non-finite values")
    return (np.rint(values / THU_TO_GU) * THU_TO_GU).astype(np.float64)


def _encoded_delta_report(values_gu: np.ndarray, cells: Sequence[tuple[int, int]]) -> dict[str, Any]:
    quantized = _quantize_field(values_gu)
    per_cell: list[dict[str, Any]] = []
    worst: dict[str, Any] | None = None
    max_abs_thu = 0
    adjacent_max_abs_thu = 0
    for cell, grid in sorted(split_field(quantized, cells).items()):
        thu = np.rint(grid / THU_TO_GU).astype(np.int64)
        deltas: list[tuple[int, int, int]] = []
        row_start = int(thu[0, 0])
        for y in range(65):
            for x in range(65):
                previous = int(thu[0, 0]) if x == 0 and y == 0 else row_start if x == 0 else int(thu[y, x - 1])
                delta = int(thu[y, x] - previous)
                deltas.append((x, y, delta))
                if abs(delta) > max_abs_thu:
                    max_abs_thu = abs(delta)
                    worst = {
                        "cell": [cell[0], cell[1]],
                        "local_vertex": [x, y],
                        "delta_thu": delta,
                        "delta_gu": delta * 8,
                        "encoding_previous": [x - 1, y] if x else [0, y - 1],
                    }
            row_start = int(thu[y, 0])
        adjacent_x = int(np.max(np.abs(np.diff(thu, axis=1)))) if thu.shape[1] > 1 else 0
        adjacent_y = int(np.max(np.abs(np.diff(thu, axis=0)))) if thu.shape[0] > 1 else 0
        adjacent_max_abs_thu = max(adjacent_max_abs_thu, adjacent_x, adjacent_y)
        per_cell.append({
            "cell": [cell[0], cell[1]],
            "max_abs_encoded_delta_thu": max(abs(delta) for _, _, delta in deltas),
            "max_abs_adjacent_delta_thu": max(adjacent_x, adjacent_y),
        })
    return {
        "legal": max_abs_thu <= MAX_DELTA_THU and adjacent_max_abs_thu <= MAX_DELTA_THU,
        "max_abs_encoded_delta_thu": max_abs_thu,
        "max_abs_encoded_delta_gu": max_abs_thu * 8,
        "max_abs_adjacent_delta_thu": adjacent_max_abs_thu,
        "max_abs_adjacent_delta_gu": adjacent_max_abs_thu * 8,
        "limit_thu": MAX_DELTA_THU,
        "limit_gu": MAX_DELTA_GU,
        "worst_location": worst,
        "per_cell": per_cell,
    }


def _candidate_values(
    edit: Mapping[str, Any],
    current: np.ndarray,
    points: Sequence[tuple[float, float]],
    blend_function: Any,
    blend: np.ndarray,
) -> np.ndarray:
    result = np.array(current, dtype=np.float64, copy=True)
    kind = str(edit.get("kind"))
    if kind in {"flatten_shelf", "auto_pad"}:
        target = _finite(edit.get("target_height_gu"), f"{kind}.target_height_gu")
        result = current + blend * (target - current)
    elif kind == "mound":
        delta = _finite(edit.get("height_delta_gu"), "mound.height_delta_gu")
        result = current + blend * delta
    elif kind == "cut":
        depth = _finite(edit.get("depth_gu"), "cut.depth_gu")
        if depth < 0.0:
            raise ValueError("cut depth_gu must be non-negative")
        result = current - blend * depth
    elif kind == "terrace":
        shelves = edit.get("shelves")
        if not isinstance(shelves, list) or not shelves:
            raise ValueError("terrace requires a non-empty shelves list")
        result = np.array(current, dtype=np.float64, copy=True)
        # Terrace shelves are intentionally ordered.  Each shelf composes on
        # the result of the preceding shelf, so overlap has explicit provenance.
        for shelf_index, shelf in enumerate(shelves):
            if not isinstance(shelf, Mapping):
                raise ValueError(f"terrace shelf {shelf_index} is not an object")
            polygon = _points(shelf.get("polygon"), f"terrace.shelves[{shelf_index}].polygon", minimum=3)
            if not _simple_polygon(polygon):
                raise ValueError(f"terrace shelf {shelf_index} polygon is invalid")
            shelf_target = _finite(shelf.get("target_height_gu"), f"terrace.shelves[{shelf_index}].target_height_gu")
            shelf_falloff = _finite(shelf.get("falloff_gu", edit.get("falloff_gu", 0.0)), f"terrace.shelves[{shelf_index}].falloff_gu")
            shelf_mask, shelf_blend = _support_mask(polygon, lambda point, p=polygon, f=shelf_falloff: _blend_polygon(point, p, f))
            result = result + shelf_blend * (shelf_target - result)
        return result
    elif kind == "road_grade":
        # The grade target is produced by a deterministic least-squares fit to
        # the current field's vertex samples and then bounded by the declared
        # maximum grade.  The corridor blend is the only support edit.
        line = _points(edit.get("polyline"), "road_grade.polyline", minimum=2)
        width = _finite(edit.get("width_gu"), "road_grade.width_gu")
        max_grade_percent = _finite(edit.get("max_grade_percent", 10.0), "road_grade.max_grade_percent")
        if max_grade_percent < 0.0:
            raise ValueError("road grade percent must be non-negative")
        stations: list[float] = []
        heights: list[float] = []
        for iy in range(FIELD_SIDE):
            for ix in range(FIELD_SIDE):
                corridor_blend, station = _blend_corridor(
                    (ix * FIELD_SPACING_GU, iy * FIELD_SPACING_GU), line, width / 2.0, _finite(edit.get("falloff_gu", 0.0), "road_grade.falloff_gu")
                )
                if corridor_blend > 0.999999:
                    stations.append(station)
                    heights.append(float(current[iy, ix]))
        if len(stations) < 2:
            raise ValueError("road grade corridor has fewer than two interior field vertices")
        s = np.asarray(stations, dtype=np.float64)
        z = np.asarray(heights, dtype=np.float64)
        slope, intercept = np.polyfit(s, z, 1)
        slope_limit = max_grade_percent / 100.0
        slope = max(-slope_limit, min(slope_limit, float(slope)))
        target_line = intercept + slope * s
        # Keep the fitted intercept anchored to the mean current elevation;
        # this avoids a hidden vertical retarget while still enforcing grade.
        intercept = float(np.mean(z - slope * s))
        result = np.array(current, dtype=np.float64, copy=True)
        for iy in range(FIELD_SIDE):
            for ix in range(FIELD_SIDE):
                corridor_blend, station = _blend_corridor(
                    (ix * FIELD_SPACING_GU, iy * FIELD_SPACING_GU), line, width / 2.0, _finite(edit.get("falloff_gu", 0.0), "road_grade.falloff_gu")
                )
                if corridor_blend > 0.0:
                    result[iy, ix] = current[iy, ix] + corridor_blend * (intercept + slope * station - current[iy, ix])
        return result
    else:
        raise ValueError(f"unsupported edit kind {kind!r}")
    return result


def _validate_links(
    edit: Mapping[str, Any],
    linked_to: Sequence[str],
    *,
    known_links: set[str] | None,
) -> None:
    if known_links is not None:
        unknown = sorted(set(linked_to) - known_links)
        if unknown:
            raise _failure(edit, "unknown_link", f"edit links do not resolve: {unknown}", measured=unknown, limit=sorted(known_links))


def _water_authorized(edit: Mapping[str, Any], linked_to: Sequence[str], authorized_water_links: set[str] | None) -> bool:
    if authorized_water_links is not None:
        return bool(set(linked_to) & authorized_water_links)
    return any("dock" in item.lower() or "basin" in item.lower() for item in linked_to)


def _validate_auto_pad(edit: Mapping[str, Any]) -> None:
    lot_id = edit.get("lot_id")
    required_values = ("target_height_gu", "falloff_gu", "max_cut_fill_gu", "measured_max_cut_fill_gu")
    missing_values = [key for key in required_values if key not in edit]
    if missing_values:
        raise _failure(edit, "illegal_pad", f"T1.2 auto-pad request is missing {missing_values}")
    hull = _points(edit.get("footprint_hull_xy_plan_gu"), "auto_pad.footprint_hull_xy_plan_gu", minimum=3)
    pad = _points(edit.get("pad_polygon"), "auto_pad.pad_polygon", minimum=4)
    margin = _finite(edit.get("margin_gu"), "auto_pad.margin_gu")
    if abs(margin - 256.0) > 1.0e-9:
        raise _failure(edit, "illegal_pad", "T1.2 auto-pad margin must remain exactly 256 GU", measured=margin, limit=256.0)
    xs = [point[0] for point in hull]
    ys = [point[1] for point in hull]
    expected = [
        (min(xs) - margin, min(ys) - margin),
        (max(xs) + margin, min(ys) - margin),
        (max(xs) + margin, max(ys) + margin),
        (min(xs) - margin, max(ys) + margin),
    ]
    actual = pad[:-1] if pad[0] == pad[-1] else pad
    if len(actual) != 4 or any(math.hypot(actual[i][0] - expected[i][0], actual[i][1] - expected[i][1]) > 1.0e-6 for i in range(4)):
        raise _failure(edit, "illegal_pad", "auto-pad polygon is not the exact T1.2 256-GU envelope", measured=[list(item) for item in actual], limit=[list(item) for item in expected])
    falloff = _finite(edit.get("falloff_gu"), "auto_pad.falloff_gu")
    if falloff < 512.0:
        raise _failure(edit, "illegal_pad", "auto-pad falloff must be at least 512 GU", measured=falloff, limit=512.0)
    declared = edit.get("measured_max_cut_fill_gu")
    limit = edit.get("max_cut_fill_gu")
    if declared is not None and limit is not None and _finite(declared, "auto_pad.measured_max_cut_fill_gu") > _finite(limit, "auto_pad.max_cut_fill_gu") + 1.0e-9:
        raise _failure(edit, "illegal_pad", "T1.2 measured pad cut/fill exceeds its declared maximum", measured=declared, limit=limit)
    if lot_id is None or not isinstance(lot_id, str) or not lot_id:
        raise _failure(edit, "illegal_pad", "auto-pad request has no lot_id")


def validate_edit_request(
    edit: Mapping[str, Any],
    *,
    known_links: set[str] | None = None,
    authorized_water_links: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return structured validation failures without mutating a field."""

    try:
        linked = _links(edit)
        _validate_links(edit, linked, known_links=known_links)
        kind = str(edit.get("kind", ""))
        if kind == "auto_pad":
            _validate_auto_pad(edit)
        shape_kind, points, falloff, _ = _shape_and_blend(edit)
        _ = shape_kind
        _validate_support(edit, points, falloff)
        if kind in {"flatten_shelf", "mound", "terrace"} and _finite(edit.get("target_height_gu", 1.0), "target_height_gu") < 0.0 and not _water_authorized(edit, linked, authorized_water_links):
            raise _failure(edit, "unintentional_basin", "edit target is below z=0 without an authorized dock/basin link", measured=edit.get("target_height_gu"), limit=0.0)
    except CityscapeEditError as exc:
        return [dict(exc.failure)]
    except (TypeError, ValueError) as exc:
        return [_failure(edit, "edit_invalid", str(exc)).failure]
    return []


def _minimum_required_falloff_estimate(
    edit: Mapping[str, Any],
    current: np.ndarray,
    candidate: np.ndarray,
    *,
    cells: Sequence[tuple[int, int]],
) -> float | None:
    falloff = edit.get("falloff_gu")
    if not isinstance(falloff, (int, float)) or float(falloff) <= 0.0:
        return None
    if _encoded_delta_report(candidate, cells)["legal"]:
        return None
    # Re-run the same analytic edit with a trial falloff.  This is an evidence
    # search only: the production candidate is never replaced by the trial or
    # clipped to a legal value.  Starting at one vertex spacing avoids an
    # unrepresentative zero-width step, then doubling brackets the first legal
    # result before the deterministic binary search refines it.
    trial = copy.deepcopy(dict(edit))
    high = max(float(falloff), FIELD_SPACING_GU)

    def legal_at(value: float) -> bool:
        trial["falloff_gu"] = value
        try:
            _, points, _, blend_function = _shape_and_blend(trial)
            _validate_support(trial, points, value)
            _, blend = _support_mask(points, blend_function)
            possible = _candidate_values(trial, current, points, blend_function, blend)
            return bool(_encoded_delta_report(possible, cells)["legal"])
        except (CityscapeEditError, TypeError, ValueError):
            return False

    for _ in range(16):
        if legal_at(high):
            break
        high *= 2.0
        if high >= TARGET_SPAN_GU:
            return None
    else:
        return None
    low = 0.0
    for _ in range(36):
        middle = (low + high) / 2.0
        if legal_at(middle):
            high = middle
        else:
            low = middle
    return float(high)


def apply_edit(
    edit: Mapping[str, Any],
    current_values_gu: np.ndarray,
    *,
    block: TargetBlock,
    known_links: set[str] | None = None,
    authorized_water_links: set[str] | None = None,
    cells: Sequence[tuple[int, int]] | None = None,
) -> EditApplication:
    """Apply one primitive to a copy of the current field or raise structured failure."""

    selected_cells = tuple(cells or block.cells)
    failures = validate_edit_request(edit, known_links=known_links, authorized_water_links=authorized_water_links)
    if failures:
        raise CityscapeEditError(failures[0])
    kind = str(edit.get("kind"))
    linked_to = _links(edit)
    _, points, falloff, blend_function = _shape_and_blend(edit)
    support, blend = _support_mask(points, blend_function)
    if np.any(support & outer_border_mask()):
        raise _failure(edit, "immutable_border", "edit support would alter the immutable outer border")
    current = np.asarray(current_values_gu, dtype=np.float64)
    if current.shape != (FIELD_SIDE, FIELD_SIDE) or not np.isfinite(current).all():
        raise CityscapeEditError("edit input must be a finite 449x449 field")
    candidate = _candidate_values(edit, current, points, blend_function, blend)
    changed = support & (candidate != current)
    if kind == "road_grade" and edit.get("max_cut_fill_gu") is not None:
        cut_fill_limit = _finite(edit.get("max_cut_fill_gu"), "road_grade.max_cut_fill_gu")
        if cut_fill_limit < 0.0:
            raise _failure(edit, "road_grade_cut_fill_exceeded", "road grade max_cut_fill_gu must be non-negative")
        cut_fill = np.abs(candidate - current)
        measured_cut_fill = float(np.max(cut_fill[support])) if np.any(support) else 0.0
        if measured_cut_fill > cut_fill_limit + 1.0e-9:
            raise _failure(
                edit,
                "road_grade_cut_fill_exceeded",
                "road grade fit exceeds its declared cut/fill bound",
                measured=measured_cut_fill,
                limit=cut_fill_limit,
            )
    below = support & (candidate < 0.0)
    if np.any(below) and not _water_authorized(edit, linked_to, authorized_water_links):
        location = np.argwhere(below)[0]
        raise _failure(edit, "unintentional_basin", "edit creates terrain below z=0 without an authorized dock/basin link", measured={"minimum_height_gu": float(np.min(candidate[below])), "field_vertex": [int(location[1]), int(location[0])]}, limit=0.0)
    encoding = _encoded_delta_report(candidate, selected_cells)
    if not encoding["legal"]:
        minimum = _minimum_required_falloff_estimate(edit, current, candidate, cells=selected_cells)
        raise _failure(edit, "edit_too_steep", "analytic edit would exceed signed TES3 THU delta encoding; no clipping applied", measured=encoding, limit={"max_delta_thu": MAX_DELTA_THU}, minimum_required_falloff_gu=minimum)
    delta = candidate - current
    delta_values = delta[support]
    ledger = {
        "edit_id": str(edit.get("edit_id", edit.get("lot_id"))),
        "kind": kind,
        "linked_to": list(linked_to),
        "support_vertex_count": int(np.count_nonzero(support)),
        "changed_vertex_count": int(np.count_nonzero(changed)),
        "support_bounds_gu": [
            float(np.min(np.argwhere(support)[:, 1]) * FIELD_SPACING_GU),
            float(np.min(np.argwhere(support)[:, 0]) * FIELD_SPACING_GU),
            float(np.max(np.argwhere(support)[:, 1]) * FIELD_SPACING_GU),
            float(np.max(np.argwhere(support)[:, 0]) * FIELD_SPACING_GU),
        ],
        "vertex_delta_stats": {
            "min_gu": float(np.min(delta_values)) if delta_values.size else 0.0,
            "max_gu": float(np.max(delta_values)) if delta_values.size else 0.0,
            "mean_abs_gu": float(np.mean(np.abs(delta_values))) if delta_values.size else 0.0,
        },
        "encoding_before_final_quantization": encoding,
        "provenance": {
            "linked_to": list(linked_to),
            "support_geometry_not_widened": True,
            "outer_border_mutated": False,
        },
    }
    if kind == "auto_pad":
        ledger["auto_pad_contract"] = {
            "lot_id": str(edit["lot_id"]),
            "footprint_hull_xy_plan_gu": copy.deepcopy(edit["footprint_hull_xy_plan_gu"]),
            "pad_polygon": copy.deepcopy(edit["pad_polygon"]),
            "margin_gu": float(edit["margin_gu"]),
            "target_height_gu": float(edit["target_height_gu"]),
            "falloff_gu": float(edit["falloff_gu"]),
            "max_cut_fill_gu": float(edit["max_cut_fill_gu"]),
            "measured_max_cut_fill_gu": float(edit["measured_max_cut_fill_gu"]),
            "request_copied_without_retarget": True,
        }
    return EditApplication(
        edit_id=str(edit.get("edit_id", edit.get("lot_id"))),
        kind=kind,
        linked_to=linked_to,
        values_gu=candidate,
        support_mask=support,
        changed_mask=changed,
        ledger=ledger,
    )


def quantize_once(
    values_gu: np.ndarray,
    *,
    source_values_gu: np.ndarray,
    block: TargetBlock,
    cells: Sequence[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Round a final float64 field once and run the complete THU legality gate."""

    selected_cells = tuple(cells or block.cells)
    values = np.asarray(values_gu, dtype=np.float64)
    source = np.asarray(source_values_gu, dtype=np.float64)
    quantized = _quantize_field(values)
    border = outer_border_mask()
    if not np.array_equal(quantized[border], source[border]):
        location = np.argwhere(border & (quantized != source))[0]
        raise CityscapeEditError({
            "code": "immutable_border",
            "message": "final quantization changed an immutable outer border vertex",
            "field_vertex": [int(location[1]), int(location[0])],
        })
    encoding = _encoded_delta_report(quantized, selected_cells)
    if not encoding["legal"]:
        raise CityscapeEditError({
            "code": "edit_too_steep",
            "message": "final field exceeds signed TES3 THU delta encoding; no clipping applied",
            "measured": encoding,
            "limit": {"max_delta_thu": MAX_DELTA_THU},
        })
    error = quantized - values
    ledger = {
        **encoding,
        "quantized_once": True,
        "pre_post_error_gu": {
            "min": float(np.min(error)),
            "max": float(np.max(error)),
            "mean_abs": float(np.mean(np.abs(error))),
            "max_abs": float(np.max(np.abs(error))),
        },
        "nonzero_error_vertex_count": int(np.count_nonzero(error)),
    }
    return quantized, ledger


def compose_edits(
    *,
    block: TargetBlock,
    source_values_gu: np.ndarray,
    edits: Sequence[Mapping[str, Any]],
    known_links: set[str] | None = None,
    authorized_water_links: set[str] | None = None,
) -> TerrainEditResult:
    """Apply edits in explicit deterministic order and quantize only at the end."""

    source = np.asarray(source_values_gu, dtype=np.float64)
    if source.shape != (FIELD_SIDE, FIELD_SIDE) or not np.isfinite(source).all():
        raise CityscapeEditError("source field must be finite 449x449 float64")
    current = np.array(source, dtype=np.float64, copy=True)
    edit_rows: list[Mapping[str, Any]] = []
    support_union = np.zeros_like(current, dtype=bool)
    changed_union = np.zeros_like(current, dtype=bool)
    ordered = list(enumerate(edits))
    # List order is semantic for overlapping edits; the index is retained in
    # the ledger so repeated IDs cannot hide a composition-order change.
    for order, edit in ordered:
        if not isinstance(edit, Mapping):
            raise CityscapeEditError({"code": "edit_invalid", "message": f"edit {order} is not an object"})
        application = apply_edit(
            edit,
            current,
            block=block,
            known_links=known_links,
            authorized_water_links=authorized_water_links,
        )
        row = dict(application.ledger)
        row["order"] = order
        edit_rows.append(row)
        current = application.values_gu
        support_union |= application.support_mask
        changed_union |= application.changed_mask
    quantized, final_encoding = quantize_once(
        current,
        source_values_gu=source,
        block=block,
    )
    source_unchanged = {
        "source_vertex_count": int(source.size),
        "support_vertex_count": int(np.count_nonzero(support_union)),
        "changed_vertex_count": int(np.count_nonzero(changed_union)),
        "outside_declared_support_exact": bool(np.array_equal(current[~support_union], source[~support_union])),
        "outside_declared_support_count": int(np.count_nonzero(~support_union)),
        "outer_border_exact": bool(np.array_equal(current[outer_border_mask()], source[outer_border_mask()])),
        "final_quantized_outside_support_exact": bool(np.array_equal(quantized[~support_union], _quantize_field(source)[~support_union])),
    }
    if not source_unchanged["outside_declared_support_exact"] or not source_unchanged["outer_border_exact"]:
        raise CityscapeEditError({"code": "source_value_mutated_outside_support", "message": "edit composition changed an undeclared or immutable source vertex"})
    return TerrainEditResult(
        values_gu=current,
        quantized_values_gu=quantized,
        edit_ledger=tuple(edit_rows),
        vertex_ledger={
            "support_union_vertex_count": int(np.count_nonzero(support_union)),
            "changed_union_vertex_count": int(np.count_nonzero(changed_union)),
            "outside_support_exact": source_unchanged["outside_declared_support_exact"],
            "outer_border_exact": source_unchanged["outer_border_exact"],
        },
        final_encoding=final_encoding,
        source_unchanged=source_unchanged,
    )


__all__ = [
    "CityscapeEditError",
    "EditApplication",
    "EditFailure",
    "TerrainEditResult",
    "compose_edits",
    "quantize_once",
    "validate_edit_request",
]
