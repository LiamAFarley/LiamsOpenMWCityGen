"""Profile and repair local discontinuities in the immutable road mask.

Pipeline position
------------------
The repair stage consumes the exact source mask from :mod:`road_source` and
produces a separate repaired corridor plus an auditable bridge ledger::

    effective source mask
        -> one source skeleton + endpoint/gap profile
        -> deterministic candidate ledger
        -> accepted bridge mask + repaired corridor
        -> final skeleton/graph stage

The source mask is never mutated.  Endpoint pairs and endpoint-to-interior
corridor projections are discovered with deterministic KD-tree queries,
ordered by measured distance and coordinates, and accepted only when they
satisfy a data-derived gap threshold, outward-heading checks, target-corridor
normality (for T-junctions), and a bounded source-dilation locality corridor.
A hard pixel cap remains in force even if the measured distribution is
unexpectedly broad, so the repair cannot invent continent- or ocean-scale
links.  Each accepted dilated bridge enumerates every source component it
touches and unions those labels immediately; a bounded local raster window
keeps that accounting tractable on the full map.  A union-find makes candidate
ordering connectivity-aware, and the ledger records every candidate in both
families with explicit rejection reasons and stable provenance.

No random state is used.  NumPy arrays use image coordinates ``[y, x]`` while
serialized points use ``[x, y]``.  This module does not smooth vectors or
trace graph edges; those responsibilities belong to later pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line
from skimage.morphology import disk, skeletonize


Coord = tuple[int, int]  # x, y in source-canvas pixels


@dataclass(frozen=True)
class RepairSettings:
    """Fixed repair policy; selected threshold is recorded separately."""

    profile_radius_px: float = 64.0
    threshold_quantile: float = 0.90
    minimum_selected_gap_px: float = 6.0
    hard_max_gap_px: float = 32.0
    corridor_radius_px: int = 6
    minimum_heading_cosine: float = 0.10
    minimum_target_normal_cosine: float = 0.50
    bridge_radius_px: int = 1
    component_connectivity: int = 8


@dataclass
class RepairResult:
    """Arrays, profile, and ledger produced by :func:`repair_source_mask`."""

    source_mask: np.ndarray
    source_skeleton: np.ndarray
    bridge_mask: np.ndarray
    bridge_owner: np.ndarray
    repaired_mask: np.ndarray
    source_component_labels: np.ndarray
    repaired_component_labels: np.ndarray
    metadata: dict[str, Any]
    bridge_ledger: dict[str, Any]


_NEIGHBOUR_OFFSETS: tuple[Coord, ...] = tuple(
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dx, dy) != (0, 0)
)


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Validate and copy a source mask into a boolean topology array."""

    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"road mask must be 2-D, got {values.shape}")
    if values.size == 0:
        raise ValueError("road mask must not be empty")
    return np.ascontiguousarray(values > 0, dtype=bool)


def component_labels(mask: np.ndarray, *, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Return deterministic 8-neighbour component labels and count.

    ``scipy.ndimage.label`` numbers components by scan order.  The returned
    integer labels are useful internally; stable serialized IDs are assigned
    from the sorted minimum pixel of each component by
    :func:`component_rows`.
    """

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, count = ndimage.label(_as_binary_mask(mask), structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


def _component_order(labels: np.ndarray, count: int) -> list[int]:
    """Sort raw labels by their first occupied ``(y, x)`` pixel."""

    rows: list[tuple[int, int, int]] = []
    # ``labels == value`` over the full 15-million-pixel canvas for every
    # component is needlessly quadratic.  ``find_objects`` gives one small
    # bounding slice per component and keeps the full-map audit practical.
    slices = ndimage.find_objects(labels)
    for label_value in range(1, count + 1):
        region = slices[label_value - 1] if label_value - 1 < len(slices) else None
        if region is None:
            continue
        sub = labels[region]
        ys, xs = np.nonzero(sub == label_value)
        if len(xs):
            rows.append((int(region[0].start + ys.min()), int(region[1].start + xs.min()), label_value))
    rows.sort()
    return [label_value for _y, _x, label_value in rows]


def component_rows(mask: np.ndarray, labels: np.ndarray | None = None) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Describe components with valid area/bounding-box accounting.

    The explicit ``area_px`` and ``bbox_area_px`` fields make it impossible to
    accidentally report a component area larger than its bounding box, a bug
    that can otherwise hide an image-axis mix-up in audit output.
    """

    values = _as_binary_mask(mask)
    if labels is None:
        labels, count = component_labels(values)
    else:
        labels = np.asarray(labels)
        count = int(labels.max())
        if labels.shape != values.shape:
            raise ValueError("component labels do not match mask shape")
    rows: list[dict[str, Any]] = []
    id_by_label: dict[int, str] = {}
    slices = ndimage.find_objects(labels)
    for index, label_value in enumerate(_component_order(labels, count), start=1):
        region = slices[label_value - 1] if label_value - 1 < len(slices) else None
        if region is None:
            raise AssertionError(f"component label {label_value} has no bounding slice")
        sub = labels[region]
        ys, xs = np.nonzero(sub == label_value)
        min_x = int(region[1].start + xs.min())
        max_x = int(region[1].start + xs.max())
        min_y = int(region[0].start + ys.min())
        max_y = int(region[0].start + ys.max())
        area = int(len(xs))
        bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        if area > bbox_area:
            raise AssertionError(f"component area {area} exceeds bbox area {bbox_area}")
        component_id = f"road_component_{index:06d}"
        id_by_label[int(label_value)] = component_id
        rows.append(
            {
                "component_id": component_id,
                "raw_label": int(label_value),
                "area_px": area,
                "bbox_px": {
                    "min": [min_x, min_y],
                    "max": [max_x, max_y],
                    "width": max_x - min_x + 1,
                    "height": max_y - min_y + 1,
                    "area": bbox_area,
                },
            }
        )
    return rows, id_by_label


def _pixel_neighbours(point: Coord, occupied: set[Coord]) -> list[Coord]:
    """Return sorted occupied 8-neighbours for a pixel coordinate."""

    x, y = point
    result = [(x + dx, y + dy) for dx, dy in _NEIGHBOUR_OFFSETS if (x + dx, y + dy) in occupied]
    return sorted(result, key=lambda item: (item[1], item[0]))


def _endpoint_records(skeleton: np.ndarray, labels: np.ndarray, id_by_label: Mapping[int, str]) -> list[dict[str, Any]]:
    """Extract degree-one skeleton endpoints with their inward headings."""

    ys, xs = np.nonzero(skeleton)
    occupied = {(int(x), int(y)) for y, x in zip(ys, xs)}
    records: list[dict[str, Any]] = []
    for x, y in sorted(occupied, key=lambda item: (item[1], item[0])):
        neighbours = _pixel_neighbours((x, y), occupied)
        if len(neighbours) != 1:
            continue
        component_label = int(labels[y, x])
        neighbour = neighbours[0]
        records.append(
            {
                "endpoint_index": len(records),
                "pixel": [x, y],
                "component_label": component_label,
                "component_id": id_by_label.get(component_label, f"raw_label_{component_label}"),
                "inward_neighbour": [neighbour[0], neighbour[1]],
            }
        )
    return records


def _skeleton_neighbour_map(skeleton: np.ndarray) -> dict[Coord, list[Coord]]:
    """Build one sorted neighbour list per skeleton pixel for corridor queries."""

    ys, xs = np.nonzero(skeleton)
    occupied = {(int(x), int(y)) for y, x in zip(ys, xs)}
    return {
        point: _pixel_neighbours(point, occupied)
        for point in sorted(occupied, key=lambda item: (item[1], item[0]))
    }


def _target_normal_cosine(
    target: Coord,
    source_endpoint: Coord,
    neighbours: Sequence[Coord],
) -> float:
    """Measure perpendicular approach to an interior skeleton corridor.

    For a degree-two target the two neighbouring skeleton pixels define the
    local tangent.  A valid T-junction approaches its corridor approximately
    along the tangent's normal.  At a junction, every incident neighbour pair
    is tested and the best local segment is retained, allowing attachment to
    an existing branch without a coordinate-specific exception.
    """

    if len(neighbours) < 2:
        return -1.0
    vx = float(source_endpoint[0] - target[0])
    vy = float(source_endpoint[1] - target[1])
    approach_norm = math.hypot(vx, vy)
    if approach_norm == 0:
        return -1.0
    best = -1.0
    for first_index in range(len(neighbours)):
        for second_index in range(first_index + 1, len(neighbours)):
            ax = float(neighbours[second_index][0] - neighbours[first_index][0])
            ay = float(neighbours[second_index][1] - neighbours[first_index][1])
            tangent_norm = math.hypot(ax, ay)
            if tangent_norm == 0:
                continue
            # Absolute value accepts either side of the corridor normal while
            # still rejecting a nearly parallel approach.
            normal_cosine = abs(vx * (-ay) + vy * ax) / (approach_norm * tangent_norm)
            best = max(best, float(normal_cosine))
    return best


def _corridor_candidates(
    skeleton: np.ndarray,
    labels: np.ndarray,
    endpoints: Sequence[Mapping[str, Any]],
    id_by_label: Mapping[int, str],
    *,
    profile_radius_px: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Find nearest interior/junction pixels of other components per endpoint.

    A KD-tree broad phase queries all nearby skeleton pixels.  Results are
    reduced to the nearest eligible target per *other component*, making the
    ledger finite and deterministic while preserving the nearest local
    projection needed for a T-junction.  Degree-one targets remain in the
    endpoint-to-endpoint family, so the two candidate families do not duplicate
    terminal semantics unnecessarily.
    """

    neighbours_by_pixel = _skeleton_neighbour_map(skeleton)
    target_points = np.asarray(list(neighbours_by_pixel), dtype=np.float64)
    if not len(target_points):
        return [], []
    tree = cKDTree(target_points)
    candidates: list[dict[str, Any]] = []
    nearest_distances: list[float] = []
    for endpoint in endpoints:
        endpoint_point: Coord = tuple(int(value) for value in endpoint["pixel"])
        endpoint_label = int(endpoint["component_label"])
        target_by_component: dict[int, tuple[float, Coord]] = {}
        nearby_indices = tree.query_ball_point(np.asarray(endpoint_point, dtype=np.float64), profile_radius_px)
        for target_index in nearby_indices:
            target_float = target_points[int(target_index)]
            target: Coord = (int(target_float[0]), int(target_float[1]))
            target_label = int(labels[target[1], target[0]])
            if not target_label or target_label == endpoint_label:
                continue
            target_neighbours = neighbours_by_pixel[target]
            if len(target_neighbours) < 2:
                continue
            distance = math.dist(endpoint_point, target)
            old = target_by_component.get(target_label)
            if old is None or (distance, target[1], target[0]) < (old[0], old[1][1], old[1][0]):
                target_by_component[target_label] = (distance, target)
        for target_label, (distance, target) in sorted(
            target_by_component.items(),
            key=lambda item: (item[1][0], item[1][1][1], item[1][1][0], item[0]),
        ):
            target_neighbours = neighbours_by_pixel[target]
            target_degree = len(target_neighbours)
            candidates.append(
                {
                    "candidate_type": "endpoint_to_corridor",
                    "endpoint_index": int(endpoint["endpoint_index"]),
                    "endpoint": [endpoint_point[0], endpoint_point[1]],
                    "component_a": str(endpoint["component_id"]),
                    "component_a_label": endpoint_label,
                    "inward_neighbour": list(endpoint["inward_neighbour"]),
                    "target_pixel": [target[0], target[1]],
                    "target_component": str(id_by_label[target_label]),
                    "component_b": str(id_by_label[target_label]),
                    "component_b_label": target_label,
                    "target_degree": target_degree,
                    "target_neighbours": [[item[0], item[1]] for item in target_neighbours],
                    "target_projection_method": "nearest_other_component_skeleton_pixel",
                    "distance_px": float(distance),
                }
            )
        if target_by_component:
            nearest_distances.append(float(min(value[0] for value in target_by_component.values())))
    candidates.sort(
        key=lambda row: (
            float(row["distance_px"]),
            int(row["endpoint"][1]),
            int(row["endpoint"][0]),
            int(row["target_pixel"][1]),
            int(row["target_pixel"][0]),
            str(row["target_component"]),
        )
    )
    return candidates, nearest_distances


def _line_pixels(start: Coord, end: Coord, shape: tuple[int, int]) -> list[Coord]:
    """Return a clipped Bresenham line in deterministic ``(x,y)`` order."""

    height, width = shape
    rr, cc = draw_line(start[1], start[0], end[1], end[0])
    return [
        (int(x), int(y))
        for y, x in zip(rr, cc)
        if 0 <= int(x) < width and 0 <= int(y) < height
    ]


def _bridge_raster_window(
    start: Coord,
    end: Coord,
    shape: tuple[int, int],
    radius_px: int,
) -> tuple[np.ndarray, int, int]:
    """Rasterize one bridge in a padded local window, returning its origin.

    Candidate bridges are short, but the source canvas is 15 million pixels.
    Keeping this raster local avoids a full-canvas dilation and scan for every
    accepted candidate while preserving the exact global pixels when the caller
    writes the window into the bridge mask.
    """

    line = _line_pixels(start, end, shape)
    if not line:
        return np.zeros((1, 1), dtype=bool), max(0, int(start[0])), max(0, int(start[1]))
    pad = int(radius_px) + 1  # one extra pixel models connectivity contact
    height, width = shape
    min_x = max(0, min(point[0] for point in line) - pad)
    max_x = min(width - 1, max(point[0] for point in line) + pad)
    min_y = max(0, min(point[1] for point in line) - pad)
    max_y = min(height - 1, max(point[1] for point in line) + pad)
    result = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=bool)
    local_x = np.asarray([point[0] - min_x for point in line], dtype=np.intp)
    local_y = np.asarray([point[1] - min_y for point in line], dtype=np.intp)
    result[local_y, local_x] = True
    if radius_px:
        result = ndimage.binary_dilation(result, structure=disk(radius_px))
    return result, min_x, min_y


def _touched_source_labels(
    bridge_raster: np.ndarray,
    source: np.ndarray,
    source_labels: np.ndarray,
    *,
    origin_x: int,
    origin_y: int,
    connectivity: int,
) -> list[int]:
    """Return every source component contacted by the actual dilated raster.

    The extra one-pixel dilation models the same 4/8-neighbour contact used by
    component labeling: a bridge need not overlap a source pixel to connect it
    when it lands diagonally or orthogonally adjacent.  This is the accounting
    set that must be unioned, not merely the candidate's two endpoint labels.
    """

    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    contact = ndimage.binary_dilation(bridge_raster, structure=structure)
    height, width = source.shape
    y0, x0 = max(0, int(origin_y)), max(0, int(origin_x))
    y1 = min(height, y0 + contact.shape[0])
    x1 = min(width, x0 + contact.shape[1])
    contact = contact[: y1 - y0, : x1 - x0]
    labels = np.unique(source_labels[y0:y1, x0:x1][contact & source[y0:y1, x0:x1]])
    return sorted(int(value) for value in labels if int(value) > 0)


def _heading_cosine(endpoint: Mapping[str, Any], target: Coord) -> float:
    """Compare the endpoint's outward tangent with the proposed continuation.

    The stored skeleton neighbour points *into* the existing component.  A
    bridge continues from the endpoint in the opposite direction, so the
    heading vector is ``endpoint - inward_neighbour`` rather than the raw
    neighbour vector.  Using the inward direction would systematically reject
    genuine gaps whose fragments face one another.
    """

    x, y = (int(value) for value in endpoint["pixel"])
    nx, ny = (int(value) for value in endpoint["inward_neighbour"])
    vx, vy = target[0] - x, target[1] - y
    hx, hy = x - nx, y - ny
    vnorm = math.hypot(vx, vy)
    hnorm = math.hypot(hx, hy)
    if vnorm == 0 or hnorm == 0:
        return -1.0
    return (vx * hx + vy * hy) / (vnorm * hnorm)


def _quantile(values: Sequence[float], quantile: float) -> float | None:
    """Return a stable NumPy percentile or ``None`` for an empty sequence."""

    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile * 100.0, method="linear"))


def _distance_statistics(values: Sequence[float]) -> dict[str, Any]:
    """Summarize measured endpoint distances without lossy rounding."""

    if not values:
        return {"count": 0, "min_px": None, "max_px": None, "quantiles_px": {}}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min_px": float(array.min()),
        "max_px": float(array.max()),
        "quantiles_px": {
            "p25": float(np.percentile(array, 25.0)),
            "p50": float(np.percentile(array, 50.0)),
            "p75": float(np.percentile(array, 75.0)),
            "p90": float(np.percentile(array, 90.0)),
            "p95": float(np.percentile(array, 95.0)),
        },
    }


def profile_source_mask(mask: np.ndarray, settings: RepairSettings | None = None) -> dict[str, Any]:
    """Skeletonize the source once and measure local cross-component gaps.

    The profile deliberately precedes threshold selection.  Distances are
    measured between all endpoint pairs within the bounded profiling radius,
    while ``nearest_cross_component_distances_px`` records one nearest value
    per endpoint for the data-derived quantile used by the repair rule.
    """

    policy = settings or RepairSettings()
    source = _as_binary_mask(mask)
    source_labels, source_count = component_labels(source, connectivity=policy.component_connectivity)
    source_rows, source_id_by_label = component_rows(source, source_labels)
    source_skeleton = skeletonize(source)
    endpoints = _endpoint_records(source_skeleton, source_labels, source_id_by_label)
    corridor_rows, corridor_nearest_distances = _corridor_candidates(
        source_skeleton,
        source_labels,
        endpoints,
        source_id_by_label,
        profile_radius_px=float(policy.profile_radius_px),
    )
    points = np.asarray([row["pixel"] for row in endpoints], dtype=np.float64)
    pair_rows: list[dict[str, Any]] = []
    nearest_by_endpoint: dict[int, float] = {}
    if len(points) >= 2:
        tree = cKDTree(points)
        raw_pairs = sorted(
            tree.query_pairs(float(policy.profile_radius_px), output_type="set"),
            key=lambda pair: (
                float(np.linalg.norm(points[pair[0]] - points[pair[1]])),
                pair[0],
                pair[1],
            ),
        )
        for first_index, second_index in raw_pairs:
            first = endpoints[first_index]
            second = endpoints[second_index]
            if first["component_label"] == second["component_label"]:
                continue
            first_point = tuple(int(value) for value in first["pixel"])
            second_point = tuple(int(value) for value in second["pixel"])
            distance = math.dist(first_point, second_point)
            nearest_by_endpoint[first_index] = min(nearest_by_endpoint.get(first_index, float("inf")), distance)
            nearest_by_endpoint[second_index] = min(nearest_by_endpoint.get(second_index, float("inf")), distance)
            pair_rows.append(
                {
                    "candidate_type": "endpoint_endpoint",
                    "endpoint_a_index": int(first_index),
                    "endpoint_b_index": int(second_index),
                    "endpoint_a": [first_point[0], first_point[1]],
                    "endpoint_b": [second_point[0], second_point[1]],
                    "component_a": str(first["component_id"]),
                    "component_b": str(second["component_id"]),
                    "component_a_label": int(first["component_label"]),
                    "component_b_label": int(second["component_label"]),
                    "distance_px": float(distance),
                }
            )
    nearest_values = list(nearest_by_endpoint.values())
    return {
        "source_component_count": int(source_count),
        "source_road_pixels": int(np.count_nonzero(source)),
        "skeleton_pixels": int(np.count_nonzero(source_skeleton)),
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "candidate_pair_count_within_profile_radius": len(pair_rows),
        "candidate_counts": {
            "endpoint_endpoint": len(pair_rows),
            "endpoint_to_corridor": len(corridor_rows),
        },
        "endpoint_pair_distance_statistics_px": _distance_statistics(
            [float(row["distance_px"]) for row in pair_rows]
        ),
        "nearest_cross_component_distance_statistics_px": _distance_statistics(nearest_values),
        "nearest_cross_component_distances_px": [float(value) for value in sorted(nearest_values)],
        "candidate_pairs": pair_rows,
        "endpoint_to_corridor_candidates": corridor_rows,
        "endpoint_to_corridor_distance_statistics_px": _distance_statistics(corridor_nearest_distances),
        "nearest_endpoint_to_corridor_distances_px": [
            float(value) for value in sorted(corridor_nearest_distances)
        ],
        "profile_radius_px": float(policy.profile_radius_px),
        "component_connectivity": int(policy.component_connectivity),
        "_source_skeleton": source_skeleton,
        "_source_labels": source_labels,
        "_source_rows": source_rows,
        "_source_id_by_label": source_id_by_label,
    }


def _bridge_id(row: Mapping[str, Any]) -> str:
    """Create a content-derived ID that does not depend on candidate order."""

    if row.get("candidate_type") == "endpoint_to_corridor":
        basis = (
            f"endpoint_to_corridor|{row['component_a']}|{row['endpoint']}|"
            f"{row['component_b']}|{row['target_pixel']}|{float(row['distance_px']):.9f}"
        ).encode("utf-8")
    else:
        basis = (
            f"endpoint_endpoint|{row['component_a']}|{row['endpoint_a']}|{row['component_b']}|"
            f"{row['endpoint_b']}|{float(row['distance_px']):.9f}"
        ).encode("utf-8")
    return f"road_bridge_{hashlib.sha256(basis).hexdigest()[:16]}"


def _candidate_reason(
    candidate: Mapping[str, Any],
    *,
    selected_max_gap: float,
    settings: RepairSettings,
    source: np.ndarray,
    distance_to_source: np.ndarray,
) -> tuple[str | None, dict[str, Any]]:
    """Evaluate one endpoint pair or endpoint-to-corridor projection."""

    candidate_type = str(candidate["candidate_type"])
    if candidate_type == "endpoint_to_corridor":
        first: Coord = tuple(int(value) for value in candidate["endpoint"])
        second: Coord = tuple(int(value) for value in candidate["target_pixel"])
    else:
        first = tuple(int(value) for value in candidate["endpoint_a"])
        second = tuple(int(value) for value in candidate["endpoint_b"])
    line = _line_pixels(first, second, source.shape)
    if not line:
        return "empty_line", {"line_pixel_count": 0, "corridor_fraction": 0.0}
    line_y = np.asarray([point[1] for point in line], dtype=np.intp)
    line_x = np.asarray([point[0] for point in line], dtype=np.intp)
    nearest = distance_to_source[line_y, line_x]
    corridor_fraction = float(np.mean(nearest <= float(settings.corridor_radius_px)))
    measurements: dict[str, Any] = {
        "line_pixel_count": len(line),
        "line_pixels": [[point[0], point[1]] for point in line],
        "max_distance_to_source_px": float(np.max(nearest)),
        "mean_distance_to_source_px": float(np.mean(nearest)),
        "corridor_fraction": corridor_fraction,
    }
    if candidate_type == "endpoint_to_corridor":
        measurements["heading_cosine_endpoint"] = float(
            _heading_cosine(
                {"pixel": candidate["endpoint"], "inward_neighbour": candidate["inward_neighbour"]},
                second,
            )
        )
        measurements["target_normal_cosine"] = float(
            _target_normal_cosine(second, first, [tuple(item) for item in candidate["target_neighbours"]])
        )
    else:
        measurements["heading_cosine_a"] = float(
            _heading_cosine(
                {"pixel": candidate["endpoint_a"], "inward_neighbour": candidate["inward_neighbour_a"]},
                second,
            )
        )
        measurements["heading_cosine_b"] = float(
            _heading_cosine(
                {"pixel": candidate["endpoint_b"], "inward_neighbour": candidate["inward_neighbour_b"]},
                first,
            )
        )
    if float(candidate["distance_px"]) > selected_max_gap + 1e-9:
        return "outside_selected_gap_threshold", measurements
    if float(candidate["distance_px"]) > float(settings.hard_max_gap_px) + 1e-9:
        return "outside_hard_gap_cap", measurements
    if candidate_type == "endpoint_to_corridor":
        if measurements["heading_cosine_endpoint"] < float(settings.minimum_heading_cosine):
            return "endpoint_heading_misaligned", measurements
        if measurements["target_normal_cosine"] < float(settings.minimum_target_normal_cosine):
            return "target_corridor_not_approximately_perpendicular", measurements
    else:
        if measurements["heading_cosine_a"] < float(settings.minimum_heading_cosine):
            return "heading_a_misaligned", measurements
        if measurements["heading_cosine_b"] < float(settings.minimum_heading_cosine):
            return "heading_b_misaligned", measurements
    if measurements["max_distance_to_source_px"] > float(settings.corridor_radius_px) + 1e-9:
        return "line_exits_source_dilation_land_locality_gate", measurements
    return None, measurements


class _ComponentUnionFind:
    """Deterministic disjoint-set structure for union-aware bridge ordering."""

    def __init__(self, labels: Iterable[int]) -> None:
        self._parent = {int(label): int(label) for label in labels}

    def find(self, label: int) -> int:
        """Return the current representative with path compression."""

        label = int(label)
        parent = self._parent.setdefault(label, label)
        if parent != label:
            self._parent[label] = self.find(parent)
        return self._parent[label]

    def connected(self, first: int, second: int) -> bool:
        """Return whether two source components have already been joined."""

        return self.find(first) == self.find(second)

    def union(self, first: int, second: int) -> None:
        """Join two representatives using the lower label as root."""

        first_root, second_root = self.find(first), self.find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            self._parent[second_root] = first_root
        else:
            self._parent[first_root] = second_root


def repair_source_mask(mask: np.ndarray, settings: RepairSettings | None = None) -> RepairResult:
    """Profile and deterministically bridge local source-mask gaps.

    Threshold selection is ``ceil(p90(combined nearest endpoint and corridor
    distances))`` bounded by ``minimum_selected_gap_px`` and the hard cap.  It
    is calculated from this input's actual source skeleton before any bridge is
    painted.  Every endpoint pair and nearest endpoint-to-corridor candidate in
    the profile radius receives a ledger row, even when it is rejected by the
    threshold, heading/normality/locality gates, or union-aware ordering.
    """

    policy = settings or RepairSettings()
    if policy.threshold_quantile <= 0 or policy.threshold_quantile > 1:
        raise ValueError("threshold_quantile must be in (0, 1]")
    if policy.minimum_selected_gap_px > policy.hard_max_gap_px:
        raise ValueError("minimum selected gap cannot exceed hard cap")
    if policy.corridor_radius_px < 1 or policy.bridge_radius_px < 0:
        raise ValueError("corridor and bridge radii must be non-negative, corridor >= 1")

    source = _as_binary_mask(mask)
    profile = profile_source_mask(source, policy)
    source_skeleton = np.asarray(profile.pop("_source_skeleton"), dtype=bool)
    source_labels = np.asarray(profile.pop("_source_labels"), dtype=np.int32)
    source_rows = profile.pop("_source_rows")
    source_id_by_label = profile.pop("_source_id_by_label")
    endpoint_nearest_values = [float(value) for value in profile["nearest_cross_component_distances_px"]]
    corridor_nearest_values = [
        float(value) for value in profile["nearest_endpoint_to_corridor_distances_px"]
    ]
    combined_nearest_values = endpoint_nearest_values + corridor_nearest_values
    p_quantile = _quantile(combined_nearest_values, policy.threshold_quantile)
    if p_quantile is None:
        selected_max_gap = float(policy.minimum_selected_gap_px)
        threshold_basis = "minimum_selected_gap_no_cross_component_nearby_targets"
    else:
        selected_max_gap = float(
            min(policy.hard_max_gap_px, max(policy.minimum_selected_gap_px, math.ceil(p_quantile)))
        )
        threshold_basis = (
            f"ceil(p{int(policy.threshold_quantile * 100)}_combined_endpoint_and_corridor_distance)"
        )

    distance_to_source = ndimage.distance_transform_edt(~source)
    endpoint_endpoint_rows = list(profile["candidate_pairs"])
    corridor_rows = list(profile["endpoint_to_corridor_candidates"])
    candidate_rows = endpoint_endpoint_rows + corridor_rows
    endpoints = profile["endpoints"]
    for row in endpoint_endpoint_rows:
        first = endpoints[int(row["endpoint_a_index"])]
        second = endpoints[int(row["endpoint_b_index"])]
        row["inward_neighbour_a"] = list(first["inward_neighbour"])
        row["inward_neighbour_b"] = list(second["inward_neighbour"])

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    bridge_mask = np.zeros(source.shape, dtype=bool)
    bridge_owner = np.full(source.shape, -1, dtype=np.int32)
    union_find = _ComponentUnionFind(row["raw_label"] for row in source_rows)
    candidate_order = sorted(
        range(len(candidate_rows)),
        key=lambda index: (
            float(candidate_rows[index]["distance_px"]),
            0 if candidate_rows[index]["candidate_type"] == "endpoint_to_corridor" else 1,
            int(candidate_rows[index].get("endpoint", candidate_rows[index].get("endpoint_a"))[1]),
            int(candidate_rows[index].get("endpoint", candidate_rows[index].get("endpoint_a"))[0]),
            int(candidate_rows[index].get("target_pixel", candidate_rows[index].get("endpoint_b"))[1]),
            int(candidate_rows[index].get("target_pixel", candidate_rows[index].get("endpoint_b"))[0]),
            str(candidate_rows[index]["component_b"]),
        ),
    )
    for candidate_index in candidate_order:
        row = candidate_rows[candidate_index]
        reason, measurements = _candidate_reason(
            row,
            selected_max_gap=selected_max_gap,
            settings=policy,
            source=source,
            distance_to_source=distance_to_source,
        )
        bridge = dict(row)
        bridge.update(measurements)
        bridge["candidate_index"] = int(candidate_index)
        first_label = int(row["component_a_label"])
        second_label = int(row["component_b_label"])
        bridge["union_representative_a_before"] = int(union_find.find(first_label))
        bridge["union_representative_b_before"] = int(union_find.find(second_label))
        bridge["touched_source_component_labels"] = []
        bridge["touched_source_component_ids"] = []
        bridge["touched_source_component_count"] = 0
        line_array: np.ndarray | None = None
        line_origin_x = 0
        line_origin_y = 0
        if reason is None:
            if bridge["candidate_type"] == "endpoint_to_corridor":
                first = tuple(int(value) for value in bridge["endpoint"])
                second = tuple(int(value) for value in bridge["target_pixel"])
            else:
                first = tuple(int(value) for value in bridge["endpoint_a"])
                second = tuple(int(value) for value in bridge["endpoint_b"])
            line_array, line_origin_x, line_origin_y = _bridge_raster_window(
                first,
                second,
                source.shape,
                policy.bridge_radius_px,
            )
            touched_labels = _touched_source_labels(
                line_array,
                source,
                source_labels,
                origin_x=line_origin_x,
                origin_y=line_origin_y,
                connectivity=policy.component_connectivity,
            )
            bridge["touched_source_component_labels"] = touched_labels
            bridge["touched_source_component_ids"] = [source_id_by_label[label] for label in touched_labels]
            bridge["touched_source_component_count"] = len(touched_labels)
            touched_roots = {union_find.find(label) for label in touched_labels}
            if not touched_labels:
                reason = "bridge_does_not_touch_source_component"
            elif len(touched_roots) == 1:
                reason = "components_already_connected"
        bridge["bridge_id"] = _bridge_id(bridge)
        bridge["status"] = "rejected" if reason else "accepted"
        bridge["rejection_reason"] = reason
        if reason is not None:
            # A complete ledger retains the measured line and gate values, but
            # does not duplicate an accepted bridge's raster provenance.
            rejected.append(bridge)
            continue

        assert line_array is not None
        y_slice = slice(line_origin_y, line_origin_y + line_array.shape[0])
        x_slice = slice(line_origin_x, line_origin_x + line_array.shape[1])
        bridge_mask[y_slice, x_slice] |= line_array
        accepted_index = len(accepted)
        owner_view = bridge_owner[y_slice, x_slice]
        owner_view[line_array & (owner_view < 0)] = accepted_index
        bridge["bridge_pixel_count"] = int(np.count_nonzero(line_array))
        bridge["source_component_labels"] = list(bridge["touched_source_component_labels"])
        bridge["source_component_ids"] = list(bridge["touched_source_component_ids"])
        touched_labels = [int(value) for value in bridge["touched_source_component_labels"]]
        for touched_label in touched_labels[1:]:
            union_find.union(touched_labels[0], touched_label)
        accepted.append(bridge)

    repaired = source | bridge_mask
    repaired_labels, repaired_count = component_labels(repaired, connectivity=policy.component_connectivity)
    repaired_rows, repaired_id_by_label = component_rows(repaired, repaired_labels)
    union_component_count = len({union_find.find(row["raw_label"]) for row in source_rows})
    if union_component_count != int(repaired_count):
        raise ValueError(
            "bridge union accounting does not reconcile with repaired mask: "
            f"union={union_component_count}, repaired={repaired_count}"
        )
    touched_histogram: dict[str, int] = {}
    for row in accepted:
        key = str(int(row["touched_source_component_count"]))
        touched_histogram[key] = touched_histogram.get(key, 0) + 1

    accepted_sorted = sorted(accepted, key=lambda row: str(row["bridge_id"]))
    # ``bridge_owner`` was filled in candidate-distance order while bridges
    # were accepted.  The canonical ledger is sorted by stable bridge ID, so
    # remap the sparse owner indices before handing provenance to the graph
    # stage; otherwise a pixel could name a different bridge after sorting.
    owner_remap = np.full(len(accepted), -1, dtype=np.int32)
    sorted_index_by_id = {str(row["bridge_id"]): index for index, row in enumerate(accepted_sorted)}
    for original_index, row in enumerate(accepted):
        owner_remap[original_index] = sorted_index_by_id[str(row["bridge_id"])]
    remapped_owner = np.full(bridge_owner.shape, -1, dtype=np.int32)
    active_owner = bridge_owner >= 0
    if np.any(active_owner):
        remapped_owner[active_owner] = owner_remap[bridge_owner[active_owner]]

    ledger = {
        "schema_version": 2,
        "settings": {
            "profile_radius_px": float(policy.profile_radius_px),
            "threshold_quantile": float(policy.threshold_quantile),
            "minimum_selected_gap_px": float(policy.minimum_selected_gap_px),
            "hard_max_gap_px": float(policy.hard_max_gap_px),
            "corridor_radius_px": int(policy.corridor_radius_px),
            "minimum_heading_cosine": float(policy.minimum_heading_cosine),
            "minimum_target_normal_cosine": float(policy.minimum_target_normal_cosine),
            "bridge_radius_px": int(policy.bridge_radius_px),
            "component_connectivity": int(policy.component_connectivity),
        },
        "selected_max_gap_px": float(selected_max_gap),
        "threshold_basis": threshold_basis,
        "profile": profile,
        "candidate_count": len(candidate_rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "union_component_count": int(union_component_count),
        "union_matches_repaired_component_count": bool(union_component_count == int(repaired_count)),
        "accepted_touched_source_component_count_histogram": dict(sorted(touched_histogram.items())),
        "candidate_counts": {
            "endpoint_endpoint": len(endpoint_endpoint_rows),
            "endpoint_to_corridor": len(corridor_rows),
        },
        "accepted_counts": {
            "endpoint_endpoint": sum(
                row["candidate_type"] == "endpoint_endpoint" for row in accepted
            ),
            "endpoint_to_corridor": sum(
                row["candidate_type"] == "endpoint_to_corridor" for row in accepted
            ),
        },
        "rejected_counts": {
            "endpoint_endpoint": sum(
                row["candidate_type"] == "endpoint_endpoint" for row in rejected
            ),
            "endpoint_to_corridor": sum(
                row["candidate_type"] == "endpoint_to_corridor" for row in rejected
            ),
        },
        "accepted": accepted_sorted,
        "rejected": sorted(rejected, key=lambda row: str(row["bridge_id"])),
    }
    metadata = {
        "source_component_count": len(source_rows),
        "repaired_component_count": int(repaired_count),
        "source_road_pixels": int(np.count_nonzero(source)),
        "bridge_pixels": int(np.count_nonzero(bridge_mask)),
        "repaired_road_pixels": int(np.count_nonzero(repaired)),
        "source_skeleton_pixels": int(np.count_nonzero(source_skeleton)),
        "selected_max_gap_px": float(selected_max_gap),
        "accepted_bridge_count": len(accepted),
        "rejected_candidate_count": len(rejected),
        "union_component_count": int(union_component_count),
        "union_matches_repaired_component_count": bool(union_component_count == int(repaired_count)),
        "accepted_touched_source_component_count_histogram": dict(sorted(touched_histogram.items())),
        "accepted_bridge_counts": dict(ledger["accepted_counts"]),
        "rejected_candidate_counts": dict(ledger["rejected_counts"]),
        "source_components": source_rows,
        "repaired_components": repaired_rows,
        "source_component_id_by_label": {str(key): value for key, value in source_id_by_label.items()},
        "repaired_component_id_by_label": {str(key): value for key, value in repaired_id_by_label.items()},
        "component_connectivity": int(policy.component_connectivity),
    }
    return RepairResult(
        source_mask=source.astype(np.uint8),
        source_skeleton=source_skeleton,
        bridge_mask=bridge_mask.astype(np.uint8),
        bridge_owner=remapped_owner,
        repaired_mask=repaired.astype(np.uint8),
        source_component_labels=source_labels,
        repaired_component_labels=repaired_labels,
        metadata=metadata,
        bridge_ledger=ledger,
    )


__all__ = [
    "RepairResult",
    "RepairSettings",
    "component_labels",
    "component_rows",
    "profile_source_mask",
    "repair_source_mask",
]
