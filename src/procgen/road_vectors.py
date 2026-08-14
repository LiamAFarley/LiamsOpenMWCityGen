"""Fit bounded smooth vectors and convert road pixels to TES3 world GU.

Pipeline position
------------------
This module consumes the validated raw skeleton graph and repaired corridor::

    raw graph pixel chains
        -> scale-aware RDP simplification above one-pixel quantization noise
        -> two-control linear fit or centripetal Catmull--Rom sampling
        -> dense continuous metrics, corridor/self-intersection gates, and
           explicit fallback reasons
        -> TES3 absolute-GU node/edge geometry and metrics

The source canvas is north-up, whereas TES3 exterior ``+Y`` is northward from
the source origin's southern edge.  :class:`RoadTransform` therefore flips
the source pixel row around the canvas height while preserving pixel centers.
No random state or floating-point fitting solver is used; all curves are
constructed from local control points and then bounded against the repaired
raster corridor.  The audit compares fixed-angle high-frequency turns and
zigzag reversals before and after fitting; dense sample triplet counts are kept
separate because denser sampling naturally creates more local triplets.  Arrays
are indexed ``[y, x]``; serialized points are ``[x, y]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line

from .road_graph import SkeletonGraph


Point = tuple[float, float]
Pixel = tuple[int, int]


@dataclass(frozen=True)
class RoadTransform:
    """Source-pixel-center to absolute TES3-GU transform."""

    origin_cell_x: int = -254
    origin_cell_y: int = -130
    pixels_per_cell: int = 16
    pixel_size_gu: int = 512
    canvas_width_px: int = 4992
    canvas_height_px: int = 3040

    def __post_init__(self) -> None:
        if self.pixels_per_cell <= 0 or self.pixel_size_gu <= 0:
            raise ValueError("pixels_per_cell and pixel_size_gu must be positive")
        if self.canvas_width_px <= 0 or self.canvas_height_px <= 0:
            raise ValueError("canvas dimensions must be positive")

    @property
    def origin_gu(self) -> tuple[int, int]:
        """Absolute GU of the lower-left corner of the origin cell."""

        cell_size = self.pixels_per_cell * self.pixel_size_gu
        return self.origin_cell_x * cell_size, self.origin_cell_y * cell_size

    def pixel_to_gu(self, point: Sequence[float]) -> list[float]:
        """Map a source pixel center/continuous pixel coordinate to GU."""

        x, y = float(point[0]), float(point[1])
        origin_x, origin_y = self.origin_gu
        return [
            float(origin_x + (x + 0.5) * self.pixel_size_gu),
            float(origin_y + (self.canvas_height_px - y - 0.5) * self.pixel_size_gu),
        ]

    def gu_to_pixel(self, point: Sequence[float]) -> list[float]:
        """Invert :meth:`pixel_to_gu` to continuous source-pixel coordinates."""

        x, y = float(point[0]), float(point[1])
        origin_x, origin_y = self.origin_gu
        return [
            (x - origin_x) / self.pixel_size_gu - 0.5,
            self.canvas_height_px - (y - origin_y) / self.pixel_size_gu - 0.5,
        ]

    def metadata(self) -> dict[str, Any]:
        """Return canonical transform metadata and axis conventions."""

        return {
            "coordinate_system": "TES3 exterior world GU",
            "axes": "+X east, +Y north, source image rows north-to-south",
            "origin_cell": [self.origin_cell_x, self.origin_cell_y],
            "origin_gu_lower_left": list(self.origin_gu),
            "pixels_per_cell": self.pixels_per_cell,
            "pixel_size_gu": self.pixel_size_gu,
            "canvas_size_px": [self.canvas_width_px, self.canvas_height_px],
            "pixel_center_convention": "source [x,y] denotes center at x+0.5 pixels and y+0.5 pixels; +Y flips around canvas height",
        }


@dataclass(frozen=True)
class VectorSettings:
    """Deterministic smoothing and corridor-bound settings."""

    simplify_tolerance_px: float = 2.00
    sample_spacing_px: float = 0.75
    corridor_tolerance_px: float = 3.0
    minimum_curve_points: int = 3
    pixel_quantization_noise_px: float = 1.0
    scale_tolerance_multiplier: float = 2.0
    scale_tolerance_cap_px: float = 3.0
    high_frequency_turn_angle_radians: float = 0.18
    self_intersection_min_separation_px: float = 4.0
    metric_sample_spacing_px: float = 0.5


@dataclass
class VectorResult:
    """World-coordinate graph plus smoothing metrics."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    metrics: dict[str, Any]


def _as_points(chain: Sequence[Sequence[float]]) -> list[Point]:
    """Normalize serialized coordinate rows to finite float pairs."""

    points: list[Point] = []
    for row in chain:
        if len(row) != 2:
            raise ValueError("polyline points must contain exactly two coordinates")
        point = (float(row[0]), float(row[1]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("polyline contains a non-finite point")
        if not points or point != points[-1]:
            points.append(point)
    if not points:
        raise ValueError("polyline must contain at least one point")
    return points


def densify_polyline(points: Sequence[Point], spacing_px: float = 0.5) -> list[Point]:
    """Densify every finite polyline segment at a fixed continuous spacing.

    Graph chains can contain sparse node-anchor jumps even when the underlying
    skeleton route is straight.  Treating a two-anchor line as two isolated
    KD-tree samples produces a false Hausdorff miss, so metrics and raster
    coverage use this same continuous segment sampling for both raw and smooth
    polylines.  Segment endpoints are emitted exactly, preserving anchors.
    """

    if spacing_px <= 0:
        raise ValueError("polyline densification spacing must be positive")
    source = _as_points(points)
    if len(source) == 1:
        return source
    result: list[Point] = [source[0]]
    for first, second in zip(source, source[1:]):
        distance = math.dist(first, second)
        steps = max(1, int(math.ceil(distance / float(spacing_px))))
        for step in range(1, steps + 1):
            fraction = step / steps
            result.append(
                (
                    first[0] + (second[0] - first[0]) * fraction,
                    first[1] + (second[1] - first[1]) * fraction,
                )
            )
    result[-1] = source[-1]
    return result


def _perpendicular_distance(point: Point, start: Point, end: Point) -> float:
    """Distance from ``point`` to a finite line segment."""

    ax, ay = start
    bx, by = end
    px, py = point
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def simplify_polyline(points: Sequence[Point], tolerance: float, *, closed: bool = False) -> list[Point]:
    """Ramer--Douglas--Peucker simplification with preserved endpoint anchors."""

    if tolerance < 0:
        raise ValueError("simplification tolerance must be non-negative")
    source = _as_points(points)
    if closed and len(source) > 1 and source[0] == source[-1]:
        source = source[:-1]
    if len(source) <= 2:
        return source + ([source[0]] if closed and source else [])

    keep = {0, len(source) - 1}
    stack: list[tuple[int, int]] = [(0, len(source) - 1)]
    while stack:
        first, last = stack.pop()
        max_distance = -1.0
        max_index = -1
        for index in range(first + 1, last):
            distance = _perpendicular_distance(source[index], source[first], source[last])
            if distance > max_distance:
                max_distance, max_index = distance, index
        if max_distance > tolerance and max_index >= 0:
            keep.add(max_index)
            stack.append((first, max_index))
            stack.append((max_index, last))
    result = [source[index] for index in sorted(keep)]
    if closed:
        result.append(result[0])
    return result


def _centripetal_segment(p0: Point, p1: Point, p2: Point, p3: Point, u: float) -> Point:
    """Evaluate a centripetal Catmull--Rom segment at ``u`` in [0,1].

    Chord-length parameterization with ``alpha=0.5`` prevents the overshoot
    and cusps that uniform Catmull--Rom can introduce around uneven raster
    steps.  Epsilon-spaced knot times make endpoint padding deterministic even
    when a graph chain contains repeated control points.
    """

    alpha = 0.5

    def next_time(current: float, first: Point, second: Point) -> float:
        return current + max(math.dist(first, second), 1.0e-6) ** alpha

    t0 = 0.0
    t1 = next_time(t0, p0, p1)
    t2 = next_time(t1, p1, p2)
    t3 = next_time(t2, p2, p3)
    t = t1 + (t2 - t1) * float(u)

    def blend(first: Point, second: Point, first_time: float, second_time: float) -> Point:
        denominator = second_time - first_time
        if abs(denominator) < 1.0e-12:
            return second
        first_weight = (second_time - t) / denominator
        second_weight = (t - first_time) / denominator
        return (
            first_weight * first[0] + second_weight * second[0],
            first_weight * first[1] + second_weight * second[1],
        )

    a1 = blend(p0, p1, t0, t1)
    a2 = blend(p1, p2, t1, t2)
    a3 = blend(p2, p3, t2, t3)
    b1 = blend(a1, a2, t0, t2)
    b2 = blend(a2, a3, t1, t3)
    return blend(b1, b2, t1, t2)


def sample_catmull_rom(points: Sequence[Point], spacing_px: float, *, closed: bool = False) -> list[Point]:
    """Sample a centripetal Catmull--Rom curve with exact anchors."""

    if spacing_px <= 0:
        raise ValueError("curve sample spacing must be positive")
    source = _as_points(points)
    if closed:
        if source[0] == source[-1]:
            source = source[:-1]
        if len(source) < 3:
            return source + ([source[0]] if source else [])
        result: list[Point] = []
        count = len(source)
        for index, p1 in enumerate(source):
            p0 = source[(index - 1) % count]
            p2 = source[(index + 1) % count]
            p3 = source[(index + 2) % count]
            segment_length = math.dist(p1, p2)
            steps = max(1, int(math.ceil(segment_length / spacing_px)))
            for step in range(steps):
                result.append(_centripetal_segment(p0, p1, p2, p3, step / steps))
        result.append(result[0])
        return result
    if len(source) < 3:
        return source
    result = []
    for index in range(len(source) - 1):
        p0 = source[index - 1] if index > 0 else source[index]
        p1 = source[index]
        p2 = source[index + 1]
        p3 = source[index + 2] if index + 2 < len(source) else p2
        segment_length = math.dist(p1, p2)
        steps = max(1, int(math.ceil(segment_length / spacing_px)))
        for step in range(steps):
            result.append(_centripetal_segment(p0, p1, p2, p3, step / steps))
    result.append(source[-1])
    # Explicitly restore both anchors after floating arithmetic.
    result[0] = source[0]
    result[-1] = source[-1]
    return result


def _polyline_length(points: Sequence[Point]) -> float:
    """Return Euclidean length in the input coordinate units."""

    return float(sum(math.dist(first, second) for first, second in zip(points, points[1:])))


def _turning_metrics(points: Sequence[Point], *, high_frequency_angle: float = 0.18) -> dict[str, float | int]:
    """Compute absolute turns plus high-frequency zigzag indicators."""

    angles: list[float] = []
    signs: list[int] = []
    for first, middle, last in zip(points, points[1:], points[2:]):
        ax, ay = middle[0] - first[0], middle[1] - first[1]
        bx, by = last[0] - middle[0], last[1] - middle[1]
        cross = ax * by - ay * bx
        dot = ax * bx + ay * by
        if math.hypot(ax, ay) == 0 or math.hypot(bx, by) == 0:
            continue
        angle = abs(math.atan2(cross, dot))
        angles.append(angle)
        if angle >= high_frequency_angle:
            signs.append(1 if cross > 0 else -1 if cross < 0 else 0)
    signs = [sign for sign in signs if sign]
    zigzag_reversals = sum(first != second for first, second in zip(signs, signs[1:]))
    return {
        "turn_count": int(len(angles)),
        "total_absolute_turn_radians": float(sum(angles)),
        "maximum_turn_radians": float(max(angles, default=0.0)),
        "high_frequency_turn_count": int(sum(angle >= high_frequency_angle for angle in angles)),
        "high_frequency_zigzag_reversal_count": int(zigzag_reversals),
    }


def _scale_aware_simplification_tolerance(points: Sequence[Point], policy: VectorSettings) -> float:
    """Derive a raster-noise tolerance from the chain's measured pixel scale."""

    source = _as_points(points)
    steps = np.asarray([math.dist(first, second) for first, second in zip(source, source[1:])], dtype=np.float64)
    positive = steps[steps > 1.0e-9]
    measured_step = float(np.median(positive)) if len(positive) else 0.0
    evidence_scale = max(float(policy.pixel_quantization_noise_px), measured_step)
    evidence_tolerance = min(
        float(policy.scale_tolerance_cap_px),
        evidence_scale * float(policy.scale_tolerance_multiplier),
    )
    return float(max(float(policy.simplify_tolerance_px), evidence_tolerance))


def _self_induced_loop(
    points: Sequence[Point],
    *,
    closed: bool,
    minimum_path_separation_px: float = 4.0,
) -> bool:
    """Detect genuine self-crossing, not dense-sample pixel re-rounding."""

    seen: dict[tuple[int, int], tuple[int, float]] = {}
    travelled = 0.0
    for index, point in enumerate(points):
        cell = (int(round(point[0])), int(round(point[1])))
        previous = seen.get(cell)
        if previous is not None:
            previous_index, previous_distance = previous
            is_closed_endpoint = closed and previous_index == 0 and index == len(points) - 1
            if (
                not is_closed_endpoint
                and travelled - previous_distance >= float(minimum_path_separation_px)
            ):
                return True
        else:
            seen[cell] = (index, travelled)
        if index + 1 < len(points):
            travelled += math.dist(point, points[index + 1])
    return False


def _corridor_deviation(
    points: Sequence[Point],
    distance_to_source: np.ndarray,
    *,
    sample_spacing_px: float = 0.5,
) -> float:
    """Measure maximum sampled distance to the nearest repaired pixel."""

    height, width = distance_to_source.shape
    maximum = 0.0
    for x, y in densify_polyline(points, sample_spacing_px):
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < width and 0 <= iy < height):
            return float("inf")
        maximum = max(maximum, float(distance_to_source[iy, ix]))
    return maximum


def _rasterize_polyline(points: Sequence[Point], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a sampled curve into a clipped one-pixel centerline mask."""

    result = np.zeros(shape, dtype=bool)
    height, width = shape
    # Densification is explicit here even though draw_line covers each segment;
    # it keeps raster coverage on the same continuous-polyline contract used by
    # Hausdorff and corridor measurements.
    dense = densify_polyline(points, spacing_px=0.5)
    for first, second in zip(dense, dense[1:]):
        x0, y0 = int(round(first[0])), int(round(first[1]))
        x1, y1 = int(round(second[0])), int(round(second[1]))
        rr, cc = draw_line(y0, x0, y1, x1)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        result[rr[valid], cc[valid]] = True
    return result


def _hausdorff_metrics(
    raw_points: Sequence[Point],
    smooth_points: Sequence[Point],
    *,
    sample_spacing_px: float = 0.5,
) -> dict[str, float]:
    """Compute symmetric Hausdorff distances between continuous polylines.

    Both polylines are densified at the same fixed spacing before the KD-tree
    query.  Both directed distances are retained so a low one-sided value
    cannot hide a smoothed curve that skipped a raw bend.
    """

    raw_array = np.asarray(densify_polyline(raw_points, sample_spacing_px), dtype=np.float64)
    smooth_array = np.asarray(densify_polyline(smooth_points, sample_spacing_px), dtype=np.float64)
    if raw_array.ndim != 2 or smooth_array.ndim != 2 or len(raw_array) == 0 or len(smooth_array) == 0:
        raise ValueError("Hausdorff inputs must be non-empty point sets")
    raw_tree = cKDTree(raw_array)
    smooth_tree = cKDTree(smooth_array)
    smooth_to_raw = float(np.max(raw_tree.query(smooth_array, k=1)[0]))
    raw_to_smooth = float(np.max(smooth_tree.query(raw_array, k=1)[0]))
    return {
        "smooth_to_raw_hausdorff_px": smooth_to_raw,
        "raw_to_smooth_hausdorff_px": raw_to_smooth,
        "symmetric_hausdorff_px": max(smooth_to_raw, raw_to_smooth),
    }


def _edge_width_gu(raw_chain: Sequence[Pixel], distance_inside: np.ndarray, pixel_size_gu: int) -> dict[str, float]:
    """Estimate corridor width from the distance-transform radius."""

    height, width = distance_inside.shape
    radii = [
        float(distance_inside[y, x])
        for x, y in raw_chain
        if 0 <= x < width and 0 <= y < height and distance_inside[y, x] > 0
    ]
    if not radii:
        radii = [0.0]
    values = np.asarray(radii, dtype=np.float64) * 2.0 * float(pixel_size_gu)
    return {
        "estimated_width_gu": float(np.median(values)),
        "width_gu_p10": float(np.percentile(values, 10.0)),
        "width_gu_p90": float(np.percentile(values, 90.0)),
    }


def vectorize_graph(
    graph: SkeletonGraph,
    repaired_mask: np.ndarray,
    *,
    transform: RoadTransform | None = None,
    settings: VectorSettings | None = None,
) -> VectorResult:
    """Smooth each graph edge under corridor and anchor constraints."""

    policy = settings or VectorSettings()
    world = transform or RoadTransform(
        canvas_width_px=int(np.asarray(repaired_mask).shape[1]),
        canvas_height_px=int(np.asarray(repaired_mask).shape[0]),
    )
    repaired = np.asarray(repaired_mask) > 0
    if repaired.shape != graph.skeleton.shape:
        raise ValueError("repaired mask and graph skeleton have different shapes")
    distance_outside = ndimage.distance_transform_edt(~repaired)
    distance_inside = ndimage.distance_transform_edt(repaired)
    vector_edges: list[dict[str, Any]] = []
    max_deviation = 0.0
    endpoint_displacements: list[float] = []
    raster_all = np.zeros(repaired.shape, dtype=bool)
    fallback_counts: dict[str, int] = {}
    hausdorff_values: list[float] = []
    raw_turn_count_total = 0
    smooth_turn_count_total = 0
    raw_high_frequency_turn_total = 0
    smooth_high_frequency_turn_total = 0
    raw_zigzag_total = 0
    smooth_zigzag_total = 0
    method_counts: dict[str, int] = {}
    fallback_edge_ids: dict[str, list[str]] = {}
    for raw_edge in graph.edges:
        raw_chain = _as_points(raw_edge["raw_pixel_chain"])
        raw_pixels = [(int(round(point[0])), int(round(point[1]))) for point in raw_chain]
        closed = str(raw_edge["from"]) == str(raw_edge["to"])
        effective_tolerance = _scale_aware_simplification_tolerance(raw_chain, policy)
        simplified = simplify_polyline(raw_chain, effective_tolerance, closed=closed)
        method = "catmull_rom"
        fallback_reason: str | None = None
        if len(raw_chain) < policy.minimum_curve_points:
            smooth_pixels = raw_chain
            method = "raw_skeleton_fallback"
            fallback_reason = "too_short_for_curve"
        elif closed and len(simplified) - 1 < policy.minimum_curve_points:
            smooth_pixels = raw_chain
            method = "raw_skeleton_fallback"
            fallback_reason = "closed_loop_too_short_for_curve"
        elif not closed and len(simplified) < policy.minimum_curve_points:
            # Two RDP anchors are a valid fitted straight segment.  Densify it
            # before all continuous metrics so a nearly straight route does
            # not retain raster quantization merely because Catmull--Rom needs
            # three controls.  Only a genuinely degenerate one-point chain is
            # allowed to use the raw fallback here.
            if len(simplified) == 2:
                smooth_pixels = densify_polyline(simplified, policy.sample_spacing_px)
                method = "straight_line_simplified"
                deviation = _corridor_deviation(
                    smooth_pixels,
                    distance_outside,
                    sample_spacing_px=policy.metric_sample_spacing_px,
                )
                if deviation > policy.corridor_tolerance_px + 1e-9:
                    smooth_pixels = raw_chain
                    method = "raw_skeleton_fallback"
                    fallback_reason = "straight_line_corridor_deviation"
                elif _self_induced_loop(
                    smooth_pixels,
                    closed=False,
                    minimum_path_separation_px=policy.self_intersection_min_separation_px,
                ):
                    smooth_pixels = raw_chain
                    method = "raw_skeleton_fallback"
                    fallback_reason = "straight_line_self_induced_loop"
            else:
                smooth_pixels = raw_chain
                method = "raw_skeleton_fallback"
                fallback_reason = "too_short_for_curve"
        else:
            smooth_pixels = sample_catmull_rom(
                simplified,
                policy.sample_spacing_px,
                closed=closed,
            )
            deviation = _corridor_deviation(
                smooth_pixels,
                distance_outside,
                sample_spacing_px=policy.metric_sample_spacing_px,
            )
            if deviation > policy.corridor_tolerance_px + 1e-9:
                smooth_pixels = raw_chain
                method = "raw_skeleton_fallback"
                fallback_reason = "corridor_deviation"
            elif _self_induced_loop(
                smooth_pixels,
                closed=closed,
                minimum_path_separation_px=policy.self_intersection_min_separation_px,
            ):
                smooth_pixels = raw_chain
                method = "raw_skeleton_fallback"
                fallback_reason = "self_induced_loop"
        if not smooth_pixels:
            raise ValueError(f"edge {raw_edge['id']} produced an empty smoothed polyline")
        # Anchors are taken from the original graph chain, not from a rounded
        # or re-fitted point.  For loops both ends are the same anchor.
        smooth_pixels = list(smooth_pixels)
        smooth_pixels[0] = raw_chain[0]
        if len(smooth_pixels) > 1:
            smooth_pixels[-1] = raw_chain[-1]
        deviation = _corridor_deviation(
            smooth_pixels,
            distance_outside,
            sample_spacing_px=policy.metric_sample_spacing_px,
        )
        if not math.isfinite(deviation) or deviation > policy.corridor_tolerance_px + 1e-9:
            # A raw skeleton chain is the only permitted final fallback.  It
            # is itself inside the repaired mask, so a failure here is a
            # malformed graph/corridor rather than a reason to widen bounds.
            if method != "raw_skeleton_fallback":
                smooth_pixels = raw_chain
                method = "raw_skeleton_fallback"
                fallback_reason = "post_anchor_corridor_deviation"
                deviation = _corridor_deviation(
                    smooth_pixels,
                    distance_outside,
                    sample_spacing_px=policy.metric_sample_spacing_px,
                )
            if not math.isfinite(deviation) or deviation > policy.corridor_tolerance_px + 1e-9:
                raise ValueError(f"edge {raw_edge['id']} cannot be bounded by repaired corridor")
        smooth_raster = _rasterize_polyline(smooth_pixels, repaired.shape)
        raster_all |= smooth_raster
        method_counts[method] = method_counts.get(method, 0) + 1
        if fallback_reason:
            fallback_counts[fallback_reason] = fallback_counts.get(fallback_reason, 0) + 1
            fallback_edge_ids.setdefault(fallback_reason, []).append(str(raw_edge["id"]))
        raw_gu = [world.pixel_to_gu(point) for point in raw_chain]
        smooth_gu = [world.pixel_to_gu(point) for point in smooth_pixels]
        endpoint_displacement = max(math.dist(raw_gu[0], smooth_gu[0]), math.dist(raw_gu[-1], smooth_gu[-1]))
        endpoint_displacements.append(float(endpoint_displacement))
        max_deviation = max(max_deviation, float(deviation))
        width = _edge_width_gu(raw_pixels, distance_inside, world.pixel_size_gu)
        raw_turning = _turning_metrics(
            raw_chain,
            high_frequency_angle=policy.high_frequency_turn_angle_radians,
        )
        turning = _turning_metrics(
            smooth_pixels,
            high_frequency_angle=policy.high_frequency_turn_angle_radians,
        )
        raw_turn_count_total += int(raw_turning["turn_count"])
        smooth_turn_count_total += int(turning["turn_count"])
        raw_high_frequency_turn_total += int(raw_turning["high_frequency_turn_count"])
        smooth_high_frequency_turn_total += int(turning["high_frequency_turn_count"])
        raw_zigzag_total += int(raw_turning["high_frequency_zigzag_reversal_count"])
        smooth_zigzag_total += int(turning["high_frequency_zigzag_reversal_count"])
        raw_metric_point_count = len(densify_polyline(raw_chain, policy.metric_sample_spacing_px))
        smooth_metric_point_count = len(densify_polyline(smooth_pixels, policy.metric_sample_spacing_px))
        hausdorff = _hausdorff_metrics(
            raw_chain,
            smooth_pixels,
            sample_spacing_px=policy.metric_sample_spacing_px,
        )
        hausdorff_values.append(hausdorff["symmetric_hausdorff_px"])
        raw_length_gu = _polyline_length(raw_gu)
        smooth_length_gu = _polyline_length(smooth_gu)
        inside_count = int(np.count_nonzero(smooth_raster & repaired))
        raster_count = int(np.count_nonzero(smooth_raster))
        vector_edges.append(
            {
                **dict(raw_edge),
                "raw_gu_chain": [[float(point[0]), float(point[1])] for point in raw_gu],
                "smooth_pixel_polyline": [[float(point[0]), float(point[1])] for point in smooth_pixels],
                "smooth_gu_polyline": [[float(point[0]), float(point[1])] for point in smooth_gu],
                "estimated_width_gu": width["estimated_width_gu"],
                "width_gu_p10": width["width_gu_p10"],
                "width_gu_p90": width["width_gu_p90"],
                "raw_length_gu": float(raw_length_gu),
                "length_gu": float(smooth_length_gu),
                "smoothing": {
                    "method": method,
                    "fallback_reason": fallback_reason,
                    "simplify_tolerance_px": float(effective_tolerance),
                    "configured_simplify_tolerance_px": float(policy.simplify_tolerance_px),
                    "simplified_point_count": len(simplified),
                    "raw_point_count": len(raw_chain),
                    "smooth_point_count": len(smooth_pixels),
                    "raw_metric_point_count": int(raw_metric_point_count),
                    "smooth_metric_point_count": int(smooth_metric_point_count),
                    "metric_sample_spacing_px": float(policy.metric_sample_spacing_px),
                    "max_deviation_px": float(deviation),
                    "corridor_tolerance_px": float(policy.corridor_tolerance_px),
                    "endpoint_displacement_gu": float(endpoint_displacement),
                    "raster_pixel_count": raster_count,
                    "raster_inside_repaired_pixel_count": inside_count,
                    "raster_inside_repaired_fraction": float(inside_count / raster_count) if raster_count else 1.0,
                    **hausdorff,
                    "raw_turning": raw_turning,
                    "smooth_turning": turning,
                    "high_frequency_turn_reduction_count": int(
                        raw_turning["high_frequency_turn_count"] - turning["high_frequency_turn_count"]
                    ),
                    "high_frequency_zigzag_reduction_count": int(
                        raw_turning["high_frequency_zigzag_reversal_count"]
                        - turning["high_frequency_zigzag_reversal_count"]
                    ),
                    **turning,
                },
            }
        )

    vector_nodes: list[dict[str, Any]] = []
    for node in graph.nodes:
        row = dict(node)
        row["position_gu"] = world.pixel_to_gu(row["position_px"])
        vector_nodes.append(row)
    vector_nodes.sort(key=lambda row: str(row["id"]))
    vector_edges.sort(key=lambda row: str(row["id"]))
    inside = int(np.count_nonzero(raster_all & repaired))
    total = int(np.count_nonzero(raster_all))
    metrics = {
        "transform": world.metadata(),
        "settings": {
            "simplify_tolerance_px": float(policy.simplify_tolerance_px),
            "sample_spacing_px": float(policy.sample_spacing_px),
            "corridor_tolerance_px": float(policy.corridor_tolerance_px),
            "minimum_curve_points": int(policy.minimum_curve_points),
            "pixel_quantization_noise_px": float(policy.pixel_quantization_noise_px),
            "scale_tolerance_multiplier": float(policy.scale_tolerance_multiplier),
            "scale_tolerance_cap_px": float(policy.scale_tolerance_cap_px),
            "high_frequency_turn_angle_radians": float(policy.high_frequency_turn_angle_radians),
            "self_intersection_min_separation_px": float(policy.self_intersection_min_separation_px),
            "metric_sample_spacing_px": float(policy.metric_sample_spacing_px),
        },
        "edge_count": len(vector_edges),
        "max_edge_deviation_px": float(max_deviation),
        "maximum_symmetric_hausdorff_px": float(max(hausdorff_values, default=0.0)),
        "maximum_endpoint_displacement_gu": float(max(endpoint_displacements, default=0.0)),
        "endpoint_displacement_values_gu": sorted(float(value) for value in endpoint_displacements),
        "smoothing_fallback_counts": dict(sorted(fallback_counts.items())),
        "smoothing_fallback_edge_ids": {
            reason: sorted(edge_ids) for reason, edge_ids in sorted(fallback_edge_ids.items())
        },
        "smoothing_method_counts": dict(sorted(method_counts.items())),
        "raw_fallback_edge_count": int(sum(fallback_counts.values())),
        "raw_turn_count_total": int(raw_turn_count_total),
        "smooth_turn_count_total": int(smooth_turn_count_total),
        # Dense curve sampling naturally increases the raw number of local
        # triplets.  The acceptance metric therefore uses the fixed-angle
        # high-frequency subset rather than comparing density-dependent totals.
        "sample_turn_count_delta": int(smooth_turn_count_total - raw_turn_count_total),
        "raw_high_frequency_turn_total": int(raw_high_frequency_turn_total),
        "smooth_high_frequency_turn_total": int(smooth_high_frequency_turn_total),
        "high_frequency_turn_reduction_count": int(
            raw_high_frequency_turn_total - smooth_high_frequency_turn_total
        ),
        "high_frequency_turn_reduction_fraction": float(
            (raw_high_frequency_turn_total - smooth_high_frequency_turn_total)
            / raw_high_frequency_turn_total
            if raw_high_frequency_turn_total
            else 0.0
        ),
        "raw_high_frequency_zigzag_reversal_total": int(raw_zigzag_total),
        "smooth_high_frequency_zigzag_reversal_total": int(smooth_zigzag_total),
        "high_frequency_zigzag_reduction_count": int(raw_zigzag_total - smooth_zigzag_total),
        "high_frequency_zigzag_reduction_fraction": float(
            (raw_zigzag_total - smooth_zigzag_total) / raw_zigzag_total if raw_zigzag_total else 0.0
        ),
        "centerline_raster_pixels": total,
        "centerline_raster_inside_repaired_pixels": inside,
        "centerline_raster_inside_repaired_fraction": float(inside / total) if total else 1.0,
        "self_induced_loop_edges": int(fallback_counts.get("self_induced_loop", 0)),
    }
    return VectorResult(nodes=vector_nodes, edges=vector_edges, metrics=metrics)


__all__ = [
    "RoadTransform",
    "VectorResult",
    "VectorSettings",
    "sample_catmull_rom",
    "simplify_polyline",
    "vectorize_graph",
]
