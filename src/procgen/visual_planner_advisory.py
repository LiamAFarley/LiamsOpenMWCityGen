"""Advisory visual-plan analyser for Cityforge settlement design.

Pipeline position
------------------
The analyser sits after the visual extension format gate and before a planner
hands a candidate back to the accepted T1.1/T1.2 pipeline.  It is a measured
review aid, not a layout solver: it reports hard geometric blockers and softer
urban-form advisories but never moves a stamp, invents a road, or chooses a
district.  A clean report therefore does not certify that a town is good; a
non-empty advisory list is a design conversation for the vision planner.

Inputs and outputs
------------------
``analyze_plan`` consumes a validated visual-plan extension, the exact
``TerrainBundle``, a requested ``PlanningRectangle``, the aligned road API
object, and D-STAMP geometry records.  It returns a deterministic JSON-ready
report with separate ``hard_errors`` and ``advisories`` arrays.  Every finding
has lot/door ids (empty arrays only for plan-level road/cluster findings),
measured geometry, and a concise human explanation.  An override reason is
copied into the report when it is supplied in ``advisory_overrides``.

The building-overlap check is Z-aware: it consumes the per-member 3D AABB
sidecar (``output/cityforge/stamps/stamp_volumes_v2.json``, next to the stamp
libraries) and the exact terrain field.  Each lot receives a plan-stage seat
Z = (max target-terrain height under its transformed hull) - stamp
``terrain_envelope.burial_depth_gu``, mirroring how the source city seated the
building (T1.2 performs exact seating later; this is the plan-stage
approximation).  A lot pair whose hulls overlap is excused only when EVERY
member-pair 3D conflict sits entirely below the target terrain surface
(conflict top at least ``SUBTERRANEAN_MARGIN_GU`` below the minimum terrain
height over the conflict rectangle); those pairs are recorded as
``subterranean_overlap_facts`` instead of hard errors, matching how the source
cities compose (buildings unfused only for underground overlap).  Missing or
unusable volumes/seat data fails closed with a ``stamp_volumes_unresolved``
hard error; it never falls back to a 2D-only check.

Hard errors
-----------
Building overlap (above-ground 3D member conflict), out-of-scope/water
footprint, footprint/road-corridor overlap, unresolved stamp/door/transform,
stamp volumes/seat data unresolved, and declared authored-road connections
that do not meet their target geometrically are blockers.

Advisories
----------
Door circulation, long access, tandem/rear access, repeated stamps, similar
orientations, public frontage, dense groups without open space, source
terrain/burial mismatch, and generic slope/circulation suggestions are
reported as ``advisory`` severity.  In particular, a dry slope is never a
universal hard exclusion; stamp-specific terrain evidence remains the final
compatibility authority.

Determinism and provenance
--------------------------
Records are processed in stable id order, pair measurements use exact planar
geometry helpers from T1.1, and no random numbers are used.  The report keeps
the discrete/raw terrain authority separate from the visual renderer and never
changes TES3 semantics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import cityplan
from .aligned_roads import SOURCE_ROAD_PRACTICAL_PATH_FRACTION
from .cityplan import point_in_ring, point_polyline_distance, ring_min_distance, rings_overlap_exact, rot2d_ccw
from .visual_planner_symbols import StampRenderRecord, resolve_stamps
from .visual_planner_terrain import PlanningRectangle, TerrainBundle, TerrainBundleError


SLOPE_ADVISORY_DEG = 15.0
DOOR_REACH_GU = 768.0
LONG_ACCESS_GU = 1800.0
TANDEM_RADIUS_GU = 5000.0
REPETITION_RADIUS_GU = 5500.0
OPEN_SPACE_RADIUS_GU = 1400.0
PUBLIC_FRONTAGE_GU = 1400.0
ROAD_CONNECTION_TOLERANCE_GU = 768.0

#: A member-pair conflict is subterranean when its Z-intersection TOP sits at
#: least this far below the minimum target-terrain height over the conflict
#: rectangle.  Matches how the source cities compose: buildings remain unfused
#: when their only 3D contact is underground.
SUBTERRANEAN_MARGIN_GU = 32.0
#: Canonical per-member 3D AABB sidecar, next to the accepted stamp libraries.
CANONICAL_STAMP_VOLUMES = (
    Path(__file__).resolve().parents[2] / "output/cityforge/stamps/stamp_volumes_v2.json")
#: Terrain sampling grid for seat/conflict-rectangle heights (mirrors the
#: 128-GU field spacing convention of ``TerrainBundle.terrain_metrics``).
_HEIGHT_SAMPLE_SPACING_GU = 128.0
_HEIGHT_SAMPLE_MAX = 32


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _ring_edges(ring: Sequence[Sequence[float]]) -> Iterable[tuple[tuple[float, float], tuple[float, float]]]:
    points = [(float(point[0]), float(point[1])) for point in ring]
    if points and points[0] != points[-1]:
        points.append(points[0])
    return zip(points, points[1:])


def _point_ring_distance(point: Sequence[float], ring: Sequence[Sequence[float]]) -> float:
    return min((cityplan.point_seg_distance(tuple(point), a, b)
                for a, b in _ring_edges(ring)), default=float("inf"))


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(points, points[1:]))


def _nearest_polyline(point: Sequence[float], polyline: Sequence[Sequence[float]]) -> tuple[float, tuple[float, float], int]:
    best = (float("inf"), (0.0, 0.0), -1)
    for index, (a, b) in enumerate(zip(polyline, polyline[1:])):
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq == 0.0 else max(0.0, min(1.0,
            ((float(point[0]) - ax) * dx + (float(point[1]) - ay) * dy) / length_sq))
        nearest = (ax + t * dx, ay + t * dy)
        distance = _distance(point, nearest)
        if distance < best[0]:
            best = (distance, nearest, index)
    return best


def _polyline_endpoint_distance(point: Sequence[float], polyline: Sequence[Sequence[float]]) -> float:
    if not polyline:
        return float("inf")
    return min(_distance(point, polyline[0]), _distance(point, polyline[-1]))


def _centroid(record: StampRenderRecord) -> tuple[float, float]:
    if record.hull:
        value = cityplan.polygon_centroid([list(point) for point in record.hull])
        return float(value[0]), float(value[1])
    return record.position


def _issue(severity: str, code: str, explanation: str, *, lot_ids: Iterable[str] = (),
           door_ids: Iterable[str] = (), measured: Mapping[str, Any] | None = None,
           target_ids: Iterable[str] = (), override_reason: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "lot_ids": sorted({str(value) for value in lot_ids if value}),
        "door_ids": sorted({str(value) for value in door_ids if value}),
        "target_ids": sorted({str(value) for value in target_ids if value}),
        "measured": dict(measured or {}),
        "explanation": explanation,
    }
    if override_reason:
        row["override_reason"] = override_reason
    return row


def _plan_target_map(document: Mapping[str, Any], network: Any, terrain: TerrainBundle) -> dict[str, dict[str, Any]]:
    """Collect plan-frame polylines/polygons for access and road checks."""

    targets: dict[str, dict[str, Any]] = {}
    for road in document.get("authored_roads", []):
        if isinstance(road, Mapping) and isinstance(road.get("road_id"), str):
            targets[road["road_id"]] = {
                "kind": "authored_road", "polyline": road.get("polyline_plan_gu", []),
                "width_gu": float(road.get("width_gu", 0.0)),
            }
    for alley in document.get("alleys", []):
        if isinstance(alley, Mapping) and isinstance(alley.get("alley_id"), str):
            targets[alley["alley_id"]] = {
                "kind": "alley", "polyline": alley.get("polyline_plan_gu", []),
                "width_gu": float(alley.get("width_gu", 0.0)),
            }
    for region in document.get("road_surface_polygons", []):
        if isinstance(region, Mapping) and isinstance(region.get("region_id"), str):
            targets[region["region_id"]] = {
                "kind": str(region.get("kind", "road_surface_polygon")),
                "polygon": region.get("polygon_plan_gu", []), "width_gu": 0.0,
            }
    for court in document.get("shared_courts", []):
        if isinstance(court, Mapping) and isinstance(court.get("court_id"), str):
            targets[court["court_id"]] = {
                "kind": "shared_court", "polygon": court.get("polygon_plan_gu", []), "width_gu": 0.0,
            }
    if network is not None:
        origin = terrain.origin_gu
        for source in document.get("existing_source_roads", []):
            if not isinstance(source, Mapping):
                continue
            edge_id = source.get("edge_id")
            if not isinstance(edge_id, str):
                continue
            try:
                edge = network.edge(edge_id)
            except Exception:  # noqa: BLE001 - unresolved edge is reported elsewhere
                continue
            targets[edge_id] = {
                "kind": "existing_source_road",
                "polyline": [network.to_site_local(point, origin) for point in edge.smooth_gu_polyline],
                "width_gu": float(edge.estimated_width_gu),
                "edge_id": edge_id,
            }
    return targets


def _corridor_rings(target: Mapping[str, Any]) -> list[list[tuple[float, float]]]:
    polyline = target.get("polyline")
    if not isinstance(polyline, list) or len(polyline) < 2:
        return []
    width = max(0.0, float(target.get("width_gu", 0.0)))
    if target.get("kind") == "existing_source_road":
        # Source corridors: only the practical path is forbidden to
        # buildings (the VTEX-blended band overstates it ~2.5x); authored
        # streets/alleys keep their full declared width.
        width *= SOURCE_ROAD_PRACTICAL_PATH_FRACTION
    if width == 0.0:
        return []
    half = width / 2.0
    rings: list[list[tuple[float, float]]] = []
    for a, b in zip(polyline, polyline[1:]):
        dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        nx, ny = -dy / length * half, dx / length * half
        rings.append([(float(a[0]) + nx, float(a[1]) + ny),
                      (float(b[0]) + nx, float(b[1]) + ny),
                      (float(b[0]) - nx, float(b[1]) - ny),
                      (float(a[0]) - nx, float(a[1]) - ny)])
    return rings


def _target_distance(point: Sequence[float], target: Mapping[str, Any]) -> tuple[float, tuple[float, float] | None]:
    if isinstance(target.get("polyline"), list) and len(target["polyline"]) >= 2:
        distance, nearest, _ = _nearest_polyline(point, target["polyline"])
        if target.get("kind") == "existing_source_road":
            # Source road corridors: reach is measured to the edge of the
            # PRACTICAL PATH (estimated width x practical fraction); a door
            # at the path edge is at the road.  Centerline reach wrongly
            # read such doors as unconnected and drove the designer to face
            # buildings away from the source roads (user-confirmed
            # 2026-08-12: hut door 1001 GU from centerline, ~700 GU from the
            # practical path edge -> connected).
            half = float(target.get("width_gu", 0.0)) * SOURCE_ROAD_PRACTICAL_PATH_FRACTION / 2.0
            if distance <= half:
                return 0.0, (float(point[0]), float(point[1]))
            edge_distance = distance - half
            t = edge_distance / distance if distance > 0.0 else 0.0
            edge_point = (float(point[0]) + (nearest[0] - float(point[0])) * t,
                          float(point[1]) + (nearest[1] - float(point[1])) * t)
            return edge_distance, edge_point
        return distance, nearest
    polygon = target.get("polygon")
    if isinstance(polygon, list) and len(polygon) >= 3:
        if point_in_ring((float(point[0]), float(point[1])), polygon):
            return 0.0, (float(point[0]), float(point[1]))
        return _point_ring_distance(point, polygon), None
    return float("inf"), None


def _target_has_open_space(targets: Mapping[str, Mapping[str, Any]], point: Sequence[float]) -> bool:
    for target in targets.values():
        if target.get("kind") in ("shared_court", "plaza", "market_circle", "road_surface_polygon"):
            distance, _ = _target_distance(point, target)
            if distance <= OPEN_SPACE_RADIUS_GU:
                return True
    return False


def _overrides(document: Mapping[str, Any]) -> dict[tuple[str, str, str], str]:
    result: dict[tuple[str, str, str], str] = {}
    for override in document.get("advisory_overrides", []):
        if not isinstance(override, Mapping):
            continue
        key = (str(override.get("code", "")), str(override.get("lot_id", "")),
               str(override.get("door_id", "")))
        reason = override.get("reason")
        if isinstance(reason, str) and reason.strip():
            result[key] = reason.strip()
    return result


def _with_override(row: dict[str, Any], overrides: Mapping[tuple[str, str, str], str]) -> dict[str, Any]:
    lot_ids = row.get("lot_ids", [])
    door_ids = row.get("door_ids", [])
    for lot_id in lot_ids or [""]:
        for door_id in door_ids or [""]:
            reason = overrides.get((row["code"], str(lot_id), str(door_id)))
            if reason:
                row["override_reason"] = reason
                return row
    reason = overrides.get((row["code"], str(lot_ids[0]) if lot_ids else "", ""))
    if reason:
        row["override_reason"] = reason
    return row


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["code"], tuple(row.get("lot_ids", [])),
                                         tuple(row.get("door_ids", [])), row["explanation"]))


# ---------------------------------------------------------------------------
# Z-aware building-overlap support (stamp volumes sidecar + terrain seating)
# ---------------------------------------------------------------------------

def _load_stamp_volumes(path: Path | str) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Load the per-member 3D AABB sidecar into {stamp_id: {source_id: box_local}}.

    ``box_local`` min/max are stamp-local (seed-door anchor at origin,
    world-aligned), in GU.  The sidecar's per-member ``below_ground`` flags are
    deliberately NOT used: they are an over-conservative source-space measure.
    """
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    volumes: dict[str, dict[str, Mapping[str, Any]]] = {}
    for library in data.get("libraries", {}).values():
        if not isinstance(library, Mapping):
            continue
        for stamp in library.get("stamps", []):
            if not isinstance(stamp, Mapping):
                continue
            stamp_id = stamp.get("stamp_id")
            if not isinstance(stamp_id, str) or not stamp_id:
                continue
            members: dict[str, Mapping[str, Any]] = {}
            for member in stamp.get("members", []):
                if not isinstance(member, Mapping):
                    continue
                source_id = member.get("source_id")
                box = member.get("box_local")
                if isinstance(source_id, str) and source_id and isinstance(box, Mapping):
                    members[source_id] = box
            volumes[stamp_id] = members
    return volumes


def _valid_box(box: Mapping[str, Any]) -> bool:
    """A box_local is usable when both corner pairs are finite and ordered."""
    low, high = box.get("min"), box.get("max")
    if not (isinstance(low, Sequence) and len(low) == 3 and
            isinstance(high, Sequence) and len(high) == 3):
        return False
    for lo, hi in zip(low, high):
        if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and
                math.isfinite(float(lo)) and math.isfinite(float(hi)) and float(lo) <= float(hi)):
            return False
    return True


def _seat_z(record: StampRenderRecord, terrain: TerrainBundle) -> float:
    """Plan-stage seat Z for one lot: max terrain under hull minus source burial.

    The source city sat each building with its bbox bottom buried
    ``burial_depth_gu`` below the max terrain under the footprint, so the
    plan-stage anchor seat reproduces that relationship.  T1.2 performs exact
    seating later; this is the documented approximation.  If the terrain
    bundle cannot supply heights this raises (FAILURE) -- it never seats at
    zero.
    """
    metrics = terrain.terrain_metrics(list(record.hull))
    height_max = metrics.get("height_max_gu")
    if not isinstance(height_max, (int, float)) or not math.isfinite(float(height_max)):
        raise TerrainBundleError(
            f"cannot estimate seat Z for lot {record.lot_id}: terrain heights unavailable "
            f"under hull (height_max_gu={height_max!r})")
    burial = record.terrain_envelope.get("burial_depth_gu")
    if not isinstance(burial, (int, float)) or not math.isfinite(float(burial)):
        raise TerrainBundleError(
            f"cannot estimate seat Z for lot {record.lot_id}: stamp "
            f"{record.stamp_id} terrain_envelope.burial_depth_gu is missing")
    return float(height_max) - float(burial)


def _rotated_xy_corners(box: Mapping[str, Any], yaw_deg: float) -> tuple[tuple[float, float], ...]:
    """2D-rotate the 8 box corners about the stamp-local origin (yaw only).

    Reuses the exact transform helper used for hulls (``cityplan.rot2d_ccw``);
    Z is added later as the seat offset, rotation does not touch it.  The 8
    corners are the four XY corner combinations at both Z extremes; they
    project to four distinct XY points (duplicated here for literalness).
    """
    xs = (float(box["min"][0]), float(box["max"][0]))
    ys = (float(box["min"][1]), float(box["max"][1]))
    return tuple(rot2d_ccw(x, y, yaw_deg) for x in xs for y in ys) * 2


def _sample_rect_min(terrain: TerrainBundle, rect: Sequence[float],
                     cache: dict[tuple[float, float, float, float], float | None]) -> float | None:
    """Minimum terrain height over an XY rectangle (survey-clipped).

    The rectangle is clipped to the survey frame; an empty clip returns None
    (the caller treats that as unverifiable, not subterranean).  Sampling
    mirrors the 128-GU field convention with a bounded grid, and results are
    memoized because many member pairs share the same intersection rectangle.
    """
    x0, y0, x1, y1 = (float(value) for value in rect)
    span_x, span_y = terrain.site_span_gu
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(span_x, x1), min(span_y, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    key = (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))
    if key in cache:
        return cache[key]
    count_x = max(2, min(_HEIGHT_SAMPLE_MAX,
                         int(math.ceil((x1 - x0) / _HEIGHT_SAMPLE_SPACING_GU)) + 1))
    count_y = max(2, min(_HEIGHT_SAMPLE_MAX,
                         int(math.ceil((y1 - y0) / _HEIGHT_SAMPLE_SPACING_GU)) + 1))
    best = float("inf")
    for step_y in range(count_y):
        for step_x in range(count_x):
            fx = x0 + (x1 - x0) * step_x / (count_x - 1)
            fy = y0 + (y1 - y0) * step_y / (count_y - 1)
            best = min(best, terrain.sample_height(fx, fy))
    cache[key] = best
    return best


def _member_conflicts(first: StampRenderRecord, second: StampRenderRecord,
                      boxes_a: Mapping[str, Mapping[str, Any]],
                      boxes_b: Mapping[str, Mapping[str, Any]],
                      seat_a: float, seat_b: float, terrain: TerrainBundle,
                      cache: dict[tuple[float, float, float, float], float | None],
                      ) -> list[dict[str, Any]]:
    """Member-pair 3D conflicts between two lots, each classified subterranean.

    A conflict exists when the rotated XY AABBs intersect (positive area) AND
    the world-Z intervals intersect (positive length).  It is subterranean
    when the Z-intersection TOP is at least ``SUBTERRANEAN_MARGIN_GU`` below
    the minimum terrain height over the XY intersection rectangle.
    """
    members_a = sorted(boxes_a)
    members_b = sorted(boxes_b)
    corners_a = {source_id: _rotated_xy_corners(box, first.yaw_deg)
                 for source_id, box in boxes_a.items()}
    corners_b = {source_id: _rotated_xy_corners(box, second.yaw_deg)
                 for source_id, box in boxes_b.items()}
    conflicts: list[dict[str, Any]] = []
    for source_a in members_a:
        ax0 = first.position[0] + min(point[0] for point in corners_a[source_a])
        ax1 = first.position[0] + max(point[0] for point in corners_a[source_a])
        ay0 = first.position[1] + min(point[1] for point in corners_a[source_a])
        ay1 = first.position[1] + max(point[1] for point in corners_a[source_a])
        za0 = seat_a + float(boxes_a[source_a]["min"][2])
        za1 = seat_a + float(boxes_a[source_a]["max"][2])
        for source_b in members_b:
            bx0 = second.position[0] + min(point[0] for point in corners_b[source_b])
            bx1 = second.position[0] + max(point[0] for point in corners_b[source_b])
            by0 = second.position[1] + min(point[1] for point in corners_b[source_b])
            by1 = second.position[1] + max(point[1] for point in corners_b[source_b])
            if not (ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0):
                continue
            zb0 = seat_b + float(boxes_b[source_b]["min"][2])
            zb1 = seat_b + float(boxes_b[source_b]["max"][2])
            z_lo, z_hi = max(za0, zb0), min(za1, zb1)
            if z_lo >= z_hi:
                continue
            terrain_min = _sample_rect_min(
                terrain, (max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)), cache)
            subterranean = (terrain_min is not None and
                            z_hi <= terrain_min - SUBTERRANEAN_MARGIN_GU)
            conflicts.append({
                "member_source_ids": [source_a, source_b],
                "intersection_top_z_gu": z_hi,
                "terrain_min_z_gu": terrain_min,
                "margin_gu": None if terrain_min is None else terrain_min - z_hi,
                "subterranean": bool(subterranean),
            })
    return conflicts


def _building_overlap_z_aware(
    records: Sequence[StampRenderRecord],
    volumes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    terrain: TerrainBundle,
    skip_lots: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace the 2D hull-only pair check with the Z-aware member check.

    Returns (building_overlap hard errors, subterranean_overlap_facts).
    A pair is excused only when EVERY member-pair conflict is subterranean
    (or no member conflict exists at all); otherwise the first offending
    member pair and its measured above-ground height enrich the error.
    """
    errors: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    cache: dict[tuple[float, float, float, float], float | None] = {}
    seats: dict[str, float] = {record.lot_id: _seat_z(record, terrain) for record in records}
    pairs = [(first, second) for index, first in enumerate(records)
             for second in records[index + 1:]]
    for first, second in pairs:
        if first.lot_id in skip_lots or second.lot_id in skip_lots:
            continue
        if not rings_overlap_exact([list(point) for point in first.hull],
                                   [list(point) for point in second.hull]):
            continue
        conflicts = _member_conflicts(
            first, second, volumes[first.stamp_id], volumes[second.stamp_id],
            seats[first.lot_id], seats[second.lot_id], terrain, cache)
        if conflicts and not all(conflict["subterranean"] for conflict in conflicts):
            offender = next(conflict for conflict in conflicts if not conflict["subterranean"])
            measured: dict[str, Any] = {
                "overlap": True,
                "centroid_distance_gu": _distance(_centroid(first), _centroid(second)),
                "member_source_ids": offender["member_source_ids"],
                "conflict_top_z_gu": offender["intersection_top_z_gu"],
                "terrain_min_z_gu": offender["terrain_min_z_gu"],
                "above_ground_gu": None if offender["terrain_min_z_gu"] is None
                else offender["intersection_top_z_gu"] - offender["terrain_min_z_gu"],
            }
            errors.append(_issue(
                "error", "building_overlap",
                "transformed building volumes physically overlap above the target "
                "terrain surface (3D member conflict)",
                lot_ids=[first.lot_id, second.lot_id], measured=measured))
        else:
            facts.append({
                "lot_ids": sorted([first.lot_id, second.lot_id]),
                "stamp_ids": [first.stamp_id, second.stamp_id],
                "conflict_count": len(conflicts),
                "margin_threshold_gu": SUBTERRANEAN_MARGIN_GU,
                "minimum_margin_gu": min((conflict["margin_gu"] for conflict in conflicts),
                                         default=None),
                "conflicts": [{
                    "member_source_ids": conflict["member_source_ids"],
                    "intersection_top_z_gu": conflict["intersection_top_z_gu"],
                    "terrain_min_z_gu": conflict["terrain_min_z_gu"],
                    "margin_gu": conflict["margin_gu"],
                } for conflict in conflicts],
                "excused": True,
            })
    return errors, facts


def analyze_plan(
    document: Mapping[str, Any],
    terrain: TerrainBundle,
    rectangle: PlanningRectangle,
    *,
    aligned_network: Any,
    stamp_geometry: Mapping[str, Mapping[str, Any]],
    format_issues: Sequence[Any] = (),
    stamp_volumes_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return separated hard errors/advisories for one visual-plan document."""

    hard_errors: list[dict[str, Any]] = []
    advisories: list[dict[str, Any]] = []
    overrides = _overrides(document)
    records: list[StampRenderRecord] = []

    if format_issues:
        for issue in format_issues:
            row = _issue("error", "format_invalid", str(getattr(issue, "message", issue)),
                         measured={"path": getattr(issue, "path", "$")})
            hard_errors.append(row)

    # Resolve stamps independently so a malformed visual record is a hard
    # blocker, not silently omitted from geometry/advisory analysis.
    placements = [placement for placement in document.get("stamps", []) if isinstance(placement, Mapping)]
    for placement in sorted(placements, key=lambda item: str(item.get("lot_id", ""))):
        lot_id = str(placement.get("lot_id", ""))
        stamp_id = str(placement.get("stamp_id", ""))
        stamp = stamp_geometry.get(stamp_id)
        if stamp is None:
            hard_errors.append(_issue("error", "stamp_unresolved",
                                      f"stamp {stamp_id!r} is not available in the accepted D-STAMP geometry",
                                      lot_ids=[lot_id], measured={"stamp_id": stamp_id}))
            continue
        position = placement.get("position_plan_gu")
        yaw = placement.get("yaw_deg")
        if not (isinstance(position, Sequence) and len(position) == 2 and
                all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in position) and
                isinstance(yaw, (int, float)) and math.isfinite(float(yaw))):
            hard_errors.append(_issue("error", "transform_unresolved",
                                      "stamp position/yaw cannot be transformed deterministically",
                                      lot_ids=[lot_id], measured={"position_plan_gu": position, "yaw_deg": yaw}))
            continue
        members = stamp.get("members")
        hull = stamp.get("footprint", {}).get("hull_xy_rel")
        if not isinstance(members, list) or not isinstance(hull, list) or len(hull) < 3:
            hard_errors.append(_issue("error", "stamp_geometry_unresolved",
                                      "accepted stamp has no usable members or footprint hull",
                                      lot_ids=[lot_id], measured={"stamp_id": stamp_id}))
            continue
        door_members = [member for member in members if isinstance(member, Mapping) and member.get("is_door")]
        if not door_members:
            hard_errors.append(_issue("error", "doors_unresolved",
                                      "stamp has no measured door/access anchors",
                                      lot_ids=[lot_id], measured={"stamp_id": stamp_id, "door_count": 0}))
            continue
        try:
            records.extend(resolve_stamps({"stamps": [placement]}, {stamp_id: stamp}))
        except (TypeError, ValueError, KeyError) as exc:
            hard_errors.append(_issue("error", "transform_unresolved",
                                      f"stamp transform failed: {exc}", lot_ids=[lot_id],
                                      measured={"stamp_id": stamp_id, "yaw_deg": float(yaw)}))

    record_by_lot = {record.lot_id: record for record in records}
    targets = _plan_target_map(document, aligned_network, terrain)
    # Include declared authored connection target points in the target map for
    # connection validation without inventing a route.
    declared_targets: dict[str, tuple[float, float]] = {}
    for section in ("authored_roads", "alleys", "shared_courts"):
        key = "road_id" if section == "authored_roads" else "alley_id" if section == "alleys" else "court_id"
        for record in document.get(section, []):
            if not isinstance(record, Mapping):
                continue
            for connection in record.get("connection_targets", []):
                if isinstance(connection, Mapping) and isinstance(connection.get("target_id"), str):
                    target = connection.get("at_plan_gu")
                    if isinstance(target, Sequence) and len(target) == 2:
                        declared_targets[connection["target_id"]] = (float(target[0]), float(target[1]))

    # Authored roads are hard-gated at their declared endpoints.
    for section in ("authored_roads", "alleys"):
        id_key = "road_id" if section == "authored_roads" else "alley_id"
        for road in document.get(section, []):
            if not isinstance(road, Mapping):
                continue
            road_id = str(road.get(id_key, ""))
            points = road.get("polyline_plan_gu", [])
            if not isinstance(points, list) or len(points) < 2:
                continue
            for connection in road.get("connection_targets", []):
                if not isinstance(connection, Mapping):
                    continue
                target_id = str(connection.get("target_id", ""))
                target_point = connection.get("at_plan_gu")
                if not (isinstance(target_point, Sequence) and len(target_point) == 2):
                    hard_errors.append(_issue("error", "road_connection_unresolved",
                                              f"{road_id} declares connection {target_id!r} without a target point",
                                              target_ids=[target_id], measured={"road_id": road_id}))
                    continue
                # A connection point may be an intentional junction along a
                # smooth polyline, not only an endpoint.  Measure against the
                # whole declared geometry; this keeps a road's connection
                # contract compatible with bends and plaza entries.
                endpoint_distance = _nearest_polyline(target_point, points)[0]
                tolerance = float(connection.get("tolerance_gu", ROAD_CONNECTION_TOLERANCE_GU))
                if endpoint_distance > tolerance:
                    hard_errors.append(_issue("error", "road_disconnected",
                                              f"authored {section[:-1]} {road_id} declares {target_id} but its nearest endpoint is geometrically disconnected",
                                              target_ids=[road_id, target_id],
                                              measured={"endpoint_distance_gu": endpoint_distance,
                                                       "allowed_tolerance_gu": tolerance,
                                                       "road_endpoint_gu": [list(points[0]), list(points[-1])],
                                                       "target_gu": list(target_point)}))

    # Exact transformed stamp scope/water/overlap checks.
    for record in sorted(records, key=lambda item: item.lot_id):
        lot_id = record.lot_id
        if not record.hull:
            continue
        out = [list(point) for point in record.hull if not rectangle.contains(point, requested=True)]
        if out:
            hard_errors.append(_issue("error", "building_outside_scope",
                                      "transformed building footprint lies outside the requested planning rectangle",
                                      lot_ids=[lot_id], measured={"outside_vertices": out,
                                                                   "rectangle": rectangle.requested_bounds_gu}))
        covered = cityplan.tiles_covered_by_ring([list(point) for point in record.hull], side=terrain.road_mask.shape[0])
        water_tiles = [list(tile) for tile in covered
                       if 0 <= tile[0] < terrain.water_mask.shape[1] and 0 <= tile[1] < terrain.water_mask.shape[0]
                       and terrain.water_mask[tile[1], tile[0]]]
        if water_tiles:
            hard_errors.append(_issue("error", "building_in_water",
                                      "transformed building footprint covers exact survey water tiles",
                                      lot_ids=[lot_id], measured={"water_tiles": water_tiles,
                                                                   "water_tile_count": len(water_tiles)}))
        # Every measured door must transform and remain identifiable.
        if not record.doors:
            hard_errors.append(_issue("error", "doors_unresolved",
                                      "resolved stamp has no transformed doors", lot_ids=[lot_id],
                                      measured={"stamp_id": record.stamp_id}))
        for door in record.doors:
            point = door["position_plan_gu"]
            if not rectangle.contains(point, requested=False):
                hard_errors.append(_issue("error", "door_transform_outside_scope",
                                          "measured door anchor transforms outside the render rectangle",
                                          lot_ids=[lot_id], door_ids=[door["door_id"]],
                                          measured={"door_plan_gu": point,
                                                    "render_bounds_gu": rectangle.render_bounds_gu}))

    # Building overlap is Z-aware via the stamp-volumes sidecar.  The sidecar
    # loads once per run; when it (or a used stamp's entry/member boxes) is
    # unavailable the plan fails closed with stamp_volumes_unresolved and the
    # pair check is skipped entirely -- never a silent 2D-only fallback.
    volumes_path = Path(stamp_volumes_path) if stamp_volumes_path is not None \
        else CANONICAL_STAMP_VOLUMES
    volumes: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None
    volumes_issue: str | None = None
    try:
        volumes = _load_stamp_volumes(volumes_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        volumes_issue = f"stamp volumes sidecar cannot be loaded: {exc}"
    if volumes_issue is not None:
        hard_errors.append(_issue("error", "stamp_volumes_unresolved", volumes_issue,
                                  measured={"volumes_path": str(volumes_path)}))
    unresolved_lots: set[str] = set()
    if volumes is not None:
        for record in records:
            stamp = stamp_geometry.get(record.stamp_id) or {}
            library_members = [member for member in stamp.get("members", [])
                               if isinstance(member, Mapping)]
            box_map = volumes.get(record.stamp_id)
            missing: list[str] = []
            if box_map is None:
                missing.append("(no stamp entry)")
            else:
                for member in library_members:
                    source_id = str(member.get("source_id", ""))
                    box = box_map.get(source_id)
                    if not source_id or box is None or not _valid_box(box):
                        missing.append(source_id or "(unnamed member)")
            if missing:
                hard_errors.append(_issue(
                    "error", "stamp_volumes_unresolved",
                    f"stamp {record.stamp_id} lacks usable per-member 3D boxes for "
                    "Z-aware overlap analysis",
                    lot_ids=[record.lot_id],
                    measured={"stamp_id": record.stamp_id,
                              "missing_members": sorted(set(missing))}))
                unresolved_lots.add(record.lot_id)
    if volumes is not None:
        overlap_errors, subterranean_facts = _building_overlap_z_aware(
            records, volumes, terrain, skip_lots=unresolved_lots)
        hard_errors.extend(overlap_errors)
    else:
        subterranean_facts: list[dict[str, Any]] = []

    records_by_pair = [(a, b) for index, a in enumerate(records) for b in records[index + 1:]]

    # Road corridor hard checks use referenced aligned existing roads plus
    # authored road/alleys.  Plaza/court polygons are circulation targets, not
    # forbidden corridors.
    corridor_targets = {key: value for key, value in targets.items()
                        if value.get("kind") in ("existing_source_road", "authored_road", "alley")}
    for record in records:
        for target_id, target in sorted(corridor_targets.items()):
            if target_id not in {str(item.get("target_id")) for item in document.get("advisory_overrides", []) if isinstance(item, Mapping)}:
                for corridor in _corridor_rings(target):
                    if rings_overlap_exact([list(point) for point in record.hull], [list(point) for point in corridor]):
                        hard_errors.append(_issue("error", "building_road_overlap",
                                                  "building footprint unintentionally overlaps an established/authored road corridor",
                                                  lot_ids=[record.lot_id], target_ids=[target_id],
                                                  measured={"corridor_width_gu": target.get("width_gu"),
                                                           "target_kind": target.get("kind"),
                                                           "footprint_centroid_gu": list(_centroid(record))}))
                        break

    # Door access/circulation advisories.  All measured doors are considered;
    # unused doors are explicitly exempted from the reach warning.
    for record in sorted(records, key=lambda item: item.lot_id):
        for door in record.doors:
            intent = str(door.get("intent", "public"))
            if intent == "unused":
                continue
            point = door["position_plan_gu"]
            candidates: list[tuple[float, str, str, tuple[float, float] | None]] = []
            for target_id, target in sorted(targets.items()):
                distance, nearest = _target_distance(point, target)
                candidates.append((distance, target_id, str(target.get("kind")), nearest))
            candidates.sort(key=lambda item: (item[0], item[1]))
            best = candidates[0] if candidates else (float("inf"), "", "", None)
            declared_target = door.get("target_id")
            if declared_target and declared_target in targets:
                declared_distance, declared_nearest = _target_distance(point, targets[declared_target])
                best = (declared_distance, declared_target, str(targets[declared_target].get("kind")), declared_nearest)
            if not math.isfinite(best[0]) or best[0] > DOOR_REACH_GU:
                row = _issue("advisory", "door_unconnected",
                             "no usable circulation face reaches this measured door; relate it to a road, alley, plaza, shared court, or intentional cluster",
                             lot_ids=[record.lot_id], door_ids=[door["door_id"]],
                             target_ids=[best[1]], measured={"nearest_distance_gu": best[0],
                                                              "reach_threshold_gu": DOOR_REACH_GU,
                                                              "nearest_target_kind": best[2]})
                advisories.append(_with_override(row, overrides))
            elif best[0] > LONG_ACCESS_GU:
                row = _issue("advisory", "door_access_long",
                             "door has a long measured access reach; consider a closer alley, court, plaza, or street connection",
                             lot_ids=[record.lot_id], door_ids=[door["door_id"]], target_ids=[best[1]],
                             measured={"access_distance_gu": best[0], "long_access_threshold_gu": LONG_ACCESS_GU,
                                       "target_kind": best[2]})
                advisories.append(_with_override(row, overrides))

    # Source stamp terrain evidence is compared to the exact planning field;
    # this is a warning even for a steep but dry, otherwise usable site.
    for placement in placements:
        lot_id = str(placement.get("lot_id", ""))
        record = record_by_lot.get(lot_id)
        stamp = stamp_geometry.get(str(placement.get("stamp_id", "")))
        if record is None or stamp is None:
            continue
        metrics = terrain.terrain_metrics(record.hull)
        source_env = stamp.get("terrain_envelope", {})
        observed = placement.get("terrain_evidence", {}) if isinstance(placement.get("terrain_evidence"), Mapping) else {}
        source_relief = float(observed.get("observed_relief_gu", source_env.get("footprint_relief_gu", 0.0)))
        source_slope = float(observed.get("observed_slope_deg", source_env.get("footprint_slope_deg", 0.0)))
        source_burial = float(observed.get("observed_burial_depth_gu", source_env.get("burial_depth_gu", 0.0)))
        comparisons = {
            "relief_gu": (metrics.get("relief_gu"), source_relief),
            "slope_deg": (metrics.get("slope_mean_deg"), source_slope),
            "burial_depth_gu": (None, source_burial),
        }
        for label, (measured_value, source_value) in comparisons.items():
            if measured_value is None:
                continue
            tolerance = max(128.0 if label != "slope_deg" else 5.0,
                            abs(float(source_value)) * (0.75 if label != "slope_deg" else 1.0))
            delta = abs(float(measured_value) - float(source_value))
            if delta > tolerance:
                row = _issue("advisory", "stamp_terrain_mismatch",
                             "planning terrain differs materially from the stamp-specific source envelope; review stairs, burial, and access before placement",
                             lot_ids=[lot_id], measured={"measure": label,
                                                          "planning_value": measured_value,
                                                          "source_value": source_value,
                                                          "delta": delta,
                                                          "tolerance": tolerance})
                advisories.append(_with_override(row, overrides))
        slope = metrics.get("slope_max_deg")
        if isinstance(slope, (int, float)) and slope > SLOPE_ADVISORY_DEG:
            row = _issue("advisory", "slope_advisory",
                         "generic 15-degree buildable-mask slope is advisory only; use stamp-specific relief/stairs evidence or an intentional slope-capable treatment",
                         lot_ids=[lot_id], measured={"max_slope_deg": slope,
                                                      "advisory_threshold_deg": SLOPE_ADVISORY_DEG,
                                                      "water_tiles": sum(1 for tile in cityplan.tiles_covered_by_ring([list(p) for p in record.hull]) if terrain.water_mask[tile[1], tile[0]])})
            advisories.append(_with_override(row, overrides))

    # Repetition and orientation concentration are local composition warnings,
    # never hard rejection rules.
    by_stamp: dict[str, list[StampRenderRecord]] = defaultdict(list)
    for record in records:
        by_stamp[record.stamp_id].append(record)
    for stamp_id, group in sorted(by_stamp.items()):
        if len(group) >= 2:
            distances = [_distance(_centroid(a), _centroid(b)) for index, a in enumerate(group) for b in group[index + 1:]]
            nearby = [distance for distance in distances if distance <= REPETITION_RADIUS_GU]
            if nearby:
                row = _issue("advisory", "repeated_stamp_concentration",
                             "nearby lots use the same stamp repeatedly; vary kit, orientation, setbacks, or clustering unless the repetition has a written reason",
                             lot_ids=[record.lot_id for record in group], measured={"stamp_id": stamp_id,
                                                                                     "group_count": len(group),
                                                                                     "nearby_pair_count": len(nearby),
                                                                                     "radius_gu": REPETITION_RADIUS_GU})
                advisories.append(_with_override(row, overrides))
    for first, second in records_by_pair:
        if _distance(_centroid(first), _centroid(second)) <= REPETITION_RADIUS_GU:
            delta = abs((first.yaw_deg - second.yaw_deg + 180.0) % 360.0 - 180.0)
            if delta <= 15.0:
                row = _issue("advisory", "neighbor_orientation_concentration",
                             "neighboring buildings have near-identical orientations; vary yaw or explain the shared frontage",
                             lot_ids=[first.lot_id, second.lot_id], measured={"yaw_delta_deg": delta,
                                                                                "neighbor_distance_gu": _distance(_centroid(first), _centroid(second)),
                                                                                "orientation_threshold_deg": 15.0})
                advisories.append(_with_override(row, overrides))

    # Tandem/rear and public-frontage checks use doors and open-space targets.
    for first, second in records_by_pair:
        distance = _distance(_centroid(first), _centroid(second))
        if distance > TANDEM_RADIUS_GU:
            continue
        first_front = min((_target_distance(door["position_plan_gu"], target)[0]
                           for door in first.doors for target in targets.values()), default=float("inf"))
        second_front = min((_target_distance(door["position_plan_gu"], target)[0]
                            for door in second.doors for target in targets.values()), default=float("inf"))
        # A rear lot is the one whose centre is farther from the nearest target.
        # The pair is still reported together so the planner can see the tandem
        # relationship; lack of an explicit alley/court route is the advisory
        # trigger, not the generic presence of a nearby target somewhere else.
        rear_target = second if second_front >= first_front else first
        rear_has_explicit_route = any(
            str(door.get("target_id", "")) in targets and
            targets[str(door.get("target_id"))].get("kind") in ("alley", "shared_court")
            for door in rear_target.doors)
        if max(first_front, second_front) > DOOR_REACH_GU and not rear_has_explicit_route:
            row = _issue("advisory", "rear_tandem_access",
                         "nearby/rear building has no alley, court, plaza, or road route; avoid unexplained tandem access",
                         lot_ids=[first.lot_id, second.lot_id], measured={"building_distance_gu": distance,
                                                                            "front_access_distances_gu": [first_front, second_front]})
            advisories.append(_with_override(row, overrides))

    public_categories = {"civic", "public", "commercial", "market", "tavern", "hall", "keep", "shop"}
    for placement in placements:
        lot_id = str(placement.get("lot_id", ""))
        record = record_by_lot.get(lot_id)
        if record is None:
            continue
        public_doors = [door for door in record.doors if door.get("intent") == "public"]
        if public_doors and (record.category.casefold() in public_categories or
                             placement.get("category", "").casefold() in public_categories):
            distances = [_target_distance(door["position_plan_gu"], target)[0]
                         for door in public_doors for target in targets.values()]
            best = min(distances, default=float("inf"))
            if best > PUBLIC_FRONTAGE_GU:
                row = _issue("advisory", "public_building_no_frontage",
                             "major public/commercial building lacks meaningful frontage; give its public door a road, alley, plaza, or court relationship",
                             lot_ids=[lot_id], door_ids=[door["door_id"] for door in public_doors],
                             measured={"nearest_frontage_distance_gu": best,
                                       "frontage_threshold_gu": PUBLIC_FRONTAGE_GU})
                advisories.append(_with_override(row, overrides))

    for first, second in records_by_pair:
        if _distance(_centroid(first), _centroid(second)) <= OPEN_SPACE_RADIUS_GU * 3:
            midpoint = ((_centroid(first)[0] + _centroid(second)[0]) / 2.0,
                        (_centroid(first)[1] + _centroid(second)[1]) / 2.0)
            nearby = [record for record in records if _distance(_centroid(record), midpoint) <= OPEN_SPACE_RADIUS_GU * 2]
            if len(nearby) >= 3 and not _target_has_open_space(targets, midpoint):
                row = _issue("advisory", "dense_group_no_open_space",
                             "dense building group has no nearby shared court, plaza, or other circulation/open-space target",
                             lot_ids=[record.lot_id for record in nearby],
                             measured={"group_count": len(nearby), "group_center_gu": list(midpoint),
                                       "open_space_radius_gu": OPEN_SPACE_RADIUS_GU})
                advisories.append(_with_override(row, overrides))
                break

    # A cluster-level suggestion is useful when all doors miss all circulation
    # targets, but remains an advisory rather than a synthesized route.
    unconnected = []
    for record in records:
        if any(min((_target_distance(door["position_plan_gu"], target)[0]
                   for target in targets.values()), default=float("inf")) > DOOR_REACH_GU
               for door in record.doors if door.get("intent") != "unused"):
            unconnected.append(record)
    if len(unconnected) >= 2:
        center = (sum(_centroid(record)[0] for record in unconnected) / len(unconnected),
                  sum(_centroid(record)[1] for record in unconnected) / len(unconnected))
        row = _issue("advisory", "suggest_circulation_target",
                     "unconnected building cluster may benefit from a new authored street, alley, court, or plaza; this analyser does not design it",
                     lot_ids=[record.lot_id for record in unconnected],
                     measured={"cluster_count": len(unconnected), "cluster_center_gu": list(center),
                               "unconnected_threshold_gu": DOOR_REACH_GU})
        advisories.append(_with_override(row, overrides))

    hard_errors = _sorted_rows(hard_errors)
    advisories = _sorted_rows(advisories)
    return {
        "schema_version": 1,
        "kind": "cityforge_visual_plan_advisory_report",
        "plan_id": document.get("plan_id"),
        "rectangle": rectangle.to_dict(),
        "hard_errors": hard_errors,
        "advisories": advisories,
        "subterranean_overlap_facts": subterranean_facts,
        "summary": {
            "hard_error_count": len(hard_errors),
            "advisory_count": len(advisories),
            "subterranean_overlap_pair_count": len(subterranean_facts),
            "valid_for_visual_review": not hard_errors,
            "override_count": sum(1 for row in hard_errors + advisories if row.get("override_reason")),
            "slope_is_advisory_only": True,
            "tes3_semantics_changed": False,
        },
    }


__all__ = ["CANONICAL_STAMP_VOLUMES", "SUBTERRANEAN_MARGIN_GU", "analyze_plan"]
