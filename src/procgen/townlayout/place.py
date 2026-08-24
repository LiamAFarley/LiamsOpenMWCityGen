"""Parcel-aware stamp placement for V2 townlayout (Phase 18).

Purpose
-------
Seat one stamp per occupied parcel.  Core wards (residential/craft/
market) use curb poses only: stone gaps {0, 32} GU, wood {32, 64} GU.
Interior centroid is outskirts/keep fallback.  Prefers the largest hull
that seats.

Inputs
------
Phase 17 candidate, stamp index, D-STAMP v2 library map, optional SiteContext.

Outputs
-------
``placements`` plus per-parcel rejection histograms.  Budget exhaustion is
``inconclusive``, never a false unsatisfiable.

Pipeline position
-----------------
V2 townlayout Phase 18 placement; no D-PLAN/VTEX.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import LineString, Point, Polygon
import numpy as np

from procgen.cityplan import rot2d_ccw
from procgen.frontage_fit import (
    FrontageFitError,
    _rings_conflict,
    _stamp_doors,
    _stamp_hull,
    _transform_hull,
)

from .constants import (
    ALLEY_CLEAR_WIDTH_GU,
    FRONTAGE_TOUCH_GU,
    SEATING_INSET_GU,
    SLOPE_HARD_DEG,
    SLOPE_SOFT_START_DEG,
)
from .geometry import polygon_from_ring
from .site_context import SiteContext, _plan_to_px, diagnostic_view
from .parcels import _best_road, _frame_for_road
from .stamp_index import CORE_WARDS, kit_family, list_compatible_stamps
from .validate import TownLayoutError

DOOR_GAPS_STONE = (0.0, 8.0, 16.0, 32.0, 64.0, 96.0, 160.0)
DOOR_GAPS_WOOD = (8.0, 16.0, 32.0, 48.0, 96.0, 160.0)
DOOR_GAPS_OUTSKIRTS = (32.0, 64.0, 128.0, 256.0)
YAW_PERTURB_DEG = (0.0,)
ALONG_STEP_GU = 64.0
ALONG_SLIDES = (0.0, 48.0, -48.0, 96.0, -96.0)
MAX_IDENTICAL = 16
SEARCH_BUDGET = 500000
PER_PARCEL_BUDGET = 6000
MAX_POSES_PER_STAMP = 2
MAX_POSES_PER_PARCEL = 8
TRIES_PER_STAMP = 400
HULL_SLOP_GU2 = 512.0
YAW_FLIPS_DEG = (0.0, 180.0)


def _ward_type(parcel: dict, wards: list[dict]) -> str:
    for ward in wards:
        if ward.get("ward_id") == parcel.get("ward_id"):
            return ward["ward_type"]
    return "residential"


def _gaps_for(row: dict, depth_gu: Optional[float] = None) -> tuple[float, ...]:
    fam = kit_family(row)
    if fam == "stone":
        gaps = DOOR_GAPS_STONE
    elif fam == "outskirts":
        gaps = DOOR_GAPS_OUTSKIRTS
    else:
        gaps = DOOR_GAPS_WOOD
    if depth_gu is None:
        return gaps
    slack = max(0.0, float(depth_gu) - float(row.get("obb_depth_gu") or 0.0))
    clipped = tuple(g for g in gaps if g <= slack + 16.0)
    return clipped if clipped else (min(gaps[0], slack),)


def _shape(parcel: dict, roads: Optional[list] = None) -> dict:
    poly = polygon_from_ring(parcel["polygon"])
    road = _best_road(poly, roads or []) if roads else None
    frame = _frame_for_road(poly, road) if road is not None else None
    if frame is not None and frame["along"] > 1.0 and frame["depth"] > 1.0:
        return {
            "area_gu2": float(poly.area),
            "frontage_gu": float(frame["along"]),
            "depth_gu": float(frame["depth"]),
        }
    by_target: dict[str, float] = defaultdict(float)
    for front in parcel.get("frontages") or []:
        by_target[str(front.get("target_id") or "")] += float(
            front.get("frontage_length_gu") or 0.0)
    primary = max(by_target.values()) if by_target else 1.0
    return {
        "area_gu2": float(poly.area),
        "frontage_gu": max(primary, 1.0),
        "depth_gu": float(poly.area) / max(primary, 1.0),
    }


def _ordered_fronts(parcel: dict, roads_list: list) -> list[dict]:
    poly = polygon_from_ring(parcel["polygon"])
    best = _best_road(poly, roads_list)
    best_id = best["road_id"] if best else None
    fronts = list(parcel.get("frontages") or [])
    fronts.sort(key=lambda f: (
        0 if f.get("target_id") == best_id else 1,
        0 if f.get("target_type") == "alley" else 1,
        -float(f.get("frontage_length_gu") or 0.0),
    ))
    return fronts


def _road_map(candidate: dict) -> dict[str, dict]:
    return {r["road_id"]: r for r in candidate.get("roads") or []}


def _space_map(candidate: dict) -> dict[str, dict]:
    return {s["space_id"]: s for s in candidate.get("open_spaces") or []}


def _target_line(target_id: str, roads: dict, spaces: dict) -> Optional[LineString]:
    road = roads.get(target_id)
    if road is not None:
        geom = road.get("polyline") or []
        if len(geom) >= 2:
            return LineString([(float(p[0]), float(p[1])) for p in geom])
    space = spaces.get(target_id)
    if space is not None:
        ring = space.get("polygon") or []
        if len(ring) >= 3:
            coords = [(float(p[0]), float(p[1])) for p in ring] + [
                (float(ring[0][0]), float(ring[0][1]))]
            return LineString(coords)
    return None


def _inward_normal(sample: tuple[float, float], tangent: tuple[float, float],
                   centroid: tuple[float, float]) -> tuple[float, float]:
    tx, ty = tangent
    nrm = math.hypot(tx, ty) or 1.0
    tx, ty = tx / nrm, ty / nrm
    left = (-ty, tx)
    vx, vy = centroid[0] - sample[0], centroid[1] - sample[1]
    if left[0] * vx + left[1] * vy < 0:
        left = (-left[0], -left[1])
    ln = math.hypot(*left) or 1.0
    return left[0] / ln, left[1] / ln


CURB_TOUCH_GU = FRONTAGE_TOUCH_GU


def _curb_points(parcel_poly: Polygon, line: LineString,
                 start: float, end: float, corridors: list,
                 ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Sample street-front edges. Midpoint must graze the corridor, not a side wall."""
    from shapely.ops import unary_union
    lo, hi = min(start, end), max(start, end)
    road_u = unary_union(corridors) if corridors else None
    pts = []
    coords = list(parcel_poly.exterior.coords)
    for a, b in zip(coords, coords[1:]):
        seg = LineString([a, b])
        if seg.length < 256.0:
            continue
        mid = seg.interpolate(0.5, normalized=True)
        if road_u is not None:
            if mid.distance(road_u) > CURB_TOUCH_GU:
                continue
        elif mid.distance(line) > CURB_TOUCH_GU:
            continue
        tangent = (float(b[0] - a[0]), float(b[1] - a[1]))
        n = max(3, int(seg.length / ALONG_STEP_GU) + 1)
        for i in range(n):
            frac = i / (n - 1)
            if n > 3 and (frac < 0.12 or frac > 0.88):
                continue
            pt = seg.interpolate(seg.length * i / (n - 1))
            s = line.project(pt)
            if s < lo - ALONG_STEP_GU or s > hi + ALONG_STEP_GU:
                continue
            pts.append(((float(pt.x), float(pt.y)), tangent))
    return pts


def _terrain_points(poly: Polygon) -> list[tuple[float, float]]:
    """Deterministic perimeter/centroid/interior samples for terrain gating."""
    coords = list(poly.exterior.coords)[:-1]
    points = [(float(x), float(y)) for x, y in coords]
    for a, b in zip(coords, coords[1:] + coords[:1]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        count = max(1, int(math.ceil(length / 128.0)))
        points.extend((float(a[0] + (b[0] - a[0]) * i / count),
                       float(a[1] + (b[1] - a[1]) * i / count))
                      for i in range(1, count))
    c = poly.centroid
    points.append((float(c.x), float(c.y)))
    minx, miny, maxx, maxy = poly.bounds
    x = minx + 128.0
    while x < maxx - 1e-6:
        y = miny + 128.0
        while y < maxy - 1e-6:
            if poly.covers(Point(x, y)):
                points.append((x, y))
            y += 128.0
        x += 128.0
    return points


def _terrain_ok(hull_poly: Polygon, ctx: SiteContext,
                envelope: Optional[dict]) -> Optional[str]:
    points = _terrain_points(hull_poly)
    samples = ctx.sample_many(np.asarray(points, dtype=np.float64))
    # ``buildable`` is a composite hard mask (water, slope, and out-of-grid),
    # not a water assertion.  Keep the two rejection reasons evidence-based.
    if np.any(samples["water_term"] >= 0.99):
        return "terrain_water"
    if np.any(samples["buildable"] == 0):
        return "terrain_unbuildable"
    env = envelope or {}
    slope_limit = env.get("footprint_slope_deg")
    # A zero cost represents any slope at or below the soft threshold, so use
    # that threshold as its conservative upper bound. Positive costs invert
    # SiteContext's exact linear mapping.
    recovered_slope = SLOPE_SOFT_START_DEG + np.clip(
        samples["slope_cost"], 0.0, 1.0
    ) * (SLOPE_HARD_DEG - SLOPE_SOFT_START_DEG)
    allowed_slope = min(
        SLOPE_HARD_DEG, float(slope_limit) + 5.0
    ) if slope_limit is not None else SLOPE_HARD_DEG
    if np.any(recovered_slope > allowed_slope + 1e-9):
        return "terrain_envelope"
    relief = env.get("footprint_relief_gu")
    if relief is not None and len(samples["elevation_gu"]):
        if float(np.max(samples["elevation_gu"]) - np.min(samples["elevation_gu"])) > float(relief):
            return "terrain_envelope"
    return None


def _unary_ok(hull_poly: Polygon, parcel_poly: Polygon, corridors: list,
              ctx: Optional[SiteContext], terrain_envelope: Optional[dict] = None) -> Optional[str]:
    leftover = hull_poly.difference(parcel_poly)
    if leftover.area > max(HULL_SLOP_GU2, float(hull_poly.area) * 0.002):
        return "hull_outside_parcel"
    if corridors:
        from shapely.ops import unary_union
        hit = hull_poly.intersection(unary_union(corridors))
        if not hit.is_empty and hit.area > HULL_SLOP_GU2:
            return "hull_in_road"
    if ctx is not None:
        reason = _terrain_ok(hull_poly, ctx, terrain_envelope)
        if reason is not None:
            return reason
    return None


def _as_polygon(geom) -> Optional[Polygon]:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon" and geom.area > 1.0:
        return geom
    if geom.geom_type == "MultiPolygon":
        parts = [g for g in geom.geoms
                 if g.geom_type == "Polygon" and g.area > 1.0]
        return max(parts, key=lambda g: g.area) if parts else None
    return None


def _try_pose(
    hull,
    primary,
    door_or_anchor: tuple[float, float],
    yaw: float,
    parcel_poly: Polygon,
    corridors: list,
    ctx: Optional[SiteContext],
    hist: Counter,
    budget: list[int],
    stamp_id: str,
    mode: str,
    terrain_envelope: Optional[dict] = None,
) -> Optional[dict]:
    budget[0] += 1
    if budget[0] > SEARCH_BUDGET:
        return None
    if mode == "door":
        rotated = rot2d_ccw(primary.offset[0], primary.offset[1], yaw)
        anchor = (door_or_anchor[0] - rotated[0], door_or_anchor[1] - rotated[1])
    else:
        hx = float(Polygon(hull).centroid.x)
        hy = float(Polygon(hull).centroid.y)
        rx, ry = rot2d_ccw(hx, hy, yaw)
        anchor = (door_or_anchor[0] - rx, door_or_anchor[1] - ry)
    world = _transform_hull(hull, anchor, yaw)
    hull_poly = Polygon(world)
    if hull_poly.is_empty or hull_poly.area <= 0:
        hist["invalid_hull"] += 1
        return None
    reason = _unary_ok(hull_poly, parcel_poly, corridors, ctx, terrain_envelope)
    if reason is not None:
        hist[reason] += 1
        return None
    return {
        "stamp_id": stamp_id,
        "anchor": [anchor[0], anchor[1]],
        "yaw_deg": float(yaw),
        "hull": [list(p) for p in world],
        "hull_area_gu2": float(hull_poly.area),
    }


def _aabb_fit_pose(
    hull,
    primary,
    frame: dict,
    parcel_poly: Polygon,
    corridors: list,
    ctx: Optional[SiteContext],
    hist: Counter,
    budget: list[int],
    stamp_id: str,
) -> Optional[dict]:
    """Search exact hull translations inside the street-aligned lot frame.

    The inscribed rectangle is sampled at finite resolution and is only a
    cheap translation envelope for organic parcels.  Every candidate still
    goes through ``_try_pose`` so the actual parcel polygon remains the
    authority.
    """
    tx, ty = frame["tangent"]
    nx, ny = frame["inward"]
    ox, oy = frame["origin"]
    # The sampled inscribed rectangle is useful for parcel sizing, but it is
    # not an exact polygon erosion.  Using it as a hard seating envelope can
    # reject a real stamp when one sample column falls just outside an
    # organic edge.  Search the parcel frame here; _try_pose is the exact
    # containment gate.
    desired = math.degrees(math.atan2(-ny, -nx))
    for flip in YAW_FLIPS_DEG:
        yaw = desired - primary.heading_deg + flip
        door_rot = rot2d_ccw(primary.offset[0], primary.offset[1], yaw)
        rel = []
        for hx, hy in hull:
            rx, ry = rot2d_ccw(hx, hy, yaw)
            rel.append((rx - door_rot[0], ry - door_rot[1]))
        ts = [p[0] * tx + p[1] * ty for p in rel]
        ns = [p[0] * nx + p[1] * ny for p in rel]
        t_lo, t_hi = min(ts), max(ts)
        n_lo, n_hi = min(ns), max(ns)
        w, d = t_hi - t_lo, n_hi - n_lo
        if w > frame["along"] - 4.0 or d > frame["depth"] - 4.0:
            continue
        t_start = frame["tmin"] - t_lo
        t_end = frame["tmax"] - t_hi
        n_start = frame["nmin"] - n_lo + 8.0
        n_end = frame["nmax"] - n_hi
        if t_start > t_end or n_start > n_end:
            continue

        def samples(lo: float, hi: float, step: float) -> list[float]:
            if hi - lo <= 1e-6:
                return [lo]
            values = [lo, hi, (lo + hi) * 0.5]
            value = lo + step
            while value < hi - 1e-6:
                values.append(value)
                value += step
            return sorted(set(round(v, 6) for v in values))

        # Prefer the curb and center before probing deeper/inward positions.
        t_values = samples(t_start, t_end, 8.0)
        n_values = samples(n_start, n_end, 8.0)
        n_values.sort(key=lambda value: (abs(value - n_start), value))
        for door_n in n_values:
            for door_t in t_values:
                if budget[0] >= SEARCH_BUDGET:
                    return None
                door_pt = (ox + tx * door_t + nx * door_n,
                           oy + ty * door_t + ny * door_n)
                pose = _try_pose(
                    hull, primary, door_pt, yaw, parcel_poly, corridors, ctx,
                    hist, budget, stamp_id, "door")
                if pose is not None:
                    return pose
    return None


def _generate_for_parcel(
    parcel: dict,
    stamps: list[dict],
    libraries: dict,
    roads: dict,
    spaces: dict,
    corridors: list,
    ctx: Optional[SiteContext],
    budget: list[int],
    ward_type: str,
) -> tuple[list[dict], Counter]:
    hist: Counter = Counter()
    candidates = []
    parcel_poly = polygon_from_ring(parcel["polygon"])
    centroid = (float(parcel_poly.centroid.x), float(parcel_poly.centroid.y))
    core = _as_polygon(parcel_poly.buffer(-SEATING_INSET_GU)) or parcel_poly
    core_pt = (float(core.representative_point().x),
               float(core.representative_point().y))
    fronts = _ordered_fronts(parcel, list(roads.values()))
    parcel_start = budget[0]
    shape = _shape(parcel, list(roads.values()))
    depth_gu = float(shape["depth_gu"])

    def over_budget() -> bool:
        return budget[0] - parcel_start >= PER_PARCEL_BUDGET

    def try_door_at(curb, tangent, slides, gaps, hull, primary, stamp_id,
                    before, stamp_start, candidates):
        tn = math.hypot(*tangent) or 1.0
        ux, uy = tangent[0] / tn, tangent[1] / tn
        normal = _inward_normal(curb, tangent, centroid)
        desired_curb = math.degrees(math.atan2(-normal[1], -normal[0]))
        base_curb = desired_curb - primary.heading_deg
        for slide in slides:
            if stamp_done(before, stamp_start, candidates):
                return
            for gap in gaps:
                if stamp_done(before, stamp_start, candidates):
                    return
                door_pt = (curb[0] + ux * slide + normal[0] * gap,
                           curb[1] + uy * slide + normal[1] * gap)
                for flip in YAW_FLIPS_DEG:
                    if stamp_done(before, stamp_start, candidates):
                        return
                    pose = _try_pose(
                        hull, primary, door_pt, base_curb + flip,
                        parcel_poly, corridors, ctx, hist, budget,
                        stamp_id, "door")
                    if pose is not None:
                        candidates.append(pose)

    def stamp_done(before, stamp_start, candidates) -> bool:
        return (len(candidates) - before >= MAX_POSES_PER_STAMP
                or over_budget()
                or budget[0] - stamp_start >= TRIES_PER_STAMP)

    for row in stamps:
        if len(candidates) >= MAX_POSES_PER_PARCEL or over_budget():
            break
        source = libraries.get(row["stamp_id"])
        if source is None:
            hist["stamp_geometry_unresolved"] += 1
            continue
        try:
            hull = _stamp_hull(source)
            doors = _stamp_doors(source)
        except FrontageFitError:
            hist["stamp_geometry_unresolved"] += 1
            continue
        primary = doors[0]
        before = len(candidates)
        stamp_start = budget[0]
        gaps = _gaps_for(row, depth_gu)
        road_list = list(roads.values())
        pack_road = _best_road(parcel_poly, road_list)
        pack_frame = _frame_for_road(parcel_poly, pack_road) if pack_road else None
        if pack_frame is not None and (parcel.get("frontages") or []):
            pose = _aabb_fit_pose(
                hull, primary, pack_frame, parcel_poly, corridors, ctx, hist,
                budget, row["stamp_id"])
            if pose is not None:
                candidates.append(pose)
            tx, ty = pack_frame["tangent"]
            nx, ny = pack_frame["inward"]
            ox, oy = pack_frame["origin"]
            for t_frac in (0.5, 0.38, 0.62):
                if stamp_done(before, stamp_start, candidates):
                    break
                t = pack_frame["tmin"] + pack_frame["along"] * t_frac
                for gap in gaps:
                    if stamp_done(before, stamp_start, candidates):
                        break
                    curb = (ox + tx * t + nx * pack_frame["nmin"],
                            oy + ty * t + ny * pack_frame["nmin"])
                    tangent = (tx, ty)
                    try_door_at(curb, tangent, (0.0,), (gap,), hull, primary,
                                row["stamp_id"], before, stamp_start, candidates)
        curb_samples = []
        for front in fronts:
            if stamp_done(before, stamp_start, candidates):
                break
            line = _target_line(front["target_id"], roads, spaces)
            if line is None or line.length <= 0:
                hist["target_unresolved"] += 1
                continue
            start = float(front["target_arc_start_gu"])
            end = float(front["target_arc_end_gu"])
            for curb, tangent in _curb_points(
                    parcel_poly, line, start, end, corridors):
                if math.hypot(*tangent) < 1e-6:
                    continue
                curb_samples.append((curb, tangent))
                try_door_at(curb, tangent, (0.0,), gaps, hull, primary,
                            row["stamp_id"], before, stamp_start, candidates)
                if stamp_done(before, stamp_start, candidates):
                    break
            if over_budget():
                return candidates, hist
        if len(candidates) == before and not stamp_done(before, stamp_start, candidates):
            extra = tuple(s for s in ALONG_SLIDES if s != 0.0)
            for curb, tangent in curb_samples:
                try_door_at(curb, tangent, extra, gaps, hull, primary,
                            row["stamp_id"], before, stamp_start, candidates)
                if stamp_done(before, stamp_start, candidates):
                    break
        if len(candidates) > before:
            continue
        if ward_type in CORE_WARDS:
            continue
        stamp_start = budget[0]
        interior_line = None
        for front in fronts:
            trial = _target_line(front["target_id"], roads, spaces)
            if trial is not None and trial.length > 0:
                interior_line = trial
                break
        if interior_line is not None:
            nearest = interior_line.interpolate(
                interior_line.project(Point(core_pt[0], core_pt[1])))
            outward = (float(nearest.x) - core_pt[0], float(nearest.y) - core_pt[1])
            nrm = math.hypot(*outward) or 1.0
            desired = math.degrees(math.atan2(outward[1] / nrm, outward[0] / nrm))
            base_yaw = desired - primary.heading_deg
            for flip in YAW_FLIPS_DEG:
                if stamp_done(before, stamp_start, candidates):
                    break
                for pert in YAW_PERTURB_DEG:
                    if stamp_done(before, stamp_start, candidates):
                        break
                    pose = _try_pose(
                        hull, primary, core_pt, base_yaw + flip + pert,
                        parcel_poly, corridors, ctx, hist, budget,
                        row["stamp_id"], "centroid")
                    if pose is not None:
                        candidates.append(pose)
            if over_budget():
                return candidates, hist
    candidates.sort(key=lambda c: (c["stamp_id"], c["yaw_deg"],
                                   c["anchor"][0], c["anchor"][1]))
    return candidates, hist


def place_stamps(
    candidate: dict,
    stamp_index: dict,
    libraries: dict,
    *,
    ctx: Optional[SiteContext] = None,
    candidate_id: str = "c00",
) -> dict[str, Any]:
    """Assign stamps to parcels. Exhaustion is inconclusive, not a crash."""
    from shapely.ops import unary_union

    roads = _road_map(candidate)
    spaces = _space_map(candidate)
    corridors = []
    for road in candidate.get("roads") or []:
        geom = road.get("polyline") or []
        if len(geom) < 2:
            continue
        width = float(road.get("clear_width_gu") or ALLEY_CLEAR_WIDTH_GU)
        line = LineString([(float(p[0]), float(p[1])) for p in geom])
        if line.length > 0:
            corridors.append(line.buffer(width / 2.0, cap_style=2, join_style=2))
    wards = candidate.get("wards") or []
    budget = [0]
    domains: dict[str, list[dict]] = {}
    histograms: dict[str, dict] = {}
    for parcel in candidate.get("parcels") or []:
        wtype = _ward_type(parcel, wards)
        roles = parcel.get("allowed_roles") or None
        stamps = list_compatible_stamps(
            stamp_index, _shape(parcel, candidate.get("roads") or []), wtype, roles)
        by_area = sorted(stamps, key=lambda r: (r["hull_area_gu2"], r["stamp_id"]))
        if len(by_area) <= 20:
            ordered = by_area
        else:
            seen = set()
            ordered = []
            for row in by_area[:10] + by_area[len(by_area)//3:len(by_area)//3 + 6] + by_area[-6:]:
                if row["stamp_id"] not in seen:
                    ordered.append(row)
                    seen.add(row["stamp_id"])
        if budget[0] > SEARCH_BUDGET:
            domains[parcel["parcel_id"]] = []
            histograms[parcel["parcel_id"]] = {"inconclusive": 1}
            continue
        cands, hist = _generate_for_parcel(
            parcel, ordered, libraries, roads, spaces, corridors, ctx, budget,
            wtype)
        domains[parcel["parcel_id"]] = cands
        histograms[parcel["parcel_id"]] = dict(hist)

    remaining = {pid: list(cs) for pid, cs in domains.items()}
    placed_hulls: list[list] = []
    hull_by_parcel: dict[str, list] = {}
    used = Counter()
    placements = []
    order = sorted(remaining, key=lambda pid: (-len(remaining[pid]), pid))
    assigned: dict[str, Optional[dict]] = {}
    for pid in order:
        parcel = next(p for p in candidate["parcels"] if p["parcel_id"] == pid)
        choices = []
        for cand in remaining[pid]:
            if used[cand["stamp_id"]] >= MAX_IDENTICAL:
                continue
            if any(_rings_conflict(cand["hull"], other) for other in placed_hulls):
                continue
            choices.append(cand)
        if not choices:
            assigned[pid] = None
            placements.append({
                "parcel_id": pid,
                "stamp_id": None,
                "anchor": None,
                "yaw_deg": None,
            })
            if parcel.get("required_occupancy") and not remaining[pid]:
                histograms[pid]["unsatisfiable_domain"] = (
                    histograms[pid].get("unsatisfiable_domain", 0) + 1)
            elif parcel.get("required_occupancy"):
                histograms[pid]["inconclusive"] = (
                    histograms[pid].get("inconclusive", 0) + 1)
            continue
        choices.sort(key=lambda c: (used[c["stamp_id"]],
                                    -float(c.get("hull_area_gu2") or 0.0),
                                    c["stamp_id"], c["yaw_deg"]))
        pick = choices[0]
        assigned[pid] = pick
        placed_hulls.append(pick["hull"])
        hull_by_parcel[pid] = pick["hull"]
        used[pick["stamp_id"]] += 1
        placements.append({
            "parcel_id": pid,
            "stamp_id": pick["stamp_id"],
            "anchor": pick["anchor"],
            "yaw_deg": pick["yaw_deg"],
        })
    placements.sort(key=lambda p: p["parcel_id"])
    n_ok = sum(1 for p in placements if p["stamp_id"] is not None)
    n_req = sum(1 for p in candidate.get("parcels") or []
                if p.get("required_occupancy"))
    status = "ok" if n_ok >= max(1, int(0.5 * n_req)) else "repaired"
    reports = list(candidate.get("reports") or [])
    reports.append({
        "stage": "placement",
        "status": status,
        "message": (
            f"placed={n_ok}/{len(placements)} budget={budget[0]} "
            f"identical_cap={MAX_IDENTICAL}"
        ),
    })
    out = dict(candidate)
    out["placements"] = placements
    out["placement_hulls"] = hull_by_parcel
    out["placement_histograms"] = histograms
    out["reports"] = reports
    _ = unary_union
    return out


def write_placement_diagnostic(
    ctx: SiteContext,
    product: dict,
    *,
    topdown_path: Path,
    survey: dict,
    out_png: Path,
    full_site: bool = False,
) -> None:
    from PIL import Image, ImageDraw

    image, mapping = diagnostic_view({"_diagnostic_bounds": [product.get("city_domain") or []]}, topdown_path, survey, full_site=full_site)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    def to_px(pt):
        return _plan_to_px(float(pt[0]), float(pt[1]), mapping)

    for parcel in product.get("parcels") or []:
        ring = parcel.get("polygon") or []
        if len(ring) >= 3:
            draw.polygon([to_px(p) for p in ring],
                         fill=(40, 90, 50, 70), outline=(20, 50, 25, 220))
    for road in product.get("roads") or []:
        geom = road.get("polyline") or []
        if len(geom) < 2:
            continue
        hier = road.get("hierarchy") or "street"
        color = {
            "arterial": (200, 40, 40, 110),
            "street": (230, 200, 50, 110),
            "lane": (160, 140, 70, 90),
            "alley": (120, 80, 40, 110),
            "regional_approach": (180, 80, 40, 110),
        }.get(hier, (200, 200, 200, 90))
        clear = float(road.get("clear_width_gu") or 256.0)
        line = LineString([(float(p[0]), float(p[1])) for p in geom])
        if line.length <= 0:
            continue
        buf = line.buffer(clear / 2.0, cap_style=2, join_style=2)
        geoms = [buf] if buf.geom_type == "Polygon" else list(getattr(buf, "geoms", []))
        for g in geoms:
            if g.geom_type != "Polygon" or g.area <= 0:
                continue
            draw.polygon([to_px(p) for p in g.exterior.coords], fill=color)
    # R11 carries designed centerlines before R12 materializes their exact
    # surface records. Draw them at real width so route shape can be judged
    # before any building placement is accepted as coherent.
    if product.get("stage_id") == "r11_alley_infill":
        for alley in product.get("alleys") or []:
            geom = alley.get("polyline") or []
            if len(geom) < 2:
                continue
            line = LineString(geom)
            role = alley.get("role")
            color = ((193, 147, 64, 190) if role == "plaza_mouth"
                     else (125, 82, 42, 205))
            surface = line.buffer(float(alley.get("clear_width_gu") or 224.0) / 2.0,
                                  cap_style=1, join_style=1)
            geoms = [surface] if surface.geom_type == "Polygon" else list(surface.geoms)
            for piece in geoms:
                if piece.geom_type == "Polygon":
                    draw.polygon([to_px(p) for p in piece.exterior.coords], fill=color)
            draw.line([to_px(p) for p in geom], fill=(70, 38, 18, 255), width=4)
    # R12 surfaces are rendered at their planning widths.  They are drawn
    # before hulls so the viewer can judge continuity and whether buildings
    # actually enclose the space rather than reading a debug fill.
    for surface in product.get("surfaces") or product.get("circulation_surfaces") or []:
        colors = {"alley": (125, 82, 42, 205), "plaza": (193, 147, 64, 190),
                  "front_courtyard": (166, 116, 58, 180), "back_court": (145, 95, 48, 175)}
        color = colors.get(surface.get("role"), (145, 100, 50, 170))
        for ring in surface.get("polygon") or []:
            if len(ring) >= 3:
                draw.polygon([to_px(p) for p in ring], fill=color, outline=(85, 55, 25, 230))
        if surface.get("centerline") and len(surface["centerline"]) >= 2:
            draw.line([to_px(p) for p in surface["centerline"]], fill=(70, 38, 18, 255), width=5)
    # R10-only candidate outlines make rejected geometry inspectable without
    # turning the final R13 render into a residual-space debug map.
    if product.get("stage_id") == "r10_spatial_roles":
        for candidate in product.get("spatial_role_candidates") or []:
            ring = candidate.get("polygon") or []
            if len(ring) >= 3:
                col = (50, 235, 110, 230) if candidate.get("status") == "accepted" else (235, 70, 70, 200)
                draw.line([to_px(p) for p in ring + [ring[0]]], fill=col, width=4)
                cx, cy = to_px(candidate.get("center", ring[0]))
                draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=col)
    for apron in product.get("rear_aprons") or []:
        for ring in apron.get("polygon") or []:
            if len(ring) >= 3:
                draw.polygon([to_px(p) for p in ring], fill=(224, 171, 76, 210), outline=(110, 70, 25, 240))
    for apron in product.get("door_aprons") or []:
        for ring in apron.get("polygon") or []:
            if len(ring) >= 3:
                draw.polygon([to_px(p) for p in ring], fill=(160, 105, 50, 220),
                             outline=(90, 55, 25, 240))
    for _pid, hull in (product.get("placement_hulls") or {}).items():
        if not hull:
            continue
        place = next((p for p in product.get("placements", []) if p.get("parcel_id") == _pid), {})
        if place.get("family") == "stone":
            color = (150, 150, 165, 210)
        elif place.get("family") in ("wood", "outskirts"):
            color = (155, 100, 45, 205)
        else:
            color = {"frontage": (45, 115, 220, 175), "rear": (125, 60, 185, 175),
                     "backs_to_wall": (150, 75, 35, 190)}.get(place.get("mode"), (40, 80, 180, 150))
        draw.polygon([to_px(p) for p in hull], fill=color, outline=(10, 20, 80, 255))
    for place in product.get("placements") or []:
        if place.get("stamp_id") is None or place.get("anchor") is None:
            continue
        px, py = to_px(place["anchor"])
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(220, 220, 40, 255))
        door = place.get("door_world")
        tick = place.get("outward_tick")
        if door and tick:
            x, y = to_px(door)
            ex, ey = to_px((door[0] + tick[0] * 192.0, door[1] + tick[1] * 192.0))
            draw.line((x, y, ex, ey), fill=(255, 245, 80, 255), width=2)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255, 255))
    for door in product.get("doors") or []:
        x, y = to_px(door["position"])
        col = (255, 255, 255, 255) if door.get("role") == "primary" else (255, 150, 40, 255)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
    for water in product.get("water_polygons") or []:
        draw.polygon([to_px(p) for p in water], fill=(30, 150, 210, 130), outline=(20, 80, 140, 220))
    for strip in (product.get("wall") or {}).get("strips", []):
        ring = strip.get("polygon") or []
        if len(ring) >= 3:
            col = (35, 95, 220, 110) if strip.get("mode") == "wall_lane" else (120, 65, 45, 125)
            draw.polygon([to_px(p) for p in ring], fill=col, outline=col)
    wall_ring = (product.get("wall") or {}).get("planning_polygon") or []
    if len(wall_ring) >= 3:
        draw.line([to_px(p) for p in wall_ring + [wall_ring[0]]],
                  fill=(245, 235, 180, 255), width=3)
    for court in product.get("courtyards") or []:
        ring = court.get("polygon") or []
        if len(ring) >= 3:
            draw.polygon([to_px(p) for p in ring], fill=(70, 180, 105, 45),
                         outline=(80, 235, 125, 210))
    for path in product.get("reserved_access_paths") or []:
        geom = path.get("geometry") or []
        if len(geom) >= 2:
            draw.line([to_px(p) for p in geom], fill=(40, 245, 245, 255), width=3)
    # Compact legend is intentionally part of the source-resolution render.
    legend = [("roads", (230, 200, 50, 230)), ("alley", (125, 82, 42, 230)),
              ("plaza", (193, 147, 64, 230)), ("primary door", (255, 255, 255, 255)),
              ("secondary door", (255, 150, 40, 255))]
    lx, ly = 18, 18
    draw.rectangle((lx - 8, ly - 8, lx + 170, ly + 22 * len(legend) + 6), fill=(20, 25, 30, 185))
    for i, (label, col) in enumerate(legend):
        yy = ly + i * 22
        draw.rectangle((lx, yy, lx + 14, yy + 14), fill=col)
        draw.text((lx + 22, yy - 2), label, fill=(245, 245, 235, 255))
    for mouth in product.get("access_mouths") or []:
        if mouth.get("position"):
            x, y = to_px(mouth["position"])
            draw.ellipse([x - 5, y - 5, x + 5, y + 5],
                         fill=(255, 80, 215, 255))
    Image.alpha_composite(image, overlay).save(out_png)
    _ = ctx
