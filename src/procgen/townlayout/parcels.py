"""Ward-specific parcel and alley subdivision for V2 townlayout (Phase 16).

Purpose
-------
Slice buildable blocks along the street: width cuts when a façade is two
houses wide, depth cuts with an explicit alley when two houses deep.
Leaves are one OBB frontage×depth, not a yard-area multiple.  Alleys are
road edges.  Protected plazas/courts/parks are not overwritten.

Inputs
------
Phase 14 candidate, TownBrief, stamp capability index, p50 hull area.

Outputs
-------
``parcels``, extra alley ``roads``, extra verge ``open_spaces``.

Pipeline position
-----------------
V2 townlayout Phase 16 parcels; no stamp seating/VTEX.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split as shp_split
from shapely.ops import unary_union

from .constants import (
    ALLEY_CLEAR_WIDTH_GU,
    FRONTAGE_TOUCH_GU,
    MIN_PARCEL_FRONTAGE_GU,
    MIN_VERGE_AREA_GU2,
    PACK_SLACK_OUTSKIRTS,
    STAMP_FILL_MAX,
)
from .geometry import normalize_ring, polygon_from_ring
from .openspaces import protected_open_space_ids
from .rng import stage_rng
from .site_context import SiteContext, _plan_to_px, diagnostic_view
from .stamp_index import (
    CORE_WARDS,
    WARD_BUILDING_TYPES,
    is_outskirts_only,
    kit_family,
    pack_slack,
)
from .validate import TownLayoutError

MAX_DEPTH = 10
MAX_SPLIT_ATTEMPTS = 6
FRONTAGE_EPS_GU = FRONTAGE_TOUCH_GU
MIN_FRONTAGE_GU = MIN_PARCEL_FRONTAGE_GU
LOT_FIT = 1.08
SIT_MARGIN = 1.05

WARD_GRAMMAR = {
    "craft": {"area_factor": 0.9, "gridChaos": 0.35, "sizeChaos": 0.4,
              "emptyProb": 0.0, "alleyProb": 0.0},
    "market": {"area_factor": 1.1, "gridChaos": 0.3, "sizeChaos": 0.35,
               "emptyProb": 0.04, "alleyProb": 0.0},
    "residential": {"area_factor": 1.0, "gridChaos": 0.35, "sizeChaos": 0.4,
                    "emptyProb": 0.0, "alleyProb": 0.0},
    "outskirts": {"area_factor": 1.6, "gridChaos": 0.5, "sizeChaos": 0.55,
                  "emptyProb": 0.15, "alleyProb": 0.2},
    "keep": {"area_factor": 2.5, "gridChaos": 0.15, "sizeChaos": 0.2,
             "emptyProb": 0.1, "alleyProb": 0.0},
}


def _ring(poly: Polygon) -> list[list[float]]:
    return normalize_ring([[c[0], c[1]] for c in poly.exterior.coords])["ring"]


def _iq(poly: Polygon) -> float:
    if poly.is_empty or poly.area <= 0 or poly.length <= 1e-9:
        return 0.0
    return float(4.0 * math.pi * poly.area / (poly.length ** 2))


def _parts(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "GeometryCollection":
        parts = []
        for item in geom.geoms:
            parts.extend(_parts(item))
        return parts
    if geom.geom_type == "Polygon":
        return [geom] if geom.area > 1.0 and len(geom.interiors) == 0 else []
    if geom.geom_type == "MultiPolygon":
        return [g for g in geom.geoms
                if g.geom_type == "Polygon" and g.area > 1.0
                and len(g.interiors) == 0]
    return []


def _corridors(roads: list[dict]) -> list:
    out = []
    for road in roads:
        geom = road.get("polyline") or []
        if len(geom) < 2:
            continue
        width = float(road.get("clear_width_gu") or 0.0)
        if width <= 0:
            continue
        line = LineString([(float(p[0]), float(p[1])) for p in geom])
        if line.length <= 0:
            continue
        out.append(line.buffer(width / 2.0, cap_style=2, join_style=2))
    return out


def _frontage_length(poly: Polygon, corridors: list) -> float:
    if not corridors:
        return 0.0
    union = unary_union(corridors)
    total = 0.0
    coords = list(poly.exterior.coords)
    for a, b in zip(coords, coords[1:]):
        seg = LineString([a, b])
        if seg.length <= 0:
            continue
        if seg.distance(union) <= FRONTAGE_EPS_GU:
            total += seg.length
    return float(total)


def _as_line(road: dict) -> Optional[LineString]:
    geom = road.get("polyline") or []
    if len(geom) < 2:
        return None
    line = LineString([(float(p[0]), float(p[1])) for p in geom])
    return line if line.length > 0 else None


def _contact_length(poly: Polygon, road: dict) -> float:
    line = _as_line(road)
    if line is None:
        return 0.0
    width = float(road.get("clear_width_gu") or ALLEY_CLEAR_WIDTH_GU)
    buf = line.buffer(width / 2.0, cap_style=2, join_style=2)
    total = 0.0
    coords = list(poly.exterior.coords)
    for a, b in zip(coords, coords[1:]):
        seg = LineString([a, b])
        if seg.length > 0 and seg.distance(buf) <= FRONTAGE_EPS_GU:
            total += seg.length
    return float(total)


def _best_road(poly: Polygon, roads: list[dict]) -> Optional[dict]:
    scored = []
    for road in roads:
        contact = _contact_length(poly, road)
        if contact <= 0:
            continue
        hier = road.get("hierarchy") or "street"
        prefer = 0 if hier == "alley" else 1
        scored.append((-contact, prefer, road["road_id"], road))
    if not scored:
        return None
    scored.sort()
    return scored[0][3]


def _frame_for_road(poly: Polygon, road: dict) -> Optional[dict]:
    """Along-street and inward-depth extents of a lot against one road."""
    line = _as_line(road)
    if line is None:
        return None
    c = poly.centroid
    s = min(max(line.project(c), 1.0), max(line.length - 1.0, 1.0))
    p0 = line.interpolate(s)
    p1 = line.interpolate(min(s + 32.0, line.length))
    tx, ty = float(p1.x - p0.x), float(p1.y - p0.y)
    nrm = math.hypot(tx, ty) or 1.0
    tx, ty = tx / nrm, ty / nrm
    nx, ny = -ty, tx
    vx, vy = float(c.x - p0.x), float(c.y - p0.y)
    if nx * vx + ny * vy < 0:
        nx, ny = -nx, -ny
    tvals = []
    nvals = []
    for x, y in poly.exterior.coords:
        dx, dy = float(x) - float(p0.x), float(y) - float(p0.y)
        tvals.append(dx * tx + dy * ty)
        nvals.append(dx * nx + dy * ny)
    return {
        "road": road,
        "line": line,
        "tangent": (tx, ty),
        "inward": (nx, ny),
        "origin": (float(p0.x), float(p0.y)),
        "tmin": min(tvals),
        "tmax": max(tvals),
        "nmin": min(nvals),
        "nmax": max(nvals),
        "along": float(max(tvals) - min(tvals)),
        "depth": float(max(nvals) - min(nvals)),
        "contact": _contact_length(poly, road),
    }


def _inscribed_bounds(poly: Polygon, frame: dict, step: float = 24.0) -> Optional[dict]:
    """Largest street-aligned rectangle that stays inside the lot."""
    tx, ty = frame["tangent"]
    nx, ny = frame["inward"]
    ox, oy = frame["origin"]
    tmin, tmax = frame["tmin"], frame["tmax"]
    nmin, nmax = frame["nmin"], frame["nmax"]
    if tmax - tmin < 64.0 or nmax - nmin < 64.0:
        return None
    cols = []
    t = tmin + step * 0.5
    while t <= tmax - step * 0.5:
        inside = []
        n = nmin + step * 0.5
        while n <= nmax - step * 0.5:
            pt = Point(ox + tx * t + nx * n, oy + ty * t + ny * n)
            if poly.covers(pt):
                inside.append(n)
            n += step
        if inside:
            n_lo, n_hi = min(inside), max(inside)
            if n_lo <= nmin + step * 3.0:
                cols.append((t, n_lo, n_hi))
        t += step
    if len(cols) < 2:
        return None
    best = None
    best_area = 0.0
    for i, (ti, lo_i, hi_i) in enumerate(cols):
        lo, hi = lo_i, hi_i
        for j in range(i, len(cols)):
            lo = max(lo, cols[j][1])
            hi = min(hi, cols[j][2])
            if hi - lo < 64.0 or lo > nmin + step * 3.0:
                break
            along = cols[j][0] - ti + step
            depth = hi - lo
            area = along * depth
            if area > best_area:
                best_area = area
                best = {
                    "t0": ti,
                    "t1": cols[j][0],
                    "n0": lo,
                    "n1": hi,
                    "along": float(along),
                    "depth": float(depth),
                }
    return best


def _usable_frame(poly: Polygon, road: dict) -> Optional[dict]:
    frame = _frame_for_road(poly, road)
    if frame is None:
        return None
    ins = _inscribed_bounds(poly, frame)
    if ins is None:
        return frame
    out = dict(frame)
    out["tmin"] = ins["t0"]
    out["tmax"] = ins["t1"]
    out["nmin"] = ins["n0"]
    out["nmax"] = ins["n1"]
    out["along"] = ins["along"]
    out["depth"] = ins["depth"]
    out["inscribed"] = ins
    return out


def _rect_from_frame(frame: dict) -> Optional[Polygon]:
    ins = frame.get("inscribed")
    if not ins:
        return None
    tx, ty = frame["tangent"]
    nx, ny = frame["inward"]
    ox, oy = frame["origin"]
    t0, t1 = ins["t0"], ins["t1"]
    n0, n1 = ins["n0"], ins["n1"]
    corners = [(t0, n0), (t1, n0), (t1, n1), (t0, n1)]
    ring = [(ox + tx * t + nx * n, oy + ty * t + ny * n) for t, n in corners]
    poly = Polygon(ring)
    if poly.is_empty or poly.area < 1.0:
        return None
    if poly.exterior.is_ccw is False:
        poly = Polygon(list(poly.exterior.coords)[::-1])
    return poly


def _core_rows(index: dict, roles: list[str], ward_type: str) -> list[dict]:
    rows = []
    for row in index.get("stamps") or []:
        if row.get("building_type") not in roles:
            continue
        if ward_type != "outskirts" and is_outskirts_only(row):
            continue
        rows.append(row)
    return rows or [s for s in (index.get("stamps") or [])
                    if s.get("building_type") in roles]


def _min_house(index: dict, roles: list[str], ward_type: str) -> tuple[float, float]:
    """Paired width×depth of the smallest-area real stamp, not independent mins."""
    rows = _core_rows(index, roles, ward_type)
    if not rows:
        return 800.0, 800.0
    row = min(rows, key=lambda r: (
        float(r["obb_width_gu"]) * float(r["obb_depth_gu"]), r["stamp_id"]))
    return float(row["obb_width_gu"]), float(row["obb_depth_gu"])


def _stamp_fits(row: dict, along: float, depth: float,
                area: Optional[float] = None) -> bool:
    if float(row["obb_width_gu"]) * SIT_MARGIN > along:
        return False
    if float(row["obb_depth_gu"]) * SIT_MARGIN > depth:
        return False
    if area is not None and float(row["hull_area_gu2"]) > area * STAMP_FILL_MAX:
        return False
    return True


def _any_fits(index: dict, roles: list[str], ward_type: str,
              along: float, depth: float, area: Optional[float] = None) -> bool:
    return any(_stamp_fits(row, along, depth, area)
               for row in _core_rows(index, roles, ward_type))


def _fitting_rows(index: dict, roles: list[str], ward_type: str,
                  along: Optional[float] = None,
                  depth: Optional[float] = None) -> list[dict]:
    rows = _core_rows(index, roles, ward_type)
    fit = []
    for row in rows:
        a = along if along is not None else 1.0e9
        d = depth if depth is not None else 1.0e9
        if _stamp_fits(row, a, d):
            fit.append(row)
    return fit


def _ward_of(pid: str, wards: list[dict]) -> tuple[str, str]:
    for ward in wards:
        if pid in ward.get("patch_ids", []):
            return ward["ward_id"], ward["ward_type"]
    return "ward_unknown", "residential"


def _min_width(poly: Polygon) -> float:
    if poly.is_empty or poly.length <= 1e-9:
        return 0.0
    return float(2.0 * poly.area / poly.length)


def _cut(poly: Polygon, p1, direction) -> list[Polygon]:
    dx, dy = direction
    nrm = math.hypot(dx, dy) or 1.0
    dx, dy = dx / nrm, dy / nrm
    cx, cy = float(poly.centroid.x), float(poly.centroid.y)
    vx, vy = cx - p1[0], cy - p1[1]
    vn = math.hypot(vx, vy) or 1.0
    origin = (p1[0] + vx / vn * 8.0, p1[1] + vy / vn * 8.0)
    span = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1], 1.0) * 4.0
    line = LineString([
        (origin[0] - dx * span, origin[1] - dy * span),
        (origin[0] + dx * span, origin[1] + dy * span),
    ])
    pieces = _parts(shp_split(poly, line))
    pieces.sort(key=lambda g: g.area, reverse=True)
    return pieces


def _insert_alley(poly: Polygon, p1, cut_dir) -> Optional[dict]:
    nrm = math.hypot(*cut_dir) or 1.0
    span = max(poly.bounds[2] - poly.bounds[0],
               poly.bounds[3] - poly.bounds[1], 1.0) * 4.0
    line = LineString([
        (p1[0] - cut_dir[0] / nrm * span, p1[1] - cut_dir[1] / nrm * span),
        (p1[0] + cut_dir[0] / nrm * span, p1[1] + cut_dir[1] / nrm * span),
    ]).intersection(poly)
    if line.is_empty:
        return None
    if line.geom_type == "MultiLineString":
        line = max(line.geoms, key=lambda g: g.length)
    if line.geom_type != "LineString" or line.length < MIN_FRONTAGE_GU:
        return None
    buf = line.buffer(ALLEY_CLEAR_WIDTH_GU / 2.0, cap_style=2, join_style=2)
    children = _parts(poly.difference(buf))
    if len(children) < 2:
        return None
    children.sort(key=lambda g: g.area, reverse=True)
    return {
        "children": children,
        "alley": {
            "polyline": [[c[0], c[1]] for c in line.coords],
            "line": line,
        },
    }


def _try_width_split(poly: Polygon, rng, index: dict, roles: list[str],
                     ward_type: str, frame: dict) -> Optional[list[Polygon]]:
    along = frame["along"]
    depth = frame["depth"]
    rows = _fitting_rows(index, roles, ward_type, along=along, depth=depth)
    if len(rows) < 1:
        return None
    order = sorted(rows, key=lambda r: (
        float(r["obb_width_gu"]) * float(r["obb_depth_gu"]), r["stamp_id"]))
    tx, ty = frame["tangent"]
    nx, ny = frame["inward"]
    ox, oy = frame["origin"]
    viable = []
    for row in order:
        slice_w = float(row["obb_width_gu"]) * pack_slack(row)
        leftover = along - slice_w
        if leftover < 1.0:
            continue
        if not _any_fits(index, roles, ward_type, slice_w, depth):
            continue
        if not _any_fits(index, roles, ward_type, leftover, depth):
            continue
        viable.append(slice_w)
    if not viable:
        return None
    if rng.random() < 0.65:
        slice_w = viable[0]
    else:
        slice_w = viable[int(rng.random() * min(6, len(viable)))]
    split_t = frame["tmin"] + slice_w
    if rng.random() < 0.5:
        split_t = frame["tmax"] - slice_w
    p1 = (ox + tx * split_t, oy + ty * split_t)
    pieces = _cut(poly, p1, (nx, ny))
    if len(pieces) < 2:
        return None
    return pieces[:2]


def _try_depth_split(poly: Polygon, rng, index: dict, roles: list[str],
                     ward_type: str, frame: dict) -> Optional[dict]:
    """Front row one stamp deep; alley; back row faces the alley."""
    along = frame["along"]
    depth = frame["depth"]
    rows = _fitting_rows(index, roles, ward_type, along=along, depth=depth)
    if not rows:
        return None
    order = sorted(rows, key=lambda r: (
        float(r["obb_width_gu"]) * float(r["obb_depth_gu"]), r["stamp_id"]))
    tx, ty = frame["tangent"]
    nx, ny = frame["inward"]
    ox, oy = frame["origin"]
    for row in order:
        cut_d = float(row["obb_depth_gu"]) * pack_slack(row)
        leftover = depth - cut_d - ALLEY_CLEAR_WIDTH_GU
        if leftover < 1.0:
            continue
        if not _any_fits(index, roles, ward_type, along, cut_d):
            continue
        if not _any_fits(index, roles, ward_type, along, leftover):
            continue
        n_cut = frame["nmin"] + cut_d + ALLEY_CLEAR_WIDTH_GU / 2.0
        p1 = (ox + nx * n_cut, oy + ny * n_cut)
        trial = _insert_alley(poly, p1, (tx, ty))
        if trial is not None:
            return trial
    return None

def subdivide_parcels(
    candidate: dict,
    town_brief: dict,
    stamp_index: dict,
    *,
    p50: float,
    candidate_id: str = "c00",
    master_seed: Optional[int] = None,
) -> dict[str, Any]:
    """Bisect inner blocks into parcels and explicit alleys."""
    seed = int(master_seed if master_seed is not None else town_brief["master_seed"])
    roads = list(candidate.get("roads") or [])
    corridors = _corridors(roads)
    protected = set(candidate.get("protected_space_ids")
                    or protected_open_space_ids(candidate))
    protected_polys = [
        polygon_from_ring(s["polygon"])
        for s in (candidate.get("open_spaces") or [])
        if s.get("space_id") in protected and len(s.get("polygon") or []) >= 3
    ]
    wards = candidate.get("wards") or []
    parcels: list[dict] = []
    verges: list[dict] = []
    alleys: list[dict] = []
    pack_debug: list[str] = []
    verified_rects: list[Polygon] = []
    seq = {"parcel": 0, "alley": 0, "verge": 0, "node": 0}
    emit_fail = Counter({
        "clip": 0,
        "no_front": 0,
        "plot": 0,
        "verified_discarded_road": 0,
        "verified_discarded_overlap": 0,
        "small_guard": 0,
        "no_stamp_fits": 0,
        "usable_empty": 0,
        "no_fragment": 0,
        "no_frag_frame": 0,
        "clamp_too_small": 0,
        "search_exhausted": 0,
        "fallback_failed": 0,
    })

    def clip_roads(poly: Polygon) -> Optional[Polygon]:
        live = _corridors(roads + alleys)
        remaining = poly
        if live:
            # Extra 2 GU so split leftovers cannot graze the corridor.
            remaining = poly.difference(
                unary_union(live).buffer(2.0, cap_style=2, join_style=2))
        parts = _parts(remaining)
        if not parts:
            return None
        parts.sort(key=lambda g: g.area, reverse=True)
        return parts[0]

    def usable_union(poly: Polygon):
        live = _corridors(roads + alleys)
        if live:
            poly = poly.difference(
                unary_union(live).buffer(2.0, cap_style=2, join_style=2))
        return poly

    def emit_verge(poly: Polygon) -> None:
        if poly.area < MIN_VERGE_AREA_GU2:
            return
        if _min_width(poly) < 32.0:
            return
        verges.append({
            "space_id": f"space_{candidate_id}_parcel_verge_{seq['verge']:04d}",
            "kind": "verge",
            "polygon": _ring(poly),
        })
        seq["verge"] += 1

    def emit_parcel(poly: Polygon, ward_id: str, ward_type: str,
                    roles: list[str], required: bool, frontage: float,
                    hits: list[dict], *, verified: bool = False,
                    verified_frame: Optional[dict] = None,
                    frontage_road_id: Optional[str] = None,
                    feasible_stamp_ids: Optional[list[str]] = None,
                    intended_family: Optional[str] = None) -> None:
        cleaned = poly if verified else clip_roads(poly)
        if cleaned is None:
            emit_fail["clip"] += 1
            emit_verge(poly)
            return
        poly = cleaned
        frontage = _frontage_length(poly, _corridors(roads + alleys))
        hits = road_hits(poly)
        frontages = []
        for hit in hits:
            frontages.append({
                "target_id": hit["road_id"],
                "target_type": hit["target_type"],
                "target_arc_start_gu": 0.0,
                "target_arc_end_gu": hit["length"],
                "frontage_length_gu": hit["length"],
            })
        if not frontages:
            emit_fail["no_front"] += 1
            emit_verge(poly)
            return
        parcel = {
            "parcel_id": f"parcel_{candidate_id}_{seq['parcel']:04d}",
            "ward_id": ward_id,
            "polygon": _ring(poly),
            "frontages": frontages,
            "required_occupancy": required,
            "allowed_roles": list(roles),
        }
        if verified:
            parcel["verified"] = True
            parcel["frame"] = verified_frame
            parcel["frontage_road_id"] = frontage_road_id
            parcel["feasible_stamp_ids"] = list(feasible_stamp_ids or [])
            parcel["intended_family"] = intended_family or "wood"
        parcels.append(parcel)
        seq["parcel"] += 1

    def road_hits(poly: Polygon) -> list[dict]:
        hits = []
        coords = list(poly.exterior.coords)
        for road in roads + alleys:
            geom = road.get("polyline") or []
            if len(geom) < 2:
                continue
            width = float(road.get("clear_width_gu") or ALLEY_CLEAR_WIDTH_GU)
            line = LineString([(float(p[0]), float(p[1])) for p in geom])
            buf = line.buffer(width / 2.0, cap_style=2, join_style=2)
            length = 0.0
            for a, b in zip(coords, coords[1:]):
                seg = LineString([a, b])
                if seg.length > 0 and seg.distance(buf) <= FRONTAGE_EPS_GU:
                    length += seg.length
            if length > MIN_FRONTAGE_GU or (
                    length > 0 and road.get("hierarchy") == "alley"):
                hier = road.get("hierarchy") or "street"
                target = "alley" if hier == "alley" else "street"
                if hier == "lane":
                    target = "street"
                hits.append({
                    "road_id": road["road_id"],
                    "target_type": target,
                    "length": length,
                })
        hits.sort(key=lambda h: h["road_id"])
        return hits

    def recurse(poly: Polygon, ward_id: str, ward_type: str, grammar: dict,
                roles: list[str], depth: int, rng, primary_road=None) -> None:
        live_roads = roads + alleys
        min_w, min_d = _min_house(stamp_index, roles, ward_type)
        if primary_road is not None:
            frame = _frame_for_road(poly, primary_road)
            if frame is None or frame["contact"] < MIN_FRONTAGE_GU * 0.5:
                primary_road = None
                frame = None
        else:
            frame = None
        if frame is None:
            primary_road = _best_road(poly, live_roads)
            frame = _frame_for_road(poly, primary_road) if primary_road else None
        total_front = _frontage_length(poly, _corridors(live_roads))

        def emit_or_verge() -> None:
            along = 0.0 if frame is None else frame["along"]
            deep = 0.0 if frame is None else frame["depth"]
            if (total_front < MIN_FRONTAGE_GU or _min_width(poly) < 128.0
                    or not _any_fits(stamp_index, roles, ward_type,
                                     along, deep, along * deep)):
                emit_verge(poly)
                return
            empty_prob = 0.0 if ward_type in CORE_WARDS else float(grammar["emptyProb"])
            if rng.random() < empty_prob:
                emit_verge(poly)
                return
            emit_parcel(poly, ward_id, ward_type, roles, True, total_front,
                        road_hits(poly))

        def add_alley_segment(t0: float, t1: float, n_mid: float) -> Optional[dict]:
            tx, ty = frame["tangent"]
            nx, ny = frame["inward"]
            ox, oy = frame["origin"]
            span = max(poly.bounds[2] - poly.bounds[0],
                       poly.bounds[3] - poly.bounds[1], 1.0) * 2.0
            raw = LineString([
                (ox + tx * (t0 - span) + nx * n_mid,
                 oy + ty * (t0 - span) + ny * n_mid),
                (ox + tx * (t1 + span) + nx * n_mid,
                 oy + ty * (t1 + span) + ny * n_mid),
            ])
            hit = raw.intersection(poly.buffer(ALLEY_CLEAR_WIDTH_GU / 2.0, join_style=2))
            if hit.is_empty:
                hit = raw.intersection(poly)
            if hit.geom_type == "MultiLineString":
                hit = max(hit.geoms, key=lambda g: g.length)
            if hit.geom_type != "LineString" or hit.length < MIN_FRONTAGE_GU:
                return None
            coords = list(hit.coords)
            p_a, p_b = coords[0], coords[-1]

            def snap_node(pt) -> Optional[str]:
                best_id = None
                best_d = FRONTAGE_EPS_GU * 2.0
                for road in roads + alleys:
                    line = _as_line(road)
                    if line is None:
                        continue
                    dist = line.distance(Point(pt[0], pt[1]))
                    if dist >= best_d:
                        continue
                    pa = line.coords[0]
                    pb = line.coords[-1]
                    da = math.hypot(pt[0] - pa[0], pt[1] - pa[1])
                    db = math.hypot(pt[0] - pb[0], pt[1] - pb[1])
                    best_id = road["node_a"] if da <= db else road["node_b"]
                    best_d = dist
                return best_id

            seq["node"] += 1
            nid = seq["node"]
            alley = {
                "road_id": f"road_{candidate_id}_alley_{seq['alley']:04d}",
                "node_a": snap_node(p_a) or f"node_{candidate_id}_alley_{nid:04d}_a",
                "node_b": snap_node(p_b) or f"node_{candidate_id}_alley_{nid:04d}_b",
                "polyline": [list(p_a), list(p_b)],
                "hierarchy": "alley",
                "clear_width_gu": ALLEY_CLEAR_WIDTH_GU,
                "paint_surface": "settlement_dirt",
                "source_edge_ids": [],
                "boundary_edge_ids": [],
            }
            alleys.append(alley)
            seq["alley"] += 1
            return alley

        def emit_plot(t0: float, t1: float, n0: float, n1: float) -> bool:
            along = t1 - t0
            deep = n1 - n0
            if along < 400.0 or deep < 400.0:
                emit_fail["small_guard"] += 1
                return False
            if not _any_fits(stamp_index, roles, ward_type, along, deep):
                emit_fail["no_stamp_fits"] += 1
                return False

            usable = usable_union(poly)
            if usable.is_empty:
                emit_fail["usable_empty"] += 1
                return False

            requested = dict(frame)
            requested["inscribed"] = {
                "t0": t0, "t1": t1, "n0": n0, "n1": n1,
                "along": t1 - t0, "depth": n1 - n0,
            }
            req = _rect_from_frame(requested)
            if req is None or req.is_empty:
                emit_fail["no_fragment"] += 1
                return False

            if usable.geom_type == "Polygon":
                fragments = [usable]
            else:
                fragments = _parts(usable)
            scored = []
            for fragment in fragments:
                intersection_area = fragment.intersection(req).area
                if intersection_area > 0.0:
                    scored.append((intersection_area, fragment))
            if not scored:
                emit_fail["no_fragment"] += 1
                return False
            fragment = max(scored, key=lambda item: item[0])[1]
            frag_frame = _frame_for_road(fragment, frame["road"])
            if frag_frame is None:
                emit_fail["no_frag_frame"] += 1
                return False

            t0 = max(t0, frag_frame["tmin"])
            t1 = min(t1, frag_frame["tmax"])
            n0 = max(n0, frag_frame["nmin"])
            n1 = min(n1, frag_frame["nmax"])
            if t1 - t0 < 400.0 or n1 - n0 < 400.0:
                emit_fail["clamp_too_small"] += 1
                return False

            corridor_union = _corridors(roads + alleys)
            for back_trim in range(0, 257, 16):
                for side_trim in range(0, 257, 16):
                    for left_trim in range(0, 129, 16):
                        ct0 = t0 + left_trim + 2.0
                        ct1 = t1 - side_trim - 2.0
                        cn0 = n0 + 2.0
                        cn1 = n1 - back_trim - 2.0
                        if ct1 <= ct0 or cn1 <= cn0:
                            continue
                        if not _any_fits(stamp_index, roles, ward_type,
                                         ct1 - ct0, cn1 - cn0):
                            continue
                        fake = dict(frag_frame)
                        fake["inscribed"] = {
                            "t0": ct0, "t1": ct1, "n0": cn0, "n1": cn1,
                            "along": ct1 - ct0, "depth": cn1 - cn0,
                        }
                        rect = _rect_from_frame(fake)
                        if rect is None or not usable.covers(rect):
                            continue
                        if any(rect.intersection(old).area > 1.0
                               for old in verified_rects):
                            continue
                        rows = _fitting_rows(
                            stamp_index, roles, ward_type,
                            along=ct1 - ct0, depth=cn1 - cn0)
                        if not rows:
                            continue
                        if (_frontage_length(rect, corridor_union)
                                < MIN_PARCEL_FRONTAGE_GU):
                            continue
                        verified_frame = {
                            "tangent": [float(v) for v in frag_frame["tangent"]],
                            "inward": [float(v) for v in frag_frame["inward"]],
                            "origin": [float(v) for v in frag_frame["origin"]],
                            "tmin": float(ct0), "tmax": float(ct1),
                            "nmin": float(cn0), "nmax": float(cn1),
                            "along": float(ct1 - ct0),
                            "depth": float(cn1 - cn0),
                        }
                        before = seq["parcel"]
                        emit_parcel(
                            rect, ward_id, ward_type, roles, True,
                            ct1 - ct0, [], verified=True,
                            verified_frame=verified_frame,
                            frontage_road_id=frame["road"]["road_id"],
                            feasible_stamp_ids=sorted(
                                row["stamp_id"] for row in rows),
                            intended_family=kit_family(rows[0]),
                        )
                        if seq["parcel"] > before:
                            verified_rects.append(rect)
                            return True
                        emit_fail["search_exhausted"] += 1
                        return False

            emit_fail["search_exhausted"] += 1
            safe = _inscribed_bounds(fragment, frag_frame)
            safe_rect = None
            if safe is not None:
                safe_fake = dict(frag_frame)
                safe_fake["inscribed"] = safe
                safe_rect = _rect_from_frame(safe_fake)
            if (safe_rect is not None
                    and not any(safe_rect.intersection(old).area > 1.0
                                for old in verified_rects)
                    and usable.covers(safe_rect)
                    and _frontage_length(safe_rect, corridor_union)
                    >= MIN_PARCEL_FRONTAGE_GU):
                safe_along = float(safe["along"])
                safe_depth = float(safe["depth"])
                rows = _fitting_rows(
                    stamp_index, roles, ward_type,
                    along=safe_along, depth=safe_depth)
                if rows:
                    verified_frame = {
                        "tangent": [float(v) for v in frag_frame["tangent"]],
                        "inward": [float(v) for v in frag_frame["inward"]],
                        "origin": [float(v) for v in frag_frame["origin"]],
                        "tmin": float(safe["t0"]), "tmax": float(safe["t1"]),
                        "nmin": float(safe["n0"]), "nmax": float(safe["n1"]),
                        "along": safe_along, "depth": safe_depth,
                    }
                    before = seq["parcel"]
                    emit_parcel(
                        safe_rect, ward_id, ward_type, roles, True,
                        safe_along, [], verified=True,
                        verified_frame=verified_frame,
                        frontage_road_id=frame["road"]["road_id"],
                        feasible_stamp_ids=sorted(
                            row["stamp_id"] for row in rows),
                        intended_family=kit_family(rows[0]),
                    )
                    if seq["parcel"] > before:
                        verified_rects.append(safe_rect)
                        return True
            emit_fail["fallback_failed"] += 1
            return False

        def pack_row(t0: float, t1: float, n0: float, n1: float) -> None:
            t = t0
            row_d = n1 - n0
            while t1 - t >= min_w * 0.98:
                remain = t1 - t
                fitting = [
                    row for row in _core_rows(stamp_index, roles, ward_type)
                    if _stamp_fits(row, remain, row_d)
                ]
                if not fitting:
                    break
                fitting.sort(key=lambda r: (
                    float(r["obb_width_gu"]),
                    float(r["obb_depth_gu"]),
                    r["stamp_id"]))
                placed = False
                picks = list(fitting)
                if rng.random() < 0.45 and len(picks) > 1:
                    picks = picks[1:] + picks[:1]
                for pick in picks:
                    for slack in (pack_slack(pick), 1.02, 1.0):
                        slice_w = float(pick["obb_width_gu"]) * slack
                        if slice_w > remain + 1.0:
                            if remain >= float(pick["obb_width_gu"]) * 0.98:
                                slice_w = remain
                            else:
                                continue
                        leftover = remain - slice_w
                        if leftover > 1.0 and leftover < min_w * 0.98:
                            if emit_plot(t, t + slice_w, n0, n1):
                                return
                            continue
                        if emit_plot(t, t + slice_w, n0, n1):
                            t += slice_w
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    break

        if frame is None or depth >= MAX_DEPTH:
            emit_or_verge()
            return
        if depth > 0 and frame["along"] < min_w * 0.98:
            emit_verge(poly)
            return

        empty_prob = 0.0 if ward_type in CORE_WARDS else float(grammar["emptyProb"])
        if rng.random() < empty_prob:
            emit_verge(poly)
            return

        t0, t1 = frame["tmin"] + 16.0, frame["tmax"] - 16.0
        n0, n1 = frame["nmin"] + 16.0, frame["nmax"] - 16.0
        along = t1 - t0
        packed = []
        n = n0
        while n1 - n >= 400.0:
            leftover_n = n1 - n
            row_d = None
            for row in sorted(
                    _core_rows(stamp_index, roles, ward_type),
                    key=lambda r: (float(r["obb_depth_gu"]), r["stamp_id"])):
                d = float(row["obb_depth_gu"]) * pack_slack(row)
                if not _stamp_fits(row, along, min(d, leftover_n)):
                    continue
                if d > leftover_n:
                    continue
                rest = leftover_n - d
                if rest < 1.0:
                    row_d = leftover_n
                    break
                if rest >= ALLEY_CLEAR_WIDTH_GU + 400.0:
                    row_d = d
                    break
                row_d = leftover_n
                break
            if row_d is None:
                if leftover_n >= 400.0 and _any_fits(
                        stamp_index, roles, ward_type, along, leftover_n):
                    packed.append((n, n1))
                break
            packed.append((n, n + row_d))
            n += row_d
            rest = n1 - n
            if rest >= 400.0 + ALLEY_CLEAR_WIDTH_GU:
                add_alley_segment(t0, t1, n + ALLEY_CLEAR_WIDTH_GU / 2.0)
                n += ALLEY_CLEAR_WIDTH_GU
            else:
                break

        if not packed:
            emit_or_verge()
            return
        for n_lo, n_hi in packed:
            pack_row(t0, t1, n_lo, n_hi)
        pack_debug.append(
            f"{ward_type}:{along:.0f}x{n1-n0:.0f} rows={len(packed)}")

        used = _rect_from_frame(frame)
        if used is not None and depth + 1 < MAX_DEPTH:
            leftover = poly.difference(used.buffer(32.0, join_style=2))
            for part in _parts(leftover):
                if part.area >= min_w * min_d * 0.8:
                    recurse(part, ward_id, ward_type, grammar, roles, depth + 1,
                            rng, None)

    _ = p50
    blocks = list(candidate.get("buildable_blocks") or [])
    if candidate.get("keep_buildable"):
        blocks.append(candidate["keep_buildable"])
    for block in blocks:
        poly = polygon_from_ring(block["polygon"])
        for prot in protected_polys:
            poly = poly.difference(prot)
        pieces = _parts(poly)
        if not pieces and poly.geom_type == "Polygon" and poly.area > 1:
            pieces = [poly]
        pid = block.get("patch_id") or ""
        ward_id, ward_type = _ward_of(pid, wards)
        if block.get("block_id", "").startswith("keep_block"):
            ward_type = "keep"
            keep_ward = next((w for w in wards if w.get("ward_type") == "keep"), None)
            if keep_ward:
                ward_id = keep_ward["ward_id"]
        grammar = WARD_GRAMMAR[ward_type]
        roles = list(WARD_BUILDING_TYPES.get(ward_type, ("house",)))
        rng = stage_rng(seed, candidate_id, "parcels", block.get("block_id", pid))
        for piece in pieces:
            recurse(piece, ward_id, ward_type, grammar, roles, 0, rng)

    kept: list[dict] = []
    for parcel in parcels:
        p = polygon_from_ring(parcel["polygon"])
        if parcel.get("verified"):
            cleaned = p
        else:
            cleaned = clip_roads(p)
        if cleaned is None:
            if parcel.get("verified"):
                emit_fail["verified_discarded_road"] += 1
            emit_verge(p)
            continue
        if not parcel.get("verified"):
            parcel["polygon"] = _ring(cleaned)
        if _frontage_length(cleaned, _corridors(roads + alleys)) < MIN_FRONTAGE_GU:
            if parcel.get("verified"):
                emit_fail["verified_discarded_road"] += 1
            emit_verge(cleaned)
            continue
        kept.append(parcel)
    parcels = kept

    union_roads = _corridors(roads + alleys)
    if union_roads:
        road_u = unary_union(union_roads)
        leftover = []
        for parcel in parcels:
            p = polygon_from_ring(parcel["polygon"])
            if p.intersects(road_u) and p.intersection(road_u).area > 1.0:
                if parcel.get("verified"):
                    emit_fail["verified_discarded_road"] += 1
                emit_verge(p)
                continue
            leftover.append(parcel)
        parcels = leftover
    kept2: list[dict] = []
    for parcel in parcels:
        pa = polygon_from_ring(parcel["polygon"])
        overlap = False
        for other in kept2:
            pb = polygon_from_ring(other["polygon"])
            if pa.intersects(pb) and pa.intersection(pb).area > 1.0:
                overlap = True
                break
        if overlap:
            if parcel.get("verified"):
                emit_fail["verified_discarded_overlap"] += 1
            emit_verge(pa)
            continue
        kept2.append(parcel)
    parcels = kept2

    used_ids = {
        frontage["target_id"]
        for parcel in parcels
        for frontage in parcel.get("frontages") or []
    }
    pruned = [alley for alley in alleys
              if alley["road_id"] not in used_ids]
    alleys = [alley for alley in alleys
              if alley["road_id"] in used_ids]

    reports = list(candidate.get("reports") or [])
    reports.append({
        "stage": "parcels",
        "status": "ok",
        "message": (
            f"parcels={len(parcels)} alleys={len(alleys)} "
            f"verges={len(verges)} "
            f"verified={sum(1 for p in parcels if p.get('verified'))} "
            f"verified_discarded_road={emit_fail['verified_discarded_road']} "
            f"verified_discarded_overlap={emit_fail['verified_discarded_overlap']} "
            f"alley_pruned={len(pruned)} "
            "emit_fail="
            + ",".join(
                f"{key}={emit_fail[key]}"
                for key in (
                    "small_guard", "no_stamp_fits", "usable_empty",
                    "no_fragment", "no_frag_frame", "clamp_too_small",
                    "search_exhausted", "fallback_failed",
                )
            )
        ),
    })
    out = dict(candidate)
    out["parcels"] = parcels
    out["roads"] = roads + alleys
    out["open_spaces"] = list(candidate.get("open_spaces") or []) + verges
    out["reports"] = reports
    return out


def write_parcels_diagnostic(
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

    ward_color = {
        "market": (220, 40, 160, 90),
        "craft": (200, 120, 40, 90),
        "residential": (60, 150, 80, 90),
        "outskirts": (180, 180, 60, 80),
        "keep": (90, 90, 90, 90),
    }
    ward_of = {}
    for ward in product.get("wards") or []:
        for pid in ward.get("patch_ids") or []:
            ward_of[pid] = ward["ward_type"]
    by_parcel_ward = {p["parcel_id"]: p.get("ward_id") for p in product.get("parcels") or []}
    ward_type_of = {w["ward_id"]: w["ward_type"] for w in product.get("wards") or []}
    for space in product.get("open_spaces") or []:
        ring = space.get("polygon") or []
        if len(ring) < 3:
            continue
        kind = space.get("kind")
        fill = (220, 40, 160, 70) if kind == "plaza" else (140, 160, 90, 50)
        draw.polygon([to_px(p) for p in ring], fill=fill, outline=(80, 40, 80, 180))
    for parcel in product.get("parcels") or []:
        ring = parcel.get("polygon") or []
        if len(ring) < 3:
            continue
        wtype = ward_type_of.get(parcel.get("ward_id"), "residential")
        draw.polygon([to_px(p) for p in ring], fill=ward_color.get(wtype, (80, 120, 80, 80)),
                     outline=(20, 40, 20, 220))
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
    wall = product.get("wall")
    if wall:
        ring = wall.get("planning_polygon") or []
        if len(ring) >= 3:
            pts = [to_px(p) for p in ring] + [to_px(ring[0])]
            draw.line(pts, fill=(90, 50, 20, 255), width=3)
    Image.alpha_composite(image, overlay).save(out_png)
    _ = (ctx, by_parcel_ward, ward_of)
