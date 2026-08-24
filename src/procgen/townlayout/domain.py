"""Stage 04 compact, land-neighbour domain growth.

This stage selects a connected patch union inside a bounded search envelope.
The selected union follows terrain, shared patch boundaries, and mandatory
source-road geometry while the envelope remains construction-only metadata.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from .constants import PARCEL_YARD_FACTOR, VERTEX_EPS_GU
from .geometry import normalize_ring, polygon_from_ring
from .site_context import SiteContext, _plan_to_px, diagnostic_view
from .validate import TownLayoutError

SUIT_MIN = 0.25
AREA_EPS_GU2 = 1.0


def _iq(poly: Polygon) -> float:
    if poly.is_empty or poly.area <= 0 or poly.length <= VERTEX_EPS_GU:
        return 0.0
    return float(4.0 * math.pi * poly.area / poly.length ** 2)


def _patch_poly(patch: dict) -> Polygon:
    return polygon_from_ring(patch["polygon"])


def _mean_suit(ctx: SiteContext, poly: Polygon) -> float:
    p = poly.representative_point()
    return float(ctx.sample(float(p.x), float(p.y)).get("suitability", 0.0))


def _capacity(area: float, p50: float) -> float:
    parcel = float(p50) * PARCEL_YARD_FACTOR
    if parcel <= 0:
        raise TownLayoutError("invalid_polygon: parcel area <= 0")
    return area / parcel


def _start_patch_id(patches: list[dict], cx: float, cy: float) -> str:
    point = Point(cx, cy)
    containing = sorted(p["patch_id"] for p in patches
                        if _patch_poly(p).covers(point))
    if containing:
        return containing[0]
    return min(patches, key=lambda p: (
        _patch_poly(p).centroid.distance(point), p["patch_id"]))["patch_id"]


def _frontier(inner: set[str], by_id: dict[str, dict]) -> list[str]:
    return sorted({nid for pid in inner for nid in by_id[pid].get(
        "neighbour_patch_ids", []) if nid in by_id and nid not in inner})


def _road_band(approaches: Optional[list]) -> Optional[object]:
    lines = []
    for approach in approaches or []:
        if not approach.get("mandatory"):
            continue
        points = approach.get("inside_polyline_plan_gu") or []
        if len(points) >= 2:
            line = LineString(points)
            if not line.is_empty and line.length > VERTEX_EPS_GU:
                lines.append(line)
    if not lines:
        return None
    return unary_union(lines).buffer(2048.0)


def _score_add(ctx: SiteContext, current: Polygon, candidate: Polygon,
               road_band) -> tuple[float, dict]:
    suitability = max(0.0, min(1.0, _mean_suit(ctx, candidate)))
    shared = current.boundary.intersection(candidate.boundary).length
    shared_norm = max(0.0, min(1.0, shared / max(0.35 * candidate.length, 1.0)))
    road_overlap = 0.0
    if road_band is not None and candidate.area > 0:
        road_overlap = max(0.0, min(1.0, math.sqrt(
            candidate.intersection(road_band).area / candidate.area)))
    parts = {"suitability": suitability, "shared_boundary": shared_norm,
             "road_overlap": road_overlap}
    if road_band is None:
        return 0.43 * suitability + 0.57 * shared_norm, parts
    return 0.30 * suitability + 0.40 * shared_norm + 0.30 * road_overlap, parts


def _line_components(geom) -> list[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    return [part for part in getattr(geom, "geoms", []) if isinstance(part, LineString)]


def _roadside_report(ctx: SiteContext, selected: Polygon, approaches: Optional[list]) -> dict:
    site = box(0.0, 0.0, float(ctx.span_gu[0]), float(ctx.span_gu[1]))
    water = unary_union(ctx.water_polygons()) if ctx.water_polygons() else None
    eligible = covered = blocked = 0
    for approach in approaches or []:
        if not approach.get("mandatory"):
            continue
        points = approach.get("inside_polyline_plan_gu") or []
        if len(points) < 2:
            continue
        clipped = LineString(points).intersection(selected)
        for line in _line_components(clipped):
            if line.length < 2048.0:
                continue
            distance = 512.0
            while distance <= line.length - 512.0 + AREA_EPS_GU2:
                point = line.interpolate(distance)
                if (point.distance(site.boundary) <= AREA_EPS_GU2 or
                        (water is not None and point.distance(water) <= AREA_EPS_GU2)):
                    blocked += 1
                    distance += 1024.0
                    continue
                coords = list(line.coords)
                before = line.interpolate(max(0.0, distance - 1.0))
                after = line.interpolate(min(line.length, distance + 1.0))
                dx, dy = after.x - before.x, after.y - before.y
                length = math.hypot(dx, dy) or 1.0
                nx, ny = -dy / length, dx / length
                eligible += 1
                depths = []
                for sign in (-1.0, 1.0):
                    ray = LineString([(point.x, point.y),
                                      (point.x + sign * nx * 4096.0,
                                       point.y + sign * ny * 4096.0)])
                    hit = ray.intersection(selected.boundary)
                    depths.append(min((point.distance(g) for g in getattr(hit, "geoms", [hit])
                                       if not g.is_empty), default=0.0))
                if min(depths) >= 768.0:
                    covered += 1
                distance += 1024.0
    return {"eligible_station_count": eligible,
            "two_sided_coverage_fraction": covered / eligible if eligible else 0.0,
            "blocked_station_count": blocked}


def valid_domain_addition(current: Polygon, candidate: Polygon) -> bool:
    """Return whether an addition remains a single hole-free city polygon."""
    proposed = current.union(candidate)
    return proposed.geom_type == "Polygon" and not proposed.interiors


def grow_city_domain(ctx: SiteContext, candidate: dict, town_brief: dict,
                     *, approaches: Optional[list] = None,
                     rewrite_domain_meta: Optional[dict] = None) -> dict[str, Any]:
    """Grow compactly, retaining valid snapshots and failing closed."""
    patches = list(candidate["patches"])
    if len(patches) < 3:
        raise TownLayoutError("invalid_polygon: need at least 3 patches")
    by_id = {p["patch_id"]: p for p in patches}
    cx, cy = ctx.candidate_centers[0]
    start = _start_patch_id(patches, cx, cy)
    est = ctx.estimated_urban_area_gu2
    amin, amax = float(est["min"]), float(est["max"])
    preferred = float(est["preferred"])
    p50 = float(ctx.stamp_footprint_stats["p50"])
    road_band = _road_band(approaches)
    search_boundary = None
    if rewrite_domain_meta:
        disk = rewrite_domain_meta.get("unclipped_disk")
        site = rewrite_domain_meta.get("site_box")
        if disk is not None and site is not None:
            search_boundary = disk.boundary.difference(site.boundary.buffer(1.0))
    inner = {start}
    current = _patch_poly(by_id[start])
    snapshots: list[tuple[float, float, float, set[str], Polygon]] = []
    rejected: dict[str, int] = {"multipart": 0, "holes": 0,
                                "search_envelope_contact": 0}

    while True:
        area = float(current.area)
        cap = _capacity(area, p50)
        if amin <= area <= amax:
            snapshots.append((abs(area - preferred), area, _iq(current), set(inner), current))
        if area >= preferred:
            break
        candidates = []
        for pid in _frontier(inner, by_id):
            proposed = current.union(_patch_poly(by_id[pid]))
            if proposed.geom_type != "Polygon":
                rejected["multipart"] += 1
                continue
            if proposed.interiors:
                rejected["holes"] += 1
                continue
            if (search_boundary is not None and
                    _patch_poly(by_id[pid]).boundary.intersection(search_boundary).length > 1.0):
                rejected["search_envelope_contact"] += 1
                continue
            _score, parts = _score_add(ctx, current, _patch_poly(by_id[pid]), road_band)
            candidates.append((pid, proposed, parts))
        if not candidates:
            break
        ranked = []
        for pid, proposed, parts in candidates:
            score = (0.30 * parts["suitability"] + 0.40 * parts["shared_boundary"] +
                     0.30 * parts["road_overlap"]) if road_band is not None else (
                     0.43 * parts["suitability"] + 0.57 * parts["shared_boundary"])
            ranked.append((-score, pid, proposed, parts))
        ranked.sort(key=lambda x: (x[0], x[1]))
        _score, chosen, current, _parts = ranked[0]
        inner.add(chosen)

    if not snapshots:
        available = _capacity(float(current.area), p50)
        required = _capacity(amin, p50)
        raise TownLayoutError(
            f"insufficient_compact_capacity: available={available:.3f} required={required:.3f} "
            f"frontier_rejections={rejected}")
    snapshots.sort(key=lambda item: (item[0], item[3]))
    _distance, area, iq, inner, core_union = snapshots[0]
    cap = _capacity(area, p50)
    if _mean_suit(ctx, core_union) < SUIT_MIN:
        raise TownLayoutError("low_suitability: inner mean suitability < 0.25")

    # Outskirts are a connected subset of the selected set's own boundary;
    # never append an external patch after the authoritative snapshot.
    outskirts: set[str] = set()
    share = float(town_brief["ward_mix"].get("outskirts", 0.0))
    if share > 0:
        fringe = {p["patch_id"] for p in patches if p["patch_id"] in inner
                  and any(n not in inner for n in p.get("neighbour_patch_ids", []))}
        budget = area * share
        if fringe:
            seed = min(fringe, key=lambda pid: (-_patch_poly(by_id[pid]).area, pid))
            outskirts.add(seed)
            while True:
                frontier = sorted({n for pid in outskirts
                                   for n in by_id[pid].get("neighbour_patch_ids", [])
                                   if n in fringe and n not in outskirts},
                                  key=lambda pid: (-_patch_poly(by_id[pid]).area, pid))
                if not frontier:
                    break
                chosen = frontier[0]
                if sum(_patch_poly(by_id[x]).area for x in outskirts) + _patch_poly(by_id[chosen]).area > budget:
                    break
                outskirts.add(chosen)

    selected_union = unary_union([_patch_poly(by_id[pid]) for pid in inner])
    if selected_union.geom_type != "Polygon" or selected_union.interiors:
        raise TownLayoutError("invalid_polygon: selected union is not hole-free")
    area = float(selected_union.area)
    iq = _iq(selected_union)
    cap = _capacity(area, p50)
    radial = sorted(math.hypot(_patch_poly(by_id[pid]).centroid.x - cx,
                               _patch_poly(by_id[pid]).centroid.y - cy)
                    for pid in inner)
    hull_area = selected_union.convex_hull.area or 1.0
    gap_ratio = max(0.0, 1.0 - area / hull_area)
    reports = list(candidate.get("reports") or [])
    roadside = _roadside_report(ctx, selected_union, approaches)
    contact = {"search_envelope_boundary_gu": 0.0,
               "real_site_boundary_gu": 0.0, "water_boundary_gu": 0.0}
    if rewrite_domain_meta:
        disk = rewrite_domain_meta.get("unclipped_disk")
        site = rewrite_domain_meta.get("site_box")
        if disk is not None:
            artificial = disk.boundary.difference(site.boundary.buffer(1.0))
            contact["search_envelope_boundary_gu"] = float(
                selected_union.boundary.intersection(artificial).length)
        if site is not None:
            contact["real_site_boundary_gu"] = float(
                selected_union.boundary.intersection(site.boundary).length)
    water_geom = unary_union(ctx.water_polygons()) if ctx.water_polygons() else None
    if water_geom is not None:
        contact["water_boundary_gu"] = float(
            selected_union.boundary.intersection(water_geom.boundary).length)
    if contact["search_envelope_boundary_gu"] <= 1.0:
        contact["search_envelope_boundary_gu"] = 0.0
    elif contact["search_envelope_boundary_gu"] > 1.0:
        raise TownLayoutError("domain_search_envelope_exhausted")
    reports.append({"stage": "domain", "status": "ok", "message": (
        f"inner={len(inner)} outskirts={len(outskirts)} area={area:.1f} "
        f"capacity={cap:.2f} iq={iq:.3f} radial_max_median={radial[-1]:.1f},{radial[len(radial)//2]:.1f} "
        f"convex_hull_gap_ratio={gap_ratio:.4f} frontier_rejections={rejected}")})
    new_patches = []
    for patch in patches:
        item = dict(patch)
        pid = patch["patch_id"]
        item["inside_city"] = pid in inner or pid in outskirts
        item["inside_wall"] = False
        if pid in outskirts:
            item["morphology_region"] = "outskirts"
        new_patches.append(item)
    out = dict(candidate)
    out.update({"patches": new_patches,
                "city_domain": normalize_ring([[x, y] for x, y in selected_union.exterior.coords])["ring"],
                "morphology_regions": ([{"region_id": "organic", "patch_ids": sorted(inner)}] +
                                        ([{"region_id": "outskirts", "patch_ids": sorted(outskirts)}] if outskirts else [])),
                "reports": reports})
    metrics = dict(candidate.get("domain_metrics") or {})
    metrics.update({"selected_area_gu2": area, "capacity": cap, "capacity_band": [amin, amax],
                    "iq": iq, "radial_max_gu": radial[-1],
                    "radial_median_gu": radial[len(radial)//2],
                    "convex_hull_gap_ratio": gap_ratio,
                    "frontier_rejections": rejected,
                    **contact, "roadside_coverage": roadside,
                    "score_weights": {"suitability": 0.30 if road_band is not None else 0.43,
                                       "shared_boundary": 0.40 if road_band is not None else 0.57,
                                       "road_overlap": 0.30 if road_band is not None else 0.0}})
    out["domain_metrics"] = metrics
    out["water_metrics"] = dict(candidate.get("water_metrics") or {})
    return out


def write_domain_diagnostic(ctx: SiteContext, product: dict, *, topdown_path: Path,
                            survey: dict, out_png: Path, full_site: bool = False) -> None:
    from PIL import Image, ImageDraw
    image, mapping = diagnostic_view({"_diagnostic_bounds": [product.get("city_domain") or []]},
                                     topdown_path, survey, full_site=full_site)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0)); draw = ImageDraw.Draw(overlay, "RGBA")
    to_px = lambda p: _plan_to_px(float(p[0]), float(p[1]), mapping)
    for patch in product.get("patches", []):
        ring = patch.get("polygon") or []
        if len(ring) >= 3:
            fill = (40, 160, 80, 80) if patch.get("inside_city") else (0, 0, 0, 0)
            draw.polygon([to_px(p) for p in ring], fill=fill, outline=(40, 70, 40, 180))
    ring = product.get("city_domain") or []
    if len(ring) >= 3: draw.polygon([to_px(p) for p in ring], outline=(0, 220, 255, 255))
    px, py = to_px(ctx.candidate_centers[0]); draw.ellipse([px-5, py-5, px+5, py+5], fill=(255,255,0,255))
    Image.alpha_composite(image, overlay).save(out_png)
