"""Semantic anchors (market plaza, optional keep) for V2 townlayout (Phase 7).

Purpose
-------
Score inner patches with the locked first-pass formula, reserve the
winning market patch as a plaza open space, and optionally reserve a
keep patch.  Temple is never placed in v1.

Inputs
------
A Phase 6 domain candidate, SiteContext, TownBrief, optional approaches.

Outputs
-------
The candidate plus ``anchors`` and ``open_spaces``.  Score evidence is
stored on the market/keep anchor via ``reports``.

Pipeline position
-----------------
V2 townlayout Phase 7 anchors; no walls, parcels, or VTEX.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import Polygon

from .geometry import normalize_ring, polygon_from_ring, simple_polygon_parts
from .site_context import SiteContext, _plan_to_px, diagnostic_view
from .validate import TownLayoutError

IQ_EPS = 1e-9
MARKET_MIN_FOOTPRINTS = 2.0
MARKET_MAX_FOOTPRINTS = 3.0


def _iq(poly: Polygon) -> float:
    if poly.is_empty or poly.area <= 0 or poly.length <= IQ_EPS:
        return 0.0
    return float(4.0 * math.pi * poly.area / (poly.length ** 2))


def _inner_core(patches: list[dict]) -> list[dict]:
    return [
        p for p in patches
        if p.get("inside_city") and p.get("morphology_region") != "outskirts"
    ]


def _city_radius(candidate: dict, ctx: SiteContext) -> float:
    domain = candidate.get("city_domain") or []
    if len(domain) >= 3:
        area = polygon_from_ring(domain).area
        return math.sqrt(max(area, 1.0) / math.pi)
    return math.sqrt(max(float(ctx.estimated_urban_area_gu2["preferred"]), 1.0) / math.pi)


def _market_score(ctx: SiteContext, patch: dict, cx: float, cy: float,
                  radius: float, crossings: list[tuple[float, float]]) -> dict:
    poly = polygon_from_ring(patch["polygon"])
    c = poly.centroid
    suit = float(ctx.sample(float(c.x), float(c.y)).get("suitability", 0.0))
    dist_c = math.hypot(c.x - cx, c.y - cy)
    near_center = 1.0 - min(1.0, dist_c / max(radius, 1.0))
    compact = _iq(poly)
    # Stage 04 market placement is center-led.  Approaches must not pull the
    # reservation toward a gate; roads are authored later.
    near_app = 0.0
    score = (
        2.0 * suit
        + 1.5 * near_center
        + 1.0 * compact
        + 1.0 * near_app
    )
    return {
        "score": score,
        "suitability_mean": suit,
        "near_center": near_center,
        "compactness": compact,
        "near_approach": near_app,
    }


def _market_reservation(ctx: SiteContext, patch: dict) -> tuple[list, float]:
    """Return a compact court clipped inside the market patch, never the patch."""
    poly = polygon_from_ring(patch["polygon"])
    target = float(ctx.stamp_footprint_stats["p50"]) * ((MARKET_MIN_FOOTPRINTS + MARKET_MAX_FOOTPRINTS) / 2.0)
    center = poly.representative_point()
    radius = math.sqrt(max(target, 1.0) / math.pi)
    court = poly.intersection(center.buffer(radius, resolution=8))
    parts = simple_polygon_parts(court)
    if not parts:
        raise TownLayoutError("insufficient_compact_capacity: market reservation has no land")
    court = max(parts, key=lambda p: (p.area, -p.centroid.x, -p.centroid.y))
    ring = normalize_ring([[x, y] for x, y in court.exterior.coords])["ring"]
    return ring, float(court.area)


def _keep_score(ctx: SiteContext, patch: dict, city_mean_z: float,
                inner_ids: set[str], edges: list[dict]) -> dict:
    poly = polygon_from_ring(patch["polygon"])
    c = poly.centroid
    sample = ctx.sample(float(c.x), float(c.y))
    z = float(sample.get("elevation_gu", 0.0))
    elev = max(-2.0, min(2.0, 2.0 * (z - city_mean_z) / 500.0))
    compact = _iq(poly)
    pid = patch["patch_id"]
    on_wall = 0.0
    for edge in edges:
        left, right = edge.get("patch_left"), edge.get("patch_right")
        ids = {left, right}
        if pid in ids and (ids - inner_ids):
            on_wall = 1.0
            break
    water_adj = 1.0 if sample.get("water_term", 0.0) >= 0.99 else 0.0
    score = elev + 1.0 * compact + 1.0 * on_wall - 1.0 * water_adj
    return {
        "score": score,
        "elevation_term": elev,
        "compactness": compact,
        "wall_edge_fraction": on_wall,
        "water_adjacent": water_adj,
    }


def place_anchors(
    ctx: SiteContext,
    candidate: dict,
    town_brief: dict,
    *,
    approaches: Optional[list] = None,
    candidate_id: str = "c00",
) -> dict[str, Any]:
    """Reserve market (and optional keep) patches on the inner city."""
    patches = list(candidate["patches"])
    inner = _inner_core(patches)
    presence = town_brief["anchors"]
    if presence["market"] == "required" and not inner:
        raise TownLayoutError("missing_anchor: market required but no inner patch")
    cx, cy = ctx.candidate_centers[0]
    radius = _city_radius(candidate, ctx)
    crossings = []
    for item in approaches or []:
        pt = item.get("crossing_plan_gu") if isinstance(item, dict) else None
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            crossings.append((float(pt[0]), float(pt[1])))

    anchors: list[dict] = []
    open_spaces: list[dict] = []
    reports = list(candidate.get("reports") or [])
    reserved: set[str] = set()

    if presence["market"] != "absent" and inner:
        scored = []
        for patch in inner:
            ev = _market_score(ctx, patch, cx, cy, radius, crossings)
            scored.append((-ev["score"], patch["patch_id"], ev, patch))
        scored.sort()
        _s, pid, ev, patch = scored[0]
        reserved.add(pid)
        anchors.append({
            "anchor_id": f"anchor_{candidate_id}_market",
            "kind": "market",
            "patch_id": pid,
            "polygon": patch["polygon"],
        })
        market_ring, market_area = _market_reservation(ctx, patch)
        anchors[-1]["polygon"] = market_ring
        open_spaces.append({
            "space_id": f"space_{candidate_id}_plaza",
            "kind": "plaza",
            "polygon": market_ring,
        })
        reports.append({
            "stage": "anchors",
            "status": "ok",
            "message": (
                f"market {pid} score={ev['score']:.3f} "
                f"suit={ev['suitability_mean']:.3f} compact={ev['compactness']:.3f} "
                f"reservation_area={market_area:.1f} p50_footprints={market_area / float(ctx.stamp_footprint_stats['p50']):.3f}"
            ),
        })
    elif presence["market"] == "required":
        raise TownLayoutError("missing_anchor: market required")

    keep_mode = presence.get("keep", "absent")
    if keep_mode in ("required", "optional"):
        remaining = [p for p in inner if p["patch_id"] not in reserved]
        if not remaining and keep_mode == "required":
            raise TownLayoutError("missing_anchor: keep required but no remaining patch")
        if remaining:
            inner_ids = {p["patch_id"] for p in inner}
            zs = []
            for patch in inner:
                c = polygon_from_ring(patch["polygon"]).centroid
                zs.append(float(ctx.sample(float(c.x), float(c.y)).get("elevation_gu", 0.0)))
            city_mean_z = sum(zs) / max(len(zs), 1)
            scored = []
            for patch in remaining:
                ev = _keep_score(
                    ctx, patch, city_mean_z, inner_ids,
                    candidate.get("boundary_edges") or [])
                scored.append((-ev["score"], patch["patch_id"], ev, patch))
            scored.sort()
            _s, pid, ev, patch = scored[0]
            if keep_mode == "optional" and ev["score"] < 0:
                reports.append({
                    "stage": "anchors",
                    "status": "ok",
                    "message": f"keep skipped (best score {ev['score']:.3f} < 0)",
                })
            else:
                reserved.add(pid)
                anchors.append({
                    "anchor_id": f"anchor_{candidate_id}_keep",
                    "kind": "keep",
                    "patch_id": pid,
                    "polygon": patch["polygon"],
                })
                reports.append({
                    "stage": "anchors",
                    "status": "ok",
                    "message": f"keep {pid} score={ev['score']:.3f}",
                })
        elif keep_mode == "required":
            raise TownLayoutError("missing_anchor: keep required")

    if presence.get("temple") == "required":
        # v1 schema allows the key; Falkreath fixture is absent. Do not place.
        reports.append({
            "stage": "anchors",
            "status": "ok",
            "message": "temple required but v1 does not place a temple building",
        })

    out = dict(candidate)
    out["anchors"] = anchors
    out["open_spaces"] = open_spaces
    out["reports"] = reports
    return out


def write_anchors_diagnostic(
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

    for patch in product.get("patches", []):
        ring = patch.get("polygon") or []
        if len(ring) < 3:
            continue
        pts = [to_px(p) for p in ring]
        if patch.get("inside_city") and patch.get("morphology_region") != "outskirts":
            draw.polygon(pts, fill=(40, 160, 80, 50), outline=(20, 80, 40, 180))
        elif patch.get("inside_city"):
            draw.polygon(pts, outline=(180, 140, 40, 200))
    for space in product.get("open_spaces", []):
        ring = space.get("polygon") or []
        if len(ring) >= 3:
            draw.polygon([to_px(p) for p in ring], fill=(220, 40, 160, 90),
                         outline=(255, 80, 200, 255))
    for anchor in product.get("anchors", []):
        if anchor.get("kind") == "keep":
            ring = anchor.get("polygon") or []
            if len(ring) >= 3:
                draw.polygon([to_px(p) for p in ring], fill=(80, 80, 80, 80),
                             outline=(20, 20, 20, 255))
    Image.alpha_composite(image, overlay).save(out_png)
