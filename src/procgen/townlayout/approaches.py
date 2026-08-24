"""Road-rewrite domain and aligned source approaches for V2 townlayout.

Purpose
-------
Build the provisional rewrite-domain disk (preferred urban area + margin,
clipped to the site box) and clip aligned centerlines to that domain so
each boundary crossing becomes a Phase 1 ``SourceApproach``.

Inputs
------
A Phase 3 ``SiteContext`` plus either synthetic plan-GU polylines or an
``AlignedNetwork`` from ``procgen.aligned_roads``.

Outputs
-------
Open CCW domain ring; ``site_approaches.json`` dict (rewrite_domain,
approaches, edge_ledger); optional diagnostic PNG on ``site_topdown.png``.

Pipeline position
-----------------
V2 townlayout Phase 4 rewrite domain / approaches; no patches, walls,
or VTEX.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Optional

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    box,
)

from procgen.aligned_roads import AlignedNetwork

from .constants import REWRITE_MARGIN_GU, TRANSITION_STUB_LENGTH_GU, VERTEX_EPS_GU
from .geometry import normalize_ring, polygon_from_ring
from .site_context import SiteContext, _plan_to_px, diagnostic_view
from .validate import TownLayoutError

QUAD_SEGS = 64  # explicit Point.buffer resolution (logged, not silent repair)


def build_rewrite_domain(
    ctx: SiteContext,
    *,
    radius_gu: Optional[float] = None,
    margin_gu: float = REWRITE_MARGIN_GU,
    metadata: Optional[dict[str, Any]] = None,
) -> list[list[float]]:
    """Build the bounded circular search envelope and optionally retain its geometry."""
    if not ctx.candidate_centers:
        raise TownLayoutError("no_buildable_center")
    cx, cy = ctx.candidate_centers[0]
    if radius_gu is None:
        preferred = float(ctx.estimated_urban_area_gu2["preferred"])
        if preferred <= 0:
            raise TownLayoutError("invalid_polygon: preferred urban area <= 0")
        radius_gu = math.sqrt(preferred / math.pi)
    radius_gu = float(radius_gu) + float(margin_gu)
    if radius_gu <= 0:
        raise TownLayoutError("invalid_polygon: rewrite domain radius <= 0")
    disk = Point(float(cx), float(cy)).buffer(radius_gu, quad_segs=QUAD_SEGS)
    site = box(0.0, 0.0, float(ctx.span_gu[0]), float(ctx.span_gu[1]))
    clipped = disk.intersection(site)
    if clipped.is_empty or clipped.area <= 0:
        raise TownLayoutError("invalid_polygon: rewrite domain empty after clip")
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda g: g.area)
    if clipped.geom_type != "Polygon":
        raise TownLayoutError("invalid_polygon: rewrite domain is not a polygon")
    if len(clipped.interiors) > 0:
        raise TownLayoutError("invalid_polygon: holes not supported in v1")
    if metadata is not None:
        metadata.update({
            "unclipped_disk": disk,
            "site_box": site,
            "search_radius_gu": float(radius_gu),
            "search_clearance_gu": float(margin_gu),
        })
    return normalize_ring([[c[0], c[1]] for c in clipped.exterior.coords])["ring"]


def _as_lines(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length > VERTEX_EPS_GU and len(geom.coords) >= 2 else []
    if isinstance(geom, MultiLineString):
        out = []
        for part in geom.geoms:
            out.extend(_as_lines(part))
        return out
    if isinstance(geom, GeometryCollection):
        out = []
        for part in geom.geoms:
            out.extend(_as_lines(part))
        return out
    return []


def _coords(line: LineString) -> list[list[float]]:
    return [[float(x), float(y)] for x, y, *_ in line.coords]


def _unit(dx: float, dy: float) -> Optional[tuple[float, float]]:
    n = math.hypot(dx, dy)
    if n <= VERTEX_EPS_GU:
        return None
    return (dx / n, dy / n)


def _boundary_points(geom) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if geom is None or geom.is_empty:
        return points
    if isinstance(geom, Point):
        points.append((float(geom.x), float(geom.y)))
    elif isinstance(geom, MultiPoint):
        for p in geom.geoms:
            points.append((float(p.x), float(p.y)))
    elif isinstance(geom, LineString):
        coords = list(geom.coords)
        if coords:
            points.append((float(coords[0][0]), float(coords[0][1])))
            points.append((float(coords[-1][0]), float(coords[-1][1])))
    elif isinstance(geom, (MultiLineString, GeometryCollection)):
        for part in geom.geoms:
            points.extend(_boundary_points(part))
    return points


def _inward_tangent(line: LineString, crossing: tuple[float, float],
                    domain: Polygon) -> tuple[float, float]:
    pt = Point(crossing)
    dist = float(line.project(pt))
    eps = min(1.0, max(line.length * 0.01, 1e-3))
    fwd = line.interpolate(min(dist + eps, line.length))
    back = line.interpolate(max(dist - eps, 0.0))
    candidates = [
        _unit(fwd.x - crossing[0], fwd.y - crossing[1]),
        _unit(back.x - crossing[0], back.y - crossing[1]),
    ]
    for vec in candidates:
        if vec is None:
            continue
        probe = Point(crossing[0] + vec[0] * 8.0, crossing[1] + vec[1] * 8.0)
        if domain.covers(probe) or domain.contains(probe):
            return vec
    c = domain.centroid
    vec = _unit(c.x - crossing[0], c.y - crossing[1])
    if vec is None:
        raise TownLayoutError("zero_tangent: cannot orient inward tangent")
    return vec


def _nearest_outside(crossing: tuple[float, float],
                     outside_lines: list[LineString]) -> Optional[list[list[float]]]:
    best = None
    best_d = None
    for line in outside_lines:
        coords = _coords(line)
        if len(coords) < 2:
            continue
        d0 = math.hypot(coords[0][0] - crossing[0], coords[0][1] - crossing[1])
        d1 = math.hypot(coords[-1][0] - crossing[0], coords[-1][1] - crossing[1])
        d = min(d0, d1)
        if best_d is None or d < best_d:
            best_d = d
            # Orient so the crossing end is last (road arriving at the city).
            best = coords if d1 <= d0 else list(reversed(coords))
    return best


def _nearest_inside(crossing: tuple[float, float],
                    inside_lines: list[LineString]) -> Optional[list[list[float]]]:
    """Return the source-road continuation oriented inward from a crossing."""
    best = None
    best_d = None
    for line in inside_lines:
        coords = _coords(line)
        if len(coords) < 2:
            continue
        d0 = math.hypot(coords[0][0] - crossing[0], coords[0][1] - crossing[1])
        d1 = math.hypot(coords[-1][0] - crossing[0], coords[-1][1] - crossing[1])
        d = min(d0, d1)
        if best_d is None or d < best_d:
            best_d = d
            best = coords if d0 <= d1 else list(reversed(coords))
    return best


def extract_approaches(
    ctx: SiteContext,
    domain_ring: list,
    edges: Iterable[dict],
    *,
    candidate_id: str = "c00",
) -> dict[str, Any]:
    """Clip plan-GU polylines to the rewrite domain and emit SourceApproaches."""
    domain = polygon_from_ring(domain_ring)
    ledger: list[dict] = []
    approaches: list[dict] = []
    index = 0
    sorted_edges = sorted(edges, key=lambda e: str(e["id"]))
    for edge in sorted_edges:
        edge_id = str(edge["id"])
        chain = edge["plan_polyline"]
        if len(chain) < 2:
            continue
        line = LineString([(float(p[0]), float(p[1])) for p in chain])
        if line.is_empty or not line.intersects(domain):
            continue
        outside_lines = _as_lines(line.difference(domain))
        inside_lines = _as_lines(line.intersection(domain))
        crossings = _boundary_points(line.intersection(domain.boundary))
        # Stable along-chain order; do not merge nearby crossings.
        crossings = sorted(set((round(x, 6), round(y, 6)) for x, y in crossings),
                           key=lambda p: line.project(Point(p)))
        approach_ids: list[str] = []
        invalid: list[dict] = []
        for crossing in crossings:
            outside = _nearest_outside(crossing, outside_lines)
            inside = _nearest_inside(crossing, inside_lines)
            if outside is None or inside is None:
                continue
            sample = ctx.sample(crossing[0], crossing[1])
            if not sample.get("buildable", False):
                invalid.append({
                    "crossing_plan_gu": [crossing[0], crossing[1]],
                    "reason": "unbuildable_crossing",
                })
                continue
            tangent = _inward_tangent(line, crossing, domain)
            stub_end = [
                crossing[0] + tangent[0] * TRANSITION_STUB_LENGTH_GU,
                crossing[1] + tangent[1] * TRANSITION_STUB_LENGTH_GU,
            ]
            approach_id = f"approach_{candidate_id}_{index:04d}"
            index += 1
            approach_ids.append(approach_id)
            approaches.append({
                "approach_id": approach_id,
                "source_edge_id": edge_id,
                "crossing_plan_gu": [crossing[0], crossing[1]],
                "inward_tangent": [tangent[0], tangent[1]],
                "mandatory": True,
                "outside_polyline_plan_gu": outside,
                "inside_polyline_plan_gu": inside,
                "transition_stub_plan_gu": [
                    [crossing[0], crossing[1]],
                    stub_end,
                ],
            })
        ledger.append({
            "source_edge_id": edge_id,
            "outside_polylines_plan_gu": [_coords(g) for g in outside_lines],
            "inside_polylines_plan_gu": [_coords(g) for g in inside_lines],
            "approach_ids": approach_ids,
            "invalid_crossings": invalid,
        })
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "rewrite_domain": {"polygon": domain_ring},
        "approaches": approaches,
        "edge_ledger": ledger,
    }


def build_site_approaches(
    ctx: SiteContext,
    network: AlignedNetwork,
    *,
    candidate_id: str = "c00",
    radius_gu: Optional[float] = None,
    margin_gu: float = REWRITE_MARGIN_GU,
    domain_ring: Optional[list] = None,
) -> dict[str, Any]:
    """Rewrite domain + approaches from an aligned network in world GU."""
    if domain_ring is None:
        domain_ring = build_rewrite_domain(ctx, radius_gu=radius_gu, margin_gu=margin_gu)
    origin = ctx.origin_world_gu
    span = ctx.span_gu
    world_edges = network.edges_in_rect(
        origin[0], origin[1], origin[0] + span[0], origin[1] + span[1],
    )
    plan_edges = []
    for edge in world_edges:
        chain = network.edge_site_chain(edge.id, origin)
        plan_edges.append({"id": edge.id, "plan_polyline": chain})
    return extract_approaches(ctx, domain_ring, plan_edges, candidate_id=candidate_id)


def write_approaches_diagnostic(
    ctx: SiteContext,
    product: dict,
    *,
    topdown_path: Path,
    survey: dict,
    out_png: Path,
    full_site: bool = False,
) -> None:
    """Overlay rewrite domain, outside roads, stubs, and crossings on topdown."""
    from PIL import Image, ImageDraw, ImageFont

    image, mapping = diagnostic_view({"_diagnostic_bounds": [product["rewrite_domain"]["polygon"]]}, topdown_path, survey, full_site=full_site)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    def to_px(pt) -> tuple[int, int]:
        return _plan_to_px(float(pt[0]), float(pt[1]), mapping)

    domain = product["rewrite_domain"]["polygon"]
    domain_px = [to_px(p) for p in domain]
    if len(domain_px) >= 3:
        draw.polygon(domain_px, fill=(0, 170, 255, 50), outline=(0, 220, 255, 220))

    for entry in product.get("edge_ledger", []):
        for poly in entry.get("outside_polylines_plan_gu", []):
            if len(poly) < 2:
                continue
            draw.line([to_px(p) for p in poly], fill=(255, 255, 255, 220), width=2)

    for approach in product.get("approaches", []):
        stub = approach.get("transition_stub_plan_gu") or []
        if len(stub) >= 2:
            draw.line([to_px(p) for p in stub], fill=(80, 255, 80, 255), width=3)
        crossing = approach["crossing_plan_gu"]
        px, py = to_px(crossing)
        r = 5
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 0, 180, 255))
        label = str(approach.get("source_edge_id", ""))
        if font is not None and label:
            draw.text((px + 7, py - 10), label, fill=(255, 255, 80, 255), font=font)

    Image.alpha_composite(image, overlay).save(out_png)
