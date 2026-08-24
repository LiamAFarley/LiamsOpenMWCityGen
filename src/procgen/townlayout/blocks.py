"""Road-dependent buildable-block insets for V2 townlayout (Phase 13).

Purpose
-------
Subtract road corridors and a palisade setback from each inner patch to
leave ``buildable_block`` polygons.  This is the Gate A 2D street map.

Inputs
------
Phase 12 candidate with roads, wall, and wards.

Outputs
-------
``buildable_blocks`` on the candidate plus a blocks diagnostic PNG.

Pipeline position
-----------------
V2 townlayout Phase 13 insets; no parcels/VTEX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .constants import MIN_PATCH_AREA_GU2
from .geometry import normalize_ring, polygon_from_ring, simple_polygon_parts
from .site_context import SiteContext, _plan_to_px, diagnostic_view
from .validate import TownLayoutError

WALL_HALF_GU = 80.0
WATER_AREA_SLOP_GU2 = 128.0 ** 2
WARD_SETBACK = {
    "market": 0.0,
    "keep": 0.0,
    "craft": 48.0,
    "residential": 80.0,
    "outskirts": 128.0,
}


def _ward_of(pid: str, wards: list[dict]) -> str:
    for ward in wards:
        if pid in ward.get("patch_ids", []):
            return ward["ward_type"]
    return "residential"


def _polygonal(geometry):
    """Discard zero-dimensional overlay remnants before polygon differences."""
    if geometry.is_empty:
        return Polygon()
    if geometry.geom_type == "Polygon":
        return geometry
    parts = [g for g in getattr(geometry, "geoms", []) if g.geom_type == "Polygon" and g.area > 0]
    return unary_union(parts) if parts else Polygon()


def _measure_patch_partition(poly: Polygon, components: dict[str, Polygon],
                             patch_id: str) -> dict[str, float]:
    """Measure an independently constructed, mutually exclusive partition."""
    names = ("water", "protected", "emitted", "reserved")
    geoms = [components[name] for name in names]
    union = unary_union(geoms)
    gap = float(poly.difference(union).area)
    outside = float(union.difference(poly).area)
    overlap = max(0.0, sum(float(g.area) for g in geoms) - float(union.area))
    residual = float(poly.symmetric_difference(union).area)
    if max(gap, outside, overlap, residual) > WATER_AREA_SLOP_GU2:
        raise TownLayoutError(
            f"invalid_polygon: block partition residual on {patch_id} "
            f"gap={gap:.3f} outside={outside:.3f} overlap={overlap:.3f}")
    return {
        "water_area_gu2": float(components["water"].area),
        "protected_area_gu2": float(components["protected"].area),
        "block_verge_area_gu2": float(components["emitted"].area),
        "corridor_area_gu2": float(components["reserved"].area),
        "gap_area_gu2": gap, "outside_area_gu2": outside,
        "overlap_area_gu2": overlap, "reconciliation_residual_gu2": residual,
    }


def inset_blocks(candidate: dict, *, water_polygons: Optional[list] = None,
                 apply_ward_setback: bool = True) -> dict[str, Any]:
    """Subtract corridors, protected spaces, and water from every city patch.

    ``apply_ward_setback`` is the compatibility switch for stamp-first stages.
    The historical Phase 13 caller keeps the ward setback; Stage 07 asks for
    the actual curb-facing block and applies its family-specific setback once,
    in the placement walk, rather than stacking two setback systems.
    """
    corridors = []
    for road in candidate.get("roads") or []:
        geom = road.get("polyline") or []
        if len(geom) < 2:
            continue
        width = float(road.get("clear_width_gu") or 0.0)
        if width <= 0:
            continue
        line = LineString([(float(p[0]), float(p[1])) for p in geom])
        if line.length <= 0:
            continue
        corridors.append(line.buffer(width / 2.0, cap_style=2, join_style=2))
    wall_buf = None
    wall_lane_strips = []
    wall = candidate.get("wall")
    if wall and wall.get("planning_polygon"):
        wall_poly = polygon_from_ring(wall["planning_polygon"])
        wall_lane_strips = [s for s in wall.get("strips", []) if s.get("mode") == "wall_lane"]
        if wall_lane_strips:
            wall_buf = unary_union([polygon_from_ring(s["polygon"]) for s in wall_lane_strips])

    water_parts = water_polygons
    if water_parts is None:
        water_parts = candidate.get("water_polygons") or []
    water_parts = [polygon_from_ring(w) if not isinstance(w, Polygon) else w
                   for w in water_parts]
    water_union = unary_union(water_parts) if water_parts else None

    blocks = []
    verges = []
    seq = 0
    inner = [
        p for p in candidate["patches"]
        if p.get("inside_city")
    ]
    protected = []
    for space in candidate.get("open_spaces") or []:
        if space.get("kind") in ("plaza", "court", "park"):
            protected.append(polygon_from_ring(space["polygon"]))
    protected_union = unary_union(protected) if protected else None
    patch_metrics = []
    for patch in inner:
        poly = polygon_from_ring(patch["polygon"])
        original_area = float(poly.area)
        corridor_union = unary_union(corridors) if corridors else None
        if wall_buf is not None and patch.get("inside_wall"):
            corridor_union = unary_union([x for x in (corridor_union, wall_buf) if x is not None])
        water_geom = poly.intersection(water_union) if water_union is not None else Polygon()
        water_geom = _polygonal(water_geom)
        corridor_geom = (poly.intersection(corridor_union)
                         if corridor_union is not None else Polygon())
        corridor_geom = _polygonal(corridor_geom)
        corridor_exclusive = corridor_geom.difference(water_geom)
        corridor_exclusive = _polygonal(corridor_exclusive)
        protected_geom = (poly.intersection(protected_union)
                          if protected_union is not None else Polygon())
        protected_geom = _polygonal(protected_geom)
        protected_geom = protected_geom.difference(water_geom).difference(corridor_exclusive)
        remaining = poly.difference(water_geom).difference(protected_geom).difference(corridor_exclusive)
        setback = (WARD_SETBACK[_ward_of(patch["patch_id"], candidate.get("wards") or [])]
                   if apply_ward_setback else 0.0)
        before_setback = remaining
        if setback > 0 and remaining.geom_type == "Polygon" and remaining.area > 0:
            shrunk = remaining.buffer(-setback)
            if not shrunk.is_empty and shrunk.area > 0:
                remaining = shrunk
        setback_geom = before_setback.difference(remaining)
        parts: list[Polygon] = []
        try:
            parts = (simple_polygon_parts(remaining, area_tolerance=WATER_AREA_SLOP_GU2)
                     if not remaining.is_empty else [])
        except TownLayoutError as exc:
            raise TownLayoutError(f"{exc} on {patch['patch_id']}") from exc
        parts = [p for p in parts if p.geom_type == "Polygon"]
        parts.sort(key=lambda p: (-float(p.area), float(p.centroid.x), float(p.centroid.y)))
        threshold = MIN_PATCH_AREA_GU2 / 8.0
        qualifying = [p for p in parts if p.area >= threshold]
        # Shapely can retain zero-area polygon shells at raster/corridor
        # contacts.  They are not serializable polygons and must be reserved,
        # not emitted as blocks or verge records.
        slivers = [p for p in parts if p.area < threshold and p.area > 1.0]
        discarded_slivers = [p for p in parts if p.area <= 1.0]
        for part_no, part in enumerate(qualifying):
            ring = normalize_ring([[c[0], c[1]] for c in part.exterior.coords])["ring"]
            block_item = {
                "block_id": f"block_{patch['patch_id']}_part{part_no:02d}",
                "patch_id": patch["patch_id"],
                "polygon": ring,
                "wall_strip_ids": sorted(s["strip_id"] for s in wall_lane_strips
                    if polygon_from_ring(ring).intersection(polygon_from_ring(s["polygon"])).area > 1.0),
            }
            blocks.append(block_item)
            seq += 1
        for part_no, extra in enumerate(slivers):
            ring = normalize_ring([[c[0], c[1]] for c in extra.exterior.coords])["ring"]
            verges.append({
                "space_id": f"space_verge_{patch['patch_id']}_part{part_no:02d}",
                "kind": "verge", "reason": "water_or_corridor_sliver",
                "polygon": ring,
            })
        emitted_geom = unary_union(qualifying + slivers) if (qualifying or slivers) else Polygon()
        # Degenerate shells have zero area; omitting them from the emitted
        # partition is within the existing raster reconciliation slop.
        reserved_geom = unary_union([corridor_exclusive, setback_geom]).difference(emitted_geom)
        measured = _measure_patch_partition(
            poly, {"water": water_geom, "protected": protected_geom,
                   "emitted": emitted_geom, "reserved": reserved_geom},
            patch["patch_id"])
        patch_metrics.append({
            "patch_id": patch["patch_id"], "patch_area_gu2": original_area,
            **measured, "reconciliation_error_gu2": measured["reconciliation_residual_gu2"],
        })
        # Corridor must not overlap remaining block (hard error).
        if qualifying and corridors:
            union_c = unary_union(corridors)
            if any(p.intersection(union_c).area > 1.0 for p in qualifying):
                raise TownLayoutError(
                    f"invalid_polygon: block overlaps road corridor on {patch['patch_id']}")

    reports = list(candidate.get("reports") or [])
    reports.append({
        "stage": "blocks",
        "status": "ok",
        "message": f"blocks={len(blocks)} verges={len(verges)} water_overlap={sum(m['water_area_gu2'] for m in patch_metrics):.3f}",
    })
    open_spaces = list(candidate.get("open_spaces") or []) + verges
    out = dict(candidate)
    out["buildable_blocks"] = blocks
    out["open_spaces"] = open_spaces
    out["reports"] = reports
    out["water_metrics"] = dict(candidate.get("water_metrics") or {})
    block_water_overlap = 0.0
    if water_union is not None:
        block_water_overlap = sum(
            polygon_from_ring(block["polygon"]).intersection(water_union).area
            for block in blocks)
    out["water_metrics"].update({
        "block_water_intersection_gu2": float(block_water_overlap),
        "block_patch_reconciliation": patch_metrics,
        "max_partition_gap_gu2": max((m["gap_area_gu2"] for m in patch_metrics), default=0.0),
        "max_partition_outside_gu2": max((m["outside_area_gu2"] for m in patch_metrics), default=0.0),
        "max_partition_overlap_gu2": max((m["overlap_area_gu2"] for m in patch_metrics), default=0.0),
        "max_reconciliation_residual_gu2": max((m["reconciliation_residual_gu2"] for m in patch_metrics), default=0.0),
    })
    if water_parts:
        out["water_polygons"] = [normalize_ring([[c[0], c[1]] for c in p.exterior.coords])["ring"]
                                 for p in water_parts]
    return out


def write_blocks_diagnostic(
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

    # Water is deliberately loud: blue fill is land exclusion, red is an
    # actual serialized overlap (should be empty apart from raster slop).
    for water in product.get("water_polygons") or []:
        if len(water) >= 3:
            draw.polygon([to_px(p) for p in water], fill=(20, 80, 255, 150), outline=(0, 20, 255, 255))
    for road in product.get("roads") or []:
        geom = road.get("polyline") or []
        if len(geom) < 2:
            continue
        hier = road.get("hierarchy") or "street"
        color = {
            "arterial": (200, 40, 40, 230),
            "street": (230, 200, 50, 230),
            "lane": (160, 140, 70, 200),
            "regional_approach": (180, 80, 40, 230),
        }.get(hier, (200, 200, 200, 200))
        width = {"arterial": 4, "street": 2, "lane": 1, "regional_approach": 5}.get(hier, 2)
        draw.line([to_px(p) for p in geom], fill=color, width=width)
    for block in product.get("buildable_blocks") or []:
        ring = block.get("polygon") or []
        if len(ring) >= 3:
            draw.polygon([to_px(p) for p in ring], fill=(80, 160, 90, 90),
                         outline=(20, 60, 30, 220))
            if product.get("water_polygons"):
                from shapely.ops import unary_union
                if polygon_from_ring(ring).intersection(unary_union(
                        [polygon_from_ring(w) for w in product["water_polygons"]])).area > 0.0:
                    draw.polygon([to_px(p) for p in ring], fill=(255, 0, 0, 210))
    wall = product.get("wall")
    if wall:
        ring = wall.get("planning_polygon") or []
        if len(ring) >= 3:
            pts = [to_px(p) for p in ring] + [to_px(ring[0])]
            draw.line(pts, fill=(90, 50, 20, 255), width=3)
    Image.alpha_composite(image, overlay).save(out_png)
