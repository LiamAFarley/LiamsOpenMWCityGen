"""Cityforge T1.1 D-PLAN overlay renderer - deterministic 2D compositing.

Pipeline position
------------------
Second executable gate of the Cityforge plan chain: renders a *validated*
``city_plan.json`` as a legible 2D overlay on the accepted 4096x4096
``site_topdown.png`` terrain render (Pillow only; no Blender).  Runs the
strict validator internally and refuses to render any plan with errors, so
an overlay file only ever exists for a validated plan.  The first real
Falkreath overlay gate is T1.6 (user review) - this tool is proven on the
synthetic fixture only.

Coordinate mapping
------------------
Exact GU<->pixel mapping from ``site_survey.json#frame.render_mapping``:
``px_x = origin_px[0] + (gu_x - origin_gu[0]) * px_per_gu`` and
``px_y = origin_px[1] - (gu_y - origin_gu[1]) * px_per_gu`` with
``y_down_image`` true, so plan-frame GU (0,0) is the SW corner and +y
(north) is upward on screen.

Output canvas
-------------
Width 4096 (the site render).  Height 4096 + BANNER_BAND + LEGEND_BAND:
the banner band sits above the map, the legend band below it, so no title
or legend pixel can ever cover planned geometry (verified in the render
audit as ``geometry_under_bands_px``, always 0 by construction).

Usage
-----
    python tools/cityforge/render_plan.py --plan <city_plan.json> --out overlay.png
        [--banner-text "SYNTHETIC VALIDATION FIXTURE - NOT A FALKREATH DESIGN"]
        [--audit-out render_audit.json] [bundle paths as in validate_city_plan.py]

Exit codes: 0 = rendered; 1 = plan invalid (nothing written);
2 = configuration failure.

Determinism
-----------
Same inputs -> byte-identical PNG (Pillow's PNG encoder is deterministic
for identical pixel data; fonts are Pillow's embedded deterministic font).
The render audit records every input hash plus the output SHA-256.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen import cityplan  # noqa: E402
from procgen.censusio import sha256_file, write_deterministic  # noqa: E402
from procgen.cityplan import (  # noqa: E402
    Bundle,
    SITE_SPAN_GU,
    TILE_SIZE_GU,
    close_ring,
    point_polyline_distance,
    polygon_centroid,
    yaw_hull,
)

CANONICAL_SURVEY = "output/cityforge/sites/falkreath_v1/site_survey.json"
CANONICAL_BRIEF = "output/cityforge/briefs/falkreath_v1/kit_brief.json"
CANONICAL_PALETTE = "output/cityforge/briefs/falkreath_v1/region_palette.json"
CANONICAL_LIBRARIES = (
    "output/cityforge/stamps/karthgad_nord_v1.json",
    "output/cityforge/stamps/markarth_side_stone_v1.json",
)
CANONICAL_CENTERLINES = ("output/mapdata/roads/tamriel_aligned_centerlines_v1/"
                         "tamriel_aligned_centerlines_v1.json")
CANONICAL_TOPDOWN = "output/cityforge/sites/falkreath_v1/site_topdown.png"

BANNER_BAND_PX = 56
LEGEND_BAND_PX = 232
MAP_ORIGIN_Y_PX = BANNER_BAND_PX

# Fixed deterministic drawing palette (plan-frame layers).
COLOR = {
    "district_fill": (255, 200, 60, 46),
    "district_outline": (255, 200, 60, 255),
    "district_label": (120, 80, 0, 255),
    "road_street": (200, 40, 40, 255),
    "road_approach": (230, 120, 30, 255),
    "road_path": (60, 120, 60, 255),
    "road_dock_lane": (0, 100, 180, 255),
    "road_label": (90, 10, 10, 255),
    "road_arrow": (0, 0, 0, 255),
    "external_marker": (180, 0, 180, 255),
    "centerline_context": (120, 120, 120, 200),
    "lot_fill": (40, 140, 255, 60),
    "lot_outline": (0, 70, 180, 255),
    "lot_label": (0, 40, 120, 255),
    "door_anchor": (255, 255, 255, 255),
    "door_anchor_edge": (0, 0, 0, 255),
    "heading_arrow": (0, 150, 0, 255),
    "warning_marker": (255, 120, 0, 255),
    "boundary_palisade": (120, 70, 30, 255),
    "boundary_label": (70, 40, 10, 255),
    "gate_marker": (255, 220, 0, 255),
    "feature_dock": (0, 140, 200, 255),
    "feature_well": (0, 90, 150, 255),
    "feature_statue": (150, 90, 0, 255),
    "feature_market": (200, 90, 160, 255),
    "feature_boat": (60, 140, 190, 255),
    "feature_signpost": (100, 70, 40, 255),
    "feature_keep_trees": (30, 120, 30, 255),
    "edit_fill": (200, 60, 200, 40),
    "edit_outline": (150, 20, 150, 255),
    "hint_outline": (0, 160, 0, 255),
    "legend_bg": (245, 245, 245, 255),
    "legend_text": (20, 20, 20, 255),
    "banner_bg": (30, 30, 30, 255),
    "banner_text": (255, 60, 60, 255),
}
ROAD_CLASS_COLOR = {
    "street": COLOR["road_street"],
    "approach": COLOR["road_approach"],
    "path": COLOR["road_path"],
    "dock_lane": COLOR["road_dock_lane"],
}
FEATURE_COLOR = {
    "well": COLOR["feature_well"],
    "statue": COLOR["feature_statue"],
    "market_stalls": COLOR["feature_market"],
    "dock": COLOR["feature_dock"],
    "boat": COLOR["feature_boat"],
    "signpost": COLOR["feature_signpost"],
    "keep_trees": COLOR["feature_keep_trees"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic Cityforge D-PLAN overlay renderer (T1.1)")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--audit-out", default=None,
                        help="render audit JSON path (default: <out>.audit.json)")
    parser.add_argument("--site-survey", default=CANONICAL_SURVEY)
    parser.add_argument("--kit-brief", default=CANONICAL_BRIEF)
    parser.add_argument("--region-palette", default=CANONICAL_PALETTE)
    parser.add_argument("--stamp-libraries", nargs="+", default=list(CANONICAL_LIBRARIES))
    parser.add_argument("--centerlines", default=CANONICAL_CENTERLINES)
    parser.add_argument("--topdown", default=CANONICAL_TOPDOWN)
    parser.add_argument("--banner-text", default=None,
                        help="text drawn in the top banner band (fixture runs "
                             "pass the SYNTHETIC VALIDATION FIXTURE banner)")
    return parser


def _gu_to_px(mapping: dict, x: float, y: float) -> tuple[float, float]:
    px_x = mapping["origin_px"][0] + x * mapping["px_per_gu"]
    px_y = mapping["origin_px"][1] - y * mapping["px_per_gu"]
    return (px_x, px_y)


def _load_plan(plan_path: Path) -> dict:
    with open(plan_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_hulls(plan: dict, bundle: Bundle, result: dict) -> dict:
    """lot_id -> world plan-frame hull points for every resolved lot
    (explicit or selector pick, per the validator's resolution report)."""
    out: dict = {}
    resolution = {r.get("lot_id"): r for r in
                  result.get("summary", {}).get("lot_resolution", [])}
    for lot in plan.get("lots", []):
        if not isinstance(lot, dict) or not isinstance(lot.get("lot_id"), str):
            continue
        lot_id = lot["lot_id"]
        res = resolution.get(lot_id)
        stamp_id = (res or {}).get("stamp_id")
        if not stamp_id or stamp_id not in bundle.stamp_geometry:
            continue
        hull = bundle.stamp_geometry[stamp_id].get("footprint", {}).get("hull_xy_rel")
        position = lot.get("position")
        yaw = lot.get("yaw_deg")
        if not (isinstance(hull, list) and isinstance(position, list) and
                len(position) == 2 and isinstance(yaw, (int, float))):
            continue
        out[lot_id] = yaw_hull(hull, float(yaw),
                               (float(position[0]), float(position[1])))
    return out


def _font(size: int):
    return ImageFont.load_default(size=size)


def _label(draw, xy, text, fill, font, anchor="mm", stroke=2):
    draw.text(xy, text, font=font, fill=fill,
              stroke_width=stroke, stroke_fill=(255, 255, 255, 255), anchor=anchor)


def _label_box(draw, xy, text, fill, font, pad=3):
    """Label on an opaque white rounded box: readable over any layer
    (outline, fill, terrain) without relying on stroke contrast."""
    box = draw.textbbox((0, 0), text, font=font, anchor="mm",
                        stroke_width=2)
    w = box[2] - box[0] + 2 * pad
    h = box[3] - box[1] + 2 * pad
    left, top = xy[0] - w / 2, xy[1] - h / 2
    draw.rounded_rectangle((left, top, left + w, top + h), radius=3,
                           fill=(255, 255, 255, 235),
                           outline=(60, 60, 60, 255), width=1)
    draw.text(xy, text, font=font, fill=fill, anchor="mm", stroke_width=1,
              stroke_fill=(255, 255, 255, 255))


def _composite_polygon(base: "Image.Image", points, fill, outline=None,
                       width: int = 1) -> None:
    """Draw a polygon with an alpha fill onto ``base`` using a temp layer +
    ``alpha_composite``.  ``ImageDraw`` on an RGBA image *replaces* pixels
    (alpha included) instead of blending, which would erase everything
    underneath a translucent fill; compositing preserves translucency."""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.polygon(points, fill=fill, outline=outline, width=width)
    base.alpha_composite(layer)


def render_plan(plan: dict, bundle: Bundle, result: dict, topdown_path: Path,
                banner_text: Optional[str],
                keep_overlay: bool = False) -> tuple["Image.Image", dict, Optional["Image.Image"]]:
    """Compose the overlay; returns (canvas image, render audit dict,
    overlay image when ``keep_overlay`` is set - a test/debug aid)."""
    mapping = bundle.survey_frame["render_mapping"]["site_topdown.png"]
    topdown = Image.open(topdown_path).convert("RGBA")
    if topdown.size != tuple(mapping["resolution"]):
        raise cityplan.BundleError(
            f"site_topdown.png is {topdown.size}, expected "
            f"{tuple(mapping['resolution'])} from the survey mapping")

    canvas = Image.new("RGBA", (4096, 4096 + BANNER_BAND_PX + LEGEND_BAND_PX),
                       (245, 245, 245, 255))
    canvas.paste(topdown, (0, MAP_ORIGIN_Y_PX))
    draw = ImageDraw.Draw(canvas)

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # ---- background context: aligned centerline edges inside the site ----
    # The canonical bundle carries an AlignedNetwork (the one supported road
    # entry point); synthetic unit fixtures may still inject a plain dict.
    plan_origin = tuple(bundle.survey_frame["origin_gu"])
    window_edges = 0
    if bundle.aligned_network is not None:
        context_edges = bundle.aligned_network.edges.values()
        chain_getter = lambda edge: edge.smooth_gu_polyline  # noqa: E731
    else:
        context_edges = bundle.centerlines["edges"]
        chain_getter = lambda edge: edge["smooth_gu_polyline"]  # noqa: E731
    for edge in context_edges:
        chain = chain_getter(edge)
        pts = [(q[0] - plan_origin[0], q[1] - plan_origin[1]) for q in chain]
        if not any(0.0 <= p[0] <= SITE_SPAN_GU and 0.0 <= p[1] <= SITE_SPAN_GU
                   for p in pts):
            continue
        px = [_gu_to_px(mapping, p[0], p[1]) for p in pts]
        odraw.line(px, fill=COLOR["centerline_context"], width=1, joint="curve")
        window_edges += 1

    # ---- districts (translucent fill + labels) ----------------------------
    for idx, district in enumerate(plan.get("districts", [])):
        if not isinstance(district, dict):
            continue
        poly = district.get("polygon")
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        px = [_gu_to_px(mapping, p[0], p[1]) for p in poly]
        _composite_polygon(overlay, px, COLOR["district_fill"],
                           outline=COLOR["district_outline"], width=3)
        centroid = polygon_centroid(poly)
        cpx = _gu_to_px(mapping, centroid[0], centroid[1])
        _label(odraw, (cpx[0], cpx[1]),
               district.get("district_id", f"d{idx}"),
               COLOR["district_label"], _font(20))
        zone = district.get("texture_zone")
        if zone:
            _label(odraw, (cpx[0], cpx[1] + 22), f"zone {zone}",
                   (140, 100, 30, 255), _font(14))

    # ---- roads ------------------------------------------------------------
    roads = plan.get("roads", [])
    road_px: dict = {}
    for road in roads:
        if not isinstance(road, dict) or not isinstance(road.get("road_id"), str):
            continue
        polyline = road.get("polyline")
        if not isinstance(polyline, list) or len(polyline) < 2:
            continue
        px = [_gu_to_px(mapping, p[0], p[1]) for p in polyline]
        road_px[road["road_id"]] = px
        width_px = max(2.0, float(road.get("width_gu", 512.0)) * mapping["px_per_gu"])
        color = ROAD_CLASS_COLOR.get(road.get("class"), COLOR["road_path"])
        odraw.line(px, fill=color, width=int(round(width_px)), joint="curve")
        # direction arrow at the far end of the last segment
        a, b = px[-2], px[-1]
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        head = 14
        odraw.polygon([
            (b[0], b[1]),
            (b[0] - head * math.cos(ang - 0.45), b[1] - head * math.sin(ang - 0.45)),
            (b[0] - head * math.cos(ang + 0.45), b[1] - head * math.sin(ang + 0.45)),
        ], fill=COLOR["road_arrow"])
        # id label near the polyline midpoint (by arc length)
        total = sum(math.hypot(px[i + 1][0] - px[i][0], px[i + 1][1] - px[i][1])
                    for i in range(len(px) - 1))
        target, acc = total / 2.0, 0.0
        mid = px[0]
        for i in range(len(px) - 1):
            seg = math.hypot(px[i + 1][0] - px[i][0], px[i + 1][1] - px[i][1])
            if acc + seg >= target:
                t = (target - acc) / seg if seg else 0.0
                mid = (px[i][0] + t * (px[i + 1][0] - px[i][0]),
                       px[i][1] + t * (px[i + 1][1] - px[i][1]))
                break
            acc += seg
        _label(odraw, (mid[0], mid[1]),
               road["road_id"], COLOR["road_label"], _font(16))

    # ---- external-connection markers --------------------------------------
    for road in roads:
        if not isinstance(road, dict):
            continue
        polyline = road.get("polyline")
        if not isinstance(polyline, list) or len(polyline) < 2:
            continue
        px = road_px.get(road["road_id"], [])
        if not px:
            continue
        for ref in road.get("connects", []):
            if not isinstance(ref, str):
                continue
            anchor = _external_px(bundle, mapping, ref, plan)
            if anchor is None:
                continue
            # nearest road endpoint to the external anchor
            end = min(px, key=lambda q: math.hypot(q[0] - anchor[0], q[1] - anchor[1]))
            odraw.polygon([(end[0] - 5, end[1] - 5), (end[0] + 5, end[1] - 5),
                           (end[0] + 5, end[1] + 5), (end[0] - 5, end[1] + 5)],
                          fill=COLOR["external_marker"])
            _label(odraw, (end[0], end[1] + 12),
                   ref, COLOR["external_marker"], _font(11))

    # ---- lots: footprints, door anchors, heading arrows, ids, warnings ----
    lots = plan.get("lots", [])
    hulls = _resolve_hulls(plan, bundle, result)
    warning_lots = set(result.get("summary", {}).get("lot_warnings", {}).keys())
    lot_id_by_name = {l.get("lot_id"): l for l in lots if isinstance(l, dict)}
    for lot in lots:
        if not isinstance(lot, dict) or not isinstance(lot.get("lot_id"), str):
            continue
        lot_id = lot["lot_id"]
        hull = hulls.get(lot_id)
        position = lot.get("position")
        if isinstance(position, list) and len(position) == 2:
            ax = _gu_to_px(mapping, position[0], position[1])
            if hull:
                px = [_gu_to_px(mapping, p[0], p[1]) for p in hull]
                _composite_polygon(overlay, px, COLOR["lot_fill"],
                                   outline=COLOR["lot_outline"], width=3)
            # access-heading arrow (starts at the anchor edge so it stays
            # visible for headings that point back into the hull)
            yaw = float(lot.get("yaw_deg") or 0.0)
            request = lot.get("request") or {}
            stamp_id = request.get("stamp_id")
            heading = yaw
            if stamp_id and stamp_id in bundle.stamp_geometry:
                heading += math.degrees(
                    bundle.stamp_geometry[stamp_id].get("access_heading_rad", 0.0))
            arrow_len = 256.0 * mapping["px_per_gu"]
            rad = math.radians(heading)
            start = (ax[0] + 8.0 * math.cos(rad), ax[1] - 8.0 * math.sin(rad))
            tip = (ax[0] + arrow_len * math.cos(rad),
                   ax[1] - arrow_len * math.sin(rad))
            odraw.line((start[0], start[1], tip[0], tip[1]),
                       fill=COLOR["heading_arrow"], width=3)
            # door anchor (drawn last so the arrow never covers it)
            r = 6
            odraw.ellipse((ax[0] - r, ax[1] - r, ax[0] + r, ax[1] + r),
                          fill=COLOR["door_anchor"],
                          outline=COLOR["door_anchor_edge"], width=2)
            # lot id
            _label_box(odraw, (ax[0], ax[1] - 24), lot_id,
                       COLOR["lot_label"], _font(13))
            # warning marker
            if lot_id in warning_lots:
                w = 10
                cy = ax[1] - 46
                odraw.ellipse((ax[0] - w, cy - w, ax[0] + w, cy + w),
                              fill=COLOR["warning_marker"],
                              outline=(0, 0, 0, 255), width=2)
                _label(odraw, (ax[0], cy),
                       "!", (0, 0, 0, 255), _font(13))

    # ---- boundaries + gates ------------------------------------------------
    for boundary in plan.get("boundaries", []):
        if not isinstance(boundary, dict):
            continue
        poly = boundary.get("polygon")
        if not isinstance(poly, list) or len(poly) < 4:
            continue
        ring = close_ring(poly)
        px = [_gu_to_px(mapping, p[0], p[1]) for p in ring]
        odraw.line(px, fill=COLOR["boundary_palisade"], width=5, joint="curve")
        c = polygon_centroid(poly)
        cpx = _gu_to_px(mapping, c[0], c[1])
        _label(odraw, (cpx[0], cpx[1]),
               boundary.get("boundary_id", ""), COLOR["boundary_label"], _font(15))
        for gate in boundary.get("gates", []):
            if not isinstance(gate, dict):
                continue
            gpos = gate.get("position")
            if not isinstance(gpos, list) or len(gpos) != 2:
                continue
            gx, gy = _gu_to_px(mapping, gpos[0], gpos[1])
            s = 8
            odraw.polygon([(gx, gy - s), (gx + s, gy),
                           (gx, gy + s), (gx - s, gy)],
                          fill=COLOR["gate_marker"], outline=(0, 0, 0, 255), width=2)
            _label(odraw, (gx, gy - 16),
                   gate.get("gate_id", ""), (120, 100, 0, 255), _font(12))

    # ---- features ----------------------------------------------------------
    for feature in plan.get("features", []):
        if not isinstance(feature, dict):
            continue
        fpos = feature.get("position")
        if not isinstance(fpos, list) or len(fpos) != 2:
            continue
        fx, fy = _gu_to_px(mapping, fpos[0], fpos[1])
        kind = feature.get("kind", "")
        color = FEATURE_COLOR.get(kind, (150, 150, 150, 255))
        if kind == "dock":
            odraw.polygon([(fx - 12, fy + 8), (fx + 12, fy + 8), (fx + 6, fy - 8),
                           (fx - 6, fy - 8)], fill=color, outline=(0, 0, 0, 255), width=2)
        elif kind == "well":
            odraw.ellipse((fx - 8, fy - 8, fx + 8, fy + 8), fill=color,
                          outline=(0, 0, 0, 255), width=2)
            odraw.ellipse((fx - 3, fy - 3, fx + 3, fy + 3), fill=(0, 0, 0, 255))
        elif kind == "statue":
            odraw.polygon([(fx, fy - 10), (fx - 6, fy + 6), (fx + 6, fy + 6)],
                          fill=color, outline=(0, 0, 0, 255), width=2)
        elif kind == "market_stalls":
            odraw.rectangle((fx - 12, fy - 8, fx + 12, fy + 8), fill=color,
                            outline=(0, 0, 0, 255), width=2)
        elif kind == "boat":
            odraw.polygon([(fx - 12, fy + 6), (fx + 12, fy + 6), (fx + 4, fy - 6),
                           (fx - 4, fy - 6)], fill=color, outline=(0, 0, 0, 255), width=2)
            odraw.line((fx, fy - 6, fx, fy - 14), fill=(0, 0, 0, 255), width=2)
        elif kind == "signpost":
            odraw.line((fx, fy - 10, fx, fy + 8), fill=(0, 0, 0, 255), width=3)
            odraw.line((fx - 9, fy - 4, fx + 9, fy - 4), fill=color, width=3)
        elif kind == "keep_trees":
            odraw.ellipse((fx - 8, fy - 14, fx + 8, fy + 2), fill=color,
                          outline=(0, 0, 0, 255), width=2)
            odraw.line((fx, fy + 2, fx, fy + 8), fill=(0, 0, 0, 255), width=2)
        else:
            odraw.ellipse((fx - 6, fy - 6, fx + 6, fy + 6), fill=color,
                          outline=(0, 0, 0, 255), width=2)
        _label(odraw, (fx, fy + 18), feature.get("feature_id", ""),
               (60, 60, 60, 255), _font(12))

    # ---- terrain edits -----------------------------------------------------
    for edit in plan.get("terrain_edits", []):
        if not isinstance(edit, dict):
            continue
        poly = edit.get("polygon")
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        px = [_gu_to_px(mapping, p[0], p[1]) for p in poly]
        _composite_polygon(overlay, px, COLOR["edit_fill"],
                           outline=COLOR["edit_outline"], width=2)
        c = polygon_centroid(poly)
        cpx = _gu_to_px(mapping, c[0], c[1])
        _label(odraw, (cpx[0], cpx[1]),
               edit.get("edit_id", ""), (140, 20, 140, 255), _font(13))

    # ---- wilderness hints --------------------------------------------------
    for hint in plan.get("wilderness_hints", []):
        if not isinstance(hint, dict):
            continue
        poly = hint.get("polygon")
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        ring = close_ring(poly)
        for a, b in zip(ring, ring[1:]):
            ax = _gu_to_px(mapping, a[0], a[1])
            bx = _gu_to_px(mapping, b[0], b[1])
            n = max(2, int(math.hypot(bx[0] - ax[0], bx[1] - ax[1]) / 24))
            for i in range(n + 1):
                t = i / n
                px = ax[0] + t * (bx[0] - ax[0])
                py = ax[1] + t * (bx[1] - ax[1])
                odraw.ellipse((px - 2, py - 2,
                               px + 2, py + 2),
                              fill=COLOR["hint_outline"])
        c = polygon_centroid(poly)
        cpx = _gu_to_px(mapping, c[0], c[1])
        _label(odraw, (cpx[0], cpx[1]),
               f"{hint.get('hint', '')} x{hint.get('density', 1.0)}",
               (0, 110, 0, 255), _font(13))

    # ---- composite ---------------------------------------------------------
    canvas.alpha_composite(overlay, (0, MAP_ORIGIN_Y_PX))
    overlay_saved = overlay if keep_overlay else None

    # ---- banner band -------------------------------------------------------
    if banner_text:
        bdraw = ImageDraw.Draw(canvas)
        bdraw.rectangle((0, 0, 4096, BANNER_BAND_PX), fill=COLOR["banner_bg"])
        bdraw.text((2048, BANNER_BAND_PX / 2), banner_text,
                   font=_font(30), fill=COLOR["banner_text"],
                   stroke_width=2, stroke_fill=(0, 0, 0, 255), anchor="mm")

    # ---- legend band -------------------------------------------------------
    _draw_legend(draw, plan, result, banner_text is not None)

    return (canvas, {
        "canvas_size": list(canvas.size),
        "map_band_px": [0, BANNER_BAND_PX, 4096, BANNER_BAND_PX + 4096],
        "banner_text": banner_text,
        "banner_band_px": [0, 0, 4096, BANNER_BAND_PX],
        "legend_band_px": [0, BANNER_BAND_PX + 4096, 4096,
                           BANNER_BAND_PX + 4096 + LEGEND_BAND_PX],
        "geometry_under_bands_px": 0,
        "drawn": {
            "districts": len([d for d in plan.get("districts", []) if isinstance(d, dict)]),
            "roads": len(roads),
            "lots": len(lots),
            "boundaries": len([b for b in plan.get("boundaries", []) if isinstance(b, dict)]),
            "gates": sum(len(b.get("gates", [])) for b in plan.get("boundaries", [])
                         if isinstance(b, dict)),
            "features": len([f for f in plan.get("features", []) if isinstance(f, dict)]),
            "terrain_edits": len([e for e in plan.get("terrain_edits", []) if isinstance(e, dict)]),
            "texture_zones": len(plan.get("texture_zones", [])),
            "wilderness_hints": len(plan.get("wilderness_hints", [])),
            "external_markers": _count_external_markers(plan, bundle),
            "source_centerline_edges_in_site": window_edges,
            "warning_lot_markers": len(warning_lots),
        },
    }, overlay_saved)


def _count_external_markers(plan: dict, bundle: Bundle) -> int:
    count = 0
    for road in plan.get("roads", []):
        if not isinstance(road, dict):
            continue
        for ref in road.get("connects", []):
            if not isinstance(ref, str):
                continue
            if ref in bundle.edge_ids or ref in bundle.node_ids or \
                    ref in bundle.map_exits:
                count += 1
    return count


def _external_px(bundle: Bundle, mapping: dict, ref: str, plan: dict):
    """Pixel position of an external ref (edge/node/exit) or a plan element
    (road id / gate id) for the connection markers."""
    anchor = cityplan._external_anchor_point(bundle, ref)
    if anchor is not None:
        return _gu_to_px(mapping, anchor[0], anchor[1])
    for road in plan.get("roads", []):
        if isinstance(road, dict) and road.get("road_id") == ref:
            polyline = road.get("polyline")
            if isinstance(polyline, list) and len(polyline) >= 2:
                p = polyline[len(polyline) // 2]
                return _gu_to_px(mapping, p[0], p[1])
    for boundary in plan.get("boundaries", []):
        if not isinstance(boundary, dict):
            continue
        for gate in boundary.get("gates", []):
            if isinstance(gate, dict) and gate.get("gate_id") == ref and \
                    isinstance(gate.get("position"), list) and len(gate["position"]) == 2:
                return _gu_to_px(mapping, gate["position"][0], gate["position"][1])
    return None


def _draw_legend(draw, plan: dict, result: dict, banner: bool) -> None:
    top = MAP_ORIGIN_Y_PX + 4096 + 8
    col_x = 16
    entries = [
        ("Districts", COLOR["district_fill"]),
        ("Roads (street/approach/path/dock_lane)", COLOR["road_street"]),
        ("Road direction arrow", COLOR["road_arrow"]),
        ("External connection marker", COLOR["external_marker"]),
        ("Source centerline (context)", COLOR["centerline_context"]),
        ("Lot footprint (explicit/selector)", COLOR["lot_fill"]),
        ("Door anchor + access heading", COLOR["heading_arrow"]),
        ("Warning marker", COLOR["warning_marker"]),
        ("Boundary palisade ring", COLOR["boundary_palisade"]),
        ("Gate", COLOR["gate_marker"]),
        ("Features (well/statue/market/dock/boat/signpost/trees)", COLOR["feature_dock"]),
        ("Terrain edit", COLOR["edit_outline"]),
        ("Wilderness hint", COLOR["hint_outline"]),
    ]
    row = 0
    for text, color in entries:
        x = col_x + (row // 5) * 1350
        y = top + (row % 5) * 22
        draw.rectangle((x, y + 4, x + 14, y + 18), fill=color,
                       outline=(0, 0, 0, 255), width=1)
        draw.text((x + 20, y + 2), text, font=_font(13), fill=COLOR["legend_text"])
        row += 1
    y = top + 5 * 22 + 6
    draw.text((col_x, y),
              f"plan_id={result.get('plan_id')}  "
              f"errors={result['error_count']}  warnings={result['warning_count']}  "
              f"banner={banner}",
              font=_font(13), fill=COLOR["legend_text"])


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        bundle = Bundle.from_paths(
            site_survey=args.site_survey,
            kit_brief=args.kit_brief,
            region_palette=args.region_palette,
            stamp_libraries=args.stamp_libraries,
            centerlines=args.centerlines,
        )
        plan = _load_plan(Path(args.plan))
        result = cityplan.validate_plan(plan, bundle)
    except cityplan.BundleError as exc:
        print(f"configuration failure: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}", file=sys.stderr)
        return 2

    if not result["valid"]:
        print(f"refusing to render invalid plan: {result['error_count']} errors",
              file=sys.stderr)
        for issue in result["issues"]:
            if issue["severity"] == "error":
                print(f"  [{issue['code']}] {issue['path']}: {issue['message']}",
                      file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas, audit, _ = render_plan(plan, bundle, result, Path(args.topdown),
                                    args.banner_text)
    canvas.save(out_path, format="PNG")

    audit["output_png_sha256"] = sha256_file(out_path)
    audit["output_png_size"] = [canvas.size[0], canvas.size[1]]
    audit["input_hashes"] = dict(sorted(bundle.hashes.items()))
    audit["input_hashes"]["plan"] = sha256_file(Path(args.plan))
    audit["input_hashes"]["site_topdown.png"] = sha256_file(Path(args.topdown))
    audit["validation_summary"] = {
        "valid": result["valid"],
        "issue_count": result["issue_count"],
        "error_count": result["error_count"],
        "warning_count": result["warning_count"],
    }
    audit_path = Path(args.audit_out) if args.audit_out else \
        out_path.with_name(out_path.stem + ".audit.json")
    write_deterministic(audit_path, audit)

    print(f"rendered {out_path} sha256={audit['output_png_sha256']}")
    print(f"audit {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
