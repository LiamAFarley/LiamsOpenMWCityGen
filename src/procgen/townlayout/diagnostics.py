"""Direct vector diagnostic for Phase 21 checkpoints.

Inputs are a checkpoint, the authoritative site top-down image, and survey
metadata.  The renderer draws serialized geometry directly into the city
viewport and includes a compact in-frame legend and counts.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

from .site_context import _plan_to_px, diagnostic_view, resolve_topdown_png
from .validate import TownLayoutError


def render_macro_diagnostic(product: dict, survey_path: str | Path,
                            output_path: str | Path) -> None:
    survey_file = Path(survey_path)
    topdown = resolve_topdown_png(survey_file)
    if topdown is None:
        raise TownLayoutError("missing_diagnostic_input: site_topdown.png")
    survey = json.loads(survey_file.read_text(encoding="utf-8"))
    image, mapping = diagnostic_view(
        {"_diagnostic_bounds": [product.get("city_domain") or [],
                                 (product.get("rewrite_domain") or {}).get("polygon") or []]},
        topdown, survey, full_site=False,
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    to_px = lambda point: _plan_to_px(float(point[0]), float(point[1]), mapping)
    wards = {pid: ward["ward_type"] for ward in product.get("wards", [])
             for pid in ward.get("patch_ids", [])}
    colors = {"market": (220, 40, 160, 125), "craft": (220, 120, 30, 115),
              "residential": (40, 170, 80, 115), "outskirts": (180, 140, 40, 115),
              "keep": (80, 80, 80, 125)}
    for patch in product.get("patches", []):
        ring = patch.get("polygon") or []
        if patch.get("inside_city") and len(ring) >= 3:
            kind = wards.get(patch.get("patch_id"), "residential")
            points = [to_px(point) for point in ring]
            draw.polygon(points, fill=colors.get(kind, colors["residential"]),
                         outline=(20, 20, 20, 220))
            if patch.get("patch_id"):
                centroid = Polygon(ring).representative_point()
                draw.text(to_px((centroid.x, centroid.y)), str(patch["patch_id"]),
                          fill=(255, 255, 255, 255), font=ImageFont.load_default())
    domain = product.get("city_domain") or []
    if len(domain) >= 3:
        draw.line([to_px(point) for point in domain + [domain[0]]],
                  fill=(0, 230, 255, 255), width=3)
    envelope = (product.get("rewrite_domain") or {}).get("polygon") or []
    if len(envelope) >= 3:
        dashed = envelope + [envelope[0]]
        for start, end in zip(dashed, dashed[1:]):
            x0, y0 = to_px(start); x1, y1 = to_px(end)
            steps = max(1, int(((x1-x0)**2 + (y1-y0)**2) ** 0.5 / 8))
            for i in range(0, steps, 2):
                a, b = i / steps, min(1.0, (i + 1) / steps)
                draw.line((x0 + (x1-x0)*a, y0 + (y1-y0)*a,
                           x0 + (x1-x0)*b, y0 + (y1-y0)*b),
                          fill=(120, 220, 255, 150), width=2)
    for edge in product.get("aligned_roads", {}).get("edges", []):
        chain = edge.get("plan_polyline") or []
        if len(chain) >= 2:
            draw.line([to_px(point) for point in chain], fill=(255, 255, 255, 220), width=2)
    for approach in product.get("approaches", []):
        crossing = approach.get("crossing_plan_gu")
        if crossing:
            px, py = to_px(crossing)
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(255, 0, 180, 255))
    legend = (f"R1 MACRO  patches={len(product.get('patches', []))}"
              f" city={sum(bool(p.get('inside_city')) for p in product.get('patches', []))}"
              f" roads={len(product.get('aligned_roads', {}).get('edges', []))}"
              f" approaches={len(product.get('approaches', []))}")
    draw.rectangle((8, 8, min(image.width - 8, 8 + len(legend) * 7), 26),
                   fill=(0, 0, 0, 190))
    draw.text((12, 12), legend, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    Image.alpha_composite(image, overlay).save(output_path)


def render_ports_diagnostic(product: dict, survey_path: str | Path,
                            output_path: str | Path) -> None:
    """Render the frozen ring, true entries, gaps, and exterior continuations."""
    survey_file = Path(survey_path)
    topdown = resolve_topdown_png(survey_file)
    if topdown is None:
        raise TownLayoutError("missing_diagnostic_input: site_topdown.png")
    survey = json.loads(survey_file.read_text(encoding="utf-8"))
    ring = product.get("planning_ring", {}).get("ring") or []
    image, mapping = diagnostic_view({"_diagnostic_bounds": [ring]}, topdown, survey,
                                     full_site=False)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    to_px = lambda point: _plan_to_px(float(point[0]), float(point[1]), mapping)
    for edge in product.get("edges", []):
        if edge.get("edge_role") == "block" and edge.get("geometry"):
            draw.line([to_px(point) for point in edge["geometry"]],
                      fill=(80, 80, 80, 180), width=2)
    if len(ring) >= 3:
        draw.line([to_px(point) for point in ring + [ring[0]]],
                  fill=(0, 240, 255, 255), width=4)
    for item in product.get("regional_outside_polylines", []):
        chain = item.get("polyline") or []
        if len(chain) >= 2:
            status = item.get("status")
            color = ((255, 80, 80, 210) if status == "excluded"
                     else (255, 170, 40, 170) if status == "retained_bridge_dependent"
                     else (185, 185, 185, 145))
            draw.line([to_px(point) for point in chain], fill=color, width=2)
    for node in product.get("nodes", []):
        if node.get("kind") in ("source_junction", "source_terminus"):
            px, py = to_px(node["position"])
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=(255, 230, 40, 255))
            draw.text((px + 10, py - 6), node["node_id"], fill=(255, 230, 40, 255),
                      font=ImageFont.load_default())
    for crossing in product.get("source_crossings", []):
        point = crossing.get("position")
        if not point:
            continue
        px, py = to_px(point)
        status = crossing.get("status")
        excluded = status == "excluded"
        internal_gap = status == "internal_gap_crossing"
        if internal_gap:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3),
                         fill=(145, 145, 155, 230))
            draw.text((px + 6, py - 6), "internal gap",
                      fill=(175, 175, 185, 240), font=ImageFont.load_default())
        else:
            color = (255, 50, 50, 255) if excluded else (255, 0, 190, 255)
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color)
        if excluded:
            draw.text((px + 9, py - 8), str(crossing.get("reason", "excluded")),
                      fill=(255, 100, 100, 255), font=ImageFont.load_default())
    for port in product.get("ports", []):
        point = port.get("position")
        if point:
            px, py = to_px(point)
            draw.ellipse((px - 9, py - 9, px + 9, py + 9),
                         outline=(255, 0, 190, 255), width=4)
            draw.text((px + 11, py + 8), port["port_id"], fill=(255, 80, 210, 255),
                      font=ImageFont.load_default())
    legend = (f"R2 PORTS  ports={len(product.get('ports', []))}"
              f" internal_gap={product.get('stage_metrics', {}).get('internal_gap_crossing_count', 0)}"
              f" texture_port={product.get('stage_metrics', {}).get('texture_port_count', 0)}"
              f" excluded={product.get('stage_metrics', {}).get('excluded_crossing_count', 0)}"
              f" bridge_dep={product.get('stage_metrics', {}).get('bridge_dependent_count', 0)}"
              f" retracted={product.get('stage_metrics', {}).get('boundary_retraction_count', 0)}")
    draw.rectangle((8, 8, min(image.width - 8, 8 + len(legend) * 7), 26),
                   fill=(0, 0, 0, 190))
    draw.text((12, 12), legend, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    Image.alpha_composite(image, overlay).save(output_path)
