"""Stage A/B/C review renderers: direct source-resolution topology + terrain.

Purpose
-------
Render townlayout road checkpoints into two same-extent, full-town images:
a clean topology panel (plain background, exact vector geometry, legend
panel with counts/checks/scale/orientation) and a terrain panel (same
geometry over the authoritative ``site_topdown.png``).  No source-road
polyline is ever drawn.

Inputs
------
Stage checkpoint product dicts plus the site survey JSON path (for the
terrain backdrop via ``site_context.diagnostic_view``).

Outputs
-------
PNG files at caller-given paths.

Pipeline position
-----------------
Phase 21 visual review artifacts for r2a/r2b/r2c; acceptance evidence only,
never a correctness proof.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont
from shapely.ops import unary_union

from .arterial_graph import build_fine_graph
from .geometry import polygon_from_ring
from .site_context import _plan_to_px, diagnostic_view, resolve_topdown_png
from .validate import TownLayoutError

MARGIN_GU = 2048.0
FONT = ImageFont.load_default()


def _draw_poly(draw, to_px, points, fill, width=1, dashed=False):
    px = [to_px(p) for p in points]
    if len(px) < 2:
        return
    if not dashed:
        draw.line(px, fill=fill, width=width)
        return
    for a, b in zip(px[:-1], px[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length <= 0:
            continue
        steps = max(1, int(length / 10))
        for i in range(0, steps, 2):
            u0, u1 = i / steps, min(1.0, (i + 1) / steps)
            draw.line([(a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0),
                       (a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1)],
                      fill=fill, width=width)


def _draw_ring_fill(draw, to_px, ring, fill, outline):
    px = [to_px(p) for p in ring]
    if len(px) >= 3:
        draw.polygon(px, fill=fill, outline=outline)


def _dot(draw, to_px, point, radius, fill, label=None, label_fill=None):
    px, py = to_px(point)
    draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=fill)
    if label:
        draw.text((px + radius + 2, py - radius), label,
                  fill=label_fill or fill, font=FONT)


def _legend(draw: ImageDraw.ImageDraw, image, title: str, lines: list[str]) -> None:
    all_lines = [title] + lines
    width = max(len(line) for line in all_lines) * 7 + 16
    height = len(all_lines) * 13 + 12
    draw.rectangle((8, 8, min(image.width - 8, 8 + width), 8 + height),
                   fill=(0, 0, 0, 200))
    for i, line in enumerate(all_lines):
        draw.text((14, 14 + i * 13), line,
                  fill=(255, 255, 255, 255), font=FONT)


def _scale_and_north(draw, image, mapping, gu=2048):
    px_per_gu = float(mapping["px_per_gu"])
    w = int(round(gu * px_per_gu))
    x0, y0 = image.width - w - 20, image.height - 30
    draw.line([(x0, y0), (x0 + w, y0)], fill=(255, 255, 255, 255), width=3)
    draw.text((x0, y0 - 14), f"{gu} GU", fill=(255, 255, 255, 255), font=FONT)
    nx, ny = image.width - 30, image.height - 100
    draw.line([(nx, ny), (nx, ny - 30)], fill=(255, 255, 255, 255), width=3)
    draw.polygon([(nx - 5, ny - 30), (nx + 5, ny - 30), (nx, ny - 40)],
                 fill=(255, 255, 255, 255))
    draw.text((nx - 3, ny - 56), "N", fill=(255, 255, 255, 255), font=FONT)


def _view(product: dict[str, Any], survey_path: str | Path):
    survey_file = Path(survey_path)
    topdown = resolve_topdown_png(survey_file)
    if topdown is None:
        raise TownLayoutError("missing_diagnostic_input: site_topdown.png")
    survey = json.loads(survey_file.read_text(encoding="utf-8"))
    domain = product.get("city_domain") or []
    if len(domain) < 3:
        raise TownLayoutError("render: city_domain missing")
    image, mapping = diagnostic_view({"_diagnostic_bounds": [domain]},
                                     topdown, survey,
                                     margin_gu=MARGIN_GU, full_site=False)
    return image, mapping


def _arterial_layers() -> dict[str, Any]:
    return {
        "fine": (200, 200, 200, 110),
        "outer": (255, 255, 255, 200),
        "chord": (140, 140, 140, 200),
        "raw": (255, 170, 40, 255),
        "smoothed": (255, 60, 60, 255),
        "corridor_fill": (255, 220, 60, 60),
        "corridor_edge": (255, 220, 60, 220),
        "port": (255, 0, 190, 255),
        "root": (255, 230, 40, 255),
        "merge": (40, 200, 255, 255),
    }


def _draw_arterials(draw, to_px, product, colors, terrain: bool) -> None:
    graph = build_fine_graph(product.get("patches") or [])
    # Full cell fabric: every fine edge is a cell outline (2026-08-18 user
    # amendment — leaf-trimming hid port context and boundary cells
    # had no outline at all).  The city outer boundary is emphasized on top.
    for edge_id in sorted(graph.edges):
        edge = graph.edges[edge_id]
        _draw_poly(draw, to_px, [graph.nodes[edge["a"]], graph.nodes[edge["b"]]],
                   colors["fine"], width=1)
    for edge_id in sorted(graph.edges):
        edge = graph.edges[edge_id]
        if edge["role"] != "fine_outer":
            continue
        _draw_poly(draw, to_px, [graph.nodes[edge["a"]], graph.nodes[edge["b"]]],
                   colors["outer"], width=2)
    corridor_rings = (product.get("corridor") or {}).get("rings") or []
    for ring in corridor_rings:
        _draw_ring_fill(draw, to_px, ring, colors["corridor_fill"], None)
    if corridor_rings:
        corridor_union = unary_union([polygon_from_ring(ring)
                                      for ring in corridor_rings])
        parts = ([corridor_union] if corridor_union.geom_type == "Polygon"
                 else list(corridor_union.geoms))
        for part in parts:
            _draw_poly(draw, to_px, list(part.exterior.coords),
                       colors["corridor_edge"], width=1)
            for interior in part.interiors:
                _draw_poly(draw, to_px, list(interior.coords),
                           colors["corridor_edge"], width=1)
    meeting = product["arterial_meeting"]["position"]
    for port in product.get("ports") or []:
        _draw_poly(draw, to_px, [port["position"], meeting],
                   colors["chord"], width=1, dashed=True)
    for barrier in product.get("raw_barrier") or []:
        if barrier.get("kind") == "fine_edge":
            _draw_poly(draw, to_px, barrier["geometry"], colors["raw"], width=2)
    for stroke in product.get("smoothed_strokes") or []:
        _draw_poly(draw, to_px, stroke["geometry"], colors["smoothed"], width=2)
    for port in product.get("ports") or []:
        _dot(draw, to_px, port["position"], 7, colors["port"], port["port_id"])
    merges = {m["merge_node_id"] for m in product.get("merge_records") or []
              if not m.get("is_root_merge")}
    for node in product.get("arterial_nodes") or []:
        if node.get("kind") == "meeting":
            _dot(draw, to_px, node["position"], 6, colors["root"],
                 "arterial meeting")
        elif node["node_id"] in merges:
            _dot(draw, to_px, node["position"], 5, colors["merge"],
                 node["node_id"])


def render_arterials(product: dict[str, Any], survey_path: str | Path,
                     out_topology: str | Path, out_terrain: str | Path) -> None:
    """Write the Stage A topology and terrain review renders (same extent)."""
    terrain_image, mapping = _view(product, survey_path)
    colors = _arterial_layers()
    to_px = lambda p: _plan_to_px(float(p[0]), float(p[1]), mapping)

    # Terrain panel.
    overlay = Image.new("RGBA", terrain_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    _draw_arterials(draw, to_px, product, colors, terrain=True)
    metrics = product.get("metrics") or {}
    legend_lines = [
        f"ports={len(product.get('ports') or [])} "
        f"routes={len(product.get('routes') or [])} "
        f"tree V={metrics.get('tree_node_count')} E={metrics.get('tree_edge_count')}",
        f"corridor water={metrics.get('corridor_water_overlap_gu2')} GU^2 "
        f"(bridge={metrics.get('bridge_connector_water_overlap_gu2', 0)} GU^2) "
        f"spill={metrics.get('corridor_city_spill_gu2')} GU^2",
        f"port residual deg: " + ", ".join(
            f"{g['port_id']}={g['heading_residual_deg']:.2f}"
            for g in metrics.get("port_metrics") or []),
        f"runtime={metrics.get('runtime_s', 0):.2f}s",
        "layers: grey cell outlines | white city outer boundary | yellow corridor | "
        "red complete arterial | orange raw shared edges | grey dashed "
        "chords | magenta boundary ports | "
        "yellow arterial meeting | cyan merges",
    ]
    _legend(draw, terrain_image, "R2A ARTERIALS (terrain)", legend_lines)
    _scale_and_north(draw, terrain_image, mapping)
    Image.alpha_composite(terrain_image, overlay).save(out_terrain)

    # Topology panel: same extent, plain background.
    topo = Image.new("RGBA", terrain_image.size, (24, 26, 30, 255))
    overlay = Image.new("RGBA", topo.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    _draw_arterials(draw, to_px, product, colors, terrain=False)
    _legend(draw, topo, "R2A ARTERIALS (topology)", legend_lines)
    _scale_and_north(draw, topo, mapping)
    Image.alpha_composite(topo, overlay).save(out_topology)


def _draw_road_blocks(draw, to_px, product, terrain: bool) -> None:
    """Draw the actual corridor-subtracted Stage-B polygons."""
    fills = {
        "island": (244, 205, 92, 125 if terrain else 185),
        "standard": (114, 177, 214, 105 if terrain else 175),
        "run": (221, 132, 92, 115 if terrain else 185),
        "exception": (255, 65, 65, 150),
    }
    for block in product.get("blocks") or []:
        _draw_ring_fill(draw, to_px, block["polygon"],
                        fills.get(block.get("actual_class"), fills["standard"]),
                        (20, 20, 20, 230))
        poly = polygon_from_ring(block["polygon"])
        _dot(draw, to_px, [poly.representative_point().x, poly.representative_point().y],
             2, (245, 245, 245, 255), block["block_id"], (245, 245, 245, 255))
    for area in product.get("isolated_areas") or []:
        _draw_ring_fill(draw, to_px, area["polygon"], (190, 80, 220, 135),
                        (240, 120, 255, 255))
    for verge in product.get("road_verges") or []:
        _draw_ring_fill(draw, to_px, verge["polygon"], (80, 220, 130, 150),
                        (90, 255, 150, 255))
    colors = _arterial_layers()
    for ring in (product.get("corridor") or {}).get("rings") or []:
        _draw_ring_fill(draw, to_px, ring, (40, 40, 40, 225), None)
    for stroke in product.get("smoothed_strokes") or []:
        _draw_poly(draw, to_px, stroke["geometry"], colors["smoothed"], width=2)
    for edge in product.get("block_edges") or []:
        if edge.get("edge_class") == "interior_candidate":
            _draw_poly(draw, to_px, edge["geometry"], (210, 210, 210, 210), width=1)
        elif edge.get("edge_class") == "outer_candidate":
            _draw_poly(draw, to_px, edge["geometry"], (255, 255, 255, 190), width=1)
    for port in product.get("ports") or []:
        _dot(draw, to_px, port["position"], 7, colors["port"], port["port_id"])


def render_road_blocks(product: dict[str, Any], survey_path: str | Path,
                       out_topology: str | Path, out_terrain: str | Path) -> None:
    """Write same-extent Stage-B topology and terrain review renders."""
    terrain_image, mapping = _view(product, survey_path)
    to_px = lambda p: _plan_to_px(float(p[0]), float(p[1]), mapping)
    metrics = product.get("metrics") or {}
    lines = [
        f"blocks={metrics.get('block_count', 0)} faces={metrics.get('atomic_face_count', 0)} "
        f"verges={metrics.get('road_verge_count', 0)} isolated={metrics.get('isolated_area_count', 0)}",
        f"lot equivalents p10={metrics.get('p10_lot_equivalents', 0):.2f} "
        f"p50={metrics.get('p50_lot_equivalents', 0):.2f} "
        f"p90={metrics.get('p90_lot_equivalents', 0):.2f} "
        f"pass={metrics.get('distribution_pass')}",
        f"gap={metrics.get('unexplained_gap_gu2', 0):.2f} GU^2 "
        f"overlap={metrics.get('overlap_gu2', 0):.2f} GU^2 "
        f"exceptions={metrics.get('exception_count', 0)} runtime={metrics.get('runtime_s', 0):.2f}s",
        "yellow=island | blue=standard | orange=run | red=exception | purple=isolated | green=verge",
        "dark corridor/red arterial | grey interior candidates | white outer candidates | magenta boundary ports",
    ]
    for base, output, title, terrain in (
            (terrain_image, out_terrain, "R2B ROAD BLOCKS (terrain)", True),
            (Image.new("RGBA", terrain_image.size, (24, 26, 30, 255)),
             out_topology, "R2B ROAD BLOCKS (topology)", False)):
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        _draw_road_blocks(draw, to_px, product, terrain)
        _legend(draw, base, title, lines)
        _scale_and_north(draw, base, mapping)
        Image.alpha_composite(base, overlay).save(output)


def _draw_minor_roads(draw, to_px, product, terrain: bool) -> None:
    palette = [(94, 151, 176, 90 if terrain else 165),
               (129, 170, 116, 90 if terrain else 165),
               (178, 143, 98, 90 if terrain else 165),
               (151, 119, 174, 90 if terrain else 165)]
    for index, block in enumerate(product.get("final_blocks") or []):
        fill = ((70, 115, 75, 55 if terrain else 100)
                if block.get("classification") == "open_landscape"
                else palette[index % len(palette)])
        for ring in block.get("polygons") or []:
            _draw_ring_fill(draw, to_px, ring, fill, (220, 220, 220, 150))
    for ring in (product.get("full_road_corridor") or {}).get("rings") or []:
        _draw_ring_fill(draw, to_px, ring, (42, 42, 42, 235), None)
    for stroke in product.get("smoothed_strokes") or []:
        _draw_poly(draw, to_px, stroke["geometry"], (255, 65, 55, 255), width=2)
    for stroke in product.get("minor_strokes") or []:
        _draw_poly(draw, to_px, stroke["geometry"], (245, 205, 95, 255), width=2)
    for port in product.get("ports") or []:
        _dot(draw, to_px, port["position"], 7, (255, 0, 190, 255), port["port_id"])
    wall = (product.get("inner_wall") or {}).get("centerline") or []
    if wall:
        _draw_poly(draw, to_px, wall + [wall[0]], (235, 225, 175, 255), width=2)
    for gate in product.get("wall_gates") or []:
        _dot(draw, to_px, gate["position"], 7, (255, 0, 190, 255), gate["gate_id"])
    degree = defaultdict(int)
    for edge in product.get("minor_road_edges") or []:
        geometry = edge.get("geometry") or []
        if len(geometry) >= 2:
            degree[tuple(geometry[0])] += 1
            degree[tuple(geometry[-1])] += 1
    for point, count in degree.items():
        if count != 2:
            _dot(draw, to_px, point, 3 if count == 1 else 4,
                 (80, 220, 255, 255) if count > 2 else (245, 205, 95, 255))


def render_minor_roads(product: dict[str, Any], survey_path: str | Path,
                       out_topology: str | Path, out_terrain: str | Path) -> None:
    """Write the complete Stage-C street/block visual review pair."""
    terrain_image, mapping = _view(product, survey_path)
    to_px = lambda p: _plan_to_px(float(p[0]), float(p[1]), mapping)
    metrics = product.get("metrics") or {}
    lines = [
        f"blocks={len(product.get('final_blocks') or [])} minor edges={metrics.get('selected_minor_edge_count', 0)} "
        f"strokes={metrics.get('minor_stroke_count', 0)} length={metrics.get('minor_road_length_gu', 0):.0f} GU",
        f"components={metrics.get('minor_component_count', 0)} exact links={metrics.get('junction_link_count', 0)} "
        f"unrooted={metrics.get('unrooted_component_count', 0)}",
        f"isolated={metrics.get('isolated_block_count', 0)} open landscape={metrics.get('open_landscape_count', 0)} "
        f"frontage failures={metrics.get('frontage_failure_count', 0)} "
        f"max degree={metrics.get('max_degree', 0)}",
        f"water={metrics.get('water_overlap_gu2', 0):.2f} GU^2 gap={metrics.get('unexplained_gap_gu2', 0):.2f} GU^2 "
        f"runtime={metrics.get('runtime_s', 0):.2f}s",
        "dark=all road space | red=arterial | yellow=minor streets | magenta=boundary ports",
        "pale line=inner wall | green wash=open landscape | cyan=junction | yellow dot=dead end",
    ]
    for base, output, title, terrain in (
            (terrain_image, out_terrain, "R2C MINOR ROADS (terrain)", True),
            (Image.new("RGBA", terrain_image.size, (24, 26, 30, 255)),
             out_topology, "R2C MINOR ROADS (topology)", False)):
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        _draw_minor_roads(draw, to_px, product, terrain)
        _legend(draw, base, title, lines)
        _scale_and_north(draw, base, mapping)
        Image.alpha_composite(base, overlay).save(output)


def render_inner_wall(product: dict[str, Any], survey_path: str | Path,
                      out_topology: str | Path, out_terrain: str | Path) -> None:
    """Render the pre-street inner wall and its arterial-only gates."""
    terrain_image, mapping = _view(product, survey_path)
    to_px = lambda p: _plan_to_px(float(p[0]), float(p[1]), mapping)
    metrics = product.get("metrics") or {}
    lines = [
        f"inner fraction={metrics.get('actual_fraction', 0):.3f} target={metrics.get('target_fraction', 0):.3f} "
        f"patches={metrics.get('selected_patch_count', 0)}/{metrics.get('city_patch_count', 0)}",
        f"wall={metrics.get('wall_length_gu', 0):.0f} GU gates={metrics.get('gate_count', 0)} "
        f"runtime={metrics.get('runtime_s', 0):.2f}s",
        "cyan fill=inner city | pale wall=centerline | red=major arterial | magenta=wall gate",
        "minor streets are deliberately absent; they are generated after this wall is frozen",
    ]
    for base, output, title, terrain in (
            (terrain_image, out_terrain, "INNER WALL (terrain)", True),
            (Image.new("RGBA", terrain_image.size, (24, 26, 30, 255)),
             out_topology, "INNER WALL (topology)", False)):
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        wall = (product.get("inner_wall") or {}).get("polygon") or []
        _draw_ring_fill(draw, to_px, wall, (40, 185, 210, 45 if terrain else 90),
                        (235, 225, 175, 255))
        for ring in (product.get("corridor") or {}).get("rings") or []:
            _draw_ring_fill(draw, to_px, ring, (35, 35, 35, 220), None)
        for stroke in product.get("smoothed_strokes") or []:
            _draw_poly(draw, to_px, stroke["geometry"], (255, 65, 55, 255), width=2)
        for gate in product.get("wall_gates") or []:
            _dot(draw, to_px, gate["position"], 8, (255, 0, 190, 255), gate["gate_id"])
        for port in product.get("ports") or []:
            _dot(draw, to_px, port["position"], 5, (255, 0, 190, 210), port["port_id"])
        _legend(draw, base, title, lines)
        _scale_and_north(draw, base, mapping)
        Image.alpha_composite(base, overlay).save(output)
