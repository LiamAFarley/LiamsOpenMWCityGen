"""Shared planning-canvas renderer for Cityforge planning bundles.

Pipeline position
------------------
This module is the deterministic drawing surface shared by the planning
bundle builder (``tools/cityforge/build_planning_bundle.py``) and the later
sketch renderer.  Given the exact D-SITE terrain evidence, a world-GU
rectangle, and the aligned road network, it renders the base planning
canvas PNG that a vision-capable design agent reads lot coordinates off
of.  It never loads stamp geometry, never runs Blender, and never authors
TES3 records.

Projection (binding; every consumer must use it)
------------------------------------------------
One orthographic north-up mapping from TES3 **world** GU (x east, y north)
to image pixels.  The map area occupies columns ``[0, map_w)`` and rows
``[TITLE_BAND_PX, TITLE_BAND_PX + map_h)``; the title band is above it and
the legend strip below:

    px = (x_gu - x0_gu) / (x1_gu - x0_gu) * map_w_px
    py = TITLE_BAND_PX + (y1_gu - y_gu) / (y1_gu - y0_gu) * map_h_px

``CanvasProjection.world_to_px`` / ``px_to_world`` expose exactly this
mapping so later tools (sketch renderer) draw geometry onto the identical
projection.  Note the coordinates are TES3 world GU, NOT survey-local plan
GU; world GU = plan GU + survey frame origin.

Inputs
------
* ``TerrainBundle`` (``src/procgen/visual_planner_terrain.py``) -- the
  accepted survey frame and exact height/slope/water/raw-VTEX evidence it
  renders (hillshade + water come from ``TerrainBundle.render_map``).
* a world-GU rectangle ``[x0, y0, x1, y1]`` in TES3 grid coordinates.
* ``AlignedNetwork`` (``src/procgen/aligned_roads.py``) -- the aligned road
  consumer product; only smooth centerline chains, corridor widths, and
  edge ids are used (source-space bundles are refused by its loader).

Outputs
-------
``render_planning_canvas`` returns ``(image, projection)``: a Pillow RGBA
image with terrain hillshade + exact water, cyan aligned source roads
(with corridor quads and edge-id labels), faint cell boundary lines every
8192 GU, a labelled GU graticule (gridlines every 1024 GU with ``x=``
labels along the top edge and ``y=`` labels along the left edge), a FAINT
UNLABELLED sub-graticule every 512 GU (visually subordinate to the 1024-GU
graticule; the 1024 line wins where they coincide), a title band (site name
+ rectangle) and a thin bottom legend strip.

Invariants
----------
* Deterministic for identical inputs: no randomness, no Blender, no
  filesystem-iteration-order dependence.
* North up; x east / y north in TES3 world GU.  Labels are exact TES3
  coordinates (e.g. ``x=-761856``).
* Cell and graticule lines are drawn at exact world-GU multiples, so a
  label value is the GU of the gridline it annotates.
* Rendering is a planning aid only; nothing drawn here becomes TES3 data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from .aligned_roads import SOURCE_ROAD_PRACTICAL_PATH_FRACTION
from .visual_planner_terrain import TerrainBundle

#: TES3 world cell size in game units (pinned by the terrain module).
CELL_SIZE_GU = 8192.0
#: Graticule spacing in game units.
GRATICULE_SPACING_GU = 1024.0
#: Faint unlabelled sub-graticule spacing (position-reading aid for
#: designers; visually subordinate to the labelled 1024-GU graticule).
SUBGRATICULE_SPACING_GU = 512.0
#: Default drawing density (pixels per 1024 GU).  Chosen so an 8-chartacter
#: ``x=`` label (~60 px at the label font) fits between adjacent gridlines
#: while the whole 2-cell-plus-margin canvas stays ~1600 px wide.
PIXELS_PER_1024_GU = 80
#: Height of the title band and the legend strip in image pixels.
TITLE_BAND_PX = 64
LEGEND_BAND_PX = 34
#: Hard cap on map-area pixels per side; larger rectangles are downscaled.
MAX_MAP_PX = 4096

_ROAD_CYAN = (0, 210, 235)
_CORRIDOR_FILL = (0, 180, 210, 42)
_CORRIDOR_OUTLINE = (0, 180, 210, 120)
_CELL_LINE = (210, 210, 210, 90)
_GRATICULE_LINE = (15, 15, 15, 36)
#: Sub-graticule is fainter and thinner-looking than the 1024-GU graticule
#: (lower alpha, same 1 px width -- the thinnest PIL line).
_SUBGRATICULE_LINE = (15, 15, 15, 20)
_LABEL_FILL = (20, 20, 20, 235)
_LABEL_STROKE = (255, 255, 255, 215)
_TITLE_BG = (28, 34, 40)
_TITLE_TEXT = (240, 244, 246)
_LEGEND_BG = (28, 34, 40)
_LEGEND_TEXT = (214, 220, 226)
_WATER_SWATCH = (56, 145, 205)

_FONT_REGULAR = r"C:\Windows\Fonts\arial.ttf"
_FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Best-effort TrueType font with the PIL default as fallback."""
    try:
        return ImageFont.truetype(_FONT_BOLD if bold else _FONT_REGULAR, size)
    except (OSError, ValueError):
        return ImageFont.load_default()


@dataclass(frozen=True)
class CanvasProjection:
    """The single orthographic world-GU -> image-pixel mapping of a canvas.

    Attributes
    ----------
    world_bounds_gu:
        ``[x0, y0, x1, y1]`` TES3 world GU actually drawn (render bounds).
    map_width_px / map_height_px:
        Size of the map area (excludes the title and legend bands).
    title_band_px:
        Image row where the map area starts (pixel Y of world y1).

    The mapping is documented in the module header; both helpers return
    float pixel coordinates so downstream tools can draw at sub-pixel
    precision.
    """

    world_bounds_gu: tuple[float, float, float, float]
    map_width_px: int
    map_height_px: int
    title_band_px: int

    @property
    def width_gu(self) -> float:
        return self.world_bounds_gu[2] - self.world_bounds_gu[0]

    @property
    def height_gu(self) -> float:
        return self.world_bounds_gu[3] - self.world_bounds_gu[1]

    def world_to_px(self, point: Sequence[float]) -> tuple[float, float]:
        """TES3 world GU -> image pixel (float), north up."""
        x0, y0, x1, y1 = self.world_bounds_gu
        px = (float(point[0]) - x0) / (x1 - x0) * self.map_width_px
        py = self.title_band_px + (y1 - float(point[1])) / (y1 - y0) * self.map_height_px
        return px, py

    def px_to_world(self, point: Sequence[float]) -> tuple[float, float]:
        """Image pixel -> TES3 world GU (inverse of ``world_to_px``)."""
        x0, y0, x1, y1 = self.world_bounds_gu
        x = x0 + float(point[0]) / self.map_width_px * (x1 - x0)
        y = y1 - (float(point[1]) - self.title_band_px) / self.map_height_px * (y1 - y0)
        return x, y


def _clip_segment_rect(
    a: Sequence[float], b: Sequence[float], rect: Sequence[float]
) -> tuple[float, float] | None:
    """Liang-Barsky clip of segment ``ab`` to the closed GU rect.

    Returns the parameter span ``(t0, t1)`` of the clipped portion, or
    ``None`` when the segment does not intersect the rect.  Deterministic
    and self-contained so the canvas does not depend on road-module
    internals beyond the public ``edges_in_rect`` query.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    x0, y0, x1, y1 = (float(v) for v in rect)
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            if ratio > t1:
                return None
            if ratio > t0:
                t0 = ratio
        else:
            if ratio < t0:
                return None
            if ratio < t1:
                t1 = ratio
    if t1 < t0:
        return None
    return t0, t1


def clip_polyline(
    points: Sequence[Sequence[float]], rect: Sequence[float]
) -> list[list[tuple[float, float]]]:
    """Clip a polyline to the GU rect, returning one span per segment.

    Each returned span is ``[start, end]`` (two points).  A polyline that
    leaves and re-enters the rect yields several spans; callers that need a
    single chain (e.g. ``site.json``) concatenate them, callers that draw
    (canvas) draw each span separately to avoid jump artifacts.
    """
    spans: list[list[tuple[float, float]]] = []
    for a, b in zip(points, points[1:]):
        span = _clip_segment_rect(a, b, rect)
        if span is None:
            continue
        t0, t1 = span
        start = (a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1]))
        end = (a[0] + t1 * (b[0] - a[0]), a[1] + t1 * (b[1] - a[1]))
        # Deduplicate the exact shared vertex of two adjacent clipped spans.
        if spans and spans[-1][1] == start:
            spans[-1][1] = start
            continue
        spans.append([start, end])
    return spans


def douglas_peucker(
    points: Sequence[Sequence[float]], epsilon_gu: float
) -> list[tuple[float, float]]:
    """Deterministic Douglas-Peucker simplification of a polyline."""
    clean = [(float(p[0]), float(p[1])) for p in points]
    if len(clean) < 3:
        return clean

    def perp_dist(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length == 0.0:
            return math.hypot(p[0] - a[0], p[1] - a[1])
        return abs(dx * (a[1] - p[1]) - (a[0] - p[0]) * dy) / length

    keep = [False] * len(clean)
    keep[0] = keep[-1] = True
    stack: list[tuple[int, int]] = [(0, len(clean) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        farthest, best = -1, -1.0
        for index in range(start + 1, end):
            dist = perp_dist(clean[index], clean[start], clean[end])
            if dist > best:
                farthest, best = index, dist
        if best > epsilon_gu:
            keep[farthest] = True
            stack.append((start, farthest))
            stack.append((farthest, end))
    return [point for point, flag in zip(clean, keep) if flag]


def _map_size(world_bounds_gu: Sequence[float], px_per_1024_gu: float) -> tuple[int, int]:
    """Map-area pixel size for a world rect, capped at ``MAX_MAP_PX``."""
    x0, y0, x1, y1 = (float(v) for v in world_bounds_gu)
    px_per_gu = px_per_1024_gu / GRATICULE_SPACING_GU
    width = max(1, int(round((x1 - x0) * px_per_gu)))
    height = max(1, int(round((y1 - y0) * px_per_gu)))
    scale = min(1.0, MAX_MAP_PX / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _draw_gridlines(
    draw: ImageDraw.ImageDraw,
    projection: CanvasProjection,
    spacing_gu: float,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    """Faint world-aligned gridlines at exact multiples of ``spacing_gu``."""
    x0, y0, x1, y1 = projection.world_bounds_gu
    for x in range(int(math.ceil(x0 / spacing_gu)), int(math.floor(x1 / spacing_gu)) + 1):
        px, _ = projection.world_to_px((x * spacing_gu, y1))
        draw.line((px, projection.title_band_px, px, projection.title_band_px + projection.map_height_px),
                  fill=color, width=width)
    for y in range(int(math.ceil(y0 / spacing_gu)), int(math.floor(y1 / spacing_gu)) + 1):
        _, py = projection.world_to_px((x0, y * spacing_gu))
        draw.line((0, py, projection.map_width_px, py), fill=color, width=width)


def _draw_graticule_labels(
    draw: ImageDraw.ImageDraw,
    projection: CanvasProjection,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """``x=`` labels along the top edge, ``y=`` labels along the left edge.

    A label anchored within half a gridline spacing of a map edge is
    skipped: an edge-anchored label would be clipped by the canvas border
    and collide with the perpendicular axis (the top-left corner in
    particular).  All interior labels are placed exactly as before.
    """
    x0, y0, x1, y1 = projection.world_bounds_gu
    top_y = projection.title_band_px + 2
    half_px = 0.5 * GRATICULE_SPACING_GU / projection.width_gu * projection.map_width_px
    for x in range(int(math.ceil(x0 / GRATICULE_SPACING_GU)),
                   int(math.floor(x1 / GRATICULE_SPACING_GU)) + 1):
        px, _ = projection.world_to_px((x * GRATICULE_SPACING_GU, y1))
        if px < half_px or px > projection.map_width_px - half_px:
            continue
        text = f"x={int(x * GRATICULE_SPACING_GU)}"
        box = draw.textbbox((0, 0), text, font=font)
        draw.text((px - (box[2] - box[0]) / 2.0, top_y), text, font=font,
                  fill=_LABEL_FILL, stroke_width=2, stroke_fill=_LABEL_STROKE)
    for y in range(int(math.ceil(y0 / GRATICULE_SPACING_GU)),
                   int(math.floor(y1 / GRATICULE_SPACING_GU)) + 1):
        _, py = projection.world_to_px((x0, y * GRATICULE_SPACING_GU))
        if (py < projection.title_band_px + half_px or
                py > projection.title_band_px + projection.map_height_px - half_px):
            continue
        text = f"y={int(y * GRATICULE_SPACING_GU)}"
        box = draw.textbbox((0, 0), text, font=font)
        draw.text((3, py - (box[3] - box[1]) / 2.0), text, font=font,
                  fill=_LABEL_FILL, stroke_width=2, stroke_fill=_LABEL_STROKE)


def _draw_roads(
    draw: ImageDraw.ImageDraw,
    projection: CanvasProjection,
    network: Any,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> list[str]:
    """Cyan aligned source roads: corridor quads, centerlines, edge labels.

    Returns the sorted edge ids drawn (for the caller's manifest/summary).
    """
    x0, y0, x1, y1 = projection.world_bounds_gu
    edges = sorted(network.edges_in_rect(x0, y0, x1, y1), key=lambda e: e.id)
    drawn: list[str] = []
    for edge in edges:
        spans = clip_polyline(edge.smooth_gu_polyline, (x0, y0, x1, y1))
        if not spans:
            continue
        # Corridor quads first (under the centerline).  Rings are drawn as
        # filled polygons; PIL clips out-of-image pixels automatically, and
        # only edges intersecting the rect are drawn, so no corridor can
        # paint far outside the map.  The band is drawn at the PRACTICAL
        # PATH width (the VTEX-blended band looks ~2.5x wider in game);
        # footprints must clear this band, road texture beyond it is fine.
        for ring in network.corridor_polygons(
                edge.id, margin_gu=0.0,
                width_scale=SOURCE_ROAD_PRACTICAL_PATH_FRACTION):
            pixels = [projection.world_to_px(p) for p in ring]
            if len(pixels) >= 3:
                draw.polygon(pixels, fill=_CORRIDOR_FILL, outline=_CORRIDOR_OUTLINE)
        # Centerline as one connected stroke (spans share clipped vertices).
        chain: list[tuple[float, float]] = []
        for span in spans:
            if not chain or chain[-1] != span[0]:
                chain.append(span[0])
            chain.append(span[1])
        pixels = [projection.world_to_px(p) for p in chain]
        if len(pixels) >= 2:
            draw.line(pixels, fill=_ROAD_CYAN, width=3)
        # Edge-id label at the midpoint of the clipped chain.
        mid = chain[len(chain) // 2]
        label = edge.id
        px, py = projection.world_to_px(mid)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((px - (box[2] - box[0]) / 2.0, py - (box[3] - box[1]) - 2), label,
                  font=font, fill=_LABEL_FILL, stroke_width=2, stroke_fill=_LABEL_STROKE)
        drawn.append(edge.id)
    return drawn


def _draw_title_band(
    image: Image.Image, title: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, TITLE_BAND_PX), fill=_TITLE_BG)
    box = draw.textbbox((0, 0), title, font=font)
    draw.text(((image.width - (box[2] - box[0])) / 2.0,
               (TITLE_BAND_PX - (box[3] - box[1])) / 2.0 - box[1]), title,
              font=font, fill=_TITLE_TEXT)
    draw.line((0, TITLE_BAND_PX - 1, image.width, TITLE_BAND_PX - 1), fill=(0, 0, 0, 255))


def _draw_legend(image: Image.Image, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(image)
    y = image.height - LEGEND_BAND_PX
    draw.rectangle((0, y, image.width, image.height), fill=_LEGEND_BG)
    items: list[tuple[tuple[int, int, int], str]] = [
        (_ROAD_CYAN, "aligned source road"),
        (_WATER_SWATCH, "water"),
        ((150, 150, 150), "cell boundary (8192 GU)"),
        ((110, 110, 110), "graticule (1024 GU)"),
        ((65, 65, 65), "sub-graticule (512 GU)"),
    ]
    cursor = 14
    swatch_h = 12
    band_y = y + (LEGEND_BAND_PX - swatch_h) / 2.0
    for color, label in items:
        draw.rectangle((cursor, band_y, cursor + 18, band_y + swatch_h), fill=color)
        cursor += 24
        draw.text((cursor, band_y - 1), label, font=font, fill=_LEGEND_TEXT)
        box = draw.textbbox((0, 0), label, font=font)
        cursor += (box[2] - box[0]) + 26


def render_planning_canvas(
    terrain: TerrainBundle,
    world_bounds_gu: Sequence[float],
    network: Any,
    *,
    site_name: str,
    title: str | None = None,
    px_per_1024_gu: float = PIXELS_PER_1024_GU,
) -> tuple[Image.Image, CanvasProjection]:
    """Render the base planning canvas for one world-GU rectangle.

    The terrain body (hillshade + exact water + shoreline) comes from
    ``TerrainBundle.render_map`` over the survey-plan rectangle that
    corresponds to ``world_bounds_gu``; the road/graticule layers are
    drawn on top in world GU.  Returns ``(image, projection)``.
    """
    if len(world_bounds_gu) != 4:
        raise ValueError("world_bounds_gu must be [x0, y0, x1, y1]")
    world = tuple(float(v) for v in world_bounds_gu)
    if world[0] >= world[2] or world[1] >= world[3]:
        raise ValueError("world rectangle minimum must not exceed maximum")
    plan_rect = terrain.rectangle(world_bounds_gu=world, context_margin_gu=0.0,
                                  full_site_inset=False)
    map_w, map_h = _map_size(plan_rect.render_bounds_gu, px_per_1024_gu)
    # World-GU render bounds actually drawn (render bounds may be clamped
    # to the survey extent; the projection must match the pixels).
    origin_x, origin_y = terrain.origin_gu
    drawn_world = (
        origin_x + plan_rect.render_bounds_gu[0],
        origin_y + plan_rect.render_bounds_gu[1],
        origin_x + plan_rect.render_bounds_gu[2],
        origin_y + plan_rect.render_bounds_gu[3],
    )
    projection = CanvasProjection(drawn_world, map_w, map_h, TITLE_BAND_PX)

    body = terrain.render_map(plan_rect, size=(map_w, map_h), hillshade=True)
    image = Image.new("RGBA", (map_w, TITLE_BAND_PX + map_h + LEGEND_BAND_PX), (0, 0, 0, 0))
    image.alpha_composite(body, (0, TITLE_BAND_PX))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    label_font = _font(12)
    _draw_gridlines(draw, projection, CELL_SIZE_GU, _CELL_LINE, 1)
    # Sub-graticule first: where a 512 line coincides with a 1024/8192 line
    # the later stronger line overwrites it (PIL lines replace pixels).
    _draw_gridlines(draw, projection, SUBGRATICULE_SPACING_GU, _SUBGRATICULE_LINE, 1)
    _draw_gridlines(draw, projection, GRATICULE_SPACING_GU, _GRATICULE_LINE, 1)
    # Roads are drawn BEFORE the graticule labels: PIL polygon/line fills
    # REPLACE overlay pixels rather than alpha-blending, so a corridor drawn
    # after a label would erase it (observed: y=-74752 label buried under a
    # corridor band).  Labels on top stay legible everywhere.
    _draw_roads(draw, projection, network, label_font)
    _draw_graticule_labels(draw, projection, label_font)
    # Thin frame around the map area so the canvas edge is unambiguous.
    draw.rectangle((0, TITLE_BAND_PX, map_w - 1, TITLE_BAND_PX + map_h - 1),
                   outline=(10, 10, 10, 200), width=1)
    image.alpha_composite(overlay)

    _draw_title_band(image, title or f"{site_name} — GU [{world[0]:.0f},{world[1]:.0f}]..[{world[2]:.0f},{world[3]:.0f}]",
                     _font(15, bold=True))
    _draw_legend(image, _font(12))
    return image, projection


__all__ = [
    "CELL_SIZE_GU", "CanvasProjection", "GRATICULE_SPACING_GU", "LEGEND_BAND_PX",
    "MAX_MAP_PX", "PIXELS_PER_1024_GU", "SUBGRATICULE_SPACING_GU", "TITLE_BAND_PX",
    "clip_polyline", "douglas_peucker", "render_planning_canvas",
]
