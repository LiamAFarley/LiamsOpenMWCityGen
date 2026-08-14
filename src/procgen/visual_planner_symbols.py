"""Deterministic Pillow symbols for the Cityforge visual-planning canvas.

Pipeline position
------------------
This module is the symbol stage between the exact D-SITE background and the
human/vision review of a visual-plan extension.  It consumes transformed
D-STAMP hulls, measured doors, and aligned-road geometry; it never authors
TES3 records and never changes the discrete survey data.

The map is intentionally separated from a right-hand selected-lot panel and a
bottom legend.  Map labels are placed by a deterministic collision gate after
 all door anchors, arrow shafts, and building bounds are known.  A label is
 never accepted over those forbidden rectangles or another accepted label, and
 failed candidates are recorded in the render audit rather than silently
 relaxed. Door arrows are drawn from the measured anchor outwards ALONG THE
 MEASURED DOOR FACING (member rotation z + lot yaw; the hull-centroid radial
 is only a fallback for members without a finite rotation) before optional
 local labels, with a dark halo and an intent-coloured shaft/head.
 Long leader lines are refused; ordinary map labels are local annotations,
 while full relationships belong in the selected-lot panel or an explicit
 routed access polyline.

Inputs and outputs
------------------
``render_plan_layers`` returns an RGBA Pillow image plus a JSON-ready audit.
The selected-lot panel exposes the full stamp/source identity, kit/category,
 every measured door and target, source terrain envelope, burial range, and any
 stair/access members recorded by D-STAMP. Door text is deliberately optional:
 the default callout is a short local ID beside the measured arrow, and the
 selected panel is the authority for full details. Access is rendered only as a
 short dashed stub or an explicitly supplied routed polyline; no target-centroid
 straight line is inferred. The adversarial proof may add finding markers, but
 normal callers still fail closed in the CLI when the advisory analyser
 returns hard errors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from . import cityplan
from .cityplan import polygon_centroid, rot2d_ccw
from .visual_planner_terrain import PlanningRectangle, TerrainBundle


MAP_MARGIN_PX = 18
HEADER_PX = 58
LEGEND_PX = 170
SIDE_PANEL_PX = 420
ARROW_LENGTH_GU = 720.0
ACCESS_STUB_LENGTH_GU = 560.0
LOCAL_LEADER_MAX_PX = 180.0
SELECTED_GLOW_WIDTH_PX = 11

COLORS = {
    "existing": (43, 221, 238, 230),
    "existing_edge": (17, 71, 78, 180),
    "existing_corridor": (43, 221, 238, 34),
    "authored_major": (145, 82, 54, 135),
    "authored_local": (193, 119, 67, 135),
    "authored_center": (255, 211, 139, 235),
    "authored_edge": (92, 57, 43, 175),
    "alley": (155, 97, 183, 150),
    "alley_center": (229, 181, 244, 225),
    "surface": (246, 195, 89, 48),
    "plaza": (251, 217, 107, 75),
    "court": (112, 188, 129, 52),
    "footprint_karthgad": (190, 227, 255, 74),
    "footprint_markarth": (255, 178, 96, 118),
    "footprint_outline": (17, 35, 55, 245),
    "overlap_bad": (249, 63, 54, 130),
    "overlap_bad_outline": (249, 63, 54, 240),
    "selected_outline": (255, 246, 176, 255),
    "door": (255, 255, 255, 255),
    "door_outline": (12, 25, 32, 255),
    "door_public": (47, 220, 93, 255),
    "door_service": (242, 157, 38, 255),
    "door_private": (176, 104, 232, 255),
    "door_unused": (135, 145, 151, 255),
    "door_unconnected": (64, 110, 235, 255),
    "door_faces_away": (232, 72, 220, 255),
    "slope": (242, 95, 44, 190),
    "annotation": (255, 240, 145, 255),
    "district": (255, 222, 122, 28),
    "boundary": (171, 93, 52, 210),
    "bad": (249, 63, 54, 250),
    "advisory": (255, 166, 48, 250),
    "access_link": (205, 240, 228, 150),
}


def _font(size: int = 14) -> ImageFont.ImageFont:
    """Load a readable deterministic system font with a Pillow fallback."""

    candidates = (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 compatibility for old workspace hosts
        return ImageFont.load_default()


def _rect_intersects(first: Sequence[float], second: Sequence[float]) -> bool:
    """Return true for positive-area rectangle intersection, not mere contact."""

    return (first[0] < second[2] and first[2] > second[0] and
            first[1] < second[3] and first[3] > second[1])


def _rect_from_points(points: Sequence[Sequence[float]], pad: float = 0.0) -> tuple[float, float, float, float]:
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(float(p[0]) for p in points) - pad,
            min(float(p[1]) for p in points) - pad,
            max(float(p[0]) for p in points) + pad,
             max(float(p[1]) for p in points) + pad)


def selected_highlight_geometry(
    hull_points: Sequence[Sequence[float]], *, tag_gap_px: float = 10.0,
    tag_width_px: float = 72.0, tag_height_px: float = 20.0,
) -> dict[str, Any]:
    """Return closed outline and adjacent tag geometry for a selected lot.

    This is pure pixel-space geometry so focused tests can prove that the
    selection signal is strong and local without rendering a proof image.
    The tag candidates are outside the footprint's bounding box; the caller's
    label collision gate still decides whether one fits without an arrow.
    """

    points = [(float(point[0]), float(point[1])) for point in hull_points]
    if not points:
        return {"outline": [], "bbox": None, "tag_candidates": []}
    bbox = _rect_from_points(points)
    outline = points + [points[0]]
    tag_candidates = [
        (bbox[2] + tag_gap_px, bbox[1] - tag_height_px),
        (bbox[0] - tag_width_px - tag_gap_px, bbox[1] - tag_height_px),
        (bbox[2] + tag_gap_px, bbox[3] + tag_gap_px),
        (bbox[0] - tag_width_px - tag_gap_px, bbox[3] + tag_gap_px),
    ]
    return {"outline": outline, "bbox": list(bbox),
            "tag_candidates": [list(point) for point in tag_candidates]}


def _composite_polygon(base: Image.Image, points: Sequence[Sequence[float]],
                       fill: tuple[int, int, int, int],
                       outline: tuple[int, int, int, int] | None = None,
                       width: int = 1) -> None:
    """Composite one translucent polygon without contaminating later layers."""

    if len(points) < 3:
        return
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.polygon(list(points), fill=fill, outline=outline, width=width)
    base.alpha_composite(layer)


class LabelPlacer:
    """Deterministic map-label collision gate with optional leader lines.

    ``forbidden`` contains map geometry rectangles known before labels are
    drawn.  It includes every building AABB, every door/arrow shaft, and thus
    conservatively protects the selected footprint as well as all other lots.
    Candidate order is caller-supplied and stable; no random nudging occurs.
    """

    def __init__(self, draw: ImageDraw.ImageDraw,
                 map_rect: tuple[float, float, float, float],
                 forbidden: Sequence[Sequence[float]]) -> None:
        self.draw = draw
        self.map_rect = tuple(float(value) for value in map_rect)
        self.forbidden = [tuple(float(value) for value in rect) for rect in forbidden]
        self.accepted: list[tuple[float, float, float, float]] = []
        self.placed: list[dict[str, Any]] = []
        self.unplaced: list[str] = []

    def add_forbidden(self, rect: Sequence[float]) -> None:
        self.forbidden.append(tuple(float(value) for value in rect))

    def _text_rect(self, text: str, point: Sequence[float], font: ImageFont.ImageFont,
                   pad: float = 4.0) -> tuple[float, float, float, float]:
        bbox = self.draw.textbbox((float(point[0]), float(point[1])), text,
                                  font=font, anchor="la", spacing=2,
                                  stroke_width=1)
        return (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)

    def _inside(self, rect: Sequence[float]) -> bool:
        return (rect[0] >= self.map_rect[0] and rect[1] >= self.map_rect[1] and
                rect[2] <= self.map_rect[2] and rect[3] <= self.map_rect[3])

    def place(self, text: str, preferred: Sequence[float], fill: tuple[int, int, int, int],
              font: ImageFont.ImageFont | None = None,
              *, candidates: Sequence[Sequence[float]] = (),
              leader_from: Sequence[float] | None = None,
              box_fill: tuple[int, int, int, int] = (12, 24, 29, 224)) -> tuple[float, float, float, float] | None:
        """Place one label or record it as unplaced without relaxing the gate."""

        font = font or _font(11)
        points = list(candidates) or [preferred]
        # A final deterministic ring around the preferred point handles labels
        # near a map edge without allowing overlap or clipping.
        if not candidates:
            x, y = float(preferred[0]), float(preferred[1])
            points.extend(((x + 12, y), (x - 12, y), (x, y + 18),
                           (x, y - 18), (x + 18, y + 18), (x - 18, y - 18)))
        for point in points:
            rect = self._text_rect(text, point, font)
            if not self._inside(rect):
                continue
            if any(_rect_intersects(rect, other) for other in self.forbidden):
                continue
            if any(_rect_intersects(rect, other) for other in self.accepted):
                continue
            if leader_from is not None:
                anchor = (max(rect[0], min(float(leader_from[0]), rect[2])),
                          max(rect[1], min(float(leader_from[1]), rect[3])))
                # Leader lines are only local label connectors.  A long line
                # makes a label look like a route claim and recreates the
                # cross-canvas spaghetti the planner is meant to avoid.
                if math.hypot(anchor[0] - float(leader_from[0]),
                              anchor[1] - float(leader_from[1])) > LOCAL_LEADER_MAX_PX:
                    continue
                self.draw.line((float(leader_from[0]), float(leader_from[1]),
                                anchor[0], anchor[1]),
                                fill=(233, 239, 211, 170), width=1)
            self.draw.rounded_rectangle(rect, radius=4, fill=box_fill,
                                        outline=(229, 239, 216, 210), width=1)
            self.draw.multiline_text((float(point[0]), float(point[1])), text,
                                     fill=fill, font=font, anchor="la", spacing=2,
                                     stroke_width=1, stroke_fill=(8, 16, 20, 230))
            self.accepted.append(rect)
            self.placed.append({"text": text, "rect": list(rect)})
            return rect
        self.unplaced.append(text)
        return None


@dataclass(frozen=True)
class StampRenderRecord:
    """Resolved transformed stamp data used by renderer, panel, and audit."""

    lot_id: str
    stamp_id: str
    source_name: str
    kit: str
    category: str
    position: tuple[float, float]
    yaw_deg: float
    hull: tuple[tuple[float, float], ...]
    doors: tuple[dict[str, Any], ...]
    source_cell: tuple[int, int] | None
    terrain_envelope: Mapping[str, Any]
    access_members: tuple[dict[str, Any], ...]
    access_heading_deg: float
    access_links: tuple[dict[str, Any], ...]


def _transform_door(stamp: Mapping[str, Any], member: Mapping[str, Any],
                    position: Sequence[float], yaw_deg: float,
                    hull_centroid: Sequence[float],
                    intent: Mapping[str, Any] | None) -> dict[str, Any]:
    offset = member.get("offset_gu", [0.0, 0.0, 0.0])
    relative = rot2d_ccw(float(offset[0]), float(offset[1]), yaw_deg)
    point = (float(position[0]) + relative[0], float(position[1]) + relative[1])
    # Door heading: the v2 library's geometric ``outward_heading_deg`` (thin-
    # axis wall normal, sign away from the body centroid; measured during
    # normalization) is authoritative.  The raw TES3 member rotation z is NOT
    # a reliable facing (mesh forward axes differ per model family; the
    # 2026-08-12 measurement proved world = pos + Rz(-rotz).local, and the
    # door mesh's local forward is model-specific), so it is only a fallback
    # for libraries predating outward_heading_deg.  The outward radial from
    # the transformed hull centroid is the last-resort proxy.
    outward = member.get("outward_heading_deg")
    rotation = member.get("rotation") or [0.0, 0.0, 0.0]
    rotation_ok = isinstance(rotation, Sequence) and len(rotation) >= 3 and all(
        math.isfinite(float(value)) for value in rotation[:3])
    if outward is not None and math.isfinite(float(outward)):
        heading = (float(outward) + float(yaw_deg)) % 360.0
    elif rotation_ok:
        heading = (math.degrees(float(rotation[2])) + float(yaw_deg)) % 360.0
    else:
        dx = point[0] - float(hull_centroid[0])
        dy = point[1] - float(hull_centroid[1])
        if abs(dx) + abs(dy) > 1e-9:
            heading = math.degrees(math.atan2(dy, dx))
        else:
            heading = math.degrees(float(stamp.get("access_heading_rad", 0.0))) + float(yaw_deg)
    door_data = member.get("door") or {}
    return {
        "door_id": str(member.get("source_id", "")),
        "position_plan_gu": [point[0], point[1]],
        "heading_deg": heading,
        "intent": (intent or {}).get("intent", "public"),
        "target_id": (intent or {}).get("target_id"),
        "source_destination_cell": door_data.get("destination_cell"),
        "step_height_gu": door_data.get("step_height_gu"),
        "structural_role": member.get("structural_role"),
    }


def resolve_stamps(document: Mapping[str, Any],
                   stamp_geometry: Mapping[str, Mapping[str, Any]]) -> list[StampRenderRecord]:
    """Resolve exact hulls, every measured door, and source access evidence."""

    records: list[StampRenderRecord] = []
    for placement in document.get("stamps", []):
        if not isinstance(placement, Mapping):
            continue
        stamp_id = str(placement.get("stamp_id", ""))
        stamp = stamp_geometry.get(stamp_id)
        if not isinstance(stamp, Mapping):
            continue
        position = placement.get("position_plan_gu")
        if not isinstance(position, Sequence) or len(position) != 2:
            continue
        yaw = float(placement.get("yaw_deg", 0.0))
        hull = stamp.get("footprint", {}).get("hull_xy_rel", [])
        transformed_hull = tuple(
            (float(position[0]) + delta[0], float(position[1]) + delta[1])
            for delta in (rot2d_ccw(float(p[0]), float(p[1]), yaw) for p in hull)
        )
        intent_map = {
            str(intent.get("door_id")): intent
            for intent in placement.get("door_intents", [])
            if isinstance(intent, Mapping) and isinstance(intent.get("door_id"), str)
        }
        explicit_targets = {
            str(target.get("door_id")): target
            for target in placement.get("door_targets", [])
            if isinstance(target, Mapping) and isinstance(target.get("door_id"), str)
        }
        for door_id, target in explicit_targets.items():
            intent = dict(intent_map.get(door_id, {"door_id": door_id}))
            intent["target_id"] = target.get("target_id")
            intent["intent"] = target.get("intent")
            intent_map[door_id] = intent
        hull_centroid = polygon_centroid([list(point) for point in transformed_hull])
        doors = tuple(
            _transform_door(stamp, member, position, yaw, hull_centroid,
                            intent_map.get(str(member.get("source_id"))))
            for member in stamp.get("members", [])
            if isinstance(member, Mapping) and bool(member.get("is_door"))
        )
        source = stamp.get("source", {})
        source_cell_value = source.get("source_cell")
        source_cell = (int(source_cell_value[0]), int(source_cell_value[1])) \
            if isinstance(source_cell_value, Sequence) and len(source_cell_value) == 2 else None
        access_members = tuple(
            dict(member) for member in stamp.get("members", [])
            if isinstance(member, Mapping) and (
                member.get("structural_role") in ("access", "connector") or
                any(token in str(member.get("model_key", "")).casefold()
                    for token in ("stair", "entr", "ramp", "terrace")))
        )
        access_links = tuple(
            dict(link) for link in placement.get("access_links", [])
            if isinstance(link, Mapping)
        )
        records.append(StampRenderRecord(
            lot_id=str(placement.get("lot_id", "")), stamp_id=stamp_id,
            source_name=str(source.get("slug", stamp_id)), kit=str(placement.get("kit", "unknown")),
            category=str(placement.get("category", stamp.get("building_type", "building"))),
            position=(float(position[0]), float(position[1])), yaw_deg=yaw,
            hull=transformed_hull, doors=doors, source_cell=source_cell,
            terrain_envelope=dict(stamp.get("terrain_envelope", {})),
            access_members=access_members,
            access_heading_deg=math.degrees(float(stamp.get("access_heading_rad", 0.0))) + yaw,
            access_links=access_links,
        ))
    return records


def _polyline_pixel(terrain: TerrainBundle, rectangle: PlanningRectangle,
                    size: tuple[int, int], points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    return [terrain.world_to_pixel(rectangle, size, point) for point in points]


def _draw_styled_road(draw: ImageDraw.ImageDraw, points: Sequence[Sequence[float]],
                      width_px: int, fill: tuple[int, int, int, int],
                      center: tuple[int, int, int, int],
                      edge: tuple[int, int, int, int]) -> None:
    """Draw a controlled-width surface with a separate edge and centreline."""

    if len(points) < 2:
        return
    draw.line(points, fill=edge, width=max(3, width_px + 5), joint="curve")
    draw.line(points, fill=fill, width=max(2, width_px), joint="curve")
    draw.line(points, fill=center, width=max(2, min(4, width_px // 4 + 1)), joint="curve")


def _nearest_polyline(point: Sequence[float], polyline: Sequence[Sequence[float]]) -> tuple[float, tuple[float, float]]:
    best = (float("inf"), (float(polyline[0][0]), float(polyline[0][1])))
    for first, second in zip(polyline, polyline[1:]):
        ax, ay = float(first[0]), float(first[1])
        bx, by = float(second[0]), float(second[1])
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq == 0 else max(0.0, min(1.0,
            ((float(point[0]) - ax) * dx + (float(point[1]) - ay) * dy) / length_sq))
        nearest = (ax + t * dx, ay + t * dy)
        distance = math.hypot(float(point[0]) - nearest[0], float(point[1]) - nearest[1])
        if distance < best[0]:
            best = (distance, nearest)
    return best


def _target_geometries(document: Mapping[str, Any], aligned_network: Any,
                       terrain: TerrainBundle) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for road in document.get("authored_roads", []):
        if isinstance(road, Mapping) and isinstance(road.get("road_id"), str):
            targets[road["road_id"]] = {"polyline": road.get("polyline_plan_gu", [])}
    for alley in document.get("alleys", []):
        if isinstance(alley, Mapping) and isinstance(alley.get("alley_id"), str):
            targets[alley["alley_id"]] = {"polyline": alley.get("polyline_plan_gu", [])}
    for region in document.get("road_surface_polygons", []):
        if isinstance(region, Mapping) and isinstance(region.get("region_id"), str):
            targets[region["region_id"]] = {"polygon": region.get("polygon_plan_gu", [])}
    for court in document.get("shared_courts", []):
        if isinstance(court, Mapping) and isinstance(court.get("court_id"), str):
            targets[court["court_id"]] = {"polygon": court.get("polygon_plan_gu", [])}
    if aligned_network is not None:
        for source in document.get("existing_source_roads", []):
            if not isinstance(source, Mapping) or not isinstance(source.get("edge_id"), str):
                continue
            try:
                edge = aligned_network.edge(source["edge_id"])
            except Exception:  # noqa: BLE001 - unresolved edge is handled by advisory stage
                continue
            targets[source["edge_id"]] = {"polyline": [
                aligned_network.to_site_local(point, terrain.origin_gu)
                for point in edge.smooth_gu_polyline]}
    return targets


def _target_point(point: Sequence[float], target: Mapping[str, Any]) -> tuple[float, float] | None:
    polyline = target.get("polyline")
    if isinstance(polyline, list) and len(polyline) >= 2:
        return _nearest_polyline(point, polyline)[1]
    polygon = target.get("polygon")
    if isinstance(polygon, list) and len(polygon) >= 3:
        centroid = polygon_centroid(polygon)
        return float(centroid[0]), float(centroid[1])
    return None


def _draw_direct_label(draw: ImageDraw.ImageDraw, point: Sequence[float], text: str,
                       fill: tuple[int, int, int, int], font: ImageFont.ImageFont,
                       *, box: bool = True) -> None:
    bbox = draw.textbbox((float(point[0]), float(point[1])), text, font=font,
                         anchor="la", spacing=2, stroke_width=1)
    if box:
        draw.rounded_rectangle((bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4),
                               radius=4, fill=(12, 24, 29, 230), outline=(229, 239, 216, 210))
    draw.multiline_text(point, text, fill=fill, font=font, anchor="la", spacing=2,
                        stroke_width=1, stroke_fill=(8, 16, 20, 230))


def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont,
                max_width: float) -> list[str]:
    lines: list[str] = []
    for raw in str(text).splitlines() or [""]:
        words = raw.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def local_door_label_candidates(
    tip: Sequence[float], heading_deg: float,
) -> tuple[tuple[float, float], ...]:
    """Return only short, local door-label positions around an arrow tip.

    The old renderer used a perimeter/top-lane fallback, which made ten
    otherwise local doors look like a network of long leader lines.  A callout
    now either fits immediately beside its own arrow or is omitted; its full
    identity remains in the selected-lot panel.
    """

    angle = math.radians(float(heading_deg))
    ux, uy = math.cos(angle), -math.sin(angle)
    nx, ny = -uy, ux
    candidates: list[tuple[float, float]] = []
    for forward, normal in ((12.0, -24.0), (12.0, 24.0),
                            (28.0, -34.0), (28.0, 34.0),
                            (-34.0, -28.0), (-34.0, 28.0)):
        candidates.append((float(tip[0]) + ux * forward + nx * normal,
                           float(tip[1]) + uy * forward + ny * normal))
    return tuple(candidates)


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw, points: Sequence[Sequence[float]],
    fill: tuple[int, int, int, int], width: int = 2, dash_px: int = 7,
) -> None:
    """Draw a deterministic dashed polyline without joining unrelated points."""

    if len(points) < 2:
        return
    for first, second in zip(points, points[1:]):
        ax, ay = float(first[0]), float(first[1])
        bx, by = float(second[0]), float(second[1])
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        step = max(1.0, float(dash_px * 2))
        distance = 0.0
        while distance < length:
            end_distance = min(length, distance + dash_px)
            ratio_a, ratio_b = distance / length, end_distance / length
            draw.line((ax + dx * ratio_a, ay + dy * ratio_a,
                       ax + dx * ratio_b, ay + dy * ratio_b),
                      fill=fill, width=width)
            distance += step


def access_render_geometry(
    door_point: Sequence[float], target: Mapping[str, Any] | None,
    explicit_polyline: Sequence[Sequence[float]] | None = None,
    *, stub_length_gu: float = ACCESS_STUB_LENGTH_GU,
) -> dict[str, Any] | None:
    """Resolve one access visualization to a stub or supplied route.

    ``explicit_polyline`` is trusted only as authored plan geometry and is
    rendered in full.  Without it, the target is used solely to choose a
    direction for a capped local stub; the function never returns a line to a
    distant centroid.  This pure helper is also the focused-test seam for the
    no-inferred-route rule.
    """

    start = (float(door_point[0]), float(door_point[1]))
    if isinstance(explicit_polyline, Sequence) and len(explicit_polyline) >= 2:
        route = [(float(point[0]), float(point[1])) for point in explicit_polyline]
        if math.hypot(route[0][0] - start[0], route[0][1] - start[1]) > 1e-6:
            route.insert(0, start)
        return {"mode": "explicit_route", "points": route,
                "length_gu": _polyline_length(route)}
    target_point = _target_point(start, target) if target else None
    if target_point is None:
        return None
    dx, dy = float(target_point[0]) - start[0], float(target_point[1]) - start[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return {"mode": "short_stub", "points": [start, start], "length_gu": 0.0}
    length = min(float(stub_length_gu), distance)
    end = (start[0] + dx / distance * length,
           start[1] + dy / distance * length)
    return {"mode": "short_stub", "points": [start, end], "length_gu": length,
            "target_distance_gu": distance}


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    return sum(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
               for a, b in zip(points, points[1:]))


def _draw_detail_panel(canvas: Image.Image, panel_x: int, top: int, bottom: int,
                       records: Sequence[StampRenderRecord], selected_lot_id: str | None,
                       advisory_report: Mapping[str, Any] | None,
                       adversarial: bool = False) -> dict[str, Any]:
    """Draw the non-map selected-lot evidence panel and return its audit."""

    draw = ImageDraw.Draw(canvas)
    width = canvas.size[0] - panel_x
    draw.rectangle((panel_x, top, canvas.size[0], bottom), fill=(9, 24, 29, 248),
                   outline=(93, 133, 125, 220), width=2)
    record = next((item for item in records if item.lot_id == selected_lot_id), None)
    if record is None and records:
        record = records[0]
    x = panel_x + 18
    y = top + 16
    body_font = _font(11)
    small_font = _font(10)
    heading_font = _font(14)
    draw.text((x, y), "SELECTED LOT DETAIL", fill=(255, 226, 120, 255), font=heading_font)
    y += 27
    lines_drawn = 0

    def section(label: str) -> None:
        nonlocal y, lines_drawn
        if y + 18 < bottom:
            draw.text((x, y), label, fill=(180, 231, 202, 255), font=body_font)
            y += 17
            lines_drawn += 1

    def field(label: str, value: Any, font: ImageFont.ImageFont = small_font) -> None:
        nonlocal y, lines_drawn
        text = f"{label}: {value}"
        for line in _wrap_lines(draw, text, font, width - 34):
            if y + 15 >= bottom:
                return
            draw.text((x, y), line, fill=(225, 235, 220, 255), font=font)
            y += 14
            lines_drawn += 1

    if record is None:
        field("status", "no selected stamp resolved")
    else:
        field("lot", record.lot_id, body_font)
        field("stamp", record.stamp_id)
        field("source", record.source_name)
        field("kit / category", f"{record.kit} / {record.category}")
        if record.source_cell is not None:
            field("source cell", f"{record.source_cell[0]}, {record.source_cell[1]}")
        field("placement yaw", f"{record.yaw_deg % 360.0:.1f}°")
        section(f"DOORS ({len(record.doors)})")
        for index, door in enumerate(record.doors, start=1):
            door_id = str(door.get("door_id", ""))
            intent = str(door.get("intent", "public"))
            target = door.get("target_id") or "unlinked"
            field(f"D{index} id", door_id)
            field("  intent / heading", f"{intent} / {float(door.get('heading_deg', 0.0)) % 360.0:.1f}°")
            field("  target / step", f"{target} / {float(door.get('step_height_gu', 0.0)):.1f} GU")
        envelope = record.terrain_envelope
        section("SOURCE TERRAIN ENVELOPE")
        field("slope / relief", f"{float(envelope.get('footprint_slope_deg', 0.0)):.1f}° / {float(envelope.get('footprint_relief_gu', 0.0)):.1f} GU")
        burial = float(envelope.get("burial_depth_gu", 0.0))
        field("burial range", f"0.0–{burial:.1f} GU")
        field("access heading", f"{record.access_heading_deg % 360.0:.1f}° from source centroid")
        if record.access_members:
            section(f"STAIR / ACCESS MEMBERS ({len(record.access_members)})")
            for member in record.access_members[:5]:
                model = str(member.get("model_key", "")).replace("\\", "/").split("/")[-1]
                field("member", f"{member.get('source_id', '')} · {model}")
        else:
            section("STAIR / ACCESS MEMBERS")
            field("recorded", "none in this stamp; door steps shown above")

    finding_count = 0
    if adversarial and advisory_report:
        section("ADVERSARIAL CALLOUTS")
        findings = list(advisory_report.get("hard_errors", [])) + list(advisory_report.get("advisories", []))
        for finding in findings:
            severity = str(finding.get("severity", "advisory")).upper()
            code = str(finding.get("code", "finding"))
            lots = ",".join(str(value) for value in finding.get("lot_ids", [])) or "plan"
            doors = ",".join(str(value) for value in finding.get("door_ids", []))
            exact = f"{lots} / {doors}" if doors else lots
            field(f"{severity} {code}", exact, small_font)
            finding_count += 1

    return {"panel_px": [panel_x, top, canvas.size[0], bottom],
            "selected_lot_id": record.lot_id if record else None,
            "panel_line_count": lines_drawn, "finding_line_count": finding_count}


def _draw_legend(canvas: Image.Image, title: str, audit: Mapping[str, Any],
                 stamp_records: Sequence[StampRenderRecord] = (),
                 context_inset: Image.Image | None = None) -> None:
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    top = height - LEGEND_PX
    draw.rectangle((0, top, width, height), fill=(14, 23, 28, 248))
    draw.text((MAP_MARGIN_PX, top + 10), title, fill=(255, 226, 120, 255), font=_font(16))
    entries = [
        (COLORS["existing"], "aligned source road + corridor"),
        (COLORS["authored_major"], "authored street (edge / centre)"),
        (COLORS["alley"], "narrow alley (edge / centre)"),
        (COLORS["surface"], "plaza / road-surface polygon"),
        (COLORS["court"], "shared court"),
        (COLORS["footprint_karthgad"], "Karthgad footprint"),
        (COLORS["footprint_markarth"], "Markarth footprint"),
        (COLORS["door_public"], "door arrow: public"),
        (COLORS["door_service"], "door arrow: service"),
        (COLORS["door_private"], "door arrow: private"),
    ]
    columns = 2
    row_height = 20
    for index, (colour, label) in enumerate(entries):
        column, row = divmod(index, 5)
        x = MAP_MARGIN_PX + column * 270
        y = top + 37 + row * row_height
        draw.rectangle((x, y, x + 15, y + 13), fill=colour, outline=(235, 243, 223, 220))
        draw.text((x + 22, y - 2), label, fill=(230, 238, 225, 255), font=_font(10))
    if stamp_records:
        key_x = width - (context_inset.width if context_inset is not None else 0) - 360
        key_x = max(570, key_x)
        key_y = top + 10
        draw.text((key_x, key_y), "STAMP KEY (full source in selected panel)",
                  fill=(255, 226, 120, 255), font=_font(9))
        for index, record in enumerate(stamp_records):
            column, row = divmod(index, 3)
            x = key_x + column * 155
            y = key_y + 14 + row * 14
            kit = "K" if "karth" in record.kit.casefold() else "M"
            source = record.source_name if len(record.source_name) <= 18 else record.source_name[:15] + "…"
            draw.text((x, y), f"{record.lot_id} / {kit} / {source}",
                      fill=(208, 222, 209, 255), font=_font(8))
    if context_inset is not None:
        inset_x = width - context_inset.width - 18
        inset_y = top + 29
        draw.rectangle((inset_x - 2, inset_y - 2, inset_x + context_inset.width + 2,
                        inset_y + context_inset.height + 2), outline=(241, 232, 169, 235), width=1)
        canvas.alpha_composite(context_inset, (inset_x, inset_y))
        draw.text((inset_x, top + 11), "FULL SITE CONTEXT", fill=(255, 226, 120, 255), font=_font(9))
    summary = (f"stamps={audit.get('stamp_count', 0)}  doors={audit.get('door_count', 0)}  "
               f"roads={audit.get('authored_road_count', 0)}  labels={audit.get('label_audit', {}).get('placed_count', 0)}")
    draw.text((MAP_MARGIN_PX, height - 13), summary, fill=(188, 206, 190, 255), font=_font(9))


def render_plan_layers(
    document: Mapping[str, Any], terrain: TerrainBundle, rectangle: PlanningRectangle, *,
    aligned_network: Any, stamp_geometry: Mapping[str, Mapping[str, Any]],
    title: str = "CITYFORGE VISUAL PLANNING CANVAS",
    size: tuple[int, int] = (1440, 1180), show_contours: bool = False,
    show_slope: bool = False, include_legend: bool = True,
    include_header: bool = True, include_context_inset: bool = True,
    show_source_terrain: bool = False, show_burial_envelope: bool = False,
    advisory_report: Mapping[str, Any] | None = None,
    show_advisory_markers: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    """Compose the map, selected-lot evidence panel, and legend deterministically."""

    if size[0] < SIDE_PANEL_PX + 720 or size[1] < HEADER_PX + LEGEND_PX + 160:
        raise ValueError("planning canvas is too small for map and detail panel")
    body_top = HEADER_PX if include_header else 0
    body_bottom = size[1] - (LEGEND_PX if include_legend else 0)
    map_width = size[0] - SIDE_PANEL_PX
    body_size = (map_width, body_bottom - body_top)
    terrain_image = terrain.render_map(rectangle, size=body_size, hillshade=True,
                                       slope_advisory=show_slope, contours=show_contours)
    canvas = Image.new("RGBA", size, (12, 20, 24, 255))
    canvas.alpha_composite(terrain_image, (0, body_top))
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def px(point: Sequence[float]) -> tuple[float, float]:
        x, y = terrain.world_to_pixel(rectangle, body_size, point)
        return x, y + body_top

    records = resolve_stamps(document, stamp_geometry)
    options = document.get("render_options", {})
    selected_lot_id = str(options.get("selected_lot_id")) if isinstance(options, Mapping) and options.get("selected_lot_id") else (records[0].lot_id if records else None)
    targets = _target_geometries(document, aligned_network, terrain)

    # Existing aligned context is quiet; the explicitly referenced source edge
    # receives the measured corridor and a brighter centreline.
    existing_count = 0
    if aligned_network is not None:
        for edge in sorted(aligned_network.edges.values(), key=lambda item: item.id):
            local = [aligned_network.to_site_local(point, terrain.origin_gu)
                     for point in edge.smooth_gu_polyline]
            if not any(rectangle.contains(point, requested=False) for point in local):
                continue
            draw.line([px(point) for point in local], fill=(45, 154, 160, 105), width=1, joint="curve")
            existing_count += 1
    for source_road in document.get("existing_source_roads", []):
        if not isinstance(source_road, Mapping) or aligned_network is None:
            continue
        edge_id = str(source_road.get("edge_id", ""))
        try:
            edge = aligned_network.edge(edge_id)
        except Exception:  # noqa: BLE001 - unresolved edge remains in advisory JSON
            continue
        local = [aligned_network.to_site_local(point, terrain.origin_gu)
                 for point in edge.smooth_gu_polyline]
        if bool(source_road.get("show_corridor", True)):
            for ring in aligned_network.corridor_polygons(edge_id, margin_gu=float(source_road.get("corridor_margin_gu", 0.0))):
                _composite_polygon(overlay, [px(aligned_network.to_site_local(point, terrain.origin_gu)) for point in ring],
                                   COLORS["existing_corridor"])
        draw.line([px(point) for point in local], fill=COLORS["existing_edge"], width=7, joint="curve")
        draw.line([px(point) for point in local], fill=COLORS["existing"], width=3, joint="curve")

    # Subtle cell frame, restricted to the map column rather than the detail panel.
    min_x, min_y, max_x, max_y = rectangle.render_bounds_gu
    cell_x0, cell_x1 = math.floor(min_x / 8192.0), math.ceil(max_x / 8192.0)
    cell_y0, cell_y1 = math.floor(min_y / 8192.0), math.ceil(max_y / 8192.0)
    actual_min_x = math.floor(terrain.origin_gu[0] / 8192.0)
    actual_min_y = math.floor(terrain.origin_gu[1] / 8192.0)
    for cx in range(cell_x0, cell_x1 + 1):
        x = px((cx * 8192.0, min_y))[0]
        draw.line((x, body_top, x, body_bottom), fill=(219, 231, 199, 72), width=1)
    for cy in range(cell_y0, cell_y1 + 1):
        y = px((min_x, cy * 8192.0))[1]
        draw.line((0, y, map_width, y), fill=(219, 231, 199, 72), width=1)

    # Roads use a subdued surface with an independent centreline.  Their width
    # remains driven by the plan's measured GU value; no arbitrary ribbon is added.
    scale_x = body_size[0] / max(rectangle.width_gu, 1.0)
    for road in document.get("authored_roads", []):
        if not isinstance(road, Mapping):
            continue
        points = _polyline_pixel(terrain, rectangle, body_size, road.get("polyline_plan_gu", []))
        points = [(x, y + body_top) for x, y in points]
        width_px = max(4, int(round(float(road.get("width_gu", 512.0)) * scale_x)))
        fill = COLORS["authored_major"] if road.get("class") in ("regional", "street") else COLORS["authored_local"]
        _draw_styled_road(draw, points, width_px, fill, COLORS["authored_center"], COLORS["authored_edge"])
    for alley in document.get("alleys", []):
        if not isinstance(alley, Mapping):
            continue
        points = _polyline_pixel(terrain, rectangle, body_size, alley.get("polyline_plan_gu", []))
        points = [(x, y + body_top) for x, y in points]
        width_px = max(3, int(round(float(alley.get("width_gu", 256.0)) * scale_x)))
        _draw_styled_road(draw, points, width_px, COLORS["alley"], COLORS["alley_center"], COLORS["authored_edge"])

    # Bounded open-space surfaces are translucent and outlined, not opaque debug
    # overlays.  Their labels are placed later by the collision gate.
    for region in document.get("road_surface_polygons", []):
        if not isinstance(region, Mapping):
            continue
        colour = COLORS["plaza"] if region.get("kind") in ("plaza", "market_circle") else COLORS["surface"]
        _composite_polygon(overlay, [px(point) for point in region.get("polygon_plan_gu", [])], colour,
                           outline=(255, 219, 130, 205), width=2)
    for court in document.get("shared_courts", []):
        if not isinstance(court, Mapping):
            continue
        _composite_polygon(overlay, [px(point) for point in court.get("polygon_plan_gu", [])], COLORS["court"],
                           outline=(136, 219, 151, 220), width=2)
    for district in document.get("districts", []):
        if not isinstance(district, Mapping):
            continue
        _composite_polygon(overlay, [px(point) for point in district.get("polygon_plan_gu", [])], COLORS["district"],
                           outline=(255, 221, 134, 150), width=1)

    # Access links are deliberately local.  An authored ``access_links``
    # polyline is rendered as supplied; otherwise the target only determines a
    # capped short dashed stub.  We never draw a guessed door-to-centroid line.
    access_link_count = 0
    for record in records:
        for door in record.doors:
            target_id = door.get("target_id")
            target = targets.get(str(target_id)) if target_id else None
            explicit = next((link for link in record.access_links
                             if str(link.get("door_id", "")) == str(door.get("door_id", ""))
                             and str(link.get("target_id", "")) == str(target_id)), None)
            explicit_polyline = explicit.get("polyline_plan_gu") if explicit else None
            geometry = access_render_geometry(door["position_plan_gu"], target,
                                              explicit_polyline)
            if geometry is None:
                continue
            points = [px(point) for point in geometry["points"]]
            colour = COLORS.get(f"door_{door.get('intent', 'public')}", COLORS["access_link"])
            _draw_dashed_polyline(draw, points, (colour[0], colour[1], colour[2], 185), width=2)
            end = points[-1]
            draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4),
                         fill=(colour[0], colour[1], colour[2], 180))
            access_link_count += 1

    # Footprints are exact transformed polygons.  The selected lot is visually
    # picked out without changing its geometry or source semantics.
    for record in records:
        colour = COLORS["footprint_karthgad"] if "karth" in record.kit.casefold() else COLORS["footprint_markarth"]
        outline = COLORS["selected_outline"] if record.lot_id == selected_lot_id else COLORS["footprint_outline"]
        _composite_polygon(overlay, [px(point) for point in record.hull], colour,
                           outline=outline, width=3 if record.lot_id == selected_lot_id else 2)
        if record.lot_id == selected_lot_id and record.hull:
            # A broad dark/bright double stroke makes the selected lot legible
            # at a glance while the later arrows are composited above it.
            selected_points = selected_highlight_geometry([px(point) for point in record.hull])["outline"]
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.line(selected_points, fill=(38, 28, 8, 240),
                              width=SELECTED_GLOW_WIDTH_PX, joint="curve")
            overlay_draw.line(selected_points, fill=COLORS["selected_outline"],
                              width=4, joint="curve")

    # Reserve all building bounds and every arrow shaft before any label is
    # accepted.  This is intentionally conservative: a label cannot hide a
    # footprint, even when it is not the selected lot.
    forbidden = [_rect_from_points([px(point) for point in record.hull], pad=4.0) for record in records if record.hull]
    arrow_data: list[tuple[StampRenderRecord, int, dict[str, Any], tuple[float, float], tuple[float, float]]] = []
    arrow_length_px = max(44.0, ARROW_LENGTH_GU * scale_x)
    for record in records:
        for door_index, door in enumerate(record.doors, start=1):
            start = px(door["position_plan_gu"])
            angle = math.radians(float(door["heading_deg"]))
            tip = (start[0] + arrow_length_px * math.cos(angle),
                   start[1] - arrow_length_px * math.sin(angle))
            forbidden.append(_rect_from_points([start, tip], pad=8.0))
            arrow_data.append((record, door_index, door, start, tip))
    labeler = LabelPlacer(draw, (0, body_top, map_width, body_bottom), forbidden)

    # Draw arrows after footprint fills.  Their shafts start exactly at the
    # measured anchor and get a dark halo so direction survives terrain/outline
    # contrast at the supplied proof resolution.
    for record, door_index, door, start, tip in arrow_data:
        intent = str(door.get("intent", "public"))
        colour = COLORS.get(f"door_{intent}", COLORS["door_public"])
        angle = math.atan2(-(tip[1] - start[1]), tip[0] - start[0])
        draw.line((start[0], start[1], tip[0], tip[1]), fill=(9, 20, 24, 245), width=10)
        draw.line((start[0], start[1], tip[0], tip[1]), fill=colour, width=6)
        head = max(10.0, min(16.0, arrow_length_px * 0.28))
        left = (tip[0] - head * math.cos(angle - 0.52), tip[1] + head * math.sin(angle - 0.52))
        right = (tip[0] - head * math.cos(angle + 0.52), tip[1] + head * math.sin(angle + 0.52))
        draw.polygon([tip, left, right], fill=(9, 20, 24, 245))
        inner = max(6.0, head - 4.0)
        left = (tip[0] - inner * math.cos(angle - 0.52), tip[1] + inner * math.sin(angle - 0.52))
        right = (tip[0] - inner * math.cos(angle + 0.52), tip[1] + inner * math.sin(angle + 0.52))
        draw.polygon([tip, left, right], fill=colour)
        draw.ellipse((start[0] - 7, start[1] - 7, start[0] + 7, start[1] + 7),
                     fill=COLORS["door"], outline=COLORS["door_outline"], width=2)

    # Door IDs are local optional annotations.  A crowded tip simply keeps its
    # large arrow and anchor; the full source ID, intent, heading, target, and
    # step data remain in the selected-lot panel.
    for record, door_index, door, start, tip in arrow_data:
        suffix = str(door.get("door_id", "")).split("_")[-1]
        intent = str(door.get("intent", "public"))
        colour = COLORS.get(f"door_{intent}", COLORS["door_public"])
        labeler.place(f"D{door_index} {suffix}", tip, colour, _font(9),
                      candidates=local_door_label_candidates(tip, float(door["heading_deg"])))

    # The selected tag is intentionally adjacent to the highlighted footprint,
    # not in a distant lane.  LabelPlacer prevents it from covering its own
    # hull or any measured arrow shaft.
    selected_record = next((item for item in records if item.lot_id == selected_lot_id), None)
    if selected_record is not None and selected_record.hull:
        selected_geometry = selected_highlight_geometry([px(point) for point in selected_record.hull])
        selected_bbox = tuple(selected_geometry["bbox"])
        tag_candidates = [tuple(point) for point in selected_geometry["tag_candidates"]]
        labeler.place("SELECTED", tag_candidates[0],
                      COLORS["selected_outline"], _font(10),
                      candidates=tag_candidates)

    # Collision-aware labels: cells and infrastructure first, then lot labels.
    # Stable ordering means byte output is independent of dictionary insertion
    # order. Door labels were handled locally above and are never moved to a
    # perimeter lane.
    for cx in range(cell_x0, cell_x1):
        for cy in range(cell_y0, cell_y1):
            point = px((cx * 8192.0 + 256.0, cy * 8192.0 + 256.0))
            labeler.place(f"cell {actual_min_x + cx},{actual_min_y + cy}",
                          (point[0] - 30, point[1] - 8), (207, 224, 190, 225), _font(9),
                          candidates=((point[0] - 30, point[1] - 8), (point[0] + 8, point[1] - 8),
                                      (point[0] - 30, point[1] + 10)), leader_from=point)
    for source in document.get("existing_source_roads", []):
        if not isinstance(source, Mapping):
            continue
        target = targets.get(str(source.get("edge_id")))
        polyline = target.get("polyline") if target else None
        if isinstance(polyline, list) and polyline:
            middle = px(polyline[len(polyline) // 2])
            short_id = str(source.get("edge_id", "edge"))[-10:]
            labeler.place(f"EXISTING · {short_id}", (middle[0] + 10, middle[1] - 24),
                          (205, 255, 255, 255), _font(11),
                          candidates=((middle[0] + 10, middle[1] - 24), (middle[0] + 10, middle[1] + 8),
                                      (middle[0] - 130, middle[1] - 24)), leader_from=middle)
    for road in document.get("authored_roads", []):
        if isinstance(road, Mapping) and road.get("polyline_plan_gu"):
            polyline = road["polyline_plan_gu"]
            point = px(polyline[len(polyline) // 2])
            labeler.place(f"STREET · {road.get('road_id', 'road')}", (point[0] + 10, point[1] + 10),
                          COLORS["annotation"], _font(11),
                          candidates=((point[0] + 10, point[1] + 10), (point[0] + 10, point[1] - 28),
                                      (point[0] - 120, point[1] + 10)), leader_from=point)
    for alley in document.get("alleys", []):
        if isinstance(alley, Mapping) and alley.get("polyline_plan_gu"):
            polyline = alley["polyline_plan_gu"]
            point = px(polyline[len(polyline) // 2])
            labeler.place(f"ALLEY · {alley.get('alley_id', 'alley')}", (point[0] + 8, point[1] + 8),
                          COLORS["annotation"], _font(10),
                          candidates=((point[0] + 8, point[1] + 8), (point[0] + 8, point[1] - 25),
                                      (point[0] - 100, point[1] + 8)), leader_from=point)
    for region in document.get("road_surface_polygons", []):
        if isinstance(region, Mapping) and region.get("polygon_plan_gu"):
            point = px(polygon_centroid(region["polygon_plan_gu"]))
            labeler.place(f"PLAZA · {region.get('region_id', 'surface')}", (point[0] - 45, point[1] - 9),
                          COLORS["annotation"], _font(11),
                          candidates=((point[0] - 45, point[1] - 9), (point[0] - 45, point[1] + 18),
                                      (point[0] + 10, point[1] - 9)), leader_from=point)
    for court in document.get("shared_courts", []):
        if isinstance(court, Mapping) and court.get("polygon_plan_gu"):
            point = px(polygon_centroid(court["polygon_plan_gu"]))
            labeler.place(f"COURT · {court.get('court_id', 'court')}", (point[0] - 45, point[1] - 9),
                          COLORS["annotation"], _font(11),
                          candidates=((point[0] - 45, point[1] - 9), (point[0] - 45, point[1] + 18),
                                      (point[0] + 10, point[1] - 9)), leader_from=point)
    for district in document.get("districts", []):
        if isinstance(district, Mapping) and district.get("polygon_plan_gu"):
            point = px(polygon_centroid(district["polygon_plan_gu"]))
            labeler.place(str(district.get("label", district.get("district_id", "district"))),
                          (point[0] - 70, point[1] - 10), COLORS["annotation"], _font(11),
                          candidates=((point[0] - 70, point[1] - 10), (point[0] - 70, point[1] + 18)), leader_from=point)

    for record in records:
        if not record.hull:
            continue
        hull_px = [px(point) for point in record.hull]
        bbox = _rect_from_points(hull_px)
        label = f"{record.lot_id} · {record.category}"
        candidates = ((bbox[0], bbox[1] - 25), (bbox[2] + 8, bbox[1]),
                      (bbox[0], bbox[3] + 8), (bbox[0] - 150, bbox[1]))
        labeler.place(label, candidates[0], COLORS["annotation"], _font(10),
                      candidates=candidates, leader_from=(bbox[0], bbox[1]))

    # Source evidence is shown for the selected lot only in the map, while the
    # panel always contains the complete evidence.  This prevents a relief/burial
    # text carpet from obscuring doors or open-space surfaces.
    selected = next((item for item in records if item.lot_id == selected_lot_id), None)
    if selected is not None and (show_source_terrain or show_burial_envelope):
        center = px(polygon_centroid([list(point) for point in selected.hull]))
        metrics = terrain.terrain_metrics(selected.hull)
        extra = []
        if show_source_terrain:
            extra.append(f"site relief {float(metrics.get('relief_gu', 0.0)):.0f} GU")
        if show_burial_envelope:
            extra.append(f"source burial 0–{float(selected.terrain_envelope.get('burial_depth_gu', 0.0)):.0f} GU")
        labeler.place(" · ".join(extra), (center[0] - 70, center[1] + 28),
                      (206, 245, 236, 255), _font(10),
                      candidates=((center[0] - 70, center[1] + 28), (center[0] - 70, center[1] - 45)),
                      leader_from=center)

    for placement in document.get("stamps", []):
        if isinstance(placement, Mapping) and placement.get("intentional_slope_capable"):
            point = placement.get("position_plan_gu")
            if isinstance(point, Sequence) and len(point) == 2:
                p = px(point)
                draw.ellipse((p[0] - 13, p[1] - 13, p[0] + 13, p[1] + 13),
                             outline=COLORS["slope"], width=4)
                labeler.place("SLOPE-CAPABLE (ADVISORY)", (p[0] + 18, p[1] - 10),
                              COLORS["annotation"], _font(10),
                              candidates=((p[0] + 18, p[1] - 10), (p[0] - 150, p[1] - 10)), leader_from=p)

    for annotation in document.get("annotations", []):
        if isinstance(annotation, Mapping) and isinstance(annotation.get("position_plan_gu"), Sequence):
            point = px(annotation["position_plan_gu"])
            labeler.place(str(annotation.get("text", "")), (point[0] - 100, point[1] - 10),
                          COLORS["annotation"], _font(10),
                          candidates=((point[0] - 100, point[1] - 10), (point[0] - 100, point[1] + 18)), leader_from=point)

    # Adversarial markers are explicit review evidence, not a substitute for the
    # JSON report.  Every marker label includes exact lot ids; door ids are also
    # listed in the side panel and remain in the report.
    marker_count = 0
    if show_advisory_markers and advisory_report:
        lookup = {record.lot_id: record for record in records}
        marker_callouts: set[tuple[str, str, str]] = set()
        for severity, key, colour in (("HARD", "hard_errors", COLORS["bad"]),
                                      ("ADVISORY", "advisories", COLORS["advisory"])):
            for finding in advisory_report.get(key, []):
                lot_ids = [str(value) for value in finding.get("lot_ids", [])]
                for lot_id in lot_ids[:2]:
                    record = lookup.get(lot_id)
                    if record is None:
                        continue
                    point = px(polygon_centroid([list(p) for p in record.hull])) if record.hull else px(record.position)
                    if severity == "HARD":
                        draw.polygon([(point[0], point[1] - 14), (point[0] + 14, point[1]),
                                      (point[0], point[1] + 14), (point[0] - 14, point[1])],
                                     fill=colour, outline=(18, 18, 18, 255))
                    else:
                        draw.ellipse((point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10),
                                     fill=colour, outline=(18, 18, 18, 255))
                    door_ids = [str(value) for value in finding.get("door_ids", [])]
                    callout_key = (severity, lot_id, door_ids[0] if door_ids else "")
                    if callout_key not in marker_callouts:
                        marker_callouts.add(callout_key)
                        compact_code = str(finding.get("code", "finding")).replace("_", " ")
                        compact_code = compact_code.split(" ")[0] if severity == "ADVISORY" else compact_code
                        exact = f"{severity} · {compact_code} · {lot_id}"
                        if door_ids:
                            exact += f" · D {door_ids[0].split('_')[-1]}"
                        labeler.place(exact, (point[0] + 18, point[1] - 12), colour, _font(9),
                                      candidates=((point[0] + 18, point[1] - 12), (point[0] + 18, point[1] + 12),
                                                  (point[0] - 180, point[1] - 12), (point[0] - 180, point[1] + 12),
                                                  (point[0] + 24, point[1] - 48)), leader_from=point)
                    marker_count += 1

    canvas.alpha_composite(overlay)
    if include_header:
        header = ImageDraw.Draw(canvas)
        header.rectangle((0, 0, size[0], HEADER_PX), fill=(13, 22, 28, 248))
        header.text((MAP_MARGIN_PX, 13), title, fill=(255, 225, 112, 255), font=_font(19))
        header.text((MAP_MARGIN_PX, 38),
                    f"cells {rectangle.cell_bounds} · plan GU {rectangle.requested_width_gu:.0f}×{rectangle.requested_height_gu:.0f} · exact masks + visual interpolation",
                    fill=(198, 218, 204, 255), font=_font(10))

    panel_audit = _draw_detail_panel(canvas, map_width, body_top, body_bottom, records,
                                     selected_lot_id, advisory_report, show_advisory_markers)
    context_inset = terrain.render_full_site_inset(size=(128, 128)) if include_context_inset and include_legend else None
    audit: dict[str, Any] = {
        "canvas_size": list(size),
        "map_band_px": [0, body_top, map_width, body_bottom],
        "header_band_px": [0, 0, size[0], HEADER_PX] if include_header else None,
        "legend_band_px": [0, body_bottom, size[0], size[1]] if include_legend else None,
        "detail_panel_px": panel_audit["panel_px"],
        "geometry_under_bands_px": 0,
        "existing_context_edge_count": existing_count,
        "existing_source_road_count": len(document.get("existing_source_roads", [])),
        "authored_road_count": len(document.get("authored_roads", [])),
        "alley_count": len(document.get("alleys", [])),
        "road_surface_polygon_count": len(document.get("road_surface_polygons", [])),
        "shared_court_count": len(document.get("shared_courts", [])),
        "stamp_count": len(records), "door_count": len(arrow_data),
        "door_ids": [f"{record.lot_id}:{door['door_id']}" for record in records for door in record.doors],
        "multi_door_stamp_count": sum(1 for record in records if len(record.doors) > 1),
        "source_terrain_shown": bool(show_source_terrain),
        "burial_envelope_shown": bool(show_burial_envelope),
        "slope_capable_indicator_count": sum(1 for placement in document.get("stamps", [])
                                             if isinstance(placement, Mapping) and placement.get("intentional_slope_capable")),
        "access_link_count": access_link_count,
        "advisory_marker_count": marker_count,
        "selected_lot_id": selected_lot_id,
        "label_audit": {
            "placed_count": len(labeler.placed),
            "unplaced_count": len(labeler.unplaced),
            "unplaced": list(labeler.unplaced),
            # Every accepted label passed the geometry/label collision gate.
            # Optional context labels can be omitted when the map is dense;
            # omission is reported separately rather than misreported as an
            # overlap defect.
            "collision_free": True,
            "complete": not labeler.unplaced,
            "required_geometry_labels_unplaced": [text for text in labeler.unplaced
                                                   if text.startswith(("HARD", "ADVISORY"))],
            "forbidden_geometry_count": len(labeler.forbidden),
        },
        "road_style": {"existing_centreline": True, "authored_edge_and_centreline": True,
                       "alley_narrower_than_street": True, "authored_surface_alpha": 135},
        "adversarial_proof_markers": bool(show_advisory_markers),
    }
    if include_legend:
        _draw_legend(canvas, title, audit, records, context_inset)
    return canvas, audit


__all__ = ["ACCESS_STUB_LENGTH_GU", "COLORS", "LabelPlacer", "LEGEND_PX",
           "HEADER_PX", "SIDE_PANEL_PX", "StampRenderRecord",
           "access_render_geometry", "local_door_label_candidates",
           "render_plan_layers", "resolve_stamps", "selected_highlight_geometry",
           "_rect_intersects"]
