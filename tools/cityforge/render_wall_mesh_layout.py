"""Overlay measured wall-mesh footprints on a townlayout top-down render.

Purpose
-------
The townlayout diagnostic already contains the terrain, roads, parcels, and
building footprints. This tool adds the fitted wall's actual per-piece
placement footprints, using the wall kit's evaluated z-slice extents rather
than the planning polygon alone.

Inputs
------
An existing ``city_layout_terrain.png``, its city-layout JSON, the site survey,
and the composed wall plus wall-kit JSON documents.

Outputs
-------
A browsable PNG with straight wall runs, towers, and gatehouses overlaid and
labelled. Source images and JSON inputs are never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from procgen.townlayout.site_context import _plan_to_px, diagnostic_view


def _corners(piece: dict[str, Any], member: dict[str, Any]) -> list[list[float]]:
    footprint = piece.get("footprint_slice") or {}
    outline = footprint.get("slice_outline_xy")
    if not outline:
        lo = footprint.get("slice_min_xy")
        hi = footprint.get("slice_max_xy")
    else:
        lo = hi = None
    if not outline and (not lo or not hi):
        half_l = float(piece["length_gu"]) / 2.0
        half_t = float(piece["thickness_gu"]) / 2.0
        lo, hi = [-half_l, -half_t], [half_l, half_t]
    scale = float(member.get("scale", 1.0))
    angle = -float(member.get("rotation", [0.0, 0.0, 0.0])[2])
    c, s = math.cos(angle), math.sin(angle)
    ox, oy = (float(member["offset_gu"][0]), float(member["offset_gu"][1]))
    origin_x, origin_y = (float(member["origin_gu"][0]), float(member["origin_gu"][1]))
    result = []
    points = outline or ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))
    for x, y in points:
        x *= scale
        y *= scale
        result.append([
            origin_x + ox + c * x - s * y,
            origin_y + oy + s * x + c * y,
        ])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city-layout", type=Path, required=True)
    parser.add_argument("--city-layout-png", type=Path, required=True)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--wall", type=Path, required=True)
    parser.add_argument("--wall-kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin-gu", type=float, default=4096.0)
    args = parser.parse_args(argv)

    city = json.loads(args.city_layout.read_text(encoding="utf-8"))
    survey = json.loads(args.survey.read_text(encoding="utf-8"))
    wall = json.loads(args.wall.read_text(encoding="utf-8"))
    kit = json.loads(args.wall_kit.read_text(encoding="utf-8"))
    base = Image.open(args.city_layout_png).convert("RGBA")
    _, mapping = diagnostic_view(
        {"_diagnostic_bounds": [city["city_domain"]]},
        args.survey.with_name("site_topdown.png"),
        survey,
        margin_gu=args.margin_gu,
    )
    if tuple(base.size) != tuple(mapping["resolution"]):
        mapping["resolution"] = list(base.size)

    pieces = {piece["model_key"]: piece for piece in kit["pieces"]}
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    colors = {
        "straight": (35, 115, 235, 180),
        "stair": (155, 80, 235, 220),
        "tower": (245, 190, 45, 220),
        "gatehouse": (225, 70, 55, 220),
        "door": (255, 125, 55, 230),
    }
    counts = {}
    for member in wall["members"]:
        piece = pieces.get(member["model_key"])
        if piece is None:
            raise ValueError(f"wall piece missing from kit: {member['model_key']}")
        member = dict(member)
        member["origin_gu"] = wall.get("origin_gu", [0.0, 0.0])
        ring = [_plan_to_px(x, y, mapping) for x, y in _corners(piece, member)]
        role = str(member.get("structural_role") or piece.get("role") or "straight")
        if member.get("is_door"):
            role = "door"
        color = colors.get(role, colors["straight"])
        draw.polygon(ring, fill=color, outline=(20, 25, 45, 255))
        counts[role] = counts.get(role, 0) + 1

    legend = [("wall run", colors["straight"]), ("stair", colors["stair"]),
              ("tower", colors["tower"]),
              ("gatehouse", colors["gatehouse"]), ("gate door", colors["door"])]
    x0, y0 = 22, 22
    draw.rounded_rectangle((x0, y0, x0 + 235, y0 + 28 + 24 * len(legend)),
                           radius=8, fill=(10, 20, 25, 220))
    draw.text((x0 + 10, y0 + 7), "MEASURED WALL-MESH FOOTPRINTS",
              fill=(255, 255, 255, 255))
    for index, (label, color) in enumerate(legend):
        yy = y0 + 31 + index * 24
        draw.rectangle((x0 + 10, yy, x0 + 24, yy + 14), fill=color,
                       outline=(230, 230, 230, 255))
        draw.text((x0 + 32, yy - 1), label, fill=(240, 240, 240, 255))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).save(args.output)
    print(json.dumps({"output": str(args.output), "members": len(wall["members"]),
                      "role_counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
