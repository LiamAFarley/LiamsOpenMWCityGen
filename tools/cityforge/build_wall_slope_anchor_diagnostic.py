#!/usr/bin/env python3
"""Build a minimal wall-ramp-wall diagnostic from measured kit walk anchors.

Inputs are a validated wall kit and output stamp path. The selected wall and
slope, their semantic connection anchors, and all dimensions come from the
kit's ``slope_assembly``. The output is a real-NIF D-STAMP diagnostic consumed
by the standard wall-transition scene builder and renderer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))

from procgen.kit_house_grammar import canonical_json_bytes  # noqa: E402
from procgen.wall_kit import load_kit, piece_by_id  # noqa: E402


def _member(
    piece: dict, source_id: str, role: str, position: list[float], scale: float = 1.0
) -> dict:
    return {
        "category": "exterior",
        "is_door": False,
        "model_key": str(piece["model_key"]),
        "offset_gu": [round(float(value), 4) for value in position],
        "record_type": "STAT",
        "rotation": [0.0, 0.0, 0.0],
        "scale": round(float(scale), 9),
        "source_id": source_id,
        "structural_role": role,
    }


def build(kit: dict, stamp_id: str) -> dict:
    assembly = kit["slope_assembly"]
    wall = piece_by_id(kit, str(assembly["wall_piece_id"]))
    slope = piece_by_id(kit, str(assembly["slope_piece_id"]))
    if wall is None or slope is None:
        raise ValueError("slope assembly references a missing wall or slope piece")
    wall_surface = wall["walk_surface"]
    slope_surface = slope["walk_surface"]
    entry = [float(value) for value in slope_surface["entry_local_gu"]]
    exit_anchor = [float(value) for value in slope_surface["exit_local_gu"]]
    span = exit_anchor[0] - entry[0]
    if span <= 0.0:
        raise ValueError("slope connection anchors must increase along local x")

    overlap = float(assembly["minimum_deck_overlap_gu"])
    if abs(overlap) > 1e-9:
        raise ValueError("authored slope endpoints require exact 0 GU overlap")
    slope_width = (
        float(slope_surface["lateral_bounds_gu"][1])
        - float(slope_surface["lateral_bounds_gu"][0])
    )
    wall_width = (
        float(wall_surface["lateral_bounds_gu"][1])
        - float(wall_surface["lateral_bounds_gu"][0])
    )
    slope_scale = wall_width / slope_width
    scaled_span = span * slope_scale
    wall_length = float(wall["length_gu"])
    low_seam_x = -scaled_span / 2.0
    high_seam_x = scaled_span / 2.0
    low_wall_x = low_seam_x - wall_length / 2.0
    high_wall_x = high_seam_x + wall_length / 2.0
    low_bottom = 0.0
    low_pivot_z = low_bottom + float(wall["base_offset_gu"])
    low_deck_z = low_pivot_z + float(wall_surface["surface_z_gu"])
    slope_center_local = 0.5 * (entry[0] + exit_anchor[0]) * slope_scale
    slope_pivot_x = -slope_center_local
    slope_pivot_z = low_deck_z - entry[2] * slope_scale
    high_deck_z = slope_pivot_z + exit_anchor[2] * slope_scale
    rise = high_deck_z - low_deck_z
    high_bottom = high_deck_z - (
        float(wall["base_offset_gu"]) + float(wall_surface["surface_z_gu"])
    )
    high_pivot_z = high_bottom + float(wall["base_offset_gu"])

    members = [
        _member(wall, f"{stamp_id}_low_wall", "straight", [low_wall_x, 0.0, low_pivot_z]),
        _member(
            slope, f"{stamp_id}_slope", "slope",
            [slope_pivot_x, 0.0, slope_pivot_z], slope_scale,
        ),
        _member(wall, f"{stamp_id}_high_wall", "straight", [high_wall_x, 0.0, high_pivot_z]),
    ]
    return {
        "stamp_id": stamp_id,
        "building_type": "wall_slope_anchor_diagnostic",
        "ground_plane": {
            "mode": "steps",
            "color": [0.12, 0.32, 0.10, 1.0],
            "roughness": 0.95,
            "x_min_scene": -18.0,
            "x_split_scene": 0.0,
            "x_max_scene": 18.0,
            "y_min_scene": -5.0,
            "y_max_scene": 5.0,
            "z_low_scene": 0.0,
            "z_high_scene": round(high_bottom / 100.0, 6),
        },
        "members": members,
        "diagnostic": {
            "high_wall_bottom_gu": round(high_bottom, 3),
            "slope_anchor_span_gu": round(scaled_span, 3),
            "slope_overlap_each_end_gu": round(overlap, 3),
            "slope_rise_gu": round(rise, 3),
            "slope_scale": round(slope_scale, 9),
            "slope_width_gu": round(slope_width * slope_scale, 3),
            "walkway_width_gu": round(wall_width, 3),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stamp-id", default="falkreath_wall_slope_anchor_v1")
    args = parser.parse_args(argv)
    stamp = build(load_kit(args.kit), args.stamp_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json_bytes(stamp))
    print(json.dumps({"output": str(args.out), **stamp["diagnostic"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
