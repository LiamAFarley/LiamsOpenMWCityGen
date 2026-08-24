#!/usr/bin/env python3
"""Build a wall-stair diagnostic from measured semantic anchors in a kit JSON.

Inputs are a validated wall kit and an output stamp path. All member counts,
slot spans, walk anchors, and dimensions come from the kit. The output is a
real-NIF D-STAMP diagnostic consumed by the ordinary flat renderer.
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


def _member(piece: dict, source_id: str, role: str, position: list[float], scale: float = 1.0) -> dict:
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
    assembly = kit["stair_assembly"]
    wall = piece_by_id(kit, str(assembly["wall_piece_id"]))
    stair = piece_by_id(kit, str(assembly["stair_piece_id"]))
    if wall is None or stair is None:
        raise ValueError("stair assembly references a missing wall or stair piece")
    walkway = wall["walk_surface"]
    treads = stair["walk_surface"]
    count = int(assembly["stair_count"])
    gap = float(assembly["transition_span_gu"])
    wall_length = float(wall["length_gu"])
    deck_min, deck_max = (float(value) for value in assembly["target_lateral_bounds_gu"])
    tread_min, tread_max = (float(value) for value in treads["lateral_bounds_gu"])
    deck_width = deck_max - deck_min
    tread_width = tread_max - tread_min
    stair_scale = deck_width / (count * tread_width)

    entry = [float(value) for value in treads["entry_local_gu"]]
    exit_anchor = [float(value) for value in treads["exit_local_gu"]]
    anchor_span = (exit_anchor[0] - entry[0]) * stair_scale
    overlap = 0.5 * (anchor_span - gap)
    if overlap < float(assembly["minimum_deck_overlap_gu"]):
        raise ValueError(f"scaled stair overlap {overlap:.3f} GU is below configured minimum")

    low_wall_x = -(gap / 2.0 + wall_length / 2.0)
    high_wall_x = -low_wall_x
    low_bottom = 0.0
    low_pivot_z = low_bottom + float(wall["base_offset_gu"])
    deck_z_local = float(walkway["surface_z_gu"])
    low_deck_z = low_pivot_z + deck_z_local

    stair_pivot_x = -0.5 * (entry[0] + exit_anchor[0]) * stair_scale
    stair_pivot_z = low_deck_z - entry[2] * stair_scale
    high_deck_z = stair_pivot_z + exit_anchor[2] * stair_scale
    high_bottom = high_deck_z - (float(wall["base_offset_gu"]) + deck_z_local)
    high_pivot_z = high_bottom + float(wall["base_offset_gu"])

    members = [
        _member(wall, f"{stamp_id}_low_wall", "straight", [low_wall_x, 0.0, low_pivot_z]),
    ]
    lane_width = deck_width / count
    for index in range(count):
        target_min = deck_min + index * lane_width
        pivot_y = target_min - tread_min * stair_scale
        members.append(_member(
            stair, f"{stamp_id}_stair_{index + 1}", "stair",
            [stair_pivot_x, pivot_y, stair_pivot_z], stair_scale,
        ))
    members.append(_member(
        wall, f"{stamp_id}_high_wall", "straight", [high_wall_x, 0.0, high_pivot_z]
    ))
    return {
        "stamp_id": stamp_id,
        "building_type": "wall_stair_anchor_diagnostic",
        "ground_plane": {
            "mode": "steps", "color": [0.12, 0.32, 0.10, 1.0], "roughness": 0.95,
            "x_min_scene": -18.0, "x_split_scene": 0.0, "x_max_scene": 18.0,
            "y_min_scene": -5.0, "y_max_scene": 5.0,
            "z_low_scene": 0.0, "z_high_scene": round(high_bottom / 100.0, 6),
        },
        "members": members,
        "diagnostic": {
            "deck_overlap_each_end_gu": round(overlap, 3),
            "high_wall_bottom_gu": round(high_bottom, 3),
            "stair_anchor_span_gu": round(anchor_span, 3),
            "stair_rise_gu": round(high_deck_z - low_deck_z, 3),
            "stair_scale": round(stair_scale, 9),
            "walkway_width_gu": round(deck_width, 3),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stamp-id", default="falkreath_wall_stair_anchor_v1")
    args = parser.parse_args(argv)
    stamp = build(load_kit(args.kit), args.stamp_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json_bytes(stamp))
    print(json.dumps({"output": str(args.out), **stamp["diagnostic"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
