"""Compose a measured wall kit from an authoritative R2W inner-wall stage.

Reads the fitted centerline and arterial-only gate records emitted by
``build_town_inner_wall.py`` plus the authoritative settlement survey height
field. The resulting wall document uses the same plan GU frame as the
checkpoint, follows real terrain, and inserts measured stair/tower transitions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen.wall_compose import WallComposeError, compose_city_wall  # noqa: E402
from procgen.wall_kit import load_kit  # noqa: E402


def _terrain_from_npz(path: Path):
    data = np.load(path)
    required = {"x_gu", "y_gu", "height_gu"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"terrain NPZ missing arrays: {missing}")
    xs = np.asarray(data["x_gu"], dtype=float)
    ys = np.asarray(data["y_gu"], dtype=float)
    heights = np.asarray(data["height_gu"], dtype=float)
    if xs.ndim != 1 or ys.ndim != 1 or heights.shape != (len(ys), len(xs)):
        raise ValueError("terrain NPZ has incompatible coordinate/height shapes")
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("terrain NPZ needs at least a 2x2 height grid")
    step_x = float(xs[1] - xs[0])
    step_y = float(ys[1] - ys[0])
    if step_x <= 0.0 or step_y <= 0.0:
        raise ValueError("terrain NPZ coordinates must be strictly increasing")

    def sample(x: float, y: float) -> float:
        fx = (float(x) - xs[0]) / step_x
        fy = (float(y) - ys[0]) / step_y
        ix = min(max(int(fx), 0), len(xs) - 2)
        iy = min(max(int(fy), 0), len(ys) - 2)
        tx, ty = fx - ix, fy - iy
        return float(
            heights[iy, ix] * (1.0 - tx) * (1.0 - ty)
            + heights[iy, ix + 1] * tx * (1.0 - ty)
            + heights[iy + 1, ix] * (1.0 - tx) * ty
            + heights[iy + 1, ix + 1] * tx * ty
        )

    return sample


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inner-wall", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    terrain = parser.add_mutually_exclusive_group(required=True)
    terrain.add_argument("--terrain-npz", type=Path)
    terrain.add_argument("--flat-z", type=float)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stamp-id", default="composed_city_wall")
    args = parser.parse_args(argv)

    stage = json.loads(args.inner_wall.read_text(encoding="utf-8"))
    ring = (stage.get("inner_wall") or {}).get("centerline") or []
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        raise ValueError("inner-wall checkpoint has no fitted centerline")
    gates = []
    for gate in stage.get("wall_gates") or []:
        tangent = gate["arterial_tangent"]
        gates.append({
            "gate_id": gate.get("gate_id"),
            "position_xy": list(gate["position"]),
            "heading_deg": math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))),
        })
    kit = load_kit(args.kit)
    terrain_fn = (
        _terrain_from_npz(args.terrain_npz)
        if args.terrain_npz is not None
        else lambda _x, _y: float(args.flat_z)
    )
    try:
        wall = compose_city_wall(
            [tuple(point) for point in ring], gates, terrain_fn,
            kit, stamp_id=args.stamp_id,
        )
    except WallComposeError as exc:
        print(f"FAILURE: wall composition {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(wall, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "members": len(wall["members"]),
        "provenance": wall["provenance"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
