#!/usr/bin/env python3
"""Render a generated house stamp to a 2x3 Blender sheet PNG.

Writes ``sheet_scene.json`` beside the output PNG, then invokes Blender::

    python tools/cityforge/render_generated_house.py \\
        --stamp output/cityforge/stamps/generated/stone/house_seed0001.json \\
        --out output/cityforge/stamps/generated/stone/house_seed0001_sheet_2x3.png

Requires ``blender`` on PATH (same contract as ``tools/mesh_thumbs.py``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.kit_house_grammar import canonical_json_bytes, stamp_to_sheet_scene  # noqa: E402

MESH_THUMBS = WORKSPACE / "tools" / "mesh_thumbs.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render generated house stamp sheet")
    parser.add_argument("--stamp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--scene-only", action="store_true", help="write sheet_scene.json only")
    args = parser.parse_args()

    stamp = json.loads(args.stamp.read_text(encoding="utf-8"))
    scene = stamp_to_sheet_scene(stamp)
    if isinstance(stamp.get("ground_plane"), Mapping):
        scene["ground_plane"] = dict(stamp["ground_plane"])
    scene_path = args.out.with_suffix(".scene.json")
    scene_path.write_bytes(canonical_json_bytes(scene))
    print(f"wrote {scene_path}")
    if args.scene_only:
        return 0

    blender = shutil.which(args.blender)
    if blender is None:
        print(f"FAILURE: render blender not found on PATH ({args.blender!r})", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        blender,
        "-b",
        "--factory-startup",
        "--python",
        str(MESH_THUMBS),
        "--",
        str(scene_path),
        str(args.out),
        "--layout",
        "2x3",
        "--resolution",
        "2304x1536",
        "--margin",
        "1.60",
    ]
    print("running:", " ".join(command))
    completed = subprocess.run(command, cwd=WORKSPACE, check=False)
    if completed.returncode != 0:
        print(f"FAILURE: render blender exit {completed.returncode}", file=sys.stderr)
        return completed.returncode
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
