#!/usr/bin/env python3
"""Build a controlled render scene from a wall-stair diagnostic stamp.

Pipeline position
-----------------
This is a diagnostic scene adapter after stamp construction and before Blender
rendering. It reads a stamp plus a JSON camera/scene profile, reuses the stamp's
measured members and ground plane, and writes one direct-view scene. It contains
no geometry decisions; every camera, resolution, and framing value is supplied
by the profile so the same adapter can inspect other wall kits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))

from procgen.kit_house_grammar import canonical_json_bytes, stamp_to_sheet_scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stamp", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    stamp = json.loads(args.stamp.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    scene = stamp_to_sheet_scene(stamp)
    view = profile["views"][args.view]
    scene["scene_name"] = str(view["scene_name"])
    scene["camera"] = dict(view["camera"])
    if "lighting" in profile:
        scene["lighting"] = dict(profile["lighting"])
    if "lighting" in view:
        scene["lighting"] = dict(view["lighting"])
    if "ground_plane" in stamp:
        scene["ground_plane"] = dict(stamp["ground_plane"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json_bytes(scene))
    print(json.dumps({"scene": str(args.out), "meshes": len(scene["meshes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
