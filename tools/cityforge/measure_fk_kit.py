#!/usr/bin/env python3
"""Measure Falkreath kit NIF local AABBs via Blender and write kit_bounds.json."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_MESHES = [
    "sky/x/sky_FK_house_01_a.nif",
    "sky/x/sky_FK_house_02_a.nif",
    "sky/x/sky_FK_house_03_a.nif",
    "sky/x/sky_FK_house_04_a.nif",
    "sky/x/sky_FK_house_05_a.nif",
    "sky/x/sky_FK_house_06_a.nif",
    "sky/x/sky_FK_house_07_a.nif",
    "sky/x/sky_FK_house_08_a.nif",
    "sky/x/sky_FK_house_09_a.nif",
    "sky/x/sky_FK_house_10_a.nif",
    "sky/x/sky_FK_house_11_a.nif",
    "sky/x/sky_FK_house_12_a.nif",
    "sky/d/sky_ex_fk_door_01.nif",
    "sky/d/sky_ex_fk_door_02.nif",
    "sky/d/sky_ex_fk_door_03.nif",
    "sky/x/sky_FK_DFrame_01.nif",
    "sky/x/sky_FK_DFrame_02.nif",
    "sky/x/sky_FK_DFrame_03.nif",
    "sky/x/sky_FK_Window_01a.nif",
    "sky/x/sky_FK_Window_01b.nif",
    "sky/x/sky_FK_Window_02a.nif",
    "sky/x/sky_FK_Window_02b.nif",
    "sky/x/sky_FK_Window_03a.nif",
    "sky/x/sky_FK_Window_03b.nif",
    "sky/x/sky_FK_Window_04a.nif",
    "sky/x/sky_FK_Window_04b.nif",
    "sky/x/sky_FK_Window_04c.nif",
    "sky/x/sky_FK_Window_05a.nif",
    "sky/x/sky_FK_Window_05b.nif",
    "sky/x/sky_FK_Window_06a.nif",
    "sky/x/sky_FK_Window_06b.nif",
    "sky/x/sky_FK_Chimney_01.nif",
    "sky/x/sky_FK_Chimney_02.nif",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Falkreath kit AABB in GU")
    parser.add_argument("--out", type=Path, default=WORKSPACE / "configs" / "kits" / "falkreath" / "kit_bounds.json")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--mesh", action="append", dest="meshes")
    args = parser.parse_args()
    blender = shutil.which(args.blender)
    if blender is None:
        print(f"FAILURE: blender not found on PATH ({args.blender!r})", file=sys.stderr)
        return 1
    meshes = args.meshes or DEFAULT_MESHES
    script = WORKSPACE / "tools" / "cityforge" / "blender_kit_bounds.py"
    command = [blender, "-b", "--python", str(script), "--", str(args.out), *meshes]
    print("running:", " ".join(command))
    completed = subprocess.run(command, cwd=WORKSPACE, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
