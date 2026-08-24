#!/usr/bin/env python3
"""Render the composed city wall over the existing townlayout scene.

Purpose: visualization only.  Takes the composed wall stamp doc (plan-frame
GU), converts its members exactly like ``render_current_townlayout.py``
converts seated objects (survey origin added, southwest anchor cell
subtracted, 0.01 scene units), swaps them into the already-built
townlayout flat-render scene (same LAND plugin, lighting, camera contract),
and renders through the existing ``tools/blender_flat_render.py`` worker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.engine_transform import blender_xyz_euler_for_tes3_rotation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--wall-doc",
        type=Path,
        default=ROOT / "output/cityforge/wallkit/falkreath_wall_v1.json",
    )
    parser.add_argument(
        "--base-scene",
        type=Path,
        default=ROOT
        / "output/cityforge/townlayout/falkreath_phase21_city80/r16_proc_house/13_city_layout_fix2/3d_render_fix1/current_townlayout_scene.json",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/cityforge/wallkit")
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument(
        "--land-json",
        type=Path,
        default=ROOT
        / "output/cityforge/townlayout/falkreath_phase21_city80/r16_proc_house/16_land_fix1/townlayout_land_records.json",
    )
    parser.add_argument(
        "--land-manifest",
        type=Path,
        default=ROOT
        / "output/cityforge/townlayout/falkreath_phase21_city80/r16_proc_house/16_land_fix1/townlayout_land_records.manifest.json",
    )
    parser.add_argument(
        "--tes3conv", type=Path, default=ROOT / "tes3conv-master/tes3conv.exe"
    )
    args = parser.parse_args()

    survey = json.loads(
        (ROOT / "output/cityforge/sites/falkreath_v1/site_survey.json").read_text(encoding="utf-8")
    )
    origin_x, origin_y = (float(v) for v in survey["frame"]["origin_gu"])

    base_scene_path = args.base_scene.resolve()
    document = json.loads(base_scene_path.read_text(encoding="utf-8"))

    # Terrain anchor comes from the LATEST land manifest; it must be known
    # before member positions are converted.
    land_manifest = json.loads(args.land_manifest.resolve().read_text(encoding="utf-8"))
    cells = sorted({tuple(int(v) for v in cell) for cell in land_manifest["affected_cells"]})
    anchor = min(cells)

    wall = json.loads(args.wall_doc.resolve().read_text(encoding="utf-8"))
    # Composer members are xy-relative to the doc origin (first path point);
    # add it back to get survey-plan GU before the survey-origin shift.
    wall_ox, wall_oy = (float(v) for v in wall.get("origin_gu", [0.0, 0.0]))
    meshes = []
    for index, member in enumerate(wall["members"]):
        px, py, pz = (float(v) for v in member["offset_gu"])
        absolute_x = px + wall_ox + origin_x
        absolute_y = py + wall_oy + origin_y
        euler = blender_xyz_euler_for_tes3_rotation(member.get("rotation") or [0.0, 0.0, 0.0])
        meshes.append(
            {
                "id": f"wall_{index:04d}_{member['piece_id']}",
                "mesh": member["model_key"],
                "position": [
                    round((absolute_x - anchor[0] * 8192.0) * 0.01, 6),
                    round((absolute_y - anchor[1] * 8192.0) * 0.01, 6),
                    round(pz * 0.01, 6),
                ],
                "rotation": [round(float(v), 9) for v in euler],
                "scale": float(member.get("scale", 1.0)),
                "building": True,
                "source_reference": member["source_id"],
            }
        )

    document["meshes"] = meshes
    document["scene_name"] = f"Wall_Terrain_{wall['stamp_id']}"
    document["source"] = {"wall_doc": str(args.wall_doc.resolve()), "base_scene": str(base_scene_path)}

    # Rebuild the terrain plugin from the LATEST land records so the painted
    # roads match the current layout.
    document["terrain"]["anchor_grid"] = list(anchor)
    document["terrain"]["cells"] = [list(cell) for cell in cells]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    land_plugin = out_dir / "wall_terrain_land.esp"
    import subprocess

    conversion = subprocess.run(
        [str(args.tes3conv.resolve()), str(args.land_json.resolve()), str(land_plugin), "-o"],
        cwd=str(out_dir), check=False, capture_output=True, text=True,
    )
    if conversion.returncode != 0 or not land_plugin.is_file():
        print("FAILURE: wall terrain LAND conversion failed")
        print((conversion.stderr or conversion.stdout).strip())
        return conversion.returncode or 1
    document["terrain"]["plugin"] = str(land_plugin)
    document["terrain"]["texture_plugin"] = str(land_plugin)

    scene_path = out_dir / f"{wall['stamp_id']}_terrain_scene.json"
    scene_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    png = args.png.resolve() if args.png else out_dir / f"{wall['stamp_id']}_on_terrain.png"
    blender = ROOT / "tools/blender_flat_render.py"

    command = [
        r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
        "-b",
        "--python",
        str(blender),
        "--",
        str(scene_path),
        str(png),
    ]
    print("running:", " ".join(command))
    result = subprocess.run(command, cwd=str(ROOT), check=False)
    if result.returncode != 0 or not png.is_file():
        print(f"FAILURE: render exited {result.returncode}")
        return result.returncode or 1
    print(f"wrote {png} ({len(meshes)} wall members over townlayout terrain)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
