"""Build the current townlayout scene contract and launch the existing renderer.

Purpose: visualization only.  Convert seated stamp objects from the survey-local
townlayout frame into the flat-render scene schema, pair them with the authored
LAND/LTEX plugin, and render one oblique overview with the existing
``tools/blender_flat_render.py`` worker.  This does not author or alter ESP data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "output/cityforge/townlayout/falkreath_phase21_city80/r14_dense_final3/13_city_layout"
DEFAULT_SURVEY = ROOT / "output/cityforge/sites/falkreath_v1/site_survey.json"
DEFAULT_CONFIG = ROOT / "configs/procgen.json"
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--survey", type=Path, default=DEFAULT_SURVEY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--tes3conv", type=Path, default=ROOT / "tes3conv-master/tes3conv.exe")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE / "3d_render")
    args = parser.parse_args()

    base = args.base.resolve()
    survey = json.loads(args.survey.resolve().read_text(encoding="utf-8"))
    seated = json.loads((base / "stamp_objects_seated.json").read_text(encoding="utf-8"))
    land_json = base / "townlayout_land_records.json"
    land_manifest = json.loads((base / "townlayout_land_records.manifest.json").read_text(encoding="utf-8"))
    origin_x, origin_y = (float(v) for v in survey["frame"]["origin_gu"])
    anchor = min(tuple(int(v) for v in cell) for cell in land_manifest["affected_cells"])
    # The object product stores plan-frame GU.  Add the survey origin before
    # converting to the renderer's 0.01 scene-unit-per-GU convention.
    meshes = []
    for index, obj in enumerate(seated["objects"]):
        position = obj["world_position_gu"]
        matrix = obj["rotation_matrix_3x3"]
        absolute = (float(position[0]) + origin_x, float(position[1]) + origin_y, float(position[2]))
        scene_position = [
            round((absolute[0] - anchor[0] * 8192.0) * 0.01, 6),
            round((absolute[1] - anchor[1] * 8192.0) * 0.01, 6),
            round(absolute[2] * 0.01, 6),
        ]
        meshes.append({
            "id": f"townlayout_{index:04d}_{obj['reference_id']}",
            "mesh": obj["model_key"],
            "position": scene_position,
            "rotation_z": round(math.degrees(math.atan2(float(matrix[1][0]), float(matrix[0][0]))), 6),
            "scale": float(obj.get("scale", 1.0)),
            "building": obj.get("record_type") in {"STAT", "DOOR"},
            "source_reference": obj["reference_id"],
        })

    cells = sorted({tuple(int(v) for v in cell) for cell in land_manifest["affected_cells"]})
    document = {
        "scene_name": "Falkreath_Current_Townlayout_R13",
        "procgen_config": str(args.config.resolve()),
        "import": {
            "scale_correction": 0.01,
            "normalize_to_position": False,
            "use_existing_materials": True,
            "ignore_collision_nodes": True,
            "ignore_animations": True,
        },
        "lighting": {"sun_energy": 1.35, "sun_angle_degrees": 32.0, "world_strength": 0.9, "fill_energy": 2200.0, "fill_size_factor": 2.0, "view_look": "AgX - Medium Low Contrast"},
        "camera": {
            "mode": "ORTHO", "view": "oblique", "view_direction": [0.72, -0.72, 0.62],
            "resolution": [1600, 1000], "margin": 1.18,
            "subject": {"include_terrain": True, "padding": 8.0},
        },
        "terrain": {
            "enabled": True, "plugin": "current_townlayout_land.esp", "texture_plugin": "current_townlayout_land.esp",
            "texture_masters": [], "anchor_grid": list(anchor), "cells": [list(cell) for cell in cells],
            "decimate": 2, "max_seconds": 60.0,
        },
        "meshes": meshes,
        "source": {
            "layout": str(base / "city_layout.json"),
            "seated_objects": str(base / "stamp_objects_seated.json"),
            "land_records": str(land_json),
            "survey": str(args.survey.resolve()),
            "frame": "survey-local GU + survey origin; anchor southwest cell",
        },
    }
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    land_plugin = out / "current_townlayout_land.esp"
    conversion = subprocess.run(
        [str(args.tes3conv.resolve()), str(land_json), str(land_plugin), "-o"],
        cwd=str(out), check=False, capture_output=True, text=True,
    )
    if conversion.returncode != 0 or not land_plugin.is_file():
        print("FAILURE: current townlayout LAND conversion failed")
        print((conversion.stderr or conversion.stdout).strip())
        return conversion.returncode or 1
    document["terrain"]["plugin"] = str(land_plugin)
    document["terrain"]["texture_plugin"] = str(land_plugin)
    scene_path = out / "current_townlayout_scene.json"
    scene_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    png = out / "current_townlayout_overview.png"
    command = [str(args.blender.resolve()), "-b", "--python", str(ROOT / "tools/blender_flat_render.py"), "--", str(scene_path), str(png)]
    result = subprocess.run(command, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        print(f"FAILURE: current townlayout Blender render exited {result.returncode}")
        return result.returncode
    if not png.is_file():
        print("FAILURE: current townlayout Blender render produced no PNG")
        return 1
    print(json.dumps({"scene": str(scene_path), "render": str(png), "object_count": len(meshes), "cell_count": len(cells)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
