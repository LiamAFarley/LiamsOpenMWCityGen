#!/usr/bin/env python3
"""Render active r15 cliff-category meshes in a three-angle comparison grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "output" / "rock_openface_profiles.json"


def active_meshes(scatter_path: Path) -> list[str]:
    document = json.loads(scatter_path.read_text(encoding="utf-8"))
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if value.get("category") == "cliff" and isinstance(value.get("mesh"), str):
                found.add(value["mesh"].casefold())
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(document)
    return sorted(found)


def build_scene(meshes: list[str], profiles: dict, scatter_path: Path, output_png: Path, direction: list[float]) -> dict:
    columns = 5
    spacing = 82.0
    entries = []
    for index, mesh_key in enumerate(meshes):
        profile = profiles[mesh_key]
        bbox = profile["bbox_local_game_units"]
        span = [float(bbox["max"][i]) - float(bbox["min"][i]) for i in range(3)]
        x = (index % columns) * spacing
        y = -(index // columns) * spacing
        entries.append(
            {
                "id": f"cliff_{index + 1:02d}",
                "mesh": profile["mesh"],
                "position": [x, y, 0.0],
                "rotation_z": 0.0,
                "scale": 1.0,
                "candidate_index": index + 1,
                "candidate_mesh": profile["mesh"],
                "candidate_span_gu": [round(value, 1) for value in span],
            }
        )
    return {
        "scene_name": "Active_Cliff_Candidates",
        "import": {
            "scale_correction": 0.01,
            "normalize_to_position": False,
            "use_existing_materials": True,
            "ignore_collision_nodes": True,
            "ignore_animations": True,
            "reuse_meshes": True,
        },
        "lighting": {
            "sun_energy": 2.5,
            "sun_angle_degrees": 25.0,
            "world_strength": 0.8,
            "fill_energy": 1500.0,
            "exposure": 0.5,
        },
        "camera": {
            "mode": "ORTHO",
            "view_direction": direction,
            "resolution": [1800, 1200],
            "margin": 0.8,
        },
        "meshes": entries,
        "output": {"default_render": str(output_png.resolve())},
        "source": {"scatter": str(scatter_path.resolve()), "profiles": str(PROFILE_PATH.resolve())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scatter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"))
    args = parser.parse_args()
    profiles_raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"]
    profiles = {key.casefold(): value for key, value in profiles_raw.items()}
    meshes = active_meshes(args.scatter)
    meshes = [mesh for mesh in meshes if mesh in profiles]
    if not meshes:
        raise SystemExit("no active cliff-category meshes found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    views = {
        "se": [0.72, -0.78, 0.72],
        "sw": [-0.72, -0.78, 0.72],
        "ne": [0.72, 0.78, 0.72],
    }
    for name, direction in views.items():
        png = args.output_dir / f"active_cliffs_{name}.png"
        scene = args.output_dir / f"active_cliffs_{name}.scene.json"
        scene.write_text(json.dumps(build_scene(meshes, profiles, args.scatter, png, direction), indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(meshes)} candidates -> {png}")
    (args.output_dir / "active_cliffs_manifest.json").write_text(
        json.dumps({"candidates": [{"index": i + 1, "mesh": profiles[mesh]["mesh"]} for i, mesh in enumerate(meshes)]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
