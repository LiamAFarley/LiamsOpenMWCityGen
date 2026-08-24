#!/usr/bin/env python3
"""Render the settlement city (buildings + terrain + scatter) from the SE.

Purpose: visualization only.  Builds two flat-render scene JSONs and renders
them with the existing ``tools/blender_flat_render.py`` worker (``--statics
all`` so scatter trees/rocks render):

* ``city_se_overview`` — ORTHO, framed to the full 49-cell terrain with a tight
  margin so the terrain fills the frame (no empty space around it).
* ``city_se_city`` — ORTHO, framed to the city buildings so the town fills the
  frame.

Groundcover is deliberately excluded (near-camera instanced in Morrowind; an
overview would show an unrealistic poly concentration).

Frames
------
* building refs: seated objects (plan-frame GU) -> world GU by adding the
  survey ``frame.origin_gu``.
* scatter refs: scatter doc (already global TES3 GU).
* both -> scene ``(world_gu - anchor * 8192) * 0.01``, ``anchor`` = min terrain
  cell.
* terrain: the r15 masterless LAND/LTEX ESP, used for both LAND and textures.
* circulation: r15's 69 terrain-following alley/apron requests remain in the
  separate diagnostic image, but are not drawn over the beauty renders; the
  final LAND/VTEX colors provide the road visualization.

Does not author or alter ESP data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping

from PIL import Image, ImageDraw

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen import engine_transform  # noqa: E402

ROOT = WORKSPACE
DEFAULT_BASE = ROOT / "output/cityforge/townlayout/falkreath_phase21_city80/r15_variety_retry3/16_terrain_smooth"
DEFAULT_REALIZATION = ROOT / "output/cityforge/townlayout/falkreath_phase21_city80/r15_variety_retry3/14_realization/circulation_realization.json"
DEFAULT_SURVEY = ROOT / "output/cityforge/sites/falkreath_v1/site_survey.json"
DEFAULT_CONFIG = ROOT / "configs/procgen.json"
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe")
DEFAULT_TERRAIN = ROOT / "tamriel.esm"
DEFAULT_TERRAIN_OVERRIDE = DEFAULT_BASE / "3d_render/current_townlayout_land.esp"
SCATTER_BOUNDS = (-95, -89, -11, -5)
SE_DIRECTION = [0.72, 0.72, 0.62]


def build_meshes(seated: dict, scatter: dict, anchor: tuple[int, int], origin: tuple[float, float]):
    ax, ay = (anchor[0] * 8192.0, anchor[1] * 8192.0)
    ox, oy = origin
    meshes = []
    building_ids: list[str] = []
    for index, obj in enumerate(seated["objects"]):
        position = obj["world_position_gu"]
        matrix = obj["rotation_matrix_3x3"]
        world = (float(position[0]) + ox, float(position[1]) + oy, float(position[2]))
        mid = f"bld_{index:04d}_{obj['reference_id']}"
        assignment = request.get("surface_assignment")
        if not isinstance(assignment, Mapping) or "raw_vtex" not in assignment:
            raise ValueError(
                f"circulation request {index} has no explicit surface raw_vtex assignment"
            )
        meshes.append(
            {
                "id": mid,
                "mesh": obj["model_key"],
                "position": [
                    round((world[0] - ax) * 0.01, 6),
                    round((world[1] - ay) * 0.01, 6),
                    round(world[2] * 0.01, 6),
                ],
                "rotation_z": round(math.degrees(math.atan2(float(matrix[1][0]), float(matrix[0][0]))), 6),
                "scale": float(obj.get("scale", 1.0)),
                "building": True,
                "source_reference": obj["reference_id"],
            }
        )
        building_ids.append(mid)
    sindex = 0
    for cell in scatter["density"]["cells"]:
        for ref in cell.get("refs", []):
            pos = ref["position_gu"]
            rotation = ref.get("rotation_radians") or [0.0, 0.0, 0.0]
            euler = engine_transform.blender_xyz_euler_for_tes3_rotation(
                [float(v) for v in rotation]
            )
            meshes.append(
                {
                    "id": f"sca_{sindex:05d}_{ref.get('ref_id', '')}",
                    "mesh": ref["mesh"],
                    "position": [
                        round((float(pos[0]) - ax) * 0.01, 6),
                        round((float(pos[1]) - ay) * 0.01, 6),
                        round(float(pos[2]) * 0.01, 6),
                    ],
                    "rotation": [round(float(v), 6) for v in euler],
                    "scale": float(ref.get("scale", 1.0)),
                    "building": False,
                    "source_reference": ref.get("ref_id", ""),
                }
            )
            sindex += 1
    return meshes, building_ids


def build_surface_meshes(realization: dict, anchor: tuple[int, int], origin: tuple[float, float]) -> list[dict]:
    """Convert r15 sampled terrain-following circulation requests to scene meshes."""

    ax, ay = anchor[0] * 8192.0, anchor[1] * 8192.0
    ox, oy = origin
    requests = realization.get("terrain_following_requests")
    if not isinstance(requests, list) or len(requests) != 69:
        raise ValueError(
            f"r15 realization must contain exactly 69 terrain-following requests; "
            f"found {len(requests) if isinstance(requests, list) else 'invalid'}"
        )
    meshes: list[dict] = []
    for index, request in enumerate(requests):
        vertices = request.get("terrain_vertices")
        if not isinstance(vertices, list) or len(vertices) < 3:
            raise ValueError(f"circulation request {index} has no sampled terrain polygon")
        scene_vertices = [
            [
                round((float(point[0]) + ox - ax) * 0.01, 6),
                round((float(point[1]) + oy - ay) * 0.01, 6),
                round(float(point[2]) * 0.01 + 0.015, 6),
            ]
            for point in vertices
        ]
        meshes.append(
            {
                "id": f"circulation_{index:03d}_{request.get('realization_id', '')}",
                "vertices": scene_vertices,
                "surface_class": request.get("canonical_surface_class", "settlement_dirt"),
                "raw_vtex": int(assignment["raw_vtex"]),
                "role": request.get("role", "circulation"),
            }
        )
    return meshes


def write_circulation_diagnostic(
    *, realization: dict, seated: dict, layout: dict, output_path: Path
) -> None:
    """Write a compact plan-space diagnostic for VTEX erasure and circulation."""

    points: list[tuple[float, float]] = []

    def collect(value: object) -> None:
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
            points.append((float(value[0]), float(value[1])))
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(realization.get("terrain_following_requests", []))
    collect(realization.get("source_road_erase_requests", []))
    collect(layout.get("roads", []))
    for obj in seated.get("objects", []):
        position = obj.get("world_position_gu")
        if isinstance(position, list) and len(position) >= 2:
            points.append((float(position[0]), float(position[1])))
    if not points:
        raise ValueError("circulation diagnostic has no plan-space points")
    min_x, max_x = min(x for x, _ in points), max(x for x, _ in points)
    min_y, max_y = min(y for _, y in points), max(y for _, y in points)
    margin = 512.0
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), (32, 38, 31))
    draw = ImageDraw.Draw(image)

    def xy(point: list | tuple) -> tuple[int, int]:
        x = int((float(point[0]) - min_x + margin) / (max_x - min_x + 2 * margin) * (width - 1))
        y = int((max_y - float(point[1]) + margin) / (max_y - min_y + 2 * margin) * (height - 1))
        return x, y

    def polygons(value: object) -> list[list[list[float]]]:
        if not isinstance(value, list):
            return []
        if value and all(
            isinstance(item, list) and len(item) >= 2 and isinstance(item[0], (int, float))
            for item in value
        ):
            return [value]  # type: ignore[list-item]
        result: list[list[list[float]]] = []
        for item in value:
            result.extend(polygons(item))
        return result

    for request in realization.get("source_road_erase_requests", []):
        for polygon in polygons(request.get("polygon", [])):
            draw.polygon([xy(point) for point in polygon], fill=(72, 88, 68), outline=(104, 125, 91))
    for request in realization.get("terrain_following_requests", []):
        for polygon in polygons(request.get("polygon", [])):
            draw.polygon([xy(point) for point in polygon], fill=(142, 101, 55), outline=(205, 157, 92))
    for road in layout.get("roads", []):
        line = road.get("polyline", [])
        if len(line) >= 2:
            draw.line([xy(point) for point in line], fill=(224, 190, 118), width=5)
    for obj in seated.get("objects", []):
        position = obj.get("world_position_gu")
        if isinstance(position, list) and len(position) >= 2:
            x, y = xy(position)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(190, 72, 56))
    legend = "r15 VTEX/circulation diagnostic  |  green=source-road erase  brown=alley/apron geometry  gold=authored roads  red=buildings"
    draw.rectangle((12, 12, width - 12, 36), fill=(15, 18, 15))
    draw.text((20, 18), legend, fill=(238, 238, 220))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def make_scene(
    *,
    meshes: list,
    terrain_plugin: Path,
    terrain_override: Path,
    cells: list,
    anchor: tuple[int, int],
    camera: dict,
    config: Path,
    base: Path,
    survey: Path,
    scatter_path: Path,
    surface_meshes: list[dict],
) -> dict:
    return {
        "scene_name": "Falkreath_settlement_wilderness_city_SE",
        "procgen_config": str(config.resolve()),
        "import": {
            "scale_correction": 0.01,
            "normalize_to_position": False,
            "use_existing_materials": True,
            "ignore_collision_nodes": True,
            "ignore_animations": True,
            "reuse_meshes": True,
        },
        "lighting": {
            "sun_energy": 2.2,
            "sun_angle_degrees": 30.0,
            "world_strength": 1.4,
            "fill_energy": 2600.0,
            "fill_size_factor": 2.0,
            "exposure": 0.5,
            "view_look": "AgX - Medium High Contrast",
        },
        "camera": {
            "mode": camera.get("mode", "ORTHO"),
            "view": "oblique",
            "view_direction": camera["view_direction"],
            "resolution": [1600, 1000],
            "margin": camera.get("margin", 1.0),
            "subject": camera["subject"],
        },
        "terrain": {
            "enabled": True,
            "plugin": str(terrain_plugin.resolve()),
            "override_plugin": str(terrain_override.resolve()),
            "texture_plugin": str(terrain_override.resolve()),
            "texture_masters": [],
            "anchor_grid": list(anchor),
            "cells": [list(cell) for cell in cells],
            "decimate": 2,
            "max_seconds": 90.0,
        },
        "meshes": meshes,
        "surface_meshes": surface_meshes,
        "source": {
            "seated_objects": str(base / "stamp_objects_seated.json"),
            "scatter": str(scatter_path),
            "survey": str(survey.resolve()),
            "frame": "world GU; anchor = min terrain cell",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--survey", type=Path, default=DEFAULT_SURVEY)
    parser.add_argument("--realization", type=Path, default=DEFAULT_REALIZATION)
    parser.add_argument("--layout", type=Path, default=DEFAULT_BASE.parent / "13_city_layout/city_layout.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scatter", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, default=DEFAULT_TERRAIN)
    parser.add_argument("--terrain-override", type=Path, default=DEFAULT_TERRAIN_OVERRIDE)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = args.base.resolve()
    terrain = args.terrain.resolve()
    terrain_override = args.terrain_override.resolve()
    realization_path = args.realization.resolve()
    if "r15_variety_retry3" not in str(base).casefold() or "16_terrain_smooth" not in str(base).casefold():
        print("FAILURE: renderer base must be the authoritative r15 16_terrain_smooth artifact", file=sys.stderr)
        return 1
    if terrain.name.casefold() != "tamriel.esm":
        print("FAILURE: renderer terrain must be tamriel.esm base terrain", file=sys.stderr)
        return 1
    if not terrain.is_file() or not terrain_override.is_file():
        print(f"FAILURE: renderer terrain is missing: {terrain}", file=sys.stderr)
        return 1
    if "r15_variety_retry3" not in str(terrain_override).casefold() or terrain_override.suffix.casefold() != ".esp":
        print("FAILURE: renderer terrain override must be the authoritative r15 ESP", file=sys.stderr)
        return 1
    survey = json.loads(args.survey.resolve().read_text(encoding="utf-8"))
    seated = json.loads((base / "stamp_objects_seated.json").read_text(encoding="utf-8"))
    scatter = json.loads(args.scatter.resolve().read_text(encoding="utf-8"))
    realization = json.loads(realization_path.read_text(encoding="utf-8"))
    layout = json.loads(args.layout.resolve().read_text(encoding="utf-8"))
    if any("hut" in str(obj.get("stamp_id", "")).casefold() for obj in seated.get("objects", [])):
        print("FAILURE: r15 seated objects contain a hut stamp", file=sys.stderr)
        return 1
    origin = tuple(float(v) for v in survey["frame"]["origin_gu"])

    cells = sorted(
        {(gx, gy) for gx in range(SCATTER_BOUNDS[0], SCATTER_BOUNDS[1] + 1) for gy in range(SCATTER_BOUNDS[2], SCATTER_BOUNDS[3] + 1)}
    )
    anchor = min(cells)
    meshes, building_ids = build_meshes(seated, scatter, anchor, origin)
    # These sampled polygons are diagnostic geometry, not render content.  The
    # actual VTEX-painted LAND already shows the roads; overlaying them creates
    # confusing tan floating lines over the settlement.
    surface_meshes: list[dict] = []
    if not meshes:
        print("FAILURE: render-scene produced no meshes", file=sys.stderr)
        return 1

    # Camera ortho_scale = max(span_y*1.8, span_x/aspect*1.25)*margin.  The
    # default margin 1.18 leaves the content only filling the middle, so we use
    # a tight margin (~0.55) to zoom until the framed area fills the height.
    cameras = [
        {
            "name": "overview",
            "margin": 0.56,
            "subject": {"include_terrain": True, "padding": 0.0},
        },
        {
            "name": "city",
            "margin": 0.56,
            "subject": {"ids": building_ids, "include_terrain": False, "padding": 20.0},
        },
    ]

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_circulation_diagnostic(
        realization=realization,
        seated=seated,
        layout=layout,
        output_path=out / "circulation_diagnostic.png",
    )
    for camera in cameras:
        camera["view_direction"] = list(SE_DIRECTION)
        camera["mode"] = "ORTHO"
        scene = make_scene(
            meshes=meshes,
            terrain_plugin=terrain,
            terrain_override=terrain_override,
            cells=cells,
            anchor=anchor,
            camera=camera,
            config=args.config,
            base=base,
            survey=args.survey,
            scatter_path=args.scatter,
            surface_meshes=surface_meshes,
        )
        scene_path = out / f"city_se_{camera['name']}.json"
        scene_path.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        png = out / f"city_se_{camera['name']}.png"
        command = [
            str(args.blender.resolve()),
            "-b",
            "--python",
            str(ROOT / "tools/blender_flat_render.py"),
            "--",
            str(scene_path),
            str(png),
            "--statics",
            "all",
        ]
        result = subprocess.run(command, cwd=str(ROOT), check=False)
        if result.returncode != 0:
            print(f"FAILURE: Blender render {camera['name']} exited {result.returncode}", file=sys.stderr)
            return result.returncode
        if not png.is_file():
            print(f"FAILURE: Blender render {camera['name']} produced no PNG", file=sys.stderr)
            return 1
        print(f"rendered {png}")

    print(json.dumps({"scenes": len(cameras), "meshes": len(meshes), "output_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
