#!/usr/bin/env python3
"""Dispatch native-resolution Phase 4 roof overlay diagnostics."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate, unary_union

WORKSPACE = Path(__file__).resolve().parents[2]
WORKER = WORKSPACE / "tools" / "cityforge" / "blender_roof_overlay.py"
sys.path.insert(0, str(WORKSPACE / "src"))

from procgen.engine_transform import blender_xyz_euler_for_tes3_rotation, matrix_to_tes3_euler, tes3_euler_to_matrix  # noqa: E402
VIEWS = ("north", "east", "south", "west", "top_down", "isometric")
PALETTE = (
    [0.90, 0.20, 0.12],
    [0.15, 0.55, 0.95],
    [0.20, 0.80, 0.35],
    [0.95, 0.65, 0.10],
    [0.70, 0.25, 0.90],
    [0.10, 0.80, 0.80],
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def key(value: str) -> str:
    return str(value).replace("/", "\\").casefold()


def _relation_key(stamp_id: str, dormer_source_id: str) -> str:
    return f"{stamp_id}/{dormer_source_id}"


def _region(pieces):
    polygons = []
    for piece in pieces:
        if len(piece.get("outer", [])) >= 3:
            polygons.append(Polygon(piece["outer"], piece.get("holes", [])))
    return polygons


def _triangles_for_region(pieces):
    triangles = []
    for polygon in _region(pieces):
        for triangle in triangulate(polygon):
            if polygon.covers(triangle):
                coordinates = list(triangle.exterior.coords)[:-1]
                if len(coordinates) == 3:
                    triangles.append([[round(float(x), 6), round(float(y), 6)] for x, y in coordinates])
    return triangles


def _frame_job_data(patch: dict, frame_arrow_length_gu: float, frame_normal_length_gu: float) -> dict:
    region = unary_union(_region(patch.get("usable_region_uv", [])))
    if region.is_empty:
        raise RuntimeError(f"eligible patch has no usable diagnostic region: {patch['patch_id']}")
    anchor = region.representative_point()
    anchor_uv = (float(anchor.x), float(anchor.y))

    def safe_length(dx: float, dy: float) -> float:
        if region.covers(Point(anchor.x + dx * frame_arrow_length_gu, anchor.y + dy * frame_arrow_length_gu)):
            return frame_arrow_length_gu
        low, high = 0.0, frame_arrow_length_gu
        for _ in range(32):
            midpoint = (low + high) / 2.0
            point = Point(anchor.x + dx * midpoint, anchor.y + dy * midpoint)
            if region.covers(point):
                low = midpoint
            else:
                high = midpoint
        return low * 0.85

    return {
        "frame_anchor_uv": [round(anchor_uv[0], 6), round(anchor_uv[1], 6)],
        "frame_u_length_gu": round(safe_length(1.0, 0.0), 6),
        "frame_v_length_gu": round(safe_length(0.0, 1.0), 6),
        "frame_normal_length_gu": float(frame_normal_length_gu),
    }


def _patch_job(
    patch: dict,
    color: list[float],
    frame_arrow_length_gu: float,
    frame_normal_length_gu: float,
) -> dict:
    result = dict(patch)
    result["color"] = color
    result["fill_triangles_uv"] = _triangles_for_region(patch.get("polygon_pieces_uv", []))
    result["frame_arrow_length_gu"] = frame_arrow_length_gu
    result.update(_frame_job_data(patch, frame_arrow_length_gu, frame_normal_length_gu))
    return result


def _canonicalize_png(path: Path) -> None:
    """Remove Blender's per-render metadata while preserving decoded pixels."""
    with Image.open(path) as image:
        mode = "RGBA" if "A" in image.getbands() else "RGB"
        pixels = image.convert(mode).tobytes()
        canonical = Image.frombytes(mode, image.size, pixels)
    canonical.save(path, format="PNG", optimize=False, compress_level=9)


def _run_worker(executable: str, job: dict, prefix: Path) -> list[Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roof_diag_") as temporary:
        job_path = Path(temporary) / "job.json"
        job_path.write_text(json.dumps(job, sort_keys=True) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [executable, "-b", "--python", str(WORKER), "--", str(job_path)],
            cwd=WORKSPACE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"roof diagnostic worker failed for {prefix}")
    expected = [prefix.parent / f"{prefix.name}_{view}.png" for view in VIEWS]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"roof diagnostic worker omitted files: {missing}")
    for path in expected:
        _canonicalize_png(path)
    return expected


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _stitch_pair(left_path: Path, right_path: Path, label: str, output: Path) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    if left.size != right.size:
        raise RuntimeError(f"relation panel sizes differ: {left_path} {right_path}")
    banner = 72
    image = Image.new("RGB", (left.width * 2, left.height + banner), (18, 18, 22))
    image.paste(left, (0, banner))
    image.paste(right, (left.width, banner))
    draw = ImageDraw.Draw(image)
    draw.text((18, 18), f"{label}  LEFT = SOURCE   RIGHT = ROOF-FRAME RECONSTRUCTION", fill=(235, 235, 235), font=_font(26))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def _stitch_shell_sheet(paths: list[Path], label: str, output: Path) -> None:
    if len(paths) != len(VIEWS):
        raise RuntimeError(f"shell diagnostic view count is {len(paths)}, expected {len(VIEWS)}")
    panels = [Image.open(path).convert("RGB") for path in paths]
    size = panels[0].size
    if any(panel.size != size for panel in panels):
        raise RuntimeError(f"shell diagnostic panel sizes differ for {label}")
    banner = 72
    image = Image.new("RGB", (size[0] * 3, size[1] * 2 + banner), (18, 18, 22))
    draw = ImageDraw.Draw(image)
    draw.text((18, 18), f"{label}  ROOF PATCH DIAGNOSTICS", fill=(235, 235, 235), font=_font(26))
    for index, (view, panel) in enumerate(zip(VIEWS, panels)):
        x = (index % 3) * size[0]
        y = banner + (index // 3) * size[1]
        image.paste(panel, (x, y))
        draw.rectangle((x + 12, y + 12, x + 170, y + 48), fill=(18, 18, 22))
        draw.text((x + 20, y + 18), view, fill=(235, 235, 235), font=_font(24))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def _blender_rotation(matrix) -> list[float]:
    return list(blender_xyz_euler_for_tes3_rotation(matrix_to_tes3_euler(matrix)))


def _relation_source_context(config: dict, relation: dict):
    for site in config["sites"]:
        if str(site["site_id"]) != str(relation["site_id"]):
            continue
        source = read_json(WORKSPACE / str(site["stamp_library"]))
        stamp = next(row for row in source["stamps"] if row["stamp_id"] == relation["stamp_id"])
        members = {str(member["source_id"]): member for member in stamp["members"]}
        return stamp, members[str(relation["shell_member_source_id"])], members[str(relation["dormer_member_source_id"])]
    raise RuntimeError(f"relation site missing: {relation['site_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Phase 4 roof and dormer diagnostics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    executable = shutil.which(args.blender)
    if executable is None:
        print(f"FAILURE: blender not found on PATH ({args.blender!r})", file=sys.stderr)
        return 1
    config = read_json(args.config)
    roofs = read_json(WORKSPACE / config["outputs"]["roofs"])
    relations_doc = read_json(WORKSPACE / config["outputs"]["dormers"])
    output = WORKSPACE / config["outputs"]["diagnostics"]
    output.mkdir(parents=True, exist_ok=True)
    resolution = int(config["diagnostics"]["resolution"])
    frame_arrow_length = float(config["measurement"]["frame_arrow_length_gu"])
    frame_normal_length = float(config["measurement"]["frame_normal_length_gu"])
    selected_shells = {key(value) for value in config["diagnostics"].get("shells", [])}
    selected_relation_keys = set(config["diagnostics"].get("dormer_relations", []))
    produced: list[str] = []

    for profile in roofs["profiles"]:
        if selected_shells and key(profile["model_key"]) not in selected_shells:
            continue
        prefix = output / f"{Path(profile['model_key']).stem}_roof"
        patches = [
            _patch_job(patch, PALETTE[index % len(PALETTE)], frame_arrow_length, frame_normal_length)
            for index, patch in enumerate(profile["patches"])
            if patch.get("status") == "eligible"
        ]
        paths = _run_worker(
            executable,
            {
                "placements": [{"mesh": profile["model_key"], "position_gu": [0.0, 0.0, 0.0], "rotation_blender": [0.0, 0.0, 0.0], "scale": 1.0}],
                "patches": patches,
                "out_prefix": str(prefix),
                "resolution": resolution,
            },
            prefix,
        )
        produced.extend(str(path.relative_to(WORKSPACE)).replace("\\", "/") for path in paths)
        sheet = output / f"{prefix.name}_sheet.png"
        _stitch_shell_sheet(paths, profile["model_key"], sheet)
        produced.append(str(sheet.relative_to(WORKSPACE)).replace("\\", "/"))
    if selected_shells:
        produced_shells = {key(profile["model_key"]) for profile in roofs["profiles"] if key(profile["model_key"]) in selected_shells}
        if produced_shells != selected_shells:
            raise RuntimeError(f"configured roof shell diagnostics missing: {sorted(selected_shells - produced_shells)}")

    relation_by_key = {
        _relation_key(row["source_stamp_id"], row["dormer_member_source_id"]): row
        for row in relations_doc["relations"]
    }
    profile_by_key = {key(profile["model_key"]): profile for profile in roofs["profiles"]}
    for relation_config in config["selection"]["dormer_relations"]:
        relation_id = _relation_key(relation_config["stamp_id"], relation_config["dormer_member_source_id"])
        if selected_relation_keys and relation_id not in selected_relation_keys:
            continue
        relation = relation_by_key.get(relation_id)
        if relation is None:
            raise RuntimeError(f"configured dormer relation missing: {relation_id}")
        _stamp, shell, dormer = _relation_source_context(config, relation_config)
        profile = profile_by_key[key(shell["model_key"])]
        patch = next(patch for patch in profile["patches"] if patch["patch_id"] == relation["roof_patch_id"])
        patch_job = _patch_job(patch, PALETTE[0], frame_arrow_length, frame_normal_length)
        patch_frame = np.column_stack((np.asarray(patch["u"], float), np.asarray(patch["v"], float), np.asarray(patch["n"], float)))
        source_rotation = np.asarray(tes3_euler_to_matrix(dormer["rotation"]), float)
        shell_rotation = np.asarray(tes3_euler_to_matrix(shell["rotation"]), float)
        source_relative_rotation = shell_rotation.T @ source_rotation
        stored_relative_rotation = patch_frame @ np.asarray(relation["dormer_orientation_relative_to_roof_frame"], float)
        source_position = [float(value) for value in relation["dormer_origin_shell_frame_gu"]]
        coordinates = relation["roof_frame_coordinates_gu"]
        reconstructed_position = (
            np.asarray(patch["origin_gu"], float)
            + np.asarray(patch["u"], float) * float(coordinates["u_along_eave"])
            + np.asarray(patch["v"], float) * float(coordinates["v_up_slope"])
            + np.asarray(patch["n"], float) * float(coordinates["n_sink"])
        )
        slug = relation_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
        common = {"resolution": resolution, "patches": [patch_job], "marker_gu": source_position}
        source_prefix = output / f"{slug}_source"
        recon_prefix = output / f"{slug}_reconstructed"
        source_paths = _run_worker(
            executable,
            {**common, "placements": [
                {"mesh": shell["model_key"], "position_gu": [0.0, 0.0, 0.0], "rotation_blender": [0.0, 0.0, 0.0], "scale": 1.0},
                {"mesh": dormer["model_key"], "position_gu": source_position, "rotation_blender": _blender_rotation(source_relative_rotation), "scale": float(relation["dormer_authored_scale"])},
            ], "out_prefix": str(source_prefix)},
            source_prefix,
        )
        recon_paths = _run_worker(
            executable,
            {**common, "marker_gu": reconstructed_position.tolist(), "placements": [
                {"mesh": shell["model_key"], "position_gu": [0.0, 0.0, 0.0], "rotation_blender": [0.0, 0.0, 0.0], "scale": 1.0},
                {"mesh": dormer["model_key"], "position_gu": reconstructed_position.tolist(), "rotation_blender": _blender_rotation(stored_relative_rotation), "scale": float(relation["dormer_authored_scale"])},
            ], "out_prefix": str(recon_prefix)},
            recon_prefix,
        )
        produced.extend(str(path.relative_to(WORKSPACE)).replace("\\", "/") for path in source_paths + recon_paths)
        for view, source_path, recon_path in zip(VIEWS, source_paths, recon_paths):
            pair = output / f"{slug}_source_vs_reconstructed_{view}.png"
            _stitch_pair(source_path, recon_path, relation_id, pair)
            produced.append(str(pair.relative_to(WORKSPACE)).replace("\\", "/"))

    manifest = {"schema_version": 1, "resolution": resolution, "files": sorted(produced)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
