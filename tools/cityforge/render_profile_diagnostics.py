#!/usr/bin/env python3
"""Phase 3a/3b diagnostic renders for configured profile subsets.

Produces, per selected shell, a top-down PIL map with the evaluated ground
polygon (Phase 2 z-slice), colored facade segments with stable IDs and outward
normal arrows, usable insets, and witness occupancy points; and per small-set
attachment a Blender front/back orthographic pair along the measured mount
axis, stitched with labels.

Usage::

    python tools/cityforge/render_profile_diagnostics.py --config configs/kits/xfa_sky_nord_house/phase03_config.json
    python tools/cityforge/render_profile_diagnostics.py --config configs/kits/xfa_sky_nord_house/phase03b_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path(__file__).resolve().parents[2]

FRONT_BACK_SCRIPT = WORKSPACE / "tools" / "cityforge" / "blender_front_back.py"
OVERLAY_SCRIPT = WORKSPACE / "tools" / "cityforge" / "blender_facade_overlay.py"

FACADE_COLORS = [
    (230, 85, 85), (85, 170, 230), (95, 200, 120), (235, 190, 70),
    (190, 120, 230), (90, 210, 210), (240, 140, 60), (150, 200, 90),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_model_key(value: str) -> str:
    return str(value).replace("/", "\\").casefold()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _plane_xy_at_z(n: list[float], offset: float, z: float) -> tuple[float, float]:
    nh2 = n[0] * n[0] + n[1] * n[1]
    if nh2 <= 1e-12:
        return (0.0, 0.0)
    horizontal = (offset - n[2] * z) / nh2
    return (n[0] * horizontal, n[1] * horizontal)


def render_shell_map(shell_key: str, facade_profile: dict, ground_polygon: list,
                     out_path: Path, draw_ground_context: bool = True) -> None:
    facades = facade_profile["facades"]
    points: list[tuple[float, float]] = [tuple(p) for p in ground_polygon]
    segments = []
    for facade in facades:
        n = facade["outward_frame"]["n"]
        u = facade["outward_frame"]["u"]
        offset = facade["outward_frame"]["plane_offset_gu"]
        u0, u1 = facade["u_span_gu"]
        z0, z1 = facade["z_interval_gu"]
        plane_x, plane_y = _plane_xy_at_z(n, offset, (z0 + z1) / 2.0)
        a = (plane_x + u0 * u[0], plane_y + u0 * u[1])
        b = (plane_x + u1 * u[0], plane_y + u1 * u[1])
        segments.append((a, b, n, facade["facade_id"], z0, z1, facade))
        points.extend([a, b])
        for occ in facade["occupied_regions"]:
            plane_x, plane_y = _plane_xy_at_z(n, offset, occ["z_gu"])
            p = (plane_x + occ["u_gu"] * u[0], plane_y + occ["u_gu"] * u[1])
            points.append(p)
    if not points:
        points = [(0.0, 0.0), (1.0, 1.0)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    margin = 120
    size = 1600
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    scale = (size - 2 * margin) / span

    def to_px(p: tuple[float, float]) -> tuple[float, float]:
        return ((p[0] - min(xs)) * scale + margin, size - ((p[1] - min(ys)) * scale + margin))

    image = Image.new("RGB", (size, size + 120), (18, 18, 22))
    draw = ImageDraw.Draw(image)
    font = _font(26)
    small = _font(20)
    if draw_ground_context and len(ground_polygon) >= 3:
        draw.polygon([to_px(p) for p in ground_polygon], outline=(120, 120, 130), width=2)
    for index, (a, b, n, facade_id, z0, z1, facade) in enumerate(segments):
        color = FACADE_COLORS[index % len(FACADE_COLORS)]
        pa, pb = to_px(a), to_px(b)
        draw.line([pa, pb], fill=color, width=10)
        mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        arrow_len = 90
        tip = (mid[0] + n[0] * arrow_len, mid[1] - n[1] * arrow_len)
        draw.line([mid, tip], fill=color, width=4)
        head = 14
        ang = math.atan2(tip[1] - mid[1], tip[0] - mid[0])
        for delta in (2.6, -2.6):
            draw.line([tip, (tip[0] + head * math.cos(ang + delta), tip[1] + head * math.sin(ang + delta))],
                      fill=color, width=4)
        draw.text((mid[0] + 12, mid[1] + 12), f"{facade_id}", fill=color, font=font)
        draw.text((mid[0] + 12, mid[1] + 44), f"z {z0:.0f}..{z1:.0f}", fill=color, font=small)
        for occ in facade["occupied_regions"]:
            u = facade["outward_frame"]["u"]
            off = facade["outward_frame"]["plane_offset_gu"]
            plane_x, plane_y = _plane_xy_at_z(n, off, occ["z_gu"])
            p = (plane_x + occ["u_gu"] * u[0], plane_y + occ["u_gu"] * u[1])
            px = to_px(p)
            draw.ellipse([px[0] - 7, px[1] - 7, px[0] + 7, px[1] + 7], outline=(255, 255, 255), width=3)
    draw.text((20, size + 14), f"{facade_profile['model_key']}", fill=(230, 230, 230), font=font)
    draw.text((20, size + 50),
              f"facades={facade_profile['facade_count']}  wall band "
              f"{facade_profile['wall_band_gu'][0]:.0f}..{facade_profile['wall_band_gu'][1]:.0f} GU  "
              "white dots = witness attachments",
              fill=(170, 170, 170), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def run_front_back(mesh: str, axis: int, out_dir: Path, blender: str) -> tuple[Path, Path]:
    blender_exe = shutil.which(blender)
    if blender_exe is None:
        print(f"FAILURE: blender not found on PATH ({blender!r})", file=sys.stderr)
        raise SystemExit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    plus = out_dir / "plus_axis.png"
    minus = out_dir / "minus_axis.png"
    with tempfile.TemporaryDirectory(prefix="front_back_") as tmp:
        job_path = Path(tmp) / "job.json"
        job_path.write_text(json.dumps({
            "mesh": mesh.replace("\\", "/"),
            "normal_axis": axis,
            "out_plus": str(plus),
            "out_minus": str(minus),
            "resolution": 1024,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [blender_exe, "-b", "--python", str(FRONT_BACK_SCRIPT), "--", str(job_path)]
        print("running:", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        if completed.returncode != 0 or not plus.exists() or not minus.exists():
            print(f"FAILURE: front/back render failed for {mesh}", file=sys.stderr)
            raise SystemExit(1)
    return plus, minus


def stitch_pair(plus: Path, minus: Path, label: str, front_sign: str, out_path: Path) -> None:
    """Put the measured front on the left, regardless of signed axis."""
    front_path, back_path = (plus, minus) if front_sign == "+" else (minus, plus)
    left_rgba = Image.open(front_path).convert("RGBA")
    right_rgba = Image.open(back_path).convert("RGBA")
    background = Image.new("RGBA", left_rgba.size, (82, 82, 88, 255))
    left = Image.alpha_composite(background, left_rgba).convert("RGB")
    background = Image.new("RGBA", right_rgba.size, (82, 82, 88, 255))
    right = Image.alpha_composite(background, right_rgba).convert("RGB")
    w, h = left.size
    banner = 80
    image = Image.new("RGB", (w * 2, h + banner), (18, 18, 22))
    image.paste(left, (0, banner))
    image.paste(right, (w, banner))
    draw = ImageDraw.Draw(image)
    font = _font(30)
    draw.text((20, 20), f"{label}  LEFT = FRONT (local {front_sign}n)   RIGHT = BACK (local {'-' if front_sign == '+' else '+'}n)", fill=(230, 230, 230), font=font)
    image.save(out_path)


def run_overlay(profile: dict, out_dir: Path, blender: str) -> Path:
    blender_exe = shutil.which(blender)
    if blender_exe is None:
        print(f"FAILURE: blender not found on PATH ({blender!r})", file=sys.stderr)
        raise SystemExit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = profile["model_key"].casefold()
    prefix = out_dir / f"{Path(key).stem}_overlay"
    job_facades = []
    for index, facade in enumerate(profile["facades"]):
        color = FACADE_COLORS[index % len(FACADE_COLORS)]
        job_facades.append({
            "facade_id": facade["facade_id"],
            "color": [c / 255.0 for c in color],
            "n": facade["outward_frame"]["n"],
            "u": facade["outward_frame"]["u"],
            "offset_gu": facade["outward_frame"]["plane_offset_gu"],
            "u_span_gu": facade["u_span_gu"],
            "z_interval_gu": facade["z_interval_gu"],
            "polygon_uz": facade["polygon_uz"],
        })
    with tempfile.TemporaryDirectory(prefix="overlay_") as tmp:
        job_path = Path(tmp) / "job.json"
        job_path.write_text(json.dumps({
            "mesh": profile["model_key"].replace("\\", "/"),
            "facades": job_facades,
            "out_prefix": str(prefix),
            "resolution": 1024,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [blender_exe, "-b", "--python", str(OVERLAY_SCRIPT), "--", str(job_path)]
        print("running:", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        expected = [out_dir / f"{Path(key).stem}_overlay_{view}.png" for view in ("north", "east", "south", "west")]
        if completed.returncode != 0 or not all(p.exists() for p in expected):
            print(f"FAILURE: facade overlay render failed for {key}", file=sys.stderr)
            raise SystemExit(1)
    stitched = diag_stitch_overlay(expected, profile, out_dir / f"{Path(key).stem}_overlay_sheet.png")
    return stitched


def diag_stitch_overlay(views: list[Path], profile: dict, out_path: Path) -> Path:
    images = [Image.open(p).convert("RGB") for p in views]
    w, h = images[0].size
    legend_step_x = 118
    legend_step_y = 32
    legend_columns = max(1, (w * 2 - 40) // legend_step_x)
    legend_rows = max(1, (len(profile["facades"]) + legend_columns - 1) // legend_columns)
    banner = max(160, 64 + legend_rows * legend_step_y + 8)
    image = Image.new("RGB", (w * 2, h * 2 + banner), (18, 18, 22))
    labels = ["north", "east", "south", "west"]
    for index, (img, label) in enumerate(zip(images, labels)):
        x = (index % 2) * w
        y = banner + (index // 2) * h
        image.paste(img, (x, y))
    draw = ImageDraw.Draw(image)
    font = _font(30)
    small = _font(22)
    draw.text((20, 14), f"{profile['model_key']}  facade overlay (N E / S W)", fill=(230, 230, 230), font=font)
    for index, label in enumerate(labels):
        x = (index % 2) * w + 16
        y = banner + (index // 2) * h + 16
        ImageDraw.Draw(image).text((x, y), label, fill=(255, 255, 255), font=small)
    legend_x = 20
    legend_y = 64
    for index, facade in enumerate(profile["facades"]):
        color = FACADE_COLORS[index % len(FACADE_COLORS)]
        column = index % legend_columns
        row = index // legend_columns
        item_x = legend_x + column * legend_step_x
        item_y = legend_y + row * legend_step_y
        draw.rectangle([item_x, item_y, item_x + 22, item_y + 22], fill=color)
        draw.text((item_x + 30, item_y - 2), facade["facade_id"], fill=(220, 220, 220), font=small)
    image.save(out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Phase 3a small-set diagnostics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()

    config = read_json(args.config)
    outputs = config["outputs"]
    diag_dir = WORKSPACE / outputs["diagnostics"]
    diag_dir.mkdir(parents=True, exist_ok=True)

    facades_doc = read_json(WORKSPACE / outputs["facades"])
    mounts_doc = read_json(WORKSPACE / outputs["mounts"])
    phase02_profiles = read_json(WORKSPACE / config["phase02"]["profiles"])
    ground_by_key = {normalize_model_key(row["model_key"]): row for row in phase02_profiles["meshes"]}
    diagnostic_config = config.get("diagnostics")
    allowed_shells = None
    allowed_attachments = None
    draw_ground_context = True
    if diagnostic_config is not None:
        allowed_shells = {normalize_model_key(key) for key in diagnostic_config.get("shells", [])}
        allowed_attachments = {normalize_model_key(key) for key in diagnostic_config.get("attachments", [])}
        draw_ground_context = bool(diagnostic_config.get("draw_ground_context", True))

    produced = []
    for profile in facades_doc["profiles"]:
        key = normalize_model_key(profile["model_key"])
        if allowed_shells is not None and key not in allowed_shells:
            continue
        ground = ground_by_key[key]["ground_polygon_xy"]
        out = diag_dir / f"{Path(key).stem}_facades_topdown.png"
        render_shell_map(key, profile, ground, out, draw_ground_context)
        produced.append(out)
        print(f"[diag] wrote {out}")
        overlay = run_overlay(profile, diag_dir / f"{Path(key).stem}_overlay", args.blender)
        produced.append(overlay)
        print(f"[diag] wrote {overlay}")

    for profile in mounts_doc["profiles"]:
        key = normalize_model_key(profile["model_key"])
        if allowed_attachments is not None and key not in allowed_attachments:
            continue
        axis = "xyz".index(profile["mount_frame"]["normal_axis"])
        front_sign = profile["front_back_evidence"]["front_axis_sign"]
        work = diag_dir / f"{Path(key).stem}_front_back"
        plus, minus = run_front_back(profile["model_key"], axis, work, args.blender)
        out = diag_dir / f"{Path(key).stem}_front_back_pair.png"
        stitch_pair(plus, minus, profile["model_key"], front_sign, out)
        produced.append(out)
        print(f"[diag] wrote {out}")

    if allowed_shells is not None:
        produced_shells = {normalize_model_key(profile["model_key"]) for profile in facades_doc["profiles"]}
        missing = sorted(allowed_shells - produced_shells)
        if missing:
            print(f"FAILURE: configured diagnostic shells are absent: {missing}", file=sys.stderr)
            return 1
    if allowed_attachments is not None:
        produced_attachments = {normalize_model_key(profile["model_key"]) for profile in mounts_doc["profiles"]}
        missing = sorted(allowed_attachments - produced_attachments)
        if missing:
            print(f"FAILURE: configured diagnostic attachments are absent: {missing}", file=sys.stderr)
            return 1

    print(json.dumps({"diagnostics": [str(p) for p in produced]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
