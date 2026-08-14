"""Render and annotate the four Falkreath D-SITE planner views.

This file is deliberately a thin wrapper around the shared
``tools/blender_flat_render.py`` machinery.  The Blender worker imports its
LAND/material/water helpers directly and builds a terrain-only scene without
calling the shared loader's non-empty-mesh gate.  It therefore renders the
real remap-ESP terrain and resolved image textures, rather than replacing the
site with a 2-D map or fallback material.  The wrapper converts the shared
helper's temporary rectangular water plane into a z<=0-clipped mesh derived
from the actual terrain faces, preventing a perspective water skirt.  Pillow
is used only after Blender finishes to add screen-space planner aids (grid,
rulers, elevation tint, and an exact LAND/VTEX road-mask highlight) to those
real terrain renders.  The road view never reads or draws the rejected
world-map graph: source road tiles and perimeter-confirmed continuation spans
come from ``site_survey.json``/``land_roads.json``.

Pipeline position::

    site_survey.json + survey_fields.npz
        -> Blender terrain/water renders (temporary raw PNGs)
        -> real-texture gate and camera audit
        -> annotated output/cityforge/sites/falkreath_v1/*.png
        -> render_audit.json

The host command is::

    python tools/cityforge/render_site.py

Blender invokes the same file with ``--blender-worker`` internally.  No source
plugin, configured data root, or ``blender_flat_render.py`` file is edited.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_SURVEY = WORKSPACE / "output/cityforge/sites/falkreath_v1/site_survey.json"
DEFAULT_OUTPUT = WORKSPACE / "output/cityforge/sites/falkreath_v1"
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe")
RENDER_SIZE = 4096
RENDER_MARGIN_GU = 1024.0
SCENE_UNITS_PER_GU = 0.01


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def _survey_paths(survey: Mapping[str, Any]) -> tuple[Path, Path]:
    input_land = str(survey["inputs"]["land_source"]).split(" (sha256:", 1)[0]
    input_land_path = _workspace_path(input_land)
    if not input_land_path.is_file():
        raise FileNotFoundError(f"survey LAND source is missing: {input_land_path}")
    return input_land_path, WORKSPACE / "configs/procgen.json"


def _real_texture_gate(survey: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve every LTEX/base image used by the target LAND before Blender."""

    src = WORKSPACE / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from procgen.espland import BASE_LAND_TEXTURE_PATH, load_land, load_ltex
    from procgen.meshcheck import AssetResolver, configured_data_roots

    land_path, config_path = _survey_paths(survey)
    roots = configured_data_roots(config_path)
    resolver = AssetResolver(roots=roots)
    records = load_land(land_path, max_seconds=120.0)
    used_raw: set[int] = set()
    for record in records.values():
        if record.texture_indices is not None:
            used_raw.update(int(value) for value in record.texture_indices)
    ltex = load_ltex(land_path, max_seconds=120.0)
    resolutions: dict[str, str] = {}
    missing: list[str] = []
    for raw in sorted(used_raw):
        if raw == 0:
            declared = BASE_LAND_TEXTURE_PATH
            label = "raw_vtex_0:_land_default.dds"
        else:
            index = raw - 1
            entry = ltex.get(index)
            if entry is None:
                missing.append(f"raw_vtex_{raw}:missing LTEX index {index}")
                continue
            declared = entry.file_name
            label = f"raw_vtex_{raw}:ltex_{index}:{entry.record_id}:{declared}"
        resolved = resolver.resolve(declared, "texture")
        if resolved is None or not resolved.is_file():
            missing.append(label)
        else:
            resolutions[label] = str(resolved)
    if missing:
        raise RuntimeError("real terrain texture gate failed before Blender: " + "; ".join(missing))
    if not resolutions:
        raise RuntimeError("real terrain texture gate found no used texture images")
    return {
        "passed": True,
        "used_raw_vtex_values": sorted(used_raw),
        "resolved_count": len(resolutions),
        "missing_count": 0,
        "resolutions": resolutions,
        "configured_roots": [str(root) for root in roots],
    }


def _frame_mapping(survey: Mapping[str, Any], resolution: int) -> dict[str, Any]:
    bounds = survey["target_cells"]
    width_gu = (int(bounds["max_x"]) - int(bounds["min_x"]) + 1) * 8192.0
    height_gu = (int(bounds["max_y"]) - int(bounds["min_y"]) + 1) * 8192.0
    if width_gu != height_gu:
        raise ValueError("the current planner mapping requires a square target frame")
    total_gu = width_gu + 2.0 * RENDER_MARGIN_GU
    px_per_gu = float(resolution) / total_gu
    margin_px = RENDER_MARGIN_GU * px_per_gu
    return {
        "px_per_gu": px_per_gu,
        "origin_px": [margin_px, float(resolution) - margin_px],
        "y_down_image": True,
        "origin_semantics": "frame SW corner in image pixels",
        "resolution": [resolution, resolution],
        "margin_gu": RENDER_MARGIN_GU,
        "transform": "px_x=origin_px[0]+(gu_x-origin_gu[0])*px_per_gu; px_y=origin_px[1]-(gu_y-origin_gu[1])*px_per_gu",
    }


def _gu_to_px(mapping: Mapping[str, Any], origin_gu: Sequence[float], point_gu: Sequence[float]) -> tuple[float, float]:
    scale = float(mapping["px_per_gu"])
    origin_px = mapping["origin_px"]
    return (
        float(origin_px[0]) + (float(point_gu[0]) - float(origin_gu[0])) * scale,
        float(origin_px[1]) - (float(point_gu[1]) - float(origin_gu[1])) * scale,
    )


def _px_to_gu(mapping: Mapping[str, Any], origin_gu: Sequence[float], point_px: Sequence[float]) -> tuple[float, float]:
    scale = float(mapping["px_per_gu"])
    origin_px = mapping["origin_px"]
    return (
        float(origin_gu[0]) + (float(point_px[0]) - float(origin_px[0])) / scale,
        float(origin_gu[1]) - (float(point_px[1]) - float(origin_px[1])) / scale,
    )


def _mapping_round_trip(survey: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, Any]:
    origin = survey["frame"]["origin_gu"]
    bounds = survey["target_cells"]
    sample = [
        origin,
        [float(origin[0]) + 57344.0, float(origin[1])],
        [float(origin[0]), float(origin[1]) + 57344.0],
        [float(origin[0]) + 28672.0, float(origin[1]) + 28672.0],
        [float(-92 * 8192.0), float(-10 * 8192.0)],
    ]
    rows: list[dict[str, Any]] = []
    maximum = 0.0
    for point in sample:
        px = _gu_to_px(mapping, origin, point)
        rounded_px = [round(px[0]), round(px[1])]
        back = _px_to_gu(mapping, origin, rounded_px)
        error_px = math.hypot((back[0] - point[0]) * float(mapping["px_per_gu"]), (back[1] - point[1]) * float(mapping["px_per_gu"]))
        maximum = max(maximum, error_px)
        rows.append({"gu": [float(point[0]), float(point[1])], "px": list(px), "rounded_px": rounded_px, "round_trip_error_px": error_px})
    return {"samples": rows, "max_error_px": maximum, "under_one_pixel": maximum < 1.0, "target_bounds": bounds}


def _font(size: int):
    from PIL import ImageFont

    candidates = [
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\consola.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_shadowed_text(draw: Any, xy: tuple[float, float], text: str, font: Any, fill: tuple[int, int, int, int] = (245, 245, 230, 255)) -> None:
    x, y = xy
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 220), stroke_width=1, stroke_fill=(0, 0, 0, 220))
    draw.text((x, y), text, font=font, fill=fill, stroke_width=1, stroke_fill=(0, 0, 0, 220))


def _draw_grid_and_rulers(image: Any, survey: Mapping[str, Any], mapping: Mapping[str, Any], *, title: str, dim: bool = False) -> Any:
    from PIL import ImageDraw

    output = image.convert("RGBA")
    overlay = output.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    origin = survey["frame"]["origin_gu"]
    bounds = survey["target_cells"]
    width = int(mapping["resolution"][0])
    height = int(mapping["resolution"][1])
    if dim:
        draw.rectangle((0, 0, width, height), fill=(5, 8, 12, 90))
    # Cell grid is thin and translucent so it remains an aid rather than a
    # solid replacement for the texture below it.
    for ix in range(0, int(bounds["max_x"]) - int(bounds["min_x"]) + 2):
        point = [float(origin[0]) + ix * 8192.0, float(origin[1])]
        x, _ = _gu_to_px(mapping, origin, point)
        draw.line((x, 0, x, height), fill=(238, 229, 185, 135), width=3)
    for iy in range(0, int(bounds["max_y"]) - int(bounds["min_y"]) + 2):
        point = [float(origin[0]), float(origin[1]) + iy * 8192.0]
        _, y = _gu_to_px(mapping, origin, point)
        draw.line((0, y, width, y), fill=(238, 229, 185, 135), width=3)

    tick_color = (244, 238, 206, 220)
    x0 = float(origin[0])
    y0 = float(origin[1])
    total_x = (int(bounds["max_x"]) - int(bounds["min_x"]) + 1) * 8192
    total_y = (int(bounds["max_y"]) - int(bounds["min_y"]) + 1) * 8192
    tick_font = _font(max(16, width // 256))
    for gu_x in range(0, total_x + 1, 1024):
        px, frame_bottom = _gu_to_px(mapping, origin, [x0 + gu_x, y0])
        tick_len = 12 if gu_x % 8192 else 24
        draw.line((px, frame_bottom, px, frame_bottom + tick_len), fill=tick_color, width=2)
        if gu_x % 2048 == 0:
            label = str(int((x0 + gu_x) / 8192))
            box = draw.textbbox((0, 0), label, font=tick_font)
            _draw_shadowed_text(draw, (px - (box[2] - box[0]) / 2, frame_bottom + 28), label, tick_font)
    for gu_y in range(0, total_y + 1, 1024):
        frame_left, py = _gu_to_px(mapping, origin, [x0, y0 + gu_y])
        tick_len = 12 if gu_y % 8192 else 24
        draw.line((frame_left - tick_len, py, frame_left, py), fill=tick_color, width=2)
        if gu_y % 2048 == 0:
            label = str(int((y0 + gu_y) / 8192))
            box = draw.textbbox((0, 0), label, font=tick_font)
            _draw_shadowed_text(draw, (frame_left - box[2] - 34, py - (box[3] - box[1]) / 2), label, tick_font)

    cell_font = _font(max(18, width // 192))
    for cell_y in range(int(bounds["min_y"]), int(bounds["max_y"]) + 1):
        for cell_x in range(int(bounds["min_x"]), int(bounds["max_x"]) + 1):
            px, py = _gu_to_px(
                mapping,
                origin,
                [cell_x * 8192.0 + 4096.0, cell_y * 8192.0 + 4096.0],
            )
            label = f"{cell_x},{cell_y}"
            box = draw.textbbox((0, 0), label, font=cell_font)
            # Labels sit over a translucent dark chip, preserving readability
            # without changing the terrain render itself.
            left = px - (box[2] - box[0]) / 2 - 6
            top = py - (box[3] - box[1]) / 2 - 4
            draw.rounded_rectangle(
                (left, top, left + (box[2] - box[0]) + 12, top + (box[3] - box[1]) + 8),
                radius=5,
                fill=(9, 13, 18, 105 if not dim else 150),
            )
            _draw_shadowed_text(draw, (left + 6, top + 3), label, cell_font)

    title_font = _font(max(24, width // 100))
    draw.rounded_rectangle((24, 20, min(width - 24, 980), 78), radius=10, fill=(5, 8, 12, 190))
    _draw_shadowed_text(draw, (42, 28), title, title_font, fill=(255, 242, 193, 255))
    return overlay


def _elevation_variant(raw: Any, survey: Mapping[str, Any], mapping: Mapping[str, Any], fields: Mapping[str, Any]) -> Any:
    from PIL import Image

    image = raw.convert("RGBA")
    height_field = np.asarray(fields["height_gu"], dtype=np.float64)
    slope_field = np.asarray(fields["slope_deg"], dtype=np.float64)
    side = 1024
    origin = survey["frame"]["origin_gu"]
    bounds = survey["target_cells"]
    total_x = (int(bounds["max_x"]) - int(bounds["min_x"]) + 1) * 8192.0
    total_y = (int(bounds["max_y"]) - int(bounds["min_y"]) + 1) * 8192.0
    xs = np.clip(np.rint(np.linspace(0.0, total_x, side) / 128.0).astype(int), 0, height_field.shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(0.0, total_y, side) / 128.0).astype(int), 0, height_field.shape[0] - 1)
    height = height_field[np.ix_(ys, xs)]
    slope = slope_field[np.ix_(ys, xs)]
    low = float(np.nanmin(height))
    high = float(np.nanmax(height))
    normalized = np.clip((height - low) / max(high - low, 1.0), 0.0, 1.0)
    stops = np.asarray(
        [[32, 66, 105], [48, 112, 105], [128, 147, 91], [190, 154, 91], [231, 208, 145]],
        dtype=np.float64,
    )
    positions = np.linspace(0.0, 1.0, len(stops))
    tint = np.stack([np.interp(normalized, positions, stops[:, channel]) for channel in range(3)], axis=-1)
    # A deterministic hillshade from the measured 128-GU field gives ridges
    # and pads visual separation while retaining the real render underneath.
    dy, dx = np.gradient(height, 128.0, 128.0)
    nx, ny, nz = -dx, -dy, np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    shade = np.clip(0.60 + 0.55 * ((nx * -0.45 + ny * -0.55 + nz * 0.75) / norm), 0.42, 1.15)
    tint *= shade[..., None]
    alpha = np.full((side, side, 1), 105.0, dtype=np.float64)
    alpha[height <= 0.0, 0] = 72.0
    rgba = np.concatenate([np.clip(tint, 0, 255), alpha], axis=-1).astype(np.uint8)
    tint_image = Image.fromarray(rgba, mode="RGBA").resize(image.size, Image.Resampling.BICUBIC)
    return Image.alpha_composite(image, tint_image)


def _road_masks_for_render(
    survey: Mapping[str, Any],
    mapping: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map the canonical tile mask to output pixels without interpolation.

    Pixel centres are converted back into plan-frame GU and then into source
    tile indices.  This is intentionally a nearest/source lookup rather than
    a resized image: a gap between occupied LAND tiles cannot be bridged by
    anti-aliasing or a line joint.  The edge mask is restricted to occupied
    pixels, so its outline never paints a synthetic road in an empty tile.
    """

    grids = survey.get("tile_grids")
    if not isinstance(grids, Mapping):
        raise ValueError("survey has no tile_grids road mask")
    shape = grids.get("side")
    encoded = grids.get("road_mask")
    if not isinstance(shape, int) or not isinstance(encoded, str):
        raise ValueError("survey road mask is incomplete")
    raw = base64.b64decode(encoded, validate=True)
    expected_bytes = int(shape) * int(shape)
    if len(raw) != expected_bytes:
        raise ValueError(f"survey road mask has {len(raw)} bytes; expected {expected_bytes}")
    survey_mask = np.frombuffer(raw, dtype=np.uint8).reshape((int(shape), int(shape))).copy()

    evidence_mask_document = evidence.get("target_mask")
    if not isinstance(evidence_mask_document, Mapping):
        raise ValueError("LAND road evidence has no target_mask")
    evidence_shape = evidence_mask_document.get("shape")
    evidence_encoded = evidence_mask_document.get("base64")
    if evidence_shape != [int(shape), int(shape)] or not isinstance(evidence_encoded, str):
        raise ValueError("survey and land_roads mask shapes do not agree")
    evidence_raw = base64.b64decode(evidence_encoded, validate=True)
    canonical_mask = np.frombuffer(evidence_raw, dtype=np.uint8).reshape((int(shape), int(shape))).copy()
    if not np.array_equal(survey_mask, canonical_mask):
        raise RuntimeError("survey road_mask differs from canonical land_roads.json mask")

    tile_rows = evidence.get("road_tiles")
    if not isinstance(tile_rows, list):
        raise ValueError("LAND road evidence has no road_tiles list")
    rows_mask = np.zeros_like(canonical_mask)
    expected_tile_id = 1
    for row in tile_rows:
        if not isinstance(row, Mapping):
            raise ValueError("LAND road evidence contains a non-object road tile")
        site_tile = row.get("site_tile")
        if not isinstance(site_tile, list) or len(site_tile) != 2:
            raise ValueError("LAND road evidence tile has no site_tile")
        tile_x, tile_y = int(site_tile[0]), int(site_tile[1])
        if not (0 <= tile_x < int(shape) and 0 <= tile_y < int(shape)):
            raise ValueError(f"LAND road evidence tile is outside target mask: {site_tile}")
        if int(row.get("raw_vtex", -1)) != 78:
            raise ValueError("LAND road evidence contains a non-78 road tile")
        if str(row.get("tile_id")) != f"T{expected_tile_id:04d}":
            raise ValueError("LAND road evidence road_tiles are not deterministic row-major rows")
        rows_mask[tile_y, tile_x] = 1
        expected_tile_id += 1
    if not np.array_equal(rows_mask, canonical_mask):
        raise RuntimeError("LAND road tile rows do not exactly reproduce target_mask")

    resolution = int(mapping["resolution"][0])
    if int(mapping["resolution"][1]) != resolution:
        raise ValueError("road render requires a square mapping")
    pixel_x = np.arange(resolution, dtype=np.float64) + 0.5
    pixel_y = np.arange(resolution, dtype=np.float64) + 0.5
    scale = float(mapping["px_per_gu"])
    origin_px = mapping["origin_px"]
    plan_x = (pixel_x - float(origin_px[0])) / scale
    plan_y = (float(origin_px[1]) - pixel_y) / scale
    tile_x = np.floor(plan_x / 512.0).astype(np.int64)
    tile_y = np.floor(plan_y / 512.0).astype(np.int64)
    valid_x = (tile_x >= 0) & (tile_x < int(shape))
    valid_y = (tile_y >= 0) & (tile_y < int(shape))
    pixel_mask = np.zeros((resolution, resolution), dtype=bool)
    valid = valid_y[:, None] & valid_x[None, :]
    tile_y_grid = np.broadcast_to(tile_y[:, None], valid.shape)
    tile_x_grid = np.broadcast_to(tile_x[None, :], valid.shape)
    pixel_mask[valid] = canonical_mask[tile_y_grid[valid], tile_x_grid[valid]].astype(bool)

    # An outline is a source-mask edge, not a stroked vector.  Restricting all
    # four-neighbour differences to occupied pixels preserves isolated tiles
    # and leaves every empty source tile untouched.
    source_edge = pixel_mask & (
        np.pad(~pixel_mask[1:, :], ((0, 1), (0, 0)), constant_values=True)
        | np.pad(~pixel_mask[:-1, :], ((1, 0), (0, 0)), constant_values=True)
        | np.pad(~pixel_mask[:, 1:], ((0, 0), (0, 1)), constant_values=True)
        | np.pad(~pixel_mask[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    )
    edge_mask = source_edge
    audit = {
        "mask_shape": [int(shape), int(shape)],
        "source_mask_tile_count": int(np.count_nonzero(canonical_mask)),
        "road_tile_rows": len(tile_rows),
        "road_tile_rows_match_mask": True,
        "pixel_road_coverage": int(np.count_nonzero(pixel_mask)),
        "pixel_source_edge_coverage": int(np.count_nonzero(edge_mask)),
        "pixel_mapping": "pixel-centre inverse affine to floor(plan_gu / 512 GU); no interpolation",
        "outside_source_mask_pixels": 0,
    }
    return pixel_mask, edge_mask, audit


def _draw_continuation_markers(
    draw: Any,
    survey: Mapping[str, Any],
    mapping: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    """Add compact cyan brackets and exact tile-span labels to the image."""

    origin = survey["frame"]["origin_gu"]
    width = int(mapping["resolution"][0])
    height = int(mapping["resolution"][1])
    label_font = _font(max(12, width // 340))
    marker_color = (84, 232, 255, 245)
    label_rows = evidence.get("boundary_continuations")
    if not isinstance(label_rows, list):
        raise ValueError("LAND road evidence has no boundary_continuations list")

    def plan_to_px(point: Sequence[float]) -> tuple[float, float]:
        return _gu_to_px(
            mapping,
            origin,
            [float(origin[0]) + float(point[0]), float(origin[1]) + float(point[1])],
        )

    for row in label_rows:
        side = str(row["side"])
        span = row["target_tile_span"]
        start = int(span["start"])
        end = int(span["end"])
        border = row["plan_gu_border"]
        coordinate = float(border["coordinate_gu"])
        span_start, span_end = (float(value) for value in border["span_gu"])
        if side in ("west", "east"):
            start_px = plan_to_px((coordinate, span_start))
            end_px = plan_to_px((coordinate, span_end))
            direction = -1.0 if side == "west" else 1.0
            marker_x = start_px[0] + direction * 16.0
            draw.line((marker_x, start_px[1], marker_x, end_px[1]), fill=marker_color, width=4)
            draw.line((start_px[0], start_px[1], marker_x, start_px[1]), fill=marker_color, width=4)
            draw.line((end_px[0], end_px[1], marker_x, end_px[1]), fill=marker_color, width=4)
            label = f"{side[0].upper()} LAND EXIT y={start}-{end}"
            box = draw.textbbox((0, 0), label, font=label_font)
            label_width = box[2] - box[0]
            label_height = box[3] - box[1]
            label_x = 8 if side == "west" else width - label_width - 8
            label_y = (start_px[1] + end_px[1]) / 2.0 - label_height / 2.0
        else:
            start_px = plan_to_px((span_start, coordinate))
            end_px = plan_to_px((span_end, coordinate))
            direction = 1.0 if side == "south" else -1.0
            marker_y = start_px[1] + direction * 16.0
            draw.line((start_px[0], marker_y, end_px[0], marker_y), fill=marker_color, width=4)
            draw.line((start_px[0], start_px[1], start_px[0], marker_y), fill=marker_color, width=4)
            draw.line((end_px[0], end_px[1], end_px[0], marker_y), fill=marker_color, width=4)
            label = f"{side[0].upper()} LAND EXIT x={start}-{end}"
            box = draw.textbbox((0, 0), label, font=label_font)
            label_width = box[2] - box[0]
            label_height = box[3] - box[1]
            label_x = (start_px[0] + end_px[0]) / 2.0 - label_width / 2.0
            label_y = 8 if side == "north" else height - label_height - 8
        label_x = max(4.0, min(float(width - label_width - 4), label_x))
        label_y = max(4.0, min(float(height - label_height - 4), label_y))
        draw.rounded_rectangle(
            (label_x - 4, label_y - 3, label_x + label_width + 4, label_y + label_height + 3),
            radius=4,
            fill=(4, 12, 18, 220),
        )
        _draw_shadowed_text(draw, (label_x, label_y), label, label_font, fill=marker_color)


def _road_variant(
    raw: Any,
    survey: Mapping[str, Any],
    mapping: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Annotate real terrain with only the canonical LAND road occupancy."""

    from PIL import Image, ImageDraw

    road_evidence = survey["roads"] if evidence is None else evidence
    image = raw.convert("RGBA")
    image = image.point(lambda value: int(value * 0.58) if value < 250 else value)
    # Draw planner aids before the mask overlay so occupied source pixels and
    # their exact outline remain visible beneath the cell labels/grid.
    result = _draw_grid_and_rulers(
        image,
        survey,
        mapping,
        title="FALKREATH V1 — LAND/VTEX ROAD PAINT / z=0 WATER",
        dim=False,
    )
    pixel_mask, edge_mask, audit = _road_masks_for_render(survey, mapping, road_evidence)
    overlay = np.zeros((pixel_mask.shape[0], pixel_mask.shape[1], 4), dtype=np.uint8)
    overlay[pixel_mask, :3] = (236, 116, 34)
    overlay[pixel_mask, 3] = 82
    overlay[edge_mask, :3] = (255, 238, 112)
    overlay[edge_mask, 3] = 215
    result = Image.alpha_composite(result, Image.fromarray(overlay, mode="RGBA"))

    draw = ImageDraw.Draw(result, "RGBA")
    legend = ImageDraw.Draw(result, "RGBA")
    _draw_continuation_markers(draw, survey, mapping, road_evidence)
    legend_font = _font(max(16, int(mapping["resolution"][0]) // 220))
    width = int(mapping["resolution"][0])
    components = int(road_evidence["component_statistics"]["eight_neighbour_count"])
    spans = int(road_evidence["boundary_statistics"]["total_continuation_spans"])
    legend_height = 126
    legend.rounded_rectangle(
        (width - 760, 20, width - 24, 20 + legend_height),
        radius=10,
        fill=(5, 8, 12, 205),
    )
    legend.rectangle((width - 726, 49, width - 666, 69), fill=(236, 116, 34, 160), outline=(255, 238, 112, 215), width=2)
    _draw_shadowed_text(draw, (width - 642, 45), "raw VTEX 78 occupied tile", legend_font)
    legend.line((width - 726, 94, width - 666, 94), fill=(84, 232, 255, 245), width=4)
    _draw_shadowed_text(draw, (width - 642, 81), f"perimeter-confirmed exits ({spans} spans)", legend_font)
    _draw_shadowed_text(draw, (width - 726, 112), f"source components: {components} (8-neighbour)", legend_font)
    audit.update(
        {
            "component_count_8": components,
            "continuation_span_count": spans,
            "source_kind": "LAND/VTEX raw 78",
            "graph_geometry_used": False,
        }
    )
    return result, audit


def _annotate_outputs(
    raw_dir: Path,
    output_dir: Path,
    survey: Mapping[str, Any],
    worker_audit: Mapping[str, Any],
    texture_gate: Mapping[str, Any],
    resolution: int,
) -> dict[str, Any]:
    from PIL import Image

    mapping = _frame_mapping(survey, resolution)
    fields_path = output_dir / "survey_fields.npz"
    with np.load(fields_path) as fields:
        field_arrays = {key: fields[key] for key in fields.files}
    land_roads_path = output_dir / str(survey.get("artifacts", {}).get("land_roads", "land_roads.json"))
    if not land_roads_path.is_file():
        raise FileNotFoundError(f"canonical LAND road evidence is missing: {land_roads_path}")
    road_evidence = _read_json(land_roads_path)
    if not isinstance(road_evidence, Mapping):
        raise ValueError(f"canonical LAND road evidence is not an object: {land_roads_path}")
    if road_evidence != survey.get("roads"):
        raise RuntimeError("site_survey roads differ from canonical land_roads.json before rendering")
    raw_top = Image.open(raw_dir / "site_topdown_raw.png").convert("RGBA")
    raw_elevation = Image.open(raw_dir / "site_elevation_raw.png").convert("RGBA")
    raw_roads = Image.open(raw_dir / "site_roads_raw.png").convert("RGBA")
    raw_oblique = Image.open(raw_dir / "site_oblique_raw.png").convert("RGBA")
    expected_size = (resolution, resolution)
    for label, image in (("topdown", raw_top), ("elevation", raw_elevation), ("roads", raw_roads), ("oblique", raw_oblique)):
        if image.size != expected_size:
            raise RuntimeError(f"{label} raw render has {image.size}, expected {expected_size}")
        if len(image.getbands()) < 3:
            raise RuntimeError(f"{label} raw render has no RGB channels")

    road_image, road_overlay_audit = _road_variant(raw_roads, survey, mapping, road_evidence)
    outputs = {
        "site_topdown.png": _draw_grid_and_rulers(raw_top, survey, mapping, title="FALKREATH V1 — TOPOGRAPHY / z=0 WATER"),
        "site_elevation.png": _draw_grid_and_rulers(
            _elevation_variant(raw_elevation, survey, mapping, field_arrays),
            survey,
            mapping,
            title="FALKREATH V1 — ELEVATION TINT + HILLSHADE / z=0 WATER",
        ),
        "site_roads.png": road_image,
        "site_oblique.png": raw_oblique,
    }
    # A small title strip on the perspective image is useful in a contact
    # sheet, but the underlying pixels remain the Blender render.
    from PIL import ImageDraw

    oblique_draw = ImageDraw.Draw(outputs["site_oblique.png"], "RGBA")
    oblique_draw.rounded_rectangle((24, 20, min(resolution - 24, 900), 78), radius=10, fill=(5, 8, 12, 180))
    _draw_shadowed_text(oblique_draw, (42, 28), "FALKREATH V1 — SW→NE OBLIQUE / z=0 WATER", _font(max(24, resolution // 100)), fill=(255, 242, 193, 255))
    for name, image in outputs.items():
        image.save(output_dir / name, format="PNG", optimize=False)

    output_files = {
        name: {
            "sha256": _sha256_file(output_dir / name),
            "size_bytes": (output_dir / name).stat().st_size,
        }
        for name in outputs
    }

    origin = survey["frame"]["origin_gu"]
    round_trip = _mapping_round_trip(survey, mapping)
    if not round_trip["under_one_pixel"]:
        raise RuntimeError(f"GU/px mapping round-trip exceeded one pixel: {round_trip['max_error_px']}")
    terrain_hash = str(worker_audit["terrain_hash"])
    render_rows: dict[str, Any] = {}
    for name, worker_name in (
        ("site_topdown.png", "topdown"),
        ("site_elevation.png", "elevation"),
        ("site_roads.png", "roads"),
        ("site_oblique.png", "oblique"),
    ):
        worker_row = worker_audit["renders"][worker_name]
        row = {
            "camera_mode": worker_row["camera_mode"],
            "camera_type": worker_row["camera_type"],
            "camera_location_scene": worker_row["camera_location_scene"],
            "camera_rotation_euler": worker_row["camera_rotation_euler"],
            "ortho_scale_scene": worker_row.get("ortho_scale_scene"),
            "lens_mm": worker_row.get("lens_mm"),
            "resolution": [resolution, resolution],
            "terrain_hash": terrain_hash,
            "terrain_texture_gate": dict(texture_gate),
            "water_z": worker_row["water_z"],
            "water_extent_audit": worker_row["water_extent_audit"],
            "output_sha256": output_files[name]["sha256"],
            "output_size_bytes": output_files[name]["size_bytes"],
            "raw_render_path": None,
            "raw_render_is_temporary": True,
        }
        if worker_name == "roads":
            row["road_overlay_audit"] = dict(road_overlay_audit)
        if worker_name != "oblique":
            row["px_mapping"] = dict(mapping)
            row["mapping_round_trip"] = round_trip
        else:
            row["px_mapping"] = None
            row["mapping_note"] = "perspective view; use top-down affine mapping for plan coordinates"
        render_rows[name] = row

    audit = {
        "schema_version": 1,
        "survey_id": survey["survey_id"],
        "resolution": [resolution, resolution],
        "terrain_hash": terrain_hash,
        "water_z_required": 0.0,
        "real_texture_gate": dict(texture_gate),
        "mapping_round_trip": round_trip,
        "water_extent_audit": worker_audit["water_extent_audit"],
        "road_overlay_audit": road_overlay_audit,
        "land_roads_sha256": _sha256_file(land_roads_path),
        "image_hashes": output_files,
        "renders": render_rows,
        "worker": {"blender_version": worker_audit.get("blender_version"), "samples": worker_audit.get("samples")},
    }
    _write_json(output_dir / "render_audit.json", audit)

    # Publish the same exact mapping in the machine contract that the audit
    # records.  The host only changes this generated JSON, never an input.
    survey_path = output_dir / "site_survey.json"
    published_survey = dict(survey)
    published_frame = dict(survey["frame"])
    published_mapping = dict(survey["frame"].get("render_mapping", {}))
    for name in ("site_topdown.png", "site_elevation.png", "site_roads.png"):
        published_mapping[name] = dict(mapping)
    published_frame["render_mapping"] = published_mapping
    published_survey["frame"] = published_frame
    published_survey["render_audit"] = "render_audit.json"
    _write_json(survey_path, published_survey)
    return audit


def _scene_document(survey: Mapping[str, Any], resolution: int) -> dict[str, Any]:
    cells = [row["grid"] for row in survey["cells"]]
    land_value = str(survey["inputs"]["land_source"]).split(" (sha256:", 1)[0]
    return {
        "scene_name": "Cityforge_Falkreath_Site",
        "procgen_config": "configs/procgen.json",
        "import": {"scale_correction": SCENE_UNITS_PER_GU},
        "terrain": {
            "enabled": True,
            "plugin": land_value,
            "texture_plugin": land_value,
            "texture_masters": [],
            "anchor_grid": list(survey["seed_settlement"]["anchor_cell"]),
            "cells": cells,
            "decimate": 1,
            "scene_units_per_game_unit": SCENE_UNITS_PER_GU,
            "height_scale": SCENE_UNITS_PER_GU * 8.0,
            "height_offset_scene_units": 0.0,
            "require_real_textures": True,
            "max_seconds": 180.0,
        },
        "water": {
            "enabled": True,
            "z": 0.0,
            # Do not paint the empty camera margin blue: the plane covers the
            # surveyed terrain footprint, and the exact z=0 surface remains
            # visible only where the LAND terrain is below it.
            "margin": 0.0,
            "size": (7 * 8192.0) * SCENE_UNITS_PER_GU,
        },
        "camera": {"resolution": [resolution, resolution]},
        "lighting": {
            "sun_energy": 1.15,
            "sun_angle_degrees": 25.0,
            "sun_rotation_degrees": [28.0, -32.0, -35.0],
            "world_color": [0.028, 0.038, 0.055, 1.0],
            "world_strength": 0.62,
            "fill_energy": 1250.0,
            "fill_size_factor": 1.7,
            "exposure": 0.35,
            "view_look": "AgX - Medium High Contrast",
        },
    }


def _vector_list(vector: Any) -> list[float]:
    return [round(float(vector[index]), 9) for index in range(3)]


def _clip_triangle_to_z0(points: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    """Clip one terrain triangle to the half-space at or below z=0.

    ``blender_flat_render.add_water`` intentionally supplies a rectangular
    sea-level plane.  That is useful for the generic flat renderer but is not
    sufficient for a survey block: in perspective, the plane is visible
    around elevated terrain edges.  Cityforge therefore derives a water
    surface from the actual terrain triangles and keeps only their submerged
    portions.  This small Sutherland-Hodgman step is performed in world
    coordinates; the caller converts the resulting vertices into the water
    object's local space.
    """

    if len(points) < 3:
        return []
    clipped: list[tuple[float, float, float]] = []
    for current, following in zip(points, (*points[1:], points[0])):
        current_inside = float(current[2]) <= 0.0
        following_inside = float(following[2]) <= 0.0
        if current_inside and following_inside:
            clipped.append((float(following[0]), float(following[1]), 0.0))
        elif current_inside and not following_inside:
            denominator = float(following[2]) - float(current[2])
            if abs(denominator) > 1e-12:
                fraction = -float(current[2]) / denominator
                clipped.append(
                    (
                        float(current[0]) + fraction * (float(following[0]) - float(current[0])),
                        float(current[1]) + fraction * (float(following[1]) - float(current[1])),
                        0.0,
                    )
                )
        elif not current_inside and following_inside:
            denominator = float(following[2]) - float(current[2])
            if abs(denominator) > 1e-12:
                fraction = -float(current[2]) / denominator
                clipped.append(
                    (
                        float(current[0]) + fraction * (float(following[0]) - float(current[0])),
                        float(current[1]) + fraction * (float(following[1]) - float(current[1])),
                        0.0,
                    )
                )
            clipped.append((float(following[0]), float(following[1]), 0.0))
    # A vertex exactly on z=0 can be emitted twice by adjacent edge tests.
    # Remove adjacent duplicates before fan triangulation to avoid zero-area
    # polygons at the shoreline.
    deduplicated: list[tuple[float, float, float]] = []
    for point in clipped:
        if not deduplicated or any(abs(point[index] - deduplicated[-1][index]) > 1e-9 for index in range(2)):
            deduplicated.append(point)
    if len(deduplicated) > 1 and all(abs(deduplicated[0][index] - deduplicated[-1][index]) <= 1e-9 for index in range(2)):
        deduplicated.pop()
    return deduplicated


def _object_xy_measurements(obj: Any, vector_type: Any) -> dict[str, Any]:
    """Measure world-space XY bounds and surface area for Blender audit data."""

    corners = [obj.matrix_world @ vector_type(corner) for corner in obj.bound_box]
    minimum = [min(float(point[index]) for point in corners) for index in (0, 1)]
    maximum = [max(float(point[index]) for point in corners) for index in (0, 1)]
    area = 0.0
    for polygon in obj.data.polygons:
        polygon_points = [obj.matrix_world @ obj.data.vertices[index].co for index in polygon.vertices]
        if len(polygon_points) < 3:
            continue
        first = polygon_points[0]
        for index in range(1, len(polygon_points) - 1):
            second = polygon_points[index]
            third = polygon_points[index + 1]
            area += abs(
                (float(second.x) - float(first.x)) * (float(third.y) - float(first.y))
                - (float(second.y) - float(first.y)) * (float(third.x) - float(first.x))
            ) * 0.5
    size = [maximum[index] - minimum[index] for index in (0, 1)]
    return {
        "min_xy_scene": [round(value, 9) for value in minimum],
        "max_xy_scene": [round(value, 9) for value in maximum],
        "size_xy_scene": [round(value, 9) for value in size],
        "bounding_rect_area_scene2": round(size[0] * size[1], 9),
        "surface_area_xy_scene2": round(area, 9),
        "vertex_count": len(obj.data.vertices),
        "face_count": len(obj.data.polygons),
    }


def _submerged_water_surface(terrain: Any, water: Any, vector_type: Any, bpy_module: Any) -> dict[str, Any]:
    """Replace the helper rectangle with the exact below-terrain z=0 mesh."""

    world_matrix = terrain.matrix_world
    water_inverse = water.matrix_world.inverted()
    world_vertices = [world_matrix @ vertex.co for vertex in terrain.data.vertices]
    water_vertices: list[tuple[float, float, float]] = []
    water_faces: list[tuple[int, int, int]] = []
    submerged_polygons = 0
    for polygon in terrain.data.polygons:
        polygon_indices = list(polygon.vertices)
        if len(polygon_indices) < 3:
            continue
        for index in range(1, len(polygon_indices) - 1):
            triangle = [world_vertices[polygon_indices[0]], world_vertices[polygon_indices[index]], world_vertices[polygon_indices[index + 1]]]
            clipped = _clip_triangle_to_z0([(point.x, point.y, point.z) for point in triangle])
            if len(clipped) < 3:
                continue
            submerged_polygons += 1
            local_points = [water_inverse @ vector_type(point) for point in clipped]
            start = len(water_vertices)
            water_vertices.extend((float(point.x), float(point.y), float(point.z)) for point in local_points)
            for fan_index in range(1, len(local_points) - 1):
                water_faces.append((start, start + fan_index, start + fan_index + 1))

    if not water_faces:
        raise RuntimeError("terrain contains no z<=0 surface for the Cityforge water mesh")
    old_mesh = water.data
    clipped_mesh = bpy_module.data.meshes.new("ProcGen_Water_Clipped_Mesh")
    clipped_mesh.from_pydata(water_vertices, [], water_faces)
    clipped_mesh.update()
    for material in old_mesh.materials:
        clipped_mesh.materials.append(material)
    water.data = clipped_mesh
    if old_mesh.users == 0:
        bpy_module.data.meshes.remove(old_mesh)
    measurements = _object_xy_measurements(water, vector_type)
    measurements["submerged_triangle_count"] = submerged_polygons
    measurements["z_plane"] = 0.0
    return measurements


def _blender_worker(survey_path: Path, work_dir: Path, resolution: int, samples: int) -> int:
    """Build/render the four image-backed terrain scenes inside Blender."""

    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    tools_dir = WORKSPACE / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import blender_flat_render as flat  # type: ignore

    survey = _read_json(survey_path)
    document = _scene_document(survey, resolution)
    roots, resolver = flat.load_procgen_meshcheck(document.get("procgen_config"))
    render_specs = {
        "topdown": {"mode": "ORTHO", "oblique": False},
        "elevation": {"mode": "ORTHO", "oblique": False},
        "roads": {"mode": "ORTHO", "oblique": False},
        "oblique": {"mode": "PERSP", "oblique": True},
    }
    rows: dict[str, Any] = {}
    terrain_hash = survey["inputs"]["land_source"].split("sha256:", 1)[1].rstrip(")")
    for name, spec in render_specs.items():
        bpy.ops.wm.read_factory_settings(use_empty=True)
        terrain, terrain_bounds = flat.add_terrain(document, roots, resolver)
        if terrain is None or terrain_bounds is None:
            raise RuntimeError("terrain worker produced no terrain bounds")
        water = flat.add_water(terrain_bounds, document)
        if water is None:
            raise RuntimeError("terrain worker produced no z=0 water plane")
        if abs(float(water.location.z)) > 1e-9:
            raise RuntimeError(f"water plane is not exactly z=0: {water.location.z}")
        terrain_xy = _object_xy_measurements(terrain, Vector)
        helper_water_xy = _object_xy_measurements(water, Vector)
        water_xy = _submerged_water_surface(terrain, water, Vector, bpy)
        terrain_metrics = {
            key: terrain.get(key)
            for key in (
                "procgen_texture_table_count",
                "procgen_texture_indices_used",
                "procgen_texture_materials_used",
                "procgen_texture_files_resolved",
                "procgen_texture_files_missing_fallback",
                "procgen_texture_base_tiles",
                "procgen_texture_base_files_resolved",
                "procgen_texture_base_missing_fallback",
                "procgen_texture_missing_ltex_definitions",
            )
        }
        if int(terrain_metrics.get("procgen_texture_files_missing_fallback") or 0) != 0:
            raise RuntimeError(f"render terrain has missing nonzero textures: {terrain_metrics}")
        if int(terrain_metrics.get("procgen_texture_base_missing_fallback") or 0) != 0:
            raise RuntimeError(f"render terrain has missing base textures: {terrain_metrics}")
        if int(terrain_metrics.get("procgen_texture_files_resolved") or 0) <= 0:
            raise RuntimeError(f"render terrain resolved no real LTEX images: {terrain_metrics}")

        # Keep the actual image node as the source of the small emission lift.
        # This improves legibility of the dark highland textures in a planner
        # view without introducing a fallback color or replacing real pixels.
        for material in terrain.data.materials:
            if material is None or not material.use_nodes:
                continue
            shader = material.node_tree.nodes.get("Principled BSDF")
            image_node = next(
                (
                    node
                    for node in material.node_tree.nodes
                    if node.bl_idname == "ShaderNodeTexImage" and node.image is not None
                ),
                None,
            )
            if shader is not None and image_node is not None and "Emission Color" in shader.inputs:
                material.node_tree.links.new(image_node.outputs["Color"], shader.inputs["Emission Color"])
                if "Emission Strength" in shader.inputs:
                    shader.inputs["Emission Strength"].default_value = 0.28

        minimum, maximum = terrain_bounds
        lighting = document["lighting"]
        flat.add_sun(lighting)
        flat.add_fill_light(minimum, maximum, lighting)
        frame_origin = Vector((float(survey["frame"]["origin_gu"][0]), float(survey["frame"]["origin_gu"][1]), 0.0))
        anchor = survey["seed_settlement"]["anchor_cell"]
        frame_center_abs = Vector(
            (
                float(survey["frame"]["origin_gu"][0]) + 7.0 * 8192.0 / 2.0,
                float(survey["frame"]["origin_gu"][1]) + 7.0 * 8192.0 / 2.0,
                0.0,
            )
        )
        anchor_abs = Vector((float(anchor[0]) * 8192.0, float(anchor[1]) * 8192.0, 0.0))
        center_scene = Vector(
            ((frame_center_abs.x - anchor_abs.x) * SCENE_UNITS_PER_GU,
             (frame_center_abs.y - anchor_abs.y) * SCENE_UNITS_PER_GU,
             (minimum.z + maximum.z) / 2.0)
        )
        bpy.ops.object.camera_add(location=(center_scene.x, center_scene.y, maximum.z + 750.0))
        camera = bpy.context.object
        camera.name = f"Cityforge_Falkreath_{name}_Camera"
        camera.data.type = spec["mode"]
        if not spec["oblique"]:
            camera.data.ortho_scale = (7.0 * 8192.0 + 2.0 * RENDER_MARGIN_GU) * SCENE_UNITS_PER_GU
            camera.rotation_euler = (0.0, 0.0, 0.0)
        else:
            # Camera is southwest of the block and looks northeast, making
            # the direction explicit rather than relying on a default view.
            direction = Vector((-0.62, -0.62, 0.48)).normalized()
            distance = 2.25 * (7.0 * 8192.0 + 2.0 * RENDER_MARGIN_GU) * SCENE_UNITS_PER_GU
            camera.location = center_scene + direction * distance
            target = Vector((center_scene.x, center_scene.y, (minimum.z + maximum.z) * 0.36))
            flat.point_camera(camera, target)
            camera.data.lens = 52.0
        camera.data.clip_start = 0.1
        camera.data.clip_end = 10000.0
        output_path = work_dir / f"site_{name}_raw.png"
        document_for_render = dict(document)
        document_for_render["camera"] = {"mode": spec["mode"], "resolution": [resolution, resolution]}
        flat.configure_render(document_for_render, output_path, camera)
        bpy.context.scene.view_settings.exposure = 1.05
        bpy.context.scene.cycles.samples = int(samples)
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.render.film_transparent = False
        bpy.ops.render.render(write_still=True)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"Blender did not write raw render: {output_path}")
        rows[name] = {
            "camera_mode": spec["mode"],
            "camera_type": camera.data.type,
            "camera_location_scene": _vector_list(camera.location),
            "camera_rotation_euler": _vector_list(camera.rotation_euler),
            "ortho_scale_scene": float(camera.data.ortho_scale) if camera.data.type == "ORTHO" else None,
            "lens_mm": float(camera.data.lens) if camera.data.type == "PERSP" else None,
            "resolution": [resolution, resolution],
            "terrain_bounds_scene": {"min": _vector_list(minimum), "max": _vector_list(maximum)},
            "water_extent_audit": {
                "terrain_mesh_xy": terrain_xy,
                "helper_plane_before_clipping_xy": helper_water_xy,
                "clipped_submerged_water_xy": water_xy,
                "helper_plane_vs_terrain_aabb_excess_scene": [
                    round(helper_water_xy["size_xy_scene"][0] - terrain_xy["size_xy_scene"][0], 9),
                    round(helper_water_xy["size_xy_scene"][1] - terrain_xy["size_xy_scene"][1], 9),
                ],
            },
            "terrain_metrics": terrain_metrics,
            "water_z": float(water.location.z),
            "raw_render_path": str(output_path),
        }
    audit = {
        "blender_version": bpy.app.version_string,
        "samples": int(samples),
        "terrain_hash": terrain_hash,
        "renders": rows,
        "water_extent_audit": rows["topdown"]["water_extent_audit"],
    }
    _write_json(work_dir / "worker_audit.json", audit)
    return 0


def _host_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", default=str(DEFAULT_SURVEY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--blender", default=str(DEFAULT_BLENDER))
    parser.add_argument("--resolution", type=int, default=RENDER_SIZE)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args(list(argv))
    if args.resolution != RENDER_SIZE:
        raise RuntimeError(f"D-SITE requires the fixed {RENDER_SIZE}x{RENDER_SIZE} render canvas")
    survey_path = _workspace_path(args.survey)
    output_dir = _workspace_path(args.output_dir)
    if not survey_path.is_file():
        raise FileNotFoundError(f"survey JSON is missing: {survey_path}")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"survey output directory is missing: {output_dir}")
    survey = _read_json(survey_path)
    texture_gate = _real_texture_gate(survey)
    blender = Path(args.blender)
    if not blender.is_file():
        discovered = shutil.which("blender")
        if discovered is None:
            raise FileNotFoundError(f"Blender executable is missing: {blender}")
        blender = Path(discovered)
    with tempfile.TemporaryDirectory(prefix="cityforge-falkreath-render-") as temporary:
        work_dir = Path(temporary)
        command = [
            str(blender),
            "-b",
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "--blender-worker",
            "--survey",
            str(survey_path),
            "--work-dir",
            str(work_dir),
            "--resolution",
            str(args.resolution),
            "--samples",
            str(args.samples),
        ]
        completed = subprocess.run(command, cwd=str(WORKSPACE), check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Blender render worker failed with exit code {completed.returncode}")
        worker_audit = _read_json(work_dir / "worker_audit.json")
        audit = _annotate_outputs(work_dir, output_dir, survey, worker_audit, texture_gate, args.resolution)
    print(f"renders={len(audit['renders'])} resolution={audit['resolution']}")
    print(f"real_texture_gate={audit['real_texture_gate']['resolved_count']} resolved, 0 missing")
    print(f"water_z={audit['water_z_required']} mapping_max_error_px={audit['mapping_round_trip']['max_error_px']}")
    print(f"render_audit={output_dir / 'render_audit.json'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--blender-worker" in raw:
        parser = argparse.ArgumentParser()
        parser.add_argument("--blender-worker", action="store_true")
        parser.add_argument("--survey", required=True)
        parser.add_argument("--work-dir", required=True)
        parser.add_argument("--resolution", type=int, required=True)
        parser.add_argument("--samples", type=int, required=True)
        args = parser.parse_args(raw)
        return _blender_worker(Path(args.survey), Path(args.work_dir), args.resolution, args.samples)
    return _host_main(raw)


if __name__ == "__main__":
    # Blender's Python exposes bpy only inside the worker invocation.  The
    # host path remains importable/testable with the regular project Python.
    try:
        import bpy  # type: ignore  # noqa: F401
        inside_blender = True
    except ImportError:
        inside_blender = False
    if inside_blender:
        blender_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
        raise SystemExit(main(blender_argv))
    raise SystemExit(main())
