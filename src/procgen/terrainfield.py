"""Shared terrain-field core for the Tamriel Reworked heightmap pipeline.

Purpose
    Build and consume the canonical multi-source LAND corpus: stream TES3
    landmass plugins into one global vertex-height field plus a per-cell
    ownership grid, derive seam geometry, and provide hillshade/crop rendering
    used by every review artifact in this pipeline.

Pipeline position
    Stage A writes ``corpus_*.npz`` (+ manifest json). Stage B (seam atlas),
    Stage C/D (solve), and Stage E/F (authoring/verification) read that npz;
    authoring re-streams the original plugins itself and never mutates them.

Inputs
    ``configs/tamriel_reworked_v1.json``: ordered source plugin specs
    (``role`` = ``base`` for the canvas plugin, ``owner`` for landmass mods),
    output paths, render parameters. Plugin files are opened read-only.

Outputs
    npz arrays (all optional consumers go through :func:`load_corpus`):
      ``tam_h``  float32 GU, raw vertex heights of the base plugin (NaN = void)
      ``oth_h``  float32 GU, vertex heights of winning owners (NaN = void)
      ``cell_owner`` uint8 (0 = void, else source index + 1 into
      ``source_names``; owners resolve later-in-config-wins)
      ``cell_height_source`` uint8 (0 = synthesize, otherwise the authoritative
      source code for that cell)
      ``synth_height_cells`` bool, the explicit required-height mask
      ``gx0``, ``gy0`` int cell coords of array origin
    Vertex ``(row, col)`` = ``((cy - gy0) * 64 + v, (cx - gx0) * 64 + u)``;
    spacing 128 GU; heights in game units (THU x 8).

Invariants
    - Grid bounds are derived from the scanned cell union, never assumed.
    - Owner overlap among ``owner`` sources resolves by config order (later
      wins); the base plugin never wins a contested cell.
    - Nothing in this module writes plugins; outputs live under output/.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from procgen.espland import THU_TO_GU, iter_land

CELL_SIDE_GU = 8192
LAND_VERTS = 65
CELL_QUADS = 64
VERTEX_STEP_GU = CELL_SIDE_GU // CELL_QUADS


class TerrainFieldError(RuntimeError):
    """Raised for corpus/config inconsistencies that must fail closed."""


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class SourceScan:
    name: str
    path: Path
    role: str
    cells: set[tuple[int, int]]
    height_cells: set[tuple[int, int]]
    flat_height_gu: dict[tuple[int, int], float]
    peak_gu: float | None
    peak_cell: tuple[int, int] | None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return min(xs), min(ys), max(xs), max(ys)


def _scan_grids(name: str, path: Path, role: str) -> SourceScan:
    cells: set[tuple[int, int]] = set()
    height_cells: set[tuple[int, int]] = set()
    flat_height_gu: dict[tuple[int, int], float] = {}
    for rec in iter_land(path):
        cells.add(rec.grid)
        if rec.heights_thu is not None:
            try:
                if np.asarray(rec.heights_thu).shape == (LAND_VERTS, LAND_VERTS):
                    height_cells.add(rec.grid)
                    arr = np.asarray(rec.heights_thu, dtype=np.float32)
                    if np.ptp(arr) == 0:
                        flat_height_gu[rec.grid] = float(arr.flat[0] * THU_TO_GU)
            except (TypeError, ValueError):
                pass
    if not cells:
        raise TerrainFieldError(f"source {name!r} ({path}) exposes no LAND records")
    return SourceScan(name=name, path=path, role=role, cells=cells,
                      height_cells=height_cells,
                      flat_height_gu=flat_height_gu,
                      peak_gu=None, peak_cell=None)


def _grid_bounds(scans: list[SourceScan]) -> tuple[int, int, int, int]:
    gx0 = min(s.bbox[0] for s in scans)
    gy0 = min(s.bbox[1] for s in scans)
    gx1 = max(s.bbox[2] for s in scans)
    gy1 = max(s.bbox[3] for s in scans)
    return gx0, gy0, gx1, gy1


def build_corpus(config: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Stream every configured source twice (grids, then heights).

    Returns ``(arrays, manifest)`` ready for :func:`save_corpus`.
    """
    sources = config["sources"]
    if not sources:
        raise TerrainFieldError("config lists no sources")
    roles = [s["role"] for s in sources]
    if roles.count("base") != 1:
        raise TerrainFieldError("exactly one source must have role 'base'")

    scans = [_scan_grids(s["name"], Path(s["path"]), s["role"]) for s in sources]
    gx0, gy0, gx1, gy1 = _grid_bounds(scans)
    ncx = gx1 - gx0 + 1
    ncy = gy1 - gy0 + 1
    rows = ncy * CELL_QUADS + 1
    cols = ncx * CELL_QUADS + 1

    by_name = {s.name: s for s in scans}
    base_idx = next(i for i, s in enumerate(scans) if s.role == "base")
    base_scan = scans[base_idx]

    # Ownership: owner sources resolved in config order, LATER wins (e.g.
    # TR outranks vanilla Morrowind on shared narrow-sea island cells);
    # base fills everything unclaimed. Grid codes are source index + 1
    # so that 0 can mean void without colliding with source index 0.
    cell_owner_flat: dict[tuple[int, int], int] = {}
    owner_order: list[int] = []
    for i, s in enumerate(scans):
        if s.role != "owner":
            continue
        owner_order.append(i)
        for cell in s.cells:
            cell_owner_flat[cell] = i
    for cell in base_scan.cells:
        cell_owner_flat.setdefault(cell, base_idx)

    cell_owner = np.zeros((ncy, ncx), dtype=np.uint8)
    for (cx, cy), idx in cell_owner_flat.items():
        cell_owner[cy - gy0, cx - gx0] = idx + 1

    # Resolve height authority independently from cell ownership. A winning
    # owner LAND record without VHGT must not leave an earlier owner's raster
    # under the cell; base VHGT is the only permitted owner-stub fallback.
    cell_height_source = np.zeros((ncy, ncx), dtype=np.uint8)
    fallback_cfg = config.get("flat_owner_fallback", {})
    fallback_enabled = bool(fallback_cfg.get("enabled", True))
    fallback_max_height_gu = float(fallback_cfg.get("max_height_gu", -2000.0))
    flat_owner_fallback_cells: list[list[int]] = []
    for (cx, cy), idx in cell_owner_flat.items():
        if (cx, cy) in scans[idx].height_cells:
            source_idx = idx
            flat_gu = scans[idx].flat_height_gu.get((cx, cy))
            if (
                fallback_enabled
                and idx != base_idx
                and flat_gu is not None
                and flat_gu <= fallback_max_height_gu
                and (cx, cy) in base_scan.height_cells
            ):
                source_idx = base_idx
                flat_owner_fallback_cells.append([cx, cy])
        elif (cx, cy) in base_scan.height_cells:
            source_idx = base_idx
        else:
            source_idx = -1
        if source_idx >= 0:
            cell_height_source[cy - gy0, cx - gx0] = source_idx + 1
    synth_height_cells = (cell_owner != 0) & (cell_height_source == 0)

    fields: dict[str, np.ndarray | None] = {
        "tam_h": np.full((rows, cols), np.nan, dtype=np.float32),
        "oth_h": np.full((rows, cols), np.nan, dtype=np.float32),
    }
    dup_stats: dict[str, dict[str, float]] = {}

    def fill(scan_idx: int, key: str, winning_only: bool = False) -> None:
        scan = scans[scan_idx]
        stats = {"max_delta_gu": 0.0, "conflicts": 0}
        dup_stats[key] = stats
        peak = None
        peak_cell = None
        for rec in iter_land(scan.path):
            thu = rec.heights_thu
            if thu is None:
                continue
            arr = np.asarray(thu, dtype=np.float32)
            if arr.shape != (LAND_VERTS, LAND_VERTS):
                continue
            arr = arr * THU_TO_GU
            cx, cy = rec.grid
            if (winning_only and
                    cell_height_source[cy - gy0, cx - gx0] != scan_idx + 1):
                continue
            r0 = (cy - gy0) * CELL_QUADS
            c0 = (cx - gx0) * CELL_QUADS
            sl = fields[key][r0:r0 + LAND_VERTS, c0:c0 + LAND_VERTS]
            prev_valid = ~np.isnan(sl)
            if prev_valid.any():
                deltas = np.abs(sl - arr)[prev_valid]
                dmax = float(deltas.max())
                if dmax > stats["max_delta_gu"]:
                    stats["max_delta_gu"] = dmax
                stats["conflicts"] += int((deltas > 0.5).sum())
            sl[:, :] = arr
            m = float(arr.max())
            if peak is None or m > peak:
                peak = m
                peak_cell = rec.grid
        scan.peak_gu = peak
        scan.peak_cell = peak_cell

    fill(base_idx, "tam_h")
    for i in owner_order:
        fill(i, "oth_h", winning_only=True)

    seams = seam_edges(cell_owner, base_idx + 1, gy0, gx0)

    manifest = {
        "grid": {"gx0": gx0, "gy0": gy0, "gx1": gx1, "gy1": gy1,
                 "cells_x": ncx, "cells_y": ncy},
        "source_names": [s.name for s in scans],
        "base_source": base_scan.name,
        "sources": {
            s.name: {
                "path": str(s.path),
                "role": s.role,
                "lands": len(s.cells),
                "height_lands": len(s.height_cells),
                "bbox_xyxy": list(s.bbox),
                "peak_gu": s.peak_gu,
                "peak_cell": list(s.peak_cell) if s.peak_cell else None,
            }
            for s in scans
        },
        "retained_cells": int((cell_owner == base_idx + 1).sum()),
        "deleted_cells": sum(1 for c in base_scan.cells
                             if cell_owner_flat[c] != base_idx),
        "seam_edges": len(seams),
        "seam_tam_cells": len({e[0] for e in seams}),
        "duplicate_vertex_audit": dup_stats,
        "cell_height_source_counts": {
            str(int(code)): int((cell_height_source == code).sum())
            for code in np.unique(cell_height_source)
        },
        "synth_height_cells": int(synth_height_cells.sum()),
        "flat_owner_fallback_cells": flat_owner_fallback_cells,
        "expected_counts": config.get("expected_counts_v1"),
    }
    _warn_drift(manifest)

    arrays = {
        "tam_h": fields["tam_h"],
        "oth_h": fields["oth_h"],
        "cell_owner": cell_owner,
        "cell_height_source": cell_height_source,
        "synth_height_cells": synth_height_cells,
        "gx0": np.int32(gx0),
        "gy0": np.int32(gy0),
    }
    return arrays, manifest


def _warn_drift(manifest: dict) -> None:
    expected = manifest.get("expected_counts")
    if not expected:
        return
    checks = [
        ("deleted_cells", expected.get("deleted")),
        ("retained_cells", expected.get("retained")),
        ("seam_edges", expected.get("seam_edges")),
        ("seam_tam_cells", expected.get("seam_tam_cells")),
    ]
    for actual_key, want in checks:
        if want is not None and manifest[actual_key] != want:
            print(f"DRIFT WARNING: {actual_key}={manifest[actual_key]} "
                  f"(expected_counts_v1 says {want}) — owner mods changed?")
    peaks = expected.get("peaks_gu") or {}
    for name, want in peaks.items():
        got = manifest["sources"].get(name, {}).get("peak_gu")
        if got is not None and want is not None and not math.isclose(got, want, rel_tol=0, abs_tol=0.5):
            print(f"DRIFT WARNING: peak_gu[{name}]={got} (expected {want})")


def save_corpus(arrays: dict[str, np.ndarray], manifest: dict,
                npz_path: str | Path, manifest_path: str | Path) -> None:
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def load_corpus(npz_path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = {
        "gx0": int(arrays.pop("gx0")),
        "gy0": int(arrays.pop("gy0")),
    }
    return arrays, meta


def seam_edges(cell_owner: np.ndarray, base_code: int, gy0: int, gx0: int
               ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """4-adjacency pairs (tam_cell, owner_cell) between retained base cells
    and cells finally owned by another source. ``base_code`` is the grid code
    of the base source (source index + 1; 0 = void)."""
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    ncy, ncx = cell_owner.shape
    retained = np.argwhere(cell_owner == base_code)
    for ry, rx in retained:
        cy, cx = ry + gy0, rx + gx0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx_, ny_ = rx + dx, ry + dy
            if 0 <= nx_ < ncx and 0 <= ny_ < ncy:
                other = int(cell_owner[ny_, nx_])
                if other != 0 and other != base_code:
                    edges.append(((cx, cy), (cx + dx, cy + dy)))
    return edges


def hillshade(heights: np.ndarray, azimuth_deg: float = 315.0,
              altitude_deg: float = 45.0, z_scale: float = 1.0) -> np.ndarray:
    """Lambertian hillshade of a GU height field (vertex spacing fixed).
    Returns float 0..1 with 0 where input is NaN."""
    h = np.where(np.isnan(heights), np.float32(0.0), heights) * np.float32(z_scale)
    dz_dy, dz_dx = np.gradient(h, np.float32(VERTEX_STEP_GU))
    nx_, ny_, nz_ = -dz_dx, -dz_dy, np.float32(1.0)
    norm = np.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_)
    az = np.float32(math.radians(azimuth_deg))
    alt = np.float32(math.radians(altitude_deg))
    lx = np.float32(math.cos(alt) * math.sin(az))
    ly = np.float32(math.cos(alt) * math.cos(az))
    lz = np.float32(math.sin(alt))
    shade = (nx_ * lx + ny_ * ly + nz_ * lz) / norm
    out = np.clip(shade, 0.0, 1.0)
    out[np.isnan(heights)] = 0.0
    return out


def composite_field(tam_h: np.ndarray, oth_h: np.ndarray) -> np.ndarray:
    """World-context view: base heights everywhere, owner heights where the
    base is void. Used so review crops keep identical framing pre/post solve."""
    out = np.array(tam_h, dtype=np.float32, copy=True)
    hole = np.isnan(out)
    out[hole] = oth_h[hole]
    return out


def hypsometric_rgb(heights: np.ndarray, shade: np.ndarray,
                    stops: list[list[float]],
                    void_gray: float = 0.45) -> np.ndarray:
    """Colorize heights through piecewise-linear GU stops, modulated by
    hillshade. ``stops`` = [[gu, r, g, b], ...] ascending; NaN -> flat gray."""
    hs = np.asarray(heights, dtype=np.float32)
    valid = ~np.isnan(hs)
    gu = np.where(valid, hs, stops[0][0])
    xs = np.array([s[0] for s in stops], dtype=np.float32)
    rgb = np.stack([np.interp(gu, xs, np.array([s[1] for s in stops], np.float32)),
                    np.interp(gu, xs, np.array([s[2] for s in stops], np.float32)),
                    np.interp(gu, xs, np.array([s[3] for s in stops], np.float32))],
                   axis=-1)
    lum = np.float32(0.35) + np.float32(0.65) * shade[..., None]
    rgb = rgb * lum
    rgb[~valid] = np.float32(void_gray * 255.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_split_window(tam_h: np.ndarray, oth_h: np.ndarray,
                        cell_owner: np.ndarray, base_code: int,
                        r_lo: int, r_hi: int, c_lo: int, c_hi: int,
                        render_cfg: dict, pad: int = 64) -> np.ndarray:
    """Ownership-split composite (tam heights on retained cells, owner
    heights on owner cells) + hillshade + hypsometric tint for one window.
    Returns an HxWx3 uint8 RGB array covering exactly [r_lo:r_hi, c_lo:c_hi]."""
    pr_lo = max(0, r_lo - pad)
    pr_hi = min(tam_h.shape[0], r_hi + pad)
    pc_lo = max(0, c_lo - pad)
    pc_hi = min(tam_h.shape[1], c_hi + pad)
    win = np.array(tam_h[pr_lo:pr_hi, pc_lo:pc_hi], dtype=np.float32, copy=True)
    cy0 = pr_lo // 64
    cy1 = min(cell_owner.shape[0], (pr_hi + 63) // 64)
    cx0 = pc_lo // 64
    cx1 = min(cell_owner.shape[1], (pc_hi + 63) // 64)
    own_cell = (cell_owner[cy0:cy1, cx0:cx1] != 0) & \
               (cell_owner[cy0:cy1, cx0:cx1] != base_code)
    own_v = np.repeat(np.repeat(own_cell, 64, axis=0), 64, axis=1)
    own_v = np.pad(own_v,
                   ((0, (cy1 - cy0) * 64 + 1 - own_v.shape[0]),
                    (0, (cx1 - cx0) * 64 + 1 - own_v.shape[1])), mode="edge")
    or0 = cy0 * 64 - pr_lo
    oc0 = cx0 * 64 - pc_lo
    own_v = own_v[or0:or0 + (pr_hi - pr_lo), oc0:oc0 + (pc_hi - pc_lo)]
    win[own_v] = oth_h[pr_lo:pr_hi, pc_lo:pc_hi][own_v]
    sh = hillshade(win,
                   azimuth_deg=render_cfg["azimuth_deg"],
                   altitude_deg=render_cfg["altitude_deg"],
                   z_scale=float(render_cfg["vertical_exaggeration"]))
    sh = sh[r_lo - pr_lo:r_hi - pr_lo, c_lo - pc_lo:c_hi - pc_lo]
    return hypsometric_rgb(win[r_lo - pr_lo:r_hi - pr_lo,
                               c_lo - pc_lo:c_hi - pc_lo], sh,
                           render_cfg["hypsometric_stops_gu"])


def save_shade_png(shade: np.ndarray, path: str | Path, px_per_vertex: int = 2,
                   overlay=None, title: str | None = None) -> Path:
    """Write a PNG from a 2D shade array (grayscale) or HxWx3 uint8 RGB."""
    if shade.ndim == 3:
        img = Image.fromarray(shade, mode="RGB")
    else:
        img = Image.fromarray((shade * 255).astype(np.uint8), mode="L")
    if px_per_vertex != 1:
        img = img.resize((img.width * px_per_vertex, img.height * px_per_vertex),
                         Image.Resampling.NEAREST)
    if shade.ndim == 2:
        img = Image.merge("RGB", (img, img, img))
    draw = ImageDraw.Draw(img)
    if overlay is not None:
        overlay(draw, px_per_vertex)
    # TES3 vertex rows run south->north; flip so renders are north-up like
    # standard Tamriel maps. Overlays were drawn in array space and flip
    # with the terrain; the title is drawn after the flip in screen space.
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if title:
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), title, fill=(255, 80, 80))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
