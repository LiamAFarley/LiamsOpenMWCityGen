"""Run targeted MFD erosion and the narrow post-erosion seam lock.

The input is the real solved v3 field for one configured region. This command
does not regenerate the harmonic bridge; it routes over that bridge plus a
fixed owner halo, writes cycle snapshots, then applies a fresh 1-2 cell direct
harmonic lock at the owner seam.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen.terrain_blend import solve_surface  # noqa: E402
from procgen.terrain_erosion import erode_field  # noqa: E402
from procgen.terrainfield import (  # noqa: E402
    load_config,
    render_split_window,
    save_shade_png,
)
from procgen import terrain_metrics as tmet  # noqa: E402
from procgen.terrain_blend import build_context  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _owner_vertex_mask(ctx: dict) -> np.ndarray:
    """Expand authoritative owner cells to vertices, including shared edges."""
    r0, r1, c0, c1 = map(int, ctx["win"])
    gy0, gx0 = int(ctx["gy0"]), int(ctx["gx0"])
    source = ctx["cell_height_source"]
    base_code = int(ctx["base_code"])
    if r0 % 64 or c0 % 64:
        raise ValueError("erosion currently requires cell-aligned context windows")
    cy0 = r0 // 64 + gy0
    cx0 = c0 // 64 + gx0
    ncy = (r1 - r0) // 64
    ncx = (c1 - c0) // 64
    sy0, sx0 = cy0 - gy0, cx0 - gx0
    cells = source[sy0:sy0 + ncy, sx0:sx0 + ncx]
    owner_cells = (cells != 0) & (cells != base_code)
    quads = np.repeat(np.repeat(owner_cells, 64, axis=0), 64, axis=1)
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    mask[:-1, :-1] |= quads
    mask[1:, :-1] |= quads
    mask[:-1, 1:] |= quads
    mask[1:, 1:] |= quads
    return mask


def _final_seam_lock(ctx: dict, eroded: np.ndarray, cfg: dict):
    """Solve a narrow direct Laplace correction over the eroded field."""
    v3 = cfg["solve"].get("v3", {})
    ecfg = cfg.get("erosion", {})
    band_cells = float(ecfg.get("final_lock_cells", 2.0))
    active = ctx["smask"] & (ctx["dist_seam"] <= band_cells * 64.0)
    active |= ctx["seam_v"]
    labels, _ = ndimage.label(
        active, structure=ndimage.generate_binary_structure(2, 1)
    )
    seam_labels = np.unique(labels[ctx["seam_v"]])
    active = np.isin(labels, seam_labels)
    interior = ndimage.binary_erosion(
        active, structure=ndimage.generate_binary_structure(2, 1), border_value=0
    )
    ring = active & ~ctx["seam_v"] & ~interior
    hard = ctx["seam_v"] | ring
    hard_vals = eroded.copy()
    hard_vals[ctx["seam_v"]] = ctx["own_view"][ctx["seam_v"]]
    final_ctx = dict(ctx)
    final_ctx.update(
        target=eroded,
        tam_h=eroded,
        smask=active,
        ring_v=ring,
        hard=hard,
        hard_vals=hard_vals,
    )
    final_v3 = dict(v3)
    final_v3["surface"] = dict(v3.get("surface", {}))
    field, report = solve_surface(final_ctx, final_v3)
    report["final_lock_cells"] = band_cells
    report["final_lock_active_vertices"] = int(active.sum())
    report["final_lock_boundary_vertices"] = int(ring.sum())
    return field, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    ctx = build_context(ROOT, cfg, args.region)
    solve_dir = _resolve(cfg["paths"]["solve_out_dir"]) / "v3"
    field_path = solve_dir / f"{args.region}_v3_field.npz"
    if not field_path.exists():
        print(f"FAILURE: harmonic field missing: {field_path}")
        return 1
    with np.load(field_path) as z:
        full = z["field"].astype(np.float32)
    r0, r1, c0, c1 = map(int, ctx["win"])
    field = full[r0:r1, c0:c1].copy()

    ecfg = dict(cfg.get("erosion", {}))
    owner_v = _owner_vertex_mask(ctx)
    halo_cells = float(ecfg.get("owner_halo_cells", 6.0))
    owner_halo = owner_v & (
        ndimage.distance_transform_edt(~ctx["smask"]) <= halo_cells * 64.0
    )
    fixed = ctx["seam_v"] | ctx["ring_v"] | owner_halo
    out_dir = solve_dir / "erosion"
    out_dir.mkdir(parents=True, exist_ok=True)
    rcfg = ctx["render"]
    review_bbox = ctx["region"].get("review_bbox_cells")
    if review_bbox:
        bx0, by0, bx1, by1 = map(int, review_bbox)
        wr_lo = max(0, (by0 - ctx["gy0"]) * 64)
        wr_hi = min(ctx["tam_h"].shape[0], (by1 - ctx["gy0"]) * 64 + 65)
        wc_lo = max(0, (bx0 - ctx["gx0"]) * 64)
        wc_hi = min(ctx["tam_h"].shape[1], (bx1 - ctx["gx0"]) * 64 + 65)
    else:
        wr_lo, wr_hi, wc_lo, wc_hi = r0, r1, c0, c1
    ppv = int(rcfg["px_per_vertex"])

    def snapshot(cycle: int, snapshot_field: np.ndarray):
        full_snapshot = full.copy()
        full_snapshot[r0:r1, c0:c1] = snapshot_field
        rgb = render_split_window(
            full_snapshot, ctx["own_full"], ctx["cell_owner"],
            ctx["base_code"], wr_lo, wr_hi, wc_lo, wc_hi, rcfg
        )
        save_shade_png(
            rgb, out_dir / f"{args.region}_erosion_cycle_{cycle:02d}.png",
            ppv, title=f"{args.region} erosion cycle {cycle}"
        )

    eroded, erosion_report = erode_field(
        field, ctx["smask"], owner_halo, fixed, ecfg,
        snapshot_callback=snapshot,
    )
    locked, lock_report = _final_seam_lock(ctx, eroded, cfg)
    full_locked = full.copy()
    full_locked[r0:r1, c0:c1] = locked
    after = render_split_window(
        full_locked, ctx["own_full"], ctx["cell_owner"], ctx["base_code"],
        wr_lo, wr_hi, wc_lo, wc_hi, rcfg
    )
    save_shade_png(
        after, out_dir / f"{args.region}_erosion_final.png", ppv,
        title=f"{args.region} erosion + final seam lock"
    )
    seam_bbox = ctx["region"].get("seam_crop_bbox_cells")
    if seam_bbox:
        sx0, sy0, sx1, sy1 = map(int, seam_bbox)
        margin = int(ctx["region"].get("seam_crop_margin_cells", 2))
        sr0 = max(0, (sy0 - margin - ctx["gy0"]) * 64)
        sr1 = min(ctx["tam_h"].shape[0], (sy1 + margin - ctx["gy0"]) * 64 + 65)
        sc0 = max(0, (sx0 - margin - ctx["gx0"]) * 64)
        sc1 = min(ctx["tam_h"].shape[1], (sx1 + margin - ctx["gx0"]) * 64 + 65)
        seam_zoom = render_split_window(
            full_locked, ctx["own_full"], ctx["cell_owner"], ctx["base_code"],
            sr0, sr1, sc0, sc1, rcfg
        )
        save_shade_png(
            seam_zoom, out_dir / f"{args.region}_erosion_seam_zoom.png", ppv,
            title=f"{args.region} erosion seam zoom"
        )
    seam_c0 = tmet.seam_c0(locked, ctx["own_view"], ctx["seam_v"])
    c1 = tmet.seam_c1_normals(
        locked, ctx["own_view"], ctx["seam_v"], ctx["nx"], ctx["ny"]
    )
    report = {
        "region": args.region,
        "owner_halo_cells": halo_cells,
        "erosion": erosion_report,
        "final_lock": lock_report,
        "seam_c0_max_gu": seam_c0,
        "seam_c1_normals": c1,
        "timings": {"total_s": round(time.time() - t0, 1)},
    }
    with open(out_dir / f"{args.region}_erosion_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=lambda value: int(value))
    np.savez_compressed(out_dir / f"{args.region}_erosion_field.npz", field=full_locked)
    print(json.dumps(report, indent=2, default=lambda value: int(value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
