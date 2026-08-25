"""v3 seam synthesis driver — Milestone 2 (surface solve, no erosion).

Purpose
    Thin driver for the v3 constrained surface solve on one seam region:
    builds the shared context (terrain_blend.build_context) against the
    relief-amplified target, solves the screened-Poisson surface with
    owner normal-slope rows, reports seam metrics (terrain_metrics), and
    renders the review set: target-before crop, solved-after crop, owner
    reference crop, and a stacked comparison.

Inputs
    --config JSON, --region (under solve.regions); corpus npz + manifest,
    seam atlas, relief-scaled field npz (solve.v3.relief_npz).

Outputs (under paths.solve_out_dir / v3 /)
    ``<region>_v3_target.png`` (pre-solve amplified target)
    ``<region>_v3_after.png`` / ``_wide.png``
    ``<region>_v3_reference.png`` / ``_v3_comparison.png``
    ``<region>_v3_metrics.json`` / ``_v3_field.npz``

Pipeline position
    v3 Milestone 2 of tamriel-reworked-heightmap. Writes only its own
    output directory; no erosion happens in this stage.

Invariants
    Seam verts exactly equal own_view; owner cells never modified;
    deterministic (CG on a fixed SPD system, no RNG).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen.terrainfield import (  # noqa: E402
    load_config, render_split_window, save_shade_png,
)
from procgen.terrain_blend import build_context, solve_surface  # noqa: E402
from procgen import terrain_metrics as tmet  # noqa: E402


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tamriel_reworked_v1.json"))
    ap.add_argument("--region", default="tr_vvardenfell_wall")
    args = ap.parse_args()
    cfg = load_config(_resolve(ROOT, args.config))
    t0 = time.time()

    tc = time.time()
    ctx = build_context(ROOT, cfg, args.region)
    t_context = time.time() - tc

    ts_ = time.time()
    v3 = cfg["solve"].get("v3", {})
    field, sinfo = solve_surface(ctx, v3)
    t_solve = time.time() - ts_

    tm = time.time()
    c0 = tmet.seam_c0(field, ctx["own_view"], ctx["seam_v"])
    c1 = tmet.seam_c1_normals(field, ctx["own_view"], ctx["seam_v"],
                              ctx["nx"], ctx["ny"])
    curv = tmet.curvature_jump(field, ctx["own_view"], ctx["seam_v"],
                               ctx["nx"], ctx["ny"])
    t_metrics = time.time() - tm

    outdir = _resolve(ROOT, cfg["paths"]["solve_out_dir"]) / "v3"
    outdir.mkdir(parents=True, exist_ok=True)
    r_lo, r_hi, c_lo, c_hi = ctx["win"]
    by0, by1 = ctx["bbox"][1], ctx["bbox"][3]
    bx0, bx1 = ctx["bbox"][0], ctx["bbox"][2]
    gy0, gx0 = ctx["gy0"], ctx["gx0"]
    mm = int(ctx["region"].get("review_margin_cells", 6)) * 64
    wr_lo = max(0, (by0 - gy0) * 64 - mm)
    wr_hi = min(ctx["tam_h"].shape[0], (by1 - gy0) * 64 + 65 + mm)
    wc_lo = max(0, (bx0 - gx0) * 64 - mm)
    wc_hi = min(ctx["tam_h"].shape[1], (bx1 - gx0) * 64 + 65 + mm)
    rcfg = ctx["render"]
    ppv = int(rcfg["px_per_vertex"])
    oth_view_full = np.where(np.isfinite(ctx["oth_h"]), ctx["oth_h"],
                             ctx["tam_h"]).astype(np.float32)

    def to_full(win_field: np.ndarray) -> np.ndarray:
        full = ctx["target_full"].copy()
        rl, rh, cl, ch = ctx["win"]
        full[rl:rh, cl:ch] = win_field
        return full

    target_full = to_full(ctx["target"])
    field_full = to_full(field)
    edge_error = tmet.band_edge_continuity(field, ctx["target"], ctx["smask"],
                                           ctx["seam_v"])
    edge_tol = float(v3.get("band_edge_tolerance_gu", 1e-3))
    if edge_error > edge_tol:
        raise AssertionError(f"band-edge continuity {edge_error:.6g} GU > {edge_tol} GU")

    tgt = render_split_window(target_full, oth_view_full, ctx["cell_owner"],
                              ctx["base_code"], wr_lo, wr_hi, wc_lo, wc_hi, rcfg)
    save_shade_png(tgt, outdir / f"{args.region}_v3_target.png", ppv,
                   title=f"{args.region} TARGET (relief-scaled, pre-solve)")
    aft = render_split_window(field_full, oth_view_full, ctx["cell_owner"],
                              ctx["base_code"], wr_lo, wr_hi, wc_lo, wc_hi, rcfg)
    save_shade_png(aft, outdir / f"{args.region}_v3_after.png", ppv,
                   title=f"{args.region} v3 surface (no erosion)")
    wide = 12 * 64
    a2 = render_split_window(field_full, oth_view_full, ctx["cell_owner"],
                             ctx["base_code"],
                             max(0, wr_lo - wide), min(ctx["tam_h"].shape[0], wr_hi + wide),
                             max(0, wc_lo - wide), min(ctx["tam_h"].shape[1], wc_hi + wide),
                             rcfg)
    save_shade_png(a2, outdir / f"{args.region}_v3_wide.png", max(1, ppv // 2),
                   title=f"{args.region} v3 wide")

    dist_seam = ctx["dist_seam"]
    ref_img = render_split_window(ctx["tam_h"], oth_view_full,
                                  ctx["cell_owner"], ctx["base_code"],
                                  wr_lo, wr_hi, wc_lo, wc_hi, rcfg)
    save_shade_png(ref_img, outdir / f"{args.region}_v3_reference.png", ppv,
                   title=f"{args.region} OWNER REFERENCE (same frame, TR side)")
    a_img = Image.open(outdir / f"{args.region}_v3_after.png")
    b_img = Image.open(outdir / f"{args.region}_v3_reference.png")
    combo = Image.new("RGB", (max(a_img.width, b_img.width),
                              a_img.height + b_img.height + 60), (0, 0, 0))
    combo.paste(b_img, (0, 30))
    combo.paste(a_img, (0, a_img.height + 60))
    dd = ImageDraw.Draw(combo)
    dd.text((10, 7), "OWNER REFERENCE (same frame, TR side)", fill=(120, 220, 255))
    dd.text((10, a_img.height + 37), "OURS (v3 surface, no erosion)",
            fill=(255, 215, 100))
    combo.save(outdir / f"{args.region}_v3_comparison.png")

    seam_ys, seam_xs = np.nonzero(ctx["seam_v"])
    seam_margin = int(ctx["region"].get("seam_crop_margin_cells", 2)) * 64
    # Regions may contain several disconnected seam runs.  A review crop can
    # select one authoritative wall bbox in cell coordinates without changing
    # the standard full comparison framing.
    crop_bbox = ctx["region"].get("seam_crop_bbox_cells")
    if crop_bbox:
        cbx0, cby0, cbx1, cby1 = map(int, crop_bbox)
        seam_r0 = (cby0 - ctx["gy0"]) * 64
        seam_r1 = (cby1 - ctx["gy0"]) * 64 + 65
        seam_c0 = (cbx0 - ctx["gx0"]) * 64
        seam_c1 = (cbx1 - ctx["gx0"]) * 64 + 65
    else:
        seam_r0, seam_r1 = int(seam_ys.min()), int(seam_ys.max()) + 1
        seam_c0, seam_c1 = int(seam_xs.min()), int(seam_xs.max()) + 1
    crop = render_split_window(
        field_full, oth_view_full, ctx["cell_owner"], ctx["base_code"],
        max(0, seam_r0 - seam_margin),
        min(ctx["tam_h"].shape[0], seam_r1 + seam_margin),
        max(0, seam_c0 - seam_margin),
        min(ctx["tam_h"].shape[1], seam_c1 + seam_margin), rcfg)
    save_shade_png(crop, outdir / f"{args.region}_v3_seam_zoom.png", 4,
                   title=f"{args.region} v3 seam zoom (4 px/vertex)")

    metrics = {
        "region": args.region,
        "surface": sinfo,
        "seam_c0_max_gu": c0,
        "seam_c1_normals": c1,
        "curvature_jump": curv,
        "band_edge_max_abs_gu": edge_error,
        "timings": {"context_s": round(t_context, 1),
                    "solve_s": round(t_solve, 1),
                    "metrics_s": round(t_metrics, 1),
                    "total_s": round(time.time() - t0, 1)},
    }
    with open(outdir / f"{args.region}_v3_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=lambda o: int(o))
    np.savez_compressed(outdir / f"{args.region}_v3_field.npz", field=field_full,
                        smask=ctx["smask"],
                        win=np.array(ctx["win"], dtype=np.int32),
                        gx0=np.int32(ctx["gx0"]), gy0=np.int32(ctx["gy0"]))
    for kk, vv in metrics.items():
        print(f"  {kk}: {vv}")
    print(f"wrote {outdir} (total {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
