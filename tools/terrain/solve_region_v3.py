"""v3 seam synthesis driver — Milestone 2 rev2 (surface solve, no erosion).

Purpose
    Thin driver for the revised v3 constrained surface solve (Sol High
    review sequence, steps 2-4/7/9): builds the shared context
    (terrain_blend.build_context — edge-rasterized seam, corridor seeds,
    per-edge slope constraints), solves the screened-Poisson surface
    through the family-tagged equation assembler, enforces the quality
    gates (C0 exactness, band-edge continuity, max first-edge drop, slope
    residual), and renders the review set including a zoomed wall crop and
    seam-normal height profiles.

Inputs
    --config JSON, --region (under solve.regions); corpus npz + manifest,
    seam atlas, relief-scaled field npz (staleness-guarded).

Outputs (under paths.solve_out_dir / v3 /)
    ``<region>_v3_target.png`` / ``_after.png`` / ``_wide.png`` /
    ``_reference.png`` / ``_comparison.png`` / ``_seam_zoom.png`` /
    ``_v3_metrics.json`` / ``_v3_field.npz``

Pipeline position
    v3 Milestone 2 of tamriel-reworked-heightmap. Writes only its own
    output directory; no erosion happens in this stage.

Invariants
    Seam verts exactly equal own_view (eliminated boundary rows);
    band-edge continuity vs the scaled target within tolerance; owner
    cells never modified; deterministic.
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
from procgen.terrain_blend import build_context, load_target, solve_surface  # noqa: E402
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
    quality = v3.get("quality", {})
    prof = tmet.normal_profiles(field, ctx["own_view"], ctx["seam_v"],
                                ctx["nx"], ctx["ny"],
                                count=int(quality.get("profile_count", 64)),
                                in_verts=int(quality.get("profile_in_verts", 16)),
                                out_verts=int(quality.get("profile_out_verts", 8)))
    t_metrics = time.time() - tm

    # scaled-field canvas (band edge must be continuous with THIS)
    canvas = load_target(ROOT, cfg, ctx["tam_h"])
    r_lo, r_hi, c_lo, c_hi = ctx["win"]

    def to_full(win_field: np.ndarray) -> np.ndarray:
        full = canvas.copy()
        full[r_lo:r_hi, c_lo:c_hi] = win_field
        return full

    field_full = to_full(field)
    target_full = to_full(ctx["target"])

    edge_error = tmet.band_edge_continuity(field, ctx["target"], ctx["smask"],
                                           ctx["seam_v"])
    edge_tol = float(v3.get("band_edge_tolerance_gu", 1e-3))
    slope_res_fail = float(quality.get("slope_residual_fail_gu", 200.0))
    drop_fail = float(quality.get("max_first_edge_drop_gu", 2500.0))
    slope_res = sinfo["residuals"].get("slope_rms", 0.0)

    failures = []
    if c0 > 1e-3:
        failures.append(f"seam C0 {c0:.1f} GU != 0")
    if edge_error > edge_tol:
        failures.append(f"band-edge continuity {edge_error:.3g} > {edge_tol}")
    if slope_res > slope_res_fail:
        failures.append(f"slope residual RMS {slope_res:.1f} > {slope_res_fail}")
    max_drop = prof["max_first_edge_drop_gu"]
    if max_drop > drop_fail:
        failures.append(f"max first-edge drop {max_drop:.0f} > {drop_fail}")

    outdir = _resolve(ROOT, cfg["paths"]["solve_out_dir"]) / "v3"
    outdir.mkdir(parents=True, exist_ok=True)
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

    # zoomed seam crop (config-selectable wall bbox, default full seam extent)
    seam_ys, seam_xs = np.nonzero(ctx["seam_v"])
    seam_margin = int(ctx["region"].get("seam_crop_margin_cells", 2)) * 64
    crop_bbox = ctx["region"].get("seam_crop_bbox_cells")
    if crop_bbox:
        cbx0, cby0, cbx1, cby1 = map(int, crop_bbox)
        sr0 = (cby0 - ctx["gy0"]) * 64
        sr1 = (cby1 - ctx["gy0"]) * 64 + 65
        sc0 = (cbx0 - ctx["gx0"]) * 64
        sc1 = (cbx1 - ctx["gx0"]) * 64 + 65
    else:
        sr0, sr1 = int(seam_ys.min()), int(seam_ys.max()) + 1
        sc0, sc1 = int(seam_xs.min()), int(seam_xs.max()) + 1
    crop = render_split_window(
        field_full, oth_view_full, ctx["cell_owner"], ctx["base_code"],
        max(0, sr0 - seam_margin), min(ctx["tam_h"].shape[0], sr1 + seam_margin),
        max(0, sc0 - seam_margin), min(ctx["tam_h"].shape[1], sc1 + seam_margin),
        rcfg)
    save_shade_png(crop, outdir / f"{args.region}_v3_seam_zoom.png", 4,
                   title=f"{args.region} v3 seam zoom (4 px/vertex)")

    metrics = {
        "region": args.region,
        "surface": sinfo,
        "seam_c0_max_gu": c0,
        "seam_c1_normals": c1,
        "curvature_jump": curv,
        "band_edge_max_abs_gu": edge_error,
        "max_first_edge_drop_gu": max_drop,
        "normal_profiles": prof["profiles"],
        "quality_failures": failures,
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
        if kk != "normal_profiles":
            print(f"  {kk}: {vv}")
    print(f"wrote {outdir} (total {time.time() - t0:.1f}s)")
    if failures:
        print("FAILURE: quality gates violated:")
        for f in failures:
            print(f"  - {f}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
