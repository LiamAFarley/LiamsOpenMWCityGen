"""Relief-response preview and invariant checks (v3 Milestone 1).

Purpose
    Apply the broad-relief transform (src/procgen/terrain_relief.py) to the
    full corpus field at the configured gains, run the invariant self-check,
    and emit the Milestone-1 review packet:

      - relief-response curve plot (gain vs broad elevation);
      - full-map hillshade previews at 1x / 2x / 3x;
      - tr_vvardenfell_wall region crops (original vs 3x);
      - metrics JSON (E0/E1, gain percentiles, underwater identity,
        fine-RMS retention, self-check results, timings).

Inputs
    --config JSON; corpus npz; seam atlas (region bbox for the crop).

Outputs
    paths.solve_out_dir / v3 / relief/ : preview PNGs, scaled field npz
    (max gain only), relief_metrics.json.

Pipeline position
    v3 Milestone 1; the max-gain scaled field becomes the seam-synthesis
    target for all later stages.

Invariants
    Verified here via terrain_relief.selfcheck; underwater terrain is
    bit-exact unchanged; output NaN pattern equals input NaN pattern.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen.terrainfield import (  # noqa: E402
    load_config, load_corpus, hillshade, hypsometric_rgb, save_shade_png,
    render_split_window,
)
from procgen import terrainstyle as ts  # noqa: E402
from procgen import terrain_relief as tr  # noqa: E402


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def downsample(field: np.ndarray, factor: int) -> np.ndarray:
    H = (field.shape[0] // factor) * factor
    W = (field.shape[1] // factor) * factor
    x = field[:H, :W]
    blocks = x.reshape(H // factor, factor, W // factor, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=(1, 3)).astype(np.float32)


def curve_plot(info_by_gain: dict, out_path: Path) -> None:
    W, H = 900, 520
    img = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(img)
    lo, hi = 0.0, max(8000.0, max(v["E_full_gu"] for v in info_by_gain.values()) * 1.1)
    def px(e): return int(70 + (W - 110) * e / hi)
    def py(g): return int(H - 60) - int((H - 100) * (g - 1.0) / 2.5)
    for gy in (1.0, 1.5, 2.0, 2.5, 3.0):
        d.line([(70, py(gy)), (W - 40, py(gy))], fill=(225, 225, 225))
        d.text((30, py(gy) - 6), f"{gy:.1f}x", fill=(90, 90, 90))
    for frac in (0.25, 0.5, 0.75, 1.0):
        e = lo + (hi - lo) * frac
        d.line([(px(e), 40), (px(e), H - 60)], fill=(240, 240, 240))
        d.text((px(e) - 14, H - 48), f"{int(e)}", fill=(90, 90, 90))
    colors = {2.0: (120, 160, 255), 3.0: (230, 90, 60)}
    for gain_key, info in info_by_gain.items():
        g = float(gain_key)
        g1 = float(info.get("gentle_gain", 1.5))
        eg = float(info["E_gentle_gu"])
        ef = float(info["E_full_gu"])
        def curve_gain(e):
            t1 = min(e / max(eg, 1e-6), 1.0)
            s1 = t1 ** 3 * (t1 * (6 * t1 - 15) + 10)
            val = 1.0 + (g1 - 1.0) * s1
            t2 = min(max((e - eg) / max(ef - eg, 1e-6), 0.0), 1.0)
            s2 = t2 ** 3 * (t2 * (6 * t2 - 15) + 10)
            return val + (g - g1) * s2
        pts = []
        for i in range(201):
            e = hi * i / 200.0
            pts.append((px(e), py(min(curve_gain(e), 3.2))))
        d.line(pts, fill=colors.get(g, (60, 60, 60)), width=3)
        d.text((px(min(ef, hi)) + 4, py(g) - 14),
               f"{g:.0f}x (gentle->{int(eg)}, full->{int(ef)})",
               fill=colors.get(g, (60, 60, 60)))
    d.text((70, 12), "relief response: gain vs broad elevation D (GU)", fill=(30, 30, 30))
    img.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tamriel_reworked_v1.json"))
    ap.add_argument("--gains", default="2,3")
    ap.add_argument("--region", default="tr_vvardenfell_wall")
    args = ap.parse_args()
    cfg = load_config(_resolve(ROOT, args.config))
    rcfg = cfg["render"]
    relief_cfg = dict(cfg.get("terrain_relief", {}))
    outdir = _resolve(ROOT, cfg["paths"]["solve_out_dir"]) / "v3" / "relief"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    check = tr.selfcheck(relief_cfg)
    print("selfcheck:", json.dumps({k: v for k, v in check.items() if k != "info"}))
    if not check["all_pass"]:
        print("FAILURE: relief invariants failed; see above")
        return 1

    arrays, meta = load_corpus(_resolve(ROOT, cfg["paths"]["corpus_npz"]))
    with open(_resolve(ROOT, cfg["paths"]["corpus_manifest"]), encoding="utf-8") as fh:
        manifest = json.load(fh)
    base_code = manifest["source_names"].index(manifest["base_source"]) + 1
    tam_h = arrays["tam_h"]
    oth_view_full = np.where(np.isfinite(arrays["oth_h"]), arrays["oth_h"],
                             tam_h).astype(np.float32)

    with open(_resolve(ROOT, cfg["paths"]["seam_atlas_json"]), encoding="utf-8") as fh:
        atlas = json.load(fh)
    region = cfg["solve"]["regions"][args.region]
    atlas_by_id = {r["cluster"]: r for r in atlas["clusters"]}
    xs, ys = [], []
    for cid in region["cluster_ids"]:
        x0, y0, x1, y1 = atlas_by_id[cid]["bbox_cells_xyxy"]
        xs += [x0, x1]; ys += [y0, y1]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
    gy0, gx0 = meta["gy0"], meta["gx0"]
    mm = 6 * 64
    wr_lo = max(0, (by0 - gy0) * 64 - mm)
    wr_hi = min(tam_h.shape[0], (by1 - gy0) * 64 + 65 + mm)
    wc_lo = max(0, (bx0 - gx0) * 64 - mm)
    wc_hi = min(tam_h.shape[1], (bx1 - gx0) * 64 + 65 + mm)
    ppv = int(rcfg["px_per_vertex"])

    before = render_split_window(tam_h, oth_view_full, arrays["cell_owner"],
                                 base_code, wr_lo, wr_hi, wc_lo, wc_hi, rcfg)
    save_shade_png(before, outdir / f"{args.region}_relief_before.png", ppv,
                   title=f"{args.region} BEFORE relief (1x)")

    gains = [float(g) for g in args.gains.split(",")]
    info_by_gain = {}
    metrics = {"selfcheck": {k: v for k, v in check.items() if k != "info"},
               "selfcheck_info": check.get("info", {}), "gains": {}}
    scaled_max = None
    for g in gains:
        tg = time.time()
        scaled, info = tr.relief_scale(tam_h, {**relief_cfg, "max_gain": g})
        info_by_gain[g] = info
        uw_err = info.get("underwater_max_delta_gu", 0.0)
        print(f"  gain {g:.0f}x: gentle_end={info['E_gentle_gu']} "
              f"full_end={info['E_full_gu']} gentle_gain={info['gentle_gain']} "
              f"underwater_err={uw_err} fine_rms_ratio={info.get('fine_rms_ratio')} "
              f"({time.time() - tg:.1f}s)")
        metrics["gains"][str(g)] = info
        ds = downsample(scaled, 6)
        sh = hillshade(ds, azimuth_deg=rcfg["azimuth_deg"],
                       altitude_deg=rcfg["altitude_deg"],
                       z_scale=float(rcfg["vertical_exaggeration"]))
        rgb = hypsometric_rgb(ds, sh, rcfg["hypsometric_stops_gu"])
        save_shade_png(rgb, outdir / f"fullmap_relief_{int(g)}x.png", 1,
                       title=f"FULL MAP relief {int(g)}x (1px = {6 * 128} GU)")
        if abs(g - float(relief_cfg.get("max_gain", 3.0))) < 1e-6:
            scaled_max = scaled
        del scaled

    crop = render_split_window(scaled_max, oth_view_full, arrays["cell_owner"],
                               base_code, wr_lo, wr_hi, wc_lo, wc_hi, rcfg)
    save_shade_png(crop, outdir / f"{args.region}_relief_{int(relief_cfg['max_gain'])}x.png",
                   ppv, title=f"{args.region} AFTER relief ({int(relief_cfg['max_gain'])}x)")
    np.savez_compressed(outdir / "relief_scaled_field.npz", field=scaled_max,
                        gx0=np.int32(meta["gx0"]), gy0=np.int32(meta["gy0"]))
    curve_plot(info_by_gain, outdir / "relief_response_curve.png")

    metrics["timings"] = {"total_s": round(time.time() - t0, 1)}
    with open(outdir / "relief_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=lambda o: int(o))
    print(f"wrote {outdir} (total {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
