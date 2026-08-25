"""Solve one seam region: rebuild base-side terrain to continue owner terrain.

Purpose
    Regional prototype of the tamriel-reworked blend solve. For a configured
    set of atlas clusters, rebuild the base (tamriel) heights inside a blend
    band so that:
      - seam vertices match owner heights exactly (Dirichlet),
      - the outer band edge relaxes to untouched base ambient (Dirichlet),
      - the interior is a harmonic (Laplace) relaxation between the two,
        solved on a coarse grid (``coarse_factor`` block reduction) and
        bilinearly upsampled — the harmonic base is smooth by construction,
      - style-matched fractal detail (fBm + ridged, amplitude driven by local
        slope, scaled to the owner's measured roughness) is added with a
        distance-based fade from every boundary,
      - a D8 flow-accumulation carve (descending order, steepest descent)
        cuts dendritic valleys so the result reads as dissected terrain.

Inputs
    --config JSON, --region key under ``solve.regions`` (cluster id list),
    corpus npz + seam atlas json + corpus manifest.

Outputs (under paths.solve_out_dir)
    ``<region>_field.npz``  full-size patched base field + solve mask
    ``<region>_manifest.json``  audit (seam exactness, changed verts, stats)
    ``<region>_before.png`` / ``_after.png`` / ``_after_wide.png``  review
    renders in the same ownership-split framing as the Stage B crops.

Pipeline position
    Prototype for Stage D of tamriel-reworked-heightmap. Writes only its own
    output directory; corpus and plugins stay untouched.

Invariants
    Deterministic (single seeded RNG); zero writes outside the solve mask;
    seam vertices exactly equal owner heights before THU quantization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen.terrainfield import (  # noqa: E402
    load_config,
    load_corpus,
    seam_edges,
    render_split_window,
    save_shade_png,
)


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def fbm_noise(shape: tuple[int, int], cfg: dict, rng: np.random.Generator
              ) -> np.ndarray:
    """Bilinear-upsampled value-noise fBm in [-1,1] mixed with ridged crests."""
    total = np.zeros(shape, dtype=np.float32)
    ridge = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    norm = 0.0
    for o in range(int(cfg["octaves"])):
        g = 4 * (2 ** o)
        grid = rng.random((g + 1, g + 1), dtype=np.float32)
        zoomed = ndimage.zoom(grid, (shape[0] / g, shape[1] / g),
                              order=1)[:shape[0], :shape[1]]
        total += amp * zoomed
        r = 1.0 - np.abs(2.0 * zoomed - 1.0)
        ridge += amp * (r * r)
        norm += amp
        amp *= float(cfg["persistence"])
    total /= norm
    ridge /= norm
    w = float(cfg["ridge_weight"])
    return ((1.0 - w) * (total - 0.5) + w * (ridge - 0.5)) * 2.0


def coarse_laplace(smask: np.ndarray, known: np.ndarray, tam_w: np.ndarray,
                   ds: int) -> np.ndarray:
    """Harmonic base on a ds-fold block-reduced grid; returns the upsampled
    full-resolution base (Dirichlet vertices re-imposed by the caller)."""
    H, W = smask.shape
    Hc = -(-H // ds)
    Wc = -(-W // ds)
    pad_h = Hc * ds - H
    pad_w = Wc * ds - W
    sm = np.pad(smask.astype(np.float32), ((0, pad_h), (0, pad_w)), mode="edge")
    kn = np.pad(known, ((0, pad_h), (0, pad_w)), mode="edge")
    tw = np.pad(tam_w, ((0, pad_h), (0, pad_w)), mode="edge")

    def blocks(a):
        return a.reshape(Hc, ds, Wc, ds)

    smask_c = blocks(sm).mean(axis=(1, 3))
    kn_valid = np.isfinite(kn)
    kn_count = blocks(kn_valid.astype(np.float32)).sum(axis=(1, 3))
    kn_sum = np.where(kn_valid, kn, 0.0)
    kn_sum = blocks(kn_sum).sum(axis=(1, 3))
    tw_mean = blocks(np.where(np.isfinite(tw), tw, 0.0)).sum(axis=(1, 3)) / \
        np.maximum(blocks(np.isfinite(tw).astype(np.float32)).sum(axis=(1, 3)), 1.0)

    fixed_val = np.where(kn_count > 0, kn_sum / np.maximum(kn_count, 1e-9),
                         tw_mean)
    unk_c = (smask_c >= 0.999) & (kn_count == 0)

    idx = -np.ones((Hc, Wc), dtype=np.int64)
    idx[unk_c] = np.arange(int(unk_c.sum()))
    n = int(unk_c.sum())
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    for r in range(Hc):
        for c in range(Wc):
            if not unk_c[r, c]:
                continue
            i = idx[r, c]
            acc = 0.0
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if rr < 0 or rr >= Hc or cc < 0 or cc >= Wc:
                    continue            # zero-flux (Neumann) edge
                if unk_c[rr, cc]:
                    rows.append(i); cols.append(idx[rr, cc]); vals.append(-1.0)
                else:
                    acc += fixed_val[rr, cc]
            rows.append(i); cols.append(i); vals.append(4.0)
            rhs.append(acc)
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    sol = spsolve(mat, np.asarray(rhs, dtype=float))
    coarse = fixed_val.copy()
    coarse[unk_c] = sol.reshape(-1)
    up = ndimage.zoom(coarse, ds, order=3)[:H, :W]
    return up.astype(np.float32)


def d8_carve(field: np.ndarray, smask: np.ndarray, coef_gu: float,
             fade_verts: float, max_gu: float, slope_min: float,
             eps_seed: int, accum_cap: float) -> np.ndarray:
    """Descending-order D8 flow accumulation carve, faded near boundaries,
    depth-capped and slope-gated so plains are not trenched."""
    rng = np.random.default_rng(eps_seed)
    h = field + rng.uniform(0.0, 0.25, field.shape).astype(np.float32)
    h[~smask] = np.inf
    H, W = h.shape
    index = np.arange(H * W, dtype=np.int64).reshape(H, W)
    recv = np.full((H, W), -1, dtype=np.int64)
    best = np.full((H, W), np.float32(1e30))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
        sh_ = np.roll(h, (-dy, -dx), axis=(0, 1))
        dist = 128.0 * (1.0 if dx == 0 or dy == 0 else 2.0 ** 0.5)
        with np.errstate(invalid="ignore"):
            slope_to = (h - sh_) / np.float32(dist)
        take = (slope_to > 0) & (slope_to < best) & np.isfinite(sh_)
        shifted = np.roll(index, (-dy, -dx), axis=(0, 1))
        recv = np.where(take, shifted, recv)
        best = np.where(take, slope_to, best)

    flat = h.ravel()
    recv_flat = recv.ravel()
    accum = np.ones(flat.size, dtype=np.float32)
    order = np.argsort(flat)[::-1]          # descending: donors before receivers
    for i in order:
        if not np.isfinite(flat[i]):
            continue
        j = recv_flat[i]
        if j >= 0 and np.isfinite(flat[j]):
            accum[j] += accum[i]
    accum = accum.reshape(H, W)
    accum = np.minimum(accum, float(accum_cap))
    gy, gx = np.gradient(field, 128.0)
    slope = np.sqrt(gx * gx + gy * gy) * 128.0
    slope = np.where(np.isfinite(slope), slope, 0.0)
    gate = np.clip((slope - slope_min) / 100.0, 0.0, 1.0)
    dist_in = ndimage.distance_transform_edt(smask)
    fade = np.clip(dist_in / fade_verts, 0.0, 1.0)
    gate = np.clip((slope - slope_min) / 100.0, 0.0, 1.0)
    gate = gate * (recv >= 0)              # don't dig pits at flow sinks
    carve = np.minimum(coef_gu * np.log2(1.0 + accum), max_gu)
    return field - carve * gate * fade * smask


def _health(name: str, a: np.ndarray, mask: np.ndarray) -> None:
    m = mask & np.isfinite(a)
    if not m.any():
        print(f"  [health] {name}: NO FINITE VALUES in mask")
        return
    v = a[m]
    print(f"  [health] {name}: finite {int(m.sum())}/{int(mask.sum())} "
          f"min={v.min():.1f} max={v.max():.1f}")


def solve_region(cfg: dict, region_name: str) -> int:
    t0 = time.time()
    arrays, meta = load_corpus(_resolve(ROOT, cfg["paths"]["corpus_npz"]))
    with open(_resolve(ROOT, cfg["paths"]["seam_atlas_json"]), encoding="utf-8") as fh:
        atlas = json.load(fh)
    with open(_resolve(ROOT, cfg["paths"]["corpus_manifest"]), encoding="utf-8") as fh:
        manifest = json.load(fh)
    names = manifest["source_names"]
    base_code = names.index(manifest["base_source"]) + 1

    tam_h = arrays["tam_h"]
    oth_h = arrays["oth_h"]
    cell_owner = arrays["cell_owner"]
    gy0, gx0 = meta["gy0"], meta["gx0"]

    scfg = cfg["solve"]
    region = scfg["regions"][region_name]
    cluster_ids = set(region["cluster_ids"])
    atlas_by_id = {r["cluster"]: r for r in atlas["clusters"]}

    edges = seam_edges(cell_owner, base_code, gy0, gx0)
    by_tam = {}
    for a, b in edges:
        by_tam.setdefault(a, []).append(b)

    retained = cell_owner == base_code
    blend = int(scfg["blend_cells"])
    solve_cells: set[tuple[int, int]] = set()

    def expand_from(bbox):
        x0, y0, x1, y1 = bbox
        start = None
        for cand in ((x1, y1), (x1 - 1, y1), (x1, y1 - 1), (x0, y0)):
            if cand in by_tam:
                start = cand
                break
        if start is None:
            raise SystemExit(f"FAILURE: region cluster bbox {bbox} has no seam edge")
        q = deque([(start, 0)])
        seen = {start}
        while q:
            cell, d = q.popleft()
            solve_cells.add(cell)
            if d >= blend:
                continue
            x, y = cell
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb in seen:
                    continue
                ry, rx = nb[1] - gy0, nb[0] - gx0
                if 0 <= ry < retained.shape[0] and 0 <= rx < retained.shape[1] \
                        and retained[ry, rx]:
                    seen.add(nb)
                    q.append((nb, d + 1))

    for cid in sorted(cluster_ids):
        expand_from(atlas_by_id[cid]["bbox_cells_xyxy"])

    xs = [c[0] for c in solve_cells]
    ys = [c[1] for c in solve_cells]
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)
    print(f"solve cells: {len(solve_cells)} bbox x[{bx0}..{bx1}] y[{by0}..{by1}]")

    pad = blend + 2
    r_lo = max(0, (by0 - gy0) * 64 - pad * 64)
    r_hi = min(tam_h.shape[0], (by1 - gy0) * 64 + 65 + pad * 64)
    c_lo = max(0, (bx0 - gx0) * 64 - pad * 64)
    c_hi = min(tam_h.shape[1], (bx1 - gx0) * 64 + 65 + pad * 64)

    solve_mask = np.zeros(tam_h.shape, dtype=bool)
    for (cx, cy) in solve_cells:
        r0 = (cy - gy0) * 64
        c0 = (cx - gx0) * 64
        solve_mask[r0:r0 + 65, c0:c0 + 65] = True

    smask = solve_mask[r_lo:r_hi, c_lo:c_hi]
    tam_w = tam_h[r_lo:r_hi, c_lo:c_hi]
    oth_w = oth_h[r_lo:r_hi, c_lo:c_hi]

    owner_here = np.zeros(tam_h.shape, dtype=np.uint8)
    for a, b in edges:
        if a in solve_cells:
            owner_here[b[1] - gy0, b[0] - gx0] = cell_owner[b[1] - gy0, b[0] - gx0]
    cy0, cx0 = r_lo // 64, c_lo // 64
    cy1 = min(cell_owner.shape[0], -(-r_hi // 64))
    cx1 = min(cell_owner.shape[1], -(-c_hi // 64))
    own_cells_w = (owner_here[cy0:cy1, cx0:cx1] != 0) & \
                  (owner_here[cy0:cy1, cx0:cx1] != base_code)
    own_v = np.repeat(np.repeat(own_cells_w, 64, axis=0), 64, axis=1)
    own_v = np.pad(own_v, ((0, max(0, smask.shape[0] - own_v.shape[0])),
                           (0, max(0, smask.shape[1] - own_v.shape[1]))),
                   mode="edge")[:smask.shape[0], :smask.shape[1]]
    seam_v = own_v & smask

    ring_v = smask & ~ndimage.binary_erosion(smask)   # erosion never wraps

    known = np.full(smask.shape, np.nan, dtype=np.float32)
    known[ring_v] = tam_w[ring_v]          # ring first: seam must win below
    known[seam_v] = oth_w[seam_v]
    # Interpolate owner NaN runs along each seam edge (stub LAND cells) so
    # ridge lines stay constrained instead of notching the solve.
    for a, b in edges:
        if a not in solve_cells:
            continue
        if a[0] != b[0]:
            cw = (max(a[0], b[0]) - gx0) * 64 - c_lo
            lo = (a[1] - gy0) * 64 - r_lo
            vals = known[lo:lo + 65, cw]
        else:
            rw = (max(a[1], b[1]) - gy0) * 64 - r_lo
            lo = (a[0] - gx0) * 64 - c_lo
            vals = known[rw, lo:lo + 65]
        fin = np.isfinite(vals)
        if fin.sum() >= 2 and not fin.all():
            idxs = np.arange(vals.size)
            vals[:] = np.interp(idxs, idxs[fin], vals[fin])
    dirich_v = np.isfinite(known) & smask
    known[~dirich_v] = 0.0

    base = coarse_laplace(smask, np.where(dirich_v, known, np.nan), tam_w,
                          int(scfg["coarse_factor"]))
    _health("base", base, smask)
    field = np.where(smask, base, tam_w).astype(np.float32)
    field[dirich_v] = known[dirich_v]
    _health("field_after_dirich", field, smask)

    # Style measurement: owner roughness (detrended) near the seam.
    own_fill = np.where(np.isnan(oth_w), 0.0, oth_w)
    own_res = oth_w - ndimage.uniform_filter(own_fill, 16)
    own_res_valid = own_res[own_v & ~np.isnan(oth_w)]
    sigma_owner = float(np.std(own_res_valid)) if own_res_valid.size else 150.0

    dist_in = ndimage.distance_transform_edt(smask)
    fade = np.clip(dist_in / float(scfg["fade_verts"]), 0.0, 1.0)
    gy, gx = np.gradient(field, 128.0)
    slope = np.sqrt(gx * gx + gy * gy) * 128.0
    slope = np.where(np.isfinite(slope), slope, 0.0)
    amb = ndimage.uniform_filter(np.where(smask, field, np.nan_to_num(field)), 129)
    relief = np.clip(field - amb, 0.0, None)
    amp = np.clip(float(scfg["amp_slope_coef"]) * slope
                  + float(scfg.get("amp_relief_coef", 0.0)) * relief
                  + float(scfg["amp_base_gu"]), 0.0, float(scfg["amp_max_gu"]))
    rng = np.random.default_rng(int(scfg["seed"]))
    noise = fbm_noise(smask.shape, scfg, rng)
    core = smask & (dist_in > float(scfg["fade_verts"]))
    n = noise[smask]
    n = (n - n.mean()) / max(float(n.std()), 1e-6)
    hf = np.zeros_like(noise)
    hf[smask] = n
    detail_hf = hf * amp
    if core.any():
        sd = float(np.std(detail_hf[core]))
        if sd > 1.0:
            detail_hf *= float(np.clip(
                sigma_owner * float(scfg["style_match"]) / sd, 0.25, 6.0))
    mcfg = dict(scfg)
    mcfg["octaves"] = int(scfg.get("massif_octaves", 3))
    massif = fbm_noise(smask.shape, mcfg, rng)
    m = massif[smask]
    m = (m - m.mean()) / max(float(m.std()), 1e-6)
    massif_u = np.zeros_like(noise)
    massif_u[smask] = m
    detail = (detail_hf
              + float(scfg.get("massif_amp_gu", 0.0)) * massif_u) * fade
    field = field + detail * smask
    field[dirich_v] = known[dirich_v]
    _health("field_after_detail", field, smask)

    field = d8_carve(field, smask, float(scfg["carve_coef_gu"]),
                     float(scfg["carve_fade_verts"]),
                     float(scfg["carve_max_gu"]),
                     float(scfg["carve_slope_min_gu_per_vert"]),
                     int(scfg["seed"]) + 1,
                     float(scfg.get("carve_accum_cap", 8192)))
    field[dirich_v] = known[dirich_v]
    _health("field_after_carve", field, smask)
    print(f"  [health] seam nonfinite: {int((seam_v & ~np.isfinite(field)).sum())}, "
          f"owner NaN at seam: {int((seam_v & ~np.isfinite(oth_w)).sum())}")

    out_field = tam_h.copy()
    region_vals = out_field[r_lo:r_hi, c_lo:c_hi]
    changed = smask & (np.abs(region_vals - field) > 1e-4)
    region_vals[smask] = field[smask]

    seam_ok = seam_v & np.isfinite(oth_w)
    seam_delta = np.abs(field[seam_ok] - oth_w[seam_ok])
    seam_max = float(seam_delta.max()) if seam_delta.size else -1.0
    interior = ndimage.binary_erosion(smask)
    out_win = out_field[r_lo:r_hi, c_lo:c_hi]
    fill = np.where(np.isfinite(out_win), out_win, 0.0)
    gy2, gx2 = np.gradient(fill, 128.0)
    slope_after = np.sqrt(gx2 * gx2 + gy2 * gy2) * 128.0

    outdir = _resolve(ROOT, cfg["paths"]["solve_out_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / f"{region_name}_field.npz",
                        field=out_field, solve_mask=solve_mask,
                        win=(np.int32(r_lo), np.int32(r_hi),
                             np.int32(c_lo), np.int32(c_hi)),
                        gx0=np.int32(gx0), gy0=np.int32(gy0))
    audit = {
        "region": region_name,
        "clusters": sorted(cluster_ids),
        "solve_cells": len(solve_cells),
        "bbox_cells_xyxy": [int(bx0), int(by0), int(bx1), int(by1)],
        "seam_vertices": int(seam_v.sum()),
        "seam_vertices_finite_owner": int(seam_ok.sum()),
        "seam_max_delta_gu": seam_max,
        "changed_vertices": int(changed.sum()),
        "outside_writes": int((np.abs(out_field - tam_h) > 0).sum() - int(changed.sum())),
        "sigma_owner_gu": round(sigma_owner, 1),
        "max_slope_in_region_gu_per_vert": round(float(slope_after[interior].max()), 1),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(outdir / f"{region_name}_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, default=lambda o: int(o))
    for k, v in audit.items():
        print(f"  {k}: {v}")

    m = int(region.get("review_margin_cells", 6)) * 64
    wr_lo = max(0, (by0 - gy0) * 64 - m)
    wr_hi = min(tam_h.shape[0], (by1 - gy0) * 64 + 65 + m)
    wc_lo = max(0, (bx0 - gx0) * 64 - m)
    wc_hi = min(tam_h.shape[1], (bx1 - gx0) * 64 + 65 + m)
    rcfg = cfg["render"]
    ppv = int(rcfg["px_per_vertex"])
    before = render_split_window(tam_h, oth_h, cell_owner, base_code,
                                 wr_lo, wr_hi, wc_lo, wc_hi, rcfg)
    save_shade_png(before, outdir / f"{region_name}_before.png", ppv,
                   title=f"{region_name} BEFORE (corpus)")
    after = render_split_window(out_field, oth_h, cell_owner, base_code,
                                wr_lo, wr_hi, wc_lo, wc_hi, rcfg)
    save_shade_png(after, outdir / f"{region_name}_after.png", ppv,
                   title=f"{region_name} AFTER solve")
    wide = 16 * 64
    a2 = render_split_window(out_field, oth_h, cell_owner, base_code,
                             max(0, wr_lo - wide),
                             min(tam_h.shape[0], wr_hi + wide),
                             max(0, wc_lo - wide),
                             min(tam_h.shape[1], wc_hi + wide), rcfg)
    save_shade_png(a2, outdir / f"{region_name}_after_wide.png",
                   max(1, ppv // 2), title=f"{region_name} AFTER (wide)")
    print(f"wrote {outdir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tamriel_reworked_v1.json"))
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    cfg = load_config(_resolve(ROOT, args.config))
    if args.region not in cfg["solve"]["regions"]:
        print(f"FAILURE: unknown region {args.region!r}")
        return 1
    return solve_region(cfg, args.region)


if __name__ == "__main__":
    raise SystemExit(main())
