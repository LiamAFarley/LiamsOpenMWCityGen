"""Panel: three terrain-synthesis variants for one seam region, judged side by side.

Purpose
    Stage D redesign panel (see runs/2026-08-24_terrain_synthesis_brainstorm.md).
    Builds the shared harmonic base + measured style spec for a configured
    region, then synthesizes the blend band three ways:

      P1 noise  - domain-warped band-matched multifractal (measured amplitudes)
      P2 erode  - seed + vectorized droplet erosion with border-flow seeding
      P3 clone  - owner exemplar patches, rotated/mirrored and feather-quilted

    All variants re-impose exact Dirichlet seam heights and taper HF detail
    toward the MEASURED AMBIENT tamriel roughness outward (never to zero).
    Emits per-variant crops, a combined panel image, and a metric table
    (band-sigma ratios, orientation correlation, border exactness,
    sigma-vs-distance cliff detector).

Inputs
    --config JSON, --region key under solve.regions; corpus npz, seam atlas,
    corpus manifest. Parameters under ``solve.panel``.

Outputs (under paths.solve_out_dir / panel/)
    ``<region>_p1_noise.png`` / ``_p2_erode.png`` / ``_p3_clone.png`` (+ wide),
    ``<region>_panel_combined.png``, ``<region>_panel_metrics.json``.

Pipeline position
    Stage D decision artifact for tamriel-reworked-heightmap; writes only its
    own output directory.

Invariants
    Deterministic (seeded); zero writes outside each variant's solve mask;
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
from scipy import ndimage
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen.terrainfield import (  # noqa: E402
    load_config, load_corpus, seam_edges, render_split_window, save_shade_png,
)
from procgen import terrainstyle as ts  # noqa: E402
from solve_region_blend import coarse_laplace  # noqa: E402


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def build_context(cfg: dict, region_name: str):
    arrays, meta = load_corpus(_resolve(ROOT, cfg["paths"]["corpus_npz"]))
    with open(_resolve(ROOT, cfg["paths"]["seam_atlas_json"]), encoding="utf-8") as fh:
        atlas = json.load(fh)
    with open(_resolve(ROOT, cfg["paths"]["corpus_manifest"]), encoding="utf-8") as fh:
        manifest = json.load(fh)
    names = manifest["source_names"]
    base_code = names.index(manifest["base_source"]) + 1
    tam_h, oth_h = arrays["tam_h"], arrays["oth_h"]
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
    blend = int(scfg.get("panel", {}).get("blend_cells",
                                          scfg.get("blend_cells", 6)))
    solve_cells = set()

    def expand_from(bbox):
        x0, y0, x1, y1 = bbox
        start = next((c for c in ((x1, y1), (x1 - 1, y1), (x1, y1 - 1), (x0, y0))
                      if c in by_tam), None)
        if start is None:
            raise SystemExit(f"FAILURE: bbox {bbox} has no seam edge")
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
    pad = blend + 2
    r_lo = max(0, (by0 - gy0) * 64 - pad * 64)
    r_hi = min(tam_h.shape[0], (by1 - gy0) * 64 + 65 + pad * 64)
    c_lo = max(0, (bx0 - gx0) * 64 - pad * 64)
    c_hi = min(tam_h.shape[1], (bx1 - gx0) * 64 + 65 + pad * 64)

    solve_mask = np.zeros(tam_h.shape, dtype=bool)
    for (cx, cy) in solve_cells:
        solve_mask[(cy - gy0) * 64:(cy - gy0) * 64 + 65,
                   (cx - gx0) * 64:(cx - gx0) * 64 + 65] = True

    smask = solve_mask[r_lo:r_hi, c_lo:c_hi]
    tam_w = tam_h[r_lo:r_hi, c_lo:c_hi]
    oth_w = oth_h[r_lo:r_hi, c_lo:c_hi]
    # Intended final world view: owner heights where they exist, tamriel
    # heights under height-less owner stubs (user ruling: no missing cells).
    own_view = np.where(np.isfinite(oth_w), oth_w, tam_w).astype(np.float32)

    owner_here = np.zeros(tam_h.shape, dtype=np.uint8)
    for a, b in edges:
        if a in solve_cells:
            owner_here[b[1] - gy0, b[0] - gx0] = cell_owner[b[1] - gy0, b[0] - gx0]
    cy0, cx0 = r_lo // 64, c_lo // 64
    cy1 = min(cell_owner.shape[0], -(-r_hi // 64))
    cx1 = min(cell_owner.shape[1], -(-c_hi // 64))
    oc = owner_here[cy0:cy1, cx0:cx1]
    own_cells = (oc != 0) & (oc != base_code)
    own_v = np.repeat(np.repeat(own_cells, 64, axis=0), 64, axis=1)
    own_v = np.pad(own_v, ((0, max(0, smask.shape[0] - own_v.shape[0])),
                           (0, max(0, smask.shape[1] - own_v.shape[1]))),
                   mode="edge")[:smask.shape[0], :smask.shape[1]]
    seam_v = own_v & smask

    ring_v = smask & ~ndimage.binary_erosion(smask)   # erosion never wraps

    known = np.full(smask.shape, np.nan, dtype=np.float32)
    known[ring_v] = tam_w[ring_v]          # ring first: seam must win below
    known[seam_v] = own_view[seam_v]       # owner heights; tam heights under stubs
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

    return dict(arrays=arrays, tam_h=tam_h, oth_h=oth_h, cell_owner=cell_owner,
                base_code=base_code, names=names, gy0=gy0, gx0=gx0,
                smask=smask, tam_w=tam_w, oth_w=oth_w, own_view=own_view,
                seam_v=seam_v,
                ring_v=ring_v, dirich_v=dirich_v, known=known,
                own_v=own_v, bbox=(bx0, by0, bx1, by1),
                win=(r_lo, r_hi, c_lo, c_hi), render=cfg["render"],
                scfg=scfg, region=region, region_name=region_name)


def ambient_mask(ctx, ring_v, width_verts=384):
    """Retained tamriel cells just OUTSIDE the solve band (for ambient style)."""
    tam_h = ctx["tam_h"]
    r_lo, r_hi, c_lo, c_hi = ctx["win"]
    grow = ndimage.binary_dilation(ring_v, iterations=width_verts)
    return grow & ~ndimage.binary_dilation(
        ctx["smask"], iterations=8) & np.isfinite(tam_h[r_lo:r_hi, c_lo:c_hi])


def taper_scale(dist_from_seam: np.ndarray, blend_verts: float,
                inner: float, outer: float) -> np.ndarray:
    """1 at the seam -> ``outer`` at the band edge; never through zero."""
    t = np.clip(dist_from_seam / max(blend_verts, 1.0), 0.0, 1.0)
    s = inner + (outer - inner) * (t * t * (3 - 2 * t))
    return s.astype(np.float32)


def droplet_erosion(field: np.ndarray, smask: np.ndarray, seeds_idx: np.ndarray,
                    rng: np.random.Generator, n_droplets: int = 100_000,
                    steps: int = 140, dt: float = 0.35,
                    inertia: float = 0.06, capacity: float = 2.5,
                    deposit: float = 0.6, erode: float = 0.12,
                    evap: float = 0.015) -> np.ndarray:
    """Batched particle hydraulic erosion (numpy port of the classical
    droplet model). Erodes only inside ``smask``; seeds may concentrate
    droplets at border flow-entry points."""
    h = np.where(smask, field, np.float32(-1e9)).astype(np.float32)
    H, W = h.shape
    hi = np.argwhere(smask)
    n_seeds = seeds_idx.size
    pick = rng.random(n_droplets)
    use_seed = pick < (0.5 if n_seeds else 0.0)
    rand_i = rng.integers(0, hi.shape[0], n_droplets)
    ry = hi[rand_i, 0].astype(np.float64)
    rx = hi[rand_i, 1].astype(np.float64)
    if n_seeds:
        s = seeds_idx[rng.integers(0, n_seeds, n_droplets)]
        sy, sx = np.unravel_index(s, h.shape)
        ry = np.where(use_seed, sy.astype(np.float64) + rng.random(n_droplets) - 0.5, ry)
        rx = np.where(use_seed, sx.astype(np.float64) + rng.random(n_droplets) - 0.5, rx)
    vel_r = np.zeros(n_droplets)
    vel_c = np.zeros(n_droplets)
    water = np.ones(n_droplets)
    sed = np.zeros(n_droplets)
    alive = smask[ry.astype(int), rx.astype(int)]

    for _ in range(steps):
        if not alive.any():
            break
        gy, gx = np.gradient(h, 128.0)
        gy = np.clip(np.nan_to_num(gy, 0.0), -40.0, 40.0)
        gx = np.clip(np.nan_to_num(gx, 0.0), -40.0, 40.0)
        ir = np.clip(ry.astype(int), 0, H - 1)
        ic = np.clip(rx.astype(int), 0, W - 1)
        g_r = gy[ir, ic]
        g_c = gx[ir, ic]
        vel_r = (vel_r - dt * inertia * g_r * 128.0 * alive) * (1 - dt * 0.15)
        vel_c = (vel_c - dt * inertia * g_c * 128.0 * alive) * (1 - dt * 0.15)
        speed = np.clip(np.sqrt(vel_r * vel_r + vel_c * vel_c), 1e-6, 60.0)
        step_r = np.where(alive, vel_r / speed, 0.0)
        step_c = np.where(alive, vel_c / speed, 0.0)
        ry = np.clip(ry + step_r, 0, H - 1 - 1e-3)
        rx = np.clip(rx + step_c, 0, W - 1 - 1e-3)
        ir2 = np.clip(ry.astype(int), 0, H - 1)
        ic2 = np.clip(rx.astype(int), 0, W - 1)
        inside = smask[ir2, ic2] & np.isfinite(h[ir2, ic2])
        newly_dead = alive & ~inside
        vel_r[newly_dead] = 0.0
        vel_c[newly_dead] = 0.0
        water[newly_dead] = 0.0
        alive = inside & alive & (water > 0.01)
        if not alive.any():
            break
        h_here = h[ir, ic]
        h_next = h[ir2, ic2]
        c_eq = np.maximum(h_here - h_next, 0.0) * speed * water * capacity * 1e-3
        diff = c_eq - sed
        dep = dt * deposit * diff
        ero = dt * erode * -np.minimum(diff, 0.0)
        sed += (dep - ero) * alive
        flat = h.ravel()
        np.add.at(flat, ir * W + ic, ero * alive * water * 0.01)
        np.add.at(flat, ir2 * W + ic2, -dep * alive * water * 0.01)
        h = flat.reshape(H, W)
        water *= (1 - dt * evap)
    return field - np.where(smask, field - h, 0.0)


def thermal_pass(field: np.ndarray, smask: np.ndarray, talus: float = 220.0,
                 rate: float = 0.5, iters: int = 3) -> np.ndarray:
    h = field.copy()
    for _ in range(iters):
        total = np.zeros_like(h)
        for ax, sh in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = np.roll(h, (sh, 0) if ax == 0 else (0, sh), axis=(0, 1))
            d = h - nb
            move = np.maximum(np.abs(d) - talus, 0.0) * rate * 0.25 * np.sign(d)
            total += move
        h = h - total * smask
    return h


def quilt_band(band_field: np.ndarray, owner_valid: np.ndarray,
               target: np.ndarray, rng: np.random.Generator,
               patch: int, step_div: int, feather: int) -> np.ndarray:
    """Transplant one frequency band of owner terrain over ``target`` using
    rotated/mirrored patches with feathered overlap. Band fields are ~zero
    mean, so only texture — never absolute height — is carried."""
    H, W = target.shape
    step = max(8, patch // int(step_div))
    oys, oxs = np.nonzero(owner_valid)
    if oys.size == 0:
        return np.zeros((H, W), np.float32)
    half = patch // 2
    margin = half + 1
    cand = (oys >= margin) & (oys < H - margin) & (oxs >= margin) & (oxs < W - margin)
    cand_idx = np.nonzero(cand)[0]
    if cand_idx.size == 0:
        return np.zeros((H, W), np.float32)
    acc = np.zeros((H, W), np.float32)
    wgt = np.zeros((H, W), np.float32)
    tys, txs = np.nonzero(target)
    ty_lo, ty_hi = int(tys.min()), int(tys.max())
    tx_lo, tx_hi = int(txs.min()), int(txs.max())
    for ty in range(ty_lo, ty_hi + 1, step):
        for tx in range(tx_lo, tx_hi + 1, step):
            for _try in range(12):
                j = cand_idx[int(rng.integers(0, cand_idx.size))]
                oy, ox = int(oys[j]), int(oxs[j])
                sy, sx = oy - half, ox - half
                if sy < 0 or sx < 0 or sy + patch > H or sx + patch > W:
                    continue
                src = band_field[sy:sy + patch, sx:sx + patch]
                if np.isfinite(src).all():
                    break
            else:
                continue
            k = int(rng.integers(0, 8))
            src = np.rot90(src, k % 4)
            if k >= 4:
                src = src[:, ::-1]
            ph, pw = src.shape
            ty2, tx2 = min(H, ty + ph), min(W, tx + pw)
            if ty2 <= ty or tx2 <= tx:
                continue
            src = src[:ty2 - ty, :tx2 - tx]
            hh, ww = src.shape
            yy = np.arange(hh, dtype=np.float32)[:, None]
            xx = np.arange(ww, dtype=np.float32)[None, :]
            a = np.minimum(np.minimum(yy + 1.0, hh - yy),
                           np.minimum(xx + 1.0, ww - xx))
            a = np.clip(np.broadcast_to(a, (hh, ww)) / feather, 0.0, 1.0)
            acc[ty:ty2, tx:tx2] += src * a
            wgt[ty:ty2, tx:tx2] += a
    return acc / np.maximum(wgt, 1e-6)


def d8_graph(h: np.ndarray, valid: np.ndarray):
    """Steepest-descent receivers (8-dir), flow accumulation, step distances.
    Cells without a downhill receiver get recv = -1 (sinks / flats)."""
    H, W = h.shape
    index = np.arange(H * W, dtype=np.int64).reshape(H, W)
    recv = np.full((H, W), -1, dtype=np.int64)
    best = np.full((H, W), np.float32(1e30))
    dist = np.zeros((H, W), np.float32)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
        sh_ = np.roll(h, (-dy, -dx), axis=(0, 1))
        d = 128.0 * (1.0 if dx == 0 or dy == 0 else 2.0 ** 0.5)
        with np.errstate(invalid="ignore"):
            slope_to = (h - sh_) / np.float32(d)
        take = (slope_to > 0) & (slope_to < best) & np.isfinite(sh_)
        shifted = np.roll(index, (-dy, -dx), axis=(0, 1))
        recv = np.where(take, shifted, recv)
        best = np.where(take, slope_to, best)
        dist = np.where(take, np.float32(d), dist)
    flat = h.ravel()
    recv_flat = recv.ravel()
    accum = np.ones(flat.size, dtype=np.float32)
    for i in np.argsort(flat)[::-1]:
        if not np.isfinite(flat[i]):
            continue
        j = recv_flat[i]
        if j >= 0 and np.isfinite(flat[j]):
            accum[j] += accum[i]
    return recv, accum.reshape(H, W), dist


def stream_power(field: np.ndarray, smask: np.ndarray, dirich_v: np.ndarray,
                 known: np.ndarray, pcfg: dict, rng_seed: int) -> np.ndarray:
    """Explicit stream-power incision + hillslope creep on the D8 graph.
    Sinks (recv == -1) are never incised, so no artificial pits."""
    h = field.copy()
    iters = int(pcfg.get("erode_iters", 8))
    inner = int(pcfg.get("erode_inner", 4))
    k = float(pcfg.get("erode_k", 0.002))
    m = float(pcfg.get("erode_m", 0.8))
    creep = float(pcfg.get("erode_creep", 0.1))
    cap = float(pcfg.get("erode_cap_per_iter", 60))
    for _ in range(iters):
        recv, accum, dist = d8_graph(h, smask)
        recv_flat = recv.ravel()
        dist_flat = dist.ravel()
        donor = np.nonzero(smask.ravel() & (recv_flat >= 0))[0]
        accum_flat = accum.ravel()
        for _ in range(inner):
            hr = h.ravel()
            S = np.zeros(hr.size, np.float32)
            ok = recv_flat >= 0
            S[ok] = np.maximum((hr - hr[np.where(ok, recv_flat, 0)]) /
                               np.maximum(dist_flat, 1.0), 0.0)[ok]
            inc = np.minimum(k * np.power(accum_flat, m) * S, cap)
            add = np.zeros(hr.size, np.float32)
            add[donor] = -inc[donor]
            lap = (np.roll(h, 1, 0) + np.roll(h, -1, 0) +
                   np.roll(h, 1, 1) + np.roll(h, -1, 1) - 4.0 * h)
            add += (creep * lap).ravel() * smask.ravel()
            h = (hr + add).reshape(h.shape)
            h[dirich_v] = known[dirich_v]
    return h


def slab_terrain(field: np.ndarray, smask: np.ndarray, layers: int = 16,
                 window_verts: float = 20.0) -> np.ndarray:
    """Hatchling slab filter (EDT port): stacked threshold distance-steps.
    Reads as soil creep / aging — softens and organizes slopes."""
    x = np.where(smask, field, np.nan)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi - lo < 1e-3:
        return field
    xn = ((field - lo) / (hi - lo)).astype(np.float32)
    out = np.zeros_like(xn)
    for i in range(layers):
        t = i / max(layers - 1, 1)
        above = xn > t
        d_pos = ndimage.distance_transform_edt(~above)
        d_neg = ndimage.distance_transform_edt(above)
        v = np.clip(0.5 + (d_neg - d_pos) / (2.0 * window_verts), 0.0, 1.0)
        out += v
    out /= layers
    return (out * (hi - lo) + lo).astype(np.float32)


def nn_fill(field: np.ndarray) -> np.ndarray:
    """Replace NaNs with the nearest finite value (stub LAND cells) so band
    decomposition never sees hard rectangular holes."""
    bad = ~np.isfinite(field)
    if not bad.any():
        return field
    idx = ndimage.distance_transform_edt(bad, return_distances=False,
                                         return_indices=True)
    return field[tuple(idx)]


def run_variant(variant: str, ctx, base: np.ndarray, ref: dict, amb: dict,
                seeds_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scfg = ctx["scfg"]
    pcfg = scfg.get("panel", {})
    smask, dirich_v, known = ctx["smask"], ctx["dirich_v"], ctx["known"]
    seam_v = ctx["seam_v"]
    sizes = ref["band_sizes"]
    dist_seam = ndimage.distance_transform_edt(~seam_v)
    blend_verts = float(pcfg.get("blend_cells", scfg.get("blend_cells", 6))) * 64
    p = (np.clip(dist_seam / max(blend_verts, 1.0), 0.0, 1.0)) ** 2 * (3 - 2 * np.clip(
        dist_seam / max(blend_verts, 1.0), 0.0, 1.0))
    amb_field = ts.band_matched_detail(smask.shape, amb["band_sigma"],
                                       sizes=sizes, rng=rng,
                                       warp_strength=float(pcfg.get("warp_strength", 24)))

    if variant == "noise":
        detail = ts.band_matched_detail(smask.shape, ref["band_sigma"],
                                        sizes=sizes, rng=rng,
                                        warp_strength=float(pcfg.get("warp_strength", 24)))
        field = base + (detail * (1 - p) + amb_field * p) * smask
    elif variant == "clone":
        owner_valid = ctx["own_v"] & np.isfinite(ctx["oth_w"])
        owner_details, _ = ts.band_stack(nn_fill(ctx["own_view"]), sizes)
        amb_details, _ = ts.band_stack(amb_field, sizes)
        detail = np.zeros_like(base)
        for i, k in enumerate(sizes):
            patch = int(pcfg.get("clone_patch_by_band", {}).get(str(k), 128))
            feather = int(pcfg.get("clone_feather_by_band", {}).get(str(k), 40))
            qk = quilt_band(owner_details[i], owner_valid, smask, rng,
                            patch, int(pcfg.get("clone_step_div", 2)), feather)
            detail += (1.0 - p) * qk + p * amb_details[i]
        field = base + detail * smask
    elif variant == "spl":
        seed_scale = float(pcfg.get("erode_seed_sigma_scale", 0.45))
        seed_sigma = [s * seed_scale for s in ref["band_sigma"]]
        seed = ts.band_matched_detail(smask.shape, seed_sigma, sizes=sizes,
                                      rng=rng, warp_strength=32.0)
        field = base + seed * smask
        field[dirich_v] = known[dirich_v]
        field = stream_power(field, smask, dirich_v, known, pcfg, int(scfg["seed"]))
        ours = (field - base) * smask
        field = base + (ours * (1 - p) + amb_field * p) * smask
    elif variant == "slab":
        seed_sigma = [0.5 * s for s in ref["band_sigma"]]
        seed = ts.band_matched_detail(smask.shape, seed_sigma, sizes=sizes,
                                      rng=rng, warp_strength=28.0)
        field = slab_terrain(base + seed * smask, smask,
                             layers=int(pcfg.get("slab_layers", 16)),
                             window_verts=float(pcfg.get("slab_window_verts", 20)))
        ours = (field - base) * smask
        field = base + (ours * (1 - p) + amb_field * p) * smask
    elif variant == "hybrid":
        # clone band-transplant near the seam (TR texture carries across),
        # stream-power structure beyond, ambient crossfade at the band edge.
        owner_valid = ctx["own_v"] & np.isfinite(ctx["oth_w"])
        owner_details, _ = ts.band_stack(nn_fill(ctx["own_view"]), sizes)
        quilt_detail = np.zeros_like(base)
        for i, k in enumerate(sizes):
            patch = int(pcfg.get("clone_patch_by_band", {}).get(str(k), 128))
            feather = int(pcfg.get("clone_feather_by_band", {}).get(str(k), 40))
            qk = quilt_band(owner_details[i], owner_valid, smask, rng,
                            patch, int(pcfg.get("clone_step_div", 2)), feather)
            quilt_detail += qk
        seed_scale = float(pcfg.get("erode_seed_sigma_scale", 0.45))
        seed_sigma = [s * seed_scale for s in ref["band_sigma"]]
        seed = ts.band_matched_detail(smask.shape, seed_sigma, sizes=sizes,
                                      rng=rng, warp_strength=32.0)
        field_spl = base + seed * smask
        field_spl[dirich_v] = known[dirich_v]
        field_spl = stream_power(field_spl, smask, dirich_v, known, pcfg,
                                 int(scfg["seed"]))
        detail_spl = (field_spl - base) * smask
        carry_verts = float(pcfg.get("hybrid_carry_cells", 2.5)) * 64
        t = np.clip(dist_seam / max(carry_verts, 1.0), 0.0, 1.0)
        w_clone = 1.0 - (t * t * (3 - 2 * t))
        w_amb = p
        w_spl = np.clip(1.0 - w_clone - w_amb, 0.0, 1.0)
        detail = (w_clone * quilt_detail + w_spl * detail_spl
                  + w_amb * amb_field)
        field = base + detail * smask
    elif variant == "droplet":
        seed_scale = float(pcfg.get("erode_seed_sigma_scale", 0.45))
        seed_sigma = [s * seed_scale for s in ref["band_sigma"]]
        seed = ts.band_matched_detail(smask.shape, seed_sigma, sizes=sizes,
                                      rng=rng, warp_strength=32.0)
        field = base + seed * smask
        field[dirich_v] = known[dirich_v]
        field = droplet_erosion(field, smask, seeds_idx, rng,
                                n_droplets=int(pcfg.get("droplets", 120_000)))
        field = np.maximum(field, base - float(pcfg.get("droplet_downcut_cap", 500)))
        field = thermal_pass(field, smask)
        ours = (field - base) * smask
        field = base + (ours * (1 - p) + amb_field * p) * smask
    else:
        raise SystemExit(f"FAILURE: unknown variant {variant}")
    field[dirich_v] = known[dirich_v]
    return field


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tamriel_reworked_v1.json"))
    ap.add_argument("--region", default="tr_vvardenfell_wall")
    ap.add_argument("--variants", default="noise,clone,spl,slab,droplet")
    args = ap.parse_args()
    cfg = load_config(_resolve(ROOT, args.config))
    t0 = time.time()
    ctx = build_context(cfg, args.region)
    scfg = ctx["scfg"]
    smask, tam_w, oth_w = ctx["smask"], ctx["tam_w"], ctx["oth_w"]
    own_view = ctx["own_view"]
    seam_v, dirich_v, known = ctx["seam_v"], ctx["dirich_v"], ctx["known"]

    base = coarse_laplace(smask, np.where(dirich_v, known, np.nan), tam_w,
                          int(scfg["coarse_factor"]))
    field0 = np.where(smask, base, tam_w).astype(np.float32)
    field0[dirich_v] = known[dirich_v]

    sizes = tuple(int(s) for s in scfg.get("panel", {}).get("bands", ts.DEFAULT_BANDS))
    ref = ts.measure_style(own_view, ctx["own_v"], sizes=sizes)
    amb_m = ambient_mask(ctx, ctx["ring_v"])
    amb = ts.measure_style(ctx["tam_w"], amb_m, sizes=sizes)
    print(f"style ref band_sigma={ref['band_sigma']} amb={amb['band_sigma']}")

    # border flow-entry seeds: owner-side accumulation arriving at the seam
    of = np.where(ctx["own_v"], own_view, np.inf).astype(np.float32)
    H, W = of.shape
    index = np.arange(H * W, dtype=np.int64).reshape(H, W)
    recv = np.full((H, W), -1, dtype=np.int64)
    best = np.full((H, W), np.float32(1e30))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh_ = np.roll(of, (-dy, -dx), axis=(0, 1))
        with np.errstate(invalid="ignore"):
            st = (of - sh_) / 128.0
        take = (st > 0) & (st < best) & np.isfinite(sh_)
        recv = np.where(take, np.roll(index, (-dy, -dx), axis=(0, 1)), recv)
        best = np.where(take, st, best)
    flat_of = of.ravel()
    recv_flat = recv.ravel()
    accum = np.ones(flat_of.size, dtype=np.float32)
    for i in np.argsort(flat_of)[::-1]:
        if not np.isfinite(flat_of[i]):
            continue
        j = recv_flat[i]
        if j >= 0 and np.isfinite(flat_of[j]):
            accum[j] += accum[i]
    seam_flat = seam_v.ravel()
    seed_idx = np.nonzero(seam_flat & (accum > 64) & np.isfinite(flat_of))[0]
    print(f"border flow-entry seeds: {seed_idx.size}")

    rng = np.random.default_rng(int(scfg["seed"]))
    outdir = _resolve(ROOT, cfg["paths"]["solve_out_dir"]) / "panel"
    outdir.mkdir(parents=True, exist_ok=True)

    r_lo, r_hi, c_lo, c_hi = ctx["win"]
    by0, by1, bx0, bx1 = (ctx["bbox"][1], ctx["bbox"][3],
                          ctx["bbox"][0], ctx["bbox"][2])
    gy0, gx0 = ctx["gy0"], ctx["gx0"]
    m = int(ctx["region"].get("review_margin_cells", 6)) * 64
    wr_lo = max(0, (by0 - gy0) * 64 - m)
    wr_hi = min(ctx["tam_h"].shape[0], (by1 - gy0) * 64 + 65 + m)
    wc_lo = max(0, (bx0 - gx0) * 64 - m)
    wc_hi = min(ctx["tam_h"].shape[1], (bx1 - gx0) * 64 + 65 + m)
    rcfg = ctx["render"]
    ppv = int(rcfg["px_per_vertex"])

    metrics = {"region": args.region, "ref_band_sigma": ref["band_sigma"],
               "ambient_band_sigma": amb["band_sigma"], "variants": {}}
    tiles = []
    for variant in [v.strip() for v in args.variants.split(",")]:
        tv = time.time()
        vrng = np.random.default_rng(int(scfg["seed"]) + hash(variant) % 1000)
        field = run_variant(variant, ctx, field0, ref, amb, seed_idx, vrng)
        seam_ok = seam_v & np.isfinite(own_view)
        seam_max = float(np.abs(field[seam_ok] - own_view[seam_ok]).max()) \
            if seam_ok.any() else -1.0
        prof = ts.sigma_distance_profile(field, ndimage.distance_transform_edt(~seam_v),
                                         smask, band=16)
        mrow = {
            "seam_max_delta_gu": seam_max,
            "band_sigma_ratio_vs_ref": [
                round(a / max(b, 1e-6), 2) for a, b in zip(
                    ts.measure_style(field, smask, sizes=sizes)["band_sigma"],
                    ref["band_sigma"])],
            "orientation_corr": round(ts.orientation_corr(
                ts.measure_style(field, smask, sizes=sizes)["orientation_hist"],
                ref["orientation_hist"]), 3),
            "sigma16_distance_profile": prof,
            "profile_cliff": ts.profile_cliff(prof),
            "elapsed_s": round(time.time() - tv, 1),
        }
        metrics["variants"][variant] = mrow
        print(f"  {variant}: seam_max={seam_max:.1f} "
              f"band_ratio={[r for r in mrow['band_sigma_ratio_vs_ref']]} "
              f"ori_corr={mrow['orientation_corr']} cliff={mrow['profile_cliff']} "
              f"({mrow['elapsed_s']}s)")

        out_field = ctx["tam_h"].copy()
        out_field[r_lo:r_hi, c_lo:c_hi] = np.where(smask, field, tam_w)
        oth_view_full = np.where(np.isfinite(ctx["oth_h"]), ctx["oth_h"],
                                 ctx["tam_h"]).astype(np.float32)
        rgb = render_split_window(out_field, oth_view_full,
                                  ctx["cell_owner"], ctx["base_code"],
                                  wr_lo, wr_hi, wc_lo, wc_hi, rcfg)
        p = save_shade_png(rgb, outdir / f"{args.region}_{variant}.png", ppv,
                           title=f"{args.region} [{variant}]")
        tiles.append((variant, Image.open(p)))
        wide = 12 * 64
        a2 = render_split_window(out_field, oth_view_full, ctx["cell_owner"],
                                 ctx["base_code"],
                                 max(0, wr_lo - wide), min(ctx["tam_h"].shape[0], wr_hi + wide),
                                 max(0, wc_lo - wide), min(ctx["tam_h"].shape[1], wc_hi + wide),
                                 rcfg)
        save_shade_png(a2, outdir / f"{args.region}_{variant}_wide.png",
                       max(1, ppv // 2), title=f"{args.region} [{variant}] wide")

    # combined panel: stack variant crops with labels
    if tiles:
        imgs = []
        for name, im in tiles:
            canvas = Image.new("RGB", (im.width, im.height + 28), (24, 24, 24))
            canvas.paste(im, (0, 28))
            d = ImageDraw.Draw(canvas)
            d.text((8, 6), f"{args.region} — {name}", fill=(255, 220, 120))
            imgs.append(canvas)
        total_h = sum(i.height for i in imgs)
        total_w = max(i.width for i in imgs)
        combo = Image.new("RGB", (total_w, total_h), (0, 0, 0))
        y = 0
        for i in imgs:
            combo.paste(i, (0, y))
            y += i.height
        combo.save(outdir / f"{args.region}_panel_combined.png")

    with open(outdir / f"{args.region}_panel_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=lambda o: int(o))
    print(f"wrote {outdir} (total {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
