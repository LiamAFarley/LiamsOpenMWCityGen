"""Seam-context construction and the v3 constrained surface solve.

Purpose
    Shared machinery for v3 seam synthesis (v3 plan Phases B/C/F):

    - :func:`build_context` — one canonical per-region context: window,
      masks, hard constraints (seam + outer ring), the relief-amplified
      target field, seam distance, inward normals, and the adaptive
      blend-width field (plan section 8: width grows with local relief,
      2..10 cells).
    - :func:`solve_surface` — screened-Poisson least-squares surface with
      explicit owner normal-slope rows at the seam (C1 tendency, plan
      section 5), solved by conjugate gradient on the normal equations with
      a Jacobi preconditioner and a target-seeded initial guess.

    Constraint layout per unknown x in the solve band:
      data   : wd(x) * H(x) = wd(x) * target(x)
               wd ramps 0.05 -> 1.0 with distance from seam over the
               adaptive local blend width (plan section 8).
      smooth : ws * (4H(x) - sum H(neighbors)) = sum_fixed_rhs
      slope  : w * H(u) = w * (h_seam + 64 * s_own)
               for seam verts s whose inward neighbor u is unknown;
               s_own is the owner-side inward slope (plan section 5.3).

Pipeline position
    v3 Milestone 2 (no erosion). Consumes the corpus npz and the
    Milestone-1 relief-scaled field; produces the pre-erosion surface.

Invariants
    Seam verts are hard constraints equal to own_view (owner heights, tam
    heights under stubs); owner cells are never unknowns; deterministic;
    unknowns exist only inside the solve band.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

from procgen.terrainfield import load_corpus, seam_edges
from procgen.terrain_relief import relief_scale


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def nn_fill(field: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(field)
    if not bad.any():
        return field
    idx = ndimage.distance_transform_edt(bad, return_distances=False,
                                         return_indices=True)
    return field[tuple(idx)]


def load_target(root: Path, cfg: dict, tam_h: np.ndarray) -> np.ndarray:
    """Relief-amplified target: cached npz if present, else computed."""
    rel = cfg.get("solve", {}).get("v3", {}).get("relief_npz")
    if rel:
        p = Path(rel)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            with np.load(p) as z:
                return z["field"].astype(np.float32)
    scaled, _ = relief_scale(tam_h, cfg.get("terrain_relief", {}))
    return scaled.astype(np.float32)


def build_context(root: Path, cfg: dict, region_name: str) -> dict:
    arrays, meta = load_corpus(root / cfg["paths"]["corpus_npz"])
    with open(root / cfg["paths"]["seam_atlas_json"], encoding="utf-8") as fh:
        atlas = json.load(fh)
    with open(root / cfg["paths"]["corpus_manifest"], encoding="utf-8") as fh:
        manifest = json.load(fh)
    base_code = manifest["source_names"].index(manifest["base_source"]) + 1
    tam_h = arrays["tam_h"]
    cell_owner = arrays["cell_owner"]
    gy0, gx0 = int(meta["gy0"]), int(meta["gx0"])

    target_full = load_target(root, cfg, tam_h)
    own_full = np.where(np.isfinite(arrays["oth_h"]), arrays["oth_h"],
                        tam_h).astype(np.float32)

    v3 = cfg["solve"].get("v3", {})
    region = cfg["solve"]["regions"][region_name]
    atlas_by_id = {r["cluster"]: r for r in atlas["clusters"]}
    edges = seam_edges(cell_owner, base_code, gy0, gx0)
    by_tam = {}
    for a, b in edges:
        by_tam.setdefault(a, []).append(b)

    retained = cell_owner == base_code
    blend = int(v3.get("blend_cells_max", 10))
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

    for cid in sorted(region["cluster_ids"]):
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

    solve_mask_full = np.zeros(tam_h.shape, dtype=bool)
    for (cx, cy) in solve_cells:
        solve_mask_full[(cy - gy0) * 64:(cy - gy0) * 64 + 65,
                        (cx - gx0) * 64:(cx - gx0) * 64 + 65] = True

    smask = solve_mask_full[r_lo:r_hi, c_lo:c_hi]
    tam_w = tam_h[r_lo:r_hi, c_lo:c_hi]
    target = target_full[r_lo:r_hi, c_lo:c_hi]
    own_view = own_full[r_lo:r_hi, c_lo:c_hi]
    oth_w = arrays["oth_h"][r_lo:r_hi, c_lo:c_hi]

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
    ring_v = smask & ~ndimage.binary_erosion(smask)
    dist_seam = ndimage.distance_transform_edt(~seam_v)

    hard = np.zeros(smask.shape, dtype=bool)
    hard_vals = np.full(smask.shape, np.nan, np.float32)
    hard[ring_v] = True
    hard_vals[ring_v] = target[ring_v]
    hard[seam_v] = True
    hard_vals[seam_v] = own_view[seam_v]   # seam wins over ring at corners

    # Use both relief and the height jump that must be traversed. Relief alone
    # collapses to the minimum on the flat side of a tall owner wall.
    wmin = float(v3.get("blend_width_min_cells", 2.0))
    wmax = float(v3.get("blend_width_max_cells", 10.0))
    loc_rel = np.abs(target - ndimage.uniform_filter(
        np.where(np.isfinite(target), target, 0.0), 65))
    loc_rel[~smask] = 0.0
    lo_r = float(np.percentile(loc_rel[smask], 20)) if smask.any() else 0.0
    hi_r = float(np.percentile(loc_rel[smask], 95)) if smask.any() else 1.0
    rn = np.clip((loc_rel - lo_r) / max(hi_r - lo_r, 1e-6), 0.0, 1.0)
    width_from_relief = wmin + (wmax - wmin) * rn
    mismatch_grade = float(v3.get("max_blend_grade_gu_per_cell", 2500.0))
    mismatch = np.zeros_like(target, dtype=np.float32)
    seam_valid = seam_v & np.isfinite(own_view) & np.isfinite(target)
    mismatch[seam_valid] = np.abs(own_view[seam_valid] - target[seam_valid])
    if seam_v.any():
        nearest = ndimage.distance_transform_edt(
            ~seam_v, return_distances=False, return_indices=True)
        mismatch = mismatch[tuple(nearest)]
    width_from_mismatch = np.clip(mismatch / max(mismatch_grade, 1e-6),
                                   wmin, wmax)
    width_cells = np.maximum(width_from_relief, width_from_mismatch)

    # inward normals on seam verts: direction from owner cell into tam cell
    ny = np.zeros(smask.shape, np.float32)
    nx = np.zeros(smask.shape, np.float32)
    for a, b in edges:
        if a not in solve_cells:
            continue
        ry, rx = b[1] - gy0, b[0] - gx0
        dy_, dx_ = a[1] - b[1], a[0] - b[0]
        ey0, ex0 = ry * 64 - r_lo, rx * 64 - c_lo
        if dx_ != 0:
            col = ex0 + 64 if dx_ > 0 else ex0
            seg = (slice(max(0, ey0), min(smask.shape[0], ey0 + 65)),
                   slice(col, col + 1))
            ny[seg] = 0.0
            nx[seg] = float(dx_)
        else:
            row = ey0 + 64 if dy_ > 0 else ey0
            seg = (slice(row, row + 1),
                   slice(max(0, ex0), min(smask.shape[1], ex0 + 65)))
            ny[seg] = float(dy_)
            nx[seg] = 0.0
    keep = seam_v
    nx[~keep] = 0.0
    ny[~keep] = 0.0

    return dict(tam_h=tam_h, oth_h=arrays["oth_h"], cell_owner=cell_owner,
                base_code=base_code,
                gy0=gy0, gx0=gx0, names=manifest["source_names"],
                smask=smask, tam_w=tam_w, target=target, own_view=own_view,
                oth_w=oth_w, own_v=own_v, seam_v=seam_v, ring_v=ring_v,
                dist_seam=dist_seam, hard=hard, hard_vals=hard_vals,
                 width_cells=width_cells, nx=nx, ny=ny,
                 target_full=target_full,
                bbox=(bx0, by0, bx1, by1), win=(r_lo, r_hi, c_lo, c_hi),
                render=cfg["render"], region=region, region_name=region_name,
                edges=edges, solve_cells=solve_cells)


def solve_surface(ctx: dict, v3: dict) -> tuple[np.ndarray, dict]:
    surf = v3.get("surface", {})
    smask, hard, hard_vals = ctx["smask"], ctx["hard"], ctx["hard_vals"]
    target, seam_v = ctx["target"], ctx["seam_v"]
    nx, ny, own_view = ctx["nx"], ctx["ny"], ctx["own_view"]

    unk = smask & ~hard
    H, W = smask.shape
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[unk] = np.arange(int(unk.sum()))
    n = int(unk.sum())

    wd_field = 0.05 + 0.95 * smootherstep(
        ctx["dist_seam"] / np.maximum(ctx["width_cells"] * 64.0, 1.0))
    ws = float(surf.get("smooth_weight", 1.0))
    wg = float(surf.get("gradient_weight", 6.0))

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs: list[np.ndarray] = []

    # data rows
    ii = idx[unk]
    wv = wd_field[unk].astype(np.float64)
    rows.append(ii)
    cols.append(ii)
    vals.append(wv)
    rhs.append(wv * target[unk].astype(np.float64))

    # membrane rows: ws*(4H - sum nb) = sum_fixed  (vectorized shifts)
    acc_fixed = np.zeros(n, np.float64)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        sel = unk.copy()
        if dy == 1:
            sel[-1, :] = False
        if dy == -1:
            sel[0, :] = False
        if dx == 1:
            sel[:, -1] = False
        if dx == -1:
            sel[:, 0] = False
        shifted_idx = np.roll(idx, (dy, dx), axis=(0, 1))
        shifted_hard = np.roll(hard, (dy, dx), axis=(0, 1))
        shifted_val = np.where(shifted_hard,
                               np.roll(hard_vals, (dy, dx), axis=(0, 1)), 0.0)
        rr = idx[sel]
        jj = shifted_idx[sel]
        is_unk = jj >= 0
        rows.append(rr[is_unk])
        cols.append(jj[is_unk])
        vals.append(np.full(int(is_unk.sum()), -ws, dtype=np.float64))
        rhs.append(np.zeros(int(is_unk.sum()), dtype=np.float64))
        acc_fixed[rr[~is_unk]] += ws * shifted_val[sel][~is_unk]
    core_idx = idx[unk]
    rows.append(core_idx)
    cols.append(core_idx)
    vals.append(np.full(core_idx.size, 4.0 * ws, dtype=np.float64))
    rhs.append(acc_fixed)

    # slope rows: (H(u) - H(s)) / 64 = s_own  ->  H(u) = h_s + 64*s_own
    seam_list = np.nonzero(seam_v.ravel())[0]
    su = []
    sv = []
    sb = []
    for i in seam_list:
        sny, snx = ny.ravel()[i], nx.ravel()[i]
        uy = i // W + int(round(sny))
        ux = i % W + int(round(snx))
        if not (0 <= uy < H and 0 <= ux < W):
            continue
        ju = idx[uy, ux]
        if ju < 0:
            continue
        oy = i // W - int(round(sny))
        ox = i % W - int(round(snx))
        if not (0 <= oy < H and 0 <= ox < W) or not np.isfinite(own_view[oy, ox]):
            continue
        s_own = (own_view.ravel()[i] - own_view[oy, ox]) / 64.0
        h_s = hard_vals.ravel()[i]
        su.append(ju)
        sv.append(wg)
        sb.append(wg * (h_s + 64.0 * s_own))
    rows.append(np.asarray(su, dtype=np.int64))
    cols.append(np.asarray(su, dtype=np.int64))
    vals.append(np.asarray(sv, dtype=np.float64))
    rhs.append(np.asarray(sb, dtype=np.float64))

    rows = np.concatenate([np.asarray(r).ravel() for r in rows])
    cols = np.concatenate([np.asarray(c).ravel() for c in cols])
    vals = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in vals])
    b_vec = np.concatenate([np.asarray(b, dtype=np.float64).ravel() for b in rhs])
    n_rows = rows.size
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n_rows, n)).tocsr()
    AtA = (A.T @ A).tocsr()
    Atb = A.T @ b_vec
    diag = AtA.diagonal()
    M = sparse.diags(1.0 / np.maximum(diag, 1e-9))

    x0 = target[unk].astype(np.float64)
    x, status = cg(AtA, Atb, x0=x0, M=M,
                   rtol=float(surf.get("cg_tol", 1e-4)),
                   maxiter=int(surf.get("cg_maxiter", 400)))

    out = target.copy()
    flat = out.ravel()
    flat[np.nonzero(unk.ravel())[0]] = x.astype(np.float32)
    out = flat.reshape(out.shape)
    out[hard] = hard_vals[hard]
    # Outside the band is the relief-scaled target, not the original corpus.
    out[~smask] = target[~smask]
    edge = smask & ~ndimage.binary_erosion(smask) & ~seam_v
    edge_error = float(np.max(np.abs(out[edge] - target[edge]))) if edge.any() else 0.0
    info = {"unknowns": n, "cg_status": int(status),
            "data_weight_near_seam": 0.05,
            "slope_rows": len(su),
            "blend_width_cells_min": float(ctx["width_cells"][smask].min()),
            "blend_width_cells_max": float(ctx["width_cells"][smask].max()),
            "band_edge_max_abs_gu": edge_error}
    return out, info
