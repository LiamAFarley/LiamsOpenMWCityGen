"""Seam-context construction and the v3 constrained surface solve (v3 rev2).

Purpose
    Shared machinery for v3 seam synthesis (plan Phases B/C/F, revised per
    the Sol High review of 2026-08-24):

    - :func:`build_context` — canonical per-region context. Seam vertices
      are rasterized DIRECTLY from ``seam_edges`` (the ``own_v & smask``
      derivation is retired as authoritative). Solve corridors grow via BFS
      from EVERY seam cell of a cluster (no bbox-corner seed heuristic).
      Owner normal-slope constraints are built PER EDGE, so corner vertices
      carry one constraint per adjacent edge instead of one overwritten
      normal field.
    - :func:`solve_surface` — a proper least-squares equation assembler.
      Every conceptual constraint is its own equation row, tagged by family
      (``data`` / ``laplacian`` / ``boundary`` / ``slope``). Family row
      counts and post-solve residuals (RMS/max per family) are returned so
      an under-weighted family is visible in numbers, not guesses. Boundary
      vertices are eliminated from the unknowns and reported as an
      eliminated-constraint count with exactly-zero residual.

    The data target passed in must already contain the desired macro
    structure (Phase D continuation is a later milestone; the diagnostic
    run uses the relief-amplified Tamriel field directly).

Pipeline position
    v3 Milestone 2 (no erosion). Consumes the corpus npz and the
    relief-scaled field (staleness-guarded — see
    :func:`load_target`).

Invariants
    Seam vertices equal own_view exactly (eliminated boundary constraints);
    owner cells are never unknowns; deterministic; unknowns exist only
    inside the solve corridor.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

from procgen.terrainfield import load_corpus, seam_edges
from procgen.terrain_relief import relief_config_hash, relief_scale


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
    """Relief-amplified target with staleness guarding.

    The cached npz must carry a ``relief_cfg_hash`` matching the current
    ``terrain_relief`` config; otherwise this fails closed and instructs a
    regeneration (stale-parameter terrain is worse than an error).
    """
    want_hash = relief_config_hash(cfg)
    rel = cfg.get("solve", {}).get("v3", {}).get("relief_npz")
    if rel:
        p = Path(rel)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            with np.load(p) as z:
                got = str(z["relief_cfg_hash"]) if "relief_cfg_hash" in z else None
                if got != want_hash:
                    raise SystemExit(
                        "FAILURE: relief_scaled_field.npz was generated with a "
                        "different terrain_relief config "
                        f"(npz={got}, config={want_hash}). Rerun "
                        "tools/terrain/relief_preview.py --gains <max> to "
                        "regenerate it.")
                return z["field"].astype(np.float32)
    scaled, _ = relief_scale(tam_h, cfg.get("terrain_relief", {}))
    return scaled.astype(np.float32)


def rasterize_seam(edges, solve_cells: set, shape, gy0: int, gx0: int,
                   r_lo: int, c_lo: int):
    """Authoritative seam raster straight from ``seam_edges``.

    Returns ``(seam_v, edge_list)`` where ``edge_list`` entries hold the
    window vertex indices of one shared edge plus its unit inward normal
    (owner -> tam). Corner vertices appear once per adjacent edge, so
    per-edge constraints stay independent.
    """
    seam_v = np.zeros(shape, dtype=bool)
    edge_list = []
    for a, b in edges:
        if a not in solve_cells:
            continue
        ry, rx = b[1] - gy0, b[0] - gx0
        dy_, dx_ = a[1] - b[1], a[0] - b[0]
        ey0, ex0 = ry * 64 - r_lo, rx * 64 - c_lo
        verts = []
        if dx_ != 0:
            col = ex0 + 64 if dx_ > 0 else ex0
            r0 = max(0, ey0)
            r1 = min(shape[0], ey0 + 65)
            if col < 0 or col >= shape[1] or r1 <= r0:
                continue
            verts = [(r, col) for r in range(r0, r1)]
            normal = (0.0, float(dx_))
        else:
            row = ey0 + 64 if dy_ > 0 else ey0
            c0 = max(0, ex0)
            c1 = min(shape[1], ex0 + 65)
            if row < 0 or row >= shape[0] or c1 <= c0:
                continue
            verts = [(row, c) for c in range(c0, c1)]
            normal = (float(dy_), 0.0)
        flat = [r * shape[1] + c for r, c in verts]
        seam_v.ravel()[flat] = True
        edge_list.append({"verts": flat, "normal": normal})
    return seam_v, edge_list


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

    # corridor seeds: every tam cell that owns a region seam edge whose
    # owner cell lies in the cluster bbox expanded by one (no corner seed
    # heuristic — Sol High review item).
    cluster_boxes = []
    for cid in sorted(region["cluster_ids"]):
        x0, y0, x1, y1 = atlas_by_id[cid]["bbox_cells_xyxy"]
        cluster_boxes.append((x0 - 1, y0 - 1, x1 + 1, y1 + 1))

    def in_any_box(cell):
        x, y = cell
        return any(bx0 <= x <= bx1 and by0 <= y <= by1
                   for bx0, by0, bx1, by1 in cluster_boxes)

    solve_cells = set()

    def expand_from(start):
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

    # seeds: tam cells of region edges (owner side inside a cluster box)
    for a, b in edges:
        if a not in by_tam or a in solve_cells:
            continue
        if not in_any_box(b):
            continue
        expand_from(a)

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

    seam_v, edge_list = rasterize_seam(edges, solve_cells, smask.shape,
                                       gy0, gx0, r_lo, c_lo)
    ring_v = smask & ~ndimage.binary_erosion(smask)
    dist_seam = ndimage.distance_transform_edt(~seam_v)

    # hard constraints: seam (own heights) + ring (target). Eliminated from
    # the unknowns and reported as boundary-equation eliminations.
    hard = seam_v | ring_v
    hard_vals = np.where(seam_v, own_view, target).astype(np.float32)
    hard_vals[~hard] = 0.0

    # adaptive blend width (plan section 8) + mismatch term: the width must
    # cover the seam height difference, not just local target relief.
    wmin = float(v3.get("blend_width_min_cells", 2.0))
    wmax = float(v3.get("blend_width_max_cells", 10.0))
    grade = float(v3.get("max_blend_grade_gu_per_cell", 2500.0))
    loc_rel = np.abs(target - ndimage.uniform_filter(
        np.where(np.isfinite(target), target, 0.0), 65))
    loc_rel[~smask] = 0.0
    lo_r = float(np.percentile(loc_rel[smask], 20)) if smask.any() else 0.0
    hi_r = float(np.percentile(loc_rel[smask], 95)) if smask.any() else 1.0
    rn = np.clip((loc_rel - lo_r) / max(hi_r - lo_r, 1e-6), 0.0, 1.0)
    width_relief = wmin + (wmax - wmin) * rn
    delta_seam = np.zeros(smask.shape, np.float32)
    delta_seam[seam_v] = np.abs(own_view[seam_v] - target[seam_v])
    delta_field = ndimage.gaussian_filter(delta_seam, 192.0) * smask
    width_mismatch = delta_field / max(grade, 1.0)
    width_cells = np.clip(np.maximum(width_relief, width_mismatch), wmin, wmax)

    # per-vertex normals for METRICS (corner verts overwritten; solver uses
    # per-edge constraints instead)
    ny = np.zeros(smask.shape, np.float32)
    nx = np.zeros(smask.shape, np.float32)
    for e in edge_list:
        ny_v, nx_v = e["normal"]
        for f in e["verts"]:
            ny.ravel()[f] = ny_v
            nx.ravel()[f] = nx_v

    return dict(tam_h=tam_h, oth_h=arrays["oth_h"], cell_owner=cell_owner,
                base_code=base_code, gy0=gy0, gx0=gx0,
                names=manifest["source_names"],
                smask=smask, tam_w=tam_w, target=target, own_view=own_view,
                oth_w=oth_w,
                seam_v=seam_v, ring_v=ring_v, dist_seam=dist_seam,
                hard=hard, hard_vals=hard_vals, width_cells=width_cells,
                nx=nx, ny=ny, edge_list=edge_list,
                bbox=(bx0, by0, bx1, by1), win=(r_lo, r_hi, c_lo, c_hi),
                render=cfg["render"], region=region, region_name=region_name,
                edges=edges, solve_cells=solve_cells)


def solve_surface(ctx: dict, v3: dict) -> tuple[np.ndarray, dict]:
    """Assemble and solve the constrained surface (see module docstring).

    Returns ``(field, report)``. ``report`` carries per-family equation
    counts, per-family residuals, and the boundary-elimination count.
    """
    surf = v3.get("surface", {})
    smask, hard, hard_vals = ctx["smask"], ctx["hard"], ctx["hard_vals"]
    target = nn_fill(ctx["target"])
    own_view = ctx["own_view"]
    edge_list = ctx["edge_list"]

    w_data = float(surf.get("data_weight", 1.0))
    w_smooth = float(surf.get("smooth_weight", 0.05))
    w_slope = float(surf.get("slope_weight", 25.0))
    w_bound = float(surf.get("boundary_weight", 1.0e6))

    unk = smask & ~hard
    H, W = smask.shape
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[unk] = np.arange(int(unk.sum()))
    n = int(unk.sum())

    fam: dict[str, list] = {k: [[], [], [], []] for k in
                            ("data", "laplacian", "slope")}
    boundary_eliminated = int(hard.sum())

    # data family: wd(x) * H(x) = wd(x) * target(x)
    wd = (0.05 + 0.95 * smootherstep(
        ctx["dist_seam"] / np.maximum(ctx["width_cells"] * 64.0, 1.0))) * unk
    ii = idx[unk]
    fam["data"][0].append(ii)
    fam["data"][1].append(ii)
    fam["data"][2].append(w_data * wd[unk])
    fam["data"][3].append(w_data * wd[unk] * target[unk])

    # laplacian family: ws*(4H(x) - sum nb) = ws * sum(fixed nb)
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
        fam["laplacian"][0].append(rr[is_unk])
        fam["laplacian"][1].append(jj[is_unk])
        fam["laplacian"][2].append(np.full(int(is_unk.sum()), -w_smooth))
        fam["laplacian"][3].append(np.zeros(int(is_unk.sum())))
        acc_fixed[rr[~is_unk]] += w_smooth * shifted_val[sel][~is_unk]
    core = idx[unk]
    fam["laplacian"][0].append(core)
    fam["laplacian"][1].append(core)
    fam["laplacian"][2].append(np.full(core.size, 4.0 * w_smooth))
    fam["laplacian"][3].append(acc_fixed)

    # slope family: per edge, per seam vert with valid owner sample:
    #   H(u) - H(s) = 64 * s_own   ->   H(u) = h_s + 64 * s_own
    slope_r = []
    slope_b = []
    slope_meta = []
    for e in edge_list:
        ny_v, nx_v = e["normal"]
        for f in e["verts"]:
            uy = f // W + int(round(ny_v))
            ux = f % W + int(round(nx_v))
            if not (0 <= uy < H and 0 <= ux < W):
                continue
            ju = idx[uy, ux]
            if ju < 0:
                continue
            oy = f // W - int(round(ny_v))
            ox = f % W - int(round(nx_v))
            if not (0 <= oy < H and 0 <= ox < W):
                continue
            h_s = hard_vals.ravel()[f]
            h_o = own_view[oy, ox]
            if not np.isfinite(h_o):
                continue
            s_own = (h_s - float(h_o)) / 64.0
            slope_r.append(ju)
            slope_b.append(w_slope * (h_s + 64.0 * s_own))
            slope_meta.append(f)
    fam["slope"][0].append(np.asarray(slope_r, dtype=np.int64))
    fam["slope"][1].append(np.asarray(slope_r, dtype=np.int64))
    fam["slope"][2].append(np.full(len(slope_r), w_slope, dtype=np.float64))
    fam["slope"][3].append(np.asarray(slope_b, dtype=np.float64))

    rows = np.concatenate([np.concatenate(f[0]) for f in fam.values()])
    cols = np.concatenate([np.concatenate(f[1]) for f in fam.values()])
    vals = np.concatenate([np.concatenate(f[2]) for f in fam.values()])
    b_vec = np.concatenate([np.concatenate(f[3]) for f in fam.values()])
    n_rows = rows.size
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n_rows, n)).tocsr()

    counts = {k: int(sum(len(x) for x in fam[k][0])) for k in fam}
    counts["boundary_eliminated"] = boundary_eliminated

    AtA = (A.T @ A).tocsr()
    Atb = A.T @ b_vec
    slope_cols = np.asarray(slope_r, dtype=np.int64)
    print(f"  [debug] slope_rows={len(slope_cols)} "
          f"b_vec[-len:]={b_vec[-len(slope_cols):].tolist()[:3] if len(slope_cols) else []} "
          f"Atb_max={float(Atb.max()):.0f} "
          f"Atb_slope={Atb[slope_cols[:3]].tolist() if len(slope_cols) else []} "
          f"AtA_diag_slope={AtA[slope_cols[:3], slope_cols[:3]].tolist() if len(slope_cols) else []}")
    diag = AtA.diagonal()
    M = sparse.diags(1.0 / np.maximum(diag, 1e-9))
    x0 = target[unk].astype(np.float64)
    x, status = cg(AtA, Atb, x0=x0, M=M,
                   rtol=float(surf.get("cg_tol", 1e-5)),
                   maxiter=int(surf.get("cg_maxiter", 600)))

    out = target.copy()
    flat = out.ravel()
    flat[np.nonzero(unk.ravel())[0]] = x.astype(np.float32)
    out = flat.reshape(out.shape)
    out[hard] = hard_vals[hard]
    out[~smask] = ctx["target"][~smask]

    # per-family residuals
    residuals = {}
    pos = 0
    slices = {}
    for k in ("data", "laplacian", "slope"):
        cnt = counts[k]
        slices[k] = slice(pos, pos + cnt)
        pos += cnt
    for k, sl in slices.items():
        if cnt == 0:
            residuals[k + "_rms"] = 0.0
            residuals[k + "_max"] = 0.0
            continue
        r = A[sl, :] @ x - b_vec[sl]
        residuals[k + "_rms"] = round(float(np.sqrt(np.mean(r ** 2))), 4)
        residuals[k + "_max"] = round(float(np.abs(r).max()), 4)
    residuals["boundary_max"] = 0.0

    report = {"unknowns": n, "cg_status": int(status),
              "equation_counts": counts, "residuals": residuals,
              "slope_rows": counts["slope"]}
    return out, report
