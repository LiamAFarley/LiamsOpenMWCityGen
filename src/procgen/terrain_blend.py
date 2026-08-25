"""Terrain seam context and the v3 harmonic correction solve.

Purpose
    Build the production seam representation from cell ownership and solve a
    bounded baseline transition on the generated side of each seam. The
    unknown is an additive correction ``C = H_final - H_target``. A direct
    discrete Laplace solve is used rather than least-squares Laplacian rows;
    squaring that operator would create a biharmonic surface that can ring
    beyond its boundary values.

Pipeline position
    v3 Milestone 2, after relief and before macro continuation or erosion.

Invariants
    Seam and outer-ring values are exact. First-inland owner slope samples are
    exact Dirichlet anchors. The correction solve has no data family and no
    ``A.T @ A`` normal-equation step. All writes are returned to callers; this
    module does not author plugins.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
try:
    import pyamg
except ImportError:  # optional local acceleration for the Poisson solve
    pyamg = None
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

from procgen.terrainfield import load_corpus, seam_edges
from procgen.terrain_relief import relief_config_hash
from procgen.terrain_inpaint import compose_authoritative_field


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def nn_fill(field: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(field)
    if not bad.any():
        return field
    idx = ndimage.distance_transform_edt(
        bad, return_distances=False, return_indices=True)
    return field[tuple(idx)]


def load_target(root: Path, cfg: dict, tam_h: np.ndarray) -> np.ndarray:
    """Load the relief target and reject a cache made from stale config."""
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
                        "regenerate it."
                    )
                return z["field"].astype(np.float32)
    raise SystemExit(
        "FAILURE: relief target cache is missing; run "
        "tools/terrain/relief_preview.py before the harmonic solve"
    )


def rasterize_seam(edges, solve_cells: set, shape, gy0: int, gx0: int,
                   r_lo: int, c_lo: int):
    """Rasterize authoritative production seam edges into a window.

    ``edges`` must come from :func:`terrainfield.seam_edges`. Each returned
    edge retains its own owner-to-Tamriel normal; corners are therefore not
    collapsed into one averaged normal for the solver.
    """
    seam_v = np.zeros(shape, dtype=bool)
    edge_list = []
    for a, b in edges:
        if a not in solve_cells:
            continue
        ry, rx = b[1] - gy0, b[0] - gx0
        dy_, dx_ = a[1] - b[1], a[0] - b[0]
        ey0, ex0 = ry * 64 - r_lo, rx * 64 - c_lo
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
    if not edge_list:
        raise ValueError("seam rasterization produced no edges")
    return seam_v, edge_list


def _cell_vertex_mask(cells: set[tuple[int, int]], shape, gy0: int, gx0: int):
    mask = np.zeros(shape, dtype=bool)
    for cx, cy in cells:
        r0, c0 = (cy - gy0) * 64, (cx - gx0) * 64
        r1, c1 = min(shape[0], r0 + 65), min(shape[1], c0 + 65)
        if r0 < shape[0] and c0 < shape[1] and r1 > 0 and c1 > 0:
            mask[max(0, r0):r1, max(0, c0):c1] = True
    return mask


def build_context(root: Path, cfg: dict, region_name: str) -> dict:
    """Build the production seam window and its bounded active corridor."""
    arrays, meta = load_corpus(root / cfg["paths"]["corpus_npz"])
    with open(root / cfg["paths"]["seam_atlas_json"], encoding="utf-8") as fh:
        atlas = json.load(fh)
    with open(root / cfg["paths"]["corpus_manifest"], encoding="utf-8") as fh:
        manifest = json.load(fh)
    base_code = manifest["source_names"].index(manifest["base_source"]) + 1
    tam_h_raw = arrays["tam_h"]
    cell_owner = arrays["cell_owner"]
    gy0, gx0 = int(meta["gy0"]), int(meta["gx0"])
    v3 = cfg["solve"].get("v3", {})
    region = cfg["solve"]["regions"][region_name]
    atlas_by_id = {r["cluster"]: r for r in atlas["clusters"]}
    cell_height_source = arrays.get("cell_height_source")
    if cell_height_source is None:
        raise SystemExit(
            "FAILURE: corpus lacks cell_height_source; rebuild the corpus"
        )
    raw_working = compose_authoritative_field(
        tam_h_raw, arrays["oth_h"], cell_height_source, base_code
    )
    target_full = load_target(root, cfg, raw_working)
    # The cached relief stage already includes the required-cell synthesis.
    # Keeping it as the context composite preserves scaled base terrain and
    # owner heights while preventing a second raw-field inpaint.
    tam_h = target_full
    own_full = compose_authoritative_field(
        target_full, arrays["oth_h"], cell_height_source, base_code
    )
    edges = seam_edges(cell_owner, base_code, gy0, gx0)
    retained = cell_owner == base_code
    blend = int(v3.get("blend_cells_max", 10))
    outer_apron_cells = float(v3.get("outer_apron_cells", 1.0))
    blend = max(
        blend,
        int(np.ceil(float(v3.get("blend_width_max_cells", 10.0))
                    + outer_apron_cells)),
    )

    cluster_boxes = []
    for cid in sorted(region["cluster_ids"]):
        x0, y0, x1, y1 = atlas_by_id[cid]["bbox_cells_xyxy"]
        cluster_boxes.append((x0 - 1, y0 - 1, x1 + 1, y1 + 1))
    review_bbox = region.get("review_bbox_cells")
    selection_boxes = [tuple(map(int, review_bbox))] if review_bbox else cluster_boxes
    def in_any_box(cell):
        x, y = cell
        return any(x0 <= x <= x1 and y0 <= y <= y1
                   for x0, y0, x1, y1 in selection_boxes)

    solve_cells: set[tuple[int, int]] = set()

    def expand_from(start):
        q = deque([(start, 0)])
        seen = {start}
        while q:
            cell, distance = q.popleft()
            solve_cells.add(cell)
            if distance >= blend:
                continue
            x, y = cell
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb in seen:
                    continue
                ry, rx = nb[1] - gy0, nb[0] - gx0
                if (0 <= ry < retained.shape[0] and 0 <= rx < retained.shape[1]
                        and retained[ry, rx]):
                    seen.add(nb)
                    q.append((nb, distance + 1))

    for a, b in edges:
        if a not in solve_cells and in_any_box(b):
            expand_from(a)
    if not solve_cells:
        raise SystemExit(f"FAILURE: no retained solve cells for region {region_name}")

    xs = [c[0] for c in solve_cells]
    ys = [c[1] for c in solve_cells]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
    pad = blend + 2
    r_lo = max(0, (by0 - gy0) * 64 - pad * 64)
    r_hi = min(tam_h.shape[0], (by1 - gy0) * 64 + 65 + pad * 64)
    c_lo = max(0, (bx0 - gx0) * 64 - pad * 64)
    c_hi = min(tam_h.shape[1], (bx1 - gx0) * 64 + 65 + pad * 64)

    solve_mask_full = _cell_vertex_mask(solve_cells, tam_h.shape, gy0, gx0)
    solve_mask_base = solve_mask_full[r_lo:r_hi, c_lo:c_hi]
    tam_w = tam_h[r_lo:r_hi, c_lo:c_hi]
    target = target_full[r_lo:r_hi, c_lo:c_hi]
    own_view = own_full[r_lo:r_hi, c_lo:c_hi]
    oth_w = arrays["oth_h"][r_lo:r_hi, c_lo:c_hi]
    # Feature analysis needs the effective authoritative owner footprint, not
    # the broader generated solve corridor. Cells whose height authority was
    # deliberately downgraded to the base source are not owner features.
    owner_cells = {
        (int(x + gx0), int(y + gy0))
        for y, x in zip(*np.nonzero(
            (cell_height_source != 0) & (cell_height_source != base_code)
        ))
    }
    owner_mask_full = _cell_vertex_mask(owner_cells, tam_h.shape, gy0, gx0)
    owner_mask = owner_mask_full[r_lo:r_hi, c_lo:c_hi]
    seam_v, edge_list = rasterize_seam(
        edges, solve_cells, solve_mask_base.shape, gy0, gx0, r_lo, c_lo
    )
    dist_seam = ndimage.distance_transform_edt(~seam_v)

    wmin = float(v3.get("blend_width_min_cells", 2.0))
    wmax = float(v3.get("blend_width_max_cells", 10.0))
    grade = float(v3.get("max_blend_grade_gu_per_cell", 2500.0))
    finite_target = np.where(np.isfinite(target), target, 0.0)
    loc_rel = np.abs(target - ndimage.uniform_filter(finite_target, 65))
    loc_rel[~solve_mask_base] = 0.0
    lo_r = (float(np.percentile(loc_rel[solve_mask_base], 20))
            if solve_mask_base.any() else 0.0)
    hi_r = (float(np.percentile(loc_rel[solve_mask_base], 95))
            if solve_mask_base.any() else 1.0)
    rn = np.clip((loc_rel - lo_r) / max(hi_r - lo_r, 1e-6), 0.0, 1.0)
    seam_width = np.full(solve_mask_base.shape, np.nan, dtype=np.float32)
    seam_width[seam_v] = wmin
    seam_valid = seam_v & np.isfinite(own_view) & np.isfinite(target)
    seam_delta = np.abs(own_view[seam_valid] - target[seam_valid])
    seam_mismatch_width = seam_delta / max(grade, 1.0)
    seam_relief_width = wmin + (wmax - wmin) * rn[seam_valid]
    seam_width[seam_valid] = np.clip(
        np.maximum(seam_mismatch_width, seam_relief_width),
        wmin,
        wmax,
    )
    nearest = ndimage.distance_transform_edt(
        ~seam_v, return_distances=False, return_indices=True
    )
    requested_width_cells = np.clip(seam_width[tuple(nearest)], wmin, wmax)
    solve_width_cells = np.minimum(
        requested_width_cells + outer_apron_cells,
        wmax + outer_apron_cells,
    )
    active_mask = solve_mask_base & (
        dist_seam <= solve_width_cells * 64.0
    )
    active_mask |= seam_v
    # Pixelwise adaptive widths can leave one-pixel islands at diagonal
    # ownership corners. Keep only 4-connected components that contain seam
    # vertices; the harmonic operator itself is 4-neighbor coupled.
    labels, _ = ndimage.label(
        active_mask,
        structure=ndimage.generate_binary_structure(2, 1),
    )
    seam_labels = np.unique(labels[seam_v])
    connected = np.isin(labels, seam_labels)
    mask_islands_removed = int(np.count_nonzero(active_mask & ~connected))
    active_mask = connected
    # Every non-seam active vertex touching inactive terrain is an exact outer
    # Dirichlet boundary. The distance edge is only the requested transition
    # limit; treating the rest of the active-mask perimeter as no-flux would
    # hand a nonzero correction directly back to untouched terrain.
    active_interior = ndimage.binary_erosion(
        active_mask,
        structure=ndimage.generate_binary_structure(2, 1),
        border_value=0,
    )
    outer_v = active_mask & ~seam_v & ~active_interior
    hard = seam_v | outer_v
    hard_vals = np.zeros_like(target, dtype=np.float32)
    hard_vals[outer_v] = target[outer_v]
    hard_vals[seam_v] = own_view[seam_v]
    unknown = active_mask & ~hard
    if np.any(unknown & ~active_interior):
        bad = np.argwhere(unknown & ~active_interior)[:20].tolist()
        raise AssertionError(
            "unknown harmonic vertices touch inactive terrain: "
            f"{bad}"
        )

    ny = np.zeros(active_mask.shape, np.float32)
    nx = np.zeros(active_mask.shape, np.float32)
    for edge in edge_list:
        ny_v, nx_v = edge["normal"]
        for flat in edge["verts"]:
            ny.ravel()[flat] = ny_v
            nx.ravel()[flat] = nx_v

    return dict(
        tam_h=tam_h, tam_h_raw=tam_h_raw, oth_h=arrays["oth_h"],
        cell_owner=cell_owner, cell_height_source=cell_height_source,
        own_full=own_full,
        base_code=base_code, gy0=gy0, gx0=gx0,
        names=manifest["source_names"], smask=active_mask, tam_w=tam_w,
        target=target, own_view=own_view, oth_w=oth_w,
        owner_mask=owner_mask, owner_field=own_view,
        seam_v=seam_v,
        ring_v=outer_v, dist_seam=dist_seam, hard=hard, hard_vals=hard_vals,
        width_cells=requested_width_cells,
        solve_width_cells=solve_width_cells,
        nx=nx, ny=ny, edge_list=edge_list,
        bbox=(bx0, by0, bx1, by1), win=(r_lo, r_hi, c_lo, c_hi),
        render=cfg["render"], region=region, region_name=region_name,
        edges=edges, solve_cells=solve_cells,
        mask_islands_removed=mask_islands_removed,
    )


def solve_harmonic_correction(
    target: np.ndarray,
    active_mask: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_final_height: np.ndarray,
    *,
    linear_solver: str = "amg_rs_cg",
    cg_tol: float = 1e-6,
    cg_maxiter: int = 200,
    amg_max_coarse: int = 500,
) -> tuple[np.ndarray, dict]:
    """Solve ``L_uu C_u = rhs`` directly for an additive correction.

    Neighbors outside ``active_mask`` are no-flux boundaries. Neighbors inside
    the active domain are either unknowns or exact Dirichlet values. This
    direct Laplace system preserves the discrete maximum principle for the
    correction and avoids the ``L.T @ L`` biharmonic operator.
    """
    target = nn_fill(target).astype(np.float64)
    active = np.asarray(active_mask, dtype=bool)
    fixed = np.asarray(fixed_mask, dtype=bool)
    if np.any(fixed & ~active):
        raise ValueError("fixed vertices must lie inside the active domain")
    correction_fixed = np.zeros_like(target, dtype=np.float64)
    correction_fixed[fixed] = (
        fixed_final_height[fixed].astype(np.float64) - target[fixed]
    )
    unknown = active & ~fixed
    H, W = active.shape
    idx = np.full((H, W), -1, dtype=np.int64)
    n = int(unknown.sum())
    if n == 0:
        correction = correction_fixed
        final = target + correction
        final[fixed] = fixed_final_height[fixed]
        return final.astype(np.float32), {
            "unknowns": 0, "cg_status": 0,
            "linear_solver": linear_solver,
            "cg_iterations": 0,
            "solver_setup_s": 0.0,
            "solver_solve_s": 0.0,
            "equation_counts": {"data": 0, "laplacian": 0, "slope": 0,
                                 "slope_anchors": 0,
                                 "boundary_eliminated": int(fixed.sum())},
            "residuals": {"data_rms": 0.0, "data_max": 0.0,
                           "laplacian_rms": 0.0, "laplacian_max": 0.0,
                           "slope_rms": 0.0, "slope_max": 0.0,
                           "data_weighted_rms": 0.0,
                           "laplacian_weighted_rms": 0.0,
                           "slope_weighted_rms": 0.0, "boundary_max": 0.0},
            "assembly": {"rows": 0, "nnz": 0, "empty_equation_rows": 0},
            "correction_bounds": {"min": 0.0, "max": 0.0,
                                   "fixed_min": 0.0, "fixed_max": 0.0},
        }
    idx[unknown] = np.arange(n, dtype=np.int64)
    row_parts = []
    col_parts = []
    val_parts = []
    rhs = np.zeros(n, dtype=np.float64)
    degree = np.zeros(n, dtype=np.float64)

    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        valid_center = unknown.copy()
        if dy == 1:
            valid_center[-1, :] = False
        elif dy == -1:
            valid_center[0, :] = False
        if dx == 1:
            valid_center[:, -1] = False
        elif dx == -1:
            valid_center[:, 0] = False
        neighbor_active = np.roll(active, shift=(-dy, -dx), axis=(0, 1))
        neighbor_fixed = np.roll(fixed, shift=(-dy, -dx), axis=(0, 1))
        neighbor_idx = np.roll(idx, shift=(-dy, -dx), axis=(0, 1))
        neighbor_correction = np.roll(
            correction_fixed, shift=(-dy, -dx), axis=(0, 1)
        )
        valid = valid_center & neighbor_active
        eq_ids = idx[valid]
        nb_ids = neighbor_idx[valid]
        np.add.at(degree, eq_ids, 1.0)
        is_unknown = nb_ids >= 0
        if np.any(is_unknown):
            row_parts.append(eq_ids[is_unknown])
            col_parts.append(nb_ids[is_unknown])
            val_parts.append(np.full(int(is_unknown.sum()), -1.0))
        is_fixed = neighbor_fixed[valid]
        if np.any(is_fixed):
            np.add.at(
                rhs, eq_ids[is_fixed],
                neighbor_correction[valid][is_fixed],
            )

    if np.any(degree <= 0.0):
        bad = np.nonzero(degree <= 0.0)[0][:20]
        raise ValueError(f"harmonic system has isolated unknowns: {bad.tolist()}")
    row_parts.append(np.arange(n, dtype=np.int64))
    col_parts.append(np.arange(n, dtype=np.int64))
    val_parts.append(degree)
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    vals = np.concatenate(val_parts)
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    empty_rows = int(np.count_nonzero(np.diff(A.indptr) == 0))
    if empty_rows:
        raise AssertionError(f"harmonic system contains {empty_rows} empty rows")
    setup_t0 = time.perf_counter()
    ml = None
    if linear_solver == "amg_rs_cg":
        if pyamg is None:
            raise RuntimeError("linear_solver='amg_rs_cg' requires pyamg")
        ml = pyamg.ruge_stuben_solver(
            A,
            max_coarse=int(amg_max_coarse),
        )
        M = ml.aspreconditioner(cycle="V")
    elif linear_solver == "amg_sa_cg":
        if pyamg is None:
            raise RuntimeError("linear_solver='amg_sa_cg' requires pyamg")
        ml = pyamg.smoothed_aggregation_solver(
            A,
            symmetry="symmetric",
            max_coarse=int(amg_max_coarse),
        )
        M = ml.aspreconditioner(cycle="V")
    elif linear_solver == "jacobi_cg":
        diag = A.diagonal()
        if np.any(diag <= 0.0):
            bad = np.nonzero(diag <= 0.0)[0][:20]
            raise ValueError(
                "non-positive harmonic diagonal at "
                f"{bad.tolist()}"
            )
        M = sparse.diags(1.0 / diag)
    else:
        raise ValueError(f"unknown linear_solver={linear_solver!r}")
    solver_setup_s = time.perf_counter() - setup_t0

    iterations = 0

    def _cg_callback(_):
        nonlocal iterations
        iterations += 1

    solve_t0 = time.perf_counter()
    c, status = cg(
        A,
        rhs,
        x0=np.zeros(n, dtype=np.float64),
        M=M,
        rtol=float(cg_tol),
        atol=0.0,
        maxiter=int(cg_maxiter),
        callback=_cg_callback,
    )
    solver_solve_s = time.perf_counter() - solve_t0
    if status != 0:
        raise RuntimeError(f"harmonic CG failed with status {status}")
    if not np.all(np.isfinite(c)):
        raise FloatingPointError("harmonic CG returned non-finite correction")
    correction = correction_fixed.copy()
    correction[unknown] = c
    final = (target + correction).astype(np.float32)
    final[fixed] = fixed_final_height[fixed]
    residual = A @ c - rhs
    fixed_corr = correction_fixed[fixed]
    active_corr = correction[active]
    bounds = {
        "min": float(np.min(active_corr)),
        "max": float(np.max(active_corr)),
        "fixed_min": float(np.min(fixed_corr)) if fixed_corr.size else 0.0,
        "fixed_max": float(np.max(fixed_corr)) if fixed_corr.size else 0.0,
    }
    return final, {
        "unknowns": n, "cg_status": int(status),
        "linear_solver": linear_solver,
        "cg_iterations": int(iterations),
        "solver_setup_s": round(solver_setup_s, 4),
        "solver_solve_s": round(solver_solve_s, 4),
        **({"amg_levels": int(len(ml.levels))} if ml is not None else {}),
        "equation_counts": {"data": 0, "laplacian": n, "slope": 0,
                             "slope_anchors": 0,
                             "boundary_eliminated": int(fixed.sum())},
        "residuals": {
            "data_rms": 0.0, "data_max": 0.0,
            "laplacian_rms": float(np.sqrt(np.mean(residual ** 2))),
            "laplacian_max": float(np.max(np.abs(residual))),
            "slope_rms": 0.0, "slope_max": 0.0,
            "data_weighted_rms": 0.0,
            "laplacian_weighted_rms": float(np.sqrt(np.mean(residual ** 2))),
            "slope_weighted_rms": 0.0, "boundary_max": 0.0,
        },
        "assembly": {"rows": n, "nnz": int(A.nnz),
                      "empty_equation_rows": empty_rows},
        "correction_bounds": bounds,
    }


def solve_surface(ctx: dict, v3: dict) -> tuple[np.ndarray, dict]:
    """Adapt a production seam context to the direct harmonic solver."""
    surf = v3.get("surface", {})
    target = nn_fill(ctx["target"]).astype(np.float64)
    active = ctx["smask"]
    fixed = ctx["hard"].copy()
    fixed_final = ctx["hard_vals"].astype(np.float64).copy()
    anchors = np.zeros_like(fixed, dtype=bool)
    anchor_values = np.zeros_like(target, dtype=np.float64)
    anchor_claims: dict[tuple[int, int], list[float]] = {}
    edge_incidence: dict[int, list[tuple[int, int]]] = {}
    for edge in ctx["edge_list"]:
        normal = (
            int(round(edge["normal"][0])),
            int(round(edge["normal"][1])),
        )
        for flat in edge["verts"]:
            edge_incidence.setdefault(int(flat), []).append(normal)
    for flat in edge_incidence:
        edge_incidence[flat] = list(dict.fromkeys(edge_incidence[flat]))
    corner_vertices = {
        flat: normals
        for flat, normals in edge_incidence.items()
        if len(normals) == 2
        and normals[0][0] * normals[1][0]
        + normals[0][1] * normals[1][1] == 0
    }
    corner_diagonal_vertices: set[tuple[int, int]] = set()
    corner_anchor_values: dict[tuple[int, int], float] = {}
    corner_anchor_skipped = 0
    for flat, normals in corner_vertices.items():
        sy, sx = divmod(flat, target.shape[1])
        dy1, dx1 = normals[0]
        dy2, dx2 = normals[1]
        h_corner = float(fixed_final[sy, sx])
        # Each corner leg gets its own first-inland continuation. The
        # diagonal point carries both owner slopes; claims are never averaged.
        for dy, dx in normals:
            uy, ux = sy + dy, sx + dx
            oy, ox = sy - dy, sx - dx
            if not (0 <= uy < target.shape[0] and 0 <= ux < target.shape[1]
                    and 0 <= oy < target.shape[0] and 0 <= ox < target.shape[1]):
                corner_anchor_skipped += 1
                continue
            owner_h = float(ctx["own_view"][oy, ox])
            if not np.isfinite(owner_h) or not active[uy, ux] or fixed[uy, ux]:
                corner_anchor_skipped += 1
                continue
            corner_anchor_values[(uy, ux)] = (
                h_corner + (h_corner - owner_h) - target[uy, ux]
            )
        uy, ux = sy + dy1 + dy2, sx + dx1 + dx2
        if (0 <= uy < target.shape[0] and 0 <= ux < target.shape[1]
                and active[uy, ux] and not fixed[uy, ux]):
            corner_diagonal_vertices.add((uy, ux))
            oy1, ox1 = sy - dy1, sx - dx1
            oy2, ox2 = sy - dy2, sx - dx2
            if (0 <= oy1 < target.shape[0] and 0 <= ox1 < target.shape[1]
                    and 0 <= oy2 < target.shape[0] and 0 <= ox2 < target.shape[1]):
                owner1 = float(ctx["own_view"][oy1, ox1])
                owner2 = float(ctx["own_view"][oy2, ox2])
                if np.isfinite(owner1) and np.isfinite(owner2):
                    corner_anchor_values[(uy, ux)] = (
                        h_corner + (h_corner - owner1)
                        + (h_corner - owner2) - target[uy, ux]
                    )
                else:
                    corner_anchor_skipped += 1
        else:
            corner_anchor_skipped += 1
    ordinary_seam_samples_eligible = 0
    ordinary_anchors_created = 0
    ordinary_anchors_skipped_inactive = 0
    ordinary_anchors_skipped_invalid_owner = 0
    ordinary_anchors_skipped_boundary = 0
    conflict_tol = float(surf.get("anchor_conflict_tolerance_gu", 0.0))
    conflict_policy = str(surf.get("anchor_conflict_policy", "error"))
    if conflict_policy not in {"error", "skip"}:
        raise ValueError(f"unknown anchor_conflict_policy: {conflict_policy}")
    fixed_conflict_spread = 0.0
    fixed_conflicts = []

    for edge in ctx["edge_list"]:
        dy = int(round(edge["normal"][0]))
        dx = int(round(edge["normal"][1]))
        for flat in edge["verts"]:
            if int(flat) in corner_vertices:
                continue
            sy, sx = divmod(flat, target.shape[1])
            uy, ux = sy + dy, sx + dx
            oy, ox = sy - dy, sx - dx
            if not (0 <= uy < target.shape[0] and 0 <= ux < target.shape[1]
                    and 0 <= oy < target.shape[0] and 0 <= ox < target.shape[1]):
                continue
            if not np.isfinite(ctx["own_view"][oy, ox]):
                ordinary_anchors_skipped_invalid_owner += 1
                continue
            ordinary_seam_samples_eligible += 1
            if not active[uy, ux]:
                ordinary_anchors_skipped_inactive += 1
                continue
            if (uy, ux) in corner_diagonal_vertices:
                continue
            if (uy, ux) in corner_anchor_values:
                continue
            h_seam = float(fixed_final[sy, sx])
            desired_height = h_seam + (h_seam - float(ctx["own_view"][oy, ox]))
            desired = desired_height - target[uy, ux]
            if fixed[uy, ux]:
                if ctx["ring_v"][uy, ux]:
                    ordinary_anchors_skipped_boundary += 1
                    continue
                existing = fixed_final[uy, ux] - target[uy, ux]
                fixed_conflict_spread = max(
                    fixed_conflict_spread, abs(existing - desired)
                )
                if abs(existing - desired) > conflict_tol:
                    fixed_conflicts.append({
                        "vertex": [int(uy), int(ux)],
                        "spread_gu": abs(existing - desired),
                    })
                continue
            anchor_claims.setdefault((uy, ux), []).append(desired)
            ordinary_anchors_created += 1
    anchor_spread_max = 0.0
    anchor_conflicts = []
    for (uy, ux), all_claims in anchor_claims.items():
        claims = all_claims
        if fixed[uy, ux]:
            existing = fixed_final[uy, ux] - target[uy, ux]
            spread = max(abs(existing - value) for value in claims)
            fixed_conflict_spread = max(fixed_conflict_spread, spread)
            if spread > conflict_tol:
                fixed_conflicts.append({
                    "vertex": [int(uy), int(ux)],
                    "spread_gu": float(spread),
                    "claims_gu": [float(v) for v in claims],
                })
            continue
        spread = max(claims) - min(claims)
        anchor_spread_max = max(anchor_spread_max, spread)
        if spread > conflict_tol:
            anchor_conflicts.append({
                "vertex": [int(uy), int(ux)],
                "spread_gu": float(spread),
                "claims_gu": [float(v) for v in claims],
            })
            continue
        anchors[uy, ux] = True
        anchor_values[uy, ux] = float(np.mean(claims))
    for (uy, ux), value in corner_anchor_values.items():
        anchors[uy, ux] = True
        anchor_values[uy, ux] = value
    if (anchor_conflicts or fixed_conflicts) and conflict_policy == "error":
        first = (anchor_conflicts + fixed_conflicts)[0]
        raise ValueError(
            f"owner slope anchor conflict at {first['vertex']}: "
            f"spread {first['spread_gu']} GU > {conflict_tol} GU"
        )
    fixed |= anchors
    fixed_final[anchors] = target[anchors] + anchor_values[anchors]
    field, report = solve_harmonic_correction(
        target, active, fixed, fixed_final,
        linear_solver=str(surf.get("linear_solver", "amg_rs_cg")),
        cg_tol=float(surf.get("cg_tol", 1e-6)),
        cg_maxiter=int(surf.get("cg_maxiter", 200)),
        amg_max_coarse=int(surf.get("amg_max_coarse", 500)),
    )
    report["equation_counts"]["slope_anchors"] = int(anchors.sum())
    report["slope_rows"] = int(anchors.sum())
    report["anchor_count"] = int(anchors.sum())
    report["fixed_count"] = int(fixed.sum())
    report["anchor_conflict_tolerance_gu"] = conflict_tol
    report["anchor_spread_max_gu"] = max(anchor_spread_max,
                                          fixed_conflict_spread)
    report["anchor_conflict_policy"] = conflict_policy
    report["anchor_conflicts"] = anchor_conflicts + fixed_conflicts
    report["mask_islands_removed"] = int(ctx.get("mask_islands_removed", 0))
    report["corner_vertices"] = int(len(corner_vertices))
    report["corner_anchors"] = int(len(corner_anchor_values))
    report["corner_anchors_skipped"] = int(corner_anchor_skipped)
    report["ordinary_seam_samples_eligible"] = int(
        ordinary_seam_samples_eligible
    )
    report["ordinary_anchors_created"] = int(ordinary_anchors_created)
    report["ordinary_anchors_skipped_inactive"] = int(
        ordinary_anchors_skipped_inactive
    )
    report["ordinary_anchors_skipped_invalid_owner"] = int(
        ordinary_anchors_skipped_invalid_owner
    )
    report["ordinary_anchors_skipped_boundary"] = int(
        ordinary_anchors_skipped_boundary
    )
    report["active_boundary_vertices"] = int(ctx["ring_v"].sum())
    report["active_vertices"] = int(active.sum())
    field[ctx["hard"]] = ctx["hard_vals"][ctx["hard"]]
    field[~active] = ctx["target"][~active]
    return field, report
