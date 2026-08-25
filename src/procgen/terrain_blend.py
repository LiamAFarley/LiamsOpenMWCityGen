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
    """Solve the v3 constrained seam surface.

    IMPORTANT ASSEMBLY RULE:
    Matrix ROW indices are equation IDs.
    Matrix COLUMN indices are unknown-height IDs.

    They are NOT interchangeable.

    Families:
      data:
          one equation per unknown

      laplacian:
          one equation per unknown, containing multiple coefficients

      slope:
          one equation per valid owner->Tamriel seam derivative constraint

    Hard seam/ring vertices are eliminated from the unknown vector and enter
    neighboring equations through their RHS values.
    """
    surf = v3.get("surface", {})

    smask = ctx["smask"]
    hard = ctx["hard"]
    hard_vals = ctx["hard_vals"]
    target = nn_fill(ctx["target"]).astype(np.float64)
    own_view = ctx["own_view"]
    edge_list = ctx["edge_list"]

    # Accept the old target_weight key as a fallback while migrating config,
    # but production config should use data_weight explicitly.
    w_data = float(
        surf.get("data_weight", surf.get("target_weight", 1.0))
    )
    w_smooth = float(surf.get("smooth_weight", 0.05))
    w_slope = float(surf.get("slope_weight", 25.0))

    if w_data <= 0.0:
        raise ValueError("surface.data_weight must be > 0")
    if w_smooth <= 0.0:
        raise ValueError("surface.smooth_weight must be > 0")
    if w_slope <= 0.0:
        raise ValueError("surface.slope_weight must be > 0")

    # ------------------------------------------------------------------
    # Unknown indexing
    # ------------------------------------------------------------------

    unk = smask & ~hard
    H, W = smask.shape

    idx = np.full((H, W), -1, dtype=np.int64)
    n = int(unk.sum())

    if n == 0:
        out = target.astype(np.float32)
        out[hard] = hard_vals[hard]
        out[~smask] = ctx["target"][~smask]
        return out, {
            "unknowns": 0,
            "cg_status": 0,
            "equation_counts": {
                "data": 0,
                "laplacian": 0,
                "slope": 0,
                "boundary_eliminated": int(hard.sum()),
            },
            "residuals": {
                "data_rms": 0.0,
                "data_max": 0.0,
                "laplacian_rms": 0.0,
                "laplacian_max": 0.0,
                "slope_rms": 0.0,
                "slope_max": 0.0,
                "data_weighted_rms": 0.0,
                "laplacian_weighted_rms": 0.0,
                "slope_weighted_rms": 0.0,
                "boundary_max": 0.0,
            },
            "slope_rows": 0,
            "assembly": {
                "rows": 0,
                "nnz": 0,
                "empty_equation_rows": 0,
            },
        }

    idx[unk] = np.arange(n, dtype=np.int64)

    # Unknown IDs themselves are 0..n-1.
    unknown_ids = idx[unk]

    # ==================================================================
    # FAMILY 1: DATA
    #
    # One equation per unknown:
    #
    #     w(x) H_i = w(x) target_i
    #
    # Local equation IDs are 0..n-1.
    # ==================================================================

    wd_field = (
        0.05
        + 0.95
        * smootherstep(
            ctx["dist_seam"]
            / np.maximum(ctx["width_cells"] * 64.0, 1.0)
        )
    )

    data_coeff = (
        w_data * wd_field[unk].astype(np.float64)
    )

    data_rows_local = np.arange(n, dtype=np.int64)
    data_cols = unknown_ids.copy()
    data_vals = data_coeff

    b_data = (
        data_coeff
        * target[unk].astype(np.float64)
    )

    n_data_eq = n

    # ==================================================================
    # FAMILY 2: LAPLACIAN
    #
    # Exactly ONE equation per unknown:
    #
    #   degree_i * H_i - sum(H_unknown_neighbor)
    #       = sum(H_fixed_neighbor)
    #
    # multiplied by w_smooth.
    #
    # A Laplacian equation has several NONZEROS but one RHS.
    # ==================================================================

    lap_row_parts: list[np.ndarray] = []
    lap_col_parts: list[np.ndarray] = []
    lap_val_parts: list[np.ndarray] = []

    b_lap = np.zeros(n, dtype=np.float64)
    lap_degree = np.zeros(n, dtype=np.float64)

    for dy, dx in (
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
    ):
        # sel identifies unknown centers for which this neighbor lies
        # inside the array.
        sel = unk.copy()

        if dy == 1:
            sel[-1, :] = False
        elif dy == -1:
            sel[0, :] = False

        if dx == 1:
            sel[:, -1] = False
        elif dx == -1:
            sel[:, 0] = False

        # IMPORTANT:
        # np.roll(a, -dy) makes a[y] receive original a[y + dy].
        #
        # The previous implementation used the opposite shift direction.
        neighbor_idx = np.roll(
            idx,
            shift=(-dy, -dx),
            axis=(0, 1),
        )
        neighbor_hard = np.roll(
            hard,
            shift=(-dy, -dx),
            axis=(0, 1),
        )
        neighbor_hard_vals = np.roll(
            hard_vals,
            shift=(-dy, -dx),
            axis=(0, 1),
        )

        eq_ids = idx[sel]
        nb_ids = neighbor_idx[sel]
        nb_is_unknown = nb_ids >= 0
        nb_is_hard = neighbor_hard[sel]

        # Because the solve corridor has a hard outer ring, every
        # non-unknown neighbor of an unknown should be hard.
        #
        # If this fires, the corridor topology itself is broken and we
        # should NOT silently treat a missing neighbor as height zero.
        bad_neighbor = (~nb_is_unknown) & (~nb_is_hard)
        if np.any(bad_neighbor):
            bad_count = int(np.count_nonzero(bad_neighbor))
            raise AssertionError(
                "Laplacian assembly found "
                f"{bad_count} unknown->non-hard/non-unknown neighbor(s). "
                "The solve corridor must be enclosed by hard boundary "
                "vertices."
            )

        # Every valid neighboring raster edge contributes to degree.
        np.add.at(
            lap_degree,
            eq_ids,
            1.0,
        )

        # Unknown neighbor:
        #
        #     ... - w_smooth * H_j
        #
        if np.any(nb_is_unknown):
            lap_row_parts.append(
                eq_ids[nb_is_unknown].astype(np.int64)
            )
            lap_col_parts.append(
                nb_ids[nb_is_unknown].astype(np.int64)
            )
            lap_val_parts.append(
                np.full(
                    int(np.count_nonzero(nb_is_unknown)),
                    -w_smooth,
                    dtype=np.float64,
                )
            )

        # Hard neighbor:
        #
        #     ... = + w_smooth * H_fixed
        #
        if np.any(nb_is_hard):
            fixed_eq = eq_ids[nb_is_hard]
            fixed_val = neighbor_hard_vals[sel][nb_is_hard]

            np.add.at(
                b_lap,
                fixed_eq,
                w_smooth * fixed_val.astype(np.float64),
            )

    # Diagonal:
    #
    #     degree_i * w_smooth * H_i
    #
    lap_row_parts.append(
        np.arange(n, dtype=np.int64)
    )
    lap_col_parts.append(
        np.arange(n, dtype=np.int64)
    )
    lap_val_parts.append(
        w_smooth * lap_degree
    )

    lap_rows_local = np.concatenate(lap_row_parts)
    lap_cols = np.concatenate(lap_col_parts)
    lap_vals = np.concatenate(lap_val_parts)

    n_lap_eq = n

    # ==================================================================
    # FAMILY 3: OWNER NORMAL-SLOPE CONTINUATION
    #
    # One equation for each valid seam edge sample:
    #
    #     H_inland = H_seam + (H_seam - H_owner_outward)
    #
    # This is exactly a one-vertex continuation of the owner's normal
    # derivative.
    #
    # DO NOT divide by 64 and then multiply by 64. Adjacent samples are
    # one raster edge apart.
    # ==================================================================

    slope_cols_list: list[int] = []
    slope_desired_list: list[float] = []
    slope_seam_meta: list[int] = []

    hard_flat = hard_vals.ravel()

    for edge in edge_list:
        normal_y, normal_x = edge["normal"]

        dy = int(round(normal_y))
        dx = int(round(normal_x))

        for seam_flat in edge["verts"]:
            sy = seam_flat // W
            sx = seam_flat % W

            # One vertex inward, on generated/Tamriel side.
            uy = sy + dy
            ux = sx + dx

            if not (0 <= uy < H and 0 <= ux < W):
                continue

            inland_unknown = idx[uy, ux]

            # If the immediately-inland point is itself hard/ring/seam,
            # there is no unknown to constrain.
            if inland_unknown < 0:
                continue

            # One vertex outward, on owner side.
            oy = sy - dy
            ox = sx - dx

            if not (0 <= oy < H and 0 <= ox < W):
                continue

            h_owner_out = float(own_view[oy, ox])

            if not np.isfinite(h_owner_out):
                continue

            h_seam = float(hard_flat[seam_flat])

            if not np.isfinite(h_seam):
                continue

            # Continue the one-edge owner derivative through the seam:
            #
            # owner_out -> seam = (h_seam - h_owner_out)
            #
            # so:
            #
            # seam -> inland should initially have the same delta.
            desired_inland = (
                h_seam
                + (h_seam - h_owner_out)
            )

            slope_cols_list.append(
                int(inland_unknown)
            )
            slope_desired_list.append(
                float(desired_inland)
            )
            slope_seam_meta.append(
                int(seam_flat)
            )

    slope_cols = np.asarray(
        slope_cols_list,
        dtype=np.int64,
    )
    slope_desired = np.asarray(
        slope_desired_list,
        dtype=np.float64,
    )

    n_slope_eq = int(slope_cols.size)

    # Local slope equation IDs are 0..m-1.
    slope_rows_local = np.arange(
        n_slope_eq,
        dtype=np.int64,
    )

    slope_vals = np.full(
        n_slope_eq,
        w_slope,
        dtype=np.float64,
    )

    b_slope = (
        w_slope * slope_desired
    )

    # ==================================================================
    # GLOBAL EQUATION-ROW OFFSETS
    #
    # THIS IS THE PART THE OLD ASSEMBLER WAS MISSING.
    #
    # data rows:
    #       [0, n)
    #
    # laplacian rows:
    #       [n, 2n)
    #
    # slope rows:
    #       [2n, 2n+m)
    #
    # Columns ALWAYS remain unknown IDs [0, n).
    # ==================================================================

    data_offset = 0
    lap_offset = n_data_eq
    slope_offset = n_data_eq + n_lap_eq

    data_rows = (
        data_rows_local + data_offset
    )
    lap_rows = (
        lap_rows_local + lap_offset
    )
    slope_rows = (
        slope_rows_local + slope_offset
    )

    n_rows = (
        n_data_eq
        + n_lap_eq
        + n_slope_eq
    )

    rows = np.concatenate(
        [
            data_rows,
            lap_rows,
            slope_rows,
        ]
    )

    cols = np.concatenate(
        [
            data_cols,
            lap_cols,
            slope_cols,
        ]
    )

    vals = np.concatenate(
        [
            data_vals,
            lap_vals,
            slope_vals,
        ]
    )

    # ONE RHS value per EQUATION, not per matrix nonzero.
    b_vec = np.concatenate(
        [
            b_data,
            b_lap,
            b_slope,
        ]
    )

    # ==================================================================
    # HARD ASSEMBLY INVARIANTS
    # ==================================================================

    if not (
        rows.size
        == cols.size
        == vals.size
    ):
        raise AssertionError(
            "Sparse COO arrays have inconsistent lengths: "
            f"rows={rows.size}, cols={cols.size}, vals={vals.size}"
        )

    if b_vec.size != n_rows:
        raise AssertionError(
            "RHS length must equal number of EQUATIONS: "
            f"len(b)={b_vec.size}, n_rows={n_rows}"
        )

    if rows.size:
        if int(rows.min()) < 0:
            raise AssertionError(
                f"negative equation row index {int(rows.min())}"
            )
        if int(rows.max()) >= n_rows:
            raise AssertionError(
                "equation row index outside matrix: "
                f"max={int(rows.max())}, n_rows={n_rows}"
            )

    if cols.size:
        if int(cols.min()) < 0:
            raise AssertionError(
                f"negative unknown column index {int(cols.min())}"
            )
        if int(cols.max()) >= n:
            raise AssertionError(
                "unknown column index outside matrix: "
                f"max={int(cols.max())}, n={n}"
            )

    A = sparse.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_rows, n),
        dtype=np.float64,
    ).tocsr()

    # Every equation in this implementation must contain at least one
    # coefficient. The old broken assembler produced huge numbers of
    # EMPTY equation rows; this catches that exact failure.
    row_nnz = np.diff(A.indptr)
    empty_rows = int(np.count_nonzero(row_nnz == 0))

    if empty_rows != 0:
        empty_idx = np.nonzero(row_nnz == 0)[0][:20]
        raise AssertionError(
            "Sparse system contains empty equation rows. "
            f"count={empty_rows}, first={empty_idx.tolist()}"
        )

    # Explicit equation slices. Do NOT infer them from nnz counts.
    data_slice = slice(
        data_offset,
        data_offset + n_data_eq,
    )
    lap_slice = slice(
        lap_offset,
        lap_offset + n_lap_eq,
    )
    slope_slice = slice(
        slope_offset,
        slope_offset + n_slope_eq,
    )

    # ==================================================================
    # NORMAL EQUATIONS
    # ==================================================================

    AtA = (A.T @ A).tocsr()
    Atb = A.T @ b_vec

    # Diagnostic specifically for the historical failure.
    #
    # For flat_step:
    #
    # coefficient = 25
    # b = 250000
    #
    # so each unique slope equation should contribute about 6.25e6 to
    # its unknown before other families are added.
    if n_slope_eq:
        A_slope = A[slope_slice, :]
        b_slope_check = b_vec[slope_slice]

        Atb_slope_only = (
            A_slope.T @ b_slope_check
        )

        sample_cols = slope_cols[:3]

        print(
            "  [assembly] "
            f"unknowns={n} "
            f"eqs={n_rows} "
            f"nnz={A.nnz} "
            f"empty_rows={empty_rows}"
        )

        print(
            "  [assembly] "
            f"data_eq={n_data_eq} "
            f"lap_eq={n_lap_eq} "
            f"slope_eq={n_slope_eq}"
        )

        print(
            "  [assembly] "
            f"slope_rhs_sample="
            f"{b_slope_check[:3].tolist()} "
            f"slope_Atb_only_sample="
            f"{Atb_slope_only[sample_cols].tolist()} "
            f"total_Atb_sample="
            f"{Atb[sample_cols].tolist()}"
        )

    # ==================================================================
    # SOLVE
    # ==================================================================

    diag = AtA.diagonal()

    if np.any(~np.isfinite(diag)):
        raise FloatingPointError(
            "non-finite AtA diagonal"
        )

    if np.any(diag <= 0.0):
        bad = np.nonzero(diag <= 0.0)[0][:20]
        raise AssertionError(
            "normal matrix has non-positive diagonal entries at "
            f"{bad.tolist()}"
        )

    M = sparse.diags(
        1.0 / diag
    )

    x0 = target[unk].astype(np.float64)

    x, status = cg(
        AtA,
        Atb,
        x0=x0,
        M=M,
        rtol=float(surf.get("cg_tol", 1e-6)),
        maxiter=int(surf.get("cg_maxiter", 800)),
    )

    if not np.all(np.isfinite(x)):
        raise FloatingPointError(
            "CG returned non-finite terrain values"
        )

    # ==================================================================
    # RECONSTRUCT FIELD
    # ==================================================================

    out = target.astype(np.float32)

    out_flat = out.ravel()
    unk_flat = np.nonzero(unk.ravel())[0]

    out_flat[unk_flat] = x.astype(np.float32)

    out = out_flat.reshape(out.shape)

    # Exact hard projection AFTER solve.
    out[hard] = hard_vals[hard]

    # Everything outside the solve corridor is exactly the scaled target.
    out[~smask] = ctx["target"][~smask]

    # ==================================================================
    # RESIDUAL DIAGNOSTICS
    # ==================================================================

    def weighted_stats(sl: slice) -> tuple[float, float]:
        if sl.stop <= sl.start:
            return 0.0, 0.0

        r = (
            A[sl, :] @ x
            - b_vec[sl]
        )

        return (
            float(np.sqrt(np.mean(r * r))),
            float(np.max(np.abs(r))),
        )

    data_wrms, data_wmax = weighted_stats(
        data_slice
    )
    lap_wrms, lap_wmax = weighted_stats(
        lap_slice
    )
    slope_wrms, slope_wmax = weighted_stats(
        slope_slice
    )

    # Report physically interpretable GU residuals separately from
    # WEIGHTED least-squares residuals.
    #
    # Data error in GU.
    data_error_gu = (
        x
        - target[unk].astype(np.float64)
    )

    data_rms_gu = float(
        np.sqrt(np.mean(data_error_gu ** 2))
    )
    data_max_gu = float(
        np.max(np.abs(data_error_gu))
    )

    # Laplacian residual is GU when divided by its scalar weight.
    lap_rms_gu = (
        lap_wrms / w_smooth
        if w_smooth > 0.0
        else 0.0
    )
    lap_max_gu = (
        lap_wmax / w_smooth
        if w_smooth > 0.0
        else 0.0
    )

    # THIS is the slope residual the quality gate should use.
    #
    # It is direct height error in GU, not 25x weighted equation error.
    if n_slope_eq:
        slope_error_gu = (
            x[slope_cols]
            - slope_desired
        )

        slope_rms_gu = float(
            np.sqrt(
                np.mean(
                    slope_error_gu ** 2
                )
            )
        )

        slope_max_gu = float(
            np.max(
                np.abs(
                    slope_error_gu
                )
            )
        )
    else:
        slope_rms_gu = 0.0
        slope_max_gu = 0.0

    counts = {
        "data": int(n_data_eq),
        "laplacian": int(n_lap_eq),
        "slope": int(n_slope_eq),
        "boundary_eliminated": int(hard.sum()),
    }

    residuals = {
        # Physical / interpretable values
        "data_rms": round(data_rms_gu, 4),
        "data_max": round(data_max_gu, 4),

        "laplacian_rms": round(lap_rms_gu, 4),
        "laplacian_max": round(lap_max_gu, 4),

        "slope_rms": round(slope_rms_gu, 4),
        "slope_max": round(slope_max_gu, 4),

        # Weighted objective-space values
        "data_weighted_rms": round(data_wrms, 4),
        "data_weighted_max": round(data_wmax, 4),

        "laplacian_weighted_rms": round(lap_wrms, 4),
        "laplacian_weighted_max": round(lap_wmax, 4),

        "slope_weighted_rms": round(slope_wrms, 4),
        "slope_weighted_max": round(slope_wmax, 4),

        "boundary_max": 0.0,
    }

    report = {
        "unknowns": n,
        "cg_status": int(status),
        "equation_counts": counts,
        "residuals": residuals,
        "slope_rows": int(n_slope_eq),

        "assembly": {
            "rows": int(n_rows),
            "nnz": int(A.nnz),
            "empty_equation_rows": int(empty_rows),
        },
    }

    return out, report
