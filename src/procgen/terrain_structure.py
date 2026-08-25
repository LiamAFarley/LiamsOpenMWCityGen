"""Multiscale owner-field continuation and screened-Poisson structural solve.

Pipeline position
    Stages 5-6. Owner and Stage-3 terrain pyramids are converted to complete
    macro and meso residual fields. Each residual is continued across the
    generated corridor as one harmonic field; no sparse feature lines or raw
    owner-height profiles are injected.

Solver
    The generic correction ``C = H_structural - H0`` satisfies
    ``(L + W) C = W * (Hguide - H0)`` with seam and outer-boundary Dirichlet
    constraints. Run A sets ``W = 0`` for a pure second-order harmonic band
    solve, which remains AMG-friendly.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

try:
    import pyamg
except ImportError:  # pragma: no cover
    pyamg = None


DEFAULTS = {
    "macro_width_cells": 8.0,
    "meso_width_cells": 4.0,
    "fine_keep_at_seam": 0.2,
    "fine_restore_distance_cells": 6.0,
    "linear_solver": "amg_rs_cg",
    "cg_tol": 1e-6,
    "cg_maxiter": 200,
    "amg_max_coarse": 500,
}


def _smootherstep(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _normalized_band(field: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    values = np.where(valid, field, 0.0).astype(np.float32)
    weights = ndimage.gaussian_filter(
        valid.astype(np.float32), float(sigma), mode="nearest"
    )
    smoothed = ndimage.gaussian_filter(values, float(sigma), mode="nearest")
    return np.divide(
        smoothed,
        weights,
        out=np.zeros_like(smoothed, dtype=np.float32),
        where=weights > 1e-6,
    )


def _complete_band_boundary(active: np.ndarray, seam: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(
        active, structure=ndimage.generate_binary_structure(2, 1), border_value=0
    )
    return active & ~seam & ~eroded


def _seam_band_values(
    owner_band: np.ndarray,
    target_band: np.ndarray,
    ctx: dict,
    seam: np.ndarray,
) -> tuple[np.ndarray, dict]:
    values = target_band.astype(np.float32, copy=True)
    assigned = np.zeros(seam.shape, dtype=bool)
    conflicts = 0
    max_spread = 0.0
    for edge in ctx["edge_list"]:
        normal = tuple(int(round(v)) for v in edge["normal"])
        for flat in edge["verts"]:
            sy, sx = divmod(int(flat), target_band.shape[1])
            oy, ox = sy - normal[0], sx - normal[1]
            if not (seam[sy, sx] and 0 <= oy < owner_band.shape[0] and 0 <= ox < owner_band.shape[1]):
                continue
            candidate = float(owner_band[oy, ox])
            if not np.isfinite(candidate):
                continue
            if assigned[sy, sx]:
                spread = abs(float(values[sy, sx]) - candidate)
                max_spread = max(max_spread, spread)
                if spread > 1e-3:
                    conflicts += 1
                continue
            values[sy, sx] = candidate
            assigned[sy, sx] = True
    return values, {"seam_claim_conflicts": conflicts, "seam_claim_spread_max": max_spread}


def _first_inland_band_anchors(
    owner_band: np.ndarray,
    target_band: np.ndarray,
    ctx: dict,
    active: np.ndarray,
    seam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    incidence: dict[int, set[tuple[int, int]]] = {}
    for edge in ctx["edge_list"]:
        normal = tuple(int(round(v)) for v in edge["normal"])
        for flat in edge["verts"]:
            incidence.setdefault(int(flat), set()).add(normal)

    anchor_mask = np.zeros(active.shape, dtype=bool)
    anchor_values = target_band.astype(np.float32, copy=True)
    skipped_corner = 0
    skipped_invalid = 0
    for edge in ctx["edge_list"]:
        normal = tuple(int(round(v)) for v in edge["normal"])
        for flat in edge["verts"]:
            flat = int(flat)
            if len(incidence.get(flat, ())) != 1:
                skipped_corner += 1
                continue
            sy, sx = divmod(flat, target_band.shape[1])
            oy, ox = sy - normal[0], sx - normal[1]
            by, bx = oy - normal[0], ox - normal[1]
            fy, fx = sy + normal[0], sx + normal[1]
            if not (
                0 <= oy < owner_band.shape[0]
                and 0 <= ox < owner_band.shape[1]
                and 0 <= by < owner_band.shape[0]
                and 0 <= bx < owner_band.shape[1]
            ):
                skipped_invalid += 1
                continue
            if not (0 <= fy < target_band.shape[0] and 0 <= fx < target_band.shape[1]):
                skipped_invalid += 1
                continue
            if not active[fy, fx] or seam[fy, fx]:
                skipped_invalid += 1
                continue
            b0 = float(owner_band[oy, ox])
            bout = float(owner_band[by, bx])
            if not np.isfinite(b0) or not np.isfinite(bout):
                skipped_invalid += 1
                continue
            candidate = b0 + (b0 - bout)
            if anchor_mask[fy, fx]:
                if abs(float(anchor_values[fy, fx]) - candidate) > 1e-3:
                    skipped_corner += 1
                continue
            anchor_mask[fy, fx] = True
            anchor_values[fy, fx] = candidate
    return anchor_mask, anchor_values, {
        "first_inland_anchor_count": int(anchor_mask.sum()),
        "first_inland_anchor_skipped_corner": skipped_corner,
        "first_inland_anchor_skipped_invalid": skipped_invalid,
    }


def _solve_band(
    name: str,
    target_band: np.ndarray,
    owner_band: np.ndarray,
    ctx: dict,
    generated: np.ndarray,
    width_cells: float,
    config: dict,
) -> tuple[np.ndarray, dict]:
    active = generated & (ctx["dist_seam"] <= float(width_cells) * 64.0)
    seam = np.asarray(ctx["seam_v"], dtype=bool) & active
    outer = _complete_band_boundary(active, seam)
    fixed = seam | outer
    seam_values, seam_report = _seam_band_values(owner_band, target_band, ctx, seam)
    anchor_mask, anchor_values, anchor_report = _first_inland_band_anchors(
        owner_band, target_band, ctx, active, seam
    )
    fixed |= anchor_mask
    fixed_values = target_band.astype(np.float32, copy=True)
    fixed_values[seam] = seam_values[seam]
    fixed_values[anchor_mask] = anchor_values[anchor_mask]
    solved, solve_report = solve_screened_structure(
        target_band,
        active,
        fixed,
        fixed_values,
        target_band,
        np.zeros(target_band.shape, dtype=np.float32),
        config,
    )
    report = {
        "name": name,
        "width_cells": float(width_cells),
        "active_vertices": int(active.sum()),
        "outer_boundary_vertices": int(outer.sum()),
        "fixed_vertices": int(fixed.sum()),
        "correction_min": float((solved - target_band)[active].min(initial=0.0)),
        "correction_max": float((solved - target_band)[active].max(initial=0.0)),
        **seam_report,
        **anchor_report,
        "solve": solve_report,
    }
    return solved.astype(np.float32), report


def build_multiscale_structural_fields(
    h0: np.ndarray,
    ctx: dict,
    features: dict,
    config: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Continue complete owner macro/meso bands into generated terrain."""
    c = dict(DEFAULTS)
    if config:
        c.update(config)
    active = np.asarray(ctx["smask"], dtype=bool)
    owner_mask = np.asarray(features["owner_mask"], dtype=bool)
    generated = active & ~owner_mask & np.isfinite(h0)
    generated_valid = generated & np.isfinite(h0)

    target8 = _normalized_band(h0, generated_valid, 8.0)
    target24 = _normalized_band(h0, generated_valid, 24.0)
    target64 = _normalized_band(h0, generated_valid, 64.0)
    target_macro = target24 - target64
    target_meso = target8 - target24
    owner_macro = features["H24"] - features["H64"]
    owner_meso = features["H8"] - features["H24"]

    fine_low = _normalized_band(h0, generated_valid, 4.0)
    distance = np.nan_to_num(ctx["dist_seam"], nan=0.0)
    restore = max(float(c["fine_restore_distance_cells"]) * 64.0, 1.0)
    keep = float(c["fine_keep_at_seam"]) + (
        1.0 - float(c["fine_keep_at_seam"])
    ) * _smootherstep(distance / restore)
    cleaned = h0.astype(np.float32, copy=True)
    cleaned[generated] = fine_low[generated] + keep[generated] * (
        h0[generated] - fine_low[generated]
    )

    macro_band, macro_report = _solve_band(
        "macro", target_macro, owner_macro, ctx, generated,
        float(c["macro_width_cells"]), c,
    )
    macro = cleaned.astype(np.float32, copy=True)
    macro[generated] += (macro_band - target_macro)[generated]
    meso_band, meso_report = _solve_band(
        "meso", target_meso, owner_meso, ctx, generated,
        float(c["meso_width_cells"]), c,
    )
    macro_meso = macro.astype(np.float32, copy=True)
    macro_meso[generated] += (meso_band - target_meso)[generated]
    for field in (cleaned, macro, macro_meso):
        field[owner_mask] = h0[owner_mask]
    return {
        "stage3": h0.astype(np.float32, copy=True),
        "cleaned": cleaned,
        "macro": macro,
        "macro_meso": macro_meso,
    }, {
        "owner_vertices": int(owner_mask.sum()),
        "generated_vertices": int(generated.sum()),
        "fine_keep_at_seam": float(c["fine_keep_at_seam"]),
        "fine_restore_distance_cells": float(c["fine_restore_distance_cells"]),
        "macro": macro_report,
        "meso": meso_report,
    }


def solve_screened_structure(
    h0: np.ndarray,
    active: np.ndarray,
    fixed: np.ndarray,
    fixed_values: np.ndarray,
    guide_value: np.ndarray,
    guide_weight: np.ndarray,
    config: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve the sparse screened-Poisson structural correction."""
    c = dict(DEFAULTS)
    if config:
        c.update(config)
    h0 = np.nan_to_num(h0).astype(np.float64)
    active = np.asarray(active, dtype=bool)
    fixed = np.asarray(fixed, dtype=bool)
    weights = np.clip(np.nan_to_num(guide_weight), 0.0, None).astype(np.float64)
    if np.any(fixed & ~active):
        raise ValueError("structural fixed vertices must lie inside active domain")
    correction_fixed = np.zeros(h0.shape, dtype=np.float64)
    correction_fixed[fixed] = fixed_values[fixed] - h0[fixed]
    unknown = active & ~fixed
    n = int(unknown.sum())
    if n == 0:
        return h0.astype(np.float32), {"unknowns": 0, "guide_rows": 0}
    idx = np.full(active.shape, -1, dtype=np.int64)
    idx[unknown] = np.arange(n, dtype=np.int64)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n, dtype=np.float64)
    degree = np.zeros(n, dtype=np.float64)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        center = unknown.copy()
        if dy == 1:
            center[-1, :] = False
        elif dy == -1:
            center[0, :] = False
        if dx == 1:
            center[:, -1] = False
        elif dx == -1:
            center[:, 0] = False
        neighbor_active = np.roll(active, (-dy, -dx), axis=(0, 1))
        neighbor_fixed = np.roll(fixed, (-dy, -dx), axis=(0, 1))
        neighbor_idx = np.roll(idx, (-dy, -dx), axis=(0, 1))
        neighbor_corr = np.roll(correction_fixed, (-dy, -dx), axis=(0, 1))
        valid = center & neighbor_active
        eq = idx[valid]
        degree[eq] += 1.0
        nb = neighbor_idx[valid]
        is_unknown = nb >= 0
        rows.append(eq[is_unknown])
        cols.append(nb[is_unknown])
        vals.append(np.full(int(is_unknown.sum()), -1.0))
        is_fixed = neighbor_fixed[valid]
        if np.any(is_fixed):
            np.add.at(rhs, eq[is_fixed], neighbor_corr[valid][is_fixed])
    if np.any(degree <= 0.0):
        raise ValueError("screened structural system has isolated unknowns")
    guide_delta = guide_value.astype(np.float64) - h0
    rhs += weights[unknown] * guide_delta[unknown]
    rows.append(np.arange(n, dtype=np.int64))
    cols.append(np.arange(n, dtype=np.int64))
    vals.append(degree + weights[unknown])
    A = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    setup_t0 = time.perf_counter()
    solver = str(c.get("linear_solver", "amg_rs_cg"))
    ml = None
    if solver == "amg_rs_cg":
        if pyamg is None:
            raise RuntimeError("structural AMG requires pyamg")
        ml = pyamg.ruge_stuben_solver(A, max_coarse=int(c["amg_max_coarse"]))
        preconditioner = ml.aspreconditioner(cycle="V")
    elif solver == "jacobi_cg":
        preconditioner = sparse.diags(1.0 / A.diagonal())
    else:
        raise ValueError(f"unsupported structural solver {solver!r}")
    setup_s = time.perf_counter() - setup_t0
    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1

    solve_t0 = time.perf_counter()
    correction, status = cg(
        A, rhs, M=preconditioner, rtol=float(c["cg_tol"]), atol=0.0,
        maxiter=int(c["cg_maxiter"]), callback=callback,
    )
    solve_s = time.perf_counter() - solve_t0
    if status != 0:
        raise RuntimeError(f"structural CG failed with status {status}")
    out = h0.copy()
    full_correction = correction_fixed.copy()
    full_correction[unknown] = correction
    out[active] += full_correction[active]
    out[fixed] = fixed_values[fixed]
    residual = A @ correction - rhs
    report = {
        "unknowns": n,
        "guide_rows": int(np.count_nonzero(weights[unknown] > 0.0)),
        "guide_weight_max": float(weights.max(initial=0.0)),
        "linear_solver": solver,
        "cg_iterations": int(iterations),
        "solver_setup_s": round(setup_s, 4),
        "solver_solve_s": round(solve_s, 4),
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "correction_min": float(full_correction[active].min()),
        "correction_max": float(full_correction[active].max()),
    }
    if ml is not None:
        report["amg_levels"] = int(len(ml.levels))
    return out.astype(np.float32), report
