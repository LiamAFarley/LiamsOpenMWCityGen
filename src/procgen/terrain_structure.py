"""Sparse semantic continuation and screened-Poisson structural solve.

Pipeline position
    Stages 5-6. Feature rasters from :mod:`terrain_features` are reduced to a
    small set of ridge, valley, plateau, and scarp guide curves crossing the
    owner seam. Guides add low-frequency structure; they never copy owner
    pixels or create dense seam profiles.

Solver
    The correction ``C = H_structural - H0`` satisfies
    ``(L + W) C = W * (Hguide - H0)`` with the accepted exact seam and outer
    boundary Dirichlet constraints. This is second-order and AMG-friendly.
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
    "guide_max_cells": 6.0,
    "massif_guide_max_cells": 8.0,
    "ridge_weight": 0.7,
    "valley_weight": 1.0,
    "plateau_top_weight": 0.3,
    "scarp_weight": 0.8,
    "guide_ribbon_sigma_verts": 8.0,
    "guide_seed_stride_verts": 64,
    "guide_turn_deg_per_8_verts": 12.0,
    "guide_score_threshold": 0.35,
    "guide_decay_fraction": 0.55,
    "linear_solver": "amg_rs_cg",
    "cg_tol": 1e-6,
    "cg_maxiter": 200,
    "amg_max_coarse": 500,
}


def _smootherstep(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _feature_kind(features: dict, oy: int, ox: int, config: dict) -> tuple[str, float] | None:
    threshold = float(config.get("guide_score_threshold", 0.35))
    candidates = [
        ("ridge", float(features["ridge_score"][oy, ox]),
         float(config.get("ridge_weight", 0.7))),
        ("valley", float(features["valley_score"][oy, ox]),
         float(config.get("valley_weight", 1.0))),
    ]
    if features["plateau_top_mask"][oy, ox]:
        candidates.append(("plateau", 1.0,
                          float(config.get("plateau_top_weight", 0.3))))
    if features["scarp_mask"][oy, ox]:
        candidates.append(("scarp", 1.0,
                          float(config.get("scarp_weight", 0.8))))
    kind, score, weight = max(candidates, key=lambda item: item[1])
    if score < threshold and kind not in {"plateau", "scarp"}:
        return None
    return kind, weight


def _add_ribbon(
    source_sum: np.ndarray,
    source_weight: np.ndarray,
    active: np.ndarray,
    owner_mask: np.ndarray,
    points: list[tuple[float, float, float]],
    amplitude: float,
    base_weight: float,
    sigma: float,
    decay_fraction: float,
) -> int:
    """Add sparse guide samples; the caller performs one global Gaussian pass."""
    H, W = source_sum.shape
    added = 0
    for distance, fy, fx in points:
        cy, cx = int(round(fy)), int(round(fx))
        if not (0 <= cy < H and 0 <= cx < W):
            continue
        longitudinal = np.exp(-distance / max(decay_fraction, 1e-3))
        if not active[cy, cx] or owner_mask[cy, cx]:
            continue
        weight = float(base_weight) * float(longitudinal)
        source_weight[cy, cx] += weight
        source_sum[cy, cx] += weight * float(amplitude) * float(longitudinal)
        added += 1
    return added


def build_structural_guides(
    h0: np.ndarray,
    ctx: dict,
    features: dict,
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build sparse semantic guide-value and guide-weight rasters."""
    c = dict(DEFAULTS)
    if config:
        c.update(config)
    active = np.asarray(ctx["smask"], dtype=bool)
    owner_mask = np.asarray(features["owner_mask"], dtype=bool)
    source_sum = np.zeros(h0.shape, dtype=np.float32)
    source_weight = np.zeros(h0.shape, dtype=np.float32)
    H24 = features["H24"]
    H64 = features["H64"]
    H8 = features["H8"]
    anomaly = np.nan_to_num(H24 - H64, nan=0.0)
    local_relief = np.maximum(
        np.nan_to_num(H8, nan=0.0), np.nan_to_num(H24, nan=0.0)
    ) - np.minimum(
        np.nan_to_num(H8, nan=0.0), np.nan_to_num(H24, nan=0.0)
    )
    angle = features["orientation_angle"]
    stride = max(1, int(c["guide_seed_stride_verts"]))
    sigma = float(c["guide_ribbon_sigma_verts"])
    decay_fraction = float(c["guide_decay_fraction"])
    feature_seed_counts = {"ridge": 0, "valley": 0, "plateau": 0, "scarp": 0}
    guide_support_vertices = 0

    for edge in ctx["edge_list"]:
        normal = tuple(int(round(v)) for v in edge["normal"])
        last_seed_index = -stride
        for index, flat in enumerate(edge["verts"]):
            if index - last_seed_index < stride:
                continue
            sy, sx = divmod(int(flat), h0.shape[1])
            oy, ox = sy - normal[0], sx - normal[1]
            if not (0 <= oy < h0.shape[0] and 0 <= ox < h0.shape[1]):
                continue
            if not owner_mask[oy, ox] or not np.isfinite(H24[oy, ox]):
                continue
            selected = _feature_kind(features, oy, ox, c)
            if selected is None:
                continue
            kind, base_weight = selected
            last_seed_index = index
            feature_seed_counts[kind] += 1
            tangent = np.array([np.sin(angle[oy, ox]), np.cos(angle[oy, ox])])
            tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
            nvec = np.array(normal, dtype=np.float32)
            max_cells = float(c["guide_max_cells"])
            if kind == "ridge" and abs(float(anomaly[oy, ox])) > 0.5 * max(
                abs(float(local_relief[oy, ox])), 256.0
            ):
                max_cells = float(c["massif_guide_max_cells"])
            length = max_cells * 64.0
            amp = float(anomaly[oy, ox])
            if kind == "ridge":
                amp = abs(amp)
            elif kind == "valley":
                amp = -abs(amp)
            elif kind == "plateau":
                amp *= 0.35
            elif kind == "scarp":
                amp *= 0.8
            relief_limit = max(256.0, abs(float(local_relief[oy, ox])) * 0.75)
            amp = float(np.clip(amp, -relief_limit, relief_limit))
            points = []
            for distance in np.arange(0.0, length + 1.0, 8.0):
                pos = np.array([sy, sx], dtype=np.float32)
                pos += nvec * distance
                py, px = float(pos[0]), float(pos[1])
                iy, ix = int(round(py)), int(round(px))
                if not (0 <= iy < h0.shape[0] and 0 <= ix < h0.shape[1]):
                    break
                if not active[iy, ix] and distance > 0.0:
                    break
                points.append((distance / max(length, 1.0), py, px))
            guide_support_vertices += _add_ribbon(
                source_sum, source_weight, active, owner_mask, points[1:2],
                amp, base_weight, sigma, decay_fraction,
            )

    # Convolve all sparse semantic sources once. This keeps broad ribbons
    # bounded by the window size instead of materializing one patch per sample.
    kernel_area = 2.0 * np.pi * sigma * sigma
    guide_sum = ndimage.gaussian_filter(
        source_sum, sigma, mode="nearest"
    ) * kernel_area
    guide_weight = ndimage.gaussian_filter(
        source_weight, sigma, mode="nearest"
    ) * kernel_area
    guide_value = h0.astype(np.float32, copy=True)
    guided = guide_weight > 1e-6
    guide_value[guided] += guide_sum[guided] / guide_weight[guided]
    guide_weight[~active | owner_mask] = 0.0
    guided &= guide_weight > 1e-6
    guide_value[~np.isfinite(guide_value)] = h0[~np.isfinite(guide_value)]
    return guide_value, guide_weight.astype(np.float32), {
        "feature_seed_counts": feature_seed_counts,
        "guide_support_vertices": int(guide_support_vertices),
        "guide_source_points": int(guide_support_vertices),
        "guide_vertices": int(guided.sum()),
        "guide_weight_max": float(guide_weight.max(initial=0.0)),
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
