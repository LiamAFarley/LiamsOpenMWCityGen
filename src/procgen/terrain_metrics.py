"""Seam-quality metrics for the v3 blend (plan section 15).

Purpose
    Quantify seam quality beyond exact heights: C0 height error, C1
    normal-slope mismatch (the visible-crease predictor), and a curvature
    jump proxy. All statistics computed only over seam verts with valid
    data on both sides.

Pipeline position
    Used by tools/terrain/solve_region_v3.py and later verification stages.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def seam_c0(field: np.ndarray, own_view: np.ndarray, seam_v: np.ndarray) -> float:
    ok = seam_v & np.isfinite(own_view) & np.isfinite(field)
    if not ok.any():
        return -1.0
    return float(np.abs(field[ok] - own_view[ok]).max())


def band_edge_continuity(field: np.ndarray, target: np.ndarray,
                         smask: np.ndarray, seam_v: np.ndarray) -> float:
    """Maximum solved/target difference on the non-owner outer band edge."""
    edge = smask & ~ndimage.binary_erosion(smask) & ~seam_v
    ok = edge & np.isfinite(field) & np.isfinite(target)
    return float(np.abs(field[ok] - target[ok]).max()) if ok.any() else 0.0


def _inward_slope(field: np.ndarray, seam_v: np.ndarray, nx: np.ndarray,
                  ny: np.ndarray, sign: float) -> tuple[np.ndarray, np.ndarray]:
    H, W = field.shape
    ys, xs = np.nonzero(seam_v)
    uy = ys + sign * np.rint(ny[ys, xs]).astype(np.int64)
    ux = xs + sign * np.rint(nx[ys, xs]).astype(np.int64)
    ok = (uy >= 0) & (uy < H) & (ux >= 0) & (ux < W)
    slope = np.full(ys.shape, np.nan, np.float32)
    fy = uy[ok]
    fx = ux[ok]
    slope[ok] = (field[fy, fx] - field[ys[ok], xs[ok]]) / 64.0
    return slope, ok


def seam_c1_normals(field: np.ndarray, own_view: np.ndarray,
                    seam_v: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> dict:
    """|inward slope (ours) - inward slope (owner)| distribution.

    Our side samples one vertex inward (sign +1); the owner side samples
    one vertex outward (sign -1) of the same seam vertex."""
    ours, ok_a = _inward_slope(field, seam_v, nx, ny, +1)
    theirs, ok_b = _inward_slope(own_view, seam_v, nx, ny, -1)
    ok = ok_a & ok_b & np.isfinite(ours) & np.isfinite(theirs)
    if not ok.any():
        return {"count": 0}
    d = np.abs(ours[ok] - theirs[ok])
    return {"count": int(ok.sum()),
            "median": round(float(np.median(d)), 3),
            "p90": round(float(np.percentile(d, 90)), 3),
            "p99": round(float(np.percentile(d, 99)), 3),
            "max": round(float(d.max()), 3)}


def curvature_jump(field: np.ndarray, own_view: np.ndarray,
                   seam_v: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> dict:
    lap = lambda f: ndimage.laplace(np.where(np.isfinite(f), f, 0.0))
    ours, ok_a = _inward_slope(lap(field), seam_v, nx, ny, +1)
    theirs, ok_b = _inward_slope(lap(own_view), seam_v, nx, ny, -1)
    ok = ok_a & ok_b & np.isfinite(ours) & np.isfinite(theirs)
    if not ok.any():
        return {"count": 0}
    d = np.abs(ours[ok] - theirs[ok])
    return {"count": int(ok.sum()),
            "p90": round(float(np.percentile(d, 90)), 2),
            "max": round(float(d.max()), 2)}
