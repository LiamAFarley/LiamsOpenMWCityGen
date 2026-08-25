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

def normal_profiles(field: np.ndarray, own_view: np.ndarray,
                    seam_v: np.ndarray, nx: np.ndarray, ny: np.ndarray,
                    count: int = 64, in_verts: int = 16,
                    out_verts: int = 8) -> dict:
    """Height profiles across the seam at evenly sampled seam vertices.

    For each sampled vertex: owner heights for out_verts going outward and
    generated heights for in_verts going inward, plus the first-edge drop
    |gen(1) - gen(0)| — the numeric signature of a one-vertex cliff.
    """
    ys, xs = np.nonzero(seam_v)
    if ys.size == 0:
        return {"profiles": [], "max_first_edge_drop": 0.0}
    order = np.argsort(ys * field.shape[1] + xs)
    pick = np.linspace(0, ys.size - 1, min(count, ys.size)).astype(int)
    profiles = []
    max_drop = 0.0
    for p in pick:
        y, x = int(ys[order[p]]), int(xs[order[p]])
        nyv, nxv = float(ny[y, x]), float(nx[y, x])
        if nyv == 0.0 and nxv == 0.0:
            continue
        dy, dx = int(round(nyv)), int(round(nxv))
        owner = []
        for k in range(out_verts, 0, -1):
            r, c = y - dy * k, x - dx * k
            owner.append(float(own_view[r, c])
                         if (0 <= r < field.shape[0] and 0 <= c < field.shape[1]
                             and np.isfinite(own_view[r, c])) else np.nan)
        gen = []
        for k in range(0, in_verts + 1):
            r, c = y + dy * k, x + dx * k
            gen.append(float(field[r, c])
                       if (0 <= r < field.shape[0] and 0 <= c < field.shape[1]
                           and np.isfinite(field[r, c])) else np.nan)
        drop = (abs(gen[1] - gen[0])
                if len(gen) > 1 and np.isfinite(gen[0]) and np.isfinite(gen[1])
                else 0.0)
        max_drop = max(max_drop, drop)
        profiles.append({"vertex": [y, x], "owner_out": owner,
                         "gen_in": gen, "first_edge_drop": round(drop, 1)})
    return {"profiles": profiles,
            "max_first_edge_drop_gu": round(max_drop, 1)}
