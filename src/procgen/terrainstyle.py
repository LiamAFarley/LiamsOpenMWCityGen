"""Measured terrain style: band spectra, orientations, and panel metrics.

Purpose
    Replace guessed noise amplitudes with MEASURED owner-terrain statistics.
    Provides band decomposition (blur stack), style measurement (per-band
    sigma, gradient-orientation histogram, slope histogram), synthesis of
    band-matched detail, and the metric set used to judge panel variants.

Pipeline position
    Shared core for the Stage D panel (tools/terrain/panel_region.py) and
    later production blending. Consumes corpus-style numpy fields only.

Invariants
    All statistics computed only over finite values inside the given mask;
    deterministic given inputs (no RNG here except via caller-provided
    generators); band sizes are config, never hardcoded in callers.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

DEFAULT_BANDS = (2, 4, 8, 16, 32, 64)


def band_stack(field: np.ndarray, sizes=DEFAULT_BANDS, fill: float | None = None):
    """Multi-scale detail decomposition via a box-blur stack.

    Returns (details, residual): details[k] = blur_{k-1}(h) - blur_k(h)
    (finest first), residual = blur_largest(h). NaNs are filled with the
    field mean (or ``fill``) for filtering and re-masked in every band.
    """
    h = np.asarray(field, dtype=np.float32)
    valid = np.isfinite(h)
    if fill is None:
        fill = float(np.nan_to_num(h[valid]).mean()) if valid.any() else 0.0
    x = np.where(valid, h, np.float32(fill))
    out = []
    prev = x
    for k in sizes:
        cur = ndimage.uniform_filter(x, size=int(k) * 2 + 1, mode="nearest")
        d = prev - cur
        d[~valid] = 0.0
        out.append(d)
        prev = cur
    res = prev
    res[~valid] = 0.0
    return out, res


def measure_style(field: np.ndarray, mask: np.ndarray, sizes=DEFAULT_BANDS,
                  ori_bins: int = 18) -> dict:
    """Style statistics over the finite part of ``mask``."""
    details, _ = band_stack(field, sizes)
    band_sigma = []
    for d in details:
        v = d[mask & (d != 0.0)]
        band_sigma.append(round(float(np.std(v)), 2) if v.size else 0.0)

    gy, gx = np.gradient(np.where(np.isfinite(field), field, 0.0), 128.0)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)          # orientations mod pi
    sel = mask & np.isfinite(field) & (mag > 1e-3)
    hist, _ = np.histogram(ang[sel], bins=ori_bins, range=(0.0, np.pi),
                           weights=mag[sel])
    hist = hist / max(float(hist.sum()), 1e-9)

    slope = mag * 128.0
    ssel = mask & np.isfinite(field)
    shist, edges = np.histogram(slope[ssel], bins=24, range=(0, 2000))
    return {
        "band_sizes": [int(s) for s in sizes],
        "band_sigma": band_sigma,
        "orientation_hist": [round(float(v), 5) for v in hist],
        "slope_hist": [int(v) for v in shist],
        "slope_edges": [round(float(v), 1) for v in edges],
        "slope_mean_gu_per_vert": round(float(slope[ssel].mean()), 1),
    }


def band_matched_detail(shape: tuple[int, int], ref_sigma: list[float],
                        sizes=DEFAULT_BANDS, rng: np.random.Generator | None = None,
                        warp_strength: float = 24.0) -> np.ndarray:
    """Synthesize detail whose per-band sigma matches ``ref_sigma``.

    White noise is domain-warped (smooth random coordinate bend) to break
    grid alignment, then bandpassed with the same blur stack and rescaled
    per band to the reference sigmas.
    """
    rng = rng or np.random.default_rng(0)
    white = rng.standard_normal(shape).astype(np.float32)
    if warp_strength > 0:
        wy = ndimage.gaussian_filter(
            rng.standard_normal(shape).astype(np.float32), warp_strength / 3)
        wx = ndimage.gaussian_filter(
            rng.standard_normal(shape).astype(np.float32), warp_strength / 3)
        wy /= max(float(np.std(wy)), 1e-6)
        wx /= max(float(np.std(wx)), 1e-6)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        coords = np.stack([yy + wy * warp_strength, xx + wx * warp_strength])
        white = ndimage.map_coordinates(white, coords, order=1, mode="reflect")
    white -= float(white.mean())

    details, _ = band_stack(white, sizes)
    out = np.zeros(shape, dtype=np.float32)
    for d, sigma in zip(details, ref_sigma):
        s = float(np.std(d))
        if s > 1e-6 and sigma > 0:
            out += d * (sigma / s)
    return out


def orientation_corr(h1: list[float], h2: list[float]) -> float:
    a = np.asarray(h1, float)
    b = np.asarray(h2, float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom > 0 else 0.0


def sigma_distance_profile(field: np.ndarray, dist: np.ndarray, mask: np.ndarray,
                           band: int = 16, bin_verts: int = 32) -> list[float]:
    """Band sigma as a function of distance-from-seam (binned)."""
    details, _ = band_stack(field, sizes=(band,))
    d = details[0]
    dmax = float(dist[mask].max()) if mask.any() else 0.0
    prof = []
    lo = 0.0
    while lo < dmax:
        sel = mask & (dist >= lo) & (dist < lo + bin_verts)
        v = d[sel & (d != 0.0)]
        prof.append(round(float(np.std(v)), 2) if v.size else 0.0)
        lo += bin_verts
    return prof


def profile_cliff(prof: list[float]) -> float:
    """Largest single-bin relative drop in the sigma-vs-distance profile —
    the numeric detector for 'a new seam on the other side'."""
    p = np.asarray(prof, float)
    if p.size < 3:
        return 0.0
    interior = p[1:-1]
    drops = (interior - p[2:]) / np.maximum(interior, 1e-6)
    return round(float(np.max(drops)), 3)
