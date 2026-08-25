"""Multiscale semantic feature analysis for authoritative owner terrain.

Pipeline position
    Stage 4 of the v3 structural-erosion path. The caller supplies a local
    owner field and mask; this module never reads the full world or edits
    authoritative terrain.

Outputs
    Gaussian owner pyramid, structure-tensor orientation/coherence, sparse
    ridge/valley/plateau/scarp/coast classifications, and a smooth erosion
    factor field. Feature rasters are float32/bool and are intended as guides,
    not texture copies.

Invariants
    Pixels outside the owner mask carry no feature authority. Underwater
    vertices have zero terrestrial erosion factor. Thresholds are relative to
    the supplied owner window and are config-driven.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


DEFAULTS = {
    "gaussian_scales_verts": [8.0, 24.0, 64.0],
    "tensor_sigma_verts": 10.0,
    "ridge_percentile": 88.0,
    "valley_percentile": 88.0,
    "plateau_elevation_percentile": 60.0,
    "plateau_slope_percentile": 35.0,
    "coastal_band_verts": 96.0,
    "feature_min_component_vertices": 32,
    "hydrology_valley_boost": 0.35,
}


def _nearest_finite(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        raise ValueError("owner feature analysis has no finite owner terrain")
    indices = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    out = np.asarray(field, dtype=np.float32).copy()
    out[~valid] = field[tuple(indices)][~valid]
    return out


def _normalized_gaussian(
    field: np.ndarray, valid: np.ndarray, sigma: float
) -> np.ndarray:
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


def _masked_derivative(
    field: np.ndarray,
    valid: np.ndarray,
    spacing: float,
    axis: int,
) -> np.ndarray:
    """Differentiate without inventing values across the owner boundary."""
    if axis not in (0, 1):
        raise ValueError("masked derivative axis must be 0 or 1")
    field = np.asarray(field, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(field)
    minus = np.zeros(valid.shape, dtype=bool)
    plus = np.zeros(valid.shape, dtype=bool)
    minus_slice = [slice(None), slice(None)]
    plus_slice = [slice(None), slice(None)]
    center_minus = [slice(None), slice(None)]
    center_plus = [slice(None), slice(None)]
    minus_slice[axis] = slice(1, None)
    center_minus[axis] = slice(None, -1)
    plus_slice[axis] = slice(None, -1)
    center_plus[axis] = slice(1, None)
    minus[tuple(minus_slice)] = valid[tuple(center_minus)]
    plus[tuple(plus_slice)] = valid[tuple(center_plus)]

    backward = np.roll(field, 1, axis=axis)
    forward = np.roll(field, -1, axis=axis)
    out = np.full(field.shape, np.nan, dtype=np.float32)
    both = valid & minus & plus
    only_minus = valid & minus & ~plus
    only_plus = valid & ~minus & plus
    out[both] = (forward[both] - backward[both]) / (2.0 * spacing)
    out[only_minus] = (field[only_minus] - backward[only_minus]) / spacing
    out[only_plus] = (forward[only_plus] - field[only_plus]) / spacing
    return out


def _robust_unit(field: np.ndarray, valid: np.ndarray, percentile: float) -> np.ndarray:
    positive = field[valid & np.isfinite(field) & (field > 0.0)]
    if positive.size == 0:
        return np.zeros(field.shape, dtype=np.float32)
    scale = max(float(np.percentile(positive, percentile)), 1e-9)
    return np.clip(field / scale, 0.0, 2.0).astype(np.float32)


def _remove_small_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    if minimum <= 1:
        return mask
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(minimum)
    keep[0] = False
    return keep[labels]


def analyze_owner_features(
    owner_field: np.ndarray,
    owner_mask: np.ndarray,
    config: dict | None = None,
    *,
    owner_accumulation: np.ndarray | None = None,
) -> dict:
    """Analyze owner relief and return semantic feature rasters."""
    c = dict(DEFAULTS)
    if config:
        c.update(config)
    owner_mask = np.asarray(owner_mask, dtype=bool)
    valid = owner_mask & np.isfinite(owner_field)
    filled = _nearest_finite(owner_field, valid)
    scales = [float(v) for v in c["gaussian_scales_verts"]]
    if len(scales) != 3:
        raise ValueError("gaussian_scales_verts must contain [8, 24, 64]")
    H8 = _normalized_gaussian(filled, valid, scales[0])
    H24 = _normalized_gaussian(filled, valid, scales[1])
    H64 = _normalized_gaussian(filled, valid, scales[2])
    H8[~owner_mask] = np.nan
    H24[~owner_mask] = np.nan
    H64[~owner_mask] = np.nan

    spacing = 128.0
    derivative_valid = owner_mask & np.isfinite(H24)
    gy = _masked_derivative(H24, derivative_valid, spacing, axis=0)
    gx = _masked_derivative(H24, derivative_valid, spacing, axis=1)
    gradient_valid = np.isfinite(gx) & np.isfinite(gy)
    sigma_tensor = float(c["tensor_sigma_verts"])
    gx_finite = np.nan_to_num(gx, nan=0.0)
    gy_finite = np.nan_to_num(gy, nan=0.0)
    Jxx = _normalized_gaussian(gx_finite * gx_finite, gradient_valid, sigma_tensor)
    Jyy = _normalized_gaussian(gy_finite * gy_finite, gradient_valid, sigma_tensor)
    Jxy = _normalized_gaussian(gx_finite * gy_finite, gradient_valid, sigma_tensor)
    tensor_mean = 0.5 * (Jxx + Jyy)
    tensor_radius = np.sqrt(
        np.maximum(0.0, 0.25 * (Jxx - Jyy) ** 2 + Jxy * Jxy)
    )
    tensor_hi = tensor_mean + tensor_radius
    tensor_lo = np.maximum(tensor_mean - tensor_radius, 0.0)
    coherence = np.divide(
        tensor_hi - tensor_lo,
        tensor_hi + tensor_lo,
        out=np.zeros_like(tensor_hi, dtype=np.float32),
        where=(tensor_hi + tensor_lo) > 1e-12,
    )
    # Tensor major axis is the gradient normal; rotate it to the elongated
    # terrain direction so guide curves follow ridges and valley floors.
    orientation = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + np.pi / 2.0
    orientation[~owner_mask] = 0.0
    coherence[~owner_mask] = 0.0

    dxx = _masked_derivative(gx, np.isfinite(gx), spacing, axis=1)
    dyy = _masked_derivative(gy, np.isfinite(gy), spacing, axis=0)
    dxy = _masked_derivative(gx, np.isfinite(gx), spacing, axis=0)
    dxx = np.nan_to_num(dxx, nan=0.0)
    dyy = np.nan_to_num(dyy, nan=0.0)
    dxy = np.nan_to_num(dxy, nan=0.0)
    hmean = 0.5 * (dxx + dyy)
    hradius = np.sqrt(np.maximum(0.0, 0.25 * (dxx - dyy) ** 2 + dxy * dxy))
    eig_lo = hmean - hradius
    eig_hi = hmean + hradius

    local_ridge = np.maximum(H8 - H64, 0.0)
    local_valley = np.maximum(H64 - H8, 0.0)
    curvature_ridge = np.maximum(-eig_lo, 0.0)
    curvature_valley = np.maximum(eig_hi, 0.0)
    ridge_score = (
        _robust_unit(curvature_ridge, valid, 90.0)
        * _robust_unit(local_ridge, valid, 90.0)
        * (0.35 + 0.65 * coherence)
    )
    valley_score = (
        _robust_unit(curvature_valley, valid, 90.0)
        * _robust_unit(local_valley, valid, 90.0)
        * (0.35 + 0.65 * coherence)
    )
    if owner_accumulation is not None:
        valley_score *= 1.0 + float(c["hydrology_valley_boost"]) * _robust_unit(
            np.log1p(np.maximum(owner_accumulation, 0.0)), valid, 90.0
        )

    def threshold_mask(score: np.ndarray, percentile: float) -> np.ndarray:
        vals = score[valid & np.isfinite(score) & (score > 0.0)]
        threshold = float(np.percentile(vals, percentile)) if vals.size else np.inf
        mask = valid & (score >= threshold)
        return _remove_small_components(
            mask, int(c["feature_min_component_vertices"])
        )

    ridge_mask = threshold_mask(ridge_score, float(c["ridge_percentile"]))
    valley_mask = threshold_mask(valley_score, float(c["valley_percentile"]))

    land = valid & (filled > 0.0)
    underwater = valid & ~land
    shoreline = land & ndimage.binary_dilation(underwater, iterations=1)
    coastal_band = ndimage.binary_dilation(
        shoreline, iterations=max(1, int(float(c["coastal_band_verts"])))
    ) & land

    slope24 = np.hypot(gx_finite, gy_finite)
    slope24[~owner_mask] = 0.0
    elevation_values = H64[land]
    slope_values = slope24[land]
    elevation_cut = (
        float(np.percentile(elevation_values, c["plateau_elevation_percentile"]))
        if elevation_values.size else np.inf
    )
    slope_cut = (
        float(np.percentile(slope_values, c["plateau_slope_percentile"]))
        if slope_values.size else 0.0
    )
    plateau_top = land & (H64 >= elevation_cut) & (slope24 <= slope_cut)
    plateau_top = _remove_small_components(
        plateau_top, int(c["feature_min_component_vertices"])
    )
    scarp_score = _normalized_gaussian(
        np.maximum(np.abs(eig_lo) + np.abs(eig_hi), 0.0), valid,
        max(2.0, sigma_tensor / 2.0)
    )
    scarp = land & ~plateau_top & (scarp_score >= np.percentile(
        scarp_score[land], 80.0
    ) if land.any() else np.inf)
    scarp &= ndimage.binary_dilation(plateau_top, iterations=16)
    scarp = _remove_small_components(
        scarp, int(c["feature_min_component_vertices"] // 2)
    )

    factor = np.ones(filled.shape, dtype=np.float32)
    factor[land & (slope24 < slope_cut)] = 0.35
    factor[plateau_top] = 0.2
    factor[scarp] = 0.9
    factor[valley_mask] = np.maximum(factor[valley_mask], 1.2)
    factor[coastal_band] = np.minimum(factor[coastal_band], 0.45)
    factor[underwater] = 0.0
    factor = _normalized_gaussian(factor, valid, 10.0)
    factor[underwater] = 0.0
    factor[~owner_mask] = 0.0

    feature_counts = {
        "owner_vertices": int(valid.sum()),
        "ridge_vertices": int(ridge_mask.sum()),
        "valley_vertices": int(valley_mask.sum()),
        "plateau_vertices": int(plateau_top.sum()),
        "scarp_vertices": int(scarp.sum()),
        "underwater_vertices": int(underwater.sum()),
        "shoreline_vertices": int(shoreline.sum()),
        "coastal_vertices": int(coastal_band.sum()),
    }
    return {
        "owner_field": filled.astype(np.float32),
        "owner_mask": owner_mask,
        "H8": H8.astype(np.float32),
        "H24": H24.astype(np.float32),
        "H64": H64.astype(np.float32),
        "orientation_angle": orientation.astype(np.float32),
        "orientation_coherence": coherence.astype(np.float32),
        "slope24": slope24.astype(np.float32),
        "ridge_score": ridge_score.astype(np.float32),
        "valley_score": valley_score.astype(np.float32),
        "ridge_mask": ridge_mask,
        "valley_mask": valley_mask,
        "plateau_top_mask": plateau_top,
        "scarp_mask": scarp,
        "underwater_mask": underwater,
        "shoreline_mask": shoreline,
        "coastal_band": coastal_band,
        "erosion_factor": factor.astype(np.float32),
        "feature_counts": feature_counts,
    }
