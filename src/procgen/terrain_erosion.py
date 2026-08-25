"""Effective geomorphic refinement for a structural terrain window.

Pipeline position
    Stages 7-8. Structural continuation must already have selected the
    mountain, valley, plateau, and coast forms. This module routes rainfall
    through cached owner inflow plus a refreshed two-receiver generated graph,
    then applies calibrated implicit stream-power incision and gentle
    terrain-dependent hillslope transport.

Inputs
    Structural field, generated/owner/fixed masks, Stage-4 feature rasters,
    and the JSON erosion configuration.

Outputs
    Eroded local field and diagnostics including routing defects, owner inflow,
    calibrated ``Kdt``, c-distribution, and height-delta percentiles.

Invariants
    Owner/seam/ring fixed values are restored every cycle. Depression filling,
    routing noise, and owner routing are never written into the rendered field.
    The implicit update cannot overshoot a receiver, and underwater vertices
    receive no terrestrial incision.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage

from procgen.terrain_hydrology import (
    build_owner_inflow,
    prepare_generated_routing,
    priority_flood_routing_surface,
)


def _smootherstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _neighbor_mean(height: np.ndarray, valid: np.ndarray) -> np.ndarray:
    finite = np.isfinite(height) & valid
    source = np.where(finite, height, 0.0).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    total = ndimage.convolve(source, kernel, mode="nearest")
    count = ndimage.convolve(finite.astype(np.float32), kernel, mode="nearest")
    return np.divide(total, count, out=height.copy(), where=count > 0.0)


def _terrain_factor(features: dict, shape: tuple[int, int]) -> np.ndarray:
    owner_mask = np.asarray(features["owner_mask"], dtype=bool)
    factor = np.asarray(features["erosion_factor"], dtype=np.float32)
    out = np.ones(shape, dtype=np.float32)
    out[owner_mask] = factor[owner_mask]
    if owner_mask.any():
        indices = ndimage.distance_transform_edt(
            ~owner_mask, return_distances=False, return_indices=True
        )
        nearest = factor[tuple(indices)]
        out[~owner_mask] = nearest[~owner_mask]
    out[~np.isfinite(out)] = 1.0
    return ndimage.gaussian_filter(out, 8.0, mode="nearest").astype(np.float32)


def _receiver_state(graph: dict, height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_h = height.ravel()
    r1 = graph["receiver_1"]
    r2 = graph["receiver_2"]
    w1 = graph["weight_1"].astype(np.float32)
    w2 = graph["weight_2"].astype(np.float32)
    valid1 = r1 >= 0
    valid2 = r2 >= 0
    h1 = np.zeros(r1.shape, dtype=np.float32)
    h2 = np.zeros(r2.shape, dtype=np.float32)
    h1[valid1] = flat_h[r1[valid1]]
    h2[valid2] = flat_h[r2[valid2]]
    l1 = graph["length_1"]
    l2 = graph["length_2"]
    hrec = w1 * h1 + w2 * h2
    lrec = w1 * l1 + w2 * l2
    return hrec.reshape(height.shape), lrec.reshape(height.shape)


def _calibrate_kdt(
    accumulation: np.ndarray,
    receiver_length: np.ndarray,
    generated_mask: np.ndarray,
    config: dict,
) -> tuple[float, dict]:
    area_ref = float(config.get("area_reference_vertices", 256.0))
    m = float(config.get("stream_power_m", 0.5))
    area_start = float(config.get("channel_area_start_vertices", 32.0))
    area_full = float(config.get("channel_area_full_vertices", 256.0))
    ahat = np.maximum(accumulation / max(area_ref, 1.0), 0.0)
    q = np.divide(
        np.power(ahat, m),
        receiver_length,
        out=np.zeros_like(ahat, dtype=np.float32),
        where=receiver_length > 0.0,
    )
    channel = _channel_strength(accumulation, generated_mask, config)
    channel = channel * (receiver_length > 0.0)
    candidates = generated_mask & (accumulation >= area_start) & (channel > 0.05)
    values = (q * channel)[candidates & np.isfinite(q)]
    if values.size == 0:
        raise ValueError("no channel candidates available for Kdt calibration")
    target = float(config.get("target_c_p90", 0.15))
    q90 = max(float(np.percentile(values, 90.0)), 1e-9)
    kdt = target / q90
    kdt = float(np.clip(
        kdt,
        float(config.get("kdt_min", 1e-4)),
        float(config.get("kdt_max", 1000.0)),
    ))
    return kdt, {
        "area_reference_vertices": area_ref,
        "channel_candidates": int(values.size),
        "q_median": float(np.percentile(values, 50.0)),
        "q_p75": float(np.percentile(values, 75.0)),
        "q_p90": q90,
        "q_p95": float(np.percentile(values, 95.0)),
        "q_max": float(values.max()),
        "target_c_p90": target,
        "chosen_kdt": kdt,
    }


def _channel_strength(
    accumulation: np.ndarray, generated_mask: np.ndarray, config: dict
) -> np.ndarray:
    """Return a spatially softened activation of established catchments."""
    area_start = float(config.get("channel_area_start_vertices", 32.0))
    area_full = float(config.get("channel_area_full_vertices", 256.0))
    log_a = np.log1p(np.maximum(accumulation, 0.0))
    raw = _smootherstep(
        (log_a - np.log1p(area_start)) /
        max(np.log1p(area_full) - np.log1p(area_start), 1e-6)
    )
    sigma = float(config.get("channel_activation_sigma_verts", 8.0))
    num = ndimage.gaussian_filter(
        raw * generated_mask.astype(np.float32), sigma, mode="nearest"
    )
    den = ndimage.gaussian_filter(
        generated_mask.astype(np.float32), sigma, mode="nearest"
    )
    channel = np.divide(
        num, den, out=np.zeros_like(raw, dtype=np.float32), where=den > 1e-6
    )
    channel[~generated_mask] = 0.0
    return channel


def _distribution(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: 0.0 for key in ("median", "p75", "p90", "p95", "p99", "max")}
    return {
        "median": float(np.percentile(values, 50.0)),
        "p75": float(np.percentile(values, 75.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(values.max()),
    }


def erode_field(
    field: np.ndarray,
    generated_mask: np.ndarray,
    owner_mask: np.ndarray,
    fixed_mask: np.ndarray,
    features: dict,
    config: dict,
    *,
    snapshot_callback: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Run calibrated implicit erosion on editable generated terrain."""
    generated_mask = np.asarray(generated_mask, dtype=bool)
    owner_mask = np.asarray(owner_mask, dtype=bool)
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    domain = generated_mask | owner_mask
    editable = generated_mask & ~fixed_mask
    if not editable.any():
        raise ValueError("erosion has no editable generated vertices")
    if not np.isfinite(field[domain]).all():
        raise ValueError("erosion domain contains non-finite terrain")
    work = np.asarray(field, dtype=np.float32).copy()
    initial = work.copy()
    fixed_values = work.copy()
    cycles = int(config.get("cycles", 24))
    snapshot_cycles = {int(v) for v in config.get(
        "snapshot_cycles", [0, 4, 8, 16, 24]
    )}
    reroute_every = max(1, int(config.get("reroute_every", 2)))
    sea_level = float(config.get("sea_level_gu", 0.0))
    hillslope_enabled = bool(config.get("hillslope_enabled", True))
    hillslope_strength = float(config.get("hillslope_strength", 0.02))
    owner_domain = owner_mask & np.isfinite(work)
    static_route = priority_flood_routing_surface(work, domain)
    owner_inflow, owner_report = build_owner_inflow(
        static_route, owner_domain, generated_mask
    )
    graph = None
    accumulation = None
    route_reports = []
    kdt_report = None
    kdt = None
    terrain_factor = _terrain_factor(features, work.shape)
    area_start = float(config.get("channel_area_start_vertices", 32.0))
    area_full = float(config.get("channel_area_full_vertices", 256.0))
    m = float(config.get("stream_power_m", 0.5))
    n_exp = float(config.get("stream_power_n", 1.0))
    cycle_reports = {}
    if snapshot_callback and 0 in snapshot_cycles:
        snapshot_callback(0, work.copy())

    for cycle in range(1, cycles + 1):
        if graph is None or (cycle - 1) % reroute_every == 0:
            graph, accumulation, route_report = prepare_generated_routing(
                work, generated_mask, owner_mask, owner_inflow, config,
                seed_offset=cycle,
            )
            route_reports.append({"cycle": cycle, **route_report})
            hrec, lrec = _receiver_state(graph, work)
            if kdt is None:
                kdt, kdt_report = _calibrate_kdt(
                    accumulation, lrec, generated_mask, config
                )
        hrec, lrec = _receiver_state(graph, work)
        ahat = np.maximum(accumulation / float(
            config.get("area_reference_vertices", 256.0)
        ), 0.0)
        q = np.divide(
            np.power(ahat, m), lrec,
            out=np.zeros_like(ahat, dtype=np.float32),
            where=lrec > 0.0,
        )
        # Threshold on the actual catchment, then blur the activation field.
        # Blurring log(A) itself moves channel heads below the threshold and
        # makes the calibrated erosion silently inert on bounded windows.
        channel = _channel_strength(accumulation, generated_mask, config)
        c_eff = kdt * q * channel * terrain_factor
        valid = editable & (work > sea_level) & (lrec > 0.0)
        flat_work = work.ravel()
        flat_hrec = hrec.ravel()
        flat_c = c_eff.ravel()
        delta = np.zeros(work.size, dtype=np.float32)
        flat_valid = valid.ravel()
        delta[flat_valid] = (
            flat_c[flat_valid] * (flat_hrec[flat_valid] - flat_work[flat_valid])
            / (1.0 + flat_c[flat_valid])
        ).astype(np.float32)
        work += delta.reshape(work.shape)
        positive_before = work.copy()
        if hillslope_enabled and cycle % max(1, int(
            config.get("hillslope_every", 1)
        )) == 0:
            mean = _neighbor_mean(work, domain)
            diffuse = hillslope_strength * terrain_factor * (mean - work)
            diffuse *= 1.0 - 0.75 * channel
            diffuse[~editable] = 0.0
            diffuse[work <= sea_level] = 0.0
            work += diffuse.astype(np.float32)
        work[fixed_mask] = fixed_values[fixed_mask]
        if snapshot_callback and cycle in snapshot_cycles:
            snapshot_callback(cycle, work.copy())
        cycle_reports[str(cycle)] = {
            "active_channel_vertices": int(np.count_nonzero(valid & (channel > 0.0))),
            "c_distribution": _distribution(c_eff[valid]),
            "incision_gu": _distribution(
                np.maximum(-delta.reshape(work.shape)[valid], 0.0)
            ),
            "hillslope_max_positive_gu": float(
                np.maximum(work - positive_before, 0.0).max(initial=0.0)
            ),
        }

    delta = work[editable] - initial[editable]
    local_relief = ndimage.maximum_filter(initial, size=17) - ndimage.minimum_filter(
        initial, size=17
    )
    p95_relief = max(float(np.percentile(local_relief[editable], 95.0)), 1e-6)
    report = {
        "cycles": cycles,
        "reroute_every": reroute_every,
        "generated_vertices": int(generated_mask.sum()),
        "editable_vertices": int(editable.sum()),
        "owner_vertices": int(owner_mask.sum()),
        "owner_inflow": owner_report,
        "routing_rebuilds": len(route_reports),
        "routing_reports": route_reports,
        "kdt_calibration": kdt_report,
        "c_distribution_final": cycle_reports[str(cycles)]["c_distribution"],
        "cycle_reports": cycle_reports,
        "erosion_delta_gu": _distribution(np.abs(delta)),
        "max_incision_gu": float(np.maximum(-delta, 0.0).max(initial=0.0)),
        "max_positive_change_gu": float(np.maximum(delta, 0.0).max(initial=0.0)),
        "p95_delta_over_p95_local_relief": float(
            np.percentile(np.abs(delta), 95.0) / p95_relief
        ),
        "terrain_factor": {
            "min": float(terrain_factor[editable].min()),
            "median": float(np.median(terrain_factor[editable])),
            "max": float(terrain_factor[editable].max()),
        },
    }
    return work, report
