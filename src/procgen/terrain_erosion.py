"""Targeted geomorphic erosion for a solved terrain window.

Purpose
    Route water across generated terrain and a fixed owner halo with an
    eight-neighbor multi-flow direction (MFD) router, then apply modest
    stream-power incision and hillslope relaxation to editable vertices.

Inputs
    A solved local height field, generated/seam/fixed masks, and a JSON erosion
    configuration. Owner halo vertices participate in routing but are never
    changed.

Outputs
    The edited local field plus compact diagnostics. An optional callback
    receives requested snapshot fields without retaining all snapshots in
    memory.

Invariants
    The owner halo, exact seam, and supplied fixed ring are restored after
    every cycle. Routing depressions are adjusted on a copy only; the rendered
    field is not globally priority-filled. No random jitter is used.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage
from skimage.morphology import reconstruction

try:
    from numba import njit
except ImportError:  # pragma: no cover - the production environment supplies it
    njit = None


if njit is not None:

    @njit(cache=True)
    def _mfd_accumulation(height, domain, exponent):
        """Accumulate unit rainfall in descending-height topological order."""
        flat_h = height.ravel()
        flat_domain = domain.ravel()
        order = np.argsort(-flat_h)
        accumulation = np.ones(flat_h.size, dtype=np.float64)
        h, w = height.shape
        dys = (-1, -1, -1, 0, 0, 1, 1, 1)
        dxs = (-1, 0, 1, -1, 1, -1, 0, 1)
        distances = (1.4142135623730951, 1.0, 1.4142135623730951,
                     1.0, 1.0, 1.4142135623730951, 1.0,
                     1.4142135623730951)
        for q in range(order.size):
            flat_i = order[q]
            if not flat_domain[flat_i]:
                continue
            y = flat_i // w
            x = flat_i - y * w
            weights = np.zeros(8, dtype=np.float64)
            total = 0.0
            here = flat_h[flat_i]
            for k in range(8):
                yy = y + dys[k]
                xx = x + dxs[k]
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                flat_j = yy * w + xx
                if not flat_domain[flat_j] or flat_h[flat_j] >= here:
                    continue
                slope = (here - flat_h[flat_j]) / distances[k]
                weight = slope ** exponent
                weights[k] = weight
                total += weight
            if total > 0.0:
                amount = accumulation[flat_i]
                for k in range(8):
                    if weights[k] == 0.0:
                        continue
                    yy = y + dys[k]
                    xx = x + dxs[k]
                    flat_j = yy * w + xx
                    accumulation[flat_j] += amount * weights[k] / total
        return accumulation.reshape((h, w))

else:  # pragma: no cover - prevents an opaque import failure on small fixtures

    def _mfd_accumulation(height, domain, exponent):
        if height.size > 100_000:
            raise RuntimeError("numba is required for production erosion windows")
        order = np.argsort(-height.ravel())
        acc = np.ones(height.size, dtype=np.float64)
        h, w = height.shape
        for flat_i in order:
            if not domain.ravel()[flat_i]:
                continue
            y, x = divmod(int(flat_i), w)
            receivers = []
            weights = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and domain[yy, xx]:
                        d = 2 ** 0.5 if dy and dx else 1.0
                        s = (height[y, x] - height[yy, xx]) / d
                        if s > 0:
                            receivers.append((yy, xx))
                            weights.append(s ** exponent)
            total = sum(weights)
            if total:
                for (yy, xx), weight in zip(receivers, weights):
                    acc[yy * w + xx] += acc[flat_i] * weight / total
        return acc.reshape(height.shape)


def _component_boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(
        mask, structure=np.ones((3, 3), dtype=bool), border_value=0
    )


def priority_flood_routing_surface(
    height: np.ndarray, component: np.ndarray
) -> np.ndarray:
    """Resolve depressions for routing while preserving the actual field."""
    if not np.isfinite(height[component]).all():
        raise ValueError("erosion routing component contains non-finite terrain")
    spill = float(np.max(height[component])) + 1.0
    mask = np.full(height.shape, spill, dtype=np.float32)
    mask[component] = height[component]
    seed = np.full(height.shape, spill, dtype=np.float32)
    boundary = _component_boundary(component)
    seed[boundary] = height[boundary]
    filled = reconstruction(seed, mask, method="erosion")
    return np.asarray(filled, dtype=np.float32)


def _local_lowest_relief(height: np.ndarray, component: np.ndarray):
    """Return the lowest downhill receiver relief and its distance."""
    lowest = np.full(height.shape, np.inf, dtype=np.float32)
    distances = np.full(height.shape, np.inf, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            d = 181.01933598375618 if dy and dx else 128.0
            shifted = np.full(height.shape, np.inf, dtype=np.float32)
            ys = slice(max(0, dy), min(height.shape[0], height.shape[0] + dy))
            xs = slice(max(0, dx), min(height.shape[1], height.shape[1] + dx))
            src_y = slice(max(0, -dy), min(height.shape[0], height.shape[0] - dy))
            src_x = slice(max(0, -dx), min(height.shape[1], height.shape[1] - dx))
            shifted[ys, xs] = height[src_y, src_x]
            lower = component & (shifted < height)
            replace = lower & (shifted < lowest)
            lowest[replace] = shifted[replace]
            distances[replace] = d
    relief = np.maximum(height - lowest, 0.0)
    return relief, distances


def _neighbor_mean(height: np.ndarray, component: np.ndarray) -> np.ndarray:
    total = np.zeros(height.shape, dtype=np.float32)
    count = np.zeros(height.shape, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            shifted = np.zeros(height.shape, dtype=np.float32)
            valid = np.zeros(height.shape, dtype=bool)
            ys = slice(max(0, dy), min(height.shape[0], height.shape[0] + dy))
            xs = slice(max(0, dx), min(height.shape[1], height.shape[1] + dx))
            src_y = slice(max(0, -dy), min(height.shape[0], height.shape[0] - dy))
            src_x = slice(max(0, -dx), min(height.shape[1], height.shape[1] - dx))
            shifted[ys, xs] = height[src_y, src_x]
            valid[ys, xs] = component[src_y, src_x]
            take = component & valid
            total[take] += shifted[take]
            count[take] += 1.0
    return np.divide(total, count, out=height.copy(), where=count > 0)


def _deterministic_perturbation(shape, amplitude_gu: float) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    return amplitude_gu * 0.5 * (
        np.sin(xx / 23.0) + np.cos(yy / 29.0)
    )


def erode_field(
    field: np.ndarray,
    generated_mask: np.ndarray,
    owner_halo_mask: np.ndarray,
    fixed_mask: np.ndarray,
    config: dict,
    *,
    snapshot_callback: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Run configured MFD erosion on editable generated terrain."""
    if not np.isfinite(field[generated_mask | owner_halo_mask]).all():
        raise ValueError("erosion domain contains non-finite heights")
    domain = generated_mask | owner_halo_mask
    editable = generated_mask & ~fixed_mask
    if not editable.any():
        raise ValueError("erosion has no editable generated vertices")
    work = np.asarray(field, dtype=np.float32).copy()
    fixed_values = work.copy()
    cycles = int(config.get("cycles", 16))
    snapshot_cycles = {int(v) for v in config.get("snapshot_cycles", [0, 4, 8, 16])}
    exponent = float(config.get("mfd_exponent", 1.3))
    stream_strength = float(config.get("incision_strength_gu", 3.0))
    accumulation_ref_percentile = float(
        config.get("accumulation_ref_percentile", 95.0)
    )
    stream_m = float(config.get("stream_power_m", 0.5))
    stream_n = float(config.get("stream_power_n", 1.0))
    stability_fraction = float(config.get("stability_fraction", 0.25))
    relaxation = float(config.get("hillslope_relaxation", 0.03))
    perturbation = float(config.get("routing_perturbation_gu", 8.0))
    sea_level = float(config.get("sea_level_gu", 0.0))
    labels, count = ndimage.label(domain, structure=np.ones((3, 3), dtype=bool))
    components = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        if ys.size:
            components.append((label, int(ys.min()), int(ys.max()) + 1,
                               int(xs.min()), int(xs.max()) + 1))
    if snapshot_callback and 0 in snapshot_cycles:
        snapshot_callback(0, work.copy())
    total_incision = 0.0
    max_incision = 0.0
    max_accumulation = 0.0
    for cycle in range(1, cycles + 1):
        for label, r0, r1, c0, c1 in components:
            component = labels[r0:r1, c0:c1] == label
            h = work[r0:r1, c0:c1]
            edit = editable[r0:r1, c0:c1]
            route = priority_flood_routing_surface(h, component)
            route += _deterministic_perturbation(route.shape, perturbation) * component
            accumulation = _mfd_accumulation(route, component, exponent)
            candidates = edit & (h > sea_level)
            if not candidates.any():
                continue
            ref = float(np.percentile(accumulation[candidates],
                                      accumulation_ref_percentile))
            ref = max(ref, 1.0)
            relief, distances = _local_lowest_relief(h, component)
            slope = relief / np.maximum(distances, 1.0)
            ahat = accumulation / ref
            incision = stream_strength * np.power(ahat, stream_m) * np.power(
                slope, stream_n
            )
            max_incise = stability_fraction * relief
            delta = np.minimum(np.maximum(incision, 0.0), max_incise)
            delta[~candidates] = 0.0
            h -= delta.astype(np.float32)
            total_incision += float(delta.sum())
            max_incision = max(max_incision, float(delta.max(initial=0.0)))
            max_accumulation = max(max_accumulation,
                                   float(accumulation.max(initial=0.0)))
            if relaxation > 0.0:
                mean = _neighbor_mean(h, component)
                h[candidates] += relaxation * (mean[candidates] - h[candidates])
            work[r0:r1, c0:c1] = h
        work[fixed_mask] = fixed_values[fixed_mask]
        if snapshot_callback and cycle in snapshot_cycles:
            snapshot_callback(cycle, work.copy())
    report = {
        "cycles": cycles,
        "components": len(components),
        "domain_vertices": int(domain.sum()),
        "editable_vertices": int(editable.sum()),
        "owner_halo_vertices": int(owner_halo_mask.sum()),
        "total_incision_gu": round(total_incision, 3),
        "max_incision_gu": round(max_incision, 3),
        "max_accumulation": round(max_accumulation, 3),
        "mfd_exponent": exponent,
        "stream_power_m": stream_m,
        "stream_power_n": stream_n,
        "stability_fraction": stability_fraction,
    }
    return work, report
