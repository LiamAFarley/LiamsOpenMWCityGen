"""Cached owner inflow and two-receiver generated routing for v3 erosion.

Pipeline position
    Stage 7. Owner routing is built once from the structural pre-erosion field;
    generated routing is rebuilt only at the configured cadence. Both use a
    routing-only depression-resolved surface and never replace rendered terrain.

Outputs
    Compact int32 receiver arrays, float32 receiver weights/lengths, a
    descending topological order, static owner inflow, and accumulation.

Invariants
    Receivers are strictly lower in the routing surface, each source has at
    most two receivers, weights sum to one for routed flow, and owner rainfall
    enters generated terrain only at owner-to-generated crossings.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import reconstruction

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


_DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
_DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
_DIST = np.sqrt(_DY.astype(np.float64) ** 2 + _DX.astype(np.float64) ** 2)


def priority_flood_routing_surface(
    height: np.ndarray, domain: np.ndarray
) -> np.ndarray:
    """Fill depressions on a routing copy while preserving the real field."""
    if not np.isfinite(height[domain]).all():
        raise ValueError("routing domain contains non-finite terrain")
    spill = float(np.max(height[domain])) + 1.0
    mask = np.full(height.shape, spill, dtype=np.float32)
    mask[domain] = height[domain]
    boundary = domain & ~ndimage.binary_erosion(
        domain, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    seed = np.full(height.shape, spill, dtype=np.float32)
    seed[boundary] = height[boundary]
    return np.asarray(reconstruction(seed, mask, method="erosion"), dtype=np.float32)


if njit is not None:

    @njit(cache=True)
    def _build_receivers_numba(height, source, receiver_domain):
        h, w = height.shape
        r1 = np.full(h * w, -1, dtype=np.int32)
        r2 = np.full(h * w, -1, dtype=np.int32)
        w1 = np.zeros(h * w, dtype=np.float32)
        w2 = np.zeros(h * w, dtype=np.float32)
        l1 = np.zeros(h * w, dtype=np.float32)
        l2 = np.zeros(h * w, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                flat = y * w + x
                if not source[y, x]:
                    continue
                here = height[y, x]
                # The local negative gradient gives a continuous direction.
                left = height[y, x - 1] if x > 0 else here
                right = height[y, x + 1] if x + 1 < w else here
                up = height[y - 1, x] if y > 0 else here
                down = height[y + 1, x] if y + 1 < h else here
                vx = left - right
                vy = up - down
                norm = (vx * vx + vy * vy) ** 0.5
                best_a = -1.0
                best_b = -1.0
                idx_a = -1
                idx_b = -1
                len_a = 0.0
                len_b = 0.0
                for k in range(8):
                    yy = y + _DY[k]
                    xx = x + _DX[k]
                    if yy < 0 or yy >= h or xx < 0 or xx >= w:
                        continue
                    if not receiver_domain[yy, xx] or height[yy, xx] >= here:
                        continue
                    drop = here - height[yy, xx]
                    d = _DIST[k]
                    if norm > 1e-9:
                        direction = (vx * (_DX[k] / d) + vy * (_DY[k] / d)) / norm
                        angular = max(direction, 0.0)
                    else:
                        angular = 0.0
                    score = angular * drop
                    if score <= 0.0:
                        continue
                    if score > best_a:
                        best_b = best_a
                        idx_b = idx_a
                        len_b = len_a
                        best_a = score
                        idx_a = yy * w + xx
                        len_a = d * 128.0
                    elif score > best_b:
                        best_b = score
                        idx_b = yy * w + xx
                        len_b = d * 128.0
                if idx_a < 0:
                    # A flat/poorly resolved gradient still gets the steepest
                    # lower receiver instead of silently losing rainfall.
                    best_drop = 0.0
                    for k in range(8):
                        yy = y + _DY[k]
                        xx = x + _DX[k]
                        if yy < 0 or yy >= h or xx < 0 or xx >= w:
                            continue
                        if not receiver_domain[yy, xx]:
                            continue
                        drop = here - height[yy, xx]
                        if drop > best_drop:
                            best_drop = drop
                            idx_a = yy * w + xx
                            len_a = _DIST[k] * 128.0
                if idx_a >= 0:
                    if idx_b >= 0:
                        total = best_a + best_b
                        r1[flat] = idx_a
                        r2[flat] = idx_b
                        w1[flat] = best_a / total
                        w2[flat] = best_b / total
                        l1[flat] = len_a
                        l2[flat] = len_b
                    else:
                        r1[flat] = idx_a
                        w1[flat] = 1.0
                        l1[flat] = len_a
        return r1, r2, w1, w2, l1, l2


    @njit(cache=True)
    def _accumulate_generated_numba(order, receiver1, receiver2, weight1,
                                    weight2, source_mask, owner_inflow):
        n = receiver1.size
        accumulation = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if source_mask[i]:
                accumulation[i] = 1.0 + owner_inflow[i]
        for q in range(order.size):
            i = order[q]
            amount = accumulation[i]
            if amount <= 0.0:
                continue
            j = receiver1[i]
            if j >= 0:
                accumulation[j] += amount * weight1[i]
            j = receiver2[i]
            if j >= 0:
                accumulation[j] += amount * weight2[i]
        return accumulation


    @njit(cache=True)
    def _accumulate_owner_numba(order, receiver1, receiver2, weight1,
                                weight2, owner_mask):
        n = receiver1.size
        accumulation = np.zeros(n, dtype=np.float64)
        inflow = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if owner_mask[i]:
                accumulation[i] = 1.0
        for q in range(order.size):
            i = order[q]
            if not owner_mask[i]:
                continue
            amount = accumulation[i]
            j = receiver1[i]
            if j >= 0:
                if owner_mask[j]:
                    accumulation[j] += amount * weight1[i]
                else:
                    inflow[j] += amount * weight1[i]
            j = receiver2[i]
            if j >= 0:
                if owner_mask[j]:
                    accumulation[j] += amount * weight2[i]
                else:
                    inflow[j] += amount * weight2[i]
        return accumulation, inflow

else:  # pragma: no cover

    def _build_receivers_numba(*args):
        raise RuntimeError("numba is required for production routing")

    def _accumulate_generated_numba(*args):
        raise RuntimeError("numba is required for production routing")

    def _accumulate_owner_numba(*args):
        raise RuntimeError("numba is required for production routing")


def _topological_order(height: np.ndarray, source: np.ndarray) -> np.ndarray:
    flat = np.flatnonzero(source.ravel())
    return flat[np.argsort(-height.ravel()[flat], kind="stable")].astype(np.int32)


def build_two_receiver_graph(
    routing_surface: np.ndarray,
    source_mask: np.ndarray,
    receiver_domain: np.ndarray,
) -> dict:
    """Build a compact angular two-receiver graph for one routing surface."""
    r1, r2, w1, w2, l1, l2 = _build_receivers_numba(
        routing_surface.astype(np.float32), source_mask, receiver_domain
    )
    order = _topological_order(routing_surface, source_mask)
    routed = (r1 >= 0) & source_mask.ravel()
    if np.any(routed):
        weight_sum = w1[routed] + w2[routed]
        if np.max(np.abs(weight_sum - 1.0)) > 1e-5:
            raise AssertionError("two-receiver routing weights do not normalize")
        if np.any(r1[routed] >= 0):
            flat_h = routing_surface.ravel()
            if np.any(flat_h[r1[routed]] >= flat_h[np.flatnonzero(routed)]):
                raise AssertionError("routing receiver is not strictly lower")
    return {
        "receiver_1": r1,
        "receiver_2": r2,
        "weight_1": w1,
        "weight_2": w2,
        "length_1": l1,
        "length_2": l2,
        "order": order,
        "source_mask": source_mask.ravel().copy(),
        "receiver_domain": receiver_domain.ravel().copy(),
        "shape": routing_surface.shape,
    }


def build_owner_inflow(
    routing_surface: np.ndarray,
    owner_mask: np.ndarray,
    generated_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Route one static rainfall pass through owner cells to generated edges."""
    source = owner_mask
    receiver_domain = owner_mask | generated_mask
    graph = build_two_receiver_graph(routing_surface, source, receiver_domain)
    _, inflow = _accumulate_owner_numba(
        graph["order"], graph["receiver_1"], graph["receiver_2"],
        graph["weight_1"], graph["weight_2"], graph["source_mask"]
    )
    report = {
        "owner_vertices": int(owner_mask.sum()),
        "generated_crossing_vertices": int(np.count_nonzero(inflow)),
        "owner_inflow_total": float(inflow.sum()),
        "owner_routed_vertices": int(np.count_nonzero(
            graph["receiver_1"] >= 0
        )),
    }
    return inflow.reshape(routing_surface.shape).astype(np.float32), report


def prepare_generated_routing(
    height: np.ndarray,
    generated_mask: np.ndarray,
    owner_mask: np.ndarray,
    owner_inflow: np.ndarray,
    config: dict,
    *,
    seed_offset: int = 0,
) -> tuple[dict, np.ndarray, dict]:
    """Build generated routing once for a cycle cadence."""
    domain = generated_mask | owner_mask
    amplitude = float(config.get("routing_perturbation_gu", 6.0))
    sigma = float(config.get("routing_perturbation_sigma_verts", 4.0))
    frac = float(config.get("routing_perturbation_relief_fraction", 0.02))
    rng = np.random.default_rng(int(config.get("seed", 0)) + seed_offset)
    noise = ndimage.gaussian_filter(
        rng.normal(size=height.shape).astype(np.float32), sigma, mode="nearest"
    )
    noise /= max(float(np.std(noise[generated_mask])), 1e-6)
    local_relief = ndimage.maximum_filter(height, size=17) - ndimage.minimum_filter(
        height, size=17
    )
    amp = np.minimum(amplitude, np.maximum(local_relief, 1.0) * frac)
    routing_surface = height.astype(np.float32, copy=True)
    routing_surface[generated_mask] += (noise * amp)[generated_mask]
    routing_surface = priority_flood_routing_surface(routing_surface, domain)
    graph = build_two_receiver_graph(
        routing_surface, generated_mask, generated_mask
    )
    accumulation = _accumulate_generated_numba(
        graph["order"], graph["receiver_1"], graph["receiver_2"],
        graph["weight_1"], graph["weight_2"], graph["source_mask"],
        owner_inflow.ravel().astype(np.float64),
    ).reshape(height.shape)
    graph["routing_surface"] = routing_surface
    graph["routing_defects"] = int(np.count_nonzero(
        generated_mask & (graph["receiver_1"].reshape(height.shape) < 0)
    ))
    return graph, accumulation.astype(np.float32), {
        "routing_domain_vertices": int(domain.sum()),
        "generated_vertices": int(generated_mask.sum()),
        "routing_defects": graph["routing_defects"],
        "perturbation_sigma_verts": sigma,
        "perturbation_max_gu": amplitude,
    }
