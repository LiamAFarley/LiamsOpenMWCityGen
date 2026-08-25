"""Semantic plateau/canyon primitive inference for a local terrain seam.

Pipeline position
    Primitive P1-P5 structural checkpoint after the accepted Stage-3 field and
    owner feature analysis. P1 in this module identifies seam-crossing plateau
    components and fits low-frequency owner top surfaces; later stages attach
    support, canyon candidates, and reconciliation fields.

Authority
    Only vertices inside ``owner_mask`` are used to infer a primitive. The
    generated Stage-3 field is a compatibility/background signal, never an
    owner-height source. All expensive component work is performed on the
    configured semantic grid; full-resolution seam vertices are retained for
    contact and final candidate sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Any

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

try:
    import pyamg
except ImportError:  # pragma: no cover
    pyamg = None


@dataclass
class TerrainPrimitive:
    primitive_id: int
    kind: str
    confidence: float
    owner_component_id: int
    owner_bbox: tuple[int, int, int, int]
    seam_vertices: np.ndarray
    support_mask: np.ndarray | None = None
    target_height: np.ndarray | None = None
    target_weight: np.ndarray | None = None
    erosion_class: str = "plateau"
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _smootherstep(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


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


def _slices_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[slice, slice, tuple[int, int, int, int]]:
    r0, r1, c0, c1 = map(int, bbox)
    r0 = max(0, min(shape[0], r0))
    r1 = max(r0, min(shape[0], r1))
    c0 = max(0, min(shape[1], c0))
    c1 = max(c0, min(shape[1], c1))
    return slice(r0, r1), slice(c0, c1), (r0, r1, c0, c1)


def _block_mean(field: np.ndarray, valid: np.ndarray, factor: int) -> np.ndarray:
    h, w = field.shape
    sh = (h + factor - 1) // factor
    sw = (w + factor - 1) // factor
    ph = sh * factor - h
    pw = sw * factor - w
    values = np.where(valid, field, 0.0).astype(np.float32)
    values = np.pad(values, ((0, ph), (0, pw)))
    mask = np.pad(valid.astype(np.float32), ((0, ph), (0, pw)))
    values = values.reshape(sh, factor, sw, factor).sum(axis=(1, 3))
    counts = mask.reshape(sh, factor, sw, factor).sum(axis=(1, 3))
    return np.divide(
        values,
        counts,
        out=np.zeros((sh, sw), dtype=np.float32),
        where=counts > 0.0,
    )


def _block_max(mask: np.ndarray, factor: int) -> np.ndarray:
    h, w = mask.shape
    sh = (h + factor - 1) // factor
    sw = (w + factor - 1) // factor
    ph = sh * factor - h
    pw = sw * factor - w
    padded = np.pad(mask, ((0, ph), (0, pw)), constant_values=False)
    return padded.reshape(sh, factor, sw, factor).max(axis=(1, 3))


def _upsample_semantic(mask: np.ndarray, factor: int, shape: tuple[int, int]) -> np.ndarray:
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)[
        : shape[0], : shape[1]
    ]


def _owner_seam_mask(ctx: dict, shape: tuple[int, int]) -> np.ndarray:
    """Return owner-side vertices immediately adjacent to production seam legs."""
    out = np.zeros(shape, dtype=bool)
    for edge in ctx["edge_list"]:
        normal = tuple(int(round(v)) for v in edge["normal"])
        for flat in edge["verts"]:
            sy, sx = divmod(int(flat), shape[1])
            oy, ox = sy - normal[0], sx - normal[1]
            if 0 <= oy < shape[0] and 0 <= ox < shape[1]:
                out[oy, ox] = True
    return out


def _huber_plane_fit(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    *,
    allow_quadratic: bool,
    quadratic_regularization: float,
    quadratic_trigger_gu: float,
) -> tuple[dict[str, Any], np.ndarray]:
    if values.size < 3:
        raise ValueError("plateau top fit needs at least three owner samples")
    cy = float(np.mean(rows))
    cx = float(np.mean(cols))
    scale = max(float(np.ptp(np.concatenate((rows, cols)))), 1.0)
    y = (rows - cy) / scale
    x = (cols - cx) / scale
    affine = np.column_stack((x, y, np.ones_like(x)))
    design = affine
    order = "affine"
    weights = np.ones(values.size, dtype=np.float64)
    coeff = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(8):
        weighted = design * weights[:, None]
        coeff = np.linalg.lstsq(weighted, values * weights, rcond=None)[0]
        residual = values - design @ coeff
        mad = float(np.median(np.abs(residual - np.median(residual))))
        delta = max(64.0, 1.5 * mad)
        weights = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-6))
    residual = values - design @ coeff
    rms = float(np.sqrt(np.mean(residual * residual)))
    p95 = float(np.percentile(np.abs(residual), 95))

    if allow_quadratic and p95 > float(quadratic_trigger_gu) and values.size >= 8:
        quadratic = np.column_stack((x, y, np.ones_like(x), x * x, x * y, y * y))
        regularizer = np.zeros((6, 6), dtype=np.float64)
        regularizer[3:, 3:] = float(quadratic_regularization)
        lhs = quadratic.T @ (weights[:, None] * quadratic) + regularizer
        rhs = quadratic.T @ (weights * values)
        qcoeff = np.linalg.solve(lhs, rhs)
        qresidual = values - quadratic @ qcoeff
        if float(np.sqrt(np.mean(qresidual * qresidual))) < rms:
            design = quadratic
            coeff = qcoeff
            residual = qresidual
            rms = float(np.sqrt(np.mean(residual * residual)))
            p95 = float(np.percentile(np.abs(residual), 95))
            order = "quadratic"

    tilt_x = float(coeff[0] / scale)
    tilt_y = float(coeff[1] / scale)
    return {
        "order": order,
        "coefficients": coeff.tolist(),
        "center_row": cy,
        "center_col": cx,
        "coordinate_scale": scale,
        "fit_rms_gu": rms,
        "fit_p95_gu": p95,
        "tilt_x_gu_per_gu": tilt_x,
        "tilt_y_gu_per_gu": tilt_y,
        "tilt_magnitude_gu_per_gu": float(np.hypot(tilt_x, tilt_y)),
        "sample_count": int(values.size),
    }, residual.astype(np.float32)


def evaluate_top_fit(fit: dict[str, Any], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    output_shape = np.asarray(rows).shape
    rows = np.asarray(rows, dtype=np.float64).ravel() - float(fit.get("row_offset", 0.0))
    cols = np.asarray(cols, dtype=np.float64).ravel() - float(fit.get("col_offset", 0.0))
    cy = float(fit["center_row"])
    cx = float(fit["center_col"])
    scale = float(fit["coordinate_scale"])
    y = (rows - cy) / scale
    x = (cols - cx) / scale
    coeff = np.asarray(fit["coefficients"], dtype=np.float64)
    if fit["order"] == "quadratic":
        design = np.column_stack((x, y, np.ones_like(x), x * x, x * y, y * y))
    else:
        design = np.column_stack((x, y, np.ones_like(x)))
    return (design @ coeff).reshape(output_shape).astype(np.float32)


def _plateau_score(
    h64: np.ndarray,
    slope24: np.ndarray,
    scarp_mask: np.ndarray,
    valid: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, dict]:
    finite_h = h64[valid & np.isfinite(h64)]
    finite_slope = slope24[valid & np.isfinite(slope24)]
    if finite_h.size == 0:
        raise ValueError("plateau analysis has no finite owner H64 samples")
    z_low = float(np.percentile(finite_h, config.get("elevation_low_percentile", 60.0)))
    z_high = float(np.percentile(finite_h, config.get("elevation_high_percentile", 88.0)))
    slope_ref = float(np.percentile(
        finite_slope, config.get("flat_slope_percentile", 35.0)
    ))
    baseline = _normalized_gaussian(h64, valid, config.get("prominence_sigma_verts", 32.0))
    prominence = h64 - baseline
    positive_prominence = prominence[valid & (prominence > 0.0)]
    prom_cut = float(np.percentile(
        positive_prominence, config.get("prominence_percentile", 65.0)
    )) if positive_prominence.size else np.inf
    elevation = _smootherstep((h64 - z_low) / max(z_high - z_low, 1.0))
    flat = np.exp(-(
        slope24 / max(slope_ref, 1e-6)
    ) ** 2).astype(np.float32)
    prominence_score = np.clip(
        (prominence - prom_cut) / max(
            abs(prom_cut), float(config.get("prominence_scale_gu", 256.0))
        ) + 0.5,
        0.0,
        1.0,
    )
    near_scarp = ndimage.binary_dilation(scarp_mask, iterations=8)
    score = (elevation * flat * prominence_score).astype(np.float32)
    score[~valid] = 0.0
    candidate = valid & (score >= float(config.get("min_confidence", 0.55)))
    candidate &= (prominence >= prom_cut) | near_scarp
    return score, {
        "elevation_low_gu": z_low,
        "elevation_high_gu": z_high,
        "slope_reference_gu_per_gu": slope_ref,
        "prominence_cut_gu": prom_cut,
        "near_scarp_vertices": int((near_scarp & valid).sum()),
        "candidate_vertices": int(candidate.sum()),
        "candidate_mask": candidate,
        "prominence": prominence.astype(np.float32),
    }


def analyze_plateaus(
    h0: np.ndarray,
    ctx: dict,
    features: dict,
    bbox: tuple[int, int, int, int],
    config: dict,
) -> tuple[list[TerrainPrimitive], dict[str, np.ndarray], dict]:
    """Detect seam-contacting plateau components and fit their owner tops."""
    rs, cs, clipped = _slices_bbox(bbox, h0.shape)
    owner = np.asarray(features["owner_mask"][rs, cs], dtype=bool)
    h64 = np.asarray(features["H64"][rs, cs], dtype=np.float32)
    h24 = np.asarray(features["H24"][rs, cs], dtype=np.float32)
    slope = np.asarray(features["slope24"][rs, cs], dtype=np.float32)
    scarp = np.asarray(features["scarp_mask"][rs, cs], dtype=bool)
    valid = owner & np.isfinite(h64) & np.isfinite(h24)
    plateau_cfg = dict(config.get("plateau", {}))
    score, score_report = _plateau_score(h64, slope, scarp, valid, plateau_cfg)
    candidate = score_report.pop("candidate_mask")
    factor = max(1, int(config.get("semantic_downsample", 4)))
    sem_score = _block_mean(score, valid, factor)
    sem_candidate = _block_max(candidate, factor) & (sem_score >= float(
        plateau_cfg.get("min_confidence", 0.55)
    ) * 0.8)
    sem_candidate = ndimage.binary_closing(sem_candidate, iterations=1)
    min_vertices = int(config.get("plateau", {}).get(
        "min_component_vertices_semantic", 24
    ))
    labels, count = ndimage.label(sem_candidate, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sem_candidate &= labels > 0
    sem_candidate &= sizes[labels] >= min_vertices
    labels, count = ndimage.label(sem_candidate, structure=np.ones((3, 3), dtype=bool))

    owner_seam_full = _owner_seam_mask(ctx, h0.shape)[rs, cs]
    seam_normal_map = _seam_normals(ctx, h0.shape)
    sem_owner_seam = _block_max(owner_seam_full, factor)
    primitives: list[TerrainPrimitive] = []
    component_summaries = []
    full_labels = np.zeros(owner.shape, dtype=np.int32)
    next_id = 0
    for component_id in range(1, count + 1):
        sem_component = labels == component_id
        if not np.any(ndimage.binary_dilation(sem_component, iterations=1) & sem_owner_seam):
            continue
        full_component = (
            _upsample_semantic(sem_component, factor, owner.shape)
            & owner
            & candidate
        )
        rows, cols = np.nonzero(full_component & valid)
        if rows.size < 3:
            continue
        full_labels[full_component] = next_id + 1
        contact = np.argwhere(full_component & owner_seam_full)
        fit, residual = _huber_plane_fit(
            rows.astype(np.float64), cols.astype(np.float64), h24[rows, cols].astype(np.float64),
            allow_quadratic=bool(plateau_cfg.get("allow_quadratic_fit", True)),
            quadratic_regularization=float(plateau_cfg.get("quadratic_regularization", 0.01)),
            quadratic_trigger_gu=float(plateau_cfg.get("quadratic_trigger_gu", 512.0)),
        )
        fit["row_offset"] = clipped[0]
        fit["col_offset"] = clipped[2]
        bbox_rows = np.nonzero(full_component)[0]
        bbox_cols = np.nonzero(full_component)[1]
        local_bbox = (
            clipped[0] + int(bbox_rows.min()),
            clipped[0] + int(bbox_rows.max()) + 1,
            clipped[2] + int(bbox_cols.min()),
            clipped[2] + int(bbox_cols.max()) + 1,
        )
        confidence = float(np.mean(score[full_component]))
        contact_global = contact.astype(np.int32)
        contact_global[:, 0] += clipped[0]
        contact_global[:, 1] += clipped[2]
        seam_vertices = []
        for row, col in contact_global:
            normal = seam_normal_map.get(int(row) * h0.shape[1] + int(col))
            if normal is not None:
                seam_vertices.append((int(row) + normal[0], int(col) + normal[1]))
        seam_vertices = np.asarray(seam_vertices, dtype=np.int32).reshape((-1, 2))
        primitive = TerrainPrimitive(
            primitive_id=next_id,
            kind="plateau",
            confidence=confidence,
            owner_component_id=int(component_id),
            owner_bbox=local_bbox,
            seam_vertices=seam_vertices,
            diagnostics={
                "fit": fit,
                "fit_residual_gu": residual.tolist(),
                "owner_vertices": int(full_component.sum()),
                "seam_contact_vertices": int(contact.shape[0]),
                "owner_contact_vertices": contact_global.tolist(),
                "mean_score": confidence,
            },
        )
        primitives.append(primitive)
        component_summaries.append({
            "primitive_id": next_id,
            "owner_component_id": int(component_id),
            "owner_vertices": int(full_component.sum()),
            "seam_contact_vertices": int(contact.shape[0]),
            "bbox": list(local_bbox),
            "confidence": confidence,
            "fit": fit,
        })
        next_id += 1

    arrays = {
        "plateau_score": score,
        "plateau_candidate": candidate,
        "plateau_labels": full_labels,
        "owner_seam": owner_seam_full,
        "prominence": score_report["prominence"],
        "bbox": np.asarray(clipped, dtype=np.int32),
    }
    report = {
        "bbox": list(clipped),
        "semantic_downsample": factor,
        "semantic_shape": list(sem_candidate.shape),
        "semantic_candidate_components": int(count),
        "selected_components": component_summaries,
        "score": {k: v for k, v in score_report.items() if k != "prominence"},
    }
    return primitives, arrays, report


def _seam_normals(ctx: dict, shape: tuple[int, int]) -> dict[int, tuple[int, int]]:
    normals: dict[int, tuple[int, int]] = {}
    for edge in ctx["edge_list"]:
        normal = tuple(int(round(v)) for v in edge["normal"])
        for flat in edge["verts"]:
            normals.setdefault(int(flat), normal)
            sy, sx = divmod(int(flat), shape[1])
            oy, ox = sy - normal[0], sx - normal[1]
            if 0 <= oy < shape[0] and 0 <= ox < shape[1]:
                normals.setdefault(oy * shape[1] + ox, normal)
    return normals


def _first_generated_vertex(
    row: int,
    col: int,
    normal: tuple[int, int],
    generated: np.ndarray,
    max_steps: int,
) -> tuple[int, int, int] | None:
    """Walk from a shared seam to the first vertex outside owner authority."""
    for step in range(max(0, int(max_steps)) + 1):
        sy = int(row + normal[0] * step)
        sx = int(col + normal[1] * step)
        if 0 <= sy < generated.shape[0] and 0 <= sx < generated.shape[1]:
            if generated[sy, sx]:
                return sy, sx, step
    return None


def _semantic_support(
    seeds: list[tuple[int, int]],
    generated: np.ndarray,
    h64: np.ndarray,
    top_height: np.ndarray,
    slope: np.ndarray,
    direction: tuple[float, float],
    factor: int,
    max_cells: float,
    eta: float,
    cost_multiplier: float,
) -> tuple[np.ndarray, dict]:
    """Run a bounded, direction-biased semantic Dijkstra from seam seeds."""
    if not seeds:
        return np.full(generated.shape, np.inf, dtype=np.float32), {
            "seed_count": 0,
            "visited": 0,
            "max_cost": 0.0,
        }
    height_scale = max(float(np.percentile(np.abs(h64[np.isfinite(h64)]), 75)) if np.isfinite(h64).any() else 256.0, 256.0)
    slope_values = slope[np.isfinite(slope)]
    slope_scale = max(float(np.percentile(slope_values, 75)) if slope_values.size else 1.0, 1e-6)
    sem_h = (h64.shape[0] + factor - 1) // factor
    sem_w = (h64.shape[1] + factor - 1) // factor
    sem_generated = _block_max(generated, factor)
    sem_h64 = _block_mean(h64, np.isfinite(h64), factor)
    sem_slope = _block_mean(slope, np.isfinite(slope), factor)
    sem_top = _block_mean(top_height, np.isfinite(top_height), factor)
    dist = np.full((sem_h, sem_w), np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for row, col in seeds:
        sy = min(sem_h - 1, max(0, int(row // factor)))
        sx = min(sem_w - 1, max(0, int(col // factor)))
        if not sem_generated[sy, sx]:
            continue
        if dist[sy, sx] > 0.0:
            dist[sy, sx] = 0.0
            heapq.heappush(heap, (0.0, sy, sx))
    max_cost = max(
        float(max_cells) * 64.0 / max(float(factor), 1.0) * float(cost_multiplier),
        1.0,
    )
    dy_pref, dx_pref = direction
    norm = max(float(np.hypot(dy_pref, dx_pref)), 1e-6)
    dy_pref /= norm
    dx_pref /= norm
    while heap:
        cost, row, col = heapq.heappop(heap)
        if cost != dist[row, col] or cost > max_cost:
            continue
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = row + dy, col + dx
            if not (0 <= nr < sem_h and 0 <= nc < sem_w) or not sem_generated[nr, nc]:
                continue
            alignment = max(0.0, dy * dy_pref + dx * dx_pref)
            lateral = abs(dy * dx_pref - dx * dy_pref)
            directional = float(eta) + (1.0 - float(eta)) * alignment * alignment
            compatibility = 1.0
            if np.isfinite(sem_h64[nr, nc]) and np.isfinite(sem_top[nr, nc]):
                compatibility += abs(float(sem_h64[nr, nc] - sem_top[nr, nc])) / height_scale
            if np.isfinite(sem_slope[nr, nc]):
                compatibility += float(sem_slope[nr, nc]) / slope_scale
            compatibility += (1.0 - float(eta)) * lateral
            step_cost = compatibility / max(directional, 1e-3)
            new_cost = cost + step_cost
            if new_cost < dist[nr, nc] and new_cost <= max_cost:
                dist[nr, nc] = new_cost
                heapq.heappush(heap, (new_cost, nr, nc))
    full_dist = np.repeat(np.repeat(dist, factor, axis=0), factor, axis=1)
    full_dist = full_dist[: generated.shape[0], : generated.shape[1]].astype(np.float32)
    return full_dist, {
        "seed_count": len(seeds),
        "visited": int(np.isfinite(dist).sum()),
        "max_cost": max_cost,
        "height_scale_gu": height_scale,
        "slope_scale_gu_per_gu": slope_scale,
    }


def continue_plateau_footprints(
    h0: np.ndarray,
    ctx: dict,
    features: dict,
    primitives: list[TerrainPrimitive],
    bbox: tuple[int, int, int, int],
    config: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """Continue plateau support and owner scarps over the generated side."""
    shape = h0.shape
    primitive_cfg = config.get("plateau", {})
    scarp_cfg = config.get("scarp", {})
    factor = max(1, int(config.get("semantic_downsample", 4)))
    margin = int(np.ceil(float(config.get("bbox_margin_cells", 2.0)) * 64.0))
    max_cells = float(primitive_cfg.get("max_continuation_cells", 8.0))
    seam_seed_max_steps = int(round(float(config.get("seam_seed_max_steps", 2.0))))
    margin += int(np.ceil(max_cells * 64.0))
    r0, r1, c0, c1 = map(int, bbox)
    work_bbox = (
        max(0, r0 - margin), min(shape[0], r1 + margin),
        max(0, c0 - margin), min(shape[1], c1 + margin),
    )
    wr, wc, _ = _slices_bbox(work_bbox, shape)
    generated = np.asarray(ctx["smask"], dtype=bool) & ~np.asarray(ctx["owner_mask"], dtype=bool)
    generated_work = generated[wr, wc]
    valid = generated_work & np.isfinite(h0[wr, wc])
    h64_full = _normalized_gaussian(
        h0, np.asarray(ctx["smask"], dtype=bool) & np.isfinite(h0), 64.0
    )
    h64_work = h64_full[wr, wc]
    slope_work = np.hypot(
        np.gradient(h64_work, axis=0), np.gradient(h64_work, axis=1)
    ).astype(np.float32)
    normals = _seam_normals(ctx, shape)
    support_total = np.zeros(shape, dtype=np.float32)
    scarp_confidence = np.zeros(shape, dtype=np.float32)
    scarp_normal_y = np.zeros(shape, dtype=np.float32)
    scarp_normal_x = np.zeros(shape, dtype=np.float32)
    signed_distance = np.zeros(shape, dtype=np.float32)
    reports = []

    for primitive in primitives:
        seeds: list[tuple[int, int]] = []
        normal_rows = []
        for row, col in primitive.seam_vertices:
            row, col = int(row), int(col)
            normal = normals.get(row * shape[1] + col)
            if normal is None:
                continue
            generated_seed = _first_generated_vertex(
                row, col, normal, generated, seam_seed_max_steps
            )
            if generated_seed is None:
                continue
            gy, gx, _ = generated_seed
            if not (work_bbox[0] <= gy < work_bbox[1] and work_bbox[2] <= gx < work_bbox[3]):
                continue
            ly, lx = gy - work_bbox[0], gx - work_bbox[2]
            if generated_work[ly, lx]:
                seeds.append((ly, lx))
                normal_rows.append(normal)
        if not seeds:
            reports.append({"primitive_id": primitive.primitive_id, "seed_count": 0})
            continue
        normal_vector = np.mean(np.asarray(normal_rows, dtype=np.float32), axis=0)
        normal_vector /= max(float(np.linalg.norm(normal_vector)), 1e-6)
        direction_weight = float(primitive_cfg.get("direction_seam_weight", 0.65))
        direction = (
            float(normal_vector[0]) * direction_weight,
            float(normal_vector[1]) * direction_weight,
        )
        local_rows, local_cols = np.indices(h64_work.shape)
        global_rows = local_rows + work_bbox[0]
        global_cols = local_cols + work_bbox[2]
        top_work = evaluate_top_fit(
            primitive.diagnostics["fit"], global_rows, global_cols
        )
        distance, path_report = _semantic_support(
            seeds, valid, h64_work, top_work, slope_work, direction, factor,
            max_cells, float(primitive_cfg.get("geodesic_lateral_eta", 0.4)),
            float(primitive_cfg.get("geodesic_cost_multiplier", 2.0)),
        )
        edge_cost = float(path_report["max_cost"])
        core_cost = edge_cost * float(primitive_cfg.get("support_core_threshold", 0.65))
        support_work = valid & np.isfinite(distance)
        support_probability = np.zeros(distance.shape, dtype=np.float32)
        support_probability[support_work] = 1.0 - _smootherstep(
            (distance[support_work] - core_cost) / max(edge_cost - core_cost, 1e-6)
        )
        support_probability[~support_work] = 0.0
        support_full = np.zeros(shape, dtype=np.float32)
        support_full[wr, wc] = support_probability
        support_full *= generated

        core = support_full >= float(primitive_cfg.get("support_core_threshold", 0.65))
        core_work = core[wr, wc]
        signed_work = (
            ndimage.distance_transform_edt(core_work)
            - ndimage.distance_transform_edt(~core_work)
        )
        signed_distance[wr, wc] = np.where(
            core_work | (support_probability > 0.0), signed_work, signed_distance[wr, wc]
        )

        owner_scarp = np.asarray(features["scarp_mask"], dtype=bool) & np.asarray(ctx["owner_mask"], dtype=bool)
        width = max(1, int(round(float(scarp_cfg.get("fallback_width_verts", 8.0)))))
        local_scarp = np.zeros(shape, dtype=np.float32)
        local_sy = np.zeros(shape, dtype=np.float32)
        local_sx = np.zeros(shape, dtype=np.float32)
        for row, col in primitive.seam_vertices:
            row, col = int(row), int(col)
            normal = normals.get(row * shape[1] + col)
            if normal is None:
                continue
            nearby = owner_scarp[
                max(0, row - 8):min(shape[0], row + 9),
                max(0, col - 8):min(shape[1], col + 9),
            ].any()
            if not nearby:
                continue
            for step in range(width + 1):
                sy = row + normal[0] * step
                sx = col + normal[1] * step
                if 0 <= sy < shape[0] and 0 <= sx < shape[1] and generated[sy, sx]:
                    value = 1.0 - step / max(width, 1)
                    if value > local_scarp[sy, sx]:
                        local_scarp[sy, sx] = value
                        local_sy[sy, sx] = float(normal[0])
                        local_sx[sy, sx] = float(normal[1])
        primitive.support_mask = support_full > float(primitive_cfg.get("support_edge_threshold", 0.05))
        primitive.target_weight = support_full * float(primitive_cfg.get("weight", 1.0))
        support_total = np.maximum(support_total, support_full)
        replace = local_scarp > scarp_confidence
        scarp_confidence[replace] = local_scarp[replace]
        scarp_normal_y[replace] = local_sy[replace]
        scarp_normal_x[replace] = local_sx[replace]
        reports.append({
            "primitive_id": primitive.primitive_id,
            "direction_owner_to_generated": [float(direction[0]), float(direction[1])],
            "support_vertices": int(np.count_nonzero(support_full > 0.0)),
            "support_core_vertices": int(core.sum()),
            "scarp_vertices": int(np.count_nonzero(local_scarp > 0.0)),
            **path_report,
        })

    arrays = {
        "support_probability": support_total,
        "scarp_confidence": scarp_confidence,
        "scarp_normal_y": scarp_normal_y,
        "scarp_normal_x": scarp_normal_x,
        "signed_distance": signed_distance,
        "work_bbox": np.asarray(work_bbox, dtype=np.int32),
    }
    return arrays, {
        "work_bbox": list(work_bbox),
        "semantic_downsample": factor,
        "primitives": reports,
        "support_vertices": int(np.count_nonzero(support_total > 0.0)),
        "scarp_vertices": int(np.count_nonzero(scarp_confidence > 0.0)),
    }


def synthesize_plateau_candidates(
    h0: np.ndarray,
    ctx: dict,
    primitives: list[TerrainPrimitive],
    footprint_arrays: dict[str, np.ndarray],
    config: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Build complete plateau candidate surfaces without canyon subtraction."""
    shape = h0.shape
    valid = np.asarray(ctx["smask"], dtype=bool) & np.isfinite(h0)
    fine_low = _normalized_gaussian(h0, valid, 4.0)
    fine = h0 - fine_low
    rows, cols = np.indices(shape, dtype=np.float32)
    total_weight = np.zeros(shape, dtype=np.float32)
    weighted_height = np.zeros(shape, dtype=np.float32)
    candidate_support = np.zeros(shape, dtype=bool)
    primitive_reports = []
    plateau_cfg = config.get("plateau", {})
    fine_keep_core = float(config.get("fine_keep_core", 0.1))
    scarp_authority_gain = float(
        config.get("scarp", {}).get("authority_gain", 0.35)
    )
    scarp_confidence = np.asarray(
        footprint_arrays["scarp_confidence"], dtype=np.float32
    )
    for primitive in primitives:
        if primitive.target_weight is None:
            primitive_reports.append({"primitive_id": primitive.primitive_id, "support_vertices": 0})
            continue
        support_weight = np.asarray(primitive.target_weight, dtype=np.float32)
        support_weight = np.clip(support_weight, 0.0, None) * np.asarray(
            ctx["smask"], dtype=np.float32
        )
        top = evaluate_top_fit(primitive.diagnostics["fit"], rows, cols)
        core = support_weight >= float(plateau_cfg.get("support_core_threshold", 0.65))
        fine_keep = np.where(core, fine_keep_core, 1.0).astype(np.float32)
        background = h0 - (1.0 - fine_keep) * fine
        normalized = np.clip(
            support_weight / max(float(plateau_cfg.get("weight", 1.0)), 1e-6),
            0.0,
            1.0,
        )
        # The longitudinal support probability controls the footprint edge.
        # Scarp samples raise the local top authority but never create support
        # outside the generated footprint.
        transition = np.clip(
            normalized
            + scarp_authority_gain * scarp_confidence * (1.0 - normalized),
            0.0,
            1.0,
        )
        candidate = transition * top + (1.0 - transition) * background
        generated = np.asarray(ctx["smask"], dtype=bool) & ~np.asarray(
            ctx["owner_mask"], dtype=bool
        )
        weight = support_weight * generated
        weighted_height += weight * candidate
        total_weight += weight
        candidate_support |= weight > 0.0
        primitive.target_height = candidate.astype(np.float32)
        primitive_reports.append({
            "primitive_id": primitive.primitive_id,
            "support_vertices": int(np.count_nonzero(weight > 0.0)),
            "core_vertices": int(core.sum()),
            "candidate_min_gu": float(candidate[weight > 0.0].min(initial=0.0)),
            "candidate_max_gu": float(candidate[weight > 0.0].max(initial=0.0)),
            "top_fit": primitive.diagnostics["fit"],
        })
    plateau = h0.astype(np.float32, copy=True)
    has_candidate = total_weight > 1e-6
    plateau[has_candidate] = weighted_height[has_candidate] / total_weight[has_candidate]
    plateau[np.asarray(ctx["owner_mask"], dtype=bool)] = h0[
        np.asarray(ctx["owner_mask"], dtype=bool)
    ]
    return plateau, {
        "candidate_weight": total_weight,
        "candidate_support": candidate_support,
        "fine_residual": fine.astype(np.float32),
    }, {
        "primitive_count": len(primitives),
        "candidate_vertices": int(has_candidate.sum()),
        "candidate_weight_max": float(total_weight.max(initial=0.0)),
        "fine_keep_core": fine_keep_core,
        "primitives": primitive_reports,
    }


def _semantic_canyon_path(
    generated: np.ndarray,
    h64: np.ndarray,
    plateau_support: np.ndarray,
    seed: tuple[int, int],
    direction: tuple[float, float],
    factor: int,
    max_cells: float,
    config: dict,
) -> tuple[list[tuple[int, int]], dict]:
    """Route one major valley over the semantic generated grid."""
    sem_generated = _block_max(generated, factor)
    sem_h64 = _block_mean(h64, np.isfinite(h64), factor)
    sem_support = _block_mean(
        plateau_support, np.isfinite(plateau_support), factor
    )
    sh, sw = sem_generated.shape
    start = (
        min(sh - 1, max(0, int(seed[0] // factor))),
        min(sw - 1, max(0, int(seed[1] // factor))),
    )
    if not sem_generated[start]:
        return [], {"seed": list(seed), "visited": 0, "endpoint": None}
    finite = sem_h64[np.isfinite(sem_h64) & sem_generated]
    if finite.size == 0:
        return [], {"seed": list(seed), "visited": 0, "endpoint": None}
    low_ref = float(np.percentile(finite, 25.0))
    high_ref = float(np.percentile(finite, 75.0))
    height_scale = max(high_ref - low_ref, 256.0)
    max_cost = max(
        float(max_cells) * 64.0 / max(float(factor), 1.0)
        * float(config.get("path_cost_multiplier", 2.0)),
        1.0,
    )
    dy_pref, dx_pref = direction
    norm = max(float(np.hypot(dy_pref, dx_pref)), 1e-6)
    dy_pref /= norm
    dx_pref /= norm
    dist = np.full((sh, sw), np.inf, dtype=np.float64)
    previous = np.full((sh, sw, 2), -1, dtype=np.int32)
    dist[start] = 0.0
    heap = [(0.0, start[0], start[1])]
    while heap:
        cost, row, col = heapq.heappop(heap)
        if cost != dist[row, col] or cost > max_cost:
            continue
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = row + dy, col + dx
            if not (0 <= nr < sh and 0 <= nc < sw) or not sem_generated[nr, nc]:
                continue
            alignment = max(0.0, dy * dy_pref + dx * dx_pref)
            direction_cost = float(config.get("direction_penalty", 1.0)) * (1.0 - alignment)
            current_h = float(sem_h64[row, col]) if np.isfinite(sem_h64[row, col]) else low_ref
            next_h = float(sem_h64[nr, nc]) if np.isfinite(sem_h64[nr, nc]) else current_h
            low_cost = float(config.get("target_low_penalty", 1.5)) * np.clip(
                (next_h - low_ref) / height_scale, 0.0, 2.0
            )
            uphill_cost = float(config.get("uphill_penalty", 4.0)) * max(
                0.0, next_h - current_h
            ) / height_scale
            support_cost = float(config.get("plateau_penalty", 1.0)) * max(
                0.0, 1.0 - float(sem_support[nr, nc])
            )
            step_cost = 1.0 + direction_cost + low_cost + uphill_cost + support_cost
            new_cost = cost + step_cost
            if new_cost < dist[nr, nc] and new_cost <= max_cost:
                dist[nr, nc] = new_cost
                previous[nr, nc] = (row, col)
                heapq.heappush(heap, (new_cost, nr, nc))
    visited = np.argwhere(np.isfinite(dist))
    if visited.size == 0:
        return [], {"seed": list(seed), "visited": 0, "endpoint": None}
    distance_from_start = np.hypot(
        visited[:, 0] - start[0], visited[:, 1] - start[1]
    )
    low_bias = np.clip(
        (sem_h64[visited[:, 0], visited[:, 1]] - low_ref) / height_scale,
        0.0,
        2.0,
    )
    ranking = distance_from_start - 0.25 * low_bias
    goal = tuple(int(v) for v in visited[int(np.argmax(ranking))])
    path: list[tuple[int, int]] = []
    current = goal
    while current != start and len(path) <= int(max_cells * 64.0 / max(factor, 1)) + 2:
        path.append(current)
        previous_point = tuple(int(v) for v in previous[current])
        if previous_point[0] < 0:
            break
        current = previous_point
    path.append(start)
    path.reverse()
    return path, {
        "seed": list(seed),
        "visited": int(visited.shape[0]),
        "endpoint": list(goal),
        "semantic_path_vertices": int(len(path)),
        "max_cost": max_cost,
        "endpoint_height_gu": float(sem_h64[goal]),
    }


def continue_canyons(
    h0: np.ndarray,
    ctx: dict,
    features: dict,
    primitives: list[TerrainPrimitive],
    plateau_field: np.ndarray,
    footprint_arrays: dict[str, np.ndarray],
    config: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Continue strong seam-crossing owner valleys and subtract their relief."""
    shape = h0.shape
    canyon_cfg = config.get("canyon", {})
    owner = np.asarray(ctx["owner_mask"], dtype=bool)
    generated = np.asarray(ctx["smask"], dtype=bool) & ~owner
    h64 = _normalized_gaussian(
        h0, np.asarray(ctx["smask"], dtype=bool) & np.isfinite(h0), 64.0
    )
    valley_mask = np.asarray(features["valley_mask"], dtype=bool) & owner
    valley_score = np.asarray(features["valley_score"], dtype=np.float32)
    valley_labels, _ = ndimage.label(
        valley_mask, structure=np.ones((3, 3), dtype=bool)
    )
    normals = _seam_normals(ctx, shape)
    factor = max(1, int(config.get("semantic_downsample", 4)))
    canyon_depth = np.zeros(shape, dtype=np.float32)
    canyon_weight = np.zeros(shape, dtype=np.float32)
    canyon_line = np.zeros(shape, dtype=bool)
    reports = []

    for primitive in primitives:
        seed_by_component: dict[int, tuple[float, tuple[int, int], tuple[int, int]]] = {}
        radius = max(1, int(round(float(canyon_cfg.get("seed_search_radius_verts", 8.0)))))
        for row, col in primitive.seam_vertices:
            row, col = int(row), int(col)
            normal = normals.get(row * shape[1] + col)
            if normal is None:
                continue
            oy, ox = row - normal[0], col - normal[1]
            r0, r1 = max(0, oy - radius), min(shape[0], oy + radius + 1)
            c0, c1 = max(0, ox - radius), min(shape[1], ox + radius + 1)
            local = valley_mask[r0:r1, c0:c1]
            if not local.any():
                continue
            score = np.where(local, valley_score[r0:r1, c0:c1], -np.inf)
            vy, vx = np.unravel_index(int(np.argmax(score)), score.shape)
            component_id = int(valley_labels[r0 + vy, c0 + vx])
            if component_id <= 0:
                continue
            seed = (row, col)
            candidate_score = float(score[vy, vx])
            if candidate_score < float(canyon_cfg.get("min_confidence", 0.55)):
                continue
            prior = seed_by_component.get(component_id)
            if prior is None or candidate_score > prior[0]:
                seed_by_component[component_id] = (candidate_score, seed, normal)
        selected = sorted(seed_by_component.values(), key=lambda item: item[0], reverse=True)
        selected = selected[: int(canyon_cfg.get("max_paths_per_plateau", 3))]
        seeds = [item[1] for item in selected]
        seed_normals = [item[2] for item in selected]
        component_ids = [
            component_id for component_id, item in seed_by_component.items()
            if item in selected
        ]
        primitive_paths = []
        for seed, normal in zip(seeds, seed_normals):
            work_bbox = footprint_arrays["work_bbox"].tolist()
            wr, wc, _ = _slices_bbox(tuple(work_bbox), shape)
            work_generated = generated[wr, wc]
            work_h64 = h64[wr, wc]
            work_support = footprint_arrays["support_probability"][wr, wc]
            generated_seed = _first_generated_vertex(
                seed[0], seed[1], normal, generated,
                int(round(float(config.get("seam_seed_max_steps", 2.0)))),
            )
            if generated_seed is None:
                continue
            route_seed = (generated_seed[0], generated_seed[1])
            local_seed = (route_seed[0] - work_bbox[0], route_seed[1] - work_bbox[2])
            if not (0 <= local_seed[0] < work_generated.shape[0]
                    and 0 <= local_seed[1] < work_generated.shape[1]):
                continue
            path, path_report = _semantic_canyon_path(
                work_generated,
                work_h64,
                work_support,
                local_seed,
                (float(normal[0]), float(normal[1])),
                factor,
                float(canyon_cfg.get("max_continuation_cells", 8.0)),
                canyon_cfg,
            )
            if not path:
                continue
            full_path = [
                (int(row * factor + work_bbox[0]), int(col * factor + work_bbox[2]))
                for row, col in path
            ]
            line = np.zeros(shape, dtype=bool)
            path_heights = []
            for row, col in full_path:
                if 0 <= row < shape[0] and 0 <= col < shape[1] and generated[row, col]:
                    line[row, col] = True
                    path_heights.append(float(h64[row, col]))
            if not line.any():
                continue
            if len(path_heights) < 2:
                continue
            owner_row = seed[0] - normal[0]
            owner_col = seed[1] - normal[1]
            if 0 <= owner_row < shape[0] and 0 <= owner_col < shape[1]:
                owner_h = float(features["H24"][owner_row, owner_col])
            else:
                owner_h = float("nan")
            z0 = owner_h if np.isfinite(owner_h) else path_heights[0]
            zend = path_heights[-1]
            thalweg_values = np.linspace(z0, zend, len(path_heights), dtype=np.float32)
            if zend <= z0:
                thalweg_values = np.minimum.accumulate(thalweg_values)
            else:
                thalweg_values = np.maximum.accumulate(thalweg_values)
            line_height = np.zeros(shape, dtype=np.float32)
            valid_path = [
                (row, col) for row, col in full_path
                if 0 <= row < shape[0] and 0 <= col < shape[1] and generated[row, col]
            ]
            for (row, col), value in zip(valid_path, thalweg_values):
                line_height[row, col] = float(value)
            local_line = line[wr, wc]
            distance, nearest = ndimage.distance_transform_edt(
                ~local_line, return_indices=True
            )
            nearest_height = line_height[wr, wc][tuple(nearest)]
            local_rows, local_cols = np.indices(local_line.shape)
            global_rows = local_rows + work_bbox[0]
            global_cols = local_cols + work_bbox[2]
            top = evaluate_top_fit(primitive.diagnostics["fit"], global_rows, global_cols)
            half_width = max(
                float(canyon_cfg.get("width_cells", 4.0)) * 64.0, 1.0
            )
            bottom_half = min(
                float(canyon_cfg.get("bottom_half_width_cells", 1.0)) * 64.0,
                half_width,
            )
            normalized = distance / half_width
            q = np.zeros(distance.shape, dtype=np.float32)
            inside = normalized < 1.0
            bottom = normalized <= bottom_half / half_width
            q[bottom] = 1.0
            wall_t = (normalized - bottom_half / half_width) / max(
                1.0 - bottom_half / half_width, 1e-6
            )
            q[inside & ~bottom] = (
                1.0 - _smootherstep(wall_t[inside & ~bottom])
            ) ** float(canyon_cfg.get("wall_exponent", 1.5))
            depth = np.maximum(top - nearest_height, 0.0) * q
            depth[~work_generated] = 0.0
            canyon_depth[wr, wc] = np.maximum(canyon_depth[wr, wc], depth)
            canyon_weight[wr, wc] = np.maximum(
                canyon_weight[wr, wc],
                q * float(canyon_cfg.get("weight", 1.2)),
            )
            canyon_line |= line
            primitive_paths.append({
                **path_report,
                "full_path_vertices": len(full_path),
                "depth_max_gu": float(depth.max(initial=0.0)),
                "width_gu": half_width,
            })
        reports.append({
            "primitive_id": primitive.primitive_id,
            "owner_valley_components": len(component_ids),
            "owner_valley_vertices": int(sum(
                np.count_nonzero(valley_labels == component_id)
                for component_id in component_ids
            )),
            "seed_count": len(seeds),
            "paths": primitive_paths,
        })

    canyon_field = plateau_field.astype(np.float32, copy=True)
    canyon_field -= canyon_depth
    canyon_field[owner] = h0[owner]
    return canyon_field, {
        "canyon_depth": canyon_depth,
        "canyon_weight": canyon_weight,
        "canyon_line": canyon_line,
    }, {
        "primitive_count": len(primitives),
        "canyon_line_vertices": int(canyon_line.sum()),
        "canyon_depth_max_gu": float(canyon_depth.max(initial=0.0)),
        "primitives": reports,
    }


def _edge_conductance(
    scarp_confidence: np.ndarray,
    scarp_normal_y: np.ndarray,
    scarp_normal_x: np.ndarray,
    dy: int,
    dx: int,
    minimum: float,
    beta: float,
) -> np.ndarray:
    shifted_confidence = np.roll(scarp_confidence, (-dy, -dx), axis=(0, 1))
    shifted_ny = np.roll(scarp_normal_y, (-dy, -dx), axis=(0, 1))
    shifted_nx = np.roll(scarp_normal_x, (-dy, -dx), axis=(0, 1))
    confidence = 0.5 * (scarp_confidence + shifted_confidence)
    normal_y = 0.5 * (scarp_normal_y + shifted_ny)
    normal_x = 0.5 * (scarp_normal_x + shifted_nx)
    norm = np.hypot(normal_y, normal_x)
    alignment = np.divide(
        np.abs(dy * normal_y + dx * normal_x),
        norm,
        out=np.zeros_like(norm, dtype=np.float32),
        where=norm > 1e-6,
    )
    return float(minimum) + (1.0 - float(minimum)) * np.exp(
        -float(beta) * confidence * alignment
    )


def _solve_edge_aware_structure(
    h0: np.ndarray,
    active: np.ndarray,
    fixed: np.ndarray,
    fixed_values: np.ndarray,
    candidate: np.ndarray,
    candidate_weight: np.ndarray,
    scarp_confidence: np.ndarray,
    scarp_normal_y: np.ndarray,
    scarp_normal_x: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, dict]:
    """Solve one symmetric screened-Poisson system with scarp conductance."""
    active = np.asarray(active, dtype=bool)
    fixed = np.asarray(fixed, dtype=bool)
    if np.any(fixed & ~active):
        raise ValueError("primitive fixed vertices must lie inside active domain")
    if np.any(active & ~np.isfinite(h0)):
        raise ValueError("primitive reconciliation received non-finite background")
    unknown = active & ~fixed
    n = int(unknown.sum())
    if n == 0:
        out = h0.astype(np.float32, copy=True)
        out[fixed] = fixed_values[fixed]
        return out, {
            "unknowns": 0,
            "linear_solver": str(config.get("linear_solver", "amg_rs_cg")),
            "cg_iterations": 0,
            "correction_min": 0.0,
            "correction_max": 0.0,
        }
    active_interior = ndimage.binary_erosion(
        active, structure=ndimage.generate_binary_structure(2, 1), border_value=0
    )
    if np.any(unknown & ~active_interior):
        raise ValueError("primitive unknown touches inactive terrain")
    idx = np.full(active.shape, -1, dtype=np.int64)
    idx[unknown] = np.arange(n, dtype=np.int64)
    background_weight = float(config.get("background_weight", 0.12))
    weights = np.clip(np.nan_to_num(candidate_weight), 0.0, None).astype(np.float64)
    generated = active & ~fixed
    data_weight = weights + background_weight * generated
    guide_rhs = weights * np.nan_to_num(candidate) + background_weight * np.nan_to_num(h0)
    correction_fixed = np.zeros(h0.shape, dtype=np.float64)
    correction_fixed[fixed] = fixed_values[fixed].astype(np.float64) - h0[fixed]
    rows, cols, values = [], [], []
    rhs = guide_rhs[unknown].astype(np.float64)
    degree = np.zeros(n, dtype=np.float64)
    gmin = float(config.get("conductance_min", 0.1))
    beta = float(config.get("conductance_beta", 3.0))
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
        neighbor_fixed_value = np.roll(fixed_values, (-dy, -dx), axis=(0, 1))
        conductance = _edge_conductance(
            scarp_confidence, scarp_normal_y, scarp_normal_x,
            dy, dx, gmin, beta,
        )
        valid = center & neighbor_active
        equation_ids = idx[valid]
        edge_values = conductance[valid].astype(np.float64)
        np.add.at(degree, equation_ids, edge_values)
        neighbor_ids = neighbor_idx[valid]
        unknown_neighbor = neighbor_ids >= 0
        rows.append(equation_ids[unknown_neighbor])
        cols.append(neighbor_ids[unknown_neighbor])
        values.append(-edge_values[unknown_neighbor])
        fixed_neighbor = neighbor_fixed[valid]
        if np.any(fixed_neighbor):
            np.add.at(
                rhs,
                equation_ids[fixed_neighbor],
                edge_values[fixed_neighbor]
                * neighbor_fixed_value[valid][fixed_neighbor],
            )
    if np.any(degree <= 0.0):
        raise ValueError("primitive reconciliation has isolated unknowns")
    rows.append(np.arange(n, dtype=np.int64))
    cols.append(np.arange(n, dtype=np.int64))
    values.append(degree + data_weight[unknown])
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    solver = str(config.get("linear_solver", "amg_rs_cg"))
    setup_t0 = __import__("time").perf_counter()
    if solver == "amg_rs_cg":
        if pyamg is None:
            raise RuntimeError("primitive reconciliation requires pyamg")
        ml = pyamg.ruge_stuben_solver(
            matrix, max_coarse=int(config.get("amg_max_coarse", 500))
        )
        preconditioner = ml.aspreconditioner(cycle="V")
    elif solver == "jacobi_cg":
        diagonal = matrix.diagonal()
        if np.any(diagonal <= 0.0):
            raise ValueError("primitive Jacobi diagonal is non-positive")
        ml = None
        preconditioner = sparse.diags(1.0 / diagonal)
    else:
        raise ValueError(f"unsupported primitive solver {solver!r}")
    setup_s = __import__("time").perf_counter() - setup_t0
    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1

    solve_t0 = __import__("time").perf_counter()
    solution, status = cg(
        matrix,
        rhs,
        M=preconditioner,
        rtol=float(config.get("cg_tol", 1e-6)),
        atol=0.0,
        maxiter=int(config.get("cg_maxiter", 200)),
        callback=callback,
    )
    solve_s = __import__("time").perf_counter() - solve_t0
    if status != 0:
        raise RuntimeError(f"primitive reconciliation CG failed with status {status}")
    out = h0.astype(np.float64, copy=True)
    out[unknown] = solution
    out[fixed] = fixed_values[fixed]
    residual = matrix @ solution - rhs
    report = {
        "unknowns": n,
        "guide_rows": int(np.count_nonzero(weights[unknown] > 0.0)),
        "background_weight": background_weight,
        "linear_solver": solver,
        "cg_iterations": int(iterations),
        "solver_setup_s": round(setup_s, 4),
        "solver_solve_s": round(solve_s, 4),
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "correction_min": float((out - h0)[active].min(initial=0.0)),
        "correction_max": float((out - h0)[active].max(initial=0.0)),
        "conductance_min": gmin,
        "conductance_beta": beta,
    }
    if ml is not None:
        report["amg_levels"] = int(len(ml.levels))
    return out.astype(np.float32), report


def reconcile_primitive_candidates(
    h0: np.ndarray,
    ctx: dict,
    plateau_arrays: dict[str, np.ndarray],
    canyon_field: np.ndarray,
    canyon_arrays: dict[str, np.ndarray],
    config: dict,
) -> tuple[np.ndarray, dict]:
    """Reconcile all candidates once while preserving exact boundaries."""
    active = np.asarray(ctx["smask"], dtype=bool)
    owner = np.asarray(ctx["owner_mask"], dtype=bool) & active
    fixed = np.asarray(ctx["hard"], dtype=bool) | owner
    fixed_values = np.asarray(ctx["hard_vals"], dtype=np.float32).copy()
    fixed_values[owner] = h0[owner]
    candidate_weight = (
        np.asarray(plateau_arrays["candidate_weight"], dtype=np.float32)
        + np.asarray(canyon_arrays["canyon_weight"], dtype=np.float32)
    )
    scarp = np.asarray(plateau_arrays["scarp_confidence"], dtype=np.float32)
    result, report = _solve_edge_aware_structure(
        h0,
        active,
        fixed,
        fixed_values,
        canyon_field,
        candidate_weight,
        scarp,
        np.asarray(plateau_arrays["scarp_normal_y"], dtype=np.float32),
        np.asarray(plateau_arrays["scarp_normal_x"], dtype=np.float32),
        config,
    )
    report.update({
        "active_vertices": int(active.sum()),
        "fixed_vertices": int(fixed.sum()),
        "owner_vertices": int(owner.sum()),
        "candidate_vertices": int(np.count_nonzero(candidate_weight > 0.0)),
    })
    return result, report
