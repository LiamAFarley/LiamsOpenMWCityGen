"""Component-local harmonic synthesis for cells without authoritative VHGT.

Purpose
    Replace the old global nearest-edge fill with a bounded absolute-height
    solve for only the explicit ``synth_height_cells`` needed by a review
    region. Base VHGT and winning-owner VHGT form the fixed boundary; the
    synthesized cells are the unknown interior.

Inputs
    Raw base/owner vertex fields, ``cell_owner`` and ``cell_height_source``
    grids, and a required set of world cells.

Outputs
    A composite working field plus diagnostics. Source corpus arrays are not
    mutated and no terrain is synthesized outside the requested missing
    components.

Invariants
    Uses the same direct Laplace + AMG correction operator as the seam solve;
    no nearest-neighbor copy, clamping, or normal-equation solve is allowed.
    A component without a finite 4-neighbor boundary raises immediately.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from scipy import ndimage


def _component_cells(
    missing: set[tuple[int, int]],
    seeds: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    """Return missing-cell components that intersect the required seeds."""
    out: list[set[tuple[int, int]]] = []
    unseen = set(missing)
    for seed in sorted(seeds):
        if seed not in unseen:
            continue
        component: set[tuple[int, int]] = set()
        q = deque([seed])
        unseen.remove(seed)
        while q:
            cell = q.popleft()
            component.add(cell)
            x, y = cell
            for nb in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nb in unseen:
                    unseen.remove(nb)
                    q.append(nb)
        out.append(component)
    return out


def _cell_vertex_mask(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    gy0: int,
    gx0: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for cx, cy in cells:
        r0 = (cy - gy0) * 64
        c0 = (cx - gx0) * 64
        r1 = min(shape[0], r0 + 65)
        c1 = min(shape[1], c0 + 65)
        if r0 < shape[0] and c0 < shape[1] and r1 > 0 and c1 > 0:
            mask[max(0, r0):r1, max(0, c0):c1] = True
    return mask


def compose_authoritative_field(
    base_field: np.ndarray,
    owner_field: np.ndarray,
    cell_height_source: np.ndarray,
    base_code: int,
) -> np.ndarray:
    """Compose raw base plus only the cell-authoritative owner heights."""
    out = np.array(base_field, dtype=np.float32, copy=True)
    owner_cells = (cell_height_source != 0) & (cell_height_source != base_code)
    owner_v = np.zeros(out.shape, dtype=bool)
    owner_quads = np.repeat(np.repeat(owner_cells, 64, axis=0), 64, axis=1)
    owner_v[:-1, :-1] |= owner_quads
    owner_v[1:, :-1] |= owner_quads
    owner_v[:-1, 1:] |= owner_quads
    owner_v[1:, 1:] |= owner_quads
    take = owner_v & np.isfinite(owner_field)
    out[take] = owner_field[take]
    return out


def synthesize_missing_heights(
    base_field: np.ndarray,
    owner_field: np.ndarray,
    cell_owner: np.ndarray,
    cell_height_source: np.ndarray,
    required_cells: set[tuple[int, int]],
    gy0: int,
    gx0: int,
    base_code: int,
    *,
    linear_solver: str = "amg_rs_cg",
    cg_tol: float = 1e-6,
    cg_maxiter: int = 200,
    amg_max_coarse: int = 500,
) -> tuple[np.ndarray, dict]:
    """Synthesize required missing-height components from finite boundaries."""
    t0 = time.perf_counter()
    if base_field.shape != owner_field.shape:
        raise ValueError("base and owner fields must have equal shapes")
    # A review hole can be a true void cell (cell_owner == 0), not only an
    # owner stub. Detect incomplete 65x65 cells in the requested frame while
    # leaving complete valid cells entirely untouched.
    working = compose_authoritative_field(
        base_field, owner_field, cell_height_source, base_code
    )
    candidate_cells: set[tuple[int, int]] = set()
    for cx, cy in required_cells:
        iy, ix = cy - gy0, cx - gx0
        if not (0 <= iy < cell_owner.shape[0] and 0 <= ix < cell_owner.shape[1]):
            continue
        r0, c0 = iy * 64, ix * 64
        cell = working[r0:min(working.shape[0], r0 + 65),
                       c0:min(working.shape[1], c0 + 65)]
        if cell.shape == (65, 65) and not np.isfinite(cell).all():
            candidate_cells.add((cx, cy))
    seeds = candidate_cells
    missing_cells = set(candidate_cells)
    components = _component_cells(missing_cells, seeds)
    report = {
        "required_missing_cells": int(len(seeds)),
        "candidate_cells": [list(cell) for cell in sorted(candidate_cells)],
        "synthesized_cells": 0,
        "components": [],
        "linear_solver": linear_solver,
    }
    if not components:
        report["synthesis_s"] = round(time.perf_counter() - t0, 4)
        return working, report

    from procgen.terrain_blend import solve_harmonic_correction

    for component_index, component in enumerate(components):
        xs = [cell[0] for cell in component]
        ys = [cell[1] for cell in component]
        bx0, bx1 = min(xs) - 1, max(xs) + 1
        by0, by1 = min(ys) - 1, max(ys) + 1
        r0 = max(0, (by0 - gy0) * 64)
        c0 = max(0, (bx0 - gx0) * 64)
        r1 = min(working.shape[0], (by1 - gy0) * 64 + 65)
        c1 = min(working.shape[1], (bx1 - gx0) * 64 + 65)
        if r1 <= r0 or c1 <= c0:
            raise ValueError(
                f"missing component {component_index} has an empty raster window"
            )

        local = working[r0:r1, c0:c1]
        missing_vertices = _cell_vertex_mask(
            component, local.shape, gy0 + r0 // 64, gx0 + c0 // 64
        )
        unknown = missing_vertices & ~np.isfinite(local)
        labels, count = ndimage.label(
            unknown, structure=ndimage.generate_binary_structure(2, 1)
        )
        component_report = {
            "component": component_index,
            "cells": len(component),
            "unknown_vertices": int(unknown.sum()),
            "subcomponents": int(count),
        }
        if not unknown.any():
            report["components"].append(component_report)
            report["synthesized_cells"] += len(component)
            continue

        for sub_id in range(1, count + 1):
            sub_unknown = labels == sub_id
            boundary = (
                ndimage.binary_dilation(
                    sub_unknown,
                    structure=ndimage.generate_binary_structure(2, 1),
                )
                & ~sub_unknown
                & np.isfinite(local)
            )
            if not boundary.any():
                raise ValueError(
                    f"missing component {component_index} has no authoritative "
                    f"boundary (subcomponent {sub_id})"
                )
            active = sub_unknown | boundary
            target = np.zeros(local.shape, dtype=np.float64)
            fixed_final = np.zeros(local.shape, dtype=np.float64)
            fixed_final[boundary] = local[boundary]
            solved, solve_report = solve_harmonic_correction(
                target,
                active,
                boundary,
                fixed_final,
                linear_solver=linear_solver,
                cg_tol=cg_tol,
                cg_maxiter=cg_maxiter,
                amg_max_coarse=amg_max_coarse,
            )
            if not np.all(np.isfinite(solved[sub_unknown])):
                raise FloatingPointError(
                    f"missing component {component_index} produced non-finite "
                    f"heights (subcomponent {sub_id})"
                )
            local[sub_unknown] = solved[sub_unknown]
            component_report.setdefault("solves", []).append({
                "subcomponent": sub_id,
                "unknowns": solve_report["unknowns"],
                "cg_iterations": solve_report["cg_iterations"],
                "setup_s": solve_report["solver_setup_s"],
                "solve_s": solve_report["solver_solve_s"],
            })
        working[r0:r1, c0:c1] = local
        report["components"].append(component_report)
        report["synthesized_cells"] += len(component)

    report["synthesis_s"] = round(time.perf_counter() - t0, 4)
    return working, report
