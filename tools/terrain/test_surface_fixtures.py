"""Production-topology fixtures for the v3 harmonic seam correction.

Purpose
    Exercise the actual Morrowind spatial representation: a 5x5
    ``cell_owner`` grid, 64 intervals per LAND cell, production
    ``seam_edges()``, and production ``rasterize_seam()``. Boundary shapes are
    independent from the synthetic height fields. Corner and staircase cases
    therefore test ownership topology, not invented terrain styles.

Fixtures
    ``straight_flat``       flat owner/target height fields
    ``straight_sloped``     planar owner terrain at a straight boundary
    ``straight_mountain``   broad owner ridge at a straight boundary
    ``corner_flat``         flat fields with an L-shaped ownership boundary
    ``staircase_flat``      flat fields with a cell-topology staircase

Acceptance
    Seam C0 is exact, no NaNs occur, the direct harmonic system has no empty
    rows, the correction obeys the fixed-value maximum principle, and the
    flat straight profile is monotone from owner height to target height.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen.terrain_blend import rasterize_seam, solve_surface  # noqa: E402
from procgen.terrainfield import (  # noqa: E402
    hillshade,
    hypsometric_rgb,
    seam_edges,
    save_shade_png,
)

CELL_SIDE = 5
VERTS_PER_CELL = 64
VERT_SIDE = CELL_SIDE * VERTS_PER_CELL + 1
BASE_CODE = 1
OWNER_CODE = 2
TARGET_HEIGHT = 200.0
OWNER_HEIGHT = 10000.0
BLEND_VERTS = VERTS_PER_CELL


def _ownership(kind: str) -> np.ndarray:
    """Return only the ownership topology; heights are built separately."""
    t, o = BASE_CODE, OWNER_CODE
    if kind.startswith("straight"):
        rows = [[t] * 5, [t] * 5, [t] * 5, [o] * 5, [o] * 5]
    elif kind == "corner_flat":
        rows = [[t] * 5, [t] * 5, [o, t, t, t, t], [o] * 5, [o] * 5]
    elif kind == "staircase_flat":
        rows = [[t, t, t, t, o], [t, t, t, o, o],
                [t, t, o, o, o], [t, o, o, o, o], [o] * 5]
    else:
        raise ValueError(f"unknown fixture {kind}")
    return np.asarray(rows, dtype=np.uint8)


def _cell_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((VERT_SIDE, VERT_SIDE), dtype=bool)
    for cx, cy in cells:
        r0, c0 = cy * VERTS_PER_CELL, cx * VERTS_PER_CELL
        mask[r0:r0 + VERTS_PER_CELL + 1,
             c0:c0 + VERTS_PER_CELL + 1] = True
    return mask


def _owner_heights(kind: str) -> np.ndarray:
    yy, xx = np.mgrid[0:VERT_SIDE, 0:VERT_SIDE].astype(np.float32)
    if kind == "straight_sloped":
        # Owner is below the straight seam at y=3*64. Moving outward into
        # the owner increases height by 40 GU per raster interval.
        return OWNER_HEIGHT + (yy - 3.0 * VERTS_PER_CELL) * 40.0
    if kind == "straight_mountain":
        ridge = 6000.0 * np.exp(
            -(((yy - 2.5 * VERTS_PER_CELL) ** 2) / (150.0 ** 2)
              + ((xx - 2.0 * VERTS_PER_CELL) ** 2) / (130.0 ** 2))
        )
        return 5000.0 + ridge
    return np.full((VERT_SIDE, VERT_SIDE), OWNER_HEIGHT, dtype=np.float32)


def _base_ctx(kind: str) -> dict:
    cell_owner = _ownership(kind)
    owner_cells = {
        (x, y)
        for y in range(CELL_SIDE)
        for x in range(CELL_SIDE)
        if int(cell_owner[y, x]) == OWNER_CODE
    }
    tam_cells = {
        (x, y)
        for y in range(CELL_SIDE)
        for x in range(CELL_SIDE)
        if int(cell_owner[y, x]) == BASE_CODE
    }
    edges = seam_edges(cell_owner, BASE_CODE, 0, 0)
    seam_v, edge_list = rasterize_seam(
        edges, tam_cells, (VERT_SIDE, VERT_SIDE), 0, 0, 0, 0
    )

    target = np.full((VERT_SIDE, VERT_SIDE), TARGET_HEIGHT, dtype=np.float32)
    owner_field = _owner_heights(kind).astype(np.float32)
    own_view = np.full_like(target, TARGET_HEIGHT)
    owner_v = _cell_mask(owner_cells)
    own_view[owner_v] = owner_field[owner_v]
    tam_v = _cell_mask(tam_cells)

    # The active domain is the first real LAND-cell-width corridor on the
    # Tamriel side. No hand-built edge list or artificial scanline geometry is
    # used here; the boundary comes entirely from cell ownership.
    dist_seam = ndimage.distance_transform_edt(~seam_v).astype(np.float32)
    smask = tam_v & (dist_seam <= float(BLEND_VERTS))
    smask |= seam_v
    ring_v = smask & (dist_seam >= float(BLEND_VERTS) - 1.0)
    hard = seam_v | ring_v
    hard_vals = np.zeros_like(target)
    hard_vals[ring_v] = target[ring_v]
    hard_vals[seam_v] = own_view[seam_v]

    ny = np.zeros(smask.shape, np.float32)
    nx = np.zeros(smask.shape, np.float32)
    for edge in edge_list:
        ny_v, nx_v = edge["normal"]
        for flat in edge["verts"]:
            ny.ravel()[flat] = ny_v
            nx.ravel()[flat] = nx_v

    return {
        "cell_owner": cell_owner,
        "base_code": BASE_CODE,
        "tam_h": target.copy(),
        "target": target,
        "own_view": own_view,
        "owner_field": owner_field,
        "owner_v": owner_v,
        "tam_v": tam_v,
        "smask": smask,
        "seam_v": seam_v,
        "ring_v": ring_v,
        "hard": hard,
        "hard_vals": hard_vals,
        "dist_seam": dist_seam,
        "width_cells": np.ones_like(target),
        "edge_list": edge_list,
        "edges": edges,
        "solve_cells": tam_cells,
        "nx": nx,
        "ny": ny,
    }


def _normal_step(ctx: dict, field: np.ndarray) -> float:
    best = 0.0
    H, W = field.shape
    for edge in ctx["edge_list"]:
        dy, dx = (int(round(edge["normal"][0])),
                  int(round(edge["normal"][1])))
        for flat in edge["verts"]:
            sy, sx = divmod(flat, W)
            previous = float(field[sy, sx])
            for k in range(1, VERTS_PER_CELL + 1):
                y, x = sy + dy * k, sx + dx * k
                if not (0 <= y < H and 0 <= x < W) or not ctx["smask"][y, x]:
                    break
                current = float(field[y, x])
                best = max(best, abs(current - previous))
                previous = current
    return best


def _check(kind: str, ctx: dict, field: np.ndarray, report: dict) -> list[str]:
    fails = []
    seam = ctx["seam_v"]
    c0 = float(np.abs(field[seam] - ctx["own_view"][seam]).max())
    if c0 > 1e-3:
        fails.append(f"C0 {c0:.3g} != 0")
    if np.isnan(field[ctx["smask"]]).any():
        fails.append("NaN inside active corridor")
    bounds = report["correction_bounds"]
    correction = field.astype(np.float64) - ctx["target"].astype(np.float64)
    cmin = float(np.min(correction[ctx["smask"]]))
    cmax = float(np.max(correction[ctx["smask"]]))
    if cmin < bounds["fixed_min"] - 1e-2:
        fails.append(f"correction undershoot {cmin:.2f} < {bounds['fixed_min']:.2f}")
    if cmax > bounds["fixed_max"] + 1e-2:
        fails.append(f"correction overshoot {cmax:.2f} > {bounds['fixed_max']:.2f}")
    if kind == "straight_flat":
        edge = ctx["edge_list"][len(ctx["edge_list"]) // 2]
        flat = edge["verts"][len(edge["verts"]) // 2]
        y, x = divmod(flat, field.shape[1])
        dy, dx = int(round(edge["normal"][0])), int(round(edge["normal"][1]))
        profile = np.asarray([
            field[y + dy * k, x + dx * k]
            for k in range(VERTS_PER_CELL + 1)
        ])
        if np.any(np.diff(profile) > 1e-3):
            fails.append("straight flat profile is not monotone")
        if float(profile.min()) < TARGET_HEIGHT - 1e-3:
            fails.append(f"flat profile enters water: {profile.min():.2f} GU")
        if float(profile.max()) > OWNER_HEIGHT + 1e-3:
            fails.append(f"flat profile spikes: {profile.max():.2f} GU")
    step = _normal_step(ctx, field)
    print(f"  {kind:<20} C0={c0:.2f} normal_step={step:.2f} GU "
          f"corr=[{cmin:.1f},{cmax:.1f}] "
          f"{'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    return fails


def _render(kind: str, ctx: dict, field: np.ndarray, render_dir: Path) -> None:
    display = field.copy()
    display[ctx["owner_v"]] = ctx["owner_field"][ctx["owner_v"]]
    cfg = {
        "azimuth_deg": 315.0,
        "altitude_deg": 45.0,
        "vertical_exaggeration": 1.0,
        "hypsometric_stops_gu": [
            [-500, 30, 40, 80], [0, 70, 110, 160], [60, 92, 140, 92],
            [3000, 190, 180, 120], [6000, 165, 125, 90],
            [9000, 150, 110, 80], [11000, 225, 225, 232],
            [14000, 255, 255, 255],
        ],
    }
    shade = hillshade(display, cfg["azimuth_deg"], cfg["altitude_deg"],
                      cfg["vertical_exaggeration"])
    rgb = hypsometric_rgb(display, shade, cfg["hypsometric_stops_gu"])
    save_shade_png(rgb, render_dir / f"fixture_{kind}.png", 2,
                   title=f"{kind}: owner territory + generated Tamriel transition")


def main() -> int:
    render_dir = (ROOT / "output" / "mapdata" / "terrain" /
                  "tamriel_reworked" / "solved" / "v3" / "fixture_renders")
    render_dir.mkdir(parents=True, exist_ok=True)
    all_fails = []
    for kind in ("straight_flat", "straight_sloped", "straight_mountain",
                 "corner_flat", "staircase_flat"):
        ctx = _base_ctx(kind)
        field, report = solve_surface(
            ctx,
            {"surface": {"smooth_weight": 1.0, "slope_weight": 25.0,
                         "cg_tol": 1e-6, "cg_maxiter": 800}},
        )
        if report["assembly"]["empty_equation_rows"] != 0:
            raise AssertionError(f"{kind}: empty harmonic equation rows")
        if report["assembly"]["rows"] != report["unknowns"]:
            raise AssertionError(f"{kind}: direct L rows != unknown count")
        if report["equation_counts"]["data"] != 0:
            raise AssertionError(f"{kind}: data family remains")
        all_fails.extend(_check(kind, ctx, field, report))
        _render(kind, ctx, field, render_dir)
    if all_fails:
        print(f"FAILURE: {len(all_fails)} fixture assertion(s) failed")
        return 1
    print("ALL FIXTURES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
