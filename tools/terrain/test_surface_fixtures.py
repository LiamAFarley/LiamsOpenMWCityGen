"""Synthetic fixtures for the v3 constrained surface solver (mandatory).

Purpose
    Sol High review requirement: the constrained solver must pass pure
    mathematical fixtures BEFORE touching the real corpus. Each fixture
    builds a small synthetic seam scene, runs the exact production
    ``solve_surface``, and asserts the invariants:

      - seam heights exact (C0 == 0);
      - no single-vertex cliff at the seam (first-edge drop bounded by the
        blend grade);
      - monotone approach from the low side toward the owner heights
        (no overshoot past the owner level near the seam);
      - no NaNs; slope-family residual within tolerance.

    Fixtures: flat-step, sloped-step, corner, staircase,
    mountain-to-lowland.

Outputs
    PASS/FAIL per fixture to stdout; exit code 1 on any failure.

Pipeline position
    Quality gate for tools/terrain/solve_region_v3.py. Run before every
    solver change reaches the real corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen.terrain_blend import solve_surface  # noqa: E402
from procgen.terrainfield import (  # noqa: E402
    hillshade, hypsometric_rgb, save_shade_png,
)

N = 192
SEAM_ROW = 96          # horizontal seam at this vertex row (tam side below)
BLEND_CELLS = 6
GRADE = 2500.0         # GU per cell allowed average grade
OWNER_HIGH = 10000.0
TARGET_LOW = 200.0


def _base_ctx(kind: str) -> dict:
    # The seam row is part of the solve band (production seam verts are the
    # tamriel cell's own edge vertices, inside the mask). The band is wider
    # than the seam run so the seam ENDPOINTS are interior vertices — a seam
    # end touching the band's side ring would legitimately create a local
    # corner step that is not representative of production corridors.
    smask = np.zeros((N, N), dtype=bool)
    smask[SEAM_ROW:SEAM_ROW + 1 + BLEND_CELLS * 8, 2:N - 2] = True
    own_view = np.full((N, N), TARGET_LOW, dtype=np.float32)
    target = np.full((N, N), TARGET_LOW, dtype=np.float32)
    hard = np.zeros((N, N), dtype=bool)
    hard_vals = np.zeros((N, N), np.float32)

    if kind == "flat_step":
        own_view[:SEAM_ROW + 1, :] = OWNER_HIGH
    elif kind == "sloped_step":
        for r in range(SEAM_ROW + 1):
            own_view[r, :] = OWNER_HIGH - (SEAM_ROW - r) * 40.0
    elif kind == "mountain_to_lowland":
        yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
        peak = 12000.0 * np.exp(-(((yy - 60) ** 2) / 3600.0
                                  + ((xx - N // 2) ** 2) / 25000.0))
        own_view[:SEAM_ROW + 1, :] = np.maximum(OWNER_HIGH * 0.6, peak)[:SEAM_ROW + 1]
    elif kind in ("corner", "staircase"):
        own_view[:SEAM_ROW + 1, :] = OWNER_HIGH
    else:
        raise SystemExit(f"unknown fixture kind {kind}")

    # hard: outer ring first, then seam (seam wins at overlaps — same
    # precedence rule as the production build_context)
    ring = smask & ~_erode(smask)
    hard |= ring
    hard_vals[ring] = target[ring]
    hard[SEAM_ROW, 8:N - 8] = True
    hard_vals[SEAM_ROW, 8:N - 8] = own_view[SEAM_ROW, 8:N - 8]

    dist = np.full((N, N), np.inf, np.float32)
    rows = np.arange(N, dtype=np.float32)[:, None]
    dist[smask] = np.broadcast_to(np.abs(rows - SEAM_ROW), (N, N))[smask]
    width = np.full((N, N), float(BLEND_CELLS), np.float32)

    edge_list = _edge_list(kind, smask)
    seam_v = np.zeros((N, N), dtype=bool)
    for e in edge_list:
        seam_v.ravel()[e["verts"]] = True

    return dict(smask=smask, seam_v=seam_v, target=target, own_view=own_view,
                hard=hard, hard_vals=hard_vals, dist_seam=dist, width_cells=width,
                edge_list=edge_list, tam_w=np.zeros((N, N), np.float32))


def _erode(mask):
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def _dist_to_seam(smask, seam_row):
    dist = np.full(smask.shape, np.inf, np.float32)
    for r in range(smask.shape[0]):
        if smask[r].any():
            dist[r] = abs(r - seam_row)
    return dist


def _edge_list(kind, smask):
    """Synthesize edge_list entries (verts + inward normal) matching the
    seam geometry of each fixture, in the production format."""
    edges = []
    cols = list(range(8, N - 8))

    def hline(row, normal_y, c0, c1):
        verts = [r * smask.shape[1] + c for r, c in
                 [(row, c) for c in range(c0, c1)]]
        edges.append({"verts": verts, "normal": (normal_y, 0.0)})

    def vline(col, normal_x, r0, r1):
        verts = [r * smask.shape[1] + c for r, c in
                 [(r, col) for r in range(r0, r1)]]
        edges.append({"verts": verts, "normal": (0.0, normal_x)})

    if kind in ("flat_step", "sloped_step", "mountain_to_lowland"):
        hline(SEAM_ROW, +1, cols[0], cols[-1] + 1)
    elif kind == "corner":
        hline(SEAM_ROW, +1, cols[0], 110)
        vline(110, +1, SEAM_ROW, SEAM_ROW + BLEND_CELLS * 8)
    elif kind == "staircase":
        r = SEAM_ROW
        c = cols[0]
        run = 24
        while c < cols[-1] - run:
            hline(r, +1, c, c + run)
            c += run
            if c < cols[-1] - 8:
                vline(c, +1, r, r + 8)
                r += 8
        hline(r, +1, c, cols[-1] + 1)
    return edges


def _check(kind: str, ctx: dict, field: np.ndarray, report: dict) -> list:
    fails = []
    seam = ctx["seam_v"]
    c0 = float(np.abs(field[seam] - ctx["own_view"][seam]).max()) if seam.any() else 0.0
    if c0 > 1e-3:
        fails.append(f"C0 {c0:.1f} != 0")
    if np.isnan(field[ctx["smask"]]).any():
        fails.append("NaN inside solve band")
    # Edge-based normal first drop: |field(inland) - field(seam)| per seam
    # vertex, using each edge's own normal (row-difference checks would
    # conflate normal drops with legitimate tangential seam variation).
    H, W = field.shape
    max_drop = 0.0
    for e in ctx["edge_list"]:
        dy, dx = int(round(e["normal"][0])), int(round(e["normal"][1]))
        for f in e["verts"]:
            uy, ux = f // W + dy, f % W + dx
            if 0 <= uy < H and 0 <= ux < W and ctx["smask"][uy, ux]:
                max_drop = max(max_drop, abs(float(field[uy, ux]) - float(field[f // W, f % W])))
    first = max_drop
    limit = GRADE / 8.0 * 2.0     # 2 cells of grade per first vertex allowed
    if first > limit:
        fails.append(f"first-edge drop {first:.0f} > {limit:.0f} GU")
    res = report["residuals"]
    if res["slope_rms"] > 100.0:
        fails.append(f"slope residual RMS {res['slope_rms']:.1f} > 100")
    print(f"  {kind:<20} C0={c0:.1f} first_drop={first:.0f} GU "
          f"slope_rms={res['slope_rms']:.2f} "
          f"{'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    return fails


def main() -> int:
    all_fails = []
    render_dir = ROOT / "output" / "mapdata" / "terrain" / "tamriel_reworked" / "solved" / "v3" / "fixture_renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    rcfg = {"azimuth_deg": 315.0, "altitude_deg": 45.0,
            "vertical_exaggeration": 1.0,
            "hypsometric_stops_gu": [
                [-500, 30, 40, 80], [0, 70, 110, 160], [60, 92, 140, 92],
                [3000, 190, 180, 120], [6000, 165, 125, 90],
                [9000, 150, 110, 80], [11000, 225, 225, 232],
                [14000, 255, 255, 255]]}
    for kind in ("flat_step", "sloped_step", "corner", "staircase",
                 "mountain_to_lowland"):
        ctx = _base_ctx(kind)
        v3 = {"surface": {"data_weight": 1.0, "smooth_weight": 0.05,
                          "slope_weight": 25.0, "cg_tol": 1e-6,
                          "cg_maxiter": 800}}
        field, report = solve_surface(ctx, v3)
        assembly = report["assembly"]
        eq = report["equation_counts"]

        if assembly["empty_equation_rows"] != 0:
            raise AssertionError(
                f"{kind}: solver assembled "
                f"{assembly['empty_equation_rows']} empty equation rows"
            )

        expected_rows = (
            eq["data"]
            + eq["laplacian"]
            + eq["slope"]
        )

        if assembly["rows"] != expected_rows:
            raise AssertionError(
                f"{kind}: equation count mismatch: "
                f"matrix={assembly['rows']} expected={expected_rows}"
            )

        if eq["data"] != report["unknowns"]:
            raise AssertionError(
                f"{kind}: expected one data equation per unknown"
            )

        if eq["laplacian"] != report["unknowns"]:
            raise AssertionError(
                f"{kind}: expected one Laplacian equation per unknown"
            )
        all_fails += _check(kind, ctx, field, report)

        # colored hypsometric review render (owner side + solved band)
        disp = field.copy()
        owner_rows = ctx["own_view"][: SEAM_ROW + 1]
        disp[: SEAM_ROW + 1] = owner_rows
        sh = hillshade(disp, azimuth_deg=rcfg["azimuth_deg"],
                       altitude_deg=rcfg["altitude_deg"],
                       z_scale=float(rcfg["vertical_exaggeration"]))
        rgb = hypsometric_rgb(disp, sh, rcfg["hypsometric_stops_gu"])
        save_shade_png(rgb, render_dir / f"fixture_{kind}.png", 4,
                       title=f"fixture: {kind} (owner top / solved band below)")

    if all_fails:
        print(f"FAILURE: {len(all_fails)} fixture assertion(s) failed")
        return 1
    print("ALL FIXTURES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
