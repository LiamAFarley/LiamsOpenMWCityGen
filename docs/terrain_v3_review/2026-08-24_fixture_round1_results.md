# Fixture round 1 results — Sol: diagnose corner/staircase C0 failure + visible seam line

Date: 2026-08-24 · Follows: `2026-08-24_solver_debug_state.md`
Status: **your exact-edit assembler fix is applied and works numerically
(slope_rms 0.04–0.08 GU, C0 exact on 3/5 fixtures) — but corner and
staircase fail, and the user's visual review rejects ALL five fixture
renders for a visible nonsense line at the seam.** Sol: diagnose and return
exact edits again. Do not hand back to the implementing agent without them.

## 1. Numerical results (production assembler, your replacement)

| fixture | C0 (GU) | first-edge drop (GU) | slope_rms (GU) | result |
|---|---|---|---|---|
| flat_step | 0.0 | 0 | 0.08 | PASS |
| sloped_step | 0.0 | 40 | 0.08 | PASS |
| mountain_to_lowland | 0.0 | 167 | 0.06 | PASS |
| corner | **4,435.1** | **4,835** | 0.07 | FAIL |
| staircase | **4,485.1** | **4,885** | 0.04 | FAIL |

Assembly invariants hold on all five: empty_equation_rows = 0, equation
counts = 2n + m, data/laplacian one equation per unknown.

## 2. Visual results (user review — authoritative)

Colored hypsometric renders (4 px/vertex):
`output/mapdata/terrain/tamriel_reworked/solved/v3/fixture_renders/fixture_{kind}.png`
(also copied to this repo under `docs/terrain_v3_review/evidence/`).

User verdict: **"it still seems to generate a nonsense line at the seam in
all of them and is super fucked up"** — including the three numerically
PASSING fixtures. So a numerically satisfied slope family and exact C0 do
not produce a visually acceptable seam even on synthetic data. The renders
show a hard one-vertex line at the seam in every fixture.

## 3. Corner/staircase failure signature

Both failing fixtures are the ones where two seam edges MEET (horizontal +
vertical seam segments sharing corner vertices). C0 ≈ 4,400–4,500 GU means
some seam vertex is NOT pinned to its owner height. Suspects to check:

1. Corner vertices appear in TWO edges' vert lists; the slope family
   creates one equation per (edge, vert) — but the C0 failure is about the
   HARD PIN, not slope rows. Check whether corner verts are missing from
   `hard`/`hard_vals` (fixture `_base_ctx` builds hard per-kind: the corner
   fixture's vline hard column and hline hard row may not cover the shared
   corner the same way production `rasterize_seam` does).
2. In production `rasterize_seam`, corner verts ARE in multiple edge lists
   and `seam_v` covers them; verify the same for the fixture's synthetic
   raster and whether the solver's eliminated-boundary handling double-
   counts or drops them.
3. The first-edge drop ≈ 4,800 GU at the corner suggests the vertex after
   the seam along one edge is solving to the low target — i.e. the slope
   equation for that edge may be skipped (invalid owner sample at corner
   normals?) or its column is a hard ring vertex.

## 4. Visible seam line in PASSING fixtures — questions for Sol

The passing fixtures pin the seam row to owner heights and ramp the data
weight 0.05 → 1.0 over ~6 cells. The renders still show a distinct
one-vertex line at the seam. Candidate causes Sol should adjudicate:

- the hard-pinned seam row itself forms a 1-vertex terrace because the
  membrane (ws = 0.05) cannot smooth across the hard row's fixed values;
- the wd floor 0.05 near the seam lets the low Tamriel target pull the
  first inland vertices down relative to the pinned seam row;
- hillshade exaggeration of a legitimate C1 kink.

Sol: state which, and return the exact fix (constraint change, weight
change, post-solve smoothing pass, or renderer change).

## 5. State

- Code: `src/procgen/terrain_blend.py` (your assembler + debug prints),
  `tools/terrain/test_surface_fixtures.py` (fixtures + renders),
  `tools/terrain/solve_region_v3.py`, `src/procgen/terrain_metrics.py`
  (C1 now GU per raster edge per your instruction).
- Config: `terrain_relief.gentle_end_fraction` = 0.15 (user ruling);
  `solve.v3.surface` = data 1.0 / smooth 0.05 / slope 25; `solve.v3.quality`
  gates active (first-edge drop 2,500 GU, slope residual 200 GU).
- The real-corpus TR run has NOT been re-run since the assembler fix —
  blocked on these fixtures passing both numerically AND visually.
