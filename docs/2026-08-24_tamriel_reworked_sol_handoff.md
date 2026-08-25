# 2026-08-24 Tamriel Reworked Sol Handoff

## Purpose

Review the current Tamriel Reworked heightmap seam pipeline after the first
real `tr_vvardenfell_wall` crop. Do not assume the previous fixture or agent
completion claims are correct. Inspect the copied code and the real images
before proposing any implementation.

## Repository State

The current workspace source checkpoint is `d5131aad` in the companion
workspace. These files have been copied into this repository for review:

- `src/procgen/terrain_blend.py`
- `src/procgen/terrain_relief.py`
- `src/procgen/terrain_metrics.py`
- `tools/terrain/relief_preview.py`
- `tools/terrain/solve_region_v3.py`
- `tools/terrain/test_surface_fixtures.py`
- `configs/tamriel_reworked_v1.json`

The active Python environment has `pyamg==5.3.0` installed, but the solver
does not yet import or use it.

## Current Method

- The surface unknown is an additive correction `C = H_final - H_target`.
- The solver assembles a direct four-neighbor Laplace system
  `L_uu C_u = rhs`; it does not use `L.T @ L`, data rows, or post-hoc clamps.
- Seam and distance-defined outer vertices are exact Dirichlet values.
- First-inland owner-slope samples are Dirichlet anchors.
- Per-pixel height-mismatch blend widths are bounded by the configured
  `2..10` cells and use the configured `2500 GU/cell` mismatch grade.
- Disconnected adaptive-mask islands are removed before matrix assembly.
- Missing corpus heights are filled from the nearest finite Tamriel ESM edge
  vertex before relief and seam processing.
- Relief configuration is now `gentle_end_fraction=0.05` and
  `gentle_gain=1.6`.

## Real Crop Evidence

The crop artifacts are in the companion workspace:

- `F:\ProcGenWorkspace\output\mapdata\terrain\tamriel_reworked\solved\tr_crop_test\v3\tr_vvardenfell_wall_v3_after.png`
- `F:\ProcGenWorkspace\output\mapdata\terrain\tamriel_reworked\solved\tr_crop_test\v3\tr_vvardenfell_wall_v3_seam_zoom.png`
- `F:\ProcGenWorkspace\output\mapdata\terrain\tamriel_reworked\solved\tr_crop_test\v3\tr_vvardenfell_wall_v3_comparison.png`
- `F:\ProcGenWorkspace\output\mapdata\terrain\tamriel_reworked\solved\tr_crop_test\v3\tr_vvardenfell_wall_v3_target.png`
- `F:\ProcGenWorkspace\output\mapdata\terrain\tamriel_reworked\solved\tr_crop_test\v3\tr_vvardenfell_wall_v3_metrics.json`
- `F:\ProcGenWorkspace\output\mapdata\terrain\tamriel_reworked\solved\tr_crop_test\v3\relief\tr_vvardenfell_wall_relief_3x.png`

Measured run:

- Harmonic unknowns: `730,992`
- Matrix nonzeros: `3,618,660`
- CG status: `0` with `cg_maxiter=5000`; it failed at the previous `800`
  iteration ceiling.
- Seam C0 error: `0 GU`
- Outer-edge error: `0 GU`
- Maximum sampled normal step: `1189.2 GU`
- Saved field NaN count: `0`
- Adaptive-mask pixels removed: `97,525`
- Missing raw corpus vertices filled: `98,879,236`

Four conflicting corner first-inland anchors were skipped under the explicit
diagnostic policy. Their spreads were `40`, `64`, `1384`, and `1432 GU`.
The largest conflict is a shared diagonal first-inland vertex where two
ownership legs have materially different seam heights. The code reports these
conflicts instead of silently selecting the last processed edge.

## User-Observed Visual Defects

The user sees more than numerical seam success:

- A river appears to have been raised, producing a clear elevated river seam.
- Several long, straight/rectangular terrain seams remain visible in the
  crop, including around the TR/Tamriel transition.
- The earlier gray rectangular area was missing height data. Nearest-edge fill
  removes the gray NaN region, but the resulting ocean/edge fill still needs
  visual judgment.
- Corner transition behavior may be gradual at the corner while adjacent
  straight portions remain too abrupt.

The user can attach the listed images directly to this handoff in the Sol
conversation. Do not dismiss these defects because C0, residual, or quality
scalar gates pass; the rendered images are authoritative for visual defects.

## Questions For Sol

1. Inspect `terrain_blend.py` and determine whether `pyamg.smoothed_aggregation_solver`
   or another AMG preconditioner can reduce iteration count while preserving
   the direct Laplace operator and maximum-principle behavior. Provide a
   measured recommendation before changing the solver.
2. Diagnose the raised-river seam and the long clear seams from the actual
   height-field/ownership/mask logic. Determine whether they come from
   missing-height fill, relief amplification, per-pixel width carving,
   owner-side replacement, corner anchor skipping, or render compositing.
3. Analyze the four corner conflicts and decide whether skipping conflicting
   anchors is mathematically defensible, or whether the corner should use a
   separate multi-edge boundary treatment. Do not average large conflicting
   values without explaining the geometry.
4. Check whether the nearest-edge missing-height fill is actually selecting
   surrounding ocean values at the gray-cell locations. If not, propose a
   source-authoritative ocean fallback based on the Tamriel ESM edge.
5. Recommend the smallest root-cause correction. Do not add arbitrary
   smoothing, extrusion, clamping, or threshold changes merely to improve a
   screenshot.

## Constraints

- Do not run a full-region or broad batch generation.
- Do not modify source plugins or original Tamriel files.
- Do not replace the harmonic operator with a biharmonic/least-squares solve.
- Do not accept scalar gates as proof that the visible seams are fixed.
- Return diagnosis, evidence, and a bounded implementation plan first; wait
  for lead approval before changing the copied solver.
