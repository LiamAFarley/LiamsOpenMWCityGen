# 2026-08-25 TR v3 AMG Corner Mask Apron Run

## Scope

Implemented the supplied AMG, corner-incidence, seam-width, and one-cell
outer-apron plan against the direct harmonic baseline. No river-specific
logic, erosion, smoothing, clamping, or PDE redesign was added.

## Actual Changes

- Added optional `pyamg` import and selectable `amg_rs_cg`, `amg_sa_cg`, and
  `jacobi_cg` solver modes. The default is `amg_rs_cg`; the Laplace matrix and
  direct CG solve remain unchanged.
- Added CG iteration counts, AMG level count, setup time, and solve time to
  the surface report.
- Built orthogonal seam-corner incidence from `edge_list`, suppressed the
  ordinary anchor at the corner seam vertex, and created one diagonal corner
  continuation from both owner derivatives.
- Resolved the actual corner collision root cause: the diagonal generated
  vertex was also claimed by the two adjacent one-dimensional anchors. The
  corner claim now owns that generated point; adjacent edge anchors remain
  active elsewhere.
- Calculated relief/mismatch width only at seam vertices and propagated that
  metadata inward from the nearest seam. Interior target features no longer
  move the outer correction boundary directly.
- Added `outer_apron_cells=1.0`; the exact zero-correction outer boundary is
  now one cell beyond the requested blend width.
- Added `linear_solver`, `cg_maxiter=200`, `amg_max_coarse=500`, and
  `outer_apron_cells` to the v3 config.

## Verification

- Compilation passed.
- Existing five surface fixtures passed.
- Fresh real crop output:
  `output/mapdata/terrain/tamriel_reworked/solved/tr_amg_corner_v2/v3/`
- AMG real crop: `1,074,162` unknowns, `5` CG iterations, `8` AMG levels,
  `0.6359 s` setup, `0.4387 s` solve, `1.6 s` reported solve stage.
- Previous Jacobi crop: `730,992` unknowns, `7.2 s` solve, approximately
  `50.7 s` total. The unknown count increased because of the one-cell apron;
  the solve time still dropped substantially.
- C0 seam error: `0 GU`.
- Outer-edge error: `0 GU`.
- Maximum sampled normal step: `520 GU`.
- Empty equation rows: `0`.
- Anchor conflicts: `0`; corner vertices `22`, corner anchors `22`, skipped
  corner anchors `0`.
- AMG returned status `0` and the residuals remained below the existing gates.
- The v1-to-v2 field difference is localized to the corner correction: only
  `10` vertices differ by more than `1 GU`, with `2` above `64 GU`.

## Visual Audit Status

The v2 render shows less abrupt corner behavior and a broader primary
transition. The earlier gray missing-height block remains filled. However,
visible straight/rectilinear boundary traces remain in the seam zoom and the
comparison image. This pass therefore does not establish visual acceptance;
the remaining lines require a separate root-cause review.

The river/valley question was intentionally not investigated yet, as required
by the plan. No erosion or thalweg work was started.

## Config State

The canonical config output paths were restored to `solved/` after the fresh
comparison run. The new crop artifacts remain under `tr_amg_corner_v2`.
The canonical `solved/v3/relief` cache predates the current ramp hash and will
fail closed until regenerated; the crop reused the valid hash-matched relief
cache under `tr_crop_test`.
