# 2026-08-25 TR Primitive Checkpoint Request

## Objective

Implement Sol's terrain-primitive continuation plan through P5 for the real
`tr_vvardenfell_wall` crop, targeting only the bottom-right quadrant of the
existing review frame. The target is the southern TR canyon/plateau/mesa
terrain, not a generic continuation of every terrain form.

## Binding Scope

- Preserve the accepted Stage-3 field, relief scaling, and missing-cell path.
- Keep the failed Run A macro/meso output as evidence only; do not retune it or
  use it as the primitive generator.
- Analyze and synthesize plateau/scarp/canyon primitives on a semantic grid
  downsampled by the configured factor, then finalize at full resolution.
- Keep owner terrain immutable.
- Use one edge-aware screened-Poisson reconciliation solve and the existing
  narrow final seam lock.
- Render only the real `tr_vvardenfell_wall` target.
- Do not run erosion, hydrology, or the broad multi-region batch.

## Target Selection

The target is configured as `bottom_right` relative to the existing review
bbox. The driver resolves that fraction to local vertex coordinates and passes
it to primitive analysis; no source coordinate is hard-coded in the module.

## Required Outputs

1. owner/reference plus Stage-3 context;
2. plateau component classification;
3. continued footprint plus scarp diagnostic;
4. plateau-only candidate;
5. plateau plus canyon candidate;
6. reconciled pre-lock field;
7. post-lock field;
8. comparison sheet;
9. top/scarp grayscale diagnostic;
10. primitive, timing, seam, and final-lock metrics;
11. a dated handoff report describing detected/crossing components, fits,
    support paths, scarps/canyons, and remaining visual defects.

## Acceptance Gate

The generated side must visibly contain coherent plateau/mesa topography and
major canyon continuation without a rectangular harmonic corridor, Gaussian
bumps, ribbon scars, or a seam worse than the Stage-3 baseline. AMG convergence
alone is not acceptance. Stop for visual review after P5.
