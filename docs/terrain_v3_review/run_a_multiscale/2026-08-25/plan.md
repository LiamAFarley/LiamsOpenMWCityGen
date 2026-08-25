# 2026-08-25 Run A Implementation Plan

## Stage 1: Owner Derivatives

Patch `src/procgen/terrain_features.py` with a masked derivative operator.
Central differences are used when both neighbors are valid; one-sided
differences are used when exactly one neighbor is valid; invalid samples stay
`NaN`. Apply it to H24 for gradient and Hessian terms. Smooth tensor products
with the existing normalized-mask Gaussian routine so non-owner pixels never
contribute derivative energy.

Verification: compile the module, run a narrow synthetic masked-edge probe,
and obtain a `review-flash` read-only review before continuing.

## Stage 2: Multiscale Continuation

Replace the point-source guide mechanism in `src/procgen/terrain_structure.py`
with complete-field band continuation. Build owner and generated target bands:

- macro: `H24 - H64`, active for 8 cells;
- meso: `H8 - H24`, active for 4 cells.

For each band, solve a direct harmonic correction with owner/seam Dirichlet
values and zero correction at the complete active-band boundary. Use the
existing AMG-capable second-order solver. Use the owner band normal derivative
only for the first generated row; omit ambiguous shared-corner anchors.

Suppress generated fine detail with a config-driven smootherstep keep factor
from `0.2` at the seam to `1.0` at 6 cells. Never alter owner vertices.

Verification: compile, run a narrow synthetic band contract probe, and obtain a
second `review-flash` read-only review before wiring the real render command.

## Stage 3: Run A Driver

Extend `tools/terrain/erode_region_v3.py` with an explicit Run A mode and an
output-directory override. Run A must stop after the structural fields and
write only the four requested real TR renders plus metrics. Add all tunable
width/keep values to `configs/tamriel_reworked_v1.json`.

Verification: compile, parse JSON, obtain a third `review-flash` review, then
run only `tr_vvardenfell_wall` into the fresh run directory.

## Stop Condition

After the four renders are visually inspected and documented, stop. Do not run
erosion, final lock, or any other region in this task.
