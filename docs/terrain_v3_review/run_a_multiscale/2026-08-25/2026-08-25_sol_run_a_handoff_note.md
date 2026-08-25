# 2026-08-25 Sol Run A Handoff

## Delivered Scope

Run A was executed only for the real `tr_vvardenfell_wall` region after
flash-review gates on each structure stage. It produced the four structural
fields below and did not run erosion, hydrology, final lock, or other regions:

- Stage-3 harmonic base;
- cleaned fine-detail field;
- macro continuation;
- macro plus meso continuation.

The fresh output and metrics are under:

`output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/run_a_multiscale_tr_2026-08-25/`

## User Assessment

The user's assessment is authoritative for the visual result:

> I think the macro meso made the seams worse without adding a change in structure that improves things.

This agrees with the visual review: fine detail was reduced, but the macro and
macro+meso fields still show the broad slab and conspicuous stepped/rectangular
transition boundaries around the detailed TR wall. The Run A visual gate is
therefore failed and Run B must not start from this output.

## Numerical Evidence

- Runtime: `69.7 s`.
- Owner vertices: `7,312,449`.
- Generated vertices: `2,551,237`.
- Macro correction range: `-2025.1..+2092.7 GU`.
- Meso correction range: `-5531.7..+1457.7 GU`.
- Macro AMG iterations: `6`.
- Meso AMG iterations: `5`.
- Erosion and final lock: not run.

## Investigation Request

Please investigate why the complete-field macro/meso continuation changes the
seam appearance without producing coherent TR ridge/valley structure, with
special attention to the large meso negative correction and the stepped active
band boundaries. Treat the four PNGs as the visual evidence, not the numerical
solver convergence as acceptance.
