# 2026-08-25 Run A Visual Review

## Objective

Render only Sol's Run A structural fields for the real `tr_vvardenfell_wall`
crop after the staged derivative, multiscale, and driver reviews. No erosion,
hydrology, final lock, or other region was run.

## Actual Changes

- Replaced zero-filled owner derivatives with masked central/one-sided
  derivatives and normalized-mask tensor smoothing.
- Removed the sparse point/ribbon structural mechanism.
- Added generated-side H24-H64 macro and H8-H24 meso harmonic continuation,
  owner-band first-inland anchors, and generated-only fine-detail attenuation.
- Added `--run-a` and `--output-dir` to the structural/erosion driver; Run A
  exits before downstream hydrology and erosion.
- Removed eleven dead point/ribbon configuration keys while retaining active
  terrain-feature percentile and pyramid keys.

## Evidence Checked

Fresh output directory:

`output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/run_a_multiscale_tr_2026-08-25/`

Artifacts present:

- `tr_vvardenfell_wall_stage3_base.png`
- `tr_vvardenfell_wall_cleaned_fine.png`
- `tr_vvardenfell_wall_macro_continuation.png`
- `tr_vvardenfell_wall_macro_meso_continuation.png`
- `tr_vvardenfell_wall_run_a_metrics.json`

Real run metrics:

- Total runtime: `69.7 s`.
- Owner vertices: `7,312,449`.
- Generated vertices: `2,551,237`.
- Macro: `8` cells, `2,421,691` active vertices, correction range
  `-2025.1..+2092.7 GU`, six AMG iterations.
- Meso: `4` cells, `1,528,754` active vertices, correction range
  `-5531.7..+1457.7 GU`, five AMG iterations.
- `erosion_run: false`; `final_lock_run: false`.

## Visual Findings

1. **Stage-3 base:** The existing broad isotropic generated slab and sharp
   detailed-owner/TR wall boundary are clearly visible.
2. **Cleaned fine-detail:** Repetitive fine texture is reduced in the
   generated corridor, but the broad slab and owner/generated boundary remain.
3. **Macro continuation:** Large-scale correction reaches away from the seam,
   but the transition still reads as stepped/rectilinear bands rather than
   coherent TR ridge and valley continuation. The owner wall trace remains
   visible.
4. **Macro plus meso:** Meso detail adds additional local variation, but also
   leaves conspicuous staircase/corridor boundaries around the detailed TR
   terrain. It does not pass the requirement that the old rectangular massif
   outline no longer be obvious.

## Gate Result

Run A produced the requested real TR render set, but the structural visual gate
fails. The output is evidence only, not an accepted continuation field. Do not
run Run B, erosion, final lock, or the multi-region batch until the structural
field's slab and stepped boundary are addressed.
