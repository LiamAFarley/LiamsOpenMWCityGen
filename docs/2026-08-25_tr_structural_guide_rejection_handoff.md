# 2026-08-25 Structural Guide Rejection Sol Handoff

## Objective

Continue the real `tr_vvardenfell_wall` structural-continuation pilot from the
v3 plan and determine whether the current sparse guide geometry removes the
rectangular owner/generated terrain boundary without introducing new artifacts.

## Evidence Checked

- Real full-window command: `tools/terrain/erode_region_v3.py` with
  `configs/tamriel_reworked_v1.json`, region `tr_vvardenfell_wall`.
- Real structural, erosion, and final-lock outputs under
  `docs/terrain_v3_review/evidence/2026-08-25_structural_rejection/`.
- Focus render inspected at source-derived crop coordinates around the wall.
- Plan sections 5.1-6.1 in
  `docs/2026-08-25_tr_v3_structural_erosion_plan.md`.

## Experiments

1. A point-source/H24 first-inland profile was rejected numerically. With
   outward H24 slope extrapolation, the structural correction reached
   `-11,306.9 GU`; removing the slope did not fix it (`-11,384.9 GU`).
2. A tangent-directed sparse guide experiment was rejected visually. It
   remained bounded (`-1,761.4/+865.4 GU` before final lock) but the focus
   render still showed repeated contour-like scars and the hard wall trace.
3. Increasing sparse Gaussian support from 12 to 32 vertices was also rejected
   visually. The support became broader, but the same synthetic guide traces
   remained and the generated side still read as a smooth slab.
4. A tangent-only first-point probe produced only 24 valid crossings and did
   not change the rectangular slab. A bounded first-inland profile using
   `H24(owner) - H0(first_generated)`, capped by the measured local-relief
   envelope, produced stepped square terraces; its final-lock correction still
   reached `-5,672.9 GU` and it was rejected visually.

## Restoration

All experiments were removed from the active source/configuration. The
structural module was restored to the immediately preceding point-source guide
implementation, not the older full-ribbon artifact: the active run reports
`103` source points, `257,726` guide vertices, and structural correction
`-2,249.98/+1,137.03 GU`. The active output is the freshly regenerated exact
point-source baseline under
`docs/terrain_v3_review/evidence/2026-08-25_structural_rejection/`.
The older `erosion_structural_baseline_review/` directory is retained as
historical context only. Rejected experiments are not presented as accepted
terrain.

## Evidence Files

- `tr_vvardenfell_wall_structural_erosion_metrics.json`
- `tr_vvardenfell_wall_stage3_base.png`
- `tr_vvardenfell_wall_structural_pre_erosion.png`
- `tr_vvardenfell_wall_erosion_cycle_00.png`
- `tr_vvardenfell_wall_erosion_cycle_24.png`
- `tr_vvardenfell_wall_erosion_final.png`
- `tr_vvardenfell_wall_comparison_sheet.png`
- `tr_vvardenfell_wall_structural_focus.png`
- `tr_vvardenfell_wall_tangent_point_focus.png` (rejected probe)
- `tr_vvardenfell_wall_profile_focus.png` (rejected probe)

Verification after restoration: `src/procgen/terrain_structure.py` compiles.
No checkpoint commit or push was made because no structural guide variant
passed the visual gate.

## Sol Request

Treat this as a failed structural-continuation handoff, not as an accepted v3
result. The copied structural, feature, hydrology, erosion, driver, and config
files are the exact source state used for the real pilot. Rework the semantic
continuation mechanism against the plan and included renders; do not expand
the ten-case batch or begin world-wide erosion while the Vvardenfell crop still
has the same harmonic slab.

## Unresolved

The systemic rectangular owner/generated boundary remains unresolved. The
current guide implementation does not yet provide a credible plan-compliant
semantic continuation: normal-driven ribbons create scars, tangent-driven
ribbons still create scars, and dense first-inland profiles are forbidden by
the plan and numerically unstable on this crop. Do not expand the ten-case
batch or begin world-wide erosion until the guide geometry is redesigned and
rechecked on this real crop.
