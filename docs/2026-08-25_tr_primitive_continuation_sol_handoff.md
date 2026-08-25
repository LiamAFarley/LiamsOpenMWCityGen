# 2026-08-25 Terrain Primitive Continuation Sol Handoff

## Status

This is a **rejected real-output handoff**. The primitive checkpoint is not an
accepted terrain result and must not be used for downstream erosion, the
ten-region batch, or world-wide generation.

## Core Checkout Links

This copy is the Sol-facing entry point. The links below resolve inside
`procedural-tamriel-core`:

- [Primitive continuation plan](2026-08-25_tr_primitive_continuation_plan.md)
- [Structural erosion plan](2026-08-25_tr_v3_structural_erosion_plan.md)
- [Older Sol plan](2026-08-25_sol_plan.md)
- [Primitive stage reviews](terrain_v3_review/primitive_plateau_canyon/)
- [Run A visual review](terrain_v3_review/primitive_plateau_canyon/2026-08-25_run_a_visual_review.md)
- [Structural-guide rejection report](terrain_v3_review/primitive_plateau_canyon/2026-08-25_structural-guide-rejection-report.md)
- [Rejected primitive metrics](../output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/primitive_plateau_canyon_tr_plateau_transition_seedfix_2026-08-25/tr_plateau_transition_primitive_metrics.json)
- [Rejected primitive comparison sheet](../output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/primitive_plateau_canyon_tr_plateau_transition_seedfix_2026-08-25/tr_plateau_transition_primitive_comparison_sheet.png)
- [User-preferred plateau reference](../output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/tr_plateau_transition_v3_after.png)

The user inspected the real output and rejected it because the generated side
still contains the old Tamriel ESM fine noise, the proposed plateau candidate
forms large sharp triangular excursions, and there is no recognizable broad
sloping plateau with winding canyon continuation. The output is visibly
incoherent at the target seam despite passing several numerical contracts.

## User-Requested Transfer

Copy, commit, and push the current terrain source, plans, review records,
rejected real renders, metrics, and this handoff to
`procedural-tamriel-core`. A future Sol chat should use this document as the
entry point, propose the next design, and hand that proposal to a fresh Sol
implementation session. Do not infer acceptance from the local P1-P5 review
verdicts; those reviews were source-contract reviews and did not accept the
real visual result.

## Exact Real Run

Command:

```text
python tools/terrain/erode_region_v3.py --config configs/tamriel_reworked_v1.json --region tr_plateau_transition --primitive-checkpoint --output-dir output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/primitive_plateau_canyon_tr_plateau_transition_seedfix_2026-08-25
```

The run used the configured `tr_plateau_transition` cluster union because its
existing `tr_plateau_transition_v3_after.png` framing is the user's preferred
plateau crop. It did not run the ten-region set, hydrology, or erosion.

Measured run facts from the real metrics:

- Review frame: `2945 x 2881` vertices; primitive target: rows `1472:2945`, columns `1440:2881`.
- P1 selected four plateau components with confidences `0.8175`, `0.5923`, `0.7242`, and `0.6581`; candidate seed support was real, not an empty-detection run.
- P2 generated-side support: `73,200` vertices; P2 scarp support: `0` vertices.
- P3 candidate support: `73,072` vertices; candidate correction range: `0` to `13,555.2 GU` before reconciliation. The large excursions are the triangular artifacts visible in the candidate render.
- P4 canyon line vertices: `0`; canyon depth maximum: `0 GU`. No canyon primitive was produced in the real run.
- P5 guide rows: `73,200`; reconciliation correction range: `-2,225.1` to `+3,579.7 GU`.
- Final lock seam C0: `0 GU`; sampled C1 median and p90: `0 GU`; sampled C1 maximum: `1,628.9 GU`.
- AMG reconciliation: 4 CG iterations, `0.412 s` setup, `0.255 s` solve.
- Total run time: `61.9 s`.
- `erosion_run: false`.

The exact metrics are in:

`output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/primitive_plateau_canyon_tr_plateau_transition_seedfix_2026-08-25/tr_plateau_transition_primitive_metrics.json`

## Evidence To Copy

Copy the complete real checkpoint directory, including its PNGs and JSON:

`output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/primitive_plateau_canyon_tr_plateau_transition_seedfix_2026-08-25/`

The important views are:

- `tr_plateau_transition_owner_reference.png`
- `tr_plateau_transition_stage3_context.png`
- `tr_plateau_transition_plateau_classification.png`
- `tr_plateau_transition_plateau_footprint_scarp.png`
- `tr_plateau_transition_plateau_candidate.png`
- `tr_plateau_transition_plateau_canyon_candidate.png`
- `tr_plateau_transition_reconciled_pre_lock.png`
- `tr_plateau_transition_post_lock.png`
- `tr_plateau_transition_top_scarp_grayscale.png`
- `tr_plateau_transition_primitive_comparison_sheet.png`
- `tr_plateau_transition_primitive_metrics.json`

Also copy the user's useful prior reference crop and metrics:

- `output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/tr_plateau_transition_v3_after.png`
- `output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/tr_plateau_transition_v3_reference.png`
- `output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/tr_plateau_transition_v3_target.png`
- `output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/tr_plateau_transition_v3_seam_zoom.png`
- `output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/tr_plateau_transition_v3_comparison.png`
- `output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/tr_plateau_transition_v3_metrics.json`

The `tr_plateau_transition_v3_after.png` image is a framing/reference
artifact only. It is not evidence that the new primitive output matches it.

## What Was Implemented

The current local source contains the following primitive stages in
`src/procgen/terrain_primitives.py`:

- P0 configuration under `terrain_primitives` and `structure_mode: primitives`.
- P1 semantic plateau-component inference on a factor-4 grid, robust affine top fitting, and diagnostics.
- P2 generated-side footprint continuation using bounded semantic Dijkstra support and a shared-seam-to-first-generated resolver.
- P3 plateau candidate synthesis from fitted top planes with configured fine-detail attenuation and scarp parameters.
- P4 connected owner-valley component selection and generated-only bounded canyon routing with configured direction, low-target, uphill, plateau, width, and thalweg terms.
- P5 one symmetric edge-aware screened-Poisson reconciliation with PyAMG/Jacobi options, explicit owner/active-boundary Dirichlet values, and conductance reduction across scarp normals.

`tools/terrain/erode_region_v3.py` has a dedicated
`--primitive-checkpoint` path that renders P1 through P5 and then performs the
existing narrow seam lock. It exits before hydrology and erosion. The old
`--run-a` multiscale path remains available as historical baseline evidence.

The shared-seam resolver was added after the first real attempt exposed a
production contract difference: the production owner mask contains the shared
seam vertex, so generated support starts one normal step beyond the seam. This
fix was independently reviewed and verified on synthetic owner masks. It did
not solve the visual problem.

## Why This Design Is Rejected

The failure is not an empty classifier or a missing P5 solve:

1. P1 finds four real plateau-like owner components.
2. P2 produces a large support region, but no scarp profile is extracted.
3. P3 extrapolates each local top plane over support. The result is a large planar/triangular wedge, not the observed plateau body.
4. P4 finds no qualifying owner-valley component around the selected plateau seam contacts, so no winding canyon enters the generated field.
5. P5 reconciles the bad plateau candidates into the background. Its exact C0 and low aggregate residual do not make the geometry correct.
6. The generated side retains the old fine Tamriel noise instead of being replaced by a controlled coarse landform plus intentionally restored detail.

The next design must therefore address the observed geometry, not merely tune
the solver or lower thresholds. In particular, a single affine top plane
extrapolated from a component is insufficient for this crop's broad average
slope plus winding canyon network. Scarp extraction and canyon ownership are
currently absent in the real P4 result. The triangle-shaped candidate must not
be carried forward as a fallback.

## Frozen / Accepted Baseline

Keep these stages frozen unless a new Sol plan explicitly supersedes them:

- authoritative corpus and cell-height provenance;
- relief-first ordering and explicit missing-cell synthesis;
- the accepted direct harmonic seam architecture and complete active-boundary policy;
- the accepted corner-anchor repair and AMG configuration;
- the user's preferred `tr_boundary_union_v1` / v3 harmonic evidence as the prior seam baseline.

The primitive output in this handoff is not a replacement baseline.

## Supersession Chain

These documents are historical context, not current acceptance:

- `2026-08-25_tr_seam_failure_sol_handoff.md`: rejected owner-profile seam experiments; superseded as the active design by the structural continuation work, but still useful for explaining why raw profile mirroring and nearest-owner interpolation were rejected.
- `2026-08-25_tr_v3_erosion_and_relief_handoff.md`: rejected simplified MFD erosion and relief-curve handoff; erosion remains downstream and must not be resumed before structural acceptance.
- `2026-08-25_tr_structural_guide_rejection_handoff.md`: rejected point-source/tangent/profile guide family; the primitive plan supersedes it as the active structural direction, while its visual evidence remains relevant.
- `2026-08-25_run_a_visual_review.md`: rejected macro/meso Run A; it is the direct predecessor to the primitive plan.

The current active design document was:

`tamriel_reworked_v3_terrain_primitive_continuation_plan.md`

It is now itself **rejected at the first real plateau/canyon visual gate** in
the current implementation. Keep the plan for diagnosis and do not claim that
P0-P5 completed the checkpoint. Its later phases are queued only after a new
design passes this target.

The broader plan remains background only:

`tamriel_reworked_v3_structural_erosion_multi_region_plan.md`

The older high-level Sol plan is also background:

`8-25-2026 Sol Plan.md`

## Next Phases Still Required

The plan's later stages must not start yet. A fresh Sol proposal should first
replace or substantially revise the plateau/canyon mechanism and rerun only
the same `tr_plateau_transition` crop.

Only after the target crop is visually accepted should the work proceed to:

- a proper plateau/mesa body whose coarse surface follows the observed average rise rather than a single unconstrained wedge;
- explicit scarp extraction and cross-section fitting from the owner terrain;
- connected canyon/thalweg extraction that recognizes winding low corridors and continues them into generated terrain;
- controlled fine-detail suppression/restoration so old Tamriel noise does not remain as the dominant generated texture;
- reconciliation and final seam lock rechecked at zoom, including C1 outliers;
- the mountain/ridge, valley, cliff, and other primitive checkpoints;
- erosion only after the pre-erosion structural field is visually correct;
- the ten-region TR/Skyrim/Cyrodiil review only after the single-region checkpoint passes.

## Source / Review Links

Current local stage records:

- `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/request.md`
- `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md`
- `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/2026-08-25_p0_config_review.md`
- `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/2026-08-25_p1_plateau_analysis_review.md`
- `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/2026-08-25_p2_footprint_review.md`
- `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/2026-08-25_p3_plateau_candidate_review.md`
- `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/2026-08-25_p4_canyon_review.md`
- `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/2026-08-25_p5_reconciliation_review.md`
- `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/2026-08-25_p5_driver_review.md`
- `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/2026-08-25_shared_seam_seed_review.md`
- `.opencode/runs/2026-08-25_tamriel-reworked-heightmap/2026-08-25_structural-guide-rejection-report.md`
- `.opencode/runs/2026-08-25_tr-run-a-multiscale/2026-08-25_run_a_visual_review.md`

Current source files:

- `src/procgen/terrain_primitives.py`
- `tools/terrain/erode_region_v3.py`
- `configs/tamriel_reworked_v1.json`

All numeric claims above are copied from the real checkpoint metrics and all
visual rejection claims are the user's direct assessment of the rendered
PNG set. This handoff intentionally does not self-certify the result.
