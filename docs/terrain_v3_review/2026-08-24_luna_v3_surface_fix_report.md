# v3 surface solve defect-fix report — 2026-08-24

## Plan followed

Applied the Milestone 2 corrections without changing the screened-Poisson plus
owner-slope-row architecture: use the relief-scaled field as the solve and
embedding canvas, widen the adaptive band from owner/target mismatch, preserve
hard seam heights, and add an explicit outer-band continuity assertion.

## Root causes (ranked)

1. `solve_surface` copied the unscaled `tam_w` outside the solve mask, while
   the driver embedded the window into unscaled `tam_h`. This created a
   relief-delta-sized staircase at every band edge.
2. Adaptive width only measured target local relief. A flat Tamriel flank next
   to a high TR wall therefore received the two-cell minimum despite a large
   seam height mismatch.
3. The solve had no metric/assertion for non-owner band-edge continuity, so the
   embedding regression was not fail-closed.
4. The slope rows were firing; the rerun reports 2,356 rows. Their sign and
   weight were retained after inspection: continuation uses
   `H(inward)=H(seam)+(H(seam)-H(owner-outward))` and gradient weight remains 6.
   The membrane fixed-neighbor sign is also already correct (`rhs += ws*h`).

## Changes made

- `src/procgen/terrain_blend.py`: propagate seam mismatch to a width field and
  convert it using configurable `max_blend_grade_gu_per_cell`; embed the
  relief-scaled target outside the solve mask; expose target full-map canvas
  and solve diagnostics.
- `src/procgen/terrain_metrics.py`: add non-owner band-edge continuity metric.
- `tools/terrain/solve_region_v3.py`: embed on the scaled canvas, assert the
  configured edge tolerance, record the metric, and emit a configured,
  north-up 4 px/vertex seam crop.
- `configs/tamriel_reworked_v1.json`: add grade/tolerance knobs and the
  TR-wall seam crop bbox.

## Verification

Command:

```text
python F:\ProcGenWorkspace\tools\terrain\solve_region_v3.py --region tr_vvardenfell_wall
```

The run completed in 61.9 s total (context 6.5 s, solve 25.6 s, metrics 0.3
s), under the 120 s limit. `cg_status` was 0. Results:

- `seam_c0_max_gu`: **0.0**
- slope rows: **2,356**
- C1 normal mismatch: median **58.907**, p90 **248.319**, p99 **286.468**,
  max **302.683** GU/vertex
- band-edge max absolute difference: **0.0 GU**; assertion passed against
  the configured 0.001 GU tolerance
- adaptive width range: **2.0–10.0 cells**

Focused `py_compile`, JSON parsing, and `git diff --check` also passed.

The regenerated comparison and the new focused crop were visually inspected:

- `output/mapdata/terrain/tamriel_reworked/solved/v3/tr_vvardenfell_wall_v3_comparison.png`
- `output/mapdata/terrain/tamriel_reworked/solved/v3/tr_vvardenfell_wall_v3_seam_zoom.png`

The focused crop shows the lowland rising into the TR wall over the available
blend region rather than a one-vertex plummet. The standard comparison no
longer has the relief-scaled-versus-unscaled height step; the exact numerical
edge assertion confirms this.

## Deviations

None from the requested solver architecture or protected files. No corpus,
relief semantics, or render-helper changes were made. The new crop selects the
configured TR wall bbox because the region contains multiple disconnected
seam runs.

## Remaining risks

This verifies the requested representative TR wall only. It does not establish
quality for all 435 seam clusters, serialized VHGT equality, or downstream
erosion (Milestone 2 intentionally has no erosion). The focused render still
shows the owner terrain's sharp visual wall boundary; that is owner-side
terrain, not a measured height discontinuity in the generated band.
