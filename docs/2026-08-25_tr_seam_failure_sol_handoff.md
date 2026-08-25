# 2026-08-25 TR Seam Failure and Sol Handoff

## User-Reported Failure

The TR/Vvardenfell mountain and filled-tile boundaries remain visibly sharp.
The prior report incorrectly called red selected seams “solved” because it
verified exact seam height (`C0`) and omitted-region coverage, but did not
verify cross-seam terrain shape.

## Verified Baseline Diagnosis

Fresh 61-cluster union review:
`output/mapdata/terrain/tamriel_reworked/solved/tr_actual_c1_v1/v3/`

- Unknowns: `2,081,528`.
- AMG: 5 iterations; setup `1.350 s`, solve `0.883 s`.
- Exact seam C0 error: `0 GU`.
- Exact active-boundary error: `0 GU`.
- Cyan omitted-seam count: `0`.
- Correctly oriented C1: median `0 GU`, p90 `0 GU`, p99 `0.401 GU`, max `12,104 GU`.
- The max outlier is at global vertex `[4800, 15487]`, world cell `(-10, -47)`.
- At that sample the owner normal step is `0 GU`, while the generated first
  step is `12,104 GU`.
- Ordinary anchor coverage: eligible `6477`, created `6472`, inactive `0`,
  invalid owner `0`; five were skipped because the exact active boundary owns
  their first-inland vertex.
- The large isolated failure is a corner-diagonal claim overriding two
  incompatible adjacent edge claims; no averaging or tolerance increase was
  used.

The C1 metric was corrected in `src/procgen/terrain_metrics.py`: the owner
sample is oriented outward, so continuity is `abs(generated_delta +
owner_delta)`, not subtraction of oppositely directed raw deltas.

## Failed Experiments

These were run against the same 61-cluster union and are retained as evidence,
not accepted production outputs:

1. Raw owner-profile mirroring: `tr_profile_v2/v3/`. It produced obvious
   vertical spikes and failed at `6429.7 GU` maximum normal step.
2. Euclidean nearest-seam pixel interpolation: `tr_pixel_interp_v1/v3/`. It
   crossed raster corners and produced triangular/striped artifacts; maximum
   normal step was `3793.3 GU`.
3. Production-edge inward profile continuation: `tr_edge_profile_v1/v3/`.
   It avoided Euclidean nearest-seam assignment but still produced severe
   stripes and failed at `13793.9 GU` maximum normal step. It reported
   `746516` single-claim profile vertices and `465408` multi-edge conflicts.

The failed edge-profile branch was removed from the production path after the
review; only its output evidence and metrics are retained here. It must not be
reintroduced without a new review.

## Sol Questions

- Is the intended solution a corner-aware local surface continuation rather
  than a scalar harmonic correction field?
- How should a generated vertex shared by two raster edge normals be assigned
  when the two owner-derived first-inland heights disagree by thousands of GU?
- Should the seam solve preserve owner tangential/curvature detail, and if so,
  what bounded operator is authoritative without copying raw owner spikes?
- Should the missing/filled cells be generated from neighboring TR terrain
  before the seam solve, instead of using nearest Tamriel ESM edge fill as the
  target in those areas?

## Evidence Images

- Baseline zoom: `terrain/2026-08-25_tr_seam_review/baseline_seam_zoom.png`
- Failed edge-profile zoom: `terrain/2026-08-25_tr_seam_review/failed_edge_profile_seam_zoom.png`
- Failed edge-profile full render: `terrain/2026-08-25_tr_seam_review/failed_edge_profile_after.png`
- Baseline metrics: `terrain/2026-08-25_tr_seam_review/baseline_metrics.json`
- Failed edge-profile metrics: `terrain/2026-08-25_tr_seam_review/failed_edge_profile_metrics.json`

## Canonical State

The working config was restored to the 27-cluster region, canonical `solved/`
output path, and canonical relief path. The failed edge-profile branch is not
present in the production source.
