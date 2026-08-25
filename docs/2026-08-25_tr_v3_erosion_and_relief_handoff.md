# 2026-08-25 TR v3 Erosion and Relief Handoff

## Reason For Review

The workspace followed the v3 sequence through one real TR/Vvardenfell-wall
review frame. The user observed that the erosion cycle 00 and final renders are
visually indistinguishable and that the implementation no longer resembles the
full erosion design. Sol should revise the erosion plan before any Skyrim/Cyr
batch or world-wide run.

## Current Workspace State

- The visible solve frame is configured as `review_bbox_cells = [-40, -47, 1,
  1]`, covering the former bottom-right cyan seams as well as the mountain/TR
  boundary.
- Relief currently computes the 3x base relief target, then synthesizes only
  the incomplete cells in the review frame. The last run found 12 cells in
  three components with 4,032, 24,129, and 20,097 unknown vertices.
- The harmonic bridge covers 6,468 seam vertices, has zero cyan omissions and
  zero float-field C0 error, and converges in five AMG iterations.
- The first erosion prototype uses deterministic priority-flood routing, an
  8-neighbor MFD router, normalized stream power, 16 cycles, and a six-cell
  fixed owner halo. Its real run used 2,509,191 editable vertices and
  2,488,602 owner-halo vertices.
- The post-erosion two-cell harmonic lock converged in four AMG iterations with
  float-field C0 `0 GU`; the sampled normal-step maximum was `1,070.9 GU`.

## User-Observed Problem

The cycle renders do not show convincing large-scale local landform matching
between the TR side and Tamriel side. The first implementation is therefore
diagnostic evidence only, not an accepted erosion system. Do not expand it to
the requested roughly ten TR/Skyrim/Cyr renders until the erosion design has
been reviewed and replaced or materially corrected.

## Relief Curve Note

The current `terrain_relief` response curve reaches the configured gentle gain
of `1.6` through the initial ramp before the full curve. The user now wants the
coast-protection plateau removed: use the smooth response from 0 rather than a
fixed-gain-1 section followed by a rapid climb. This local curve edit is
deliberately not needed for the erosion architecture review and can follow the
push of this handoff.

## Files To Review

- `src/procgen/terrain_blend.py`
- `src/procgen/terrain_inpaint.py`
- `src/procgen/terrain_erosion.py`
- `tools/terrain/erode_region_v3.py`
- `configs/tamriel_reworked_v1.json`
- `docs/2026-08-25_tr_v3_erosion_and_relief_handoff.md`

The authoritative real renders and metrics remain in the workspace under
`output/mapdata/terrain/tamriel_reworked/solved/v5_missing_cells/v3/`; this core
checkout is a source/reference handoff and does not contain the large output
bundle.
