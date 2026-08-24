# Building Rule Kit Phase 3a/3b Tools

Date: 2026-08-22

## Purpose

These tools implement the staged wall/mount extraction for Phase 3 of the xFa
building rule kit. They measure wall/facade geometry, attachment mount
geometry, and access bundles from regenerated source stamp products. They do
not compose houses, edit source products, or perform roof/dormer extraction.

## Entry points

- `tools/cityforge/build_wall_mount_profiles.py` — loads the JSON config,
  launches one Blender triangle-evidence pass, and writes facade, mount, and
  access products.
- `tools/cityforge/blender_wall_mount_evidence.py` — Blender worker that only
  imports fresh NIFs and exports evaluated triangle facts.
- `tools/cityforge/render_profile_diagnostics.py` — writes shell top-down and
  facade-overlay diagnostics plus attachment front/back pairs.
- `tools/cityforge/blender_front_back.py` — Blender worker for signed-axis
  front/back renders. It renders raw `+axis` and `-axis`; the host labels them
  using the measured mount profile.
- `tools/cityforge/blender_facade_overlay.py` — Blender worker that overlays
  measured facade planes on the imported shell.

## Configuration and outputs

The Phase 3a site/model/threshold choices are in
`configs/kits/xfa_sky_nord_house/phase03_config.json`; Phase 3b uses
`phase03b_config.json`, whose role selection is inventory-driven. Phase 3a
outputs are written to `output/cityforge/building_rule_kit_v1/phase03/`; Phase
3b writes separate products under `phase03b/`, including
`selection_report.json`:

- `wall_mount_evidence.json`
- `facade_profiles.json`
- `mount_profiles.json`
- `access_bundles.json`
- `diagnostics/`

Run with:

```powershell
python tools/cityforge/build_wall_mount_profiles.py --config configs/kits/xfa_sky_nord_house/phase03_config.json
python tools/cityforge/render_profile_diagnostics.py --config configs/kits/xfa_sky_nord_house/phase03_config.json
python tools/cityforge/build_wall_mount_profiles.py --config configs/kits/xfa_sky_nord_house/phase03b_config.json
python tools/cityforge/render_profile_diagnostics.py --config configs/kits/xfa_sky_nord_house/phase03b_config.json
```

## Invariants

- Source roles come from the Phase 2 inventory, never filename substring
  matching.
- Phase 3b selects only unambiguous configured roles; doors, walls, decorations,
  fences, and mixed-role rows are recorded as skipped rather than classified.
- The Phase 3b config raises the curved-log tilt allowance and applies an
  explicit body-start cutoff to remove tall roof/rake triangles that merely
  cross the wall band. Its witness bound tolerance is zero, so dots outside a
  finite facade are omitted rather than visually floating.
- Wall-band fractions are anchored at Phase 2 robust body-bottom evidence so
  buried foundation skirts do not define the wall band.
- Coplanar disconnected panels merge into one facade; stacked near-vertical
  log strips may merge across the configured vertical gap. Near-offset
  parallel panels must also be within `merge_horizontal_gap_gu`; exact panels
  are subject to the same horizontal-gap bound so separate wings cannot be
  convex-hulled across empty intervals.
- Mount `n` is signed toward the measured visible/front face. The contact
  plane is the opposite side of that frame. Empty/open backs therefore cannot
  be labeled front merely because they are the `+axis` side. Back-plane
  contacts are polygons when they have area and measured intervals when they
  collapse to a point or line; no contact surface is invented.
- `measurement.mount_normal_axis_overrides` is the only axis exception to the
  thinnest-horizontal-bbox default. It is a normalized model-key map whose
  values are `x` or `y`; the signed face-area rule still chooses the front
  sign. This is required for architectural side-vs-end orientation cases such
  as the Phase 3b dormer and stair diagnostics.
- Witness occupancy is accepted only when the projected attachment lies within
  the measured finite facade `u/z` extent (plus configured tolerance); an
  infinite-plane nearest match cannot create a topdown dot in empty space.
- Ordinary access bundles require both door and frame; frameless observations
  remain explicitly ineligible.
- Shell topdown colors are trace IDs only: each thick line is one finite
  measured facade, its arrow is the outward normal, its `f###` label is the
  profile ID, its `z` label is the vertical interval, and white dots are
  observed attachment witnesses. The overlay sheets reuse those colors for the
  corresponding measured facade polygons.
- Phase 3b diagnostics render only the configured `diagnostics.shells` and
  `diagnostics.attachments` subset; the Phase 3b config hides the Phase 2
  convex-hull context outline because it can cross an L-notch. A broad output
  is not accepted without native-resolution inspection of that targeted set.
