# Building Rule Kit Phase 4 Tools

Date: 2026-08-23

## Purpose

These tools measure roof planes and observed dormer-to-roof relations from
current Sky xFa NIFs and current site stamp libraries. They are a
measurement-only Phase 4 pilot: they do not place dormers on new roofs, add
windows, compile a palette, compose houses, or modify source data.

## Entry points

- `tools/cityforge/build_roof_dormer_profiles.py` — reads the JSON kit config,
  validates the exact current inventory/stamp selections, launches one Blender
  evidence pass per distinct mesh, and writes the Phase 4 JSON products.
- `tools/cityforge/blender_roof_dormer_evidence.py` — imports fresh NIFs and
  emits evaluated non-degenerate triangles, normals, centroids, bounds, and
  resolved paths. It makes no grouping or eligibility decisions.
- `tools/cityforge/render_roof_dormer_diagnostics.py` — builds diagnostic jobs,
  triangulates measured polygon regions without convex-hulling, runs the
  Blender overlay worker, stitches native-resolution shell sheets and
  source/reconstruction relation pairs, and writes the flat manifest.
- `tools/cityforge/blender_roof_overlay.py` — imports the real NIF scene and
  draws measured roof fills, inset/boundary lines, and canonical frame axes in
  Blender scene units.
- `src/procgen/building_gen/roofs.py` — host-side candidate filtering,
  supported connected-component grouping, plane fitting, polygon/holes,
  canonical roof frames, contact evidence, and observed dormer round trips.

## Configuration and outputs

The pilot is configured by
`configs/kits/xfa_sky_nord_house/phase04_config.json`. It selects five real
shell meshes (`house_01_a`, `house_04_a`, `house_05_c`, `house_07_a`, and
`house_07_b`) and four explicit source-stamp dormer relations. Thresholds,
scene-unit conversion, inset size, frame lengths, and diagnostic resolution
are JSON values; no model-name role logic is embedded in the tools.

The configured output directory is
`output/cityforge/building_rule_kit_v1/phase04/` and contains:

- `roof_evidence.json` — raw evaluated triangle evidence;
- `roof_profiles.json` — eligible/ineligible patches, fitted planes, frames,
  polygon pieces, holes, boundary classifications, support, and residuals;
- `dormer_relations.json` — explicit shell/dormer relations, contact and
  clearance evidence, authored scale, roof-frame coordinates, and round trips;
- `selection_report.json` — selected shells/evidence meshes, selected
  relations, and explicit skips;
- `diagnostics/` — six views per shell, one shell sheet per shell, six source
  views, six reconstruction views, and six stitched pair views per relation,
  plus `manifest.json`.

## Invariants and visual gate

- Blender only imports fresh configured NIFs and evaluates/renders geometry;
  Python owns selection, grouping, relation matching, and eligibility.
- Roof components retain disconnected polygon pieces and holes. Convex hulls
  are not used to bridge empty roof space.
- The eave tangent comes from the longest measured near-horizontal boundary
  edge; unresolved eaves are explicitly labeled `unresolved`.
- Dormer evidence requires exact configured shell/dormer member IDs and an
  explicit stamp attachment edge. Child-window witnesses are direct dormer
  contact edges only.
- Diagnostic `u/v` tripods are anchored at a representative point inside the
  measured usable region and their in-plane lengths are clamped to that region.
  The normal leg is short and starts on the measured plane. Blender's varying
  timestamp/render-time PNG metadata is removed without changing decoded
  pixels so repeated worker renders are byte-stable.
- The Phase 4 visual gate is native-resolution inspection against imported NIF
  geometry. Floating, bridged, wall/sky/rake-covered, out-of-region, or
  detached-frame overlays reject the pilot; dormer source/reconstruction
  transform or contact disagreement also rejects it.

Run the pilot and diagnostics with:

```text
python tools/cityforge/build_roof_dormer_profiles.py --config configs/kits/xfa_sky_nord_house/phase04_config.json
python tools/cityforge/render_roof_dormer_diagnostics.py --config configs/kits/xfa_sky_nord_house/phase04_config.json
```

Phase 8 owns actual roof attachment and dormer transfer. Phase 4 products are
evidence inputs, not a composer or TownLayout implementation.
