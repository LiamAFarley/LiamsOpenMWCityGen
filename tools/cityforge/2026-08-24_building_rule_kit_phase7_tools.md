# Building Rule Kit Phase 7 Tools

Date: 2026-08-24

## Purpose

`tools/cityforge/render_building_composer_gate.py` is the first visual gate for
the generated building stamps emitted by Phase 6. It selects exact accepted
request IDs from the real Phase 6 result documents and renders their members
with the existing `tools/mesh_thumbs.py` Blender bridge. It is a stamp assembly
diagnostic, not a terrain or in-game placement renderer.

## Command

```text
python tools/cityforge/render_building_composer_gate.py \
  --config configs/kits/xfa_sky_nord_house/phase07_render_config.json
```

The driver requires a fresh output directory. All render settings, case IDs,
filenames, and the diagnostic ground-plane presentation are in the JSON
config. Source NIFs remain under configured read-only data roots.

## Output Contract

The pilot writes three flat native-resolution 2x3 sheets under
`output/cityforge/building_rule_kit_v1/phase07/`:

- `observed_c131_replay_sheet_2x3.png`
- `single_house_01_primary_sheet_2x3.png`
- `single_house_01_secondary_access_sheet_2x3.png`

Each sheet has a matching stamp copy, scene JSON, and mesh-render audit.
`manifest.json` records the Phase 6 request ID, source result document, member
and door counts, expected/actual imported piece counts, and all evidence paths.
Any missing mesh, Blender failure, piece-count mismatch, or excluded building
piece aborts the run.

## Visual Boundary

Inspect every full-resolution sheet before expanding the set. Reject floating,
buried, missing, duplicated, overlapping, cropped, or implausibly oriented
members and doors. The ground plane is only a diagnostic aid. This tool does
not validate exact final terrain seating, road connections, TownLayout
integration, ESP authoring, or in-game behavior.
