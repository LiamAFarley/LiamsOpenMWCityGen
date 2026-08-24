# 2026-08-19 Falkreath height-aware wall profile tools

These tools implement the base-shell measurement stage from
`Documentation/guides/fk_kit_wall_profile_plan.md`. They use the existing NIF
import settings (`normalize_to_position: false`, `scale_correction: 0.01`) and
emit engine-local GU data.

## blender_fk_wall_profile.py

Imports the authoritative `_a` Falkreath shells in one Blender process,
evaluates triangles, samples horizontal planes at 20 GU plus detected changes
on near-vertical geometry, and writes raw section segments with source object
and triangle-normal evidence. Source triangle outlines are retained for
diagnostic rendering. It never polygonizes or writes canonical kit data.

## measure_fk_wall_profiles.py

Launches the Blender extractor, snaps section endpoints to a documented 2 GU
grid, runs Shapely `polygonize_full`, preserves components/holes and topology
diagnostics, derives normal-supported candidate edges, and merges adjacent
sections using the 8 GU geometric tolerance. It writes
`configs/kits/falkreath/wall_profiles.json` and one source-mesh diagnostic PNG
per shell under `output/cityforge/falkreath_wall_profiles/`. Profiles are
intentionally `needs_review`; this stage does not author semantic facade
bindings, variant aliases, or generator placement.

Example:

```powershell
python tools/cityforge/measure_fk_wall_profiles.py
```

The raw extraction remains at `output/falkreath_wall_sections.raw.json` for
review. Existing `blender_fk_wall_slice.py` and `blender_fk_l_wings.py` are
not deprecated until the profile diagnostics and later migration gates are
accepted.
