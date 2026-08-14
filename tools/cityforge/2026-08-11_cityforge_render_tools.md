# Cityforge T1.5 render tools — 2026-08-11

## Purpose

This entry documents the visual Cityforge Dispatch 6 T1.5 tools. They are a
synthetic proof harness, not a Falkreath authoring tool. The pipeline position
is:

```text
accepted T1.2 placement + accepted T1.3 landscape
    -> src/procgen/cityrender.py (scene contract)
    -> tools/cityforge/render_city.py (CLI/worker launcher)
    -> Blender worker (real NIF/LTEX import and PNG renders)
    -> render_audit.json + blender_worker_audit.json
    -> host fail-closed validation
```

The canonical scene is explicitly labelled `SYNTHETIC RENDER FIXTURE - NOT A
FALKREATH DESIGN`. It contains no real city plan and never writes a production
ESP. The only plugin made during a run is a temporary, labelled,
masterless render-only scratch copy used to exercise the tes3conv import path.

## Tools

### `tools/cityforge/render_city.py`

CLI entry point. It accepts `--workspace-root`, `--output-dir`, `--blender`,
`--synthetic`, and the standard `--help` flag. It creates the scratch plugin
under the output directory from the accepted T1.3 `land_records.json`, invokes
tes3conv from a scratch working directory, writes the deterministic scene JSON,
launches Blender in background mode, and validates the worker audit and all
PNG files. `--synthetic` is mandatory for the fixture mode and refuses paths
that look like real Falkreath plan/production output.

### `src/procgen/cityrender.py`

Host-side scene builder and final audit. It resolves the exact accepted input
products, validates the 19 final T1.2 members and 49 T1.3 cells, checks the
masterless scratch plugin, resolves seven real mesh assets and seven real
terrain textures, serializes the exact engine-transform matrices, declares the
11 fixed base views plus one focused detail view per synthetic lot, and performs
the post-Blender audit. It also creates 41 ordered door/road camera candidates
per street/detail view, including nine far-side escape candidates, and selects
only candidates with final-terrain LOS (ground, lower door threshold, door,
facade) and finite-edge clearance. The imported worker samples the readable
door/facade band and fails candidates above 0.05 terrain-occluded fraction or
across the readable door/facade band. Lower foundation visibility is not gated;
T1.2's measured source burial remains authoritative.
It is deterministic: all scene JSON is canonicalized before hashing and no
RNG or wall-clock value is used in the build identity.

### `tools/cityforge/blender_render_city.py`

Blender worker. It imports every NIF by its real path with
`normalize_to_position=false`, records the hierarchy roots, and checks the
serialized matrix against each expected T1.2 matrix before rendering. It builds
the exact dense T1.3 field with per-face `VHGT` heights, `VNML`-calibrated
normals, and `VTEX`-resolved LTEX materials; each face gets a real tile UV.
The water mesh is the clipped exterior polygon at scene `z=0`, not a
rectangular plane. Blender Eevee is used with the resolved source images,
neutral exposure `1.75`, and a small texture emission term so asset appearance
remains readable while real
 lighting/shadows are retained; no proxy geometry, flat-ground fallback, or
 placeholder material is permitted. It repeats the selected camera's LOS/edge
 gate after import, then writes one worker audit and 18 PNGs.

## Inputs

The default paths are pinned to the accepted products:

* `output/cityforge/phase1/t1_2_placement_fixture/t1_1_plan/city_plan.json`
* `output/cityforge/phase1/t1_2_placement_fixture/city_placement.json`
* `output/cityforge/phase1/t1_3_landscape_fixture/cityscape_field.json`
* `output/cityforge/phase1/t1_3_landscape_fixture/t1_2_final_reseat/land_records.json`
* `output/cityforge/phase1/t1_3_landscape_fixture/t1_2_final_reseat/ltex_records.json`
* `output/cityforge/phase1/t1_3_landscape_fixture/t1_2_final_reseat/terrain_vtex.json`

The accepted `configs/procgen.json` `asset_roots` are the only read-only asset
roots. Source mod files remain untouched. Mesh paths are checked against the
T1.2 D-STAMP inventory and texture paths against the accepted T1.3 LTEX table.

## Outputs

For a canonical run, `output_dir` contains:

* `render_scene.json` — labelled scene contract and build hash;
* `synthetic_render_only_masterless.esp` — scratch-only TES3 plugin copy;
* `blender_worker_audit.json` — imports, texture resolution, matrices, terrain,
  water triangle count, and rendered-view counts;
* `render_audit.json` — host summary, PNG dimensions, byte sizes, SHA-256,
  tonal/nonblank/clip statistics, and determinism contract;
* 18 PNGs: one overview, four oblique terrain views, six base street views,
  and seven focused single-lot door-height detail views.

## Reproducible commands

From `F:\ProcGenWorkspace`:

```powershell
$env:PYTHONPATH = 'src'
python tools/cityforge/render_city.py `
  --output-dir output/cityforge/phase1/t1_5_render_fixture `
  --synthetic
```

The command is intentionally run from the workspace only for orchestration;
tes3conv itself is invoked from the generated scratch directory because it
writes relative output paths. A successful run ends with
`[cityforge-worker] ALL_DONE` and a host message naming the 18 PNGs. Any
essential stage failure is a hard failure; there is no degraded fallback.

## Verification signals

Required zeroes are `building_texture_missing_count`,
`terrain_texture_missing_count`, `matrix_mismatch_count`,
`proxy_geometry_count`, `flat_terrain_fallback_count`, and
`unresolved_model_count`. Required positive counts are 19 refs, 49 terrain
cells, 18 rendered views, 13 LOS-gated camera views with zero failures, seven
unique models, seven resolved nonzero LTEX textures, and a positive clipped-
water triangle count. Every view must pass the PNG size/nonblank/clip audit;
the edge intrusion failed-view count must be zero,
every selected camera must clear door/threshold/facade targets by at least 1.00
scene units (ground-interface target: 0.75) and the
finite edge by at least 12.0 scene units, must have zero final-field perimeter
samples within 0.04 NDC of the subject bounds, zero readable door/facade-band
failures above 0.05 terrain-occluded fraction, zero lower-envelope failures
above 0.05 across the readable door/facade band, and every focused lot must pass
its content-fit test.

The focused Python tests are in `tests/test_cityrender.py`. The final visual
gate is manual image inspection of every PNG; the dated run report records the
actual per-image observations, hashes, dimensions, and any blockers.

## Status

This replaces the earlier exploratory single-camera render attempts. The
earlier attempts were diagnostic only and are not canonical products. The
canonical tool family must stay synthetic until the lead explicitly accepts a
real Falkreath render plan.
