# Cityforge T1.5 visual render guide

## Scope and safety

T1.5 is the visual proof stage for Dispatch 6. It renders the accepted T1.2
house-only placement over the accepted T1.3 final 49-cell terrain field. It is
not the T1.6 Falkreath design gate. The fixture is labelled
`SYNTHETIC RENDER FIXTURE - NOT A FALKREATH DESIGN` in both the scene JSON and
the audit products.

The tools never edit `tamriel.esm`, PTR/TR/SHOTN/PC/Tamriel_Data files, or the
accepted T1.1/T1.2/T1.3 products. Before Blender starts, the CLI copies the
T1.3 `land_records.json` into a temporary scratch directory and round-trips it
through tes3conv to create the labelled masterless
`synthetic_render_only_masterless.esp`. The source JSON and all source plugins
remain read-only.

## Exact inputs

The host reads the accepted products below and refuses missing or hash-mismatched
inputs:

| Role | Path |
|---|---|
| T1.1 plan | `output/cityforge/phase1/t1_2_placement_fixture/t1_1_plan/city_plan.json` |
| T1.2 final placement | `output/cityforge/phase1/t1_2_placement_fixture/city_placement.json` |
| T1.3 field | `output/cityforge/phase1/t1_3_landscape_fixture/cityscape_field.json` |
| T1.3 final reseat LAND | `output/cityforge/phase1/t1_3_landscape_fixture/t1_2_final_reseat/land_records.json` |
| T1.3 final reseat LTEX | `output/cityforge/phase1/t1_3_landscape_fixture/t1_2_final_reseat/ltex_records.json` |
| T1.3 final reseat VTEX | `output/cityforge/phase1/t1_3_landscape_fixture/t1_2_final_reseat/terrain_vtex.json` |
| asset roots | `configs/procgen.json` → `asset_roots` |

No input is regenerated or “cleaned” by T1.5. In particular, the terrain
heights, normals, texture classes, pad/re-seat positions, source references,
and source rotations are consumed exactly as accepted.

## Pipeline

1. **Analysis/contract.** `src/procgen/cityrender.py` verifies the accepted
   input pins and creates the deterministic scene contract. It requires 19
   final T1.2 members, 49 final T1.3 cells, 7 unique model paths, and 7
    resolved nonzero LTEX paths. It emits the exact 11 base view specifications
    plus one focused door-height detail view for each of the 7 synthetic lots
    (18 PNGs total).
2. **Scratch TES3 path.** `tools/cityforge/render_city.py` copies the LAND
   records and invokes tes3conv from scratch CWD. The generated plugin must
   have `masters: []`; it is never copied to a production output location.
3. **Blender import.** The worker imports the real NIF hierarchies with no
   normalization. Each hierarchy root carries the exact scene-space matrix
   derived by `procgen.engine_transform`; the root matrix and serialized
   translation storage are audited before any render is accepted.
4. **Terrain.** The worker reconstructs the 449×449 field from final T1.3
   `VHGT`/`VNML`/`VTEX` records. It uses the actual `LTEX` image files and a
   per-tile UV layer. No flat ground, missing-texture color, proxy box, or
   generated substitute can pass the worker gate.
5. **Water.** The worker extracts the final field boundary and creates a
   clipped exterior polygon exactly at scene `z=0`. It rejects any rectangular
   plane and audits the positive triangle count.
6. **Views and validation.** Blender writes 18 fixed PNGs: one 3072×3072
   overview, four 2048×1536 obliques, six 1600×1000 base street perspectives,
   and seven 1600×1000 focused single-lot street/detail perspectives. Every
    street/detail camera is selected from 41 ordered door/road candidates using
    the exact final height field. Door, threshold, and facade targets require at
    least `1.00` scene-unit terrain LOS clearance; the lower building-ground
    target uses a separate `0.75` scene-unit tolerance because below-grade mesh
    may be seated into the final field. The worker also samples the imported
    NIFs across the readable door/facade band and hard-rejects any candidate
    with more than `0.05` terrain-occluded hit fraction. It deliberately does
    not gate visibility of the lower foundation envelope because T1.2 preserves
    measured source-authored burial and foundations may correctly be underground.
    Every candidate still
    requires `12.0` scene-unit camera/subject/foundation clearance from the
    finite terrain edge. The Blender worker repeats these gates after import
    and rejects any sampled final-field perimeter point projected within `0.04`
    normalized-device coordinates of the imported subject bounds.
   The host also requires each focused lot's imported bounds to remain in frame
   at a non-microscopic span and checks all PNG readability statistics.
   Neutral Eevee uses the actual image materials with the matte policy and
   exposure `1.75`; this is a readability setting only and does not recolor or
   replace terrain/building textures.

## Running the fixture

```powershell
Test-Path -LiteralPath 'F:\ProcGenWorkspace\tools\cityforge'
$env:PYTHONPATH = 'src'
python tools/cityforge/render_city.py `
  --output-dir output/cityforge/phase1/t1_5_render_fixture `
  --synthetic
```

The command must be run with Blender 4.x available either on PATH or through
`--blender <path>`. The worker is launched in background mode and prints
`[cityforge-worker] ALL_DONE` only after all 18 PNGs and its audit have been
written. The host then writes `render_audit.json`; a missing Blender executable,
missing asset, failed tes3conv round-trip, or failed visual contract is a
failure, not a reason to use a proxy render.

## Reading the audits

`render_scene.json` is the input-side contract. Its `build_hash` is the
canonical hash of the scene JSON before output paths are injected.

`blender_worker_audit.json` is the Blender-side evidence. Check:

* `counts.expected_ref_count == imported_ref_count == 19`;
* `counts.matrix_mismatch_count == 0`, with
  `max_matrix_error <= 1e-7`;
* `counts.building_texture_missing_count == 0` and
  `counts.terrain_texture_missing_count == 0`;
* `counts.terrain_cell_count == 49` and
  `counts.flat_terrain_fallback_count == 0`;
* `counts.proxy_geometry_count == 0`;
* `counts.rendered_view_count == 18`;
* `counts.terrain_los_view_count == 13` and
  `counts.terrain_los_failed_view_count == 0`;
* `counts.terrain_door_band_failed_view_count == 0`;
* `counts.terrain_edge_intrusion_failed_view_count == 0` and
  `counts.terrain_edge_intrusion_sample_count > 0`;
* `water_triangle_count > 0`.

`render_audit.json` repeats the fail-closed counters and adds per-image
dimensions, byte sizes, SHA-256, nonblank fraction, background estimate,
sample clipping, and tonal statistics. It is not a replacement for visual
inspection: all 18 PNGs must be opened and described in the run report. The
canonical synthetic fixture is generated twice by
`tools/cityforge/build_city_render_fixture.py`; the two complete directory
hash maps must be byte-identical before the first run is installed.

## Tests

```powershell
python -m pytest -q tests/test_cityrender.py
```

The tests cover host scope and view counts, the 41-candidate LOS/edge/door-band contract,
byte determinism, the accepted multi-axis and near-gimbal transform canaries,
the black/blank PNG rejection, and (when the canonical fixture exists) the
final Blender audit counters including all 13 LOS-gated camera views.

## Limitations

The synthetic final T1.2 placement currently contains yaw-only members; the
matrix audit therefore reports zero observed multi-axis/near-gimbal refs in
the rendered fixture. The explicit canary unit test still reconstructs both
accepted transform categories through the production matrix path. This is a
property of the accepted synthetic input, not a license to add an unapproved
Falkreath or production placement.
