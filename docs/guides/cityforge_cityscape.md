# Cityforge T1.3 Cityscape Landscape Engine

**Status:** active synthetic proof stage
**Date:** 2026-08-11
**Pipeline stage:** Cityforge Dispatch 6, T1.3
**Canonical scope:** `synthetic_not_a_falkreath_design`

This guide documents the implementation that actually runs. It is the
authoring boundary between the accepted T1.1/T1.2 host-side products and the
future T1.4 plugin author. T1.3 produces terrain fields, validation evidence,
and tes3conv-compatible JSON; it does not produce a production ESP or a real
Falkreath city.

## 1. Pipeline position

```text
accepted T1.1 plan + validation
accepted T1.2 planned placement + pad requests
        |
        v
real tamriel.esm LAND + accepted remap ESP
        |
        v
cityscape_field: source/effective load and 449x449 stitch
        |
        v
cityscape_edits: pass 1 intentional edits
        |
        v
planned_terrain_field.npz
        |
        v
T1.2 planned replay against the generated planned field
        |
        v
cityscape_edits: exact T1.2 auto pads + selected road grades
        |
        v
final_terrain_field.npz
        |
        +--> T1.2 final replay/re-seat
        |
        +--> cityscape_vtex: final effective VTEX painting
        |
        +--> cityscape_vnml: source convention gate and edited normals
        |
        v
cityscape_output: local LTEX + LAND tes3conv JSON
        |
        v
tes3json validation, decoded audits, diagnostics, manifest, hashes
```

`src/procgen/cityscape.py` is the orchestrator. The focused modules own the
data-format responsibilities rather than placing all logic in the driver.
The actual high-level order is: source load, source gates, source VNML
calibration, planned edits, T1.2 planned replay, final edits, T1.2 final
replay, VTEX paint, VNML payload generation, LAND/LTEX assembly, and output
audits.

## 2. Modules and responsibilities

| Module | Actual responsibility and important entry points |
|---|---|
| `src/procgen/cityscape.py` | `CityscapePaths`, `default_paths()`, and `build_cityscape()`. Wires accepted inputs, runs all hard gates, writes products, and raises hard failures. It also emits synthetic edit/image diagnostics and the validation/manifest documents. |
| `src/procgen/cityscape_field.py` | `load_target_block()`, `stitch_heights()`, `split_field()`, `rejoin_field()`, `write_field_npz()`, and `field_metadata()`. Reads source LAND plus a perimeter ring, reconciles source/effective payloads, and owns the joint field. |
| `src/procgen/cityscape_edits.py` | `validate_edit_request()`, `apply_edit()`, `compose_edits()`, and final THU encoding reports. Implements all six analytic edit kinds, ordered composition, support ledgers, bounds/water/link gates, and no-clipping encoding checks. |
| `src/procgen/cityscape_vnml.py` | `validate_source_convention()`, `source_local_one_sided_normals()`, `source_local_clamped_normals()`, `analytic_normals_for_cell()`, `compute_cell_normals()`, and `production_shared_edge_audit()`. Separates source parity hypotheses from production stitched normals. |
| `src/procgen/cityscape_vtex.py` | `load_surface_assignments()` and `paint_vtex()`. Applies explicit raw/index assignments, source road/water protections, deterministic zone quotas, margin blending, and output-owned LTEX closure. |
| `src/procgen/cityscape_output.py` | `assemble_land_records()` and `build_land_edits_document()`. Builds masterless tes3conv JSON, copies applicable source payloads, transposes VTEX for TES3 serialization, and audits decoded output. |
| `tools/cityforge/build_cityscape_fixture.py` | Cleans only the canonical generated directory, runs two clean builds, compares recursive hashes, and installs the synthetic proof products after the comparison passes. |
| `tools/cityforge/build_city_landscape.py` | One-build CLI for an explicit output directory and explicit input overrides. It emits `FAILURE: cityscape ...` on hard failure. |

## 3. Inputs and immutability

The default CLI paths are defined by `cityscape.default_paths()`:

| Input | Default path | Authority |
|---|---|---|
| Site survey | `output/cityforge/sites/falkreath_v1/site_survey.json` | accepted D-SITE survey and water mask |
| Region palette | `output/cityforge/briefs/falkreath_v1/region_palette.json` | dispatch-5 semantic/raw/LTEX contract |
| Synthetic T1.2 plan | `output/cityforge/phase1/t1_2_placement_fixture/synthetic_not_a_falkreath_design.city_plan.json` | accepted T1.1/T1.2 proof plan |
| T1.1 validation | `output/cityforge/phase1/t1_2_placement_fixture/synthetic_not_a_falkreath_design.validation.json` | zero-error plan gate |
| Accepted T1.2 placement | `.../t1_2_placement_fixture/city_placement.json` | planned-pass placement reference |
| Accepted T1.2 pad requests | `.../t1_2_placement_fixture/land_edit_requests.json` | exact auto-pad contract |
| Base LAND | `tamriel.esm` | authoritative heights and non-VTEX payload |
| Effective remap | `output/falkreath_landscape_texture_remap.esp` | effective normalized VTEX and local LTEX |
| Kit brief | `output/cityforge/briefs/falkreath_v1/kit_brief.json` | T1.2 solver input |
| Stamp libraries | `output/cityforge/stamps/karthgad_nord_v1.json`, `markarth_side_stone_v1.json` | T1.2 replay input |
| Road centerlines | `output/mapdata/roads/tamriel_aligned_centerlines_v1/tamriel_aligned_centerlines_v1.json` | aligned consumer road geometry (loaded through `src/procgen/aligned_roads.py`; direct LAND/VTEX is the in-game occupancy authority) |

The base ESM, remap ESP, accepted survey, palette, plan, validation, T1.2
products, kit, stamps, and centerlines are read-only. Generated products go
only to the explicit output directory. A scratch tes3conv ESP used for
verification belongs under the approved temporary directory and is not a
T1.3 product.

## 4. Joint terrain field and passes

### 4.1 Field geometry

The target is exactly the accepted 7x7 block, 49 cells. Each TES3 LAND cell
has a 65x65 vertex grid, but adjacent cells share one edge. The field therefore
has:

* shape `449x449` vertices;
* float64 game-unit values;
* 128 GU vertex spacing;
* extent `57,344x57,344 GU`;
* one-cell source context on every side for border normal sampling.

The source block is stitched before editing. Every shared edge is compared
exactly, split/rejoined source fields are compared exactly, and the outer
1,792-vertex ring is immutable. Values outside an edit support remain exact
source values. The field is serialized as timestamp-free NPZ with a metadata
sidecar containing pass, shape, spacing, origin, extent, provenance, and
content hashes.

### 4.2 Planned pass

Pass 1 applies plan-declared `terrain_edits` in declaration order. The actual
synthetic T1.2 plan has no real city terrain design; the fixture's diagnostic
ledger independently exercises every primitive. The planned field is written
to `planned_terrain_field.npz` and metadata before T1.2 is run against it.

The generated T1.2 planned request is compared with the accepted pad request
after canonical six-decimal JSON normalization. This avoids treating Python
tuple/list representation or harmless pre-serialization floating noise as
contract drift while still rejecting a changed hull, target, margin, falloff,
or limit.

### 4.3 Final pass

The final edit list is the exact generated T1.2 `auto_pad` request followed by
deterministically sorted plan roads whose `grade_policy` is `regrade`.
Auto-pad parameters are not widened or retargeted. The final field is written
to `final_terrain_field.npz` and metadata, then T1.2 is run again with
`terrain_pass="final"` and the planned placement reference. A final pad is
trusted only when the solver reports `reference_verified`, the final field
hash matches, and provisional count is zero.

## 5. Edit contracts and failure modes

All edit parameters must be finite. Geometry plus falloff must lie strictly
inside the 0…57,344 GU field and may not touch the immutable outer ring. Every
edit has `edit_id`, `kind`, and `linked_to`; links must resolve to plan lots,
roads, or explicitly authorized water features.

| Kind | Actual parameters and behavior |
|---|---|
| `flatten_shelf` | Polygon, `target_height_gu`, `falloff_gu`; smoothstep blend from plateau to the incoming field. |
| `mound` | `center`, `radius_gu`, `height_delta_gu`, `falloff_gu`; smoothstep radial raise/lower. Below-zero results require an authorized dock/basin link. |
| `terrace` | Ordered shelf polygons with per-shelf target heights and falloff; each shelf is validated and ledgered. |
| `cut` | Polyline, `width_gu`, `depth_gu`, `falloff_gu`; smoothstep corridor lowering. |
| `auto_pad` | Exact T1.2 footprint hull/pad polygon, 256 GU margin, target, falloff, measured/max cut-fill, and lot link. `margin_gu` other than 256 is rejected. |
| `road_grade` | Road polyline corridor, width/falloff, maximum grade, and maximum cut/fill; deterministic bounded grade operation. |

The editor records changed vertices, support masks, overlap/order provenance,
per-vertex delta statistics, and pre/post quantization error. It checks both
TES3 row-start/horizontal VHGT encoding deltas and ordinary adjacent x/y
differences. Every encoded delta must fit signed `-127…+127 THU` (`±1,016 GU`).
Illegal edits return structured `edit_too_steep` evidence including a measured
location and minimum legal falloff. No illegal value is clipped.

Synthetic diagnostics cover accepted primitives and structured failures for
out-of-bounds geometry, illegal pad margin, unknown link, unintentional basin,

## 6. VNML: source convention versus production convention

### 6.1 Production method

Production edited normals are computed only for height-edited cells from the
final joint field. The method is central differences at 128 GU spacing,
normalized from `(-dzdx, -dzdy, 1)`, with one-cell real source context at the
outer target border. It is never replaced by a source seam imitation.
Normals are quantized once using signed int8 `round(component * 127)` without
clipping; zero vectors, non-finite values, and `-128` are hard failures.

`production_shared_edge_audit()` computes the production method for all target
cells and compares every shared edge. The canonical result checks all 84
internal target-block edges with zero mismatches.

### 6.2 Source parity populations

Source parity is partitioned per source LAND record, so duplicated per-cell
edge samples remain visible:

1. **Strict interior:** local x/y both 1…63. This selects the axis/sign root.
2. **Shared internal boundary:** local x/y edge samples excluding outer target
   edges.
3. **Outer target boundary:** samples on the outermost target block edges.

All 48 axis permutations/sign combinations are ranked independently. The
identity root `[0,1,2]` / `[+,+,+]` is required for the interior, both boundary
populations, the combined boundary population, and the aggregate stitched
population. The fixed source-parity tolerance is 2 degrees p95; a root with a
systematic mean defect or a population over tolerance fails closed.

### 6.3 Independent boundary hypotheses

Two source-only hypotheses are measured without changing production:

* `source_local_one_sided_normals()` uses central differences in the interior
  and forward/backward differences at each local edge. It does not read a
  neighbouring LAND.
* `source_local_clamped_normals()` clamps edge coordinates to the nearest valid
  interior coordinate, then evaluates the central stencil there. This matches
  the measured source behavior: source edge VNML frequently reuses the
  nearest interior triplet.

The former worst canary demonstrates the distinction:

| Predictor | Angle at `[-95,-10] [0,33]` |
|---|---:|
| Production stitched | 15.287237° |
| One-sided local | 9.975337° |
| Clamped-central local | 0.241935° |

Canonical source metrics are:

| Population/method | Samples | Mean | P95 | Max | >2° count |
|---|---:|---:|---:|---:|---:|
| Strict interior, stitched central | 194,481 | 0.539589° | 1.223367° | 2.480546° | 12 |
| Shared boundary, clamped central | 10,728 | 0.570259° | 1.303244° | 6.589809° | 92 |
| Outer boundary, clamped central | 1,816 | 0.608550° | 1.386155° | 5.678923° | 27 |
| Combined boundaries, clamped central | 12,544 | 0.575803° | 1.314568° | 6.589809° | 119 |

All 131 over-2-degree residuals are retained in
`validation.json#gates.vnml_source_convention.mismatches_over_2_deg`, with
population, cell, local vertex, angle, predicted quantized normal, source
quantized normal, and component error. They are source-only residual evidence;
the boundary systematic residual is removed, the identity root remains
strongly separated, and production seams remain stitched.

## 7. VTEX painting and assignments

VTEX grids are 16x16 normalized OpenMW-order tiles per cell. Painting is a pure
function of plan hash, zone id, and tile coordinates; it does not consume
global RNG state.

Priority is:

1. Plan and solver road corridors, plus protected source raw-78 road tiles →
   raw 78.
2. Placed lot footprint tiles → settlement dirt raw 241.
3. Declared texture-zone classes → deterministic low-frequency SHA-256 ranked
   quotas.
4. Margin transitions → grass-dirt where that class is declared.

Existing source raw-78 tiles cannot be reclassified by a full-support district
zone. Raw 1 lakebed/sand is preserved and never becomes a road. An explicit
dock/basin feature may make its tile paintable, but it remains ineligible for
road painting. Source raw 92 pine and raw 33 grass remain valid source classes.

The dispatch-5 assignments are explicit:

| Surface | Raw VTEX | Local LTEX index | Record ID |
|---|---:|---:|---|
| `water_edge_sand` | 1 | 0 | `Sand` |
| `base` | 33 | 32 | `T_Sky_TerrGrassRE_01` |
| `road` | 78 | 77 | `T_Hr_TerrRoadOH_01` |
| `pine` source | 92 | 91 | `T_Sky_TerrPine_01` |
| `settlement_grass_dirt` | 142 | 141 | `T_Sky_TerrGrassDirtRE_01` |
| `settlement_cobble` | 144 | 143 | `T_Nor_Set_TxCobbleStone_01` |
| `settlement_dirt` | 241 | 240 | `T_Sky_TerrDirtRE_01` |

The painter records support masks, source/painted grid hashes, priority counts,
source-road/water counts, zone target weights, realized counts/fractions, and
the complete output local-LTEX table. Every positive output raw value must
resolve to exactly one local LTEX index (`raw - 1`).

## 8. LAND/LTEX output contract

`land_records.json` is a top-level tes3conv JSON array, not an ESP. It contains:

* one Header with `masters: []`;
* one output-owned LTEX record for every positive raw VTEX value emitted;
* one Landscape/LAND record for each of the 49 target cells.

The assembler hands final signed THU heights to `tes3json.build_land()` with
`heights_in_thu=True`. Source VCLR, WNAM, DATA flags/unknown bits, and VHGT
padding are copied. Unedited cells retain source heights, VNML, and all other
payload fields except declared VTEX painting. Height-edited cells may differ
in VHGT/VNML and declared VTEX only.

The assembler converts OpenMW-normalized VTEX order back to TES3 serialized
macro-block order. It validates the JSON before writing, decodes the emitted
LAND fields again, checks intended heights/normals/VTEX/colors/world-map/data,
checks local LTEX completeness, and records the result in `validation.json`.
T1.4 owns the actual tes3conv/production authoring stage.

## 9. Products and synthetic fixture

Canonical directory:

`output/cityforge/phase1/t1_3_cityscape_fixture/`

Important products:

| Product | Purpose |
|---|---|
| `planned_terrain_field.npz` + metadata | Pass-1 field handed to the T1.2 planned replay. |
| `final_terrain_field.npz` + metadata | Pass-2 field handed to the T1.2 final re-seat and later render stages. |
| `land_edits.json` | Height edit, encoding, VTEX, support, source, and LTEX ledgers. |
| `land_records.json` | Exact masterless LAND/LTEX JSON hand-off to T1.4. |
| `validation.json` | Source stitch, payload, VNML populations/hypotheses/residuals, THU, VTEX, record, re-seat, and synthetic diagnostics. |
| `manifest.json` | Source/output hashes and explicit no-ESP provenance. |
| `determinism.json` | Recursive two-build hash maps and equality result. |
| `diagnostic_height_delta.png` | Height edit magnitude diagnostic. |
| `diagnostic_final_slope.png` | Final terrain slope diagnostic. |
| `diagnostic_vtex_before.png` / `diagnostic_vtex_after.png` | Source and painted VTEX views. |
| `diagnostic_vtex_paint_classes.png` | Explicit water/road/surface class view. |
| `t1_2_planned_reseat/` | T1.2 planned products generated against the actual planned field. |
| `t1_2_final_reseat/` | T1.2 final products generated against the actual final field. |

The fixture is intentionally synthetic and banner/status labelled
`synthetic_not_a_falkreath_design`. It fills the proof frame to exercise all
12,544 VTEX tiles; this is stress-test geometry, not a city recommendation.
The observed final T1.2 result is 7 accepted, 0 provisional, 0 rejected;
the planned replay has 6 accepted and 1 provisional pad.

## 10. Commands

From the workspace root:

```powershell
python tools/cityforge/build_cityscape_fixture.py --workspace-root F:/ProcGenWorkspace
```

This is the canonical proof command. It removes/recreates only the generated
T1.3 directory, runs two clean builds, compares all recursive hashes, and
writes canonical products only after equality passes.

For one explicit output directory:

```powershell
python tools/cityforge/build_city_landscape.py `
  --workspace-root F:/ProcGenWorkspace `
  --output-dir C:/Users/LiamF/AppData/Local/Temp/opencode/cityscape_check
```

Focused checks:

```powershell
$env:PYTHONPATH = 'src'
python -m pytest -q tests/test_cityscape.py
python -m pytest -q tests/test_cityscape.py tests/test_regionpalette.py `
  tests/test_falkreath_landscape_remap.py tests/test_tes3json.py tests/test_cityplace.py
python -m compileall -q src/procgen/cityscape.py src/procgen/cityscape_vnml.py
```

The optional conversion check must use an approved scratch directory:

```powershell
tes3conv-master/tes3conv.exe land_records.json scratch/cityscape.esp -o
tes3conv-master/tes3conv.exe scratch/cityscape.esp scratch/cityscape.json -o
```

Then run `tes3json.validate()` and compare decoded LAND payloads. Never write
the scratch ESP into the production output directory or modify any source
plugin.

## 11. Failure modes

The core and CLI fail closed. The CLI prefix is:

```text
FAILURE: cityscape <reason>
```

Important hard failures include:

* missing accepted input or source LAND;
* source shared-edge/stitch/split/rejoin disagreement;
* source/effective payload or live-remap identity disagreement;
* non-finite, unlinked, out-of-bounds, basin, illegal-pad, or too-steep edit;
* any encoded VHGT delta outside signed `-127…+127 THU`;
* source VTEX changed outside declared support;
* raw 1 painted as road or raw 78 assignment disagreement;
* missing, duplicate, or mismatched local LTEX;
* source VNML root/boundary population failure;
* `VNML boundary root unresolved` when the real canary is not explained by the
  measured alternate source convention;
* any production stitched shared-edge normal mismatch;
* tes3json validation or decoded payload mismatch;
* T1.2 planned/final field hash/pass/re-seat mismatch;
* non-deterministic two-build output.

There is no clipped, fallback, partial, or reduced-fidelity success path.

## 12. Non-goals and downstream boundaries

T1.3 does not:

* design or place a real Falkreath city;
* author or copy a production ESP;
* generate STAT, DOOR, CELL, NPC, container, interior, quest, scatter, or
  building records;
* generate road networks or replace accepted road centerlines;
* run erosion, hydrology, or terrain simulation;
* edit cells outside the accepted 49-cell target block;
* render buildings or a city scene;
* claim fine triangle/AABB collision, which remains an accepted T1.2
  deferral;
* replace production stitched normals with source-local seam behavior.

T1.4 owns production tes3conv authoring and binary scan integration. T1.5 owns
terrain-backed review rendering. T1.6 owns the first real Falkreath plan and
the user review gate before any real placement.
