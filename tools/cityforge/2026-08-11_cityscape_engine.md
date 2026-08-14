# Cityforge T1.3 hard landscape engine — 2026-08-11

**Status:** active synthetic-proof tool family; no production city design or
ESP is produced by this stage.

## Entry points

```text
python tools/cityforge/build_cityscape_fixture.py --workspace-root F:/ProcGenWorkspace
python tools/cityforge/build_city_landscape.py --workspace-root F:/ProcGenWorkspace \
  --output-dir F:/ProcGenWorkspace/output/cityforge/phase1/t1_3_cityscape_fixture
```

The fixture wrapper removes/recreates only its generated T1.3 output directory,
runs two clean builds at the same path, compares every recursive file hash, and
writes the canonical output only after the comparison passes.  The one-build
wrapper is useful for an explicitly supplied output path.  Neither command
writes `tamriel.esm`, PTR/TR/SHOTN/PC/Tamriel_Data files, or an ESP.

## Pipeline position

```text
accepted T1.1 plan + T1.2 planned placement
    -> cityscape_field (real base LAND + effective remap VTEX)
    -> cityscape_edits (intentional / exact T1.2 pads / road grades)
    -> planned_terrain_field.npz -> T1.2 planned pass
    -> T1.2 final solver re-seat against exact final_terrain_field.npz
    -> cityscape_vtex (roads, lots, zone hash mix, water gate)
    -> cityscape_vnml (real-source root calibration + edited-cell normals)
    -> cityscape_output (local LTEX + tes3json LAND records)
    -> validation.json / manifest.json / diagnostics
```

`src/procgen/cityscape.py` is orchestration only.  The focused engines are:

| Module | Responsibility |
|---|---|
| `cityscape_field.py` | Loads the 49 target cells and an 8-way one-cell source ring; checks exact seams; stitches/splits float64 GU; writes deterministic field NPZ/metadata. Base non-VTEX payload remains owned by `tamriel.esm`; effective VTEX remains owned by the accepted remap ESP. |
| `cityscape_edits.py` | Strict finite geometry, link, basin, immutable-border, ordered-composition, auto-pad, road-grade, one-time THU quantization, and no-clipping gates. Supported primitives are `flatten_shelf`, `mound`, `terrace`, `cut`, `auto_pad`, and `road_grade`. |
| `cityscape_vnml.py` | Partitions strict interiors, shared internal edges, and outer target edges; calibrates the identity root on interiors; tests independent per-cell one-sided and nearest-interior clamped-central source hypotheses; gates boundary p95/root residuals; and separately proves all 84 stitched production shared edges are exact. Production edited normals remain stitched central differences with neighbour context. |
| `cityscape_vtex.py` | Explicit palette raw/index/id/path assignments; deterministic low-frequency SHA-256 tile allocation; road/lot/zone/margin priority; raw-1 protection; complete output-owned LTEX table. |
| `cityscape_output.py` | Masterless tes3conv JSON assembly, exact base payload copying, OpenMW-normalized-to-TES3 VTEX transposition, `tes3json.validate`, and decoded output audit. |

## Source and assignment contracts

The canonical fixture uses:

* `tamriel.esm` for source heights, VHGT padding, VNML, VCLR, WNAM, and LAND
  flags;
* `output/falkreath_landscape_texture_remap.esp` for the effective 49-cell
  normalized VTEX view and its live local LTEX table;
* `region_palette.json` plus `regionpalette.live_remap_ltex_table()` /
  `planned_vs_live_remap_check()` for dispatch-5 identity verification.

The output assignments are explicit and never derived from a surface ordinal:

| Surface | raw VTEX | local LTEX index | ID |
|---|---:|---:|---|
| `water_edge_sand` | 1 | 0 | `Sand` |
| `base` | 33 | 32 | `T_Sky_TerrGrassRE_01` |
| `road` | 78 | 77 | `T_Hr_TerrRoadOH_01` |
| `settlement_grass_dirt` | 142 | 141 | `T_Sky_TerrGrassDirtRE_01` |
| `settlement_cobble` | 144 | 143 | `T_Nor_Set_TxCobbleStone_01` |
| `settlement_dirt` | 241 | 240 | `T_Sky_TerrDirtRE_01` |

Raw 1 is protected source water/lakebed and is never a road result; only an
explicit dock/basin feature may make a water tile paintable, and it still may
not be a road.  Raw 78 is the road identity and is checked separately.  A source raw value that remains
in an emitted LAND (for example an unpainted pine raw 92 in a smaller declared
support) must also resolve to a local LTEX record; the output assembly refuses
an incomplete table.

## Terrain-edit behavior

Plan terrain edits are applied in declared order.  T1.2 pad requests are
converted one-for-one into `auto_pad` edits: exact transformed footprint hull,
exact 256 GU envelope, request target/falloff/max cut/fill, and no retarget or
widening.  Regrade roads are added only for roads explicitly marked
`grade_policy: regrade` and are sorted by edit id after the exact T1.2 pad
requests.

All shape plus falloff bounds must be strictly inside the 0…57,344 GU field;
touching the immutable outer vertex ring is rejected.  A below-zero result is
only legal for an authorized dock/basin link.  Before final serialization every
TES3 VHGT row-start and horizontal delta is checked against ±127 THU, and
ordinary adjacent x/y differences are reported too.  An illegal edit returns
`edit_too_steep` with measured location and a required-falloff estimate; no
value is clamped to make the record fit.

## Outputs

The canonical fixture is labelled `synthetic_not_a_falkreath_design` in every
machine-readable proof product.  Important files are:

* `planned_terrain_field.npz` + metadata and `final_terrain_field.npz` + metadata — 449×449
  float64-GU fields at 128 GU spacing, with exact source-border preservation;
* `land_records.json` — top-level tes3conv JSON array containing a masterless
  Header (`masters: []`), local LTEX records, and all 49 LAND records; no ESP;
* `land_edits.json` — height/VTEX ledgers, class grids, raw assignments, local
  LTEX table, field hash, and source-support evidence;
* `t1_2_planned_reseat/` and `t1_2_planned_integration.json` — T1.2's planned
  pass consumed the generated planned NPZ and reproduced the accepted
  provisional request;
* `t1_2_final_reseat/` and `t1_2_final_reseat_integration.json` — exact final
  NPZ/metadata consumed by the T1.2 final pass; the synthetic pad is accepted
  with zero provisional lots;
* `validation.json` — source stitch, payload reconciliation, live remap, VNML,
  THU, border, VTEX, record, re-seat, and structured diagnostic gates;
* `diagnostic_*` — height delta, slope, before/after VTEX, and paint-class images only;
  they are not city/building renders;
* `determinism.json` — two complete recursive file hash maps and equality;
* `manifest.json` — source/output provenance and explicit no-ESP statement.

## Verification

```text
python -m pytest -q tests/test_cityscape.py
python -m pytest -q tests/test_cityscape.py tests/test_regionpalette.py
python -c "import sys; sys.path.insert(0, 'src'); from procgen import tes3json; d=tes3json.read_json('output/cityforge/phase1/t1_3_cityscape_fixture/land_records.json'); assert not tes3json.validate(d)"
```

`validation.json#gates.vnml_source_convention` contains the three sample
populations, both independent local-edge hypotheses, the real worst-canary
comparison, edge-copy evidence, and every residual above 2 degrees.
`gates.vnml_final.production_shared_edges` must report all 84 target-block
shared edges exact; this is the production stitched method, not the source
boundary hypothesis.

For the optional tes3conv round trip, copy `land_records.json` to the approved
scratch directory, run `tes3conv.exe` there, and compare the JSON output; never
write the round-trip ESP to the source/output production path.  The T1.3 engine
itself does not invoke tes3conv because T1.4 owns the authoring conversion.

## Limitations

The accepted synthetic T1.2 plan intentionally fills the proof frame so the
paint gate exercises all 12,544 tiles; that is a stress fixture, not a design
recommendation.  No building mesh, scatter ref, NPC, interior, road mesh,
rendered city, or production plugin is authored by T1.3.
