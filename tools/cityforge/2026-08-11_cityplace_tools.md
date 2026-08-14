# 2026-08-11 — Cityforge T1.2 cityplace tools

This dated entry documents the houses-only placement tools added for Dispatch 6
T1.2.  They are host-side geometry/terrain products only: none of them edits
`tamriel.esm`, Tamriel_Data, TR/SHOTN/PC files, or a plugin.

## `solve_city_placement.py`

**Pipeline position:** accepted T1.1 `city_plan.json` + zero-error validation →
T1.2 placement → T1.3 terrain editor.

**Inputs:** the plan, its current T1.1 validation result, site survey, kit
brief, region palette, both hash-pinned D-STAMP libraries, corrected road
centerlines, and one dense `survey_fields.npz`.  `--terrain-pass planned`
produces provisional pad requests; `--terrain-pass final` additionally requires
`--planned-placement` and records the final field hash used for re-seat.

**Outputs:** `city_placement.json`, `land_edit_requests.json`,
`solver_report.json`, and `manifest.json` under `--out-dir`.  The placement
product has absolute-world GU positions, mathematical exterior cell buckets,
raw TES3 Euler radians, engine rotation matrices, source/member provenance,
and optional render-only Blender Euler values explicitly labelled as such.
No TES3/tes3conv JSON is emitted.

**Hard gates:** fresh zero-error T1.1 validation and exact shared selector
agreement; eligible stamp/member mesh/door-contract validation; source-manifest
replay; the independent 37° multi-axis oracle; exact scope/buildable/water
coverage; source terrain envelope, door step, burial, and access checks; and
strict footprint overlap/contact.  Dispatch-5 spacing is reported as guidance,
not a minimum.  Fine triangle/AABB collision is recorded as deferred because
the accepted D-STAMP libraries expose hulls rather than that fine geometry.

## `build_cityplace_fixture.py`

This is a proof harness, not a real settlement authoring tool.  It derives a
synthetic plan from the accepted T1.1 synthetic template, marks it
`SYNTHETIC...NOT A FALKREATH DESIGN`, validates it, runs planned and final
passes, and writes all evidence below
`output/cityforge/phase1/t1_2_placement_fixture/`:

* planned T1.2 products and a final-reseat subdirectory;
* structured rejected cases for terrain relief, water, out-of-scope,
  collision, no compatible stamp, and road distance;
* timestamp-free synthetic final field sidecar for the pad re-seat proof;
* a deterministic top-down outcome diagnostic, explicitly not a city render;
* input/output hashes and deterministic identities.

The builder reads the source/template files but never modifies them.  It is
expected to fail closed if source replay, the oracle, T1.1 validation, or any
fixture acceptance assertion fails.

## Core modules

* `src/procgen/cityplace.py` — orchestration, validation pinning, selector
  comparison, terrain policy, outcomes, and final re-seat contract.
* `src/procgen/cityplace_transform.py` — the sole yaw/member matrix-composition
  point and source replay/oracle gates; direct Euler yaw addition is forbidden.
* `src/procgen/cityplace_geometry.py` — exact transformed hulls, dense-field
  sampling, road metrics, and hard collision/contact checks.
* `src/procgen/cityplace_contracts.py` — field metadata/pass/frame contract and
  deterministic bilinear terrain samples.
* `src/procgen/cityplace_output.py` — deterministic JSON product assembly.

## Narrow checks

```powershell
$env:PYTHONPATH = 'src'
python -m unittest tests.test_cityplace -v
python tools/cityforge/build_cityplace_fixture.py --workspace-root .
```
