# Guide — Door-through-tes3conv proof (cityforge T0.4)

- **Tool:** `tools/cityforge/door_tes3conv_proof.py`
- **Family:** Cityforge (see `tools/cityforge/2026-08-10_cityforge_tools.md`)
- **Status:** active, proof artifact (not production city authoring)
- **Last validated:** 2026-08-10 (report:
  `.opencode/runs/cityforge-t04-door-proof/2026-08-10_door_tes3conv_report.md`)

## Pipeline position

Master plan §6 T0.4: DOOR-through-tes3conv proof. It de-risks every later
Cityforge stage that wires doors: T1.4 authoring verification chain,
interiors + door DODT wiring, NPC homes, quest teleports. It sits between
`src/procgen/tes3json.py` (authoring contract) and `src/procgen/espscan.py`
(binary verification), with `tes3conv.exe` as the ground-truth converter.

## What it does

1. Builds a deterministic masterless document (`masters: []`):
   - `Header` (Esp, version 1.3)
   - one `Door` base record (`cf_t04_door_01`, mesh `x\ex_door_01.nif`)
   - one exterior CELL `cf_t04_exterior` at grid [-95, -11]
   - one interior CELL `cf_t04_interior` (IS_INTERIOR, grid [0, 0])
   - forward door ref (refr_index 1): translation [128, 256, 384.5],
     rotation [0.5, -1.5, 2.0], destination {translation [512, 640, 96],
     rotation [0.25, -0.5, 1.75], cell "cf_t04_interior"}
   - return door ref (refr_index 2): the mirror transform, destination
     {…, cell ""} — the **empty-DNAM return door**
2. Hard-gates the document through `tes3json.validate()`.
3. Runs `tes3conv -o -c fixture.json authored.esp` then
   `tes3conv -o -c authored.esp roundtrip.json`.
4. Asserts on the round-tripped JSON: header masterlessness, DOOR base
   record, both CELLs, both refs, exact DODT values, exact transforms,
   persistence, and the empty `destination.cell`.
5. Independently scans the ESP bytes:
   - `espscan.scan_file` (record counts, cell kinds, per-ref DODT/DNAM
     presence via the unchanged `has_dodt`/`destination_cell` fields),
   - a driver-local raw byte audit (HEDR master count + file type,
     DOOR subrecords, per-cell FRMR group DODT/DNAM sizes — this is what
     proves DNAM presence/absence and sizes).
6. Writes `verification.json` (all evidence + hashes) and
   `artifacts.sha256`, prints a PASS/FAIL assertion table, exits 0/1.

## Empty-DNAM serialization (the fact this proof pins)

TES3 return doors (interior -> exterior) carry DODT but no DNAM; the engine
resolves the exit position geometrically. The tes3 crate at the pinned rev
(`51fae82b79838d76a39d0d1d0d472d7f48e8577f`) models door destinations as
`TravelDestination { translation: [f32; 3], rotation: [f32; 3], cell: String }`
where `cell` is a **non-Option** String; its `Save` writes DNAM only when
`cell` is non-empty, and `Load` defaults a missing DNAM to `""`. Observed
through tes3conv (2026-08-10):

| Door | ESP bytes | Round-trip JSON |
|---|---|---|
| forward (exterior -> interior) | `FRMR + NAME + DODT(24) + DNAM(16) + DATA(24)` | `"cell": "cf_t04_interior"` |
| return (interior -> exterior) | `FRMR + NAME + DODT(24) + DATA(24)` (no DNAM) | `"cell": ""` |

So `"destination": {…, "cell": ""}` is the correct authoring form. espscan
(unchanged) sees the return door as `has_dodt=True, destination_cell=None`;
the DNAM presence/size facts come from the driver-local raw byte audit
(return: `dodt_size=24, dnam_size=None`; forward: `dnam_size=16`).

## Resulting code changes (evidence-driven, additive)

- `src/procgen/tes3json.py` — `_validate_door_destinations` now skips the
  unknown-cell `door-link` check when the destination cell name is the empty
  string (valid return-door grammar). Non-empty names still must resolve to a
  cell or region in the document; temporary door refs with destinations are
  still rejected. Regression tests: `tests/test_tes3json.py`
  (`test_empty_dnam_return_door_destination_is_valid`,
  `test_unknown_nonempty_door_destination_cell_is_still_caught`,
  `test_temporary_door_reference_with_destination_is_still_caught`).
- `src/procgen/espscan.py`, `tools/settlement_pipeline/scan_cache.py`,
  `tests/test_espscan.py`, `tests/test_scan_cache.py` — **intentionally
  untouched** (scope correction 2026-08-10: the settlement-pipeline arc is
  off-limits and the driver-local byte audit already proves DNAM
  presence/absence; earlier exploratory edits to these four files were
  removed byte-for-byte).

## Verification signals

- Proof exits 0 and prints `PASS`; `verification.json` `"passed": true` with
  zero `failure_ids`.
- `artifacts.sha256` matches `verification.json` `artifact_hashes`.
- `verification.json` is byte-stable across re-runs (the espscan timing
  field is omitted from the evidence), so artifact hashes are reproducible.
- tes3conv.exe sha256 pinned at
  `3c259868c9deca42a658ff6c69cbc8578f8eeee99ebab5342bdb73beef650b40`.

## Limitations

- Proof-only: the mesh `x\ex_door_01.nif` is a stand-in; nothing here
  validates mesh existence or OpenMW runtime behavior (no OpenMW claims).
- Interior CELLs are authored with grid [0, 0]; vanilla editors use
  [0, junk], which the engine ignores for interiors (espscan reads grid only
  for exteriors).
- `Header.num_objects` is recomputed by tes3conv (known lossy field).
- No DNAM-at-all vs empty-string distinction exists in TES3: both are the
  return-door grammar; the JSON form is always `"cell": ""`.
