# 2026-08-10 — Cityforge tool family (T0.4 door proof, first entry)

## What this folder is

`tools/cityforge/` hosts the CLI drivers of the Cityforge arc (master plan
`.opencode/runs/city-authoring-plan/2026-08-09_city_authoring_master_plan.md`).
Engine code lives in `src/procgen/` (e.g. `citysite.py`, `citystamps.py`,
`engine_transform.py`); per-settlement configs live in `configs/`; outputs
live under `output/cityforge/`.

## Tools present today

### `door_tes3conv_proof.py` — Cityforge T0.4 door-through-tes3conv proof

Proves, reproducibly, that a masterless scratch plugin containing a DOOR base
record, one exterior and one interior CELL, a forward door reference
(exterior -> interior, DODT + non-empty DNAM destination cell name) and an
interior -> exterior return door reference (DODT + **empty DNAM**) survives
JSON -> ESP -> JSON through `tes3conv-master/tes3conv.exe`.

Why it exists: no real DOOR record had ever been pushed through tes3conv in
this workspace (master plan §2.2: "DOOR + DODT is builder-complete but
unproven through tes3conv"). Every later Cityforge door link (interiors,
NPC homes, quest teleports) depends on this byte-level grammar, so it is
de-risked first with an executable proof instead of production code.

Run it:

```
python tools/cityforge/door_tes3conv_proof.py
```

It exits 0 only when all 24 assertions pass; any failed assertion or any
essential stage failure (author, binary scan, round-trip) exits nonzero and
prints the failing assertion ids. It writes only under
`output/cityforge/proofs/door_tes3conv_v1/`.

Outputs (all sha256-hashed in `artifacts.sha256`):

| File | Content |
|---|---|
| `fixture.json` | the deterministic source authoring document (Header `masters: []`, DOOR, 2 CELLs, 2 door refs) |
| `authored.esp` | tes3conv JSON -> ESP |
| `roundtrip.json` | tes3conv ESP -> JSON |
| `verification.json` | machine-readable evidence: commands + exit codes, tool/artifact hashes, all 24 assertions with pass/fail, espscan summary, raw byte audit of the ESP, and the observed empty-DNAM serialization |

Key evidence produced (2026-08-10): the return door serializes as
`FRMR + NAME + DODT(24) + DATA(24)` with **no DNAM subrecord**, and
round-trips as `"destination": {"translation": [...], "rotation": [...],
"cell": ""}`. The forward door carries `DNAM(16)` with the interior cell
name.

Dependencies: `src/procgen/tes3json.py` (builders + validator),
`src/procgen/espscan.py` (independent binary scan), `tes3conv.exe` (pinned
hash `3c259868c9deca42a658ff6c69cbc8578f8eeee99ebab5342bdb73beef650b40`).
See `Documentation/guides/door_tes3conv_proof.md` for the detailed guide.

## T0.2 site-survey members

`build_site_survey.py` and `render_site.py` now implement the bounded Falkreath
survey checkpoint.  The former writes `land_roads.json` from direct
`tamriel.esm` LAND/VTEX raw 78 evidence plus the existing D-SITE bundle; the
latter renders real remap-LAND terrain and applies an exact source-mask road
highlight with only perimeter-confirmed continuation labels.  The operational
details and focused validation contract are in
`2026-08-10_site_survey_tools.md`.

## Planned members (not yet present)

Per the master plan §4/§6: `render_plan.py`, `render_city.py`, and the T1.x
authoring/verification drivers. Each new driver gets its own dated entry here.
