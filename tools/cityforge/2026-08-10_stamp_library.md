# 2026-08-10 — Cityforge Stamp Library Tool Family

Date: 2026-08-10 · Task: Cityforge T0.3 · D-STAMP spec:
`.opencode/runs/city-authoring-plan/2026-08-09_dstamp_unit_stamp_library_spec.md`

## What this family does

Deterministically derives **unit stamps** — complete, guaranteed-valid
building packages for re-placement — from existing extraction/split products,
writes them as hash-pinned D-STAMP v1 libraries, and packages a browsable
catalog whose every accepted stamp links a byte-identical copy of a real
existing contact sheet.

| Tool | Purpose |
|---|---|
| `src/procgen/citystamps.py` | Core derivation module: member offsets (exact float64 subtraction from the seed-door anchor), verbatim rotation/scale copying, footprint AABB + 2D hull, hull-polygon-centroid access heading, LAND terrain envelope (door steps, relief, slope, burial), classification, exclusion ledger (audited + derivation reasons), replay evidence, canonical JSON bytes. |
| `tools/cityforge/stamp_library.py` | CLI: loads the Karthgad and Markarth split products (read-only), joins by source id, applies the audited non-building exclusion list (hash-pinned), derives both libraries, writes JSON, copies previews into `catalog_v1/`, verifies determinism byte-for-byte. |
| `tools/cityforge/non_building_audit_v1.json` | Exact audited exclusion list (2026-08-10 acceptance repair): `u31_city_walls` + `u31_castle_walls` → `non_building_boundary`, `u105_herringbone` → `non_building_vehicle`, pinned to the `manual-corrections-v1` units.json sha256 with inspected preview evidence. |

## Outputs

- `output/cityforge/stamps/karthgad_nord_v1.json` — 11 Karthgad Nord kit
  stamps (source: `output/skyrim-settlements/karthgad-v1`).
- `output/cityforge/stamps/markarth_side_stone_v1.json` — 44 Markarth Side
  stone-kit stamps from the approved `manual-corrections-v1` split subset
  (provisional, hash-pinned; joined to the authoritative markarth-side-v1
  placement manifest by source id; 13 excluded incl. 3 audited
  non-building).
- `output/cityforge/stamps/catalog_v1/` — `index.html` (browsable),
  `index.md`, `index.json` (machine-readable, hash-verified), and
  `previews/<library>/<stamp_id>.png` byte-identical copies of the existing
  contact sheets (55 previews).

## Commands

```powershell
python tools/cityforge/stamp_library.py --libraries both
python tools/cityforge/stamp_library.py --libraries both --verify-determinism
python -m unittest tests.test_citystamps
```

## Key conventions

- **Anchor** = the seed door's world position; member offsets are pure
  subtraction (world-aligned, no baked rotation).  Rotations stay as the
  source-authored TES3 Euler triples — all Euler arithmetic is deferred to
  `engine_transform.py` (T0.1), never done here.
- **Access heading** = atan2(seed door − 2D hull-polygon centroid) with an
  explicit degenerate fallback (mean of hull vertices); the AABB midpoint is
  no longer used (2026-08-10 repair).
- **Exclusions are explicit**: audited review overrides first
  (`non_building_boundary` / `non_building_vehicle`, exact hash-pinned
  list), then ghost members, protocol failures, no door, missing LAND,
  missing bounds, missing preview.  Every excluded candidate appears exactly
  once with one reason; source-run components recorded as doorless/access-
  only by the extraction products are listed under
  `stats.source_recorded_exclusions` with scope `source_run_component`.
- **Replay evidence** is embedded per library: anchor+offset reconstructs the
  source absolute position exactly (0.0 GU error across all 744 members),
  rotation/scale match source verbatim, and the read-only oracle
  (`tools/karthgad_rebuild_geometry.py::placement_scene_matrix`) reproduces
  every source placement matrix (0 mismatches).
- **Terrain cross-check** (Karthgad only): uniform `espland` LAND
  re-derivation vs the landscape product; disagreement > 1 GU is an explicit
  report row, never silently accepted (currently 0 rows).

Full guide: `Documentation/guides/stamp_library.md`.
