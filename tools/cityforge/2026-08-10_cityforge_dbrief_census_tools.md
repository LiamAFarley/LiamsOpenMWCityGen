# 2026-08-10 — Cityforge D-BRIEF Census Tool Family (T0.5, dispatch 5)

Date: 2026-08-10 (revised after adversarial review, same day) · Task:
bounded non-visual census/assembly (`.opencode/runs/cityforge-t05-dbrief-census/`)

## What this family does

Builds the deterministic, measured planner-vocabulary bundle for Falkreath
v1: `kit_brief.json` (what the planner may build with), `region_palette.json`
(what the ground may look like, with an explicit planned authoring
raw/LTEX contract), `census.json` (raw measured vectors + provenance), and
`validation.json` (57 closed-world contract gates, including the live remap
ESP cross-check), all from hash-pinned
D-STAMP libraries, the accepted final Markarth terrain-backed palette, the
remap ESP, `tamriel.esm`, the site survey, and the measured Karthgad surveys.

| Tool | Purpose |
|---|---|
| `src/procgen/censusio.py` | Shared deterministic I/O: canonical JSON serializer (sort_keys, 6-dp float rounding, trailing newline), hash pinning, numpy-default quantiles, quantile summaries. |
| `src/procgen/citybrief.py` | Stamp census engine: eligibility (Castle Barracks quarantine), **derived per-library raw counts** (never literals), derived enum + capability gaps, condensed planner records, final-palette preview resolution (fails closed on ambiguity/missing/manifest-absence), Karthgad preview hash verification, source-world footprint reconstruction, same-run nearest-neighbor boundary-gap census with zero-gap class separation and **per-run separated distributions + granularity metadata**, door-step aggregation, footprint quantiles, kit-brief assembly (spacing priors are measured guidance with `usable_as_hard_minimum: false`; `collision_clearance` is the geometry solver's domain; machine-readable `semantic_surfaces_used`). |
| `src/procgen/regionpalette.py` | LAND/VTEX census engine: R072 polygon rasterization (must reproduce declared 191 cells), plugin-local LTEX resolution (raw 0 sentinel, raw N → N−1) as a **workspace-validated internal/toolchain convention with load-order caveats** (OpenMW 0.51 internal confirmation unavailable from the connected index), 49-cell remap census + base-ESM composition, Karthgad core reproduction, water-edge evidence, road identity + pinned centerline ref, closed Phase-1 surface vocabulary with **explicit `planned_assignment` (raw/index/id + masterless plugin scope)**, `validate_authoring_assignments` fail-closed checks, `planned_output_plugin.required_local_ltex`, **live remap ESP cross-check helpers (`live_remap_ltex_table` / `crosscheck_live_remap_table` / `planned_vs_live_remap_check`)** and `planned_output_plugin.live_remap_evidence`, fail-closed `require_surface`. |
| `tools/cityforge/build_city_brief.py` | CLI: pins 22 inputs, runs the census twice into fresh staging dirs, byte-compares all four files, stamps the determinism proof (with both staging hash sets embedded) into both validation files, installs the canonical bundle, re-verifies on-disk hashes. |
| `tests/test_censusio.py` | Quantile oracle (independent implementation), serializer determinism, float rounding, file pins. |
| `tests/test_citybrief.py` | Fixture tests: exclusions, dynamic raw counts (drift), enum derivation, capability gaps, polygon gap geometry, spacing quantiles, per-run separation, preview resolution failures (missing/ambiguous/excluded/manifest-mismatch/**manifest-missing**), preview-section subcounts, malformed inputs. |
| `tests/test_regionpalette.py` | Fixture tests: raw-0 sentinel, plugin-local VTEX, unresolvable raw fails, scope separation, fraction sums, rasterization count check, closed vocabulary, **authoring-assignment contract (slot+1 hazard, road reassignment, index/raw mismatch, duplicates, remap collision, local LTEX coverage)**; **live remap cross-check fixtures (missing/wrong-id/wrong-path/extra-index fail closed, planned-vs-live agreement + settlement collision)**; real-file integration pins (R072 48,896 tiles, effective 12,544 tiles, membership 47+2, Karthgad core, live remap ESP table + sha). |
| `tests/test_citybrief_integration.py` | Full real-input pipeline run: four outputs, 54 eligible, barracks ledger, previews resolved + subcounts 11/43, dynamic raw counts, authoring contract, spacing guidance contract + run separation, real cross-file closure, load-order caveats, determinism evidence, **live remap gates wired + passing, palette live evidence, live-table drift fails validation closed**, all 57 gates pass. |

## Outputs

- `output/cityforge/briefs/falkreath_v1/kit_brief.json`
- `output/cityforge/briefs/falkreath_v1/region_palette.json`
- `output/cityforge/briefs/falkreath_v1/census.json`
- `output/cityforge/briefs/falkreath_v1/validation.json`

## Commands

```powershell
python tools/cityforge/build_city_brief.py --date 2026-08-10
python -m unittest tests.test_censusio tests.test_citybrief `
    tests.test_regionpalette tests.test_citybrief_integration
```

## Key facts (2026-08-10 post-review run)

- Eligible stamps: **54** = 11 Karthgad + 44 Markarth − Castle Barracks
  (quarantined with the exact user reason; absent from every type/quantile/
  spacing count, present once in the exclusion ledger). Raw per-library
  counts are derived from the loaded libraries (gates recompute them).
- Derived enum (12 types): farm, guild, hall, house, keep, manor, mill,
  shop, smith, stable, tavern, unknown. `lodge`/`shack` unavailable → machine
  readable capability gaps (lodge, stone_wall, fence_spacing).
- Coverage: 25 houses (≥15 required), 2 taverns, 2 smiths, 5 shops, 1 farm.
- All 43 eligible Markarth previews resolve to final terrain-backed sheets
  (sha256-verified against the palette catalog + render manifest; manifest
  absence is a hard failure); zero split-render-v6 paths. Preview
  verification section: Karthgad 11 / Markarth 43 subcounts.
- Spacing: 958 same-run pairs (55 Karthgad + 903 Markarth); 28 zero-gap
  pairs; 39/54 stamps have a zero-gap nearest neighbor. Positive NN prior
  p10/p50/p90 = 20.0 / 135.8 / 3344.7 GU is **measured guidance
  (`usable_as_hard_minimum: false`)**; per-run separated stats and
  granularity (door-seeded Karthgad vs split-unit Markarth) are emitted;
  collision clearance (0.0 GU hard minimum) is the geometry solver's domain.
- Authoring contract: planned raw/index/id per surface — base 33/32, dirt
  241/240, grass-dirt 142/141, cobble 144/143, road 78/77 (protected),
  water-edge sand 1/0; masterless city plugin must define local LTEX records
  at indices 0, 32, 77, 141, 143, 240; raw is never ordinal+1.
- Door steps: n=76 from stamps (p10/p50/p90 = 115.6/209.9/591.9 GU) with the
  n=470 ground-rules cross-check (70.6/103.3/553.2 GU, survey_measured).
- Land census: R072 48,896 tiles (191 cells; AI_Grass 32,725 · Sand 10,960 ·
  MA_sulphur_rock02 3,248 · MA_lavaflow 1,360 · Tx_BC_moss 603);
  effective block 12,544 tiles (49 cells; GrassRE 9,817 · RoadOH 1,275 ·
  Sand 1,330 · Pine 122). 47/49 target cells inside R072; (−90,−11),
  (−89,−11) in R014 — recorded, not merged. Load-order caveats emitted for
  both scopes; OpenMW 0.51 internal confirmation noted unavailable.
- Karthgad core (−102,11) reproduced from Sky_Main.esm: 159/42/35/16/2/1/1 —
  matches the survey table (survey omitted one dirtRE_03 tile).
- Two fresh staging builds byte-identical across all four files; determinism
  evidence (both staging hash sets) embedded in validation.json; canonical
  hashes recorded in the guide and the run report.
- **Live remap ESP cross-check (M-8):** the CLI reads the remap ESP's LTEX
  table once via `espland.load_ltex` and fails closed unless indices/ids/
  texture paths exactly match the expected table, with road index 77 pinned
  to `T_Hr_TerrRoadOH_01` / `hr\lnd\hr_oh_road_01.dds`. The measured
  path/SHA (237c96d8…) and all four records are emitted into
  `region_palette.json#planned_output_plugin.live_remap_evidence`; planned
  assignments at shared indices must agree, and the additional masterless
  city LTEX indices 141/143/240 must stay collision-free (8 new gates → 57
  total). A drifted live table fails the canonical validation path closed
  (integration test).

Full guide: `Documentation/guides/cityforge_dbrief_census.md`.
