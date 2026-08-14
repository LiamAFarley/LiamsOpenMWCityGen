# Cityforge D-BRIEF Census — Kit Brief & Region Palette (guide)

Date: 2026-08-10 (revised after adversarial review, same day) · Task:
bounded non-visual census/assembly (dispatch 5,
`.opencode/runs/cityforge-t05-dbrief-census/`)

> Revision note: this guide reflects the post-review contract. The review
> (`.opencode/runs/cityforge-t05-dbrief-census/2026-08-10_cityforge_t05_dbrief_census_review.md`)
> found 5 important + 7 minor contract hazards; all are fixed and covered by
> tests. Changes: explicit planned authoring assignments (never slot
> ordinals), dynamic source counts, corrected VTEX/LTEX evidence wording with
> load-order caveats, source-separated spacing distributions with a
> no-hard-minimum contract, renamed preview-verification section with
> subcounts, determinism evidence embedded in validation, non-tautological
> gates, and a slimmed palette.

## 1. What this family does

Builds the deterministic, measured **planner vocabulary bundle** for the
Falkreath v1 city-authoring arc: everything a future plan may reference
(building types, stamps, spacing priors, ground surfaces, road identity) is
enumerated here from hash-pinned source data, so plans become validatable
and the kit/region stays portable. Four canonical JSON products are emitted:

| File | Role |
|---|---|
| `kit_brief.json` | What the planner may *build with*: derived `building_type_enum`, 54 condensed stamp records, spacing/door-step priors (measured guidance, not hard minimums), `collision_clearance` (geometry-solver domain), boundary pieces, street furniture, docks, capability gaps, exclusion ledger, machine-readable `semantic_surfaces_used`. |
| `region_palette.json` | What the ground may *look like*: R072 base textures (measured), effective 49-cell block textures (remap identities), settlement clearance, water-edge evidence, road identity + aligned consumer road-product ref, flora/rock proxies, closed Phase-1 semantic-surface vocabulary with **explicit planned authoring assignments** (`planned_raw_vtex` / `planned_ltex_index` / `planned_ltex_id` + masterless plugin scope and required local LTEX table), load-order caveats. |
| `census.json` | Raw measured vectors and provenance: per-stamp records, per-pair spacing samples with source-separated run stats and granularity metadata, door-step samples, footprint quantiles, LAND tile counts per scope, water/road evidence, all input pins, dynamic per-library raw stamp counts, merged preview-verification section with Karthgad/Markarth subcounts. |
| `validation.json` | 49 closed-world and cross-file contract gates (counts, exclusions, preview resolution, authoring assignments, tile totals, vocabulary closure, provenance, determinism evidence with both staging build hash sets). |

Core logic lives in `src/procgen/citybrief.py` (stamps/spacing/kit) and
`src/procgen/regionpalette.py` (LAND/VTEX/palette), with shared deterministic
I/O and quantiles in `src/procgen/censusio.py`.

## 2. Pipeline position

```
D-STAMP libraries (karthgad_nord_v1, markarth_side_stone_v1)
  + final Markarth stamp_palette_v1 catalog + render manifest
  + site_survey.json + regions.json + tamriel.esm + remap ESP
  + Sky_Main.esm (Karthgad core) + Vorndgad proxies + surveys
        │  build_city_brief.py (CLI, double-build determinism proof)
        ▼
output/cityforge/briefs/falkreath_v1/{kit_brief,region_palette,census,validation}.json
        │
        ▼
D-PLAN / D-PLACE / D-SCAPE / D-MASK stages (future) — plan validator consumes
the enums and surfaces directly; unknown references fail closed.
```

The brief is the consumption boundary for the city-authoring arc; it does not
duplicate the mesh-usage census or citywalls analysis (those products, when
they land, regenerate `boundary_pieces.stone_wall` and retire hand-curated
numbers to cross-checks).

## 3. Commands

```powershell
# canonical build (two fresh staging builds, byte-compare, install)
python tools/cityforge/build_city_brief.py --date 2026-08-10

# debug single run without the determinism proof
python tools/cityforge/build_city_brief.py --date 2026-08-10 --no-proof

# focused tests
python -m unittest tests.test_censusio tests.test_citybrief `
    tests.test_regionpalette tests.test_citybrief_integration
```

Exit codes: `0` all gates pass and canonical outputs installed; `1` hard
census failure (FAILURE protocol — nothing is written); `2` validation gates
failed.

## 4. Inputs (all read-only, hash-pinned in `census.json#inputs`)

- D-STAMP libraries `output/cityforge/stamps/{karthgad_nord_v1,
  markarth_side_stone_v1}.json` (11 + 44 raw records).
- Accepted final Markarth library
  `output/settlement-splits/markarth-side-v2/final-markarth-extraction-2026-08-10-library/`
  → `stamp_palette_v1/catalog.json` (preview/provenance authority) and
  `render_library_manifest.json` (sheet-hash cross-check).
- `output/cityforge/stamps/catalog_v1/index.json` (Karthgad preview hashes).
- `output/cityforge/sites/falkreath_v1/site_survey.json` (49 target cells,
  water, roads evidence), `output/mapdata/regions.json` (R072 polygon),
  `tamriel.esm`, `output/falkreath_landscape_texture_remap.esp` (+ report),
  `Sky_Main.esm` (Karthgad core reproduction), `output/terrain_cells.json`,
  `output/vorndgad_scatter_analysis.json`,
  `output/vorndgad_cliff_analysis.json`,
  `output/skyrim_ground_rules.json`,
  `configs/groundcover_falkreath_v1_currenttextures.ini`,
  the aligned road consumer product
  `output/mapdata/roads/tamriel_aligned_centerlines_v1/` (plus the
  source-space bundle it was derived from, topology/provenance storage
  only), the two Karthgad
  survey reports, and the Karthgad placement/buildings products.

## 5. What the census measures (and how)

### 5.1 Stamp aggregation and eligibility

- Eligible = 54 = 11 Karthgad + 44 Markarth − `markarth_side_v1__u114_castle_barracks`
  (exact reason `user-reported defective Castle Barracks extraction`,
  matching the palette catalog's quarantine). Exclusion happens **before**
  any type/count/quantile/spacing computation; the barracks appears exactly
  once in the exclusion ledger.
- `building_type_enum` is the sorted union of eligible stamp types:
  `farm, guild, hall, house, keep, manor, mill, shop, smith, stable, tavern,
  unknown` — derived, never the stale hard-coded example enum. `lodge` and
  `shack` are **unavailable** (no measured stamp); a machine-readable
  capability gap is emitted for `lodge` (plus `stone_wall` and
  `fence_spacing`, also unmeasured), so a future plan validator cannot
  request them.
- Upstream D-STAMP exclusions (walls/ship false positives, bounds-missing
  candidates, doorless components) are preserved as a summarized ledger; the
  brief never reintroduces them from the final render manifest.

### 5.2 Preview resolution

- Every eligible Markarth stamp resolves by split-record identity
  (`source.unit_id` == palette `split_record_id`) to exactly one eligible
  `building_unit` palette entry; ambiguity, missing entries, excluded
  entries, missing sheet files, a missing render-manifest record, and
  manifest/catalog hash disagreement are all hard failures (nothing is
  silently skipped). The emitted `preview_sheet` points into the final
  terrain-backed library (`.../final-markarth-extraction-2026-08-10-library/
  <file>`); the superseded `split-render-v6` path is recorded only as
  `preview_replaced_source` provenance.
- Karthgad previews are hash-verified against their source render paths and
  cross-checked with `catalog_v1/index.json`.
- `census.json#stamps.preview_verification` is the merged all-stamp
  verification section with explicit subcounts: **Karthgad 11** (hash-
  verified source renders) and **Markarth 43** (final-palette resolutions).

### 5.3 Spacing

- Source-world footprint = stored seed-door anchor `source_position_gu` +
  `footprint.hull_xy_rel` (world-aligned reconstruction; no transforms).
- Boundary gap = minimum Euclidean distance between polygon edges; pairs
  that intersect or touch are the **zero-gap class** (gap exactly 0.0) and
  are counted separately from positive gaps.
- Measured **only within the same source run** (Karthgad 11 stamps → 55
  pairs; Markarth 43 eligible → 903 pairs); never across runs.
- **Heterogeneity is explicit**: Karthgad stamps are door-seeded
  complete-building groupings of one dense core (no `source.unit_id`, so
  fused/duplicate source identification is impossible at this level);
  Markarth stamps are split-unit groupings with `unit_id` provenance. Hulls
  are D-STAMP source-world footprint hulls (approximated envelopes); zero
  gaps reflect envelope overlaps/touches, and 39/54 stamps have a zero-gap
  nearest neighbor — so the positive-only mixed sample (n=15: 3 Karthgad
  door-level + 12 Markarth building-level, computed per run) is
  **exploratory measured guidance, not a universal hard clearance**.
- Per-run separated distributions are emitted (`census.json#spacing.runs`
  with `nearest_neighbor_positive_gap_gu` per run) plus granularity
  metadata; the brief's `inter_building_gap_gu` carries
  `usable_as_hard_minimum: false`, and `collision_clearance` is the
  geometry solver's exact-hull domain (hard minimum 0.0 GU, contact-graph
  basis).
- Quantiles: linear interpolation between closest ranks (numpy default),
  reproduced by an independent test oracle on fixtures.

### 5.4 Door-step priors

- Aggregated from eligible stamps' `terrain_envelope.door_step_heights_gu`
  (one sample per door, stamp id attached): n=76, p10 ≈ 115.6, p50 ≈ 209.9,
  p90 ≈ 591.9 GU.
- Cross-checked (not copied from prose) with `skyrim_ground_rules.json`
  stats: n=470, p10 70.56 / p50 103.35 / p90 553.19 GU, marked
  `survey_measured`.

### 5.5 LAND/VTEX census (two scopes, never merged)

- **R072** (191 cells): enumerated from `regions.json` by rasterizing the
  PTR polygon at integer cell centers with the documented map transform; the
  count must reproduce the declared `cell_count` or the census aborts.
  Census in `tamriel.esm` with its **own** local LTEX table: raw 1 Sand
  10,960 · raw 33 AI_Grass 32,725 · raw 78 MA_sulphur_rock02 3,248 · raw 72
  MA_lavaflow 1,360 · raw 92 Tx_BC_moss.tga 603 tiles (48,896 total).
- **Effective 49-cell block**: census in the masterless remap ESP with its
  local LTEX table: raw 33 → `T_Sky_TerrGrassRE_01` 9,817 · raw 78 →
  `T_Hr_TerrRoadOH_01` 1,275 · raw 1 → Sand 1,330 · raw 92 →
  `T_Sky_TerrPine_01` 122 tiles (12,544 total). Base-ESM composition of the
  same 49 cells is recorded separately.
- `espland` semantics throughout: raw 0 = base sentinel (never an LTEX
  record); raw `N > 0` → owning-plugin LTEX index `N - 1`. This is the
  workspace-validated **internal/toolchain convention** (espland + remap
  round-trip evidence: the remap ESP's LAND payloads resolve through its own
  4-record LTEX table to the identities recorded in the remap report).
  **OpenMW 0.51 internal API confirmation is unavailable from the connected
  openmw-docs index (no LTEX/VTEX resolution coverage); verify against the
  engine source (`terrainstorage`/`ESMStore` LandTexture handling) before
  Phase 1 authoring.** Identity labels are load-order sensitive and the
  palette records `load_order_caveats` per scope: R072 `base_textures`
  labels describe base-ESM-only load order; with the remap ESP loaded, raw
  33/78/92 render under the remap identities (`T_Sky_TerrGrassRE_01` /
  `T_Hr_TerrRoadOH_01` / `T_Sky_TerrPine_01`); `effective_block_textures`
  labels match in-game appearance with the remap ESP loaded. Tile counts are
  unaffected by either interpretation for the current data.
- Membership is explicit: 47 of the 49 target cells lie inside R072;
  `(-90,-11)` and `(-89,-11)` lie in R014 JERALLMOUNTAINS per the planning
  polygons — recorded, not forced.
- Karthgad core cell `(-102,11)` census is **recomputed** from
  `Sky_Main.esm` (159 dirt / 42 grass-dirt / 35 cobble / 16 road-dirt / 2
  gravel / 1 rock / 1 dirtRE_03) and compared with the survey table
  (matches; the survey omitted the single dirtRE_03 tile).

### 5.6 Road and water-edge evidence

- Road identity: raw 78 is the **only protected source identity**
  (base ESM `MA_sulphur_rock02` → remap `T_Hr_TerrRoadOH_01`); raw 1 Sand is
  never road. Direct LAND/VTEX-78 tiles are the **in-game occupancy
  authority**. `road_network_ref` pins the aligned consumer product
  (`tamriel_aligned_centerlines_v1.json` + its alignment manifest) and
  records the source-space bundle it was derived from; consumers load the
  aligned product only through `src/procgen/aligned_roads.py`.
  `roads_graph_clean.json` vectors and the raw-78-only `land_roads.json` are
  explicitly not used as geometry; the XCF/BMP are provenance only.
- Water-edge: quantitative evidence only — raw-1 Sand tile counts (1,330 in
  the block), the site survey's 11 water cells, 131,072 GU shore length,
  water-mask definitions, and per-cell `water_frac`/band from
  `terrain_cells.json`.

## 6. Semantic-surface vocabulary and authoring contract (closed)

Phase-1 surfaces: `base`, `settlement_dirt`, `settlement_grass_dirt`,
`settlement_cobble`, `road`, `water_edge_sand`. Each surface carries:

- `measured_identity` — the measured source/remap raw + LTEX identity
  (separate from any planned output), with scope (base ESM vs remap ESP vs
  Sky_Main) and, for the settlement classes, provenance back to
  `census.json#land.karthgad_core_reproduction`;
- `planned_assignment` — the explicit authoring contract for the future
  **masterless city output plugin** (`masters: []`): `planned_raw_vtex`,
  `planned_ltex_index` (= raw − 1), `planned_ltex_id`, and
  `local_ltex_record_required: true`.

Planned values (review-corrected): base 33/32 `T_Sky_TerrGrassRE_01`;
settlement_dirt 241/240 `T_Sky_TerrDirtRE_01`; settlement_grass_dirt
142/141 `T_Sky_TerrGrassDirtRE_01`; settlement_cobble 144/143
`T_Nor_Set_TxCobbleStone_01`; **road 78/77 `T_Hr_TerrRoadOH_01` (protected —
never reassigned)**; water_edge_sand 1/0 `Sand`.

The `surface_ordinal` (0–5) is a **pure enumeration label**: raw_vtex is
never derived from it (`ordinal + 1` is not a raw value — e.g. it would turn
road into raw 5, which no surface carries). `validate_authoring_assignments`
fails closed on: missing explicit assignments, raw/index inconsistency
(index ≠ raw − 1), duplicate raws, road reassignment, planned LTEX index
collisions with a different remap identity (remap table {0: Sand, 32:
GrassRE, 77: RoadOH, 91: Pine}), and incomplete local-LTEX coverage.
`planned_output_plugin.required_local_ltex` lists the six local LTEX records
(indices 0, 32, 77, 141, 143, 240) the masterless plugin must define; any
additional authored raw > 0 must add its own local record (fail-closed
rule). A Phase-1 planner validator must reject any authored raw ≠
`planned_raw_vtex` for a listed surface.

### 6.1 Live remap ESP cross-check (review finding M-8)

The expected remap table constant (`REMAP_LTEX_TABLE`: indices {0, 32, 77,
91} with ids `Sand` / `T_Sky_TerrGrassRE_01` / `T_Hr_TerrRoadOH_01` /
`T_Sky_TerrPine_01` and their texture paths) is only the **expectation**.
The CLI loads the **live** table from the hash-pinned
`output/falkreath_landscape_texture_remap.esp` once via
`espland.load_ltex` (`regionpalette.live_remap_ltex_table`) and gates the
authoring contract against those measured records:

- `remap.live_table_coverage` — the live indices must be exactly
  {0, 32, 77, 91} (missing **or extra** records fail closed);
- `remap.live_identity.indexN` — each live record's id **and** normalized
  texture path must equal the expected entry;
- `remap.road77_live_protected` — live index 77 must be
  `T_Hr_TerrRoadOH_01` / `hr\lnd\hr_oh_road_01.dds` exactly;
- `authoring.planned_vs_live_remap` — every planned surface assignment at a
  shared remap index must agree with the live identity, and the valid
  **additional masterless-city LTEX indices 141/143/240** (settlement
  classes, Sky_Main-measured) must have no live remap collision;
- `authoring.live_remap_evidence_emitted` — the palette's
  `planned_output_plugin.live_remap_evidence` (path, SHA-256, all four
  records) matches the live read.

The measured path/SHA/records are emitted into `region_palette.json` under
`planned_output_plugin.live_remap_evidence`, so a regenerated remap ESP is
visible in the bundle **and** fails the gates.

## 7. Determinism

- One serializer (`censusio.deterministic_dumps`: UTF-8, sort_keys, 2-space
  indent, trailing newline, floats rounded to 6 decimals).
- The CLI builds the whole bundle **twice** into fresh staging directories,
  byte-compares all four files, stamps the determinism proof into both
  validation files (identical content), re-compares, and only then installs
  the canonical output and re-verifies the on-disk hashes. The proof embeds
  **both staging build hash sets** (pre-stamp, at byte-comparison time) plus
  the `validation_json_hash_scope` note, so the emitted artifact is
  self-verifying.
- 2026-08-10 (post-M-8) canonical hashes:

```
census.json          064e013515d669f5271974c3660f8a775ae9eada2dcd5f18674d9919ff82f9e7
kit_brief.json       f36947656cedde5fb14dc3af88dc917b63f2a1b1adc544c29d54713cad25c601
region_palette.json  942ee762d9fde15883d51bab81958c2bbd7bddff38266fcb530acd37ea9a33b3
validation.json      601e0e4842e7bd7b15d46037f81ab08ebf25907f00a5ce859db8e3bbd7e117cd
```

(Pre-M-8: `region_palette.json a6f49d2e…`, `validation.json 98a0130a…`;
`census.json`/`kit_brief.json` unchanged — the live remap evidence lives in
the palette and the validation gates.)

## 8. Validation gates (validation.json)

57 gates across: eligible count (54) and uniqueness; **derived raw
per-library counts** (recomputed from the loaded libraries, never literals)
and raw−excluded−eligible reconciliation; barracks quarantine (once, exact
reason, absent from all aggregation); coverage (≥15 houses, ≥1
tavern/smith/shop/farm, lodge unavailable); preview resolution (43 Markarth
resolved, section subcounts 11/43, no stale v6, files exist); land totals
(191×256 and 49×256, raw-0 sentinel **accounting** per scope, scope
separation, raw 78 road identity); **authoring assignment gates**
(explicit assignment, raw/index consistency, uniqueness, road 78/77,
no remap collision, local LTEX coverage, masterless scope, no slot+1
hazard); **live remap ESP cross-check gates** (`remap.live_table_coverage`,
`remap.live_identity.index0/32/77/91`, `remap.road77_live_protected`,
`authoring.planned_vs_live_remap`, `authoring.live_remap_evidence_emitted` —
see §6.1); palette closure and fraction sums; **real cross-file closure**
(kit brief `semantic_surfaces_used` ⊆ palette surfaces; every stamp type in
enum and counts keys == enum); spacing contract (`usable_as_hard_minimum`
false; run-separated distributions and granularity present); input pins;
determinism proof with embedded evidence.

## 9. Limitations / measured-data caveats

- Spacing: the positive-only mixed NN sample (n=15) is **exploratory
  guidance** (`usable_as_hard_minimum: false`); Karthgad contributes
  door-seeded units of one dense core (no unit_id → fused/duplicate
  detection impossible), Markarth contributes split units; per-run
  distributions are the authoritative separation. Collision clearance is
  the geometry solver's exact-hull domain.
- VTEX/LTEX resolution is a workspace-validated internal convention;
  OpenMW 0.51 internal confirmation is unavailable from the connected
  openmw-docs index, and identity labels are load-order dependent (caveats
  emitted in the palette).
- Palisade overlap (35–70 GU) and spacing numbers are **AABB-derived** (the
  contact graph intentionally excludes palisades); method labels are
  emitted.
- `stone_wall`, `fence_spacing`, and `lodge` are unmeasured capabilities —
  represented as gaps, never fabricated.
- The settlement-clearance pattern is one culture sample (Karthgad); a
  multi-town census will turn it into per-culture ranges.
- Vorndgad ecology is an explicit **proxy** (`proxy_region` status) until a
  Kreathi Dale profile is measured.
- The 49-cell block is not fully inside R072 (2 cells in R014) — the
  palette records the overlap rather than papering over it.
