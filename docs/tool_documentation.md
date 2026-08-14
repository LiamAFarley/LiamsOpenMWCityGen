# Procedural Tamriel Tool Documentation

This index identifies active tool families and links to their operational
guides. Detailed guides explain pipeline position, commands, inputs, outputs,
conventions, validation signals, and limitations without duplicating the same
workflow in every script description.

## Active tool families

| Tool family | Status | Main entry point | Description | Guide |
|---|---|---|---|---|
| Region Mesh Analysis | active | `tools/scatter_vorndgad_analysis.py` | Extracts measured flora, rock, cliff, terrain, texture, transform, density, and water-context evidence from source cells and aggregates it into deterministic per-mesh profiles. The current source profile is SHOTN's Vorndgad Forest (59 cells); the family name covers future region profiles. | [Detailed guide](guides/region_mesh_analysis.md) |
| Falkreath Wilderness Scatter Generation and Authoring | active | `tools/generate_falkreath_scatter.py` | Converts the measured Vorndgad products and authoritative tamriel.esm LAND into deterministic placements, then authors the accepted 49-cell document as a masterless normal `content=` TES3 plugin. | [Detailed guide](guides/falkreath_scatter_generation.md) |
| Landscape Texture Remap | active | `tools/landscape_textures/falkreath_landscape_remap.py` | Replaces selected landscape texture classes in a fixed exterior-cell proving ground (Falkreath 7x7, 49 cells) by authoring a masterless ESP containing LAND overrides and the output-owned LTEX definitions they need, without editing the source ESM. | [Detailed guide](guides/landscape_texture_swapping.md) |
| Settlement Batch Extraction and Component Pipeline | active | `tools/settlement_pipeline/run_batch.py` | One-command A1-to-flatten extraction of settlements from a configured launch JSON: scope-wide linked-door seeding, role-registry-backed component manifests, landscape LAND/LTEX + per-building height fields, render sheets, resume markers, and batch reports. Extracts evidence/components/landscape/renders; it does not author settlement plugins. | [Detailed guide](guides/settlement_batch_workflow.md) |
| Kit/Role Mesh Registry | active | `tools/kit_roles/validate_registry.py` | One strict schema-v1 JSON document classifies meshes as shell / access / connector / **boundary** with priority, conflict detection, and provenance; consumed by `component_manifests.py --role-registry` and by the settlement splitting stage (`split_units.py`). The `boundary` role (added 2026-08-07) is the NAME-tier prior for weak-link candidate families (walls/fences/gates/planks/docks/streets); which wall is a connection vs. a house yard member is decided per placement by the split stage. The per-consumer exclusion fields (`contact_exclude` / `render_exclude` / `profile_exclude`) are schema-reserved and not yet consumed by any pipeline stage. | [Detailed guide](guides/kit_role_registry_usage.md) |
| Cityforge (T0.4 door proof) | active | `tools/cityforge/door_tes3conv_proof.py` | Reproducible proof that real TES3 DOOR records and a linked exterior/interior door pair (forward DODT+DNAM, empty-DNAM return door) survive JSON → ESP → JSON through tes3conv, with independent binary scan evidence and machine-readable verification. De-risks all later Cityforge door wiring; the family will host the T1.x site/plan/city render drivers. | [Detailed guide](guides/door_tes3conv_proof.md) |
| Cityforge (T0.2 site survey) | active | `tools/cityforge/build_site_survey.py` + `tools/cityforge/render_site.py` | Builds the deterministic Falkreath 7×7 D-SITE field/mask bundle and canonical `land_roads.json` directly from `tamriel.esm` normalized LAND/VTEX raw 78 over the target plus one-cell perimeter. The road view uses only the exact source mask, 8-neighbour components, and perimeter-confirmed continuation spans—never cleaned graph geometry—and records source/count/pixel agreement audit evidence alongside the four 4096² real-texture planner views with z=0 water and exact camera mapping. | [Tool entry](../tools/cityforge/2026-08-10_site_survey_tools.md) |
| Tamriel Road Centerlines v1 | active (source-space; **deprecated as world geometry**) | `tools/cityforge/build_tamriel_road_centerlines.py` | Full-map corrected-XCF extraction, measured bounded gap repair, topology-preserving skeleton graph, anchored corridor-bounded smooth TES3-GU vectors, canonical JSON/GeoJSON/SVG, audits, and required full-map/Falkreath visual review products. Raw Sand remains separate; raw 78 is source correlation only. **The bundle's world coordinates are registered 4096 GU west of the in-game LAND/VTEX grid: topology/provenance storage only — never consumed as world geometry. Use the aligned consumer product through `src/procgen/aligned_roads.py`.** | [Detailed guide](guides/tamriel_road_centerlines.md) |
| Aligned Road Centerlines (consumer product) | active | `tools/cityforge/build_aligned_road_centerlines.py` | Derives `output/mapdata/roads/tamriel_aligned_centerlines_v1/` from the committed source bundle by exactly (+4096 GU, +0 GU) and gates it against direct `tamriel.esm` LAND/VTEX-78: full-map census 391,101, Falkreath 1,275/1,275, five canary junctions at zero residual, no-shift canary must fail, per-edge corridor report with repaired bridge spans separate; refuses non-empty outputs and writes under mod/source roots; two Pillow proofs over the occupied-tile overlay. Central consumer API `src/procgen/aligned_roads.py` (fail-closed loader: source-space refusal, hash/translation/topology/invariant gates; lookup, rect queries, local frame, corridor width, nearest centerline, corridor polygons). | [Tool entry](../tools/cityforge/2026-08-11_aligned_road_centerline_tools.md) · [Detailed guide](guides/tamriel_road_centerlines.md) |
| Settlement Splitting and House-Unit Analysis | active | `tools/settlement_pipeline/split_units.py` | Splits merged settlement components into per-house units from destination-cell building keys (fuzzy colon-family grouping; different names = different houses), classifies boundary-role chains as connections / yard members / residuals, supports explicit audited fusion exclusions, retained-member assignments, shared `fused_member_assignments`, and `manual_detached` residuals, records fused fabric blocks, and emits deterministic split JSON. In mixed Markarth house/barrow components, exact barrow masonry/fort/stair families are contextual boundary wall/terrain pieces rather than shell fabric; exact complete Markarth tavern/guild body meshes are shell families; effective role overrides are recorded in `summary.json`. Manual access rulings keep complete paired assemblies together, including c45 Grimmir block-stone refs `002506` and `002511` beside its stair and c88 Water's Edge block/stair refs `001601`/`001603`. The split also classifies contact samples against the source LAND surface: `underground_only` shell contacts are retained as audit witnesses and cannot automatically connect units. `prepare_split_render.py --all-units` + `blender_split_render.py --units-only` render the textured per-unit 2x3 sheet library, while component maps use distinct non-magenta unit hues, magenta house-to-house links, tan infrastructure links, black directly anchored labels, colored legend swatches, black circular contact-only witnesses, and no map-body marker squares. `blender_render.py` can render prepared context scenes with terrain and excluded geometry, but the full Markarth acceptance requirement is global 50%-opacity textured terrain in every map, sheet, fused, connection, residual, and overview image. `package_split_render_library.py` then creates the flat building-key-named sheets and tiled overview pages. The completed 2026-08-10 baseline is documented in [the Markarth split-render acceptance checklist](guides/markarth_split_render_acceptance.md) and its final run report. | [Detailed guide](guides/settlement_splitting.md) |
| Stamp Library Derivation (D-STAMP v1) | active | `tools/cityforge/stamp_library.py` | Deterministically derives complete building *unit stamps* from existing extraction/split products only: member offsets anchored at the seed door (exact subtraction, world-aligned), verbatim source Euler rotations/scales, LAND re-measured terrain envelopes, hull-polygon-centroid access heading, footprint/hull, classification, and an exclusion ledger with exactly one reason per dropped candidate (audited `non_building_boundary`/`non_building_vehicle` overrides from the hash-pinned `tools/cityforge/non_building_audit_v1.json`, then derivation failures). Produces `output/cityforge/stamps/karthgad_nord_v1.json` (11 Karthgad stamps) and `markarth_side_stone_v1.json` (44 Markarth split-subset stamps, provisional/hash-pinned) plus a browsable `catalog_v1/` with hash-verified preview copies. Embedding replay evidence: all 744 members reconstruct their source absolute positions exactly and the read-only transform oracle reproduces every source placement matrix. Core logic lives in `src/procgen/citystamps.py`. | [Detailed guide](guides/stamp_library.md) |
| Markarth Stamp Palette (T0.5) | active | `tools/cityforge/stamp_palette.py` | Deterministically builds the browsable static stamp palette for the accepted final Markarth Side v2 extraction library: full source verification (existence, SHA-256, PNG dimensions for all 152 manifest assets), classification of all 105 standard sheets (103 eligible: 56 units / 27 connections / 19 residuals / 1 fused; 2 user-directed Castle Barracks exclusions quarantined as Needs Repair / Excluded with the exact reason), human naming (possessives, multi-party connections, suffixes, estate variants), deterministic lossless thumbnails derived from the actual final source sheets (138 PNGs with source-SHA provenance and nonblank validation), canonical `catalog.json` + single-file `file:///` `index.html` whose default view is the 56 eligible Building Units, with tabs/search/lightbox/supporting overview+textured-map links, and a double temp-run byte-determinism proof (including all thumbnails) before the canonical write into `<library>/stamp_palette_v1/`. Core logic lives in `src/procgen/stamp_palette.py` + `src/procgen/stamp_thumbnails.py`. | [Detailed guide](guides/markarth_stamp_palette.md) |
| Cityforge D-BRIEF Census (T0.5, dispatch 5) | active | `tools/cityforge/build_city_brief.py` | Builds the deterministic measured planner-vocabulary bundle for Falkreath v1 from hash-pinned inputs: 54 eligible stamps (11 Karthgad + 44 Markarth − user-quarantined Castle Barracks), derived `building_type_enum` + capability gaps (`lodge` unavailable, never fabricated), final terrain-backed palette preview resolution for every Markarth stamp (no stale split-render-v6 previews; merged verification section with 11/43 subcounts), same-run nearest-neighbor footprint-gap census with source-separated per-run distributions (positive NN p10/p50/p90 ≈ 20.0/135.8/3344.7 GU as **measured guidance, `usable_as_hard_minimum: false`**; collision clearance is the geometry solver's domain), door-step aggregation (n=76 stamps + n=470 ground-rules cross-check), LAND/VTEX census for R072 (191 cells, 48,896 tiles) and the 49-cell remap block (12,544 tiles) with plugin-local LTEX semantics (workspace-validated internal convention; load-order caveats emitted; OpenMW 0.51 internal confirmation unavailable from the connected index) and explicit scope separation, closed Phase-1 semantic-surface vocabulary with **explicit planned authoring assignments** (planned raw/index/id + masterless plugin scope and required local LTEX records; raw is never ordinal+1; road pinned 78/77), **live remap ESP cross-check gates (M-8: live `espland.load_ltex` table vs expected indices/ids/paths, road 77 protected, planned-vs-live agreement, palette live-remap evidence)** and 57 validation gates; two fresh builds are byte-compared and the determinism evidence (both staging hash sets) is embedded before the canonical four-file install (`kit_brief.json`, `region_palette.json`, `census.json`, `validation.json`). Core logic lives in `src/procgen/citybrief.py` + `src/procgen/regionpalette.py` + `src/procgen/censusio.py`. | [Detailed guide](guides/cityforge_dbrief_census.md) |
| Cityforge D-PLAN Validator + Overlay (T1.1, dispatch 6) | active | `tools/cityforge/validate_city_plan.py` + `tools/cityforge/render_plan.py` | Strict D-PLAN v1 gate for `city_plan.json`: recursive unknown-key rejection, frame/settlement pin to the accepted site survey (exact SHA-256), kit-brief vocabulary pins with capability gaps failing closed (lodge/stone_wall/fence), **aligned-centerline** road connections (loaded through `src/procgen/aligned_roads.py`; source-space bundle refused) with measured map-edge exits re-measured from the aligned network (`exit_<side>_<edge_id>`), exact yawed-footprint geometry for explicit and shared-deterministic-selector lots (scope/buildable/water/pairwise-overlap; nothing unresolved is claimed checked), palisade gate rules, linked terrain edits, closed texture zones, deterministic structured issues (error/warning, code, path, measured/limit) + input hashes, and a deterministic Pillow 2D overlay on the accepted 4096² `site_topdown.png` (banner/legend bands outside the map; refuses invalid plans; byte-identical reruns). Synthetic proof fixture under `output/cityforge/phase1/t1_1_validation_fixture/` is banner-labelled `SYNTHETIC VALIDATION FIXTURE - NOT A FALKREATH DESIGN`; no real Falkreath design is authored (that is the T1.6 user gate). Core logic lives in `src/procgen/cityplan.py`; machine-readable schema `src/procgen/schemas/city_plan_schema_v1.json` is emitted from the code. | [Detailed guide](guides/cityforge_cityplan_validator.md) |
| Cityforge visual settlement planner | active synthetic proof | `tools/cityforge/visual_planner.py` | Reusable Pillow planning canvas over exact D-SITE terrain/water/raw-VTEX evidence and the aligned road API; versioned visual-plan extension for source roads, authored streets, alleys, courts, plaza polygons, stamp/door intents; fail-closed accepted-palette/D-STAMP eligibility gate; separated hard-error/advisory analyser; local collision-aware labels, large intent arrows, capped dashed access stubs or explicit routed polylines, strong selected-lot highlight/detail panel, and four labelled synthetic proof PNGs. A distinct adversarial fixture is allowed only through the explicit proof path. It never authors TES3, consumes XCF/source-v1 road coordinates, runs Blender, or creates a real Falkreath plan. | [Detailed guide](guides/cityforge_visual_planning.md) · [Tool entry](../tools/cityforge/2026-08-11_visual_planner_tools.md) |
| Cityforge D-PLACE Houses-Only Placement (T1.2, dispatch 6) | active | `tools/cityforge/solve_city_placement.py` + `tools/cityforge/build_cityplace_fixture.py` | Consumes only a current zero-error T1.1 plan result and the accepted site/brief/palette/D-STAMP/centerline bundle; rechecks the shared selector, replays every eligible source member, runs the independent 37° multi-axis matrix oracle, seats exact plan yaws in engine-transform matrix space, measures dense-field terrain/doors/road access, rejects exact hull overlap/contact, emits provisional 256-GU-margin/512-GU-falloff pad requests, and produces raw TES3 transform evidence without authoring an ESP. Fine triangle/AABB collision is explicitly deferred when unavailable. Synthetic planned/final proof outputs are labelled NOT A FALKREATH DESIGN and include structured rejects plus a top-down placement diagnostic (not a city render). | [Detailed guide](guides/cityforge_cityplace.md) |
| Cityforge planning bundle (pre-design input) | active | `tools/cityforge/build_planning_bundle.py` | Precompiles the self-contained per-site input set for ONE visual town-design session: planning canvas PNG with labelled GU graticule (`src/procgen/planning_canvas.py`, shared projection for the later sketch renderer), eligible-only stamps.json + preview contact sheet (fail-closed eligibility policy; quarantined Castle Barracks/conn aliases excluded), site.json with clipped source roads, fixed rules.md, and a tool-written manifest. Refuses non-empty/protected output dirs; never authors TES3. | [Tool entry](../tools/cityforge/2026-08-12_planning_bundle_tools.md) |
| Cityforge sketch-to-plan derivation | active | `tools/cityforge/plan_sketch.py` | Turns a MINIMAL design-agent sketch (roads/spaces/lots, world GU) into the full format-v1 visual plan: strict sketch schema gate (fail-closed bundle stamp quarantine), lot x,y = footprint CENTROID (anchor derived internally as `lot.xy − rot(centroid_rel, yaw)`; stored plan position stays the anchor), automatic connection_targets (1536-GU snap; out-of-range endpoints left for the existing `road_disconnected` check), door intents/access links via the library door transform, one composite PNG on the byte-identical planning canvas (streets/alleys, plaza/court fills, kit-colored hulls, intent-colored door arrows along the MEASURED door facing `rotation.z + yaw`; radial fallback only), checks.json with ONLY hard errors + door/space facts (advisory codes deferred to placement), log.json bookkeeping; exit 0 iff zero hard errors, PNG always rendered. | [Tool entry](../tools/cityforge/2026-08-12_plan_sketch_tools.md) |
| Cityforge frontage fit v1 | active | `tools/cityforge/fit_intent_sketch.py` | Fits strict authored district/marker/stamp/frontage intent against full-precision stamp geometry, named roads/spaces/source-road targets, terrain/buildability masks, and conservative 2D clearance. Emits canonical intent copy, resolved sketch, and deterministic candidate/rejection/complete-search evidence; the global selector is a complete MRV/forward-check bitset search with a finite default 1,000,000-node budget, budget exhaustion reports a third `inconclusive` status (`search_budget_exhausted`) and never claims unsatisfiability, and the first feasible assignment is not claimed to be globally rank-optimal. The resolved sketch is rendered separately by `plan_sketch.py`. Explicit door targets are never replaced by nearest-target or `--auto-face` behavior. | [Tool entry](../tools/cityforge/2026-08-13_frontage_fit_v1_tools.md) |
| Cityforge hard landscape engine (T1.3, dispatch 6) | active synthetic proof | `tools/cityforge/build_cityscape_fixture.py` + `tools/cityforge/build_city_landscape.py` | Stitches the real Falkreath 49-cell LAND block into a 449×449 float64-GU field with exact seams and immutable border, applies strict analytic edits and exact T1.2 pad/re-seat, calibrates VNML axes/signs against real source, paints deterministic explicit raw VTEX classes with road/water/support gates, closes local LTEX records, validates masterless tes3conv JSON LAND records, and emits non-city terrain/VTEX diagnostics plus two-run hashes. It does not author/copy an ESP or a real Falkreath design. | [Detailed guide](guides/cityforge_cityscape.md) · [Tool entry](../tools/cityforge/2026-08-11_cityscape_engine.md) |
| Cityforge visual render host/worker (T1.5, dispatch 6) | active synthetic proof | `tools/cityforge/render_city.py` + `src/procgen/cityrender.py` | Builds a deterministic render-only scene from the accepted T1.2 final reseat and exact T1.3 terrain, copies only to a scratch masterless plugin, imports real resolved NIF/LTEX assets in Blender, applies exact placement matrices without normalization, clips the irregular z=0 water mesh to the final field, renders 11 required base views plus 7 focused single-lot door-height detail views, selects all 13 street/detail cameras from 32 final-terrain LOS/finite-edge candidates, and fails closed on missing textures, proxy/flat fallbacks, matrix drift, terrain mismatch, LOS/edge failures, readability defects, or incomplete PNG/audit output. The canonical fixture is labelled `SYNTHETIC RENDER FIXTURE - NOT A FALKREATH DESIGN`; it does not author a Falkreath plan or production ESP. | [Detailed guide](guides/cityforge_render_city.md) · [Tool entry](../tools/cityforge/2026-08-11_cityforge_render_tools.md) |
| Cityforge Stamp Volumes (T0.4/T0.4b) | active | `tools/cityforge/build_stamp_volumes.py` | Derives per-member stamp-local 3D bounding boxes + conservative below-source-ground classification for both stamp libraries (744/744 coverage, hash-verified manifests, fail-loud coverage gate, fresh-file output). v2 default: boxes are the AABB of each member's OBB in the BUILDING-ALIGNED frame (tight for members near 0 mod 90; `obb_source`/`obb_rotz_prime_deg`/`box_tight` recorded per member; OBB = A2 model local bounds × scale rotated by member rotz + offset, with the un-inflated-world-AABB/world-AABB fallback ladder); legacy v1 libraries still processable via `--karthgad-lib/--markarth-lib`. Emits `output/cityforge/stamps/stamp_volumes_v2.json` for the Z-aware overlap check (underground-only overlap is not a collision) and per-role footprint decomposition. | [Tool entry](../tools/cityforge/2026-08-12_stamp_volumes_tools.md) |
| Cityforge Stamp Orientation Normalization (T0.4b) | active (v2 shipped 2026-08-12) | `tools/cityforge/normalize_stamp_orientation.py` + `tools/cityforge/stamp_local_bounds.py` | Re-expresses a v1 D-STAMP library in a building-aligned frame: per-stamp theta = modal non-door rotz (mod 180, 0.5-deg buckets, majority, mean of winning bucket rounded 0.1 deg); members `offset' = Rz(-theta).offset`, `rotz' = rotz - theta`; hull/aabb/bounds from per-member OBBs (model local bounds × scale rotated by member rotz + offset, in the normalized frame; fallback ladder per member, recorded); `access_heading_rad - theta_rad`; door interior data untouched. Fail-closed replay gates (position ≤ 1e-6 GU, rotation ≤ 1e-9 deg, 744/744 passed) + shell-modal frame assert; door cardinality is REPORT-ONLY facts (20/70 doors off-cardinal on source geometry — lead ruling). Produced `karthgad_nord_v2.json` / `markarth_side_stone_v2.json`; propagation complete (volumes v2, Falkreath bundle v2, fixtures, benchmark, 196-test suite green). | [Tool entry](../tools/cityforge/2026-08-12_stamp_orientation_normalization.md) · [Run report](../.opencode/runs/stamp-normalization-2026-08-12/2026-08-12_stamp_normalization_report.md) |

### 2026-08-10 Split regression safeguard

`split_units.py` now refuses `--surface-manifest` without the matching
`--surfaces-dir`. The manifest alone can otherwise produce zero measured shell
contact and spurious fusions; the safeguard prevents a non-comparable render
root from being created. The targeted regression evidence is documented in
`tools/settlement_pipeline/2026-08-10_markarth_targeted_repair.md` and the
dated run report under `.opencode/runs/markarth-regression-repair-2026-08-10/`.

The follow-up c88 unfused review and the previously missing fused audit images
are documented in
`tools/settlement_pipeline/2026-08-10_markarth_targeted_repair.md` and the
run report under `.opencode/runs/c88-unfused-review-2026-08-10/`.

### 2026-08-10 Final Markarth extraction

The final full Markarth Side v2 render uses the corrected split with both
`--surface-manifest` and its matching `--surfaces-dir`, the terrain-relative
`underground_only` contact gate, and the final c88/c31 manual access rulings.
It produces 105 terrain-backed inventory slugs, 12 maps, 105 flat sheets, and
27 overview pages. The final source/render/library paths and independent
verification results are recorded in
`.opencode/runs/final-markarth-extraction-2026-08-10/`.

### 2026-08-10 Cityforge D-BRIEF census

The T0.5 dispatch-5 census family (row above) consumes the D-STAMP libraries,
the accepted final Markarth stamp palette, the remap ESP, `tamriel.esm`, the
site survey, and the Karthgad surveys to emit the four canonical planner
vocabulary files under `output/cityforge/briefs/falkreath_v1/`. Measured
facts fixed that day (post-review contract): exactly 54 eligible stamps
after the Castle Barracks quarantine; positive nearest-neighbor gap prior
p10/p50/p90 ≈ 20.0/135.8/3344.7 GU as measured guidance, not a hard minimum
(per-run Karthgad/Markarth distributions separated); R072 = 48,896 tiles
across 191 cells, effective block = 12,544 tiles across 49 cells (47 inside
R072, 2 in R014 — scopes kept separate); explicit planned authoring
raw/LTEX assignments per semantic surface with road pinned 78/77. The
adversarial review and its fixes are recorded in
`.opencode/runs/cityforge-t05-dbrief-census/2026-08-10_cityforge_t05_dbrief_census_review.md`
and the run report. Follow-up (M-8, same day): the live remap ESP's LTEX
table is read from the hash-pinned plugin via `espland.load_ltex` and gated
fail-closed against the expected indices/ids/texture paths (road index 77 =
`T_Hr_TerrRoadOH_01`), with the measured path/SHA/records emitted into
`region_palette.json#planned_output_plugin.live_remap_evidence`; the
validation bundle now runs **57 gates** (was 49).

## Documentation rules

New tool families should follow
[`tool_documentation_agent_guide.md`](tool_documentation_agent_guide.md). Mark
diagnostic, one-off, superseded, deprecated, and external tools explicitly;
do not present generated evidence products as reusable entry-point tools.

### 2026-08-11 Cityforge D-PLAN validator + overlay (T1.1)

The T1.1 family (row above) implements the strict `city_plan.json` contract
gate and the deterministic 2D plan-overlay renderer. It consumes the
accepted T0.x bundle (site survey, kit brief, region palette, D-STAMP
libraries, **aligned road consumer product** loaded through
`src/procgen/aligned_roads.py`) and the synthetic proof fixture
under `output/cityforge/phase1/t1_1_validation_fixture/` — explicitly
labelled `SYNTHETIC VALIDATION FIXTURE - NOT A FALKREATH DESIGN`. Road
geometry authority is direct `tamriel.esm` LAND/VTEX-78 occupancy; planners
consume only the aligned product (source-space bundle and XCF/BMP are
refused), map-edge exits are measured from the aligned network, and the
obsolete `roads_graph_clean.json` / raw-78-only `land_roads.json` are not
consumed. Tests: `tests/test_cityplan*.py` + `tests/test_aligned_roads.py`
(202 focused tests, all passing). No real
Falkreath design was authored and no placement was run; that remains the
T1.6 lead-driven user gate.

### 2026-08-12 Cityforge planning bundle (pre-design input)

The bundle builder (row above) precompiles the deterministic per-site input
set for the visual design session: canvas.png (terrain/water/cyan aligned
roads/cell lines + labelled 1024-GU graticule + faint unlabelled 512-GU
sub-graticule), eligible-only stamps.json
(54 for Falkreath) + one ≤4 MB preview contact sheet, site.json with clipped
simplified source roads, the fixed rules.md, and a timestamped manifest with
input/output hashes. The shared canvas projection
(`src/procgen/planning_canvas.py`, world-GU x-east/y-north orthographic) is
the contract the sketch renderer will reuse. First bundle run:
`output/cityforge/bundles/falkreath_v1/` (verified byte-deterministic on
re-run; manifest timestamp is the only varying byte).

### 2026-08-13 Repository export (public core)

`tools/repository/export_public_core.py` deterministically copies the core
implementation (`src/procgen/` + `tools/cityforge/` + selected
`Documentation/guides/` + `tool_documentation.md`) into a separate shareable
checkout `procedural-tamriel-core/` per `public_core_manifest.json`, then
writes the authored top-level files from `tools/repository/templates/`
(README, LICENSE, pyproject.toml, AGENTS.md, .gitignore,
`docs/architecture_overview.md`, synthetic examples). Fail-closed on a
non-empty destination; `--force` rebuilds. Never commits or pushes. Detail:
[`tools/repository/2026-08-13_repository_export_tools.md`](../tools/repository/2026-08-13_repository_export_tools.md).

## Planned index coverage

Settlement/building extraction, component classification, and landscape
texture remapping are **implemented** and indexed above. Future entries should
document the processed world-map/region tools, tes3conv/TESAnnwyn and other
external command-line utilities, groundcover, roads, and region definitions.

The mesh rendering/VLM tagging pipeline is **planned, not implemented**. Its
implementation-ready plan lives at the workspace root
[`2026-08-05_mesh_render_tagging_pipeline_plan.md`](../2026-08-05_mesh_render_tagging_pipeline_plan.md)
(kept outside `Documentation/guides/` on purpose) and will be indexed here
only when `tools/mesh_corpus/*` exists.
