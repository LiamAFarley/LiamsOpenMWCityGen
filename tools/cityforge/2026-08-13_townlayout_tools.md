# 2026-08-13 Cityforge V2 townlayout tools (Phases 1–18)

| Tool | Purpose | Status |
|---|---|---|
| `validate_town_layout.py` | Strict TownBrief / CityLayout v1 gate. `--brief` or `--layout` plus `--out`. `--emit-schema [PATH]` writes `src/procgen/schemas/town_layout_schema_v1.json`. | active (Phase 1) |
| `build_town_site_context.py` | Suitability SiteContext from D-SITE NPZ + census + TownBrief. Writes `site_context.json` + diagnostic overlay on `site_topdown.png`. No aligned roads. | active (Phase 3) |
| `build_town_approaches.py` | Rewrite-domain disk + aligned source approaches. Writes `site_approaches.json` + diagnostic overlay. | active (Phase 4) |
| `build_town_patches.py` | Organic Voronoi macro patches clipped to the rewrite domain. Writes `macro_patches.json` + diagnostic overlay. | active (Phase 5A) |
| `build_town_domain.py` | Grow connected inner city from patches (capacity/area stop). Writes `city_domain.json` + diagnostic overlay. | active (Phase 6) |
| `build_town_anchors.py` | Score inner patches and reserve market plaza (optional keep). Writes `anchors.json` + diagnostic overlay. | active (Phase 7) |
| `build_town_walls.py` | Palisade planning polygon + approach-driven gates. Writes `walls.json` + diagnostic overlay. | active (Phase 8) |
| `build_town_streets.py` | Topology A* + arterial/street classification. Wall/outskirts ring is not a through-street. Writes `streets.json` + diagnostic overlay. | active (Phase 10) |
| `build_town_fortification.py` | Stamp-first Stage 06 finalizer over a frozen Stage 05 road product. Validates the exact ring/gate/node contract, assigns explicit local `backs_to_wall` or connected 256-GU `wall_lane` strips, and writes `fortification.json` plus `fortification_diagnostic.png`. | active (stamp-first Stage 06) |
| `build_town_population.py` | Stamp-first Stage 07 frontage/rear/wall D-STAMP seating with full terrain, zoning, collision, local-repeat, parcel, and required-side gates. Failed runs retain `population.json`, `validation.json`, and `population_diagnostic.png` before returning nonzero so capacity handbacks remain auditable. | active (stamp-first Stage 07; current Falkreath checkpoint failed capacity) |
| `build_town_blocks.py` | Intersection cleanup, wards, and road-corridor insets. Writes `blocks.json` + Gate A diagnostic. | active (Phase 13) |
| `build_town_stamp_index.py` | Kit-brief + D-STAMP v2 capability index (54 eligible; Castle Barracks excluded). Writes `stamp_index.json`. | active (Phase 15) |
| `build_town_parcels.py` | Ward bisect into parcels + explicit alleys. Writes `parcels.json` + diagnostic overlay. | active (Phase 16) |
| `build_town_frontage.py` | Project parcel frontage arcs and prove regional-approach access. Writes `frontage.json`. | active (Phase 17) |
| `build_town_placement.py` | Exact-hull stamp seating on parcels (curb gaps 64–384 GU plus interior centroid poses). Writes `placement.json`. | active (Phase 18) |

Core: `src/procgen/townlayout/`. Geometry uses Shapely ≥ 2.0 (`geometry.py`); silent `buffer(0)` / `make_valid` forbidden. RNG is `stage_rng` via `derive_seed`. SiteContext query API is `sample()` — later stages must not reload the NPZ. Approaches consume `aligned_roads.load_aligned_network` only. Morphology generators emit a `MacroLayoutCandidate` gated by `candidate.require_macro_layout`.

`--out-dir` for the builders must be missing or empty.

Review products (not fixtures): `output/cityforge/townlayout/falkreath_phase3/`
through `.../falkreath_phase18/`. The accepted stamp-first Stage 06 checkpoint is
under `output/cityforge/townlayout/falkreath_phase20_stamp_first_city80/stage06_fortification/`.
