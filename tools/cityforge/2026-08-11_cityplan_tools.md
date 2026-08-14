# 2026-08-11 Cityforge D-PLAN validator + overlay (T1.1)

| Tool | Purpose | Status |
|---|---|---|
| `validate_city_plan.py` | Strict D-PLAN schema/semantic/geometric gate for `city_plan.json`; deterministic structured issue list, input hashes, exit 0/1/2. Also `--emit-schema`. | active (T1.1) |
| `render_plan.py` | Deterministic 2D overlay of a *validated* plan on `site_topdown.png` (Pillow only); banner/legend bands outside the map; render audit JSON. | active (T1.1) |
| `build_cityplan_fixture.py` | Deterministic synthetic fixture builder (NOT a Falkreath design) with self-validation; writes the labelled plan + manifest under `output/cityforge/phase1/t1_1_validation_fixture/`. | active (T1.1) |

Core logic: `src/procgen/cityplan.py` (contract, geometry, bundle
handling, shared deterministic stamp selector). Machine-readable schema:
`src/procgen/schemas/city_plan_schema_v1.json` (emitted from the code, so
it cannot drift). Detailed guide:
`Documentation/guides/cityforge_cityplan_validator.md`.

What this family is **not**: it does not author plans, does not place
anything, and does not write plugin data. The first real Falkreath plan is
a lead-driven T1.6 user-gated step; T1.1 is proven on the synthetic
fixture only. Road geometry authority is direct `tamriel.esm` LAND/VTEX-78
occupancy; planners consume the **aligned consumer product**
(`output/mapdata/roads/tamriel_aligned_centerlines_v1/`) only through
`src/procgen/aligned_roads.py` (the source-space bundle and the XCF/BMP
are provenance only and are refused by the loader). Map-edge exits are
measured from the aligned network by `cityplan.measure_map_exits`
(`exit_<side>_<edge_id>`; re-measured 2026-08-11 — the displaced frame's
west exit no longer exists), and the obsolete `roads_graph_clean.json` /
raw-78-only `land_roads.json` are never consumed.

Usage examples and exit codes are in the guide.
