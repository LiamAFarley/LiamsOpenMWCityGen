# Procedural Tamriel — Core

A deterministic, code-driven pipeline that procedurally generates unpopulated
parts of **Tamriel** for **OpenMW 0.51** (the TES3 / Morrowind engine).  This
checkout is the *core implementation reference*: the engine package, the
city-generation driver scripts, and the architecture guides that explain how
the pipeline acquires its data and turns it into placed, rendered, and
authorable city plans.

This is **not** a runnable end-to-end pipeline from a clean clone.  It is a
sharable, inspectable snapshot of the core architecture so a reviewer can
understand exactly how a city is generated and where the data comes from.
The full workspace (settlement extraction, split/component staging, the large
test suite, `tamriel.esm`, the mod data roots) is deliberately not included.

## What the pipeline does

For a given city site (this work's reference site is **Falkreath**, in
`tamriel.esm` region R072, around cells x=-95..-89, y=-11..-5):

1. **Site survey** — read the real `tamriel.esm` LAND/VTEX records for the
   site and its perimeter, build elevation / water / buildability / road-mask
   fields, and derive canonical road polylines (`build_site_survey.py` +
   `src/procgen/citysite.py`, `espland.py`, `landroads.py`).
2. **Road centerlines** — the source road network is reconstructed from the
   XCF source mask, repaired, skeletonized, vectorized, and **registered to
   the in-game LAND grid** (`src/procgen/road_*.py`,
   `tools/cityforge/build_tamriel_road_centerlines.py`,
   `build_aligned_road_centerlines.py`, consumed through
   `src/procgen/aligned_roads.py`).
3. **Stamp libraries** (produced beforehand) — per-building unit stamps with
   member offsets, Euler rotations, footprints, and **measured door outward
   headings**.  Derived from accepted extraction/split products; consumers
   load them as JSON through `src/procgen/citystamps.py`.
4. **Planning bundle** — a self-contained per-site input set: planning canvas
   PNG with GU graticule, eligible `stamps.json` + preview contact sheet,
   clipped `site.json`, and fixed design rules (`build_planning_bundle.py`).
5. **Frontage fitting** — author roads/spaces/lots/door intents in a sketch
   (`fit_intent_sketch.py` + `src/procgen/frontage_fit.py`), fit every lot
   against full-precision stamp geometry, terrain, buildability, and door
   targets with a complete MRV/forward-check search, then measure composition
   preferences (`src/procgen/composition_eval.py`).
6. **Plan derivation and validation** — turn a resolved sketch into a full
   visual plan (`tools/cityforge/plan_sketch.py`) and gate it with the
   strict D-PLAN validator (`validate_city_plan.py` + `src/procgen/cityplan.py`).
7. **Placement** — seat plan yaws into engine-transform matrix space, measure
   terrain/door/road access, and reject hull overlap/contact
   (`solve_city_placement.py` + `src/procgen/cityplace*.py`,
   `engine_transform.py`).
8. **Landscape editing** — apply analytic terrain edits (pads, flattening)
   and explicit VTEX texture classes as a masterless ESP through the cityscape
   engine (`build_city_landscape.py` + `src/procgen/cityscape*.py`).
9. **Rendering** — import the real resolved NIF/LTEX assets in Blender and
   render city/plan/site views (`render_city.py`, `render_site.py`,
   `render_plan.py` + `src/procgen/cityrender.py`, `cityplan.py`).
10. **ESP authoring** — generate masterless `.esp`/`.omwaddon` plugins
    (TES3 JSON through `src/procgen/tes3json.py`), never declaring
    `tamriel.esm` as a master.

## Repository layout

```
procedural-tamriel-core/
├── README.md                  this file
├── pyproject.toml             dependency metadata (see below)
├── AGENTS.md                  agent guidance for working in this core
├── src/procgen/               the engine package (all modules)
├── tools/cityforge/           city-generation driver scripts + tool notes
├── docs/
│   ├── architecture_overview.md   the full pipeline, stage by stage
│   ├── tool_documentation.md      tool-family index (from the workspace)
│   └── guides/                    cityforge + road + rotation guides
└── examples/                  synthetic (not real-asset) contract examples
```

## Dependencies

The engine is pure Python with a small, standard scientific stack:

* **NumPy**, **Pillow**, **SciPy**, **scikit-image** (core computation,
  raster handling, morphology, geometry)
* **zstandard** (used by `tes3json.py` for optional compressed blobs)
* **gimpformats** (used only by the road *source* tools that decode the XCF
  mask — optional if you only read the aligned centerline products)
* **bpy** (Blender) is required **only** for the render drivers
  (`render_city.py`, `render_site.py`); it is not a normal pip install.

See `pyproject.toml` for the dependency groups.

## Three capability tiers

1. **Pure / synthetic** — everything in `src/procgen/` plus the non-Blender
   `tools/cityforge/` drivers.  These read/write JSON, PNG, and NumPy files
   and work without any Morrowind data.  The `examples/` fixtures exercise
   these contracts.
2. **Data-backed** — tools whose inputs come from `tamriel.esm` or the mod
   data roots (site survey, road centerlines, stamp libraries, ESP authoring
   verification).  They run correctly in the full workspace; from this core
   checkout they need user-supplied data at the same paths.
3. **Blender tools** — `render_city.py` / `render_site.py` (and the 
   `blender_render_city.py` helper) require Blender, the NIF importer addon,
   and the resolved mesh/texture assets.

## Workspace-specific defaults in the code

Some drivers carry hard-coded workspace defaults (paths such as `C:\Modding`,
`F:\ProcGenWorkspace`, `configs/procgen.json`, or a local Blender/tes3conv
location, and `tamriel.esm` as the source ESM).  Those are used when the tool
runs in the original workspace; from this core checkout you must supply the
equivalent data/config at those locations or override via CLI/config.  They do
not affect the pure/synthetic modules in tier 1.  Nothing in this checkout
ever requires those paths to be present to be *read*.

## Tests

The workspace contains an extensive deterministic test suite (geometry
conventions, frontage fit, composition, road centerlines, site survey, etc.).
It is intentionally **not** copied here; the core is meant for inspection, and
the suite depends on workspace fixtures.  See `docs/architecture_overview.md`
for the invariants the tests enforce.

## License

See `LICENSE`.  This core is architecture/algorithmic code; the game assets
and mod data it consumes are the property of their respective owners and are
**not** distributed here.
