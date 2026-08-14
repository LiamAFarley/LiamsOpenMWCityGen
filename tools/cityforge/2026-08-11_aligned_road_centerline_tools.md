# 2026-08-11 — Aligned Road Centerline Tool Entry

## Purpose

Derives the **aligned consumer product** from the committed source-space
centerline bundle and proves its registration against direct `tamriel.esm`
LAND/VTEX-78 evidence.  Direct LAND/VTEX is the in-game occupancy authority;
the source bundle (and the XCF/BMP) are topology/provenance storage only.

## Active entry point

```powershell
python tools/cityforge/build_aligned_road_centerlines.py
```

Defaults (overridable): `--source-bundle-dir output/mapdata/roads/tamriel_source_centerlines_v1`,
`--base-esm tamriel.esm`, `--output-dir output/mapdata/roads/tamriel_aligned_centerlines_v1`.

## Responsibilities

- `src/procgen/aligned_roads.py` — the **one supported planner/generator
  entry point** for road geometry:
  - `load_aligned_network()` — fail-closed loader: refuses source-space
    paths, verifies the manifest/hash chain, the declared translation
    `(+4096 GU, +0 GU)`, pinned topology counts (3847 nodes / 4142 edges),
    and per-coordinate pixel round-trip invariants at the corrected
    registration;
  - `AlignedNetwork` API — node/edge lookup by stable id, world-GU
    rectangle queries, site-local frame conversion, corridor
    width/provenance/source-vs-repair status, nearest centerline
    point/tangent/distance/edge id, corridor polygons for plan collision
    checks, and alignment version + source hashes on every loaded network;
  - direct-LAND helpers — `load_esm78_tiles`, `nearest_road_tile_distance`,
    `registration_stats`, `edge_corridor_report` (repaired bridge spans
    reported separately).
- CLI gates (each failure aborts with `FAILURE: aligned road contract ...`):
  - output safety: refuses a non-empty output directory and any write under
    a mod/source root (`C:\Modding`, `Extra Reference Mods`, the source
    bundle, the workspace root);
  - source immutability: the source canonical/audit/alpha hashes and
    `tamriel.esm` are pinned by SHA-256;
  - exact translation: every world-GU node/edge/raw/smooth coordinate
    receives only `+4096 X / +0 Y` (350,149 coordinates; IDs, pixel
    coordinates, component/bridge IDs, provenance untouched);
  - direct-LAND proof: full-map raw-78 census **391,101**, Falkreath window
    **1,275** matching `land_roads.json`, five canary junctions at **0 GU**
    residual, aligned skeleton registration that decisively beats the
    **no-shift canary** (which must fail), per-edge corridor report;
  - final consumer reload through `load_aligned_network()`.

## Outputs

`tamriel_aligned_centerlines_v1.json`, `alignment_manifest.json`,
`nodes.geojson`, `edges.geojson`, `audit.json`, `audit.txt`, and two Pillow
proofs (`falkreath_alignment_full_site.png` 7×7 site at 8 px/tile,
`falkreath_alignment_central_cells.png` cells x=-93..-92, y=-9..-8 at
16 px/tile) drawn only after every numerical gate passes.  Thin 1 px aligned
vectors sit directly over the LAND/VTEX occupied-tile overlay; the legend
distinguishes LAND road tiles, source-derived vectors, repaired bridge
segments, nodes, T-junctions, and continuation exits.

## Consumer migration (2026-08-11)

T1.1 validation, map-exit measurement, and overlay context loading now use
the aligned product exclusively through `Bundle.from_paths` /
`load_aligned_network`.  Map-edge exits were re-measured from the aligned
network: the displaced frame's west exit is gone (the corrected road no
longer crosses the site border; investigation section 5.2).  D-BRIEF
`road_network_ref` pins the aligned product; the source-space bundle is
explicitly labelled topology/provenance storage.

## Tests

```powershell
python -m pytest -q tests/test_aligned_roads.py tests/test_road_centerlines.py
```

## Detailed guide

[`Documentation/guides/tamriel_road_centerlines.md`](../../Documentation/guides/tamriel_road_centerlines.md)
