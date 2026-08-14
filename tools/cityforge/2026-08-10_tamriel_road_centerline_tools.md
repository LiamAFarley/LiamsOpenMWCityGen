# 2026-08-10 — Tamriel Road Centerline Tool Entry

> **Registration caveat (2026-08-11, binding):** the bundle produced by the
> tool below is the **source-space** product.  Its world coordinates are
> registered 4096 GU (8 px) west of the in-game `tamriel.esm` LAND/VTEX grid;
> it is topology/provenance storage, **not** world geometry.  Consumers must
> use the **aligned consumer product** (`output/mapdata/roads/tamriel_aligned_centerlines_v1/`,
> derived by `(+4096 GU, +0 GU)`) through `src/procgen/aligned_roads.py`.
> See [2026-08-11_aligned_road_centerline_tools.md](2026-08-11_aligned_road_centerline_tools.md)
> and the alignment manifest for the proof and equations.

## Active entry point

`build_tamriel_road_centerlines.py` runs the full-map road centerline pipeline.
It is not a toy renderer and does not read the rejected old road vectors.

```powershell
python tools/cityforge/build_tamriel_road_centerlines.py
```

## Responsibilities

- `src/procgen/road_source.py` — production `gimpformats` XCF extraction,
  metadata gates, corrected-parity comparison, raw VTEX evidence reading, and
  source hashes.
- `src/procgen/road_repair.py` — one source skeleton for measured endpoint and
  endpoint-to-interior-corridor profiling, perpendicular T-junction candidates,
  union-aware bounded deterministic bridge decisions, component audits, and the
  complete family-labelled accepted/rejected ledger.  Accepted dilated bridges
  enumerate and union every source component they touch; union/repaired counts
  must reconcile.
- `src/procgen/road_graph.py` — final repaired-mask skeleton, clustered nodes,
  maximal chains, loop handling, and exact graph/skeleton validation.
- `src/procgen/road_vectors.py` — scale-aware RDP simplification, bounded
  centripetal Catmull–Rom sampling, explicit dense straight-line fitting for
  two-control routes, continuous-polyline Hausdorff/coverage metrics, exact
  anchors, TES3 world-GU transform, raw-vs-smooth high-frequency turn/zigzag
  metrics, and documented unsafe-only fallbacks.
- `src/procgen/road_outputs.py` — canonical JSON, NumPy sidecars, bridge/audit
  documents, GeoJSON/SVG, and full-map/Falkreath review images.

## Inputs and output boundary

The default read-only inputs are the supplied `tesannwyn-vtex3.xcf`,
`tesannwyn-vtex3.bmp`, `tes3ltex.txt`, and corrected parity PNG under the
OpenCode temp evidence directory.  The default generated bundle is
`output/mapdata/roads/tamriel_source_centerlines_v1/`.  The CLI refuses to
overwrite a non-empty output directory.  Original XCF/BMP/palette files,
`tamriel.esm`, `C:/Modding`, existing Cityforge site outputs, and rejected old
vectors are never modified.

## Validation signals

The source stage must report parity-pixel identity and 399,600 effective road
pixels.  The repair audit reports measured nearest-endpoint distributions,
selected threshold, source/repaired components, and every bridge candidate.
The graph audit must be valid with exact skeleton coverage and zero duplicate
chains.  Repair metrics must report endpoint-endpoint and endpoint-corridor
accepted counts separately.  Vector metrics must report exact endpoints, finite
coordinates, bounded corridor deviation, and positive high-frequency
turn/zigzag reduction.  The required nine PNGs are directly inspected as part
of each full-map run.

Detailed pipeline semantics and artifact descriptions are in
[`Documentation/guides/tamriel_road_centerlines.md`](../../Documentation/guides/tamriel_road_centerlines.md).
