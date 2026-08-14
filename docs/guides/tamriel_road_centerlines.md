# Tamriel Road Centerlines v1

## Purpose and status

`tamriel_road_centerlines_v1` reconstructs a deterministic, reviewable road
centerline graph from the authoritative visible `road network.png` layer in
the supplied `tesannwyn-vtex3.xcf`.  It is a **source-derived reconstruction**,
not an original creator vector file and not a TES3 plugin authoring stage.
The source mask, repaired corridor, skeleton graph, and smoothed world-GU
vectors remain separate so later land painting or settlement tools can choose
which layer they consume.

**Registration status (2026-08-11, binding):** the committed source-space
bundle (`output/mapdata/roads/tamriel_source_centerlines_v1/`) is
**topology/provenance storage only**.  Its world coordinates are registered
4096 GU (8 px) west of the in-game `tamriel.esm` LAND/VTEX grid and must
**never** be consumed as world geometry.  All planners/generators consume the
**aligned consumer product**
(`output/mapdata/roads/tamriel_aligned_centerlines_v1/`, derived by
`(+4096 GU, +0 GU)`) exclusively through `src/procgen/aligned_roads.py`.
Direct LAND/VTEX-78 tiles are the in-game occupancy authority; the XCF/BMP
are provenance only and are never planner inputs.  See the section
"Registration vs. tamriel.esm LAND" below and the alignment manifest.

The production reader is `gimpformats`.  The rejected temporary
interleaved-RLE decoder and its squiggly renders are not imported or used.
The installed package needs a scoped in-memory compatibility shim for the
unknown, size-framed XCF property 42; the shim is restored before extraction
returns and does not modify the installed package.

## Entry point

From the workspace root:

```powershell
python tools/cityforge/build_tamriel_road_centerlines.py
```

The default output is:

```text
output/mapdata/roads/tamriel_source_centerlines_v1/
```

The CLI refuses a non-empty output directory.  Use a different empty directory
for a rerun; compare `tamriel_road_centerlines_v1.json` and `audit.json` byte
for byte.  The command always processes the complete 4992×3040 source canvas;
the Falkreath 7×7-cell products are only a diagnostic crop.

Optional paths are available for controlled evidence runs:

```powershell
python tools/cityforge/build_tamriel_road_centerlines.py `
  --source-xcf <read-only tesannwyn-vtex3.xcf> `
  --source-bmp <read-only tesannwyn-vtex3.bmp> `
  --source-palette <read-only tes3ltex.txt> `
  --corrected-parity-png <corrected road_network_effective_full.png> `
  --output-dir <new empty directory>
```

## Pipeline and invariants

### 1. Source extraction

`src/procgen/road_source.py` loads the exact visible layer and asserts the
pinned contract before producing arrays:

- XCF v011, canvas 4992×3040, 16-bit gamma integer precision;
- exact layer name `road network.png`, top-first index 3;
- layer size 4992×3040, offset `(-8, 0)`, visible, opaque, Normal mode;
- four decoded channels with nonzero paint exactly `(0, 8, 112, 255)`;
- applied 4992×3040 mask at zero mask offset;
- GIMP normal mask modulation `alpha * mask // 255`, placed on the canvas
  with XCF offset clipping.

The returned `source_effective_alpha.npy` is the exact canvas alpha array,
including intermediate mask values.  `source_binary_mask.npy` is a separate
`effective_alpha > 0` topology view.  Neither is changed by repair.  The
reader records hashes for the XCF, layer alpha, layer mask, effective alpha,
and binary mask.  The CLI also compares decoded pixels and occupancy with the
corrected parity PNG before any topology work.  The expected source occupancy
is 399,600 pixels; a contract mismatch is a hard failure.

The supplied BMP is used only for evidence rendering.  Its header advertises
16-bit BI_BITFIELDS with standard 5-6-5 masks, while its payload is consumed
as raw little-endian VTEX indices.  The reader flips its bottom-up rows to the
XCF's north-up canvas orientation.  Raw VTEX 1 is recorded as **Sand** and is
never road authority.  Raw VTEX 78 is recorded as the source correlation
`MA_sulphur_rock02`.  The XCF/BMP are **provenance only** — never geometry
authority and never planner inputs; in-game authority is direct
`tamriel.esm` LAND/VTEX-78 occupancy (see "Registration vs. tamriel.esm
LAND").

### 2. Measured connectivity repair

`src/procgen/road_repair.py` skeletonizes the immutable source once for an
endpoint profile.  It measures all cross-component endpoint pairs within a
64-pixel profiling radius and, separately, nearest skeleton/corridor targets
for every endpoint.  A corridor target is the nearest degree-two or junction
pixel in another component; its local segment neighbours are retained so a
perpendicular T-junction is measurable rather than guessed from endpoint
pairing.  The repair threshold is selected from the actual combined nearest
distance distribution (`ceil(p90(...))`) with a minimum and a hard 32-pixel
cap.  The cap is 16,384 GU at source resolution and prevents arbitrary
continent/ocean links even if the measured distribution is broad.

Candidates are ordered by distance, family, and coordinates and evaluated with
a deterministic union-find.  A candidate is accepted only when it passes:

1. selected threshold and hard cap;
2. the source endpoint outward-heading check (and both endpoint headings for
   endpoint-to-endpoint candidates);
3. for corridor targets, an approximately perpendicular local tangent/normal
   check so an endpoint can attach to an interior road pixel;
4. a straight line-of-sight whose every pixel is within the bounded six-pixel
   source-dilation/land-locality corridor;
5. the union gate, which rejects redundant joins after already accepted bridges
   have connected the source components.

Accepted line segments are dilated with the configured one-pixel bridge radius
into `bridge_mask.npy`; source and bridge masks are never merged destructively.
`bridge_ledger.json` contains every endpoint pair and endpoint-to-corridor
candidate in the profiling radius with measured distance, target projection,
headings/normality, corridor values, union representatives, status, and
rejection reason.  Accepted rows carry stable content-derived bridge IDs,
family labels, and endpoint/component/target provenance.  This is deliberately
connectivity-biased: the selected rule allows plausible local extra joins, but
long, heading-inconsistent, non-perpendicular, or redundant candidates remain
explicit rejections.  Accepted counts are reported separately for
`endpoint_endpoint` and `endpoint_to_corridor`.

Before accepting a raster, the repair stage enumerates every source component
contacted by the dilated bridge under the configured 4/8-neighbour convention,
not only the candidate's two endpoint components.  Those labels/IDs are stored
in `touched_source_component_labels` / `touched_source_component_ids` and are
unioned immediately.  The ledger's `union_component_count` must equal the
actual repaired-mask component count; a mismatch is a hard failure.

### 3. Skeleton graph

`src/procgen/road_graph.py` skeletonizes the repaired mask a second time for
the final graph.  Adjacent non-degree-two pixels are clustered into one node;
maximal degree-two chains are traced exactly once using undirected skeleton
segment signatures.  A component with no natural node receives a deterministic
loop-anchor node and a self-loop edge.  Node members and edge raw pixel chains
jointly cover the final skeleton exactly.  Validation rejects missing pixels,
extra pixels, duplicate chains, invalid references, or node-degree mismatches.

Node and edge IDs hash component/content data rather than scan counters.  Node
records carry pixel position, graph degree, kind, component ID, and cluster
membership.  Edge records carry both endpoint IDs, raw pixel chain, source vs.
bridge status, and bridge IDs.

### 4. Smooth vectors and world coordinates

`src/procgen/road_vectors.py` derives a scale-aware simplification tolerance
from the one-pixel source quantization and each chain's measured step scale
(configured floor 2.0 px, multiplier 2.0, cap 3.0 px).  It fits a deterministic
centripetal Catmull–Rom curve and samples it densely at 0.75 px, restoring the
raw first/last anchors exactly.  If RDP produces exactly two controls on a
non-degenerate open chain, the stage emits a densely sampled
`straight_line_simplified` fit rather than retaining the raster chain.  Raw and
smooth polylines are both densified at 0.5 px for Hausdorff, corridor, and
raster-coverage evidence, so sparse anchor segments are measured continuously.
Curves are checked against the repaired corridor with a documented three-pixel
maximum deviation and against genuine non-adjacent self-crossings; dense-sample
re-rounding within four travelled pixels is not misclassified as a loop.  Raw
fallbacks remain only for closed-loop degeneracy, corridor failure, or genuine
self-induced-loop safety failure, and every fallback edge ID/reason is retained
in the metrics.

World transform (source-space, kept for provenance):

```text
origin cell       (-254, -130)
pixels per cell   16
GU per pixel      512
source row 0      north; TES3 +Y is north, so y is flipped around 3040 px
pixel center      [x + 0.5, y + 0.5]
```

For source pixel `(px, py)` the exported point is:

```text
GU_x = (-254 * 16 + px + 0.5) * 512
GU_y = (-130 * 16 + 3040 - py - 0.5) * 512
```

**These source-space coordinates are deprecated for direct world-geometry
consumption** (see "Registration vs. tamriel.esm LAND" below); the aligned
consumer product adds exactly `+4096 GU` to X.

## Registration vs. tamriel.esm LAND

Measured 2026-08-11 (see
`.opencode/runs/cityforge-road-authority-alignment/2026-08-11_road_authority_alignment_investigation_report.md`):

- Direct `tamriel.esm` LAND/VTEX is the **in-game occupancy authority**:
  OpenMW-normalized raw-VTEX-78 tiles at 512 GU, tile center
  `(cx*8192 + tx*512 + 256, cy*8192 + ty*512 + 256)`.
- The source raster is registered exactly **8 px (4096 GU) west** of the
  ESM tile grid: `esm78 == raster78 sampled at px - 8` for
  **391,101 / 391,101 tiles** (full map) and **1,275 / 1,275** in the
  Falkreath window; at the committed registration only 68,326 / 391,101
  agree.  The offset is a pure rigid translation: `dX = +4096 GU`,
  `dY = 0`.
- **Aligned transform** (the consumer product):
  `GU_x' = (-254*16 + px + 8 + 0.5) * 512 = (px - 4055.5) * 512`,
  `GU_y' = (959.5 - py) * 512`.
- The XCF/BMP are **provenance only**.  They are never reopened as geometry
  authority and are never planner inputs.

Consequences (binding):

1. `tamriel_source_centerlines_v1/` keeps topology, IDs, pixel coordinates
   and provenance byte-identical; its world-GU fields are labelled
   source-space and must not feed world geometry.
2. The aligned consumer product
   `output/mapdata/roads/tamriel_aligned_centerlines_v1/` carries every
   world coordinate +4096 X / +0 Y with a locked alignment manifest; it is
   the only product planners load, through `src/procgen/aligned_roads.py`.
3. `src/procgen/aligned_roads.py` fails closed on the source-space bundle,
   hash drift, translation/topology drift, and coordinate invariants.
4. The build CLI
   `tools/cityforge/build_aligned_road_centerlines.py` derives the aligned
   product from the committed source bundle, re-measures the registration
   against direct LAND reads, and refuses non-empty output directories and
   any write under a mod/source root.

Aligned product files:

| File | Role |
|---|---|
| `tamriel_aligned_centerlines_v1.json` | canonical aligned product (nodes/edges, alignment section, determinism) |
| `alignment_manifest.json` | translation, hashes, direct-LAND proof (391101/391101, 1275/1275, canary junctions, no-shift canary) |
| `edges.geojson`, `nodes.geojson` | aligned world-GU GIS exports |
| `audit.json`, `audit.txt` | quantitative audits incl. per-edge corridor checks |
| `falkreath_alignment_full_site.png` | Pillow proof: 7×7 Falkreath site, vectors over LAND tiles |
| `falkreath_alignment_central_cells.png` | Pillow proof: cells x=-93..-92, y=-9..-8 at 16 px/tile |

Consumers (T1.1 validator, overlay renderer, T1.2 solver, T1.3 landscape)
load the aligned product through `Bundle.from_paths(..., centerlines=...)` /
`aligned_roads.load_aligned_network(...)`; the aligned product hash appears
in every validation `input_hashes` and `summary.external_references`.

Canonical edges retain raw pixel/GU chains, smooth pixel/GU polylines,
estimated width, raw and smooth lengths, sampled symmetric Hausdorff distance,
raw-vs-smooth turning metrics, high-frequency turn/zigzag reduction counts,
corridor deviation, endpoint displacement, and provenance.  The aggregate audit
uses fixed-angle high-frequency turns for reduction because dense smooth
sampling naturally increases the number of local triplets.

## Bundle contents

The canonical file is `tamriel_road_centerlines_v1.json`.  It includes source
hashes, transform metadata, algorithm/version/settings, repair settings and
the complete bridge ledger, stable nodes/edges/components, graph validation,
raw/repaired/skeleton/topology statistics, visual hashes, and a deterministic
payload hash basis.

Important sidecars:

| File | Role |
|---|---|
| `source_effective_alpha.npy` | lossless immutable effective XCF alpha |
| `source_binary_mask.npy` | immutable binary topology source |
| `repaired_mask.npy` | source plus accepted bridge corridors |
| `bridge_mask.npy` | accepted repair pixels only |
| `final_skeleton.npy` | final graph skeleton |
| `bridge_ledger.json` | schema-2 complete accepted/rejected endpoint and corridor candidate evidence |
| `audit.json`, `audit.txt` | quantitative machine/human audits |
| `edges.geojson`, `nodes.geojson` | convenient world-GU GIS exports |
| `centerlines.svg` | full-map vector review export |
| `source_metadata.json` | pinned source and parity metadata |

Required visual products:

1. `full_source_mask.png`
2. `full_repaired_mask.png`
3. `full_centerlines.png`
4. `full_centerlines_over_source.png`
5. `falkreath_source_mask_8x.png`
6. `falkreath_repair_bridges_8x.png`
7. `falkreath_centerlines_8x.png`
8. `falkreath_centerlines_over_texture_8x.png`
9. `falkreath_junction_component_diagnostic.png` with legend

The source/repair masks are pixel-faithful raster products.  Centerline review
images render vector samples directly at 8× in Falkreath so smooth bends are
not merely enlarged staircases.  The texture overlay preserves the underlying
VTEX classes; it does not reinterpret Sand as road.

## Tests and verification

Focused tests:

```powershell
python -m pytest -q tests/test_road_centerlines.py tests/test_aligned_roads.py
```

They cover the real XCF contract/occupancy, measured endpoint and
endpoint-to-corridor profiling, immutable source plus bridge ledger,
T-junction repair, cross and loop graph tracing, deterministic IDs, pixel/GU
inversion, scale-aware staircase smoothing/anchors, bounded turn reduction,
and malformed input failures — plus the aligned consumer contract: transform
canaries, network API (lookup/queries/local frame/nearest/corridors), the
fail-closed loader gates (source-space refusal, hash/translation/topology/
invariant drift), the real aligned product, T1.1 aligned bundle loading with
re-measured map exits, and direct-LAND agreement (esm-78 census, Falkreath
occupancy 1275/1275, canary junctions at zero residual, no-shift canary
failure, per-edge corridor reporting).

The full build itself is the production integration check.  After a build:

```powershell
python -c "import json; from pathlib import Path; p=Path('output/mapdata/roads/tamriel_source_centerlines_v1'); c=json.loads((p/'tamriel_road_centerlines_v1.json').read_text()); print(c['statistics']['graph_validation']['valid'])"
```

The expected result is `True`; audit component areas are checked against their
bounding-box areas, all edge references resolve, graph/skeleton coverage is
exact, smooth endpoint displacement is zero, and the corrected parity pixels
are identical.  For deterministic rerun evidence, build into a second empty
directory and compare the two canonical/audit files byte-for-byte.

The aligned consumer product is built and verified with:

```powershell
python tools/cityforge/build_aligned_road_centerlines.py
```

It refuses a non-empty output directory and any write under a mod/source
root, verifies the committed source bundle and `tamriel.esm` by SHA-256,
applies exactly (+4096 GU, +0 GU) to every world coordinate, and gates the
result against direct LAND reads (391,101 full-map raw-78 tiles, 1,275
Falkreath tiles matching `land_roads.json`, five canary junctions at zero
residual, a no-shift canary that must fail, and a per-edge corridor report
that keeps repaired bridge spans separate).  The two proof PNGs are rendered
only after every numerical gate passes.

## Limitations and provenance

These are reconstructed centerlines from a 512-GU raster source, not supplied
creator splines.  The source layer contains many small branches and loops; the
pipeline preserves them rather than silently pruning them.  Smooth vectors
improve map readability but do not add semantic road hierarchy, bridge/ford
types, or TES3 LAND records.  The source-space bundle's world coordinates are
registered 4096 GU west of the in-game grid and are deprecated for world
geometry; consumers must use the aligned product through
`src/procgen/aligned_roads.py`.  Visual quality remains a lead/user review
decision; the implementation report records direct image observations and
quantitative checks without claiming aesthetic acceptance.
