# 2026-08-12 planning bundle tools

`build_planning_bundle.py` precompiles the per-site "planning bundle": the
complete, self-contained input set for ONE visual town-design session.  A
design agent reads only the bundle directory (canvas.png, stamps_sheet.png,
stamps.json, rules.md, site.json) and never opens catalogs, stamp libraries,
or preview trees.

CLI (canonical Falkreath run):

    python tools/cityforge/build_planning_bundle.py \
      --site-name falkreath_v1 \
      --survey-dir output/cityforge/sites/falkreath_v1 \
      --roads-dir output/mapdata/roads/tamriel_aligned_centerlines_v1 \
      --stamp-libraries output/cityforge/stamps/karthgad_nord_v2.json \
                      output/cityforge/stamps/markarth_side_stone_v2.json \
      --palette "output/settlement-splits/markarth-side-v2/final-markarth-extraction-2026-08-10-library/stamp_palette_v1/catalog.json" \
      --catalog-index output/cityforge/stamps/catalog_v1/index.json \
      --cells=-93..-92,-9..-8 --out output/cityforge/bundles/falkreath_v1

Notes: values that begin with `-` must use the `--cells=...` equals form
(argparse treats a leading dash as an option).  `--out` must be fresh
(non-empty directories are refused) and outside every configured data root
(`configs/procgen.json` `paths.data_roots`) and `C:\Modding`.  The stamp
libraries are the **v2 building-aligned** libraries (T0.4b normalization):
stamps.json door headings are building-relative (cardinal at yaw 0), while
`stamps_sheet.png` previews still show the SOURCE orientation (the source
rotation is recorded per stamp as `normalization_theta_deg`).

Outputs (all deterministic except the manifest's UTC timestamp):

* `canvas.png` — terrain hillshade + exact water, cyan aligned source roads
  with corridor fills and edge-id labels, cell boundary lines every 8192 GU,
  labelled GU graticule (every 1024 GU; `x=` labels top, `y=` labels left),
  plus a FAINT UNLABELLED sub-graticule every 512 GU (visually subordinate
  to the 1024-GU graticule; the stronger line wins where they coincide),
  title band with site name + rectangle, bottom legend strip;
* `stamps.json` — one entry per ELIGIBLE stamp (54 for Falkreath; the
  quarantined `markarth_side_v1__u114_castle_barracks` and
  `markarth_side_v1__conn_114_1` are excluded by the fail-closed eligibility
  policy): kit, building type, size class, footprint dims, stable source
  `door_id` values plus door offsets + heading (relative to stamp origin),
  terrain envelope, style tags;
* `stamps_sheet.png` — one contact sheet (a `stamps_sheet_2.png` only if a
  single sheet would exceed 4 MB) of catalog preview thumbnails labelled with
  short id, footprint, door count; sorted by kit then size class;
* `site.json` — site name, cells, world-GU rectangle, clipped + simplified
  source-road chains (128 GU Douglas-Peucker), notes;
* `rules.md` — the fixed city-design rules text (site name / cells
  substituted).  The "Composition rules" section (rewritten 2026-08-12 per
  the Falkreath v1 failure analysis) now mandates corridor/band clearance
  (frontage = adjacent, never overlapping), center-then-sparse-outward
  density, no same-orientation stacking, no symmetric plazas, and buildings
  + circulation only;
* `bundle_manifest.json` — UTC timestamp, input paths+sha256, output
  paths+sha256, eligible/quarantined counts.

Shared canvas logic lives in `src/procgen/planning_canvas.py`:
`render_planning_canvas(terrain, world_bounds_gu, network, site_name=...)`
returns `(image, CanvasProjection)`; the projection is the single
orthographic north-up world-GU → pixel mapping (`world_to_px` / `px_to_world`)
that the later sketch renderer must reuse so sketch geometry lands on the
identical grid the agent read coordinates from.  Canvas coordinates are TES3
world GU (x east, y north); world GU = survey plan GU + survey frame origin.

Invariants: no random generators, no Blender, no TES3 authoring, no edits to
originals; `stamps.json` contains only measured library numbers (the accepted
library records one burial depth, so both envelope bounds carry it); roads
come exclusively from the aligned consumer product via
`src/procgen/aligned_roads.py`.

Legibility contract (added 2026-08-12 review fix): graticule labels anchored
within half a gridline of a map edge are skipped (corner clipping), labels
are drawn ABOVE road/corridor fills (PIL polygon fills replace overlay
pixels, so roads must come first), and contact-sheet tile labels are
width-constrained by `fit_label_lines()` — font shrink then underscore wrap,
never drawn past the tile bounds (full id on two lines when needed).

## 2026-08-13 frontage-fit v1

`fit_intent_sketch.py` consumes a strict authored intent (roads, spaces,
districts, real eligible stamp ids, centroid markers, and explicit named door
frontages) and writes `intent.copy.json`, `resolved.sketch.json`, and
`fit_report.json` under one fresh output directory.  The pure search module is
`src/procgen/frontage_fit.py`; it uses full-precision manifest-pinned stamp
hulls/doors, exact target assignments, conservative 2D clearance, and a
complete deterministic MRV/forward-check bitset search with a finite default
1,000,000-node budget (first feasible assignment, not globally rank-optimal;
see the frontage-fit v1 tool entry for the outcome mapping and metrics).  The
resolved sketch is then rendered by the existing `plan_sketch.py` command.  Do
**not** pass `--auto-face` to the resolved sketch: that legacy helper would
overwrite the fitter's explicit target transform.  An unsatisfiable intent
exits with `FAILURE: frontage_fit ...` and never emits a partial resolved lot
set; exhausting the search budget is a third status, `inconclusive` with
terminal code `search_budget_exhausted`, which never claims unsatisfiability.
