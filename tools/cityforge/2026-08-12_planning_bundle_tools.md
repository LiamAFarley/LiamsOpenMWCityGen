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
districts, real eligible stamp ids, centroid markers, explicit named door
frontages, and optional composition declarations) and writes
`intent.copy.json`, `resolved.sketch.json`, and `fit_report.json` under one
fresh output directory.  Road composition declarations may provide `purpose`
(`urban_street`, `service_lane`, or `connector`) and an optional
`max_unsupported_frontage_gu` bound for the first two; lots may mark an
`intentional_outlier`; and `lot_groups` support the five characters
`compact_cluster`, `irregular_two_sided`, `formal_square`, `gateway_cluster`,
and `sparse_outskirts`, with shared targets, span/gap/non-outlier/side-run
bounds, authored along-order, and formal-square plaza sectors.

The pure search module is `src/procgen/frontage_fit.py`.  It materializes the
exact eight-key candidate fact contract consumed by
`src/procgen/composition_eval.py`, whose road/group metrics feed all nine hard
finding gates.  Composition intents retain all unary-feasible candidates for a
complete deterministic proof: default passes widen 64 -> 128 -> 256 -> 512 ->
the full retained domain, with a global default 1,000,000-node budget and
truthful distinction between `global_collision_unsatisfied`,
`global_relationship_unsatisfied`, and `search_budget_exhausted` (the last is
`inconclusive`, never proof).  A no-composition intent keeps one capped 64-wide
feasibility pass and does not run improvement.

After composition feasibility succeeds, the separate improvement budget is
50,000 nodes by default and `0` disables it.  It compares the fixed
lexicographic objective only within the successful feasibility-pass domain;
the hard-valid feasibility incumbent remains solved if improvement is disabled,
exhausted, or faulted.  The full `improvement` report includes
`faulted`/`fault_code` as well as domain, budget, traversal, rejection, and
incumbent/selected-objective evidence.  This is not semantic stamp selection,
road invention, or a global beauty score.  The resolved sketch is then
rendered separately by `plan_sketch.py`; do **not** pass `--auto-face`, which
would overwrite the fitter's explicit target transform.  An unsatisfied or
inconclusive intent exits with `FAILURE: frontage_fit ...` and never emits a
partial resolved lot set.
