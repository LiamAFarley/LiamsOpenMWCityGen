# 2026-08-13 frontage-fit v1

`fit_intent_sketch.py` is the deterministic fit stage after a designer writes
an intent document and before the existing `plan_sketch.py` renderer.  It
loads the bundle's manifest-pinned, full-precision D-STAMP libraries and
aligned source roads, adds authored roads/spaces as named targets, and solves
each lot's exact centroid/yaw without changing its stamp, role, district,
marker, road, space, or target choices.

## CLI

```text
python tools/cityforge/fit_intent_sketch.py \
  --bundle output/cityforge/bundles/falkreath_v1 \
  --intent <intent.json> \
  --out <fresh-output-dir>
```

Outputs are `intent.copy.json`, `resolved.sketch.json`, and `fit_report.json`.
An impossible but well-formed intent writes an `unsatisfied` report with an
empty resolved lot list and exits 1.  Exhausting the deterministic search
budget writes `inconclusive` (terminal `search_budget_exhausted`), also exits
1, and never claims unsatisfiability.  Unknown/quarantined stamps, doors, and
targets fail closed before search.  The fitter output is passed to
`plan_sketch.py` without `--auto-face`; that legacy option would overwrite the
explicit-target transform.

The pure geometry/search implementation is
`src/procgen/frontage_fit.py`.  Shared target distance and corridor geometry
lives in `src/procgen/frontage_targets.py` so the fit report and the renderer's
explicit door facts use the same practical source-road and authored-target
semantics.  No image, Blender, TES3, or subprocess stage belongs to this tool.

## Semantics notes (2026-08-13 closure)

- **Candidate counts are plan-literal.** `candidate_count_generated` counts
  every constructed (offset × gap × yaw) sample; `candidate_count_deduplicated`
  counts unique rounded `(centroid_x, centroid_y, yaw)` keys at 0.1 GU/degree,
  deduplicated **before** unary evaluation (plan §6.3), so it equals
  generated − construction-stage rejections − duplicates;
  `candidate_count_unary_feasible` counts survivors of
  the unary gates.  A duplicate key is a count, never a rejection code;
  `duplicate_candidate` does not appear in rejection histograms.
- **Unresolved selected-stamp geometry is named rejection evidence, not an
  abort.**  A selected stamp that is eligible but missing from the loaded
  libraries, has no usable full-precision hull, or has no resolved doors
  (plan §6.4) makes that lot report `stamp_geometry_unresolved` in its
  rejection histogram and returns an `unsatisfied` report instead of a CLI
  crash.
- **Unresolvable explicit source edges fail closed in the renderer.**  A
  sketch may name a `site.json` source-road edge that the aligned network
  cannot resolve; `plan_sketch.py` then raises `FAILURE: sketch ...` via a
  named `SketchError` instead of a raw `KeyError`.
- **Facing measures the assigned target in both modes.**  For explicit rows
  and for `nearest_legacy` rows alike, `distance_gu`, the access link, and
  `facing_deviation_deg` all describe the SAME assigned target.  Legacy
  facing is measured against the assigned nearest target, not the pre-v1
  minimum over all in-reach targets; that min-any-target mismatch is
  deliberately not restored.
- **Terrain sampling rejects, never clamps.**  The TerrainBundle adapter
  raises for samples inside the requested site rectangle but outside the
  survey coverage; the fitter classifies such candidates
  `terrain_sample_unresolved` (scope before sampling, plan §6.4).

## 2026-08-13 — optional lot `frontage_side` (left/right)

`fit_intent_sketch.py` accepts an optional lot key that controls which side of
the primary polyline target a building fronts:

```json
"frontage_side": "left"
```

- **Definition.** `left` = `(-ty, tx)`, `right` = `(ty, -tx)`, the unit
  tangent of the *sampled* polyline segment relative to increasing point
  order.  The side is taken verbatim; it is never inferred from the marker or
  lot id.
- **Absent behavior is byte-for-byte unchanged.**  With no `frontage_side`
  the fitter keeps the marker-derived side logic exactly as before
  (verified: `intent.copy.json` and `resolved.sketch.json` byte-identical to
  the pre-change implementation; `fit_report.json` differs only by the new
  per-lot `frontage_side` key).
- **Polyline-only.**  The key is valid only when the lot's primary frontage
  target is a road/source-road polyline.  A plaza/court polygon primary
  target with an explicit side raises `FAILURE: frontage_fit …` from
  `fit_intent` before any candidate generation (named schema error).
- **Centroid gate.**  Each constructed candidate is rejected with the named
  rejection `frontage_centroid_wrong_side` unless its resolved footprint
  centroid is strictly on the requested side of the *same sampled segment*
  by more than the ambiguity epsilon (`AMBIGUITY_EPSILON_GU = 1e-6`).  This
  is a construction-stage rejection: it counts in the per-lot rejection
  histogram before deduplication and unary evaluation, exactly like
  `frontage_side_ambiguous`.
- **Reporting.**  `frontage_side` is preserved in `intent.copy.json` and
  reported per lot in `fit_report.json` (`null` when absent, for audit).
  It is deliberately NOT copied into `resolved.sketch.json` — the solved
  transform and explicit door targets remain the renderer contract.
- **Unchanged contracts.**  Path-edge/gap/pivot/facing/rotation logic, all
  terrain/reach/facing/clearance/collision gates, rank tuple, complete-search
  counts, and deterministic ordering are untouched; marker-derived candidates
  still deduplicate by rounded `(centroid_x, centroid_y, yaw)` before unary
  evaluation.

## 2026-08-13 — complete deterministic search (replaces the beam)

The global selector is a complete deterministic MRV/forward-check search over
immutable candidate-domain bitsets in `src/procgen/frontage_fit.py`; there is
no beam, no width, and no state cap.  Lot order is `(unary domain size, lot
id)`; candidate values iterate ascending capped `unary_feasible` position,
which preserves the existing `(candidate.rank, candidate.ordinal)` preference.

- **Budget.**  `FitConfig.search_node_budget`, finite default `1_000_000`
  nodes; `0` means unlimited.  A positive budget caps recursive search-state
  calls and raises at the search boundary, so exhaustion can never fall
  through to an unsatisfiable conclusion.  The production default stays
  finite; a budget-exhausted run is a valid stop that needs a separate lead
  decision before any unlimited diagnostic rerun.
- **Outcome mapping.**  Unary failure -> `unsatisfied` /
  `unary_unsatisfied`; exhaustive proof of no complete assignment ->
  `unsatisfied` / `global_collision_unsatisfied`; node budget reached ->
  `inconclusive` / `search_budget_exhausted`; complete assignment found ->
  `solved`.  The two failure statuses both emit an empty resolved lot set and
  list every lot id in `unresolved_lot_ids`; status and terminal code
  distinguish proof from inconclusion.
- **Report metrics.**  `search_counts` (and the same keys mirrored at the top
  level) are `search_nodes`, `search_extensions`, `compatibility_checks`
  (one `_rings_conflict` call per unordered candidate pair), and
  `search_backtracks`, plus the existing `candidate_generated` /
  `candidate_deduplicated` / `candidate_unary_feasible` and the boolean
  `search_budget_exhausted`.  The retired `states_expanded` and
  `beam_truncations` fields are gone, with no aliases.
- **Limitations.**  The solver is complete (a complete collision-free
  assignment is found whenever one exists and the budget is not exhausted),
  but the reported assignment is the first deterministic feasible one, not a
  globally rank-optimal one:
  `complete MRV/forward-check search returns the first deterministic feasible
  assignment, not a globally rank-optimal assignment`.
  Budget exhaustion is inconclusive, never proof of impossibility:
  `search_budget_exhausted is inconclusive and never proves geometric
  unsatisfiability`.
  The search recurses one Python frame per assigned lot; exceptionally large
  intents approaching the interpreter's recursion limit require partitioning
  (the limit is platform-dependent).
