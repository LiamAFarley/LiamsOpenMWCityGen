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

## 2026-08-13 — deterministic composition closure

Composition is enabled when the normalized intent declares `lot_groups`, a
road `purpose`, or a lot `intentional_outlier`.  It remains authored data: the
solver does not choose semantic roles, invent roads/spaces/targets/stamps, or
claim a universal beauty score.

### Authored vocabulary

- A road may optionally declare `purpose`: `urban_street`, `service_lane`, or
  `connector`.  `max_unsupported_frontage_gu` is optional and is legal only
  for `urban_street` or `service_lane`; it is finite and non-negative.  The
  legacy road `kind` is not converted into a purpose.
- A lot may optionally declare a real boolean `intentional_outlier`.  It
  excludes that lot only from the group's non-outlier-to-medoid distance
  metric; it does not remove the lot from span, order, side-run, road support,
  or plaza-sector measurements.
- `lot_groups` contain a unique `id`, one of `compact_cluster`,
  `irregular_two_sided`, `formal_square`, `gateway_cluster`, or
  `sparse_outskirts`, and non-empty unique `lot_ids`.  Lots cannot overlap
  groups.  Optional declarations are `shared_target_id`,
  `max_span_gu`, `max_consecutive_gap_gu`,
  `max_non_outlier_distance_gu`, positive-integer
  `max_consecutive_same_side`, and `along_order`.
- Group lot ids and group ids are normalized deterministically; `along_order`
  must be an exact permutation of the group members and retains its authored
  sequence.  `plaza_sectors` is allowed only for `formal_square` groups with a
  shared target; sector ids are unique and each range satisfies
  `0 <= start_deg < end_deg <= 360`.  Sector rows are sorted by id.

### Evaluator fact and metric contract

At each complete candidate assignment the fitter materializes exactly these
eight evaluator fact keys: `lot_id`, `centroid`, `intentional_outlier`,
`primary_target_id`, `target_arc_gu`, `target_length_gu`, `frontage_side`, and
`plaza_angle_deg`.  The arc is the canonical increasing-path projection and is
clamped only to the supplied target length; the evaluator never reprojects a
centroid or reconstructs a target.

The pure evaluator emits road length, serving-lot count, ordered projected
frontage positions, unsupported intervals, longest unsupported interval, and
authored excess.  Group output includes centroid span, actual along-target
order and gaps, maximum gap, order violation, same-side run, plaza-sector
occupancy, medoid id, non-outlier maximum distance, all corresponding excess
values, and `repeated_consecutive_gap_pair_count` for exact adjacent-gap
repetitions.  Findings are canonical and deterministic.

The nine finding codes treated as authored hard relationship gates are:

```text
road_unsupported_frontage_exceeded
group_shared_target_mismatch
along_order_violation
plaza_sector_unoccupied
group_span_exceeded
group_gap_exceeded
group_gap_unmeasurable
group_same_side_run_exceeded
group_non_outlier_distance_exceeded
```

### Complete composition proof

`candidate_count_unary_feasible` is the configured rank-best capped prefix;
`candidate_unary_feasible_all` counts the complete retained unary-feasible
domain.  Composition search does not call the capped prefix a proof.  With the
default cap, it runs complete deterministic MRV/forward-checking passes at
domain widths **64, 128, 256, 512, and full retained domain**.  A narrower
width than the configured cap, or a width that produces the same domain-size
vector, is skipped.  Each pass uses immutable compatibility bitsets, lot order
`(domain size, lot id)`, and ascending domain positions.  A hard finding at a
collision-valid leaf rejects that assignment and exhaustive backtracking
continues.

`FitConfig.search_node_budget` is global across all widening passes (default
`1_000_000`; `0` means unlimited).  The terminal meanings are deliberately
truthful:

- no unary-feasible candidate: `unsatisfied` / `unary_unsatisfied`;
- no collision-valid complete assignment after the full retained-domain proof:
  `unsatisfied` / `global_collision_unsatisfied`;
- collision-valid assignments exist, but every one is rejected by the nine
  authored relationship gates: `unsatisfied` /
  `global_relationship_unsatisfied`;
- the shared node budget is reached before proof: `inconclusive` /
  `search_budget_exhausted`, never an impossibility claim.

Unsolved and inconclusive outcomes emit an empty resolved lot set and all lot
ids in `unresolved_lot_ids`.  A composition solution is the first deterministic
hard-valid feasibility incumbent found by the progressive proof, not a claim
that every candidate is globally rank-optimal.

### Legacy and bounded improvement behavior

An intent with no composition declaration retains one capped feasibility pass
(`max_candidates_per_lot`, default `64`) and its legacy report shape.  It does
not run improvement.  This is intentionally not a universal-completeness claim
over all retained candidates.

For a solved composition intent, the separate `improvement_node_budget` is
`50_000` by default; `0` disables traversal and is not an unlimited setting.
It is separate from the feasibility budget.  Improvement starts with the
hard-valid feasibility incumbent and uses **only the exact candidate domains
and compatibility matrix of the successful feasibility pass**; it never widens
to a later or global domain.  Only hard-valid, collision-valid assignments can
replace the incumbent.

The minimized objective is exactly this lexicographic tuple:

1. descending `urban_unsupported_profile_gu` (urban street/service-lane
   longest unsupported intervals; connectors excluded);
2. descending `compact_span_profile_gu` (compact, formal-square, and gateway
   group centroid spans);
3. descending `compact_gap_profile_gu` (the same characters' measurable
   maximum consecutive gaps; absent/unmeasurable values omitted);
4. `irregular_repeated_gap_pairs` (irregular-two-sided groups only);
5. `marker_displacement_sq`, total marker displacement squared rounded once to
   six decimals;
6. `assignment_signature`, canonical `(lot_id, candidate.ordinal)` pairs.

This is a deterministic preference, not a weighted score or global aesthetic
claim.  Whether improvement is disabled, exhausts its budget, or faults, the
fit remains `solved` and retains the known hard-valid incumbent when no better
valid result is adopted.  The composition `improvement` object reports the
full evidence: `enabled`, `faulted`, `fault_code`, `domain_width`,
`domain_sizes`, `node_budget`, `nodes`, `extensions`,
`collision_valid_assignments`, `hard_valid_assignments`,
`relationship_rejections`, `budget_exhausted`, `incumbent_improved`,
`incumbent_objective`, and `selected_objective`.  An unexpected integration
fault is surfaced as `faulted: true` with `fault_code` (currently
`improvement_exception`); normal and disabled runs use `false`/`null`.

The fitter writes only `intent.copy.json`, `resolved.sketch.json`, and
`fit_report.json`.  Pass the resolved sketch to `plan_sketch.py` without
`--auto-face`; that legacy helper would overwrite the explicit door-target
transform.  This stage has no semantic stamp selection, road invention,
terrain seating, TES3 authoring, or render/visual-quality certification.
