# 2026-08-12 — Cityforge plan_sketch tools

## Purpose

`tools/cityforge/plan_sketch.py` is the sketch-to-plan derivation stage of the
restructured Cityforge pipeline: a vision-capable design agent authors ONE
minimal `sketch.json` (roads, spaces, lots) against a planning bundle, and
this CLI derives the complete format-v1 visual plan, runs the existing
hard-error analysis, writes a filtered checks file, and renders ONE composite
PNG on the byte-identical planning-canvas background.

## CLI

```
python tools/cityforge/plan_sketch.py --bundle <bundle-dir> --sketch <sketch.json> --out <dir>
```

* `--bundle`  planning bundle dir (must contain `canvas.png`, `site.json`,
  `stamps.json`, `bundle_manifest.json`).  The manifest pins the survey dir,
  aligned-road product dir, and D-STAMP library paths used to load exact
  terrain/network/stamp geometry.
* `--sketch`  sketch v1 JSON in TES3 **world GU** (x east, y north) — the same
  frame the design agent reads off the canvas graticule.
* `--out`     fresh output directory (protected-root + non-empty refusal,
  shared with the bundle builder).

Exit 0 iff the derived plan has **zero hard errors**; exit 1 otherwise (also
on any schema/runtime failure).  The PNG is ALWAYS written once derivation
succeeds — the image is how errors get diagnosed.

## Sketch schema v1 (strict; unknown keys are fatal)

```json
{"site": "falkreath_v1",
 "roads":  [{"id": "...", "kind": "street"|"alley", "width_gu": 512,
             "points": [[x,y],...]}],
 "spaces": [{"id": "...", "kind": "plaza"|"court", "polygon": [[x,y],...]}],
 "lots":   [{"id": "...", "stamp": "<bundle stamp id>", "x": -753800, "y": -71400,
             "yaw_deg": 0, "note": "optional"}],
 "notes":  "free text"}
```

Validation (violations print `FAILURE: sketch <reason>`, exit 1, **no render**):

* `site` must equal the bundle `site_name`.
* road `kind` street → width 256–1024 GU; alley → 128–512 GU; ≥ 2 points.
* space polygon ≥ 3 points; `kind` plaza or court.
* lot `stamp` must exist in bundle `stamps.json` — the bundle is
  eligibility-filtered, so this is the fail-closed quarantine gate (e.g.
  `markarth_side_v1__u114_castle_barracks` is rejected).
* lot `x,y` is the footprint **CENTROID** (what rules.md tells the designer),
  and must lie inside the bundle `rectangle_gu`.
* duplicate ids within a section are rejected.

## Outputs (all in `--out`)

| File | Content |
|---|---|
| `visual_plan.json` | full format-v1 extension (`site_survey_plan_gu`), validated with `require_valid_extension` |
| `checks.json` | ONLY the advisory's complete `hard_errors` list (all fired codes; gates acceptance), per-door `door_facts`, per-space `space_facts`, and per-pair `subterranean_overlap_facts`. Advisory codes (terrain/slope/repetition/orientation/frontage/tandem/circulation) are deliberately deferred to the placement stage and never appear |
| `plan.png` | one composite, ~1600 px: identical bundle canvas + streets/alleys, plaza/court fills, yawed kit-colored hulls with lot-id labels, intent-colored door arrows (unconnected = red), plus a 32 px overlay legend strip appended below the base legend keying all nine overlay elements in the exact drawn styles |
| `sketch.copy.json` | canonical copy of the parsed sketch |
| `log.json` | UTC timestamp + sha256 of bundle files, sketch, outputs + wall clock |

## Check semantics — Z-aware building overlap (2026-08-12)

`building_overlap` is no longer a pure 2D hull test.  The advisory consumes the
per-member 3D AABB sidecar `output/cityforge/stamps/stamp_volumes_v2.json`
(next to the stamp libraries; building-aligned per-member OBB boxes from the
v2 normalization — tight for members near 0 mod 90) and seats each lot at
`seat = max(target-terrain height under the transformed hull) − stamp
terrain_envelope.burial_depth_gu` (plan-stage approximation; T1.2 does exact
seating).  For every XY-overlapping hull pair, each member pair is transformed
(`rot2d_ccw` yaw about the lot origin + position + seat Z); a member conflict
exists when the rotated XY AABBs and the world-Z intervals both intersect.
A conflict is **subterranean** when its Z-intersection top is at least
`SUBTERRANEAN_MARGIN_GU = 32` GU below the minimum terrain height over the
XY intersection rectangle.  A lot pair is excused (recorded in
`subterranean_overlap_facts`, not as a hard error) only when EVERY member
conflict is subterranean — matching how the source cities compose (buildings
unfused only for underground overlap).  Otherwise `building_overlap` fires,
enriched with the first offending member pair's `source_id`s and the measured
above-ground intersection height.  `building_road_overlap` stays 2D and hard.

Fail-closed: a missing sidecar, a used stamp missing from the sidecar, or an
unusable member box emits `stamp_volumes_unresolved` and the pair check is
skipped — there is never a silent 2D-only fallback; a terrain bundle that
cannot supply heights is a hard FAILURE.  `checks.json` copies
`subterranean_overlap_facts` from the advisory report verbatim (facts with
per-conflict `intersection_top_z_gu` / `terrain_min_z_gu` / `margin_gu`,
not counts).

## Derivation rules

* **Frame**: every derived coordinate is `world - survey_origin` (the plan
  frame the advisory measures in).  The v2 stamp libraries (T0.4b
  normalization) are BUILDING-ALIGNED: **yaw 0 is the building's natural
  axis-aligned pose** — hulls are tight around the building and door arrows
  point cardinal at yaw 0.  Bundle preview sheets still show the source
  orientation (they are renders of the source scene; the source rotation is
  recorded per stamp as `normalization_theta_deg` in the library).
* **rectangle**: from `site.json` (cells, margin 2048, world bounds).
* **existing_source_roads**: one record per `site.json` source-road edge with
  the clipped chain endpoints as `connection_points`.
* **authored_roads / alleys**: from sketch roads (street → class `street`,
  surface `road`; alley → class `service`, surface `settlement_dirt`).
* **connection_targets**: one per polyline ENDPOINT, derived against the
  COMPLETE circulation map (every existing source road, every authored
  road/alley, every space polygon) built before any endpoint connection is
  derived, so connection validity is independent of the sketch's road array
  order; only the road's own id is excluded (no trivial self-snap).  The
  endpoint snaps to the nearest candidate within `SNAP_DISTANCE_GU = 1536`.
  In-range: declares `at_plan_gu` = its own
  coordinate (distance 0 to its own polyline, so the advisory's geometric
  check passes).  Out-of-range: still declares the nearest target but with
  `at_plan_gu` = the nearest point ON that target (> 768 GU from the road), so
  the existing `road_disconnected` hard check reports it with a measured
  distance instead of the connection being silently dropped.
* **road_surface_polygons / shared_courts**: from spaces (plaza → `kind`
  `plaza`, surface `settlement_cobble`; court → surface
  `settlement_grass_dirt`, empty `connection_targets` — required by the
  format, not hard-gated).
* **stamps**: position/yaw from the lot.  The stored position is the stamp's
  **seed-door anchor**, NOT the lot coordinate: lot `x,y` is the footprint
  centroid, so the derivation computes the `hull_xy_rel` 2D centroid
  (`cityplan.polygon_centroid`), rotates it by the lot yaw, and places the
  anchor at `lot.xy − rot2d_ccw(centroid_rel, yaw)` (the downstream
  `position_plan_gu` format is unchanged).  `door_intents` per measured door.
  Door offsets are rotated by lot yaw with the SAME transform the stamp
  libraries use (`cityplan.rot2d_ccw`, as in
  `visual_planner_symbols._transform_door`); the nearest circulation target
  within `DOOR_REACH_GU = 768` (same metric as the advisory: centerline
  distance for roads, inside/ring distance for polygons — since every allowed
  half-width ≤ 512 < 768, centerline reach subsumes corridor reach) decides
  intent: **public** for street/plaza/existing source road, **service** for
  alley/court.  Unconnected doors keep intent public with NO target (the
  existing door advisories report them).  `access_links` are straight stubs
  door → nearest point on target.
* **door arrows** (plan.png) draw the door's geometric outward heading:
  `heading = member.outward_heading_deg + yaw_deg` (mod 360) — the wall
  normal derived during the v2 library build (thin box axis, sign away from
  the body centroid; see `2026-08-12_stamp_orientation_normalization.md`
  correction note).  The raw TES3 door rotz is only a legacy fallback (it is
  NOT a reliable facing — mesh forward axes are model-specific); the
  hull-centroid radial is the last-resort fallback.  Intent colors and the
  unconnected-red rendering are unchanged.
* **districts / annotations / advisory_overrides**: empty arrays — the format
  requires only the KEYS; no record semantics are derivable from a minimal
  sketch, so none are invented.
* **render_options**: omitted (format-optional; the renderer uses the bundle
  projection directly).
* **design_notes**: the sketch's `notes` verbatim.

## Invariants

* Deterministic for identical inputs (verified byte-identical reruns of all
  outputs except `log.json`'s timestamp/wall clock).
* Rendered base is byte-identical to the bundle `canvas.png` (same terrain
  product, network, rectangle, title, pixel density — verified by hash).
* No original file is modified; outputs only land in the fresh `--out` dir.
* The tool never authors TES3 records.

## Pipeline position

`planning bundle → [design agent authors sketch.json] → plan_sketch.py →
visual_plan.json (feeds the existing visual planner/advisory tools) →
later placement stage (advisory codes deferred)`.  Reuses
`src/procgen/planning_canvas.py` (projection), `visual_planner_symbols.py`
(stamp transform/rotation, drawing helpers, label gate),
`visual_planner_advisory.py` (hard-error analysis), `visual_planner_format.py`
(validation/serialization), and the bundle builder's
`refuse_unless_fresh`/`protected_roots`/`_door_members`.

## 2026-08-13 — order-independent connection derivation (shared-fork fix)

`derive_plan` now derives `connection_targets` in two passes: pass 1 registers
EVERY authored road/alley (plus spaces) in the circulation target map, pass 2
derives each road's endpoint connections against the complete map minus that
road's own id.  Previously targets were accumulated in the sketch's road-array
order, so at a fork shared by two authored roads the road listed first could
not see the road listed later and was falsely emitted as `road_disconnected`;
reversing the array merely moved the error to the other road.  Snap distance
(1536), nearest-target tie break, emitted connection record shape, and the
fail-closed out-of-range record (`left for road_disconnected`) are unchanged.
Regression tests in `tests/test_frontage_fit.py` cover both array orders,
per-road fact equivalence under reversal, isolated endpoints still emitting
the fail-closed record, and no self-snap.
