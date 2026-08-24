# 2026-08-24 Falkreath Wall/City Generation — Blocking Issues and Failed-Attempt History

Audience: a review agent with access to this repository only. This file
summarizes the current state of the Falkreath whole-city pipeline, the blocking
problems in the reverted baseline, and what the previous (failed) implementation
attempt did wrong. Read `docs/architecture_overview.md` and
`docs/guides/rotation_conventions.md` first.

## Where the relevant code lives in this repo

- `src/procgen/townlayout/` — the town pipeline package: `inner_walls.py`
  (R2W wall composition, terrain sampling, gate straightening),
  `wall_population.py` (R5 front rows), `place.py` (placement solver),
  `spatial_roles.py` (R10 civic roles), `stamp_objects.py` (realization/seating),
  `circulation_realization.py` (roads/realization), `land_authoring.py` (LAND/VTEX).
- `src/procgen/wall_compose.py` — the wall composer (runs, slopes, towers, gates).
- `tools/cityforge/build_town_*.py` — the stage drivers R2W → R2C → R5 → R6 →
  R10 → R11 → R12 → R13, plus realization/seating/circulation/LAND.
- `tools/cityforge/render_wall_on_townlayout.py`,
  `tools/cityforge/render_wall_scatter_city.py` — verification renderers.
- `tools/generate_settlement_wilderness.py` — settlement assembly + reused-scatter
  clearing.
- `configs/kits/falkreath/wall_kit.json` — wall kit (piece IDs, rules:
  `corner_angle_threshold_deg` 65.0, `tower_spacing_gu` 0.0).
- `configs/kits/falkreath/falkreath_wall_profile.json` — wall profile.
- `configs/cityforge/terrain_grading.json` — LAND grading policy.
- `current_generation_parameters.md` — the accepted lineage parameters.

Pipeline order (each stage a separate CLI, hard 120 s ceiling per stage):
R2W inner wall → R2C minor roads → R5 wall/front rows → R6 row access →
R10 spatial roles → R11 alley infill → R12 surfaces → R13 final layout →
stamp realization → seating → circulation → LAND authoring → clearing →
reused-scatter filtering → town ESP → masterless conversion → renders.

## Current baseline

The authoritative wall/city code was surgically rolled back (2026-08-24) to
workspace commit `37d92038` ("checkpoint gate road terrain and full city
render"), which predates a large failed implementation dispatch. The accepted
pre-dispatch artifacts are wall doc
`falkreath_wall_production_continuous_slopes_v18_gateplane.json` and the R18 v20
renders. Everything after it (commit `0381f25d` and the unfinished R21 attempt)
was rejected and rolled back; R19/R20/R21 outputs are retained only as rejected
evidence.

### Known defects in the reverted R18 baseline (observed in full-resolution renders, none fixed yet)

1. Wall ring is continuous, but tower spacing is irregular: clusters on the
   north arc, long bare stretches east/south.
2. Several wall segments on the northern hill are partially buried into the
   terrain crest.
3. Buildings exist *outside* the wall on the west/southwest approaches — a
   loose unenclosed cluster.
4. Buildings press directly against the inner wall face in many spots with no
   perimeter street.
5. Gate00: interior stair block against the wall reads as a stacked box
   assembly with visible seams; square-tower crenellation tops are flat slabs.
6. Gate01: wall descends to the shore and meets sand/rock with a visible
   gap/overlap against a large boulder; huts sit outside the gate on the shingle.
7. Tower heights are uniform along the ring despite large terrain height
   differences, so towers on low ground have little effective relief.
8. Interior is dense but with patchy empty courtyards in the NE quadrant.

## Structural pipeline defects (confirmed root causes, from the rejected R19 attempt)

These were established by lead investigation before the failed dispatch and are
NOT yet fixed in the rolled-back baseline (the rollback restored pre-fix code):

1. **The wall was never an input to placement.** The rejected city combined a
   new 224-member wall with an older layout whose R5 reserve contained 199
   *older* wall footprints. Placement must reserve the exact composed-wall
   member footprints.
2. **Flat-probe vs terrain divergence.** R2W previously composed its reservation
   probe on flat terrain; recomposition on real terrain selected different
   slopes/underlays/footprints, so even a nominal R2W→R5 run could diverge from
   the visible wall. The wall must be composed once against the surveyed
   terrain field and that exact document propagated (exact JSON equality) through
   every later stage.
3. **R6→R10 ordering bug.** The chain runs R6 (blocker removals, reserved access
   mouths) before R10, but `build_town_spatial_roles.py` read R5 directly, so R6
   results were discarded; R10 then failed because a hardcoded market block lost
   its road frontage.
4. **Hardcoded civic sector block IDs** in `spatial_roles.py` are not stable when
   an authoritative wall changes the buildable blocks. Civic roles (plaza,
   front courtyard) must be selected geometrically from a JSON policy, not by
   block ID.
5. **Reused scatter ignores the new clearing.** `--scatter-json` reuse applied
   only the wall filter, not the newly generated `settlement_clearing.json`, so
   trees survived on roads outside gates.
6. **Gate bend through the passage.** Gate centers are exactly on their
   arterials, but a gate heading sampled over only 128 GU bent ~16.4° across the
   1,024-GU gatehouse span, making exact point alignment look offset. Fix
   direction: straighten each gate's arterial locally (512 GU straight + 512 GU
   blend, ≤224 GU displacement) and require ≤1 GU / ≤1° gate-to-road agreement.

## The failed implementation attempt (what went wrong)

A large single dispatch ("Luna") implemented the locked-wall repair plus a
seam/gate-road repair commit (`0381f25d`) and an unfinished R21 attempt. Its
outputs (R19 v2, R20 v3, R21 partial) are all rejected. Specific failures:

1. **R20 terrain excavation.** The wall-foundation LAND grading authored 91
   grade specifications and the building-deformation path lowered 20,424 LAND
   vertices for 112 buildings. Renders showed large rectangular excavation
   voids, exposed/floating wall foundations. Neither operation is needed for
   collision exclusion; ordinary wall members and buildings must NOT cut LAND.
2. **Scaled fillers killed slopes.** Per-run `run_fit_scale`/chord-wide mesh
   scaling was introduced to close seams; this disabled every native slope
   piece (R19 had 16 slopes; R20 had zero) and broke height transitions.
3. **Projected tower-edge contacts.** A z-slice tower-endpoint solver replaced
   native wall/tower socket contacts, producing member splits and coplanar
   contact artifacts. The correct model: wall runs may terminate inside a round
   tower body by a configured socket depth (64 GU along the wall axis);
   wall-to-wall/slope/gatehouse contacts must not overlap.
4. **Silent LAND repair loops.** A signed-i8 VHGT delta "repair" repeatedly
   mutated and clamped heights at serialization time instead of failing closed.
   Encoding must fail loudly (`FAILURE: LAND VHGT delta ...`), never silently
   move vertices.
5. **Road evidence mixing.** Deferred diagnostic polygons were rendered as fake
   road overlays and cited as authored roads. Only real LAND VTEX painting
   counts as road evidence.
6. **Scope contamination.** The dispatch touched files beyond its contract
   (including mixed concurrent scatter/cliff work), which forced a surgical
   path-scoped rollback of 15 files to `37d92038` rather than a revert.

### Process lessons (binding on future attempts)

- One locked wall, composed once on real terrain, propagated by exact JSON
  equality to every downstream stage; no stage may re-derive or re-fit it.
- Ordinary wall members adapt to terrain via native straight/slope selection,
  member Z placement, and existing inverted underlays — never via LAND edits.
- Gate platforms are the ONLY wall-related terrain exception (platform set to
  the gatehouse bottom plane, approach blended along the assigned road).
- No scaled straight/slope fillers; kit-authored scale only.
- LAND encoding is fail-closed; no silent repairs or clamps.
- Visual acceptance is full-resolution renders; numeric checks never override a
  visible defect; a stage >120 s is a failed algorithm.
- Each stage dispatch is narrow and path-scoped; no dispatch continues past a
  failed review gate.

## Design ruling: no turns without towers

`max_turn_without_tower_deg` in `falkreath_wall_profile.json` was 25.0 — an
unacceptable allowance (any unwalled turn breaks wall alignment completely).
It is now **0.0** (committed). Note: in the reverted code no module currently
reads that key; tower placement is governed by `corner_angle_threshold_deg`
(65.0) and `tower_spacing_gu` (0.0 = periodic towers off) in `wall_kit.json`.
Any future consumer must treat 0.0 as "every turn gets a tower".

## What a useful fix plan must address

1. Re-implement the locked-wall propagation (root causes 1–2 above) on top of
   the reverted baseline without repeating the R20 mistakes.
2. Fix the R6→R10 ordering and replace hardcoded civic block IDs with
   geometric selection from JSON policy (root causes 3–4).
3. Apply the clearing index when reusing scatter (root cause 5).
4. Straighten gate arterials locally and bind each gate to exactly one road
   (root cause 6), with gate platforms as the only terrain exception.
5. Preserve native slopes and kit scales; socket-based tower contacts only.
6. Keep every stage under 120 s and gate acceptance on full-resolution renders.
