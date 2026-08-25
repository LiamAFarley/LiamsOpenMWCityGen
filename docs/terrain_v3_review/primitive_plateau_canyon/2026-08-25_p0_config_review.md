# 2026-08-25 P0 Config Review — TR Primitive Plateau/Canyon

**Scope:** Read-only P0 checkpoint per `request.md` + `plan.md`. Inspected `configs/tamriel_reworked_v1.json` top-level `structure_mode` and `terrain_primitives` section. Verified JSON validity, P1–P5 parameter completeness, target declaration, and preservation of the existing failed harmonic/band config. No source code inspected, no crop executed.

**Verdict:** **PASS with observations** — P0 gate criteria satisfied; file is JSON-valid and config-driven. Three non-blocking formatting/scope observations flagged for lead triage before P1.

---

## 1. P0 Gate — Mode and Configuration

| Check | Result | Evidence |
|-------|--------|----------|
| `structure_mode` present | **PASS** | `configs/tamriel_reworked_v1.json:3` → `"structure_mode": "primitives"` |
| `terrain_primitives` present as top-level object | **PASS** | `configs/tamriel_reworked_v1.json:4-57` |
| `terrain_primitives.enabled` | **PASS** | `true` (boolean) |
| Old band/harmonic implementation preserved but not invoked in primitive mode | **PASS (with note)** | `solve.panel` byte-identical to `HEAD`; `solve.v3` surface and band definitions unchanged (see §3). New `structure_mode: "primitives"` cleanly selects primitive path; no removal of old band code required at config level. |

---

## 2. JSON Validity — All P1–P5 Values

File parses cleanly via `json.loads` / `json.dumps` round-trip. No `NaN`/`Inf`, no trailing commas, no duplicate keys. `1e-06` scientific notation valid JSON. All numeric/string/boolean types correct.

### 2.1 Top-level primitive target (P1–P5 shared)

- `semantic_downsample: 4` — **PASS** (int, matches `plan.md` P0 “semantic resolution” downsample)
- `target_review_quadrant: "bottom_right"` — **PASS** (explicit string, not hard-coded coordinates)
- `target_fraction: [0.5, 0.5, 1.0, 1.0]` — **PASS** (4-float array, correct bottom-right fraction of review bbox; config-driven per `request.md` §Target Selection)
- `bbox_margin_cells: 2.0` — **PASS** (float)

Target is explicitly `bottom_right` + `target_fraction`; no source coordinate hard-coded in config. Plan says “driver resolves that fraction to local vertex coordinates” — config satisfies that contract. **PASS** on “target is explicitly bottom_right/config-driven”.

### 2.2 `plateau` (P1–P2, P3)

All 17 keys present and typed correctly:

`min_confidence: 0.55` (float 0–1) · `min_component_vertices_semantic: 24` (int) · `elevation_low_percentile: 60.0` · `elevation_high_percentile: 88.0` · `flat_slope_percentile: 35.0` · `prominence_percentile: 65.0` · `prominence_sigma_verts: 32.0` · `max_continuation_cells: 8.0` · `ordinary_continuation_cells: 5.0` · `direction_seam_weight: 0.65` · `geodesic_lateral_eta: 0.4` · `support_core_threshold: 0.65` · `support_edge_threshold: 0.05` · `top_fit: "robust_affine"` · `allow_quadratic_fit: true` · `quadratic_regularization: 0.01` · `weight: 1.0` — **All PASS, JSON-valid, values within plausible ranges.**

### 2.3 `scarp` (P2–P3, P5 conductance)

`profile_samples_each_side_verts: 32` · `profile_bins: 32` · `fallback_width_verts: 8.0` · `conductance_min: 0.1` · `conductance_beta: 3.0` · `weight: 1.5` — **PASS.**

### 2.4 `canyon` (P4)

`min_confidence: 0.55` · `max_continuation_cells: 8.0` · `direction_penalty: 1.0` · `uphill_penalty: 4.0` · `target_low_penalty: 1.5` · `width_cells: 4.0` · `bottom_half_width_cells: 1.0` · `wall_exponent: 1.5` · `weight: 1.2` — **PASS.**

### 2.5 `reconciliation` (P5)

`linear_solver: "amg_rs_cg"` · `cg_tol: 1e-06` · `cg_maxiter: 200` · `amg_max_coarse: 500` · `background_weight: 0.12` · `conductance_min: 0.1` · `conductance_beta: 3.0` — **PASS.** Single screened-Poisson solve contracted; `amg_rs_cg` matches `solve.v3.surface` solver, consistent with “Use one edge-aware screened-Poisson reconciliation solve and the existing narrow final seam lock.”

**Overall P1–P5 JSON validity: PASS.** Every requested stage has its driving parameters present, correctly typed, and serializable.

---

## 3. Preservation — Existing Failed Harmonic / Band Config Remains Intact

Compared working copy against `HEAD:configs/tamriel_reworked_v1.json` (git diff):

- **`solve.panel` — PASS, untouched.** `blend_cells/hybrid_carry_cells/bands/warp_strength/clone_patch_by_band/clone_feather_by_band/clone_step_div/erode_*` all byte-identical. This is the dominant Run A band structure the request says “Keep the failed Run A macro/meso output as evidence only; do not retune it.”
- **`solve.seed / coarse_factor / blend_cells / fade_verts / octaves / persistence / lacunarity / ridge_weight / amp_* / style_match / massif_* / carve_*` — PASS, all 21 top-level solve scalars identical.
- **`solve.v3` — PASS (modulo one path, see observation).** `blend_cells_max`, `blend_width_min/max`, `band_edge_tolerance_gu`, `surface` (`linear_solver/cg_tol/cg_maxiter/amg_max_coarse/anchor_conflict_*`), `max_blend_grade`, `outer_apron`, `quality` all identical.
- **`sources / atlas / render / expected_counts_v1` — PASS, identical.**
- **`solve.regions.tr_vvardenfell_wall` core — PASS.** `cluster_ids` (26 ids), `review_margin_cells: 6`, `seam_crop_margin_cells: 2`, `seam_crop_bbox_cells: [-22,-29,-12,-14]` unchanged. Added field `review_bbox_cells: [-40,-47,1,1]` is additive (see observation) — core identity preserved.

**Conclusion on “remains intact and untouched”:** The failed harmonic/band config (the `solve` band tree) is intact. No retune, no scale change, no solver change.

---

## 4. Observations (Non-blocking, Lead Triage Recommended)

### 4.1 Formatting — `structure` top-level indent is 4 spaces, should be 2

`configs/tamriel_reworked_v1.json:363` reads `    "structure": {` (4 leading spaces). `hydrology` and `erosion` correctly use 2 spaces. JSON is still valid (whitespace-insensitive), but style is inconsistent and will create a noisy diff if auto-formatted later.

```json
// current (line 363)
    "structure": {
      "macro_width_cells": 8.0,
...
  "hydrology": {    // 2 spaces — correct
```

**Recommendation:** Reformat `structure` block to 2-space indent to match rest of file. No functional impact.

### 4.2 Scope creep — New top-level keys beyond `plan.md` P0

`plan.md` P0 specifies exactly: `structure_mode` + `terrain_primitives` section. The working copy also adds:

- `flat_owner_fallback: {enabled:true, max_height_gu:-2000.0}`
- `structure: {macro_width_cells, meso_width_cells, fine_keep_at_seam, ...}` (12 keys)
- `hydrology: {routing, owner_hydrology_halo_cells, ...}`
- `erosion: {enabled, cycles, snapshot_cycles, ...}`

None of these existed in `HEAD` (verified across `HEAD~10` and milestone commits `d5131aad`, `08de6a89`). They are not in `request.md` (“Do not run erosion, hydrology, or the broad multi-region batch”) and not in `plan.md` P0 spec. They appear to be forward-ported from a development branch. Functionally harmless (they do not overwrite `solve.panel`), but they expand P0 beyond its contracted scope.

**Recommendation:** Lead confirms whether `structure`/`hydrology`/`erosion`/`flat_owner_fallback` are intentional P0 inclusions or should be deferred to a later stage/branch. If deferred, remove them before P1 review to keep the P0 diff minimal and reviewable.

### 4.3 Additive changes that are plausible but undocumented

- `paths.solve_out_dir`: `solved` → `solved/v5_missing_cells` — aligns with `request.md` “Preserve the accepted Stage-3 field … and missing-cell path” via `v5_missing_cells`; likely intentional, but not mentioned in `plan.md` P0 text (only implied). **No action unless lead wants the path unchanged for baseline comparison.**
- `solve.v3.relief_npz`: `solved/v3/relief/...` → `solved/v5_missing_cells/v3/relief/...` — consistent with `solve_out_dir` change.
- `solve.regions`: 12 new region entries added (`tr_ridge_orientation`, `tr_plateau_transition`, `tr_missing_cell`, `sky_mountain`, `sky_plateau`, `sky_missing_cell`, `sky_cliff_transition`, `sky_cliff_north`, `sky_void_edge`, `cyr_mountain`, `cyr_plateau`, `cyr_lowland`) plus `tr_vvardenfell_wall.review_bbox_cells`. Request says “Render only the real `tr_vvardenfell_wall` target” and “Do not … start the broad validation batch” — extra regions are inert if not rendered, but they do widen the contract surface. Flagged for lead awareness.
- `terrain_relief`: `gentle_end_fraction: 0.05` and `gentle_gain: 1.6` removed (present in `HEAD`, absent in working copy). `plan.md` says preserve Stage-3 relief scaling — removal may be intentional (gentle response folded into new relief path) but should be explicitly acknowledged in handoff so Stage-3 baseline is not accidentally altered.

None of these break P0, but they merit a one-line rationale in the P1 handoff.

---

## 5. Summary Table

| Criterion (from prompt) | Verdict |
|--------------------------|---------|
| All P1–P5 values JSON-valid | **PASS** — every plateau/scarp/canyon/reconciliation value parses and types correctly |
| Target is explicitly `bottom_right`/config-driven | **PASS** — `target_review_quadrant: "bottom_right"` + `target_fraction: [0.5,0.5,1.0,1.0]` |
| Existing failed harmonic structure config remains intact and untouched | **PASS** — `solve.panel` and all `solve` band/surface scalars identical to `HEAD`; `sources/atlas/render/expected_counts_v1` unchanged |
| No source code inspected, no crop run | **PASS** — review was config-only |

---

## 6. Recommendation

**Clear P0 to proceed to P1.** The `terrain_primitives` section is complete, valid, and respects the `bottom_right` target fraction contract and the “keep owner terrain immutable / one screened-Poisson solve / narrow final seam lock” invariants in its reconciliation parameters. Fix the `structure` indent (4→2 spaces) and have the lead confirm the four additive top-level blocks and the `gentle_*` removal before P1 implementation begins, to keep the checkpoint diff auditable.

*Reviewer: review-flash (read-only, config-only) — 2026-08-25 — files: `configs/tamriel_reworked_v1.json`, `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/request.md`, `plan.md` — no code, no render executed.*
