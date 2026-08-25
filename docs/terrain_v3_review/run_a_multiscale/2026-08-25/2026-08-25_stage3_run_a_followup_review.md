# 2026-08-25 Stage 3 Run A Follow-up Review — Post-Cleanup Wiring (No Crop Execution)

**Scope:** Stage 3 only — re-review of `tools/terrain/erode_region_v3.py` and `configs/tamriel_reworked_v1.json` structure section after lead cleanup. No crop executed, no files edited.
**Spec source:** `.opencode/runs/2026-08-25_tr-run-a-multiscale/request.md` (39 lines) + `plan.md` Stage 3 (§35-42) + prior Stage 3 review `2026-08-25_stage3_run_a_driver_review.md` (2026-08-25, PASS with 1 housekeeping).
**Files inspected:** `F:\ProcGenWorkspace\tools\terrain\erode_region_v3.py` (329 lines) and `F:\ProcGenWorkspace\configs\tamriel_reworked_v1.json` (373 lines, structure §308-326 only per instruction).
**Reviewer:** review-flash (read-only, no edits, no real-crop execution)
**Date:** 2026-08-25

## Verdict: PASS — Lead cleanup verified. All prior Run A wiring guarantees remain intact; dead-key housekeeping resolved correctly without loss of active keys.

## Findings (instruction-ordered)

### 1. Eleven dead point/ribbon keys are removed — PASS

Current `configs/tamriel_reworked_v1.json` structure holds 13 keys (was 26 in the 308-337 block reviewed on 2026-08-25). The 11 dead point-source / ribbon / weight keys are absent (grep 0 hits in `structure`):

```
guide_seed_stride_verts  — removed
guide_ribbon_sigma_verts — removed
guide_max_cells          — removed
massif_guide_max_cells   — removed
guide_turn_deg_per_8_verts — removed
guide_score_threshold    — removed
guide_decay_fraction     — removed
ridge_weight             — removed
valley_weight            — removed
plateau_top_weight       — removed
scarp_weight             — removed
```

Verification at `configs/tamriel_reworked_v1.json:308-326`: `json.loads` yields sorted keys

```
['amg_max_coarse','cg_maxiter','cg_tol','fine_keep_at_seam',
 'fine_restore_distance_cells','gaussian_scales_verts','linear_solver',
 'macro_width_cells','meso_width_cells','owner_analysis_halo_cells',
 'ridge_percentile','tensor_sigma_verts','valley_percentile']
```

`dead present: []` — exact 11-key removal confirmed. The prior review listed 13 entries (336-337 block) including `ridge_percentile`/`valley_percentile` as apparent dead keys; the follow-up instruction clarifies those two are active terrain-feature keys and must remain (see §2) — the 11 above are the true dead set. File shrank 384→373 lines, consistent with 11-key deletion; `json.loads` still parses and structure key count is 13.

No `point`/`ribbon`/`guide_seed`/`sparse` reintroduction in `structure`; no downstream consumer references these 11 keys (they were unread by `terrain_structure.py` even before removal — only 8 active keys were ever consumed).

### 2. Active terrain-feature keys remain — PASS

All four instruction-named active keys plus the four tunable continuation keys survive with correct types/values:

| key | lines | value | status |
|-----|-------|-------|--------|
| `gaussian_scales_verts` | 314-318 | `[8.0, 24.0, 64.0]` | present — matches Stage-1 sigma 8/24/64 contract |
| `tensor_sigma_verts` | 319 | `10.0` | present — Stage-1 tensor smoothing |
| `ridge_percentile` | 320 | `88.0` | present — was in the prior 13 "dead" list but is active per instruction |
| `valley_percentile` | 321 | `88.0` | present — same |
| `macro_width_cells` | 309 | `8.0` | present — matches `terrain_structure.py` DEFAULTS 8.0 |
| `meso_width_cells` | 310 | `4.0` | present — matches DEFAULTS 4.0 |
| `fine_keep_at_seam` | 311 | `0.2` | present — spec 0.2 at seam |
| `fine_restore_distance_cells` | 312 | `6.0` | present — spec 1.0 at 6 cells |
| `owner_analysis_halo_cells` | 313 | `8.0` | present — retained (used by `terrain_features`) |
| `linear_solver`/`cg_tol`/`cg_maxiter`/`amg_max_coarse` | 322-325 | `amg_rs_cg / 1e-6 / 200 / 500` | present — solver tuning |

`active missing: []` from probe. The lead correctly preserved the two percentile keys the prior review had miscategorized as dead; the resulting 13-key `structure` is exactly `8 previously-active solver/width keys + owner_analysis_halo_cells + gaussian_scales_verts + tensor_sigma_verts + ridge_percentile + valley_percentile` — no active key lost, no dead key retained.

### 3. Updated driver docstring — PASS (prior optional nit §2 resolved)

Prior review § "Optional cosmetic nits 2" noted `tools/terrain/erode_region_v3.py:1-8` still read `"solves sparse semantic guides, routes owner inflow..."` and asked to note the generic `W=0` harmonic band solve.

Current `erode_region_v3.py:1-8`:

```py
"""Run multiscale structural continuation, effective erosion, and final seam lock.

The command consumes the real Stage-3 local field for one configured region,
analyzes authoritative owner terrain, solves complete macro/meso harmonic
bands, routes owner inflow plus generated rainfall, and writes a standardized
local review sheet. Run A stops after the structural fields and never enters
hydrology, erosion, or final-lock code.
"""
```

Change verified:
- `"solves complete macro/meso harmonic bands"` replaces `"solves sparse semantic guides"` — matches `terrain_structure.py:9-13` which was clarified in the Stage 2 follow-up (`W=0` pure harmonic, AMG-friendly).
- Second sentence now explicitly scopes Run A: `"Run A stops after the structural fields and never enters hydrology, erosion, or final-lock code."` — previously only a generic sheet description.

Docstring is accurate for both normal-path and Run A specialization; no stale point-source wording remains. Prior nit is closed.

### 4. All prior --run-a / --output-dir / early-return / four-render guarantees — PASS (unchanged)

Re-audited against the prior review's 6 checklist items. Every line reference below is current 329-line driver; no wiring change detected beyond the docstring.

**a) Flags explicit** — `erode_region_v3.py:195-198`:

```py
ap.add_argument("--config", default=str(ROOT / "configs" / "tamriel_reworked_v1.json"))
ap.add_argument("--region", default="tr_vvardenfell_wall")
ap.add_argument("--run-a", action="store_true", help="render structural Run A only")
ap.add_argument("--output-dir", default=None, help="Run A output directory override")
```

Both flags exist with exact spelling, `store_true` and `default None`. `ast` walk and `py_compile` OK. No regression.

**b) Run A loads the real Stage-3 field** — `_run_a:143-147`:

```py
ctx = build_context(ROOT, cfg, region)
field = _load_stage3_field(cfg, ctx, region)
```

`_load_stage3_field` (132-140) still resolves `cfg["paths"]["solve_out_dir"]/v3/{region}_v3_field.npz`, raises `FileNotFoundError` if missing, slices `full[r0:r1,c0:c1]` by `ctx["win"]`. Same helper as normal path (205-206). No synthetic fallback.

**c) Run A analyzes owner features** — `160-164`:

```py
owner_analysis = np.where(ctx["owner_mask"], ctx["owner_field"], np.nan)
structure_cfg = dict(cfg.get("structure", {}))
features = analyze_owner_features(owner_analysis, ctx["owner_mask"], structure_cfg)
```

Masked NaN outside owner, `structure` dict forwarded. Import is still only `analyze_owner_features` and `build_multiscale_structural_fields` (30-32) — zero `point`/`ribbon` imports.

**d) Run A calls `build_multiscale_structural_fields`** — `165-167`:

```py
fields, structure_report = build_multiscale_structural_fields(field, ctx, features, structure_cfg)
```

Single import tuple `[("build_multiscale_structural_fields", None)]` confirmed; no point-source helper reintroduced.

**e) Four structural PNGs plus metrics under fresh directory and early return** — PASS:

Four writes at `168-175`:

```py
save_run_a("Run A Stage-3 harmonic base", fields["stage3"], "stage3_base")
save_run_a("Run A cleaned fine-detail field", fields["cleaned"], "cleaned_fine")
save_run_a("Run A macro continuation", fields["macro"], "macro_continuation")
save_run_a("Run A macro plus meso continuation", fields["macro_meso"], "macro_meso_continuation")
```

`save_run_a` (154-158) renders via `_render_local` + `save_shade_png` into `out_dir / f"{region}_{suffix}.png"` — one-to-one with the request's four gate outputs.

Metrics `176-188` write `f"{region}_run_a_metrics.json"` with `artifacts` dict of the four PNG paths and flags `"erosion_run": False, "final_lock_run": False`. Segment scan `143-190` contains 0 hits for `erode_field` / `priority_flood_routing_surface` / `_final_seam_lock` / `solve_surface` / `np.savez` / `comparison_sheet` (excluding the benign metric flags) — no hydrology/erosion leakage.

Fresh directory `149-151`:

```py
default_out = _resolve(cfg["paths"]["solve_out_dir"]) / "v3" / "run_a_multiscale_tr"
out_dir = _resolve(output_dir) if output_dir else default_out
out_dir.mkdir(parents=True, exist_ok=True)
```

Distinct from normal path `solve_dir / "erosion_structural"` at `212`. `--output-dir` override honored via `_resolve`. Satisfies "below a new run-specific directory" for both default and caller-supplied paths.

Early return `202-203`:

```py
if args.run_a:
    return _run_a(cfg, args.region, args.output_dir)
```

All hydrology/erosion/final-lock code is strictly after this guard (204-325). Normal path unreachable when `args.run_a`.

**f) Normal path still uses `macro_meso` without reintroducing point-source** — `235-242`:

```py
structural_fields, structural_report = build_multiscale_structural_fields(field, ctx, features, structure_cfg)
structural = structural_fields["macro_meso"]
structural_path = save_stage("Structural continuation before erosion", structural, ...)
```

`structural` feeds `erode_field` at 267. No alternate `fields["stage3"]`/`["macro"]`/`["cleaned"]` misuse. Keyword sweep finds zero `point_guide`/`ribbon`/`guide_seed`/`build_sparse` calls; single `sparse` hit remains only the fixed docstring.

### 5. Config housekeeping outcome

Prior review required removal/deprecation of 13 entries as non-blocking before batch work. Lead executed the preferred deletion (not a comment block) for the 11 truly dead keys while correctly retaining `ridge_percentile`/`valley_percentile`. The two-percentile retention is the intended correction versus the prior review's broader 13-key list — the follow-up instruction enumerates those two as active examples. Result is a lean 13-key `structure` that no longer risks downstream reintroduction of the replaced guide mechanism.

No further cleanup required. Optional nit 3 from prior review ("mention default `.../run_a_multiscale_tr` in `--output-dir` help") remains a cosmetic help-string suggestion and does not affect wiring — code at 150 already implements the default; help text unchanged.

## Verification evidence

- **Compile:** `python -m py_compile tools/terrain/erode_region_v3.py` → `True`; `py_compile` via `compile(..., doraise=True)` succeeds.
- **JSON:** `json.loads(Path("configs/tamriel_reworked_v1.json").read_text())` → ok; `structure` key count 13, sorted list above, `dead present []`, `active missing []`.
- **Dead-key audit:** grep of dead-11 list against `structure` → 0 hits; grep of prior-13 list → only `ridge_percentile`/`valley_percentile` remain (the 2 active saves), other 11 absent.
- **Active-key audit:** `gaussian_scales_verts [8,24,64]`, `tensor_sigma_verts 10.0`, `ridge_percentile 88.0`, `valley_percentile 88.0`, `macro/meso/fine width/keep` floats verified matching `terrain_structure.py` DEFAULTS.
- **Docstring diff:** `t[:600].__repr__` shows new `"solves complete macro/meso harmonic bands"` and `"Run A stops after the structural fields and never enters hydrology, erosion, or final-lock code"`; zero `"sparse semantic guides"` hits.
- **Flag audit:** `ast` walk finds `add_argument("--run-a", store_true)` and `add_argument("--output-dir", default=None)` at 197-198.
- **Wiring probe:** `_run_a` segment 143-190 contains 4 `save_run_a(` calls + 1 metrics write, 0 hydrology/erosion/lock symbols; `main` 202-203 early return precedes every `build_context`/`erode_field`/`_final_seam_lock` reference in normal path.
- **Import sweep:** `terrain_structure` import tuple is `[("build_multiscale_structural_fields", None)]`; no `point`/`ribbon` import.
- **No real crop executed** — static inspection only per task instruction. No filesystem writes outside this review.

## Stop condition

Review stops here. No files edited; real `tr_vvardenfell_wall` crop not executed. After lead inspects this review, the next gate is the visual inspection of the four PNGs under the fresh output directory (default `.../v3/run_a_multiscale_tr` or `--output-dir` override); downstream erosion/final-lock remain out of scope.

## Trace

- Request: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\request.md` (39 lines)
- Plan: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\plan.md` (47 lines, Stage 3 §35-42)
- Prior Stage 3 review: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\2026-08-25_stage3_run_a_driver_review.md` (183 lines, PASS with 1 housekeeping — this review closes that item)
- Stage 2 reviews (context): `2026-08-25_stage2_multiscale_structure_review.md` + `2026-08-25_stage2_multiscale_structure_followup_review.md` (both PASS)
- Source inspected: `F:\ProcGenWorkspace\tools\terrain\erode_region_v3.py` (329 lines)
- Config inspected: `F:\ProcGenWorkspace\configs\tamriel_reworked_v1.json` (373 lines, structure §308-326)
