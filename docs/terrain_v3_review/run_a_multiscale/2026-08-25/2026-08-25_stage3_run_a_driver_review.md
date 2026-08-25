# 2026-08-25 Stage 3 Run A Driver Review — Wiring Only (No Crop Execution)

**Scope:** Stage 3 only — `tools/terrain/erode_region_v3.py` Run A wiring + `configs/tamriel_reworked_v1.json` structure keys. No crop executed, no files edited.
**Spec source:** `.opencode/runs/2026-08-25_tr-run-a-multiscale/request.md` (39 lines) + `plan.md` (47 lines, Stage 3) + Stage 1/2 review files.
**Files inspected:** `F:\ProcGenWorkspace\tools\terrain\erode_region_v3.py` (329 lines) and `F:\ProcGenWorkspace\configs\tamriel_reworked_v1.json` (384 lines, structure §308-337).
**Reviewer:** review-flash (read-only, no edits, no execution of real crop)
**Date:** 2026-08-25

## Verdict: PASS — Stage 3 Run A wiring is correct. One housekeeping cleanup required before batch work (dead config keys), otherwise ready to run `tr_vvardenfell_wall` into a fresh directory.

## Findings (checklist order)

### 1. `--run-a` and `--output-dir` are explicit — PASS

`tools/terrain/erode_region_v3.py:194-198`:

```py
ap.add_argument("--config", default=str(ROOT / "configs" / "tamriel_reworked_v1.json"))
ap.add_argument("--region", default="tr_vvardenfell_wall")
ap.add_argument("--run-a", action="store_true", help="render structural Run A only")
ap.add_argument("--output-dir", default=None, help="Run A output directory override")
```

Both flags exist, spelled exactly `--run-a` (boolean) and `--output-dir` (optional override, defaults `None`). Matches plan "explicit Run A mode and an output-directory override". Verified via `ast` walk and `py_compile` OK.

### 2. Run A loads the real Stage-3 field — PASS

`_run_a` at lines 143-147:

```py
ctx = build_context(ROOT, cfg, region)          # 146 — real atlas/corpus context
field = _load_stage3_field(cfg, ctx, region)    # 147 — real v3 field
```

`_load_stage3_field` (132-140) resolves `cfg["paths"]["solve_out_dir"]/v3/{region}_v3_field.npz`, raises `FileNotFoundError` if missing, reads `z["field"]` and slices `full[r0:r1, c0:c1]` by `ctx["win"]`. No synthetic fallback. Same helper is used by the normal path (205-206), so Run A exercises the identical real-field path.

### 3. Run A analyzes owner features — PASS

Lines 160-164:

```py
owner_analysis = np.where(ctx["owner_mask"], ctx["owner_field"], np.nan)  # 160
structure_cfg = dict(cfg.get("structure", {}))                             # 161
features = analyze_owner_features(owner_analysis, ctx["owner_mask"], structure_cfg)  # 162-164
```

`owner_analysis` is NaN outside `owner_mask`, matching `analyze_owner_features(mask, field)` contract (`terrain_features.py` signature `owner_field, owner_mask, config`). Config is forwarded as a plain `structure` dict.

### 4. Run A calls `build_multiscale_structural_fields` — PASS

Lines 165-167:

```py
fields, structure_report = build_multiscale_structural_fields(field, ctx, features, structure_cfg)
```

Single import from `terrain_structure` at lines 30-32 is only `build_multiscale_structural_fields`; no point-source helper is imported (verified via `ast` — zero `point`/`ribbon` imports).

### 5. Run A writes exactly four structural PNGs plus metrics under the requested fresh directory and returns before downstream code — PASS

Four PNG writes at lines 168-175:

```py
save_run_a("Run A Stage-3 harmonic base", fields["stage3"], "stage3_base")          # 168
save_run_a("Run A cleaned fine-detail field", fields["cleaned"], "cleaned_fine")    # 169
save_run_a("Run A macro continuation", fields["macro"], "macro_continuation")        # 170
save_run_a("Run A macro plus meso continuation", fields["macro_meso"], "macro_meso_continuation")  # 171-175
```

`save_run_a` (154-158) renders via `_render_local` + `save_shade_png` into `out_dir / f"{region}_{suffix}.png"`. Mapping to the request's four gate outputs: Stage-3 harmonic base / cleaned fine-detail / macro continuation / macro+meso continuation — one-to-one.

Metrics at lines 176-188:

```py
metrics = {"run":"A", "region":region, "review_bbox_vertices":list(review),
           "features":features["feature_counts"], "structure":structure_report,
           "artifacts":render_paths, "erosion_run":False, "final_lock_run":False, ...}
with open(out_dir / f"{region}_run_a_metrics.json", "w") ...
```

No `np.savez`, no `comparison_sheet`, no `erosion_final` inside `_run_a` (grep 0 hits in the 143-190 segment). `artifacts` dict holds the four PNG paths.

Fresh output directory at lines 149-151:

```py
default_out = _resolve(cfg["paths"]["solve_out_dir"]) / "v3" / "run_a_multiscale_tr"  # 149
out_dir = _resolve(output_dir) if output_dir else default_out                          # 150
out_dir.mkdir(parents=True, exist_ok=True)                                             # 151
```

Distinct from the normal path's `solve_dir / "erosion_structural"` at line 212. `--output-dir` overrides the default via `_resolve`, so caller-supplied `fresh output directory` is honored; default itself is a new run-specific sibling (`run_a_multiscale_tr`) that does not collide with `erosion_structural`. Both satisfy "below a new run-specific directory".

Early return at lines 202-203 in `main`:

```py
if args.run_a:
    return _run_a(cfg, args.region, args.output_dir)
```

Hydrology / erosion / final-lock code lives strictly after this guard (lines 204-325). Segment scan of `_run_a` shows `erode_field`, `priority_flood_routing_surface`, `_final_seam_lock`, `solve_surface`, `tmet.seam_c0`, `snapshot` all absent (except the two benign `"erosion_run": False` / docstring `"never enter hydrology or erosion"` strings). `main`'s normal path correctly continues only when `args.run_a` is false.

### 6. Normal path uses `macro_meso` as its structural input without reintroducing point-source imports — PASS

Normal path at lines 235-242:

```py
structural_fields, structural_report = build_multiscale_structural_fields(field, ctx, features, structure_cfg)  # 235
structural = structural_fields["macro_meso"]                                                                   # 238
structural_path = save_stage("Structural continuation before erosion", structural, ...)                         # 239-242
```

`structural` is then fed to `erode_field` at line 267. No alternate `structural_fields["stage3"]` / `["macro"]` / `["cleaned"]` is used as the erodible surface.

Import audit (lines 30-32) shows only `build_multiscale_structural_fields` from `procgen.terrain_structure`. Keyword sweep of the driver finds zero `point_guide`/`ribbon`/`guide_seed`/`build_sparse` imports or calls; the single `sparse` hit is the generic docstring phrase `"solves sparse semantic guides"` at line 4. After the fix to `terrain_structure.py` (NaN-preserving owner bands, pure harmonic `W=0`), no point-source mechanism is re-wired through the driver.

### 7. Structure keys in `tamriel_reworked_v1.json` — PASS with housekeeping

Present and correct at lines 308-322:

```json
"structure": {
  "macro_width_cells": 8.0,              // 309 — matches DEFAULTS 8.0
  "meso_width_cells": 4.0,               // 310 — matches DEFAULTS 4.0
  "fine_keep_at_seam": 0.2,              // 311 — spec 0.2 at seam
  "fine_restore_distance_cells": 6.0,    // 312 — spec 1.0 at 6 cells
  "owner_analysis_halo_cells": 8.0,      // 313
  "gaussian_scales_verts": [8.0,24.0,64.0],
  ...
}
```

The four tunable width/keep values required by the plan are present with correct floats and are consumed by `terrain_structure.py` (`c["macro_width_cells"]` at 242, `c["meso_width_cells"]` at 247, `c["fine_keep_at_seam"]`/`c["fine_restore_distance_cells"]` at 229-232 via `_smootherstep`). `DEFAULTS` in `terrain_structure.py:30-38` matches these values, so config overrides align with code defaults.

`terrain_structure.py` only reads eight `structure` keys (`macro_width_cells`, `meso_width_cells`, `fine_keep_at_seam`, `fine_restore_distance_cells`, `linear_solver`, `amg_max_coarse`, `cg_tol`, `cg_maxiter`) — sweep of the source confirms no other `structure` key is consumed.

## Required fixes

**One housekeeping item — non-blocking for the Run A render gate, but fix before batch runs:**

1. **Remove (or explicitly deprecate) 13 dead point-source keys still in `configs/tamriel_reworked_v1.json:323-333`.** These are remnants of the replaced guide mechanism and are now unread by `terrain_structure.py`:

   ```
   guide_seed_stride_verts 64, guide_ribbon_sigma_verts 12.0, guide_max_cells 6.0,
   massif_guide_max_cells 8.0, guide_turn_deg_per_8_verts 12.0, guide_score_threshold 0.35,
   guide_decay_fraction 0.55, ridge_percentile 88.0, valley_percentile 88.0,
   ridge_weight 0.7, valley_weight 1.0, plateau_top_weight 0.3, scarp_weight 0.8
   ```

   Either delete them (preferred — keeps `structure` to the eight active keys plus `owner_analysis_halo_cells`/`gaussian_scales_verts`/`tensor_sigma_verts` which `terrain_features` still uses) or add a `_deprecated` comment block so future readers do not assume they affect macro/meso continuation. The Stage 2 review predicted removal; leaving them inflates the config and risks a downstream reintroduction.

**Optional cosmetic nits (do not block):**

2. Driver docstring `tools/terrain/erode_region_v3.py:1-8` still says `"solves sparse semantic guides, routes owner inflow..."` — update to note the generic `W=0` harmonic band solve for Run A, matching `terrain_structure.py:9-13` which was already clarified.

3. Consider making `--output-dir` `required=False` wording in help note that the default is `.../v3/run_a_multiscale_tr` — already true via code at 150, help string could mention it for discoverability.

No code change is required in `erode_region_v3.py` logic for the Run A gate; the four checks above all pass with exact line references.

## Verification evidence

- **Compile:** `python -m py_compile tools/terrain/erode_region_v3.py` → `True`; `py_compile src/procgen/terrain_structure.py` → OK (via Stage 2 review).
- **JSON:** `json.loads(Path("configs/tamriel_reworked_v1.json").read_text())` → ok; structure keys enumerated, four width/keep values match `DEFAULTS`.
- **Flag audit:** `ast` walk of `erode_region_v3.py` finds `add_argument("--run-a", store_true)` and `add_argument("--output-dir", default=None)`; no other Run A flag.
- **Wiring probe:** `_run_a` segment 143-190 contains 4 `save_run_a(` calls + 1 metrics write, 0 hits for `erode_field`/`priority_flood`/`_final_seam_lock`/`solve_surface`/`np.savez`/`comparison_sheet`; metrics flags `erosion_run False / final_lock_run False` present.
- **Early-return probe:** `main` 202-203 `if args.run_a: return _run_a(...)` precedes every `build_context`/`_load_stage3_field`/`erode_field`/`_final_seam_lock` reference in the normal path.
- **Normal-path structural input:** line 238 `structural = structural_fields["macro_meso"]` observed.
- **Import sweep:** `terrain_structure` import tuple is `[("build_multiscale_structural_fields", None)]`; no `point`/`ribbon` import reintroduced.
- **No real crop executed** — review is static inspection only per task instruction.

## Stop condition

Review stops here. Do not execute the real `tr_vvardenfell_wall` crop until lead has inspected this review; after visual inspection of the four PNGs under the fresh output directory, downstream erosion/final-lock remain out of scope.

## Trace

- Request: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\request.md`
- Plan: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\plan.md`
- Stage 1 review: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\2026-08-25_stage1_masked_derivative_review.md` (PASS)
- Stage 2 review: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\2026-08-25_stage2_multiscale_structure_review.md` (PASS)
- Stage 2 follow-up: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\2026-08-25_stage2_multiscale_structure_followup_review.md` (PASS)
- Source inspected: `F:\ProcGenWorkspace\tools\terrain\erode_region_v3.py` (329 lines)
- Config inspected: `F:\ProcGenWorkspace\configs\tamriel_reworked_v1.json` (structure keys 308-337)
- Supporting source (not re-reviewed, referenced): `F:\ProcGenWorkspace\src\procgen\terrain_structure.py` (381 lines)
