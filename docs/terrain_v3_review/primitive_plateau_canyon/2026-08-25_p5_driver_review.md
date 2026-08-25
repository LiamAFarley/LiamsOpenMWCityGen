# 2026-08-25 P5 Driver Wiring Review — TR Primitive Plateau/Canyon Checkpoint

**Scope:** Read-only re-review after passed P5 reconciliation source review — fixes-only. Inspected `tools/terrain/erode_region_v3.py` (543 lines, `master` working tree; `primitive-checkpoint` branch not separately tracked) and its imports against `request.md` + `plan.md` §P0-P5. Re-verified the two requested blocking fixes: merged plateau+footprint dict for P5 reconciliation (closing scarp `KeyError`) and top/scarp grayscale cropped to the same review frame. Re-ran bounded driver contract check (compile, synthetic reconcile, import/order probes). No real `tr_vvardenfell_wall` crop executed.

**Verdict: PASS — both P5 driver fixes verified. All other gates remain PASS. Proceed to gated real-crop visual review per plan.md; AMG convergence alone is not visual acceptance.**

Prior verdict (2026-08-25): FAIL (B1 scarp `KeyError`, O1 diagnostic framing) — both now closed.

---

## 1. Gate Summary (re-reviewed — fixes only)

| Check (prompt) | Result | Evidence |
|---|---|---|
| **Bottom-right target fraction resolved from existing review bbox** | **PASS (unchanged)** | `_review_bbox(ctx)` `69-81` → `_primitive_target_bbox(review, primitive_cfg)` `139-154` reads `target_fraction [0.5,0.5,1.0,1.0]` (`configs/tamriel_reworked_v1.json:8`) with validation `145-147` and `r0+round((r1-r0)*frac)` `149-154`. Called `review=_review_bbox(ctx)` `182` then `target_bbox=_primitive_target_bbox(review,primitive_cfg)` `184`. No hard-coded source coordinate; generic fraction mapper. Unchanged. |
| **All P1-P5 calls ordered correctly** | **PASS (unchanged)** | `_run_primitives:220` `analyze_plateaus` → `238` `continue_plateau_footprints` → `251` `synthesize_plateau_candidates` → `257` `continue_canyons` → `263` `reconcile_primitive_candidates` → `271` `_final_seam_lock`. No `erode_field`/`build_multiscale` in `177-333` segment (probe `false`). |
| **Owner/reference and same-frame checkpoint renders produced** | **PASS — fix verified** | `save_field` `194-199` and `save_overlay` `201-211` both `_render_local(review)` `195/206` via `render_split_window(...,r0,r1,c0,c1,pad=64)` `63-66`. 8 renders + comparison sheet `291-302` `A-H` unchanged same `review`. Diagnostic now also same-frame (see row below). |
| **Overlays embed masks with correct bbox** | **PASS (unchanged)** | `_overlay_masks(image,review,masks)` `164-174` slices `mask[r0:r1,c0:c1]` `171`; `plateau_labels=_embed_crop(...,bbox,field.shape)` `223-227` then sliced to review; footprint `support/scarp/seam` already `field.shape` `243-248`. |
| **Final lock is narrow and called without erosion** | **PASS (unchanged)** | `_final_seam_lock` `84-118` `band_cells 2.0` `88` → `smask & dist_seam<=128` `89` + seam-connected filter `91-95` + `ring` `99` + `hard=seam_v|ring` `100` + `solve_surface` `114` once `271` on `reconciled`; segment contains no erosion/hydrology. |
| **Metrics/artifact paths are written** | **PASS (unchanged)** | `artifact_paths` `192` → `save_field/overlay` `198/210` + `top_scarp_grayscale` `289` + `comparison_sheet` `302-303`; `metrics` `308-325` with `review_bbox/target_bbox`, `p1-p5`, `final_lock`, `seam_c0/c1`, `artifacts`, `erosion_run:False` `323`; `FAILURE` gate `330`. |
| **Old `--run-a` baseline remains available** | **PASS (unchanged)** | `--run-a` `401` handler `411-412` before primitive dispatch; `_run_a` `347-394` intact. |
| **P5 reconcile receives merged plateau+footprint (scarp KeyError fix)** | **PASS — fix verified** | `263-265` now `reconcile_primitive_candidates(field, ctx, {**plateau_arrays, **footprint_arrays}, canyon_field, canyon_arrays, ...)` — `plateau_arrays` (`candidate_weight/support/fine_residual` `708-712`) merged with `footprint_arrays` (`scarp_confidence/normal_y/normal_x/support_probability` `618-625`) so callee `1205-1219` sees both `candidate_weight` and `scarp_*`. Synthetic merged `10×10` with outer-ring `hard` → `unknowns 64 finite True`; old `plateau_arrays` alone → `KeyError: 'scarp_confidence'` (closed). |
| **Top/scarp grayscale cropped to same review frame** | **PASS — fix verified** | `276-289`: `diagnostic=zeros(field.shape)` `276`, fill `score_bbox` `278-279`, `clip 0-1` `280`, then `diagnostic_review=diagnostic[review[0]:review[1], review[2]:review[3]]` `282` and `save_shade_png(diagnostic_review, diagnostic_path, px_per_vertex, title="P1 top/scarp score")` `283-288`. Formerly saved full `diagnostic` (`H×W`); now cropped exactly like terrain renders. Synthetic `review(10,50,20,80)` → `diagnostic_review.shape (40,60)` matches `review` window. |

**Compile/synthetic probes (this re-review):** `py_compile tools/terrain/erode_region_v3.py` → `COMPILE_OK`; `py_compile src/procgen/terrain_primitives.py` → `COMPILE_OK`; `spec.loader.exec_module(drv)` import OK; merged `10×10` reconcile `64 unknowns finite True` vs old `KeyError`; diagnostic `40×60` cropped shape verified; `--run-a`/`--primitive-checkpoint` present; no `erode_field` in primitives. No real crop.

---

## 2. Fixes Verified (closing prior Required Fixes)

### RF1 — `reconcile_primitive_candidates` `KeyError: 'scarp_confidence'` — CLOSED

- **Prior:** `263-265` passed `plateau_arrays` (P3 `candidate_weight` only) to callee that reads `plateau_arrays["scarp_confidence"]` `1209`; always `KeyError` before screened-Poisson solve.
- **Now:** `tools/terrain/erode_region_v3.py:263-265`
  ```python
  reconciled, p5_report = reconcile_primitive_candidates(
      field, ctx, {**plateau_arrays, **footprint_arrays}, canyon_field, canyon_arrays,
      primitive_cfg.get("reconciliation", {}),
  )
  ```
  Merge order `{**plateau_arrays, **footprint_arrays}` covers disjoint key sets (`candidate_weight` vs `scarp_*`); no collision. Verification: merged `10×10` with outer `hard` ring → `reconcile` returns `unknowns 64 finite True`; `plateau_arrays` alone still raises `KeyError: 'scarp_confidence'` (proves fix is the merge). Preserves one edge-aware screened-Poisson solve, exact fixed values, no erosion.

### RF2 — Top/scarp grayscale not same-frame — CLOSED

- **Prior:** `save_shade_png(diagnostic, diagnostic_path, ...)` on full `field.shape` (`H×W`), while all terrain renders are `review`-cropped → framing mismatch.
- **Now:** `tools/terrain/erode_region_v3.py:276-289`
  ```python
  diagnostic = np.zeros(field.shape, dtype=np.float32)
  score = p1_arrays["plateau_score"]
  score_bbox = tuple(int(v) for v in p1_arrays["bbox"])
  diagnostic[score_bbox[0]:score_bbox[1], score_bbox[2]:score_bbox[3]] = score
  diagnostic = np.clip(diagnostic, 0.0, 1.0)
  diagnostic_path = out_dir / f"{region}_top_scarp_grayscale.png"
  diagnostic_review = diagnostic[review[0]:review[1], review[2]:review[3]]
  save_shade_png(diagnostic_review, diagnostic_path, int(ctx["render"]["px_per_vertex"]), title="P1 top/scarp score")
  ```
  Diagnostic now slices to `review` `282` before save, matching `_render_local` terrain frame; scale uses same `px_per_vertex`. Verification: synthetic `review (10,50,20,80)` → `diagnostic_review.shape (40,60)` equals review window. `artifact_paths["top_scarp_grayscale"]` still recorded `289`.

Both fixes are narrow, driver-only (no P1-P5 module signature change required), preserve narrow final lock (`2.0 cells`) and `erosion_run:False`, and do not add hydrology/second solve.

---

## 3. Observations (non-blocking — triage for later)

### O1 — `final_lock_cells` still read from `erosion` namespace

- `_final_seam_lock` `88` reads `cfg.get("erosion",{}).get("final_lock_cells",2.0)` — scalar `2.0` correct, but semantically couples checkpoint to erosion block that `request.md` says “Do not run erosion”. Carry from prior review. Consider moving to `terrain_primitives.reconciliation` for explicitness.

### O2 — Untracked file / branch naming

- Workspace on `master` (`master` only branch); `tools/terrain/erode_region_v3.py` is untracked (543 lines) — no `primitive-checkpoint` branch commits as named in prompt. Carry. Recommend committing to expected branch before real crop for bisectability.

---

## 4. Exact References

- Driver: `tools/terrain/erode_region_v3.py:139-154` (`_primitive_target_bbox`), `60-81` (`_render_local`/`_review_bbox`), `164-174` (`_overlay_masks`), `157-161` (`_embed_crop`), `84-118` (`_final_seam_lock:2.0 cells`), `177-333` (`_run_primitives` P1 `220` P2 `238` P3 `251` P4 `257` **fixed P5 `263-265` merged** Lock `271`), **fixed diagnostic `276-289` cropped**, `308-325` (metrics), `347-394` (`_run_a`), `397-417` (dispatch)
- Module callee: `src/procgen/terrain_primitives.py:618-625` (`footprint_arrays` scarp keys), `708-712` (`plateau_arrays` candidate keys), `1191-1228` (`reconcile` reads `candidate_weight` `1205` + `scarp_confidence` `1209`)
- Config: `configs/tamriel_reworked_v1.json:7-8` (`target_review_quadrant:bottom_right` `target_fraction:[0.5,0.5,1.0,1.0]`), `57-65` (`reconciliation`), `435` (`erosion.final_lock_cells:2.0`)
- Plan/request: `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md:44-51`, `request.md` (bottom-right target, one screened-Poisson + narrow lock, no erosion)
- Probes: compile driver+module OK; synthetic merged vs `KeyError` sweep + diagnostic crop shape (`F:\ProcGenWorkspace\.opencode\tmp\p5_driver_fix_probe.py` via `python -c` in this review) — no real crop

---

## 5. Recommendation

**P5 driver re-review PASS.** Both blocking fixes present and behave correctly: merged `{**plateau_arrays, **footprint_arrays}` closes scarp `KeyError` (verified merged `64 unknowns finite True` vs old `KeyError`) and `diagnostic_review` now crops to `review[0]:review[1],review[2]:review[3]` matching terrain renders (verified `40×60` shape). All other driver gates remain PASS (target fraction generic, ordered P1-P5, same-frame renders/overlays, narrow lock without erosion, metrics/artifacts, `--run-a` preserved). Keep fix narrow; do not run erosion or broad batch. Next gate is visual inspection of the real bottom-right `tr_vvardenfell_wall` checkpoint — generate it with this wiring + existing narrow final seam lock and stop for user/lead review.

*Reviewer: review-flash (read-only, P5-driver re-review) — 2026-08-25 — files: `tools/terrain/erode_region_v3.py:1-543` (fixed `263-265`, `276-289`), `src/procgen/terrain_primitives.py:1191-1228`, `configs/tamriel_reworked_v1.json:4-65`, `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md`, `request.md` — probes: compile + synthetic merged/KeyError + diagnostic crop + ordered-contract check (§1) — no crop executed, no erosion.*
