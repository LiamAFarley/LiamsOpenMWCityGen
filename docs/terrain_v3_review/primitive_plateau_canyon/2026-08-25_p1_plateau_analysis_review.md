# 2026-08-25 P1 Plateau Analysis Review — TR Primitive Plateau/Canyon

**Scope:** Re-review of P1 fixes only per `request.md` + `plan.md` §P1. Inspected `src/procgen/terrain_primitives.py` (365 lines) `analyze_plateaus` and supporting helpers. Did not inspect footprint/canyon/reconciliation (P2-P5), did not run the real `tr_vvardenfell_wall` crop. Compile check + narrow synthetic flat-plateau probe executed (no I/O).

**Verdict: PASS — all four P1 fixes verified; no remaining P1 blocker. Proceed to P2.**

**Prior verdict (2026-08-25 initial):** CONDITIONAL PASS — RF1 (tilt not reported) + RF2 (valid-count divisor) blocking. Both now closed plus RF/Obs items for candidate intersection and prominence scale.

---

## 1. P1 Gate Summary (re-reviewed)

| Check (prompt) | Result | Evidence |
|---|---|---|
| **Owner-only authority** | **PASS** | `terrain_primitives.py:260-266` `valid = owner & finite(H64/H24)`; features limited to `owner_mask/H64/H24/slope24/scarp_mask`; `h0` used only for shape/bbox slicing `260-261, 285`. No Stage-3 field. |
| **Semantic downsampling — correct averaging** | **PASS — fix verified** | Factor `config.get("semantic_downsample",4)` at `270`; `sem_score = _block_mean(score, valid, factor)` at `271` now averages over `valid` (finite owner samples) not `owner`. `_block_mean:73-89` divides by `counts` from same `valid` mask. Probe: 4×4 block with one NaN-valid hole → `mean_valid=10.0` vs bug `mean_owner=9.375` (probe §probe_p1.py). |
| **Elevation / flatness / prominence / scarp score + config prominence_scale_gu** | **PASS — fix verified** | `_plateau_score:202-249`; `prominence_score = clip((prom - prom_cut)/max(|prom_cut|, config.get("prominence_scale_gu",256))+0.5)` at `228-232`. Config `configs/tamriel_reworked_v1.json:18` `prominence_scale_gu:256.0` wired through `plateau_cfg`. Probe: switching `prominence_scale_gu` 64→1024 changes `score` mean 0.0901→0.0888 (isolated unit test). |
| **Component filtering / contact with production seam + candidate intersection** | **PASS — fix verified** | `sem_candidate` close/size-filter `275-283`; `sem_owner_seam = _block_max(owner_seam_full,factor)` `286`; sem contact test `293`; upsample `full_component = _upsample_semantic(sem_component,factor,owner.shape) & owner & candidate` at `295-299` now intersects full-resolution `candidate` mask before `valid` sampling. Probe: single sem block upsample 16 verts with 4-vert candidate hole → `full_new = full_old -4` as expected; flat-plateau pipeline probe retains only candidate vertices in `full_labels`. |
| **Robust Huber affine/quadratic top fit + RMS/p95/tilt** | **PASS — fix verified** | `_huber_plane_fit:121-183` 8-iteration IRLS `delta=max(64,1.5*mad)` `147`; `rms/p95` `150-151`; quadratic branch `153-167` with `quadratic_regularization`/`quadratic_trigger_gu` from config `307-309`. **Tilt now reported** `169-182`: `tilt_x = coeff[0]/scale`, `tilt_y = coeff[1]/scale`, `tilt_magnitude = hypot(...)`, returned as `tilt_x_gu_per_gu / tilt_y_gu_per_gu / tilt_magnitude_gu_per_gu` plus `fit_rms_gu/fit_p95_gu/sample_count/center/scale/coefficients/order`. Flat-probe tilt `0.0`, sloped-plane probe tilt `20.0/30.0/36.06` within 0.5 GU/vert; present in `primitive.diagnostics["fit"]` `330` and `component_summaries[].fit` `345`. |
| **Config-driven thresholds** | **PASS** | All P1 thresholds via `config.get`/`plateau_cfg.get` matching `tamriel_reworked_v1.json:4-30`; no hard-coded land coordinate. |
| **No generated terrain writes / owner immutability** | **PASS** | Pure helpers, no I/O, views only; `valid` masking ensures owner-only heights. `py_compile` clean. |

Compile: `py_compile terrain_primitives.py` → `compile OK` (probe). Narrow synthetic flat-plateau probe (32×32, uniform 8000 GU plateau, flat slope 0.2 vs 5 outside, seam at left edge): 1 primitive, `owner_vertices=132`, `tilt_magnitude=0.0`, `rms=0.0`, `order=affine` — seam-touching selection and near-zero tilt behave as expected.

---

## 2. Fixes Verified (closing prior Required Fixes)

### RF1 — Report tilt — CLOSED
- Prior: no tilt key; `grep tilt` = 0.
- Now: `terrain_primitives.py:169-182` computes `tilt_x/y/magnitude` from affine coefficients (linear terms) divided by `coordinate_scale`; quadratic case reuses same linear terms (center gradient). Surfaced at `310-312` `fit["row_offset"/"col_offset"]` + tilt keys, stored `329-335` and in `report.selected_components[].fit`. Satisfies `plan.md:17` "report RMS/p95/tilt".

### RF2 — `_block_mean` denominator — CLOSED
- Prior: `sem_score = _block_mean(score, owner, factor)` biased by NaN holes.
- Now: `271` passes `valid`; `_block_mean:73-89` uses `valid` for both values and counts. Probe confirms exclusion of single NaN owner vertex.

### Candidate intersection — CLOSED
- Prior observation became required fix per prompt.
- Now: `295-299` `& candidate` ensures upsampled semantic components do not re-introduce full-resolution non-candidate fringe vertices; `300` `rows,cols = nonzero(full_component & valid)` then fits only candidate+valid samples; `321` `confidence = mean(score[full_component])` averages over same set.

### Prominence scale config — CLOSED
- Prior `224` hard-coded `256`.
- Now `229-231` `float(config.get("prominence_scale_gu",256.0))` with JSON `18` `256.0`; fallback preserved for reproducibility.

---

## 3. Observations (non-blocking — triage for later)

### O1 — Tilt key units naming
`tilt_*_gu_per_gu` suggests GU per GU, but `scale` is in vertex steps, so the computed value is GU per vertex (GU/vert). Correct magnitude for P1 decisions; if a GU-per-cell or GU-per-GU convention is adopted elsewhere, document or rename to `tilt_*_gu_per_vert` later. Not a P1 blocker.

### O2 — Quadratic tilt is center linear term
For `order=quadratic` the reported tilt is the linear-term gradient at the fitted center, not a max-gradient across the component. Sufficient for P1 `tilt` gate; max-gradient diagnostic can be added in P2/P5 if needed.

### O3 — Full-resolution seam contact reporting
`304` `contact = argwhere(full_component & owner_seam_full)` may be 0 even when sem-level contact `293` passed (owner seam column adjacent but not inside candidate block, as seen in flat probe `seam_contact_vertices=0`). Sem-level gating is authoritative per plan; full-level count is diagnostic only. No change required for P1.

### O4 — Prior observations O2/O3 retained
Scarp as candidate gate not multiplicative score term (`235-239`), and hard-coded IRLS/closing/dilation iteration counts (`275,286,142,147`) remain non-configurable. Flagged previously; leave as-is for P1.

---

## 4. Exact References

- Module: `src/procgen/terrain_primitives.py:1-365` (helpers `_block_mean:73-89`, `_block_max:92-99`, `_upsample_semantic:102-105`, `_owner_seam_mask:108-118`, `_huber_plane_fit:121-183`, `_plateau_score:202-249`, `analyze_plateaus:252-365`)
- Fixes: `valid` divisor `271`; candidate intersection `295-299`; tilt `169-182`; prominence scale `228-231`, `17-18` in JSON
- Config: `configs/tamriel_reworked_v1.json:4-30` (`semantic_downsample:4`, `plateau.min_component_vertices_semantic:24` default `276-278`, `prominence_scale_gu:18`)
- Plan/request: `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md:12-18`, `request.md`
- Probes: `F:\ProcGenWorkspace\probe_p1.py` (compile + block/upsample/tilt/prominence scale + flat-plateau pipeline); no real crop, no P2-P5 code inspected.

---

## 5. Recommendation

**P1 re-review PASS.** All four requested P1 fixes are present and behave correctly in narrow synthetic probes. No remaining P1 blocker. Proceed to P2 (footprint/scarp). Do not run the real bottom-right crop until P2 review gate per `plan.md`.

*Reviewer: review-flash (read-only, P1-only re-review) — 2026-08-25 — files: `src/procgen/terrain_primitives.py:1-365`, `configs/tamriel_reworked_v1.json`, `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md`, `request.md` — probes: compile + synthetic block/seam/Huber/score/flat-plateau probes — no crop executed, no P2-P5 inspected.*
