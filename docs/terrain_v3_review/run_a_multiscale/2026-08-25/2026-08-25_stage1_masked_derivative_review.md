# 2026-08-25 Stage 1 Masked Derivative Review — Run A

**Scope:** Stage 1 only — `src/procgen/terrain_features.py` masked derivative operator.
**Spec source:** `.opencode/runs/2026-08-25_tr-run-a-multiscale/plan.md` Stage 1 + `request.md`
**File inspected:** `F:\ProcGenWorkspace\src\procgen\terrain_features.py` (292 lines, currently untracked in git — not yet committed, no diff vs HEAD)
**Reviewer:** review-flash (read-only, no edits)
**Date:** 2026-08-25

## Verdict: PASS — Stage 1 implements the intended masked operator correctly. No blocking fixes required to proceed to Stage 2. Minor observations below.

## Findings (first)

1. **Masked derivative semantics correctly implemented.** Central diff when both same-axis neighbors valid, one-sided when exactly one valid, NaN otherwise. Matches plan wording verbatim.
2. **Gradient of H24 does not use zero-filled non-owner terrain.** H24 is set to NaN outside `owner_mask` (lines 142-144), derivative validity is `owner_mask & isfinite(H24)` (line 147), and differentiation goes through `_masked_derivative` not a zero-filled field.
3. **Hessian of H24 likewise masked.** Second derivatives differentiate `gx`/`gy` with `_masked_derivative(..., isfinite(gx/gy), ...)` (lines 175-177) — same one-sided/NaN policy, no zero-fill interpolation of H24 or gx/gy across the boundary.
4. **Structure-tensor products use normalized-mask Gaussian.** `Jxx`/`Jyy`/`Jxy` are smoothed via `_normalized_gaussian(..., gradient_valid, sigma_tensor)` (lines 154-156) where `gradient_valid = isfinite(gx) & isfinite(gy)` (line 150) and products are `gx_finite*gx_finite` etc. with `nan_to_num(...,0)` input but weight-normalized by `gradient_valid` so non-owner/NaN pixels contribute zero energy. Correct per spec "existing normalized-mask Gaussian routine so non-owner pixels never contribute derivative energy."
5. **Boundary wrap safety handled.** `np.roll` wraps but `minus`/`plus` masks are false at array edges and at valid/invalid transitions (lines 77-88), so wrapped values are never read in `out[both]`/`out[only_*]` assignments.
6. **No structural scope creep.** Only Stage-1 file touched; Stage-2/3 files not inspected per stop condition.

No speculative, hardcoded, or zero-filled leakage paths remain for owner gradient/Hessian energy.

## Exact file/line references

- `_masked_derivative` definition and contract: `src/procgen/terrain_features.py:66-99`
  - Guard `axis not in (0,1)` raise: line 73-74
  - `valid = valid & isfinite(field)`: line 76
  - `minus`/`plus` neighbor-valid construction via sliced assignment: lines 77-88
  - `backward = roll(field,1)`, `forward = roll(field,-1)`: lines 90-91
  - `out = full(NaN)`, `both = valid & minus & plus`, `only_minus`, `only_plus`: lines 92-95
  - Central `(forward-backward)/(2*spacing)`, one-sided `(field-backward)/spacing` and `(forward-field)/spacing`: lines 96-98
- H-pyramid NaN masking outside owner: lines 142-144 (`H8[~owner_mask]=nan` etc.)
- Gradient creation from masked H24: lines 146-150
  ```py
  derivative_valid = owner_mask & np.isfinite(H24)   #147
  gy = _masked_derivative(H24, derivative_valid, spacing, axis=0)  #148
  gx = _masked_derivative(H24, derivative_valid, spacing, axis=1)  #149
  gradient_valid = np.isfinite(gx) & np.isfinite(gy)              #150
  ```
- Structure-tensor smoothing with normalized mask: lines 152-156
  ```py
  gx_finite = nan_to_num(gx, nan=0.0)                              #152
  Jxx = _normalized_gaussian(gx_finite*gx_finite, gradient_valid, sigma_tensor) #154
  Jyy = _normalized_gaussian(gy_finite*gy_finite, gradient_valid, sigma_tensor) #155
  Jxy = _normalized_gaussian(gx_finite*gy_finite, gradient_valid, sigma_tensor) #156
  ```
- Hessian via masked derivative of gx/gy: lines 175-180
  ```py
  dxx = _masked_derivative(gx, np.isfinite(gx), spacing, axis=1)  #175
  dyy = _masked_derivative(gy, np.isfinite(gy), spacing, axis=0)  #176
  dxy = _masked_derivative(gx, np.isfinite(gx), spacing, axis=0)  #177
  dxx = nan_to_num(dxx, nan=0.0)                                  #178
  ```
- Normalized Gaussian helper (weights `>1e-6`, mode nearest): lines 50-63

## Required fixes

**None blocking.** Stage 1 may proceed to Stage 2 review gate. Optional low-priority observations for lead to consider (not required for Run A acceptance):

1. **Hessian symmetry — informational.** `dxy` is `∂gx/∂y` only; the symmetric counterpart `∂gy/∂x` is not averaged (line 177). Results are nominally symmetric for interior points but differ at one-sided mask edges. If strict Hessian symmetry is desired, consider `0.5*(∂gx/∂y + ∂gy/∂x)` with shared validity. Current choice is consistent with "same-axis one-sided" rule and does not violate spec.
2. **Hessian zero-fill after derivative — informational.** Lines 178-180 `nan_to_num(...,0)` zeroes non-finite Hessian outside valid region before eigenvalue computation. Curvature scores are later masked by `valid` (lines 191-198, 205-211), so external zeros do not leak into feature thresholds, but the zeroing is implicit. No fix needed; documented here for completeness.
3. **Git status — housekeeping.** File is currently untracked (`?? src/procgen/terrain_features.py` as of 2026-08-25). Stage-1 commit should add it with the `??`→`M` transition observed in `git status`; no content change needed.

Do not refactor valley/ridge, plateau, erosion factor, or any Stage 2/3 logic as part of Stage 1.

## Verification evidence

- **Compile check (narrow, read-only):** `python -m py_compile src/procgen/terrain_features.py` → exit `True` (success) — verified 2026-08-25, no syntax/type import errors. `scipy.ndimage` import resolved.

- **Synthetic masked-edge probe 1 — ramp field 5×5, spacing 1:** `field[y,x]=y*10+x`, `valid[1:4,4]=False` (holes). Results:
  - `gx` central interior `1.0`, one-sided at column 3 with missing right neighbor `gx[1,3]=1.0` (correct `(13-12)/1` not zero-filled `-6`); `gx[1,4]` = NaN, `gx[2,4]` = NaN. NaN count `gx:3, gy:5` matches per-axis missing-neighbor rule.
  - `gy` central `10.0`; `gy[2,2]=10.0`, `gy[0,0]=10.0` via one-sided down; `gy[0,4]=NaN` (no vertical neighbors) — confirms "NaN otherwise" for axis-isolated pixels.
  - Structure tensor: `Jxx_norm[2,2]=1.0` vs plain Gaussian `0.948` — normalized path preserves energy without zero bleed; `Jxx_norm[1,3]=1.0` confirms edge products contribute via weight renormalization.

- **Synthetic probe 2 — isolated pixel 3×3 single valid center:** `gx` and `gy` all NaN including center `nan` — correct, both neighbors missing → NaN, not zero.

- **Synthetic probe 3 — 2-column strip 2×3:** `gx` correctly `1.0` via `only_plus` at col0 and `only_minus` at col1; `gy` correctly `0.0` via central/average along y. Confirms one-sided branch works on minimal width.

- **Integration probe — `analyze_owner_features` 16×16 random owner field, owner_mask 10×8 window:** `H24[~owner_mask]` all NaN (not zero-filled), `orientation_angle` finite count 80 inside mask, `coherence` in `[0.9999, 0.9999]` — no crash, mask-respecting pipeline.

All probes executed via `python -c` with `sys.path.insert(0,'src')` — no source edits, no filesystem writes outside this review.

## Stop condition

Review stops here. Later structural stages (multiscale continuation, driver, renders) were not inspected or redesigned per instruction.

## Trace

- Request: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\request.md` (39 lines)
- Plan: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\plan.md` (47 lines, Stage 1 § patched terrain_features.py)
- Source: `F:\ProcGenWorkspace\src\procgen\terrain_features.py` (292 lines)
