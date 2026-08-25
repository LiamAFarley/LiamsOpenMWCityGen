# 2026-08-25 Stage 2 Multiscale Structure Follow-up Review — Run A

**Scope:** Stage 2 only — re-review of `src/procgen/terrain_structure.py` after lead fix. Stage 3 not inspected.
**File inspected:** `F:\ProcGenWorkspace\src\procgen\terrain_structure.py` (381 lines)
**Prior review:** `.opencode/runs/2026-08-25_tr-run-a-multiscale/2026-08-25_stage2_multiscale_structure_review.md` (2026-08-25, PASS with 2 informational notes)
**Reviewer:** review-flash (read-only, no edits)
**Date:** 2026-08-25

## Verdict: PASS — Lead fix verified. Follow-up closes prior informational/contract gap; remaining Stage 2 contract unchanged.

## Findings (fix-targeted)

### 1. Owner bands now preserve NaN outside authoritative owner terrain — FIX VERIFIED

Prior review flagged `owner_macro = H24-H64` and `owner_meso = H8-H24` zeroed outside owner via `nan_to_num(nan=0)` (informational; prior code at plan lines 224-225). That allowed a non-owner zero to pass a finite check and be injected as seam/anchor Dirichlet data, contradicting authoritative-owner semantics.

Current `build_multiscale_structural_fields` L225-226:

```py
owner_macro = features["H24"] - features["H64"]
owner_meso  = features["H8"]  - features["H24"]
```

No `nan_to_num` on owners. `features["H24"]/["H64"]/["H8"]` are Stage-1 masked pyramids (NaN outside `owner_mask`), so subtraction propagates NaN exactly as required. `nan_to_num` remaining in file (3 hits) is only for `ctx["dist_seam"]` (L229), `h0` (L281), and `guide_weight` (L284) — none on owner bands. Grep confirms 0 `point`/`ribbon` hits, `sparse` only as generic solver wording.

### 2. Seam Dirichlet now rejects non-owner/invalid band samples — FIX VERIFIED

`_seam_band_values` L83-87:

```py
if not (seam[sy,sx] and 0 <= oy < ...): continue
candidate = float(owner_band[oy, ox])
if not np.isfinite(candidate): continue
```

A seam vertex whose owner-adjacent `owner_band` sample is NaN/Inf is skipped — `values[sy,sx]` retains `target_band` and `assigned` stays false. Conflict counting (`seam_claim_conflicts`, `seam_claim_spread_max` L88-93) unchanged and still first-value-wins, which prior review accepted as in-spec (only anchors must omit shared corners).

Probe 1 (5x5, one valid owner cell at 1,1): seam (2,1) normal (1,0) → owner (1,1)=7.0 adopted; seam (2,2) → owner (1,2)=NaN correctly rejected, value stays 0.0, 0 conflicts. See Evidence §3 Probe 1.

### 3. First-inland anchors now reject invalid-band and non-active samples — FIX VERIFIED

`_first_inland_band_anchors` L127-145 adds `skipped_invalid` branch coverage beyond the prior corner-omission (`len(incidence)!=1` L120-122, `skipped_corner`):

- out-of-bounds `oy/by` or `fy` → `skipped_invalid` (L127-137)
- `~active[fy] | seam[fy]` → `skipped_invalid` (L138-139)
- `~isfinite(b0) | ~isfinite(bout)` → `skipped_invalid` (L141-145)

Candidate `b0 + (b0-bout)` (L146) only when both owner samples are finite and the inland cell is active. Duplicate `fy` collisions still counted as `skipped_corner` (L147-150), unchanged.

Probe 2a/2b/2c: seam (2,2) normal (1,0), fy=(3,2) — NaN `bout` → `anchor_mask` false, `skipped_invalid` 1; both finite (b0=5.0 bout=3.0) → anchor 7.0 set; `fy` inactive → skipped. See Evidence §3 Probe 2.

### 4. W=0 harmonic docstring clarified — FIX VERIFIED

Prior informational note asked to note the Run A pure-harmonic specialization. Module docstring L9-13 now reads:

> `The generic correction C ... satisfies (L+W)C = W*(Hguide-H0) with seam and outer-boundary Dirichlet constraints. Run A sets W=0 for a pure second-order harmonic band solve, which remains AMG-friendly.`

Previously `W` was described only in generic form. Correct and now unambiguous that `_solve_band` L181-188 calls `solve_screened_structure(..., guide_weight=zeros)` for `L C = boundary RHS`.

### 5. Prior multiscale contract remains intact

Checked unchanged from prior PASS:

- **No ribbon/point code.** Same two `guide_value`/`guide_weight` plumbing occurrences (L185-187, L272-273) as before; `L` assembly (L294-333) and AMG-RS/CG fallback unchanged.
- **Bands:** `target_macro = target24-target64`, `target_meso = target8-target24` (L223-224) from `_normalized_band(..., generated_valid, sigma)` with sigma 8/24/64 (L220-222); widths `macro_width_cells 8.0` / `meso 4.0` via `dist_seam <= width*64` (L169).
- **Harmonic band plumbing:** `active = generated & dist<=width*64` (L169), `outer = _complete_band_boundary(active,seam)` (L171, `active & ~seam & ~eroded`, L61-65), `fixed = seam|outer|anchor` (L172,177), `fixed_values[seam]=seam_values[seam]` and `fixed_values[anchor]=anchor_values[anchor]` (L178-180), `correction_fixed[outer]=0` via `target_band[outer]` (solver L287-288).
- **Slope anchors omit corners:** `incidence[flat]` len!=1 skip and duplicate-`fy` skip still produce `skipped_corner`; L-shaped two-normal probe (12x12 sharing flat 5,5) still yields 2 anchors / 2 skipped (Evidence Probe 4).
- **Smootherstep & owner restore:** `_smootherstep` (L42-44) and `keep = 0.2 + 0.8*smootherstep(distance/restore)` (L229-233) give keep(0)=0.2, keep(192)=0.6, keep(384)=1.0 (Evidence Probe 4); `field[owner_mask]=h0[owner_mask]` for `cleaned/macro/macro_meso` (L251-252) verified in probe 3 (owner unchanged true for all fields).
- **AMG-friendly:** `solve_screened_structure` diagonal `degree+weights` (L329), `pyamg.ruge_stuben_solver` (L340) with `jacobi_cg` fallback, synthetic harmonic interior 4.0 at col 7 between 8 and 0 confirmed (Evidence Probe 4).

## Verification evidence

- **Compile:** `python -m py_compile src/procgen/terrain_structure.py` → `compile: OK` (§8).

- **Keyword / NaN audit (§9-10):** `nan_to_num` 3 hits — only `dist_seam` L229, `h0` L281, `guide_weight` L284; owner lines L225-226 have none. `np.isfinite(candidate)` 1, `np.isfinite(b0)` 1, `skipped_invalid` 7, docstring contains `W = 0` and `pure second-order harmonic`. `point` 0, `ribbon` 0, `sparse` 4 generic-solver mentions plus header disclaimer — no feature-line code.

- **Probe 1 — seam NaN-owner rejection (5x5):** Valid owner at (1,1)=7.0 adopted at seam (2,1); NaN owner at (1,2) rejected at seam (2,2), value stays 0.0.

- **Probe 2 — anchor invalid rejection (5x5, seam 2,2 normal 1,0 fy 3,2):**
  - 2a NaN `bout` → `anchor_mask[3,2]` false, `skipped_invalid` 1
  - 2b finite b0=5.0 bout=3.0 → anchor 7.0 (`5+(5-3)`) set, `skipped_invalid` 0
  - 2c `fy` inactive → skipped, `skipped_invalid` 1

- **Probe 3 — full build with NaN-owner pyramids (10x10, owner cols 0-2, seam col 3):** `H8/H24/H64` NaN outside owner. `cleaned[owner]`/`macro[owner]` unchanged (allclose true), `macro[generated]` finite, seam `skipped_invalid` 0 path exercised; `nan_to_num` not leaking owner zeros into Dirichlet values.

- **Probe 4 — prior-contract regression (re-ran prior probes):** L-corner 12x12 two normals sharing flat (5,5) → `first_inland_anchor_count 2 skipped_corner 2`; `_complete_band_boundary` for 15x15 4-col active strip → `outer 19` includes top/bottom/far edge; harmonic solve `jacobi_cg` → col 7 = 3.9992 expected 4.0, seam error 0, smootherstep keep sequence 0.2/0.6/1.0 at 0/192/384.

All probes via `python -c` with `sys.path.insert(0,'src')` and `linear_solver=jacobi_cg` (AMG path structurally identical); no source edits, no Stage 3 inspection.

## Stop condition

Follow-up stops here. No source edits made; Stage 3 driver/configs not inspected per instruction.

## Trace

- Prior review: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\2026-08-25_stage2_multiscale_structure_review.md`
- Source: `F:\ProcGenWorkspace\src\procgen\terrain_structure.py` (381 lines)
- Plan/request: `.opencode/runs/2026-08-25_tr-run-a-multiscale/plan.md` + `request.md`
