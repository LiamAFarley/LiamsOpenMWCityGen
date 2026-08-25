# 2026-08-25 Stage 2 Multiscale Structure Review — Run A

**Scope:** Stage 2 only — `src/procgen/terrain_structure.py` multiscale continuation.
**Spec source:** `.opencode/runs/2026-08-25_tr-run-a-multiscale/plan.md` Stage 2 + `request.md`
**File inspected:** `F:\ProcGenWorkspace\src\procgen\terrain_structure.py` (380 lines, currently untracked in git — new file, no diff vs HEAD)
**Driver / Stage 3:** Not inspected per instruction.
**Reviewer:** review-flash (read-only, no edits)
**Date:** 2026-08-25

## Verdict: PASS — Stage 2 implements the stated contract. No blocking fixes required before Stage 3 wiring. One informational note below.

## Findings (contract-ordered)

1. **Point-source / ribbon code is removed.** No `point`, `ribbon`, `GP`, sparse-feature, or raw height-profile injection remains. The module header `Stages 5-6 ... no sparse feature lines or raw owner-height profiles are injected` (lines 1-7) matches the plan's "Replace the point-source guide mechanism ... with complete-field band continuation." The only `guide_value`/`guide_weight` occurrences are the generic second-order solver parameters (lines 272-273, 283, 324-325, 368-369) which are called with zero weight for a pure harmonic solve (lines 185-186) — AMG-friendly `L` without screening. A grep for `point|ribbon` returns zero hits; `guide_value` appears only twice as solver plumbing.

2. **Owner and generated target bands are H24-H64 (macro) and H8-H24 (meso).** Exactly:
   ```py
   target_macro = target24 - target64  # line 222
   target_meso  = target8  - target24  # line 223
   owner_macro  = H24 - H64            # line 224  (nan_to_num, nan=0)
   owner_meso   = H8  - H24            # line 225
   ```
   Generated targets are built from `_normalized_band(h0, generated_valid, sigma)` with sigma 8/24/64 (lines 219-221) where `generated_valid = generated & isfinite(h0)` (line 217) — i.e. generated-side-only Gaussian pyramids, matching the `sigma 8, 24, 64` request. Owner bands use `features["H24"]-features["H64"]` and `features["H8"]-features["H24"]` (Stage-1 masked pyramids), zeroed outside owner where NaN. Band widths come from `DEFAULTS` macro 8.0 / meso 4.0 cells (lines 30-31), consumed in `_solve_band` via `ctx["dist_seam"] <= width*64.0` (line 168) and reported per-band (lines 191, 245-246).

3. **Each band uses a direct AMG-capable harmonic correction.**

   *Solver is second-order and AMG-friendly.* `solve_screened_structure` (267-380) builds a 4-neighbour 5-point Laplacian (`degree` on diagonal, `-1` on unknown neighbours, lines 294-322) and adds `weights[unknown]` on the diagonal (328). With the call in `_solve_band` (180-187) passing `guide_weight = zeros(...)` (line 186) the system is `(L) C = boundary RHS` — pure Laplace with Dirichlet rows, which `pyamg.ruge_stuben_solver` handles (338-340). Fallback `jacobi_cg` exists (341-342). The synthetic probe (§13 narrow probe) exercised `jacobi_cg` on a 15×15 band and returned `residual_rms` <1e-6 and the expected harmonic interior value 4.0 between fixed 10/8/0, confirming the Laplace assembly.

   *Seam Dirichlet uses owner-band values.* `_seam_band_values` (67-95) iterates `edge_list`, computes `oy,ox = sy-normal, sx-normal` (line 81), and copies `owner_band[oy,ox]` to `values[sy,sx]` when `seam[sy,sx]` and finite (82-94). Conflicting seam claims from overlapping edges are counted (`seam_claim_conflicts`/`spread`, 88-92) — first value wins, which is acceptable because the spec only requires unambiguous *slope anchors* to omit corners.

   *Zero correction at the complete active-band perimeter.* `_complete_band_boundary` (60-64) returns `active & ~seam & ~eroded` where `eroded = binary_erosion(active, structure=4-connect, border_value=0)` (61-62). This is the full 4-neighbour ring of the distance-limited active domain excluding the seam itself. `_solve_band` sets `fixed = seam | outer` (171) and then `fixed_values = target_band copy` with only `seam` (and anchors) overwritten (177-179), so `fixed_values[outer] = target_band[outer]` and `correction_fixed[outer]=0` (287). The synthetic probe verified for a 15×4 rectangle `active 60 / eroded 26 / outer 19` and the boundary includes top/bottom/far edge and lateral edges; a full solve showed `|solved[outer]| = 0` and `|solved[seam]-10| = 0`.

   *Unambiguous first-inland slope anchors, shared corners omitted.* `_first_inland_band_anchors` (98-156) builds `incidence[flat] = {normals}` (105-109), skips any `flat` where `len(incidence)!=1` (119-121) — i.e. shared-corner seam vertices claimed by two normals — and increments `skipped_corner`. For each remaining seam vertex it checks `oy = sy-normal, by = oy-normal` (123-124) and `fy = sy+normal` (125) — the owner-adjacent, one-owner-further, and one-generated-inland vertices — requires `active[fy,fx]` and `~seam[fy,fx]` (137), finite `owner_band[oy]` and `owner_band[by]` (140-143), and sets `candidate = b0 + (b0-bout)` (145) — a one-sided owner normal derivative extrapolated one cell inland. Duplicate `fy` claims are also counted as `skipped_corner` (147-148) when they disagree by >1e-3. The L-shaped seam probe (12×12, two normals sharing flat `(5,5)`) produced `skipped_corner=3` and `anchor_count` reduced by the shared vertex, confirming the omission.

   *Per-band fixed/unknown bookkeeping.* `_solve_band` (159-201) computes `active`, `seam`, `outer`, `fixed|=anchor_mask`, and calls `solve_screened_structure` with target_band as both `h0` and `guide_value` and zero weights, so the only Dirichlet data are seam owner values, one-inland slope anchors, and zero-correction outer. Reports expose `active_vertices`, `outer_boundary_vertices`, `fixed_vertices`, `anchor_count`, `skipped_corner`, and the solver report (189-199).

4. **Generated fine detail is attenuated with a config-driven smootherstep and owner vertices are unchanged.**

   *Smootherstep attenuation.* `_smootherstep` (41-43) is `6t^5-15t^4+10t^3` clipped to [0,1]. `build_multiscale_structural_fields` (227-236) computes `fine_low = _normalized_band(h0, generated_valid, 4.0)` (227), `restore = fine_restore_distance_cells*64` (229), `keep = fine_keep_at_seam + (1-fine_keep)*smootherstep(distance/restore)` (230-232) with `c["fine_keep_at_seam"]` (default 0.2, line 32) and `c["fine_restore_distance_cells"]` (default 6.0, line 33), then `cleaned[generated] = fine_low + keep*(h0-fine_low)` (233-236). The synthetic check gives `keep(0)=0.2, keep(192)=0.6, keep(384)=1.0`, exactly the spec "0.2 at seam to 1.0 at 6 cells".

   *Owner unchanged.* After all three fields are derived, `for field in (cleaned, macro, macro_meso): field[owner_mask]=h0[owner_mask]` (250-251) restores owner verbatim. Synthetic probes (10×10 single-normal and 12×12 L-corner) assert `allclose(macro[owner_mask], h0[owner_mask])` and `cleaned[owner_mask]` — both true. `stage3` is returned as a copy of `h0` (253).

## Exact file/line references

- Module contract / no ribbon: `src/procgen/terrain_structure.py:1-13`
- Defaults (macro/meso/fine): `29-38` — `macro_width_cells 8.0`, `meso_width_cells 4.0`, `fine_keep_at_seam 0.2`, `fine_restore_distance_cells 6.0`
- Smootherstep: `41-43`
- Normalized band (generated-only pyramids): `46-57`
- Complete boundary (active ring sans seam): `60-64`
- Seam owner Dirichlet: `67-95` — `normal` rounding `78`, `oy,ox = sy-normal` `81`, `assigned/conflicts` `74-94`
- First-inland anchors (unambiguous, corners omitted): `98-156` — incidence `105-109`, `len!=1 skip` `119-121`, `oy/by/fy` `122-125`, `active & ~seam` `137`, finite check `140-143`, `b0+(b0-bout)` `145`, duplicate anchor skip `146-149`
- Per-band solve (harmonic, zero outer, direct): `159-201` — `active = generated & dist<=width*64` `168`, `outer = _complete_band_boundary(...)` `170`, `fixed=seam|outer|anchor` `171,176`, `fixed_values` seam/anchor overwrite `177-179`, `solve_screened_structure(..., zeros)` `180-187`
- Multiscale field assembly (H8/H24/H64 bands, fine attenuation, owner restore): `204-264` — `target8/24/64` `219-221`, `target_macro/meso` `222-223`, `owner_macro/meso` `224-225`, `fine_low sigma 4` `227`, `keep smootherstep` `228-232`, `cleaned` `233-236`, `macro_band/meso_band` `238-249`, `owner restore` `250-251`, return dict `252-256`
- Harmonic solver (degree + weights diagonal, AMG RS): `267-380` — `correction_fixed[fixed]=fixed_values-h0` `286-287`, Laplacian assembly `294-322`, `degree+weights` diagonal `328`, `pyamg.ruge_stuben_solver` `339`, `cg` `353-356`, `out[fixed]=fixed_values` `364`

## Required fixes

**None blocking.** Stage 2 meets all five clauses of the stated contract. The following non-blocking observations are for the lead to consider; do not hold the Stage 2 gate.

1. **Informational — seam shared-corner Dirichlet multiplicity.** `_seam_band_values` keeps the first owner value at a seam vertex claimed by two normals and counts the conflict (lines 87-92) rather than omitting the vertex. This is consistent with the spec, which only requires *anchors* to omit shared corners. If strict seam-corner omission is later desired, the same `incidence` guard used for anchors could be applied to seam values — no change required for Run A.

2. **Informational — solver docstring wording.** The module docstring (lines 10-12) still describes the general screened form `(L+W)C = W*(Hguide-H0)`. The actual Run A call uses `W=0` for a pure harmonic (`L C = boundary RHS`). Consider updating the docstring to note the harmonic specialization or that `W` is zero for the macro/meso bands — cosmetic only.

3. **Housekeeping — untracked file.** `src/procgen/terrain_structure.py` is currently `??` (untracked) per `git status` 2026-08-25. Commit it as part of the Stage-2 stage gate; no content change needed.

Do not refactor erosion, hydrology, driver, or rendering logic as part of Stage 2.

## Verification evidence

- **Compile check (narrow, read-only):** `python -m py_compile src/procgen/terrain_structure.py` → `EXIT:0` — verified 2026-08-25.

- **Synthetic probe 1 — basic band (10×10, single normal (0,1), seam col 4, owner cols 0-3):**
  `h0[y,x]=x*10+y`, `owner_mask` left 4 cols, `dist_seam` 0 at seam then `*64` per cell inland, `edge_list` one edge covering col 4. With `macro_width 2 / meso 1` the reports show `macro active 30 outer 12 fixed 30 anchor 10` and `meso active 20 outer 10 fixed 20 anchor 10`, `guide_rows 0 weight_max 0`, `skipped_corner 0`, and `owner unchanged True` for `cleaned/macro/macro_meso`. Exercises H24-H64/H8-H24 subtraction, normalized bands, zero-weight harmonic path, and owner restore.

- **Synthetic probe 2 — L-corner omission (12×12, two normals (0,1) vertical + (1,0) horizontal sharing flat (5,5)):**
  With `macro 8 / meso 4` → `macro unknowns 32 outer 21 fixed 52 anchors 16 skipped_corner 3` and `meso unknowns 16 outer 17 anchors 14 skipped_corner 3`, `owner unchanged True`, `keep 0→0.2 / 192→0.6 / 384→1.0`, `guide_rows 0`. Confirms incidence-based corner omission for anchors and that duplicate `fy` collisions are counted, not solved.

- **Synthetic probe 3 — harmonic correctness (15×15, active 5-8 cols, seam col 5=10, anchor col 6=8, outer col 8=0):**
  `active 60 eroded 26 boundary 34 outer 19` — complete ring verified (top/bottom/far edge). Solve with `jacobi_cg` gives `solved[5]=10, [6]=8, [8]=0, [7]=3.9992` (expected 4.0 for 1-D Laplace between 8 and 0), `|outer| max 0`, `|seam-10| max 0`. Confirms zero-correction outer plus Laplacian assembly.

- **Config grep:** `fine_keep_at_seam` and `fine_restore_distance_cells` appear in `DEFAULTS` and in `keep = ... * _smootherstep(distance/restore)` (230-232) — config-driven.

- **Keyword grep:** `src` contains zero hits for `point`/`ribbon`; `guide_value` only as solver plumbing.

All probes executed via `python -c` with `sys.path.insert(0,'src')` and `linear_solver=jacobi_cg` (AMG path structurally identical, `amg_rs_cg` requires `pyamg`) — no source edits, no filesystem writes outside this review.

## Stop condition

Review stops here. The driver (`tools/terrain/erode_region_v3.py`), configs, and later stages were not inspected or redesigned per instruction.

## Trace

- Request: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\request.md` (39 lines)
- Plan: `F:\ProcGenWorkspace\.opencode\runs\2026-08-25_tr-run-a-multiscale\plan.md` (47 lines, Stage 2 § "Replace point-source ... multiscale owner/target bands ... harmonic macro/meso ... smootherstep ... never alter owner")
- Source: `F:\ProcGenWorkspace\src\procgen\terrain_structure.py` (380 lines)
