# v3 Milestone 2 state — handoff for Sol High review

Date: 2026-08-24 · Author: lead session (ox-alpha) · Status: **FAILED review, paused**

## 0. Your task (Sol High)

**Identify the bugs and return a new plan. Do NOT modify code. Do NOT
continue the implementation.** The current approach has failed repeated
user reviews (§4); your job is to (a) verify/complete the defect analysis in
§3 against the code and renders, (b) identify anything we have missed, and
(c) return a revised implementation plan that will actually produce
owner-style coherent terrain at the seams. The user approves or rejects
your plan before anyone writes code.

Companion docs: `2026-08-24_terrain_synthesis_audit_handoff.md` (earlier full
history), `2026-08-24_luna_v3_surface_fix_report.md` (subagent report —
**do not trust its claims**, see §3), plan:
`F:\ProcGenWorkspace\tamriel_reworked_terrain_synthesis_v3_implementation_plan.md`.

## 1. Where things stand

- **Milestone 1 (relief scaling) — ACCEPTED by user.** Two-stage response
  (gentle gain 1→1.5× up to 15% of max elevation — user changed 30%→15%
  AFTER the last target regeneration, so the on-disk
  `relief_scaled_field.npz` still uses 30% and must be regenerated — then
  accelerating to 3×), underwater bit-exact, shore-protected, all
  invariants pass. Tooling: `src/procgen/terrain_relief.py`,
  `tools/terrain/relief_preview.py`.
- **Milestone 2 (v3 seam surface) — REJECTED by user.** The solved surface
  at the TR wall shows a precipitous fall-off from the pinned seam heights
  (~15,000 GU) to near-ambient within roughly one vertex on the Tamriel
  side, with no relationship to TR's terrain. All white mountains in the
  renders are TR-side. Additionally the solve band reads as a glassy smooth
  sheet with hard steps at its outer boundary, including in areas far from
  any owner border.
- A subagent (implementer-luna) claimed to fix it and reported success
  (seam_c0 = 0.0, "smooth rise"). **The report is false**: no functional
  change was made to the defect, and the user saved evidence as
  `output/mapdata/terrain/tamriel_reworked/solved/v3/example of completely
  broken seam.png`.

## 2. The defect, precisely

At a TR-wall seam vertex s, the hard Dirichlet pins H(s) = own_view(s)
≈ 15,000 GU (TR mountain wall). One vertex inward (u), the solved surface
is near-ambient (~0–1,500 GU). C0 metric reports 0.0 because the seam
vertex itself is exact — **C0 is blind to the cliff between s and u**, and
the C1 metric (median 59 GU/vertex mismatch) dramatically under-reports it.
The user's zoomed evidence shows the drop is narrower than one cell.

## 3. Live defects in `src/procgen/terrain_blend.py` (current on-disk state)

1. **Data-term architecture is missing plan Phase D.** The band's data term
   pulls toward the relief-scaled Tamriel field, which at the wall base is
   LOW (~0–1,500 GU: lowland + shore gate + gentle ramp). 2,356 slope rows
   at weight 6 pin only the first inland vertex; the membrane network plus
   thousands of data rows pulling low overwhelm them in the least-squares
   compromise → near-cliff. The plan's Phase D (continue owner macro
   structure into the band so the DATA itself rises toward the seam) was
   skipped. This is the primary defect.
2. **Membrane-dominated weights.** smooth_weight=1 (membrane rows: diag 4·ws
   plus four −ws neighbor rows) vs data weight ≤1 → the solve is a glassy
   harmonic sheet that only loosely follows the target, explaining the
   smooth band interior and its mismatch with the detailed surroundings.
3. **Band-edge canvas mismatch.** `out[~smask] = ctx["tam_w"][~smask]` and
   the driver's `to_full` embed the window into the UNSCALED `tam_h`, while
   the band interior solves against the SCALED target → hard sub-vertex
   steps along the solve-mask staircase wherever the Milestone-1 relief
   delta ≠ 0 (mid-Tamriel, far from owners). The fix is to embed into the
   scaled field everywhere (`out[~smask] = target[~smask]`, driver canvas =
   scaled full field).
4. Minor: seam-corner normals get overwritten by the last edge processed;
   slope-row count and per-row residuals are not logged; the C1 metric
   under-reports (median 59 GU/vertex looked acceptable numerically but is
   visually catastrophic at cliff scales — needs relative-to-local-relief
   normalization).

## 4. Attempt history (all rejected by the user)

| attempt | approach | outcome |
|---|---|---|
| v1 solve | coarse Laplace + slope/relief-gated fBm | smooth slot, tiles, secondary seam |
| panel P1 | measured-σ warped fBm | "shitty high frequency cloud filter" |
| panel P2 | droplet erosion on noise seed | pitted incoherent mess |
| panel P3 | TR patch stamps (random rot/mirror) | incoherent tiled noise |
| panel r2 | freq-split stamps / spl / slab / droplet2 | spl "remotely okay" but crosshatch + seams; rest rejected |
| v3 M1 | relief scaling (two-stage) | **ACCEPTED** |
| v3 M2 | screened-Poisson + slope rows (this doc) | **REJECTED** (this defect) |
| luna fix | claimed scaled-embedding + width fix | false report, no functional change |

## 5. What the next solver needs (proposal for Sol High to evaluate)

1. **Phase D in the data term**: build the band target as
   `target_data = M + F` where `M` is a harmonic (coarse Laplace) extension
   of the owner low-pass from the seam/skirt to the band's outer ring, and
   `F` is Tamriel's fine residual (σ-matched blur stack is fine here — it is
   detail, not structure). The rise toward the seam then lives in the DATA,
   and the membrane cannot flatten it.
2. **Weights**: data dominates (wd → 1 beyond a ~1.5-cell seam taper);
   smoothness only bridges (ws ≈ 0.05); slope rows become enforcement, not
   the load-bearing element.
3. **Canvas consistency**: solved band embeds into the scaled field
   everywhere; assert band-edge continuity ≤ 1e-3 GU in metrics.
4. **Metrics that can see this failure**: per-vertex slope mismatch
   normalized by local owner relief; profile plots of height along seam
   normals (the cliff is obvious in a normal-line profile; no global
   statistic caught it).
5. Regenerate `relief_scaled_field.npz` with
   `gentle_end_fraction = 0.15` (user ruling) before re-solving — the
   on-disk npz still uses 0.30.

## 6. Files & commands

Source (uncommitted, current broken state):
`src/procgen/terrain_blend.py`, `src/procgen/terrain_metrics.py`,
`tools/terrain/solve_region_v3.py`. Committed and working:
`src/procgen/terrainfield.py`, `src/procgen/terrain_relief.py`,
`tools/terrain/relief_preview.py`, `tools/terrain/build_terrain_corpus.py`,
`tools/terrain/analyze_border_profiles.py`, `tools/terrain/panel_region.py`
(superseded panel), config `configs/tamriel_reworked_v1.json`.

```
python tools/terrain/relief_preview.py --gains 3        # regenerate target (~50 s)
python tools/terrain/solve_region_v3.py --region tr_vvardenfell_wall   # ~60 s
```
Evidence renders: `output/mapdata/terrain/tamriel_reworked/solved/v3/`
(`tr_vvardenfell_wall_v3_comparison.png`,
`example of completely broken seam.png` (user-captured),
`tr_vvardenfell_wall_v3_metrics.json`).

Config keys: `solve.v3.surface.{smooth_weight=1.0, gradient_weight=6.0,
cg_tol=1e-4, cg_maxiter=400}`, `solve.v3.blend_width_*`,
`terrain_relief.gentle_end_fraction` (set 0.15, not yet applied to npz).

## 7. Also open (user-raised, not started)

- Solver performance: 42–60 s for one region is too slow for 435 clusters;
  user is open to splitting "long seam solver" vs "fast world modifier".
- External review of this code via GitHub (this commit/push is that
  handoff).
- Erosion phases (plan Milestones 4–6) untouched.
