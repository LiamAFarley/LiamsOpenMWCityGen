# 2026-08-25 P3 Plateau Candidate Review — TR Primitive Plateau/Canyon (Re-review)

**Scope:** Re-review of P3 fixes only per `request.md` + `plan.md` §P3. Inspected `src/procgen/terrain_primitives.py:629-713` (`synthesize_plateau_candidates:629-713` with `fine_keep_core` hoist and `scarp_authority_gain`) and `configs/tamriel_reworked_v1.json:10,33-40` (`fine_keep_core`, `scarp.authority_gain`). Verified the two prior blocking fixes: empty-set validity and scarp authority capped increase, plus global/local top-fit contract, generated-only writes, fine residual attenuation, weighted complete candidates, owner immutability, and finite/bounded outputs. Compile + empty and non-empty synthetic candidate probes executed (no I/O, no real `tr_vvardenfell_wall` crop, no P4/P5 inspected or edited).

**Verdict: PASS — both P3 blocking fixes verified. Proceed to P4.**

Prior verdict (2026-08-25 initial): CONDITIONAL PASS — RF1 (fine_keep_core unbound on empty) + RF2 (inert scarp max) blocking.

---

## 1. P3 Gate Summary (re-reviewed)

| Check (prompt) | Result | Evidence |
|---|---|---|
| **Fitted top-surface evaluation — corrected global/local contract** | **PASS** | `evaluate_top_fit:188-190` subtracts `row_offset`/`col_offset`; `analyze_plateaus:314-315` stores `clipped[0/2]`; `synthesize:641-642,662` evaluates with full-field `np.indices(shape)` globals. Flat 8000 probe: `(5,5)`→8000, offset fit `center 8/4 scale 8 offset 8/8` at global `16,12`→8000 — no shift error. Tilted P1 path preserved. |
| **Generated-only writes** | **PASS** | `support_weight` clipped `*smask` `658-661`; `weight = support_weight * generated` `684` where `generated = smask & ~owner_mask` `681-683`; `total_weight/weighted_height` accrue `weight` only `685-687`; `plateau` copy `697`, `has_candidate` blend `698-699`, owner lock `700-702`. Probe: `owner support sum 0.0`, `candidate_weight[owner] 0.0`, `max|plateau[owner]-h0[owner]| 0.0` with owner 8000 vs gen 4000±200. Core synthesis never touches owner. |
| **Explicit support/scarp transition — config-driven capped increase** | **PASS — fix verified** | `scarp_authority_gain = config.get("scarp",{}).get("authority_gain",0.35)` `648-650` (from `tamriel_reworked_v1.json:37` `0.35`); `normalized = clip(support_weight/W,0,1)` `666-670`; `transition = clip(normalized + gain * scarp_confidence * (1-normalized),0,1)` `674-679`. Probe: `support 0.88 scarp 0.75` → `trans 0.875→0.908` `cand 7664.3→7752.4` Δ+88; `0.75/0.50` → `0.750→0.794` Δ+131; `0.625/0.25` → `0.625→0.658` Δ+128; `0.50/0.00` → `0.500→0.500` no boost where scarp 0. Far outside `support 0 scarp 0` → `plateau==h0`; artificial `support 0 scarp 1` → `trans 0.35` but `weight 0` → `plateau==h0` (4185.8) — scarp raises authority within footprint only, never expands `candidate_weight` footprint. Inert `max(normalized, scarp*normalized)` removed. |
| **Primitive-core fine residual attenuation from config** | **PASS — fix verified** | `fine_keep_core = float(config.get("fine_keep_core",0.1))` **before loop** `647` (was inside loop); `core = weight >= 0.65` `663`; `fine_keep = where(core,fk,1)` `664`; `background = h0 - (1-fk)*fine` `665` with `fine = h0 - _normalized_gaussian(h0,valid,4)` `638-640`. Config `tamriel_reworked_v1.json:10` `fine_keep_core:0.1`. Probe: core `16,17 sup 0.88`: `fk0 7764.1` vs `fk0.1 7752.4` vs `fk1.0 7647.1` — rim attenuates ~105 Gu; outside core identical. Report now carries `fine_keep_core` even on empty run. |
| **Weighted complete candidate surfaces** | **PASS** | Per-primitive `candidate = trans*top + (1-trans)*bg` `680` complete; `weighted_height += weight*candidate`, `total_weight += weight` `685-687`; `plateau[has_candidate]=weighted/test` `698-699`. Two-prim overlap probe: single `8000` weight 1.0 at `16,16 → 8000`; second `8500` weight `0.5` → `7684.2` with `candidate_weight_max 1.5` — correct weighted average of per-prim blended surfaces. `candidate_support` union of `weight>0` `687`. |
| **Owner immutability** | **PASS** | No `h0` mutation; final lock `plateau[owner]=h0[owner]`. Probe `max|plateau-h0| on owner 0.0` with 8000/4000 split; also holds for empty and two-prim runs. |
| **Finite / bounded outputs** | **PASS** | Finite `h0` → `plateau` finite `True`, bounded `~3920–8000`; `candidate_weight` finite `max 1.0` single / `1.5` double; `fine_residual` finite. Empty set returns `h0` copy finite. NaN-propagation observation remains (if `h0` NaN, plateau NaN) but normal pipeline guarantees finite `h0` (valid mask); not blocking. |
| **Empty primitive set valid** | **PASS — fix verified** | `fine_keep_core` hoisted `647` before `for primitive in primitives:` `654`; loop skip path appends `{support_vertices:0}` `655-657`; final `plateau=h0.copy()` `697` and `has_candidate=total_weight>1e-6` `698` yields identity field; report `fine_keep_core` from config `711` present. Probe: `synthesize(..., [], ...) → max|plateau-h0| 0.0`, `finite True`, `candidate_vertices 0`, `candidate_weight_max 0.0`, `fine_keep_core 0.1` (and `0.05` when config `0.05`); `[weight None]` also identity. Previously crashed `UnboundLocalError` at `703`. |

Compile: `py_compile terrain_primitives.py` → `compile OK` (re-review).

Synthetic probes (re-review): 32x32 owner 8000 / generated 4000+200·sin, seam col 16, linear support `1-(c-16)/8` decaying 1.0→0 over 8 verts, scarp ribbon `1-(c-16)/4` over 4 verts, authority_gain 0.35. Empty (0 prims), weight-None, single-prim gain 0 vs 0.35, far-outside, artificial outside-support scarp, two-prim overlapping, and fine_keep sweep — all generated-only, owner-immut, finite, and scarp-capped.

---

## 2. Fixes Verified (closing prior Required Fixes)

### RF1 — `fine_keep_core` unbound on empty — CLOSED

- **Prior:** `fine_keep_core` assigned inside `for primitive in primitives:` `661-664`, report `703` accessed after loop → `UnboundLocalError` when `primitives==[]` or all `target_weight is None`.
- **Now:** `terrain_primitives.py:647-648` initializes before loop:
  ```python
  fine_keep_core = float(config.get("fine_keep_core", 0.1))
  scarp_authority_gain = float(config.get("scarp", {}).get("authority_gain", 0.35))
  ```
  Loop `654-696` reuses that value; empty report `708-712` reads the same `fine_keep_core` without loop entry. Probe: empty `synthesize(..., [], foot_zero, {fine_keep_core:0.1})` → identity, `candidate_vertices 0`, `fine_keep_core 0.1`; custom `0.05` → `0.05`; single weight-None → identity — no exception.

### RF2 — Scarp transition degenerate — CLOSED

- **Prior:** `transition = max(normalized, scarp_confidence * normalized)` always `== normalized` (product ≤ operand), scarp ribbon had no candidate effect.
- **Now:** `terrain_primitives.py:674-679` capped increase:
  ```python
  transition = np.clip(
      normalized + scarp_authority_gain * scarp_confidence * (1.0 - normalized),
      0.0, 1.0,
  )
  ```
  Config-driven `scarp_authority_gain:0.35` `tamriel_reworked_v1.json:37` with neutral default `0.35`. Candidate boost `gain*scarp*(1-norm)` → 0 at `norm 0 scarp 0` or `norm 1`, max `+0.35` at `norm 0 scarp 1` (but weight gate prevents footprint expansion: `weight = support_weight * generated`, probe `support 0 scarp 1 → weight 0.0 → plateau==h0`). Verify: `0.88/0.75` +0.033 → +88 Gu, `0.75/0.50` +0.044 → +131 Gu, `0.625/0.25` +0.033 → +128 Gu; far `0/0` unchanged. Comment `671-673` retained: *“raise local top authority but never create support outside footprint”* — now true via weight gate, not via transition clamp. Reduces to `normalized` when `gain 0` (probe `pl_g0`).

Both fixes are narrow, config-driven, and preserve generated-only and bounded guarantees.

---

## 3. Observations (non-blocking — triage for later)

### O1 — Scarp footprint edge expansion semantics

With capped formula, `support 0 scarp 1 → trans 0.35` gives `candidate` with top authority, but `weight 0` keeps `plateau==h0`. Candidate per-primitive (`target_height`) will show scarp-influenced height even where `weight 0`; final `plateau` does not. If P5 reconciliation also reads `target_height` rather than weighted `plateau`, ensure it gates by `candidate_weight` likewise. Current P3 keeps footprint edge controlled by `support_weight` alone — correct per plan.

### O2 — Scarp width vs support decay

P2 ribbon `fallback_width_verts:8` linear `1-step/width` overlaps support decay `8` verts linearly; with `gain 0.35`, wall steepening is modest (`+0.35*(1-norm)`). If wall needs sharper scarp, tune `authority_gain` (or `scarp.weight`) without code change — config-driven as required.

### O3 — Dual-use `support_core_threshold`

Same `0.65` for geodesic `core_cost` and probability `core` mask; flagged in P2. Leave until tuning.

---

## 4. Exact References

- Module: `src/procgen/terrain_primitives.py:1-713` (P3 `629-713`; fix lines `647-650` hoist + gain, `674-680` transition, `700-712` owner lock+report)
- Config: `configs/tamriel_reworked_v1.json:6-40` (`semantic_downsample:4` `6`, `fine_keep_core:0.1` `10`, `plateau.support_core_threshold:0.65` `25`, `scarp.authority_gain:0.35` `37`, `scarp.fallback_width_verts:8` `36`)
- Plan/request: `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md:28-33`, `request.md`
- Probes: `F:\ProcGenWorkspace\.opencode\tmp\p3_verify.py` (compile + empty `plateau==h0` / custom fk / weight-None / single gain-0 vs 0.35 / far-outside / artificial outside-support / two-prim / fine_keep sweep / global-local top fit) — no real crop, no P4-P5 inspected.

---

## 5. Recommendation

**P3 re-review PASS.** Both blocking fixes are present and behave correctly in empty and non-empty synthetic probes: empty set now returns finite identity field with config `fine_keep_core` reported, and scarp authority now provides a real config-driven capped boost `normalized + gain*scarp*(1-normalized)` that raises candidate height inside the support footprint (verified +88 to +131 Gu on mid-support) without expanding `candidate_weight` generated footprint (weight gate holds). All other P3 gates remain PASS (global/local top contract, generated-only, fine attenuation, weighted complete candidates, owner immutability, finite/bounded). No remaining P3 blocker. Proceed to P4 (canyon candidate). Do not run the real bottom-right `tr_vvardenfell_wall` crop until P4 review gate per `plan.md`.

*Reviewer: review-flash (read-only, P3-only re-review) — 2026-08-25 — files: `src/procgen/terrain_primitives.py:629-713`, `configs/tamriel_reworked_v1.json:6-40`, `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md`, `request.md` — probes: compile + empty/non-empty scarp/fine/weight synthetic probes (§1) — no crop executed, no P4-P5 inspected.*
