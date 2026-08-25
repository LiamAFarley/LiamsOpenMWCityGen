# 2026-08-25 P4 Canyon Candidate Review — TR Primitive Plateau/Canyon (Re-review: fixes only)

**Scope:** Re-review of P4 fixes only per `request.md` + `plan.md` §P4. Inspected `src/procgen/terrain_primitives.py:818-996` (`continue_canyons:818-996`; `_semantic_canyon_path:716-815` unchanged) and `configs/tamriel_reworked_v1.json:42-56` (`canyon.min_confidence:0.55`). Verified the two requested blocking fixes: `canyon.min_confidence` enforcement on selected valley component scores and owner-side H24 bounds guard before thalweg anchoring. All other P4 gates re-checked as regression (generated-only Dijkstra, costs, bounded length, monotonic thalweg, cross-section, ordered assignment, owner immutability, no reconciliation solve). Compile + narrow synthetic canyon probe with `min_confidence` gate sweep and edge-seam bounds guard executed (no I/O, no real `tr_vvardenfell_wall` crop, no P5 inspected or edited).

**Verdict: PASS — both P4 blocking fixes verified. Proceed to P5.**

Prior verdict (2026-08-25 initial): PASS with observations O3 (H24 bounds) + O4 (`min_confidence` unused) — both now closed.

---

## 1. P4 Gate Summary (re-reviewed)

| Check (prompt) | Result | Evidence |
|---|---|---|
| **Connected owner valley-component selection — now `min_confidence` gated** | **PASS — fix verified** | `valley_mask & owner` `835` → `ndimage.label 3×3` `837-839`; `seed_search_radius_verts:8.0` `849` around `oy=row-normal[0], ox=col-normal[1]` `855` scans `valley_mask[r0:r1,c0:c1]` `858`; best `valley_score` per `component_id` `862-872`. **New gate `868-869`**: `if candidate_score < float(canyon_cfg.get("min_confidence",0.55)): continue` skips weak valleys before dedup. Config `canyon.min_confidence:0.55` `configs/tamriel_reworked_v1.json:43` wired, default `0.55` preserves behavior. Probe gate sweep (valley at col29, 1 component 72 verts): score `0.90→ 1 comp 1 seed 16 line verts depth 4180.7`; `0.40→ 0 comp 0 verts depth 0.0` (filtered); boundary `0.55→ 1 comp` (inclusive) vs `0.54→ 0 comp` filtered — correctly enforces threshold. Selection remains per-component highest-score + `[:max_paths_per_plateau:3]` `873-874`. |
| **Owner H24 seam anchoring — now bounds-guarded** | **PASS — fix verified** | `owner_row=seed[0]-normal[0], owner_col=seed[1]-normal[1]` `917-918`; **new guard `919-923`**: `if 0≤owner_row<H and 0≤owner_col<W: owner_h = H24[owner_row,owner_col] else: owner_h = nan`; `z0 = owner_h if finite else path_heights[0]` `923` (fallback to generated H64 head). Uses `H24` not `H64`; previously unguarded `H24[owner_row,owner_col]` would IndexError on edge seams. Probe edge seam `(0,0)` normal `(-1,0)` → owner `(-1,0)` out of bounds → `owner_h nan → z0 = path_heights[0]` → no exception, `canyon_line 0 verts`, `owner immut 0.0`, `canyon_depth[owner] 0.0` — guard holds. Normal downhill probe `owner ~8000 → zend 3812` thalweg `8000→3812` interior `4556→4457` depressed by `top-nearest` depth unchanged. |
| **Generated-only A*/Dijkstra continuation** | **PASS** | `_semantic_canyon_path:728 sem_generated=_block_max(generated,4)`; start/neighbor gate `sem_generated` `737-738,764-765`; `generated=smask & ~owner` `831`; line filtered `generated[row,col]` `910`/`933`; `depth[~work_generated]=0` `965`; `canyon_field[owner]=h0[owner]` `991`. Re-probe: both gated (`0.4`) and ungated (`0.9`) runs show `canyon_line in owner 0`, `canyon_depth[owner].max 0.0`. Direct `_semantic_canyon_path` `visited 128 max_cost 256` generated-only. |
| **Low / uphill / direction / plateau costs** | **PASS** | `766-779` `direction_penalty*(1-alignment)` `target_low_penalty*clip((next_h-low_ref)/scale)` `uphill_penalty*max(0,Δh)/scale` `plateau_penalty*max(0,1-sem_support)` → `step_cost=1+Σ`; via `config.get` `767,770,773,776` to JSON `45-48`. No change in fix; still config-driven. |
| **Bounded path length** | **PASS** | `max_cost = max_cells*64/factor * path_cost_multiplier` `745-748` → `8*64/4*2=256`; heap `cost>max_cost`/`new_cost≤max_cost` `760,781»; reconstruction `len(path) ≤ max_cells*64/factor+2` `800`. Probe `max_cost 256 sem_path 16 full 16` within bound; filtered case `0` vertices trivially bounded. |
| **Monotonic thalweg profile** | **PASS** | `thalweg = linspace(z0,zend,len)` `925` → `minimum/maximum.accumulate` `926-929` monotonic; guarded `z0` fallback still yields finite `z0`. Probe downhill `z0 8000 → 3812` decreasing; edge fallback `z0=path_heights[0]` (generated H64) yields degenerate but monotonic. |
| **Configured cross-section / depth** | **PASS** | EDT `933-940`, `top=evaluate_top_fit` `945`, `half_width=width_cells*64=256` `946-948`, `bottom_half=64` `949-952`, `q` smootherstep `953-962` `wall_exponent:1.5`, `depth=max(top-nearest,0)*q` `964`, `weight=q*1.2` — all `canyon_cfg.get`. No change; probe `depth_max 4180.7` vs filtered `0.0`. |
| **Ordered path-height assignment** | **PASS** | `full_path` preserves Dijkstra `start→goal` order `903-906` (`path.reverse()` `807`); `path_heights` in same order `908-912`; `zip(valid_path, thalweg_values)` `931-936`. Unchanged. |
| **Owner immutability** | **PASS** | `canyon_field=plateau_field.copy() - canyon_depth` `989-990` with `canyon_depth` zero outside generated `965`; lock `canyon_field[owner]=h0[owner]` `991`. Probes all show `max|canyon[owner]-h0[owner]| 0.0` including edge guard and `0.4` filtered cases; `h0` not mutated. |
| **No reconciliation solve yet** | **PASS** | `continue_canyons:818-996` contains no `amg`/`poisson`/`reconciliation`/`solve`; only `distance_transform_edt`, `_block_*`, `evaluate_top_fit`, `heapq`. Still no P5 solve. |

Compile: `py_compile src/procgen/terrain_primitives.py` → `compile OK` (re-review).

Synthetic probes (re-review): 64×64 owner west `col<32` plateau `8000` downslope generated east; seam `3 verts col32 normal (0,1)` flat fit `8000`; valley `col28-31` 72 verts with score sweep `0.90/0.55 pass →16 verts 4180 Gu`, `0.40/0.54 filtered →0 verts` (min_conf gate); edge seam `(0,0)` normal `(-1,0)` out-of-bounds owner `(-1,0)` → guarded `nan→fallback`, no crash, `0 verts`, `owner immut 0.0`, `canyon_depth[owner] 0.0`. Direct `_semantic_canyon_path` `visited 128` generated-only. All generated-only, bounded, owner-immut.

---

## 2. Fixes Verified (closing prior Required Fixes)

### RF1 — `canyon.min_confidence` not enforced — CLOSED

- **Prior:** `seed_by_component` accepted any `valley_mask` component with `score > -inf` inside radius; `canyon.min_confidence:0.55` existed in JSON `43` but was unread in `continue_canyons`, so weak valleys could generate canyons.
- **Now:** `src/procgen/terrain_primitives.py:868-872`:
  ```python
  candidate_score = float(score[vy, vx])
  if candidate_score < float(canyon_cfg.get("min_confidence", 0.55)):
      continue
  prior = seed_by_component.get(component_id)
  if prior is None or candidate_score > prior[0]:
      seed_by_component[component_id] = (candidate_score, seed, normal)
  ```
  Gate applied before dedup, per-component highest score retained only if `≥ min_confidence` (inclusive). Wired to `configs/tamriel_reworked_v1.json:43` `min_confidence:0.55` with fallback `0.55`. Verification: sweep `min_conf 0.55` — `score 0.90` → 1 comp, `0.55` → 1 comp (inclusive), `0.54`/`0.40` → 0 comp (filtered). Empty and filtered cases return finite zero-depth field with `owner immut 0.0` and no paths.

### RF2 — H24 owner anchor bounds — CLOSED

- **Prior:** `owner_h = features["H24"][owner_row, owner_col]` `917` assumed in-bounds; edge seam at domain border → `IndexError`.
- **Now:** `src/procgen/terrain_primitives.py:917-923`:
  ```python
  owner_row = seed[0] - normal[0]
  owner_col = seed[1] - normal[1]
  if 0 <= owner_row < shape[0] and 0 <= owner_col < shape[1]:
      owner_h = float(features["H24"][owner_row, owner_col])
  else:
      owner_h = float("nan")
  z0 = owner_h if np.isfinite(owner_h) else path_heights[0]
  ```
  Out-of-bounds falls back to `path_heights[0]` (generated H64 head) via `nan` check — no exception, thalweg remains finite monotonic. Verification: edge seam `seed (0,0)` normal `(-1,0)` → `owner (-1,0)` → `nan→ path_heights[0]` → `continue_canyons` completes `canyon_line 0 verts depth 0.0` with `compile OK` and `owner immut 0.0`. Normal interior seams still use true `H24` (probe `8000→3812`).

Both fixes are narrow, config-driven where applicable (`min_confidence` from JSON), preserve generated-only and bounded guarantees, and do not add reconciliation logic.

---

## 3. Observations (non-blocking — triage for later)

### O1 — Sparse full-resolution centerline (carry)

- `full_path` maps each semantic cell to single full-res vertex `(row*factor+wr0, col*factor+wc0)` `903-906` → 4-vert punctuated line; EDT distance shows 4-vert scallop. If P5 needs smooth walls, densify via Bresenham between semantic centers. Not blocking for P4 diagnostics.

### O2 — Goal ranking diagonal bias (carry)

- Ranking `distance -0.25*low_bias` `796` favors corners over due seam-normal; probe endpoint `[0,15]` vs `[8,15]`. Directional step cost limits drift; acceptable for P4.

### O3 — Work-bbox recomputed per primitive (carry)

- `work_bbox`/`work_generated` recomputed inside per-primitive loop `883-887` (invariant). Harmless.

---

## 4. Exact References

- Module: `src/procgen/terrain_primitives.py:716-1001` (fixed `canyon.min_confidence` gate `868-869`, H24 guard `917-923`; unchanged Dijkstra `716-815`, seeding `848-880`, thalweg `925-929`, cross-section `946-965`, subtraction `989-991`)
- Config: `configs/tamriel_reworked_v1.json:42-56` (`canyon.min_confidence:0.55` `43`, `max_continuation_cells:8.0` `44`, `direction_penalty:1.0` `45`, `uphill_penalty:4.0` `46`, `target_low_penalty:1.5` `47`, `plateau_penalty:1.0` `48`, `path_cost_multiplier:2.0` `49`, `seed_search_radius_verts:8.0` `50`, `max_paths_per_plateau:3` `51`, `width_cells:4.0` `52`, `bottom_half_width_cells:1.0` `53`, `wall_exponent:1.5` `54`, `weight:1.2` `55`)
- Plan/request: `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md:37-42`, `request.md`
- Probes: `F:\ProcGenWorkspace\.opencode\tmp\p4_rereview_probe.py` (compile + `min_conf` sweep `0.90/0.55/0.54/0.40` → `16/16/0/0` verts, edge seam `(-1,0)` guard no crash) — no real crop, no P5 inspected.

---

## 5. Recommendation

**P4 re-review PASS.** Both blocking fixes are present and behave correctly: `canyon.min_confidence` (default `0.55`, JSON `0.55`) now gates `candidate_score` before component dedup (verified `0.90/0.55` pass → 16 verts vs `0.54/0.40` filtered → 0 verts), and owner H24 thalweg anchor is now bounds-guarded to `nan→path_heights[0]` fallback (verified edge seam `(-1,0)` no `IndexError`, normal seams still anchor to true `H24`). All other P4 gates remain PASS (generated-only semantic Dijkstra with four costs, bounded continuation `256`/`130`, monotonic `linspace`/`accumulate`, config-driven cross-section, ordered `line_height`, owner immutability, no reconciliation). No remaining P4 blocker. Proceed to P5 (reconciliation). Do not run the real bottom-right `tr_vvardenfell_wall` crop until P5 review gate per `plan.md`.

*Reviewer: review-flash (read-only, P4-only re-review) — 2026-08-25 — files: `src/procgen/terrain_primitives.py:818-996`, `configs/tamriel_reworked_v1.json:42-56`, `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md`, `request.md` — probes: compile + min_conf sweep + edge guard synthetic probes (§1) — no crop executed, no P5 inspected.*
