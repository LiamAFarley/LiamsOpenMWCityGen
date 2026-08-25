# 2026-08-25 P2 Footprint + Scarp Review — TR Primitive Plateau/Canyon (Re-review)

**Scope:** Re-review of P2 fixes only per `request.md` + `plan.md` §P2. Inspected `src/procgen/terrain_primitives.py` lines 381–626 (`_seam_normals:381-391`, `_semantic_support:394-470` with `cost_multiplier`, `continue_plateau_footprints:473-626` with direct seam seeding) and `configs/tamriel_reworked_v1.json:10-30` (`geodesic_cost_multiplier`). Verifies the two prior blocking fixes: actual seam vs owner-contact coordinates (no second normal offset) and config-driven semantic Dijkstra cost conversion using `factor` and `geodesic_cost_multiplier`. Compile + narrow+wide synthetic generated-only probes executed (no I/O, no real `tr_vvardenfell_wall` crop). No P3–P5 inspected or edited.

**Verdict: PASS — both P2 blocking fixes verified. Proceed to P3.**

Prior verdict (2026-08-25 initial): CONDITIONAL PASS — RF1 (seam double-offset) + RF2 (bounded continuation units) blocking.

---

## 1. P2 Gate Summary (re-reviewed)

| Check (prompt) | Result | Evidence |
|---|---|---|
| **Actual seam vs owner-contact coordinates** | **PASS — fix verified** | P1 `analyze_plateaus:329-333` stores `owner_contact_vertices = owner_seam` (e.g. col 511) and `seam_vertices = owner+normal` (actual seam e.g. col 512). P2 `continue_plateau_footprints:516-527` now seeds **directly** from actual seam: `gy,gx = row,col` (no `+normal`), filtered `generated_work[ly,lx]` `525`. Probe: delta seam−owner = 1 (`16-15` on 32-wide, `512-511` on 1024-wide) — one normal step as required. `seam_vertices` col `{512}` vs owner col `{511}` distinct; `diagnostics["owner_contact_vertices"]` retained. |
| **Config-driven semantic Dijkstra support** | **PASS — fix verified** | Factor `config.get("semantic_downsample",4)` `485/416-417`; `max_continuation_cells` `487`; `direction_seam_weight` `533`; `geodesic_lateral_eta` `546`; `geodesic_cost_multiplier` `547` wired to `_semantic_support:404,432-435`; `support_core/edge_threshold` `550-561/596`; `fallback_width_verts` `572`. No hard-coded land coordinate; `bbox` via `bbox_margin_cells` `486-489`. All cost knobs config-driven. |
| **Generated-only domain** | **PASS** | `generated = smask & ~owner_mask` `495`; `valid` `497`; `sem_generated = _block_max(generated,factor)` `418`; seed gate `sem_generated` `427` + expand gate `446`; `support_work = valid & finite(distance)` `551`; `support_full *= generated` `559`; `core_work` from that; scarp gate `generated[sy,sx]` `590`. Probes: `support in owner sum 0.0` on 32-wide and 1024-wide (§4). No owner write. |
| **Directional / height / slope compatibility** | **PASS** | `_semantic_support:432-461`: `height_scale` 75th-pct `412`, `slope_scale` `414`; `sem_h64/sem_slope/sem_top` Means `419-421`; `compatibility = 1 + |H64-top|/hscale + slope/sscale + (1-eta)*lateral` `451-456`; `directional = eta+(1-eta)*alignment²` `450`; `step_cost = compat/directional` `457`; `direction` = seam-normal × `direction_seam_weight` `531-536`. Probe: lateral/height penalties correctly limit extent (6.36 cells at 8-cell budget vs 1.23 at 2-cell). |
| **Bounded continuation** | **PASS — fix verified** | `_semantic_support:432-435` now `max_cost = max_cells*64/factor * cost_multiplier` (≥1.0). Config `plateau.geodesic_cost_multiplier:2.0` `23`, `max_continuation_cells:8.0` `19`, `factor:4` → `8*64/4*2=256` (was `8/0.4=20`). Probe 1024-wide: 8-cell budget → max_cost 256, visited 1720 sem cells, support col 512..919 dist 407 verts (6.36 cells, compatible-cost headroom); 2-cell budget → max_cost 64, support 512..591 dist 79 verts (1.23 cells). Small 32-wide correctly truncated by domain not cost. `margin += ceil(max_cells*64)` `488` consistent. |
| **Local signed distance** | **PASS** | `core = support_full >= core_thresh` `561`; EDT `distance_transform_edt(core_work)-distance_transform_edt(~core_work)` `563-566` inside `work_bbox` only; gating `core|support>0` `567-569`. Probe: `signed 3.0` at seam+2 → `11.0` interior, positive inside core, increases inward — correct local EDT. |
| **Explicit scarp confidence / normals** | **PASS** | Per-primitive ribbon `local_scarp` `573-595`: linear falloff `1-step/width` `591` with `fallback_width_verts:8.0` `572`, gate `owner_scarp` within 8 verts `581-584` + `generated` `590`; normals `local_sy/sx = normal` `594-595`; global max-merge `594-602`. Probe: `scarp max 1.0` at seam, falloff, normal `(0,1)` correct. Seam-anchored ribbon only (expected for P2 diagnostics; full perimeter via `signed_distance` in P5). |
| **No height synthesis yet** | **PASS** | Only `primitive.support_mask` `596` and `primitive.target_weight` `597` assigned; `target_height` remains `None` (probe `target_height None True`). No `target_height` write in P2. |

Compile: `py_compile terrain_primitives.py` → `compile OK` (re-review).
Synthetic probes (§4): 32-wide (domain-limited), 1024-wide 8-cell (multi-cell corridor ≈6.4 cells), 1024-wide 2-cell (truncated ≈1.2 cells) — all owner sum 0.0, correct seam anchoring, bounded by cost.

---

## 2. Fixes Verified (closing prior Required Fixes)

### RF1 — Seam double-offset — CLOSED

- **Prior:** `terrain_primitives.py:510-523` seeded `owner+2·normal` (actual seam `owner+normal` plus second `+normal`), leaving a 1-vert gap masked only by 4× block-fill (`16//4 == 17//4`).
- **Now:** `terrain_primitives.py:516-527` seeds directly from actual seam:
  ```python
  for row,col in primitive.seam_vertices:  # already actual seam (owner+normal, e.g. 512)
      normal = normals.get(row*W+col)
      gy,gx = row, col                     # no +normal
      if generated_work[ly,lx]: seeds.append((ly,lx))
  ```
  Direction averaging `531-532` still uses `normals.get` on the same seam vertices (correct inward normal). Verification: synthetic delta 1 on both window sizes; `support at seam 32/64` present without relying on block side-effect; `owner_contact` vs `seam_vertices` split preserved.

### RF2 — Bounded continuation units — CLOSED

- **Prior:** `max_cost = max_cells/eta` (20 at 8 cells/eta0.4) vs `step_cost` ≈1–3 per semantic cell → 8 cells (512 verts =128 sem steps) needed ~256 cost, stalled at ~10 sem steps (~0.6 cells).
- **Now:** `terrain_primitives.py:394-405` accepts `cost_multiplier` param; `432-435` converts units:
  ```python
  max_cost = max(float(max_cells)*64.0/max(float(factor),1.0) * float(cost_multiplier), 1.0)
  ```
  Caller `544-548` passes `config.get("geodesic_cost_multiplier",2.0)` from `tamriel_reworked_v1.json:23` (`2.0`). Resulting `max_cost` = `8*64/4*2=256`, `2*64/4*2=64`, etc. Probe confirms linear scaling: 8-cell → 407-vert (≈6.36-cell) extent, 2-cell → 79-vert (≈1.23-cell) under same compatible terrain (headroom for height/slope/lateral costs). `work_bbox` margin `488` already in verts and now consistent.

Both fixes are narrow, config-driven, and generated-only; no P3 candidate logic added.

---

## 3. Observations (non-blocking — triage for later)

### O1 — Tilt units naming (carry)

`tilt_*_gu_per_gu` is GU per vertex (scale in vertices). Keep until P5.

### O2 — Scarp ribbon vs full perimeter

Current `local_scarp` traces `width=8` verts outward from seam vertices along normals. Adequate for P2 diagnostics; P5 conductance needs full plateau boundary normals via `signed_distance` zero-contour or owner scarp contour polyline continuation.

### O3 — `support_core_threshold` dual use

Same `0.65` for `core_cost = edge_cost*thresh` and `core = support>=thresh`. If tuning decouples geodesic core radius from probability threshold, split into two keys.

---

## 4. Exact References

- Module: `src/procgen/terrain_primitives.py:1-626` (P2 `381-626`; fixed seeds `516-527`, cost `394-435/544-548`, config `23/485/487/533/546-547`)
- Config: `configs/tamriel_reworked_v1.json:6-30` (`semantic_downsample:4`, `max_continuation_cells:8.0`, `geodesic_cost_multiplier:2.0`, `geodesic_lateral_eta:0.4`)
- Plan/request: `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md:20-26`, `request.md`
- Probes: compile + synthetic P1→P2 continuation on 32×32 (1 primitive, owner 122, seam 10, seam col 16 vs owner 15), 1024×64 8-cell (support 27472, scarp 48, col 512..919, visited 1720, max_cost 256), 1024×64 2-cell (support 4832, col 512..591, max_cost 64) — all generated-only, corridor reaches intended multi-cell extent.

---

## 5. Recommendation

**P2 re-review PASS.** Both blocking fixes are present and behave correctly in narrow+wide synthetic probes (seam-anchored seeding, unit-correct multi-cell corridor, strictly generated domain, no height synthesis). No remaining P2 blocker. Proceed to P3 (plateau candidate surface). Do not run the real bottom-right `tr_vvardenfell_wall` crop until P3 review gate per `plan.md`.

*Reviewer: review-flash (read-only, P2-only re-review) — 2026-08-25 — files: `src/procgen/terrain_primitives.py:1-626`, `configs/tamriel_reworked_v1.json:10-30`, `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md`, `request.md` — probes: compile + synthetic Dijkstra/EDT/scarp probes (§4) — no crop executed, no P3-P5 inspected.*
