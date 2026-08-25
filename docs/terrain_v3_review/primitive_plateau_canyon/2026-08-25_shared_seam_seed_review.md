# 2026-08-25 Shared Seam-Seed Correction Review — TR Primitive Plateau/Canyon (P2/P4 only)

**Scope:** Production seam-seed correction only per task instruction. Inspected `src/procgen/terrain_primitives.py:_first_generated_vertex:400-414` and its two call sites (`continue_plateau_footprints:511,545-556` P2; `continue_canyons:917-923` P4) plus `configs/tamriel_reworked_v1.json:10` `seam_seed_max_steps`. Verified synthetic-contract preservation (step 0 when seam itself is generated) and correct 1-/2-step advance when production `owner_mask` includes the shared seam vertex. Confirmed P2 support and P4 canyon routing remain strictly generated-only under a shared-seam synthetic probe. No P1/P3/P5 logic inspected; no real `tr_vvardenfell_wall` crop executed.

**Verdict: PASS — production seam-seed correction verified. Synthetic contract preserved; P2/P4 remain generated-only with shared seam.**

---

## 1. Gate Summary (P2/P4 seam-seed only)

| Check (prompt) | Result | Evidence |
|---|---|---|
| **Helper `_first_generated_vertex` — correct walk** | **PASS** | `terrain_primitives.py:400-414` `for step in range(max(0,int(max_steps))+1): sy=row+normal[0]*step, sx=col+normal[1]*step` bounds-checked `0<=sy<H,0<=sx<W` and `if generated[sy,sx]: return sy,sx,step`. Docstring "Walk from a shared seam to the first vertex outside owner authority" matches implementation. `max_steps` clamped to `max(0,…)`, inclusive `+1`, returns `None` if no generated vertex within walk — safe fallback. |
| **Config `seam_seed_max_steps` — present and wired** | **PASS** | `configs/tamriel_reworked_v1.json:10` `"seam_seed_max_steps": 2.0`. P2 `seam_seed_max_steps = int(round(float(config.get("seam_seed_max_steps",2.0))))` `511` and P4 `int(round(float(config.get("seam_seed_max_steps",2.0))))` `919` both read with fallback `2.0`. Inclusive walk `0..2` therefore covers synthetic step-0, production step-1, and double-buffer step-2; deeper owner buffers correctly return `None` (no seed) rather than crashing. |
| **Synthetic contract preserved — seam generated returns step 0** | **PASS — unit verified** | Probe `generated=smask & ~owner` where `owner cols 0-31, seam 32` (`generated[32,32]=True`): `_first_generated_vertex(32,32,(0,1),generated,2) -> (32,32,0)`. Same for `(10,10)` synthetic `owner_max 9` -> `(10,10,0)`. Prior synthetic behavior (seed at seam) reproduced exactly; P2/P4 synthetic support probe still yields `support_verts 2048` from `start_col 32` (see §4). |
| **Production 1-step advance — shared seam owned** | **PASS — unit + integrated** | Production case `owner cols 0-32 inclusive` (`generated[32,32]=False, generated[32,33]=True`): `_first_generated_vertex(32,32,(0,1),generated,2) -> (32,33,1)`. Unit `(10,11,1)` for `owner_max 10`. Integrated P2: shared-seam probe `owner 0-32` yields `support_verts 1984 start_col 33` vs synthetic `2048 start_col 32` — one-vertex shift as expected, strictly generated-only (`support*owner sum 0.0`). Integrated P4: same production case yields `canyon_line 15 verts depth_max 4180.7 line_in_owner 0 depth_in_owner 0.0 owner_immut 0.0`. |
| **Production 2-step advance — double buffer** | **PASS — unit + integrated** | Double buffer `owner cols 0-33` (`generated[32,32]=False, generated[32,33]=False, generated[32,34]=True`): `_first_generated_vertex(32,32,(0,1),generated,2) -> (32,34,2)`. Unit `(10,12,2)` for `owner_max 11`. Integrated P2 double-buffer yields `support_verts 1920 start_col 34` generated-only; P4 yields `canyon_line 15 verts` generated-only. Beyond max `owner_max 12` -> `None` correctly (no seed, no crash). |
| **P2 support remains generated-only with shared seam** | **PASS** | `continue_plateau_footprints:519 generated=smask & ~owner_mask` `520 generated_work`, `545 generated_seed=_first_generated_vertex(...)`, `551-555` check `work_bbox` and `if generated_work[ly,lx]: seeds.append`. Downstream `valid=generated_work & isfinite`, `sem_generated=_block_max(generated,4)`, seed gate `427`, `support_work=valid & isfinite(distance)`, `support_full *= generated` `588`. Probe synthetic vs production: `support_in_owner 0.0` in all three cases (2048/1984/1920 verts). |
| **P4 canyon routing remains generated-only with shared seam** | **PASS** | `continue_canyons:860 generated=smask & ~owner` `917-926` `generated_seed=_first_generated_vertex(...)`, `923 route_seed=(gy,gx)`, `924-927` local bounds check, then `_semantic_canyon_path:728 sem_generated=_block_max(generated,4)` + `sem_generated` gates `737-738,764-765`, line filter `generated[row,col]` `947-948`, `depth[~work_generated]=0` `1002`, `canyon_field[owner]=h0[owner]` `1028`. Probe: all three cases `canyon_line_in_owner 0`, `canyon_depth[owner].max 0.0`, `owner_immut 0.0` with line verts `16/15/15`. |
| **P2/P4 do not touch P1/P3/P5 code** | **PASS** | Inspection limited to `400-414`, `505-556`, `911-935`; no edits to `analyze_plateaus`, `synthesize_plateau_candidates`, or `reconcile_primitive_candidates`. Real crop not executed per instruction. |
| **Compile** | **PASS** | `py_compile src/procgen/terrain_primitives.py` OK (implicit via import in probes). |

Full-file compile: OK (probes import succeeded). No `py_compile` error.

---

## 2. Synthetic Probes (no real crop — per instruction)

### 2.1 `_first_generated_vertex` unit
```
synthetic seam generated (A): owner_max=9  gen[seam]=True  -> (10,10,0)  # step 0 preserved
production shared seam 1-step (B): owner_max=10 gen[seam]=False -> (10,11,1)
double buffer 2-step (C): owner_max=11 gen[seam]=False -> (10,12,2)
beyond max 2 (D): owner_max=12 gen[seam]=False -> None
edge seam (0,0) north gen true -> (0,0,0) expect (0,0,0)
edge seam (0,0) north gen false out of bounds -> None expect None
negative max_steps -> None (only step0)
max_steps 0 with seam owned -> None expect None
max_steps 1 with seam owned -> (10,11,1) expect step1
```

Walk table for seam `(32,32)` normal `(0,1)` max 2 on 64×64:
- synthetic `owner 0-31`: `(32,32,0)`
- production `owner 0-32`: `(32,33,1)`
- double buffer `owner 0-33`: `(32,34,2)`

All match spec: step 0 when seam generated, step 1 when seam owned, step 2 when seam+1 also owned, None when beyond max.

### 2.2 Shared-seam integrated P2/P4
64×64 synthetic terrain: plateau `8000` at `16:48,8:32`, downslope `15 GU/col` east of `32`, seam at `col 32` normal `(0,1)`, primitive `seam_vertices [[30,32],[31,32],[32,32]]` flat fit `8000`, valley `20:40,28:32` score `0.9`.

```
SYNTHETIC seam generated (owner 0-31, seam 32 gen): gen_at_seam=True support_verts=2048 scarp=0 support_in_owner=0.0 start_col=32
  canyon line_verts=16 depth_max=4180.7 line_in_owner=0 depth_in_owner=0.0 owner_immut=0.0
PRODUCTION shared seam 1-step (owner 0-32): gen_at_seam=False support_verts=1984 scarp=0 support_in_owner=0.0 start_col=33
  canyon line_verts=15 depth_max=4180.7 line_in_owner=0 depth_in_owner=0.0 owner_immut=0.0
PRODUCTION double buffer 2-step (owner 0-33): gen_at_seam=False support_verts=1920 scarp=0 support_in_owner=0.0 start_col=34
  canyon line_verts=15 depth_max=4180.7 line_in_owner=0 depth_in_owner=0.0 owner_immut=0.0
shared-seam probe PASS
```

Observations:
- Support vertex count drops `2048->1984->1920` (-64/-128) corresponding to one/two columns of semantic advance lost to the ownership walk — expected and bounded; corridor extent remains multi-cell (`max_cost 256` regime unchanged).
- `start_col` advances `32->33->34` exactly by walk step — visual confirmation that seeds are now placed at first generated vertex, not at owned seam.
- All cases `support_in_owner 0.0`, `canyon_line_in_owner 0`, `canyon_depth_in_owner 0.0`, `owner_immut 0.0` — generated-only invariant preserved.
- Prior code without `_first_generated_vertex` would have used `Primitive.seam_vertices` directly and checked `generated_work[ly,lx]` → shared-seam seeds at `col 32` owned would fail the `generated_work` gate and yield `seed_count 0` (no support/canyon). Correction restores intended seeding.

---

## 3. Exact References

- Module: `src/procgen/terrain_primitives.py:400-414` (`_first_generated_vertex` walk `range(max(0,int(max_steps))+1)` bounds+generated check), `505-556` (`continue_plateau_footprints`: `511 seam_seed_max_steps`, `545-556` seed loop with `_first_generated_vertex` + `work_bbox`/`generated_work` gates, `560-599` support/scarp), `847-935` (`continue_canyons`: `917-926` seed loop with `_first_generated_vertex`, local seed conversion `924`, `_semantic_canyon_path:728,756` generated-only Dijkstra, `940-1008` depth/weight `generated` masking, `1028` owner lock)
- Config: `configs/tamriel_reworked_v1.json:10` `"seam_seed_max_steps": 2.0` (covers 0/1/2 steps; P2/P4 both `int(round(float(config.get(...,2.0))))`)
- Plan/request: `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md:20-42` (P2/P4 gates), `request.md` (binding scope)
- Probes: inline Python via `src/procgen/terrain_primitives.py` imports — `_first_generated_vertex` unit (4 cases + edge/negative/max0) and shared-seam integrated P2/P4 probe `shared_seam 1-step/2-step` above — no I/O, no real `tr_vvardenfell_wall` crop.

---

## 4. Recommendation

**PASS — proceed.** The production seam-seed correction is minimal, correct, and preserves the synthetic contract:

- Synthetic `generated[seam]=True` → step 0 → identical to prior behavior (`support start_col 32`).
- Production `owner includes seam` → step 1 → `(32,33,1)`; double buffer → step 2 → `(32,34,2)`; beyond max → `None` safe discard.
- Both call sites (`P2:545`, `P4:917`) route through `_first_generated_vertex` on full-image `generated=smask & ~owner` with correct `work_bbox` and `generated_work`/`local_seed` gates.
- Generated-only invariants hold for P2 (`support*owner 0.0`) and P4 (`line_in_owner 0`, `depth_in_owner 0.0`, `canyon_field[owner]=h0[owner]`) under shared-seam probes.
- Config is externalized (`seam_seed_max_steps:2.0`) and defaults safely.

No change to P1/P3/P5 required. No real crop executed per instruction — visual acceptance of the bottom-right `tr_vvardenfell_wall` remains the final gate per `plan.md`.

*Reviewer: review-flash (read-only, P2/P4 seam-seed only) — 2026-08-25 — files: `src/procgen/terrain_primitives.py:400-414,505-556,847-935`, `configs/tamriel_reworked_v1.json:10`, `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md`, `request.md` — probes: `_first_generated_vertex` unit + shared-seam P2/P4 integrated (synthetic/production/double-buffer) — no P1/P3/P5 inspected, no real crop.*
