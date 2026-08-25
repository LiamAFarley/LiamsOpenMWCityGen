# 2026-08-25 P5 Reconciliation Review — TR Primitive Plateau/Canyon

**Scope:** Source-only review of P5 per `request.md` + `plan.md` §P5. Inspected `src/procgen/terrain_primitives.py:1010-1228` (`_edge_conductance:1010-1034`, `_solve_edge_aware_structure:1037-1188`, `reconcile_primitive_candidates:1191-1228`) and `configs/tamriel_reworked_v1.json:57-65` (`reconciliation.*`). Verified the eight P5 prompt gates: one symmetric edge-aware screened-Poisson matrix (not normal equations), configured PyAMG/Jacobi solve, primitive candidate/background RHS, explicit scarp-normal conductance reduction, complete active-boundary + owner Dirichlet, no isolated unknowns, finite outputs, exact fixed values. Compile + narrow synthetic reconciliation probes executed (24×24 and 6×6, scarp vertical, Jacobi and AMG branches, zero-weight and isolated cases). No driver wiring inspected, no real `tr_vvardenfell_wall` crop executed.

**Verdict: PASS — P5 reconciliation source meets all checked gates. Proceed to gated real-crop visual review per plan.md; AMG convergence alone is not visual acceptance.**

---

## 1. P5 Gate Summary

| Check (prompt) | Result | Evidence |
|---|---|---|
| **One symmetric edge-aware screened-Poisson matrix (not normal equations)** | **PASS** | `_solve_edge_aware_structure:1083-1130` builds `M = L + diag(data_weight)`. `L` is graph Laplacian: `degree` accumulates `w_e` per active neighbor `1108`, diagonal `degree + data_weight[unknown]` `1126`, off-diagonal `-w_e` only for unknown–unknown edges `1110-1113`, fixed neighbors folded to RHS `1114-1121`. Loop over 4 directions `1087` inserts each undirected edge twice with same `_edge_conductance` value → symmetric `M` (`M-M.T nnz 0` in probe, §3). No `A^T A` or normal-equations `1050` docstring says screened-Poisson. Single `sparse.coo_matrix(..., shape=(n,n)).tocsr()` `1127-1130` solved once. |
| **Configured PyAMG/Jacobi solve** | **PASS** | `solver = config.get("linear_solver","amg_rs_cg")` `1131`; `amg_rs_cg` → `pyamg.ruge_stuben_solver(matrix, max_coarse=config.amg_max_coarse)` `1136-1137` + `aspreconditioner(cycle="V")` `1139` else `jacobi_cg` → `sparse.diags(1/diag)` `1143-1145` else `ValueError`. Config wired `configs/tamriel_reworked_v1.json:58-61` `linear_solver:amg_rs_cg cg_tol:1e-6 cg_maxiter:200 amg_max_coarse:500`. Probe: `jacobi_cg 27 iter residual 0.0033`, `amg_rs_cg 1 iter residual 4e-11 amg_levels 1` both finite, setup/solve timed `1132/1148/1155`. Raises if `pyamg is None` `1134-1135` and on `status !=0` `1166-1167`. |
| **Primitive candidate/background RHS** | **PASS** | `reconcile:1205-1207` `candidate_weight = plateau_arrays[candidate_weight] + canyon_arrays[canyon_weight]` (float32). `1075-1079` `weights = clip(nan_to_num(candidate_weight))`, `generated = active & ~fixed`, `data_weight = weights + background_weight*generated`, `guide_rhs = weights*nan_to_num(candidate) + background_weight*nan_to_num(h0)` where `candidate=canyon_field` (plateau+canyon after subtraction `995-997`) `1214-1215`. `rhs = guide_rhs[unknown]` `1083`. Config `background_weight:0.12` `62`. Outside support `weights=0` → RHS pulls to `h0` (screened Laplacian fill), not harmonic. Probe zero-weight case returns finite smooth fill `residual 0.002`. |
| **Explicit conductance reduction across scarp normals** | **PASS** | `_edge_conductance:1010-1034` `confidence=0.5*(c+roll(c,-dy,-dx))`, `ny/nx` similarly `1019-1024`, `alignment = abs(dy*ny+dx*nx)/norm` `1026-1031` (vector vs scarp normal), `w = gmin + (1-gmin)*exp(-beta*confidence*alignment)` `1032-1034` with `gmin=0.1 beta=3.0` from `1085-1086` `config conductance_min/beta` wired to JSON `38-39` and `63-64`. Horizontal across vertical scarp (alignment 1) at `±` 0.6–1.0 confidence → `0.18` (probe `0.1816`), parallel (alignment 0) → `1.0` (probe `1.0000`). Applied per-direction `1101-1104` before degree/RHS. Roll wrap excluded via `center` border mask `1089-1095` + `valid` filter. |
| **Complete active-boundary and owner Dirichlet values** | **PASS** | `reconcile:1200-1204` `active=smask`, `owner=owner_mask & active`, `fixed = hard | owner`, `fixed_values = hard_vals.copy(); fixed_values[owner]=h0[owner]` → owner + seam/outer boundary exact. Checks `fixed ⊆ active` `1052-1053` and `unknown = active & ~fixed` `1056` interiority `1068-1072` `unknown & ~eroded(active) → ValueError` (one-vertex Dirichlet ring required; caller must provide `hard` outer ring via `ctx`). After `cg`, `out[fixed]=fixed_values[fixed]` `1170` exact; probe `max|out[fixed]-fixed_vals|` `0.0` for both solvers and wrapper, `owner 0.0`. `active_vertices/fixed_vertices/owner_vertices` reported `1222-1225`. |
| **No isolated unknowns** | **PASS** | Two guards: (1) `unknown & ~active_interior → raise "touches inactive"` `1071-1072` catches single-vertex islands and border unknowns; probe `5×5 active single cell` correctly raises. (2) `degree<=0 → raise "isolated"` `1122-1123` after 4-neighbor accumulation. `degree` sized `n` `1084`, symmetric, min `~2.9` in 6×6 interior test. |
| **Finite outputs** | **PASS** | Pre-check `active & ~finite(h0) → raise` `1054-1055`; `unknown==0` early return copies `h0`+fixed `1058-1067`; post-solve `out = h0.copy(); out[unknown]=solution; out[fixed]=fixed` `1168-1170` `astype float32`. All probe runs `finite True`, residual bounded, zero-weight/canyon-free cases finite. `weights/candidate` via `nan_to_num` prevents NaN RHS leak. |
| **Exact fixed values** | **PASS** | No approximation of fixed: excluded from unknowns, RHS contribution via `w*hard_vals` added to `rhs` `1116-1121`, final overwrite `out[fixed]=fixed_values[fixed]` `1170`. Wrapper preserves `owner` immutability. Probes show `0.0` diff under both solvers and wrapper, including zero-weight path. |

Compile: `py_compile src/procgen/terrain_primitives.py` → `COMPILE_OK` (re-checked this review).

Synthetic probes (this review): 24×24 active `2:22×2:22` owner `col<6=6000`, fixed outer ring, candidate `7200` weight `1` at `10:18×6:18`, vertical scarp `col12 conf1 normal(0,1)`; `jacobi_cg 270 unknowns guide_rows 96 bg 0.12 iter27 res0.0033 corr1955.8 finite True fixed0.0`; `amg_rs_cg 1 iter res4e-11 levels1 fixed0.0`; wrapper `same`; zero-weight `finite True res0.002`; isolated single-cell `ValueError touches inactive` (correct); 6×6 symmetry `M-M.T nnz 0 diag 2.9-4.6`. No I/O, no real crop, no driver inspected.

---

## 2. Detailed Findings

### 2.1 Matrix is screened-Poisson, not normal equations — confirmed
- Construction is direct Laplacian + screening diagonal, not `A^T A`. Data weight `background_weight*generated` guarantees positive diagonal even where `candidate_weight=0`, preventing harmonic rectangular corridor (screened fill toward `h0`). `matrix @ solution - rhs` residual reported `rms` for diagnostics.

### 2.2 Solve is config-driven — confirmed
- `reconciliation.linear_solver` selects branch; `amg_max_coarse` only used for AMG, Jacobi uses diagonal preconditioner with positivity check `1142-1143`. Both paths share `cg` with `rtol=cg_tol atol=0.0 maxiter=cg_maxiter` `1158-1164` and iteration callback counting `1149-1153`. Solver timings recorded.

### 2.3 RHS correctly blends primitive candidate and background
- Weight sum merges plateau support prob and canyon `q*weight`; `canyon_field` already contains `top` authority + canyon subtraction. Where no primitive, background `0.12*h0` screens the Poisson solution. Verified via non-trivial `guide_rows 96` and zero-weight smooth fallback.

### 2.4 Conductance reduction is explicit and direction-selective
- Averaged scarp confidence + normal at edge midpoint, `abs` alignment makes reduction symmetric across edge direction. `exp(-beta*conf*alignment)` yields `gmin` floor; `beta` controls sharpness. Orthogonal edges stay `1` (diffuse), normal edges drop to `0.18` at confidence ~0.8. Roll wrap neutralized by `center` exclusion.

### 2.5 Boundary/owner completeness
- `fixed = hard | owner` is the only Dirichlet source; `hard` expected from driver as `active` outer apron (per `plan.md` narrow final seam lock later, not inside solve). The `active_interior` erosion guard fails loud if driver forgets hard, preventing silently floating unknowns. Owner overwrite after solve guarantees immutability even if solver drifts.

### 2.6 Exactness and finiteness
- Direct assignment `out[fixed]=fixed_values` guarantees bit-exact (0.0 diff) rather than penalized approximation. Finiteness enforced by early finite check and `nan_to_num` on RHS inputs only.

---

## 3. Observations (non-blocking)

### O1 — Conductance only scarp-driven
- Canyon walls do not add conductance anisotropy; scarp field from `continue_plateau_footprints` is sole edge-modulator. Acceptable per plan, but deep canyon walls will smooth across thalweg unless scarp overlaps. Monitor visual.

### O2 — `amg_levels 1` on small probe
- Tiny 270-unknown system collapses to 1 AMG level; real crop `~10k` unknowns will show more levels. Not a defect; `amg_max_coarse 500` is sensible.

### O3 — Background weight is global scalar
- `0.12` applied uniformly to all generated vertices, not spatially tapered by distance to support. This keeps far-field close to `h0` but may mildly attenuate long-range diffusion. Config-tunable.

### O4 — No erosion or second solve
- Module contains no `erode` or second `cg`; `reconcile_primitive_candidates` calls `_solve_edge_aware_structure` exactly once, correctly deferring to existing narrow final seam lock per `request.md`.

---

## 4. Exact References

- Module: `src/procgen/terrain_primitives.py:1010-1228` (`_edge_conductance:1010-1034`, `_solve_edge_aware_structure:1037-1188`, `reconcile_primitive_candidates:1191-1228`, pre-checks `1052-1072`, matrix `1083-1130`, solve `1132-1167`, report `1172-1188`)
- Config: `configs/tamriel_reworked_v1.json:33-40` (`scarp.conductance_min:0.1 beta:3.0`), `57-65` (`reconciliation linear_solver:amg_rs_cg cg_tol:1e-6 cg_maxiter:200 amg_max_coarse:500 background_weight:0.12`)
- Plan/request: `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md:44-51`, `request.md` (single edge-aware screened-Poisson + narrow lock, no erosion)
- Probes: `C:\Users\LiamF\AppData\Local\Temp\opencode\p5_repro_probe.py` (compile + 24×24 Jacobi/AMG + wrapper + zero-weight + isolated + 6×6 symmetry) — no real crop, no driver wiring

---

## 5. Recommendation

**PASS.** All eight P5 source gates verified against `request.md`/`plan.md`: one symmetric screened-Poisson `L+diag` system (not normal equations), configured `amg_rs_cg`/`jacobi_cg` solve with `pyamg`, `candidate_weight + 0.12*h0` RHS, scarp-normal conductance `gmin + (1-gmin)exp(-beta*conf*alignment)` with orthogonal vs parallel selectivity, `hard|owner` exact Dirichlet with perimeter and `owner` completeness, isolated/ inactive-touch guards, finite outputs, and exact `0.0` fixed preservation under both solvers. No blocking defect found. Keep fix narrow; do not run erosion or broad batch. Next gate is visual inspection of the real `bottom_right` `tr_vvardenfell_wall` checkpoint — generate it with this screened-Poisson reconciliation + existing narrow final seam lock and stop for user/lead review.

*Reviewer: review-flash (read-only, P5-only source review) — 2026-08-25 — files: `src/procgen/terrain_primitives.py:1010-1228`, `configs/tamriel_reworked_v1.json:57-65`, `.opencode/runs/2026-08-25_tr-primitive-plateau-canyon/plan.md`, `request.md` — probes: compile + synthetic Jacoby/AMG/wrapper/isolated/symmetry (§1) — no crop executed, no driver inspected.*
