# Solver debug state — Sol: propose exact edits (no implementation by us)

Date: 2026-08-24 · Status: fixtures expose the bug; we are NOT fixing it
ourselves. **Sol: read this + `terrain_blend.py::solve_surface`, propose the
exact edited code blocks.**

## Symptom (synthetic fixture flat_step, 192x192, production assembler)

- Owner half-plane = 10,000 GU (rows <= 96), Tamriel side = 200 GU.
- Seam row = 96 (hard, own heights). Solve band = rows 96..144.
- Expected: smooth rise from 200 to ~10,000 across the band.
- Actual solved profile at col 100, rows 95..100:
  `[200, 10000, 1, 124, 171, 189]` — vertex 97 collapses to ~1 GU.
- `seam_c0` reports 0.0 (seam row itself is hard-pinned correctly).
- CG status 0 ("converged").

## Debug evidence (printed inside solve_surface after assembly)

```
slope_rows=174
b_vec[-174:] = [250000.0, 250000.0, 250000.0, ...]   # slope RHS present & correct
Atb_max = 252
Atb[slope_cols[:3]] = [251.5, 251.0, 251.0]          # should be ~6,250,000
AtA_diag[slope_cols[:3]] = [637.57, 637.57, 637.57]  # 25^2 + membrane/data terms, plausible
```

Interpretation: the slope equations exist in `A` (vals 25, rhs 250,000 — the
residual report reads 250,000 from the same slice), but **Aᵀ·b_vec contributes
nothing at the slope columns** (Atb ≈ 251 instead of ≈ 6.25M). The solved
value 1 GU is what remains when the slope family is effectively zero-weighted.

## Prime suspects for Sol to check in `solve_surface`

1. Row/RHS alignment across families: `rows` is concatenated per family, but
   the Laplacian family appends FIVE row-entries per unknown (four neighbor
   terms + one diagonal) while its RHS appends four zeros + one `acc_fixed`
   vector — verify the per-family entry counts match 1:1 between
   `rows/cols/vals` and `rhs` (off-by-N here silently shifts every later
   family's RHS).
2. `fam["slope"]` uses `slope_r` for BOTH rows and cols (diagonal rows) —
   verify `vals`/`rhs` ordering matches after the concatenations.
3. `A.T @ b_vec` with CSR `A` — confirm no dtype/int overflow in
   `rows`/`cols` (int64) and that `n_rows == len(b_vec) == len(vals)` at
   assembly time (assert them).

## Files (this repo, current state)

- `src/procgen/terrain_blend.py` — assembler + debug print (line ~"Atb_max")
- `tools/terrain/test_surface_fixtures.py` — the 5 fixtures (flat/sloped/
  corner/staircase/mountain) that reproduce the failure in <5 s
- `tools/terrain/solve_region_v3.py`, `src/procgen/terrain_metrics.py`,
  `src/procgen/terrain_relief.py`, `configs/tamriel_reworked_v1.json`

Reproduce: `python tools/terrain/test_surface_fixtures.py`

## Also pending (user rulings, not started)

- `terrain_relief.gentle_end_fraction` must be 0.15 (config still 0.30);
  `relief_scaled_field.npz` must be regenerated after the change (npz now
  carries a config hash; `load_target` refuses stale files).
- Config `solve.v3.quality` gates exist; the driver auto-fails on
  first-edge drop > 2,500 GU and slope residual > 200 GU.
