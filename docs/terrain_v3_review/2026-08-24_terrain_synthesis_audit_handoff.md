# Audit handoff — Tamriel Reworked terrain synthesis (2026-08-24)

Audience: an independent auditing agent. This documents everything tried,
what failed and why, current state, and where to verify. The user has
rejected all synthesis renders so far; the latest approach (v2) is written
but **not yet run**. Audit questions are at the end.

## 1. Task

Produce `output/esps/tamriel_reworked.esm`, a replacement for
`tamriel.esm` (Habasi full-map canvas) that:

1. Deletes every CELL+LAND position owned by a landmass mod (vanilla
   Morrowind/Bloodmoon, TR_Mainland, Sky_Main, Cyr_Main) so those mods own
   their territories regardless of load order.
2. Blends retained-tamriel terrain into every new border with EXACT vertex
   match at the seam, in the **visual style of the owner mod** — the user's
   words: "coherent well designed heightmaps with realistic structure… a
   more natural fractally style look instead of this wacky noise matching
   that doesn't correspond to anything in particular."
3. Optionally amplifies interior mountains later (cap = configurable factor
   × the highest owner-mod peak; base 21,168 GU measured).

User rulings (binding, chronological):
- Owner precedence: config order, **later wins** — TR outranks vanilla
  Morrowind on shared narrow-sea/island cells (TR's additions deliberate).
- Vanilla Morrowind/Bloodmoon count as owners (150 cells deleted).
- HF_Main excluded until release; new owners join via config edit + rerun.
- Erosion: first deferred, then pulled INTO this phase ("matching terrain
  style might involve building some amount of terrain generation… it'll be
  hard to get similar terrain at these seams without some way of handling
  plateau/valleys stuff and proper mountains").
- Height-less owner stub LANDs: use tamriel.esm heights instead ("we
  shouldn't have any missing cells").
- Stamps declared useless by the user after round 2; noise matching
  declared useless; spl (stream power) "the only one that looks remotely
  okay but even then… very aggressive seams at the mountain edges and…
  really ugly crosshatched noise everywhere".
- Review format: large seam-centered heightmap crops, identical framing
  before/after.

## 2. Verified ground truth (fresh streaming scans, 2026-08-24)

| Source | LAND cells | Peak GU | tam∩source |
|---|---|---|---|
| Morrowind.esm (Steam) | 1,390 | 18,952 @(3,9) | 147 |
| Bloodmoon.esm | 150 | 4,800 | 3 |
| TR_Mainland.esm | 2,592 | 21,168 @(-20,-21) | 2,480 |
| Sky_Main.esm | 297 | 10,528 | 297 |
| Cyr_Main.esm | 510 | 12,096 | 457 |
| tamriel.esm | 32,086 | 21,272 @(-72,1) | — |

- tamriel.esm contains ONLY: TES3 header (masters=[]), 32,086 CELL,
  32,086 LAND, 141 LTEX.
- Delete set 3,252 cells → retain 28,834. Seams: 531 edges / 446
  tam-side cells (TR 314 / Sky 92 / Cyr 90 / MW 22 / BM 13).
- Current seam severity: median worst-vertex mismatch ≈1,064 GU, max
  8,568 GU (pre-work measurements).
- Morrowind.esm has **98 height-less LAND stubs** (VTEX-only records over
  water). Where shared with TR, TR supplies real heights (later-wins).
  TR also has some height-less cells (127 seam verts had NaN owner heights).
- openmw.cfg load order: Tamriel_Data → TR_Mainland → Sky_Main → Cyr_Main
  → tamriel.esm → (our generated plugins; falkreath_r18_town.esp last).
- Measured style gap (band σ via blur-stack, GU): TR ≈ [503, 350, 497,
  716, 1019, 1249] at band sizes 2/4/8/16/32/64 verts vs ambient tamriel
  [37, 29, 49, 78, 127, 238] — TR is 10–30× "hillier" at every scale.
- OpenMW's LAND cross-plugin conflict resolution is NOT documented in the
  openmw-docs snapshot; the deletion design sidesteps the question.

## 3. Implementation inventory (all paths relative to F:\ProcGenWorkspace)

Committed (master):
- `a3b51ec3` Stage A: `src/procgen/terrainfield.py` (corpus build/load,
  seam edges, hillshade, hypsometric tint, ownership-split window renderer),
  `tools/terrain/build_terrain_corpus.py`, `configs/tamriel_reworked_v1.json`,
  run docs. Corpus verified: retained 28,834 / deleted 3,252 / 531 seams;
  tam_h internal duplicate audit = 0 conflicts; oth_h = 219 verts
  (owner-vs-owner), max 1,224 GU.
- `2ac0f11c` Stage B: `tools/terrain/analyze_border_profiles.py` — 435 seam
  clusters classified (cliff-wall 236 / sharp-mixed 71 / plateau-step 62 /
  smooth-rise 26 / already-matched 36 / void-owner 4); review crops under
  `output/mapdata/terrain/tamriel_reworked/review_crops/` (435 PNGs +
  `_index.md`).
- `26571ce1` Panel round 1: `tools/terrain/panel_region.py`,
  `src/procgen/terrainstyle.py` (band stack, style measurement, band-matched
  noise synthesis, metrics).
- `336f1e66` Panel round 2: frequency-split clone, stream-power (spl),
  slab filter, reworked droplets; metrics JSON + renders under
  `output/mapdata/terrain/tamriel_reworked/solved/panel/`.

Uncommitted working tree:
- `tools/terrain/solve_region_v2.py` — NEW, written, **never executed**.
  Implements the "owner-anchored erosion synthesis" approach (§5). All
  parameters have in-code defaults via `cfg.get`; the `solve.v2` config
  block has NOT been added to the config yet (TODO before production —
  workspace rule: values belong in JSON).

Key data artifacts:
- `output/mapdata/terrain/tamriel_reworked/corpus_v1.npz` (+ manifest):
  `tam_h`/`oth_h` float32 GU vertex fields (NaN=void), `cell_owner` uint8
  (0=void, else source index+1; owners later-in-config win), origin ints.
  Vertex (row,col) = ((cy-gy0)*64+v, (cx-gx0)*64+u), 128 GU spacing,
  shared cell edges stored ONCE (build-time overwrite audit).
- `output/mapdata/terrain/tamriel_reworked/seam_atlas_v1.json` — per-cluster
  classification + features (dmed/dmax, slope, decay, profiles).
- `output/mapdata/terrain/tamriel_reworked/solved/` — v1 solve + panel
  renders/metrics. `solved/panel/tr_vvardenfell_wall_COMPARISON_spl_clone_hybrid.png`
  is the round-2 comparison sheet.

## 4. Chronology of attempts, outcomes, and root causes

### Round 1 — regional solve (smooth base + gated noise)
Harmonic (coarse ds=8 Laplace) base between seam Dirichlet and ambient
ring; detail = fBm noise, amplitude = slope/relief-gated, σ-rescaled.
User verdict: mountain slot with smooth apron, square-tile pattern, new
rough→smooth seam one cell in, circular shape.
Root causes identified:
- Band-σ matching reproduces MARGINAL statistics, not structure. Two fields
  with identical per-band σ look nothing alike; TR's HF is the OUTPUT of
  drainage organization, not additive texture.
- Amplitude gated on local slope piled all roughness into the steep slot.
- Detail faded to ZERO at the solve boundary → new interior seam.
- Coarse ds=8 grid + box-filter bandpass + bilinear zoom → terracing/tiles.
- Laplace base gives value continuity but NOT slope continuity at the seam.

### Panel round 1 — noise / droplet / clone (all rejected)
- noise: as above.
- droplet: eroded a NOISE seed (garbage in); ~14 droplet-visits per vertex
  (100k × 140 steps over 12.8M verts) — an order of magnitude too sparse
  for channel emergence; constants outside the channel-feedback regime;
  thousands of below-sea pits. "Pitted incoherent mess" (user).
- clone: random rot/mirror patches ignore local geometry (ridge orientation
  vs flank flow) and carried absolute heights → elevation chaos + tile
  seams. "Incoherent tiled noise" (user).

### Panel round 2 — freq-split clone / spl / slab / droplet2
- clone v2: per-band patch transplant, texture-only (no heights), per-band
  patch sizes/grids, crossfade to ambient per band. Metrics best (ori corr
  0.94–0.98) but user verdict: "stamps are completely useless… incoherent
  tiled noise that in no way matches the geometry". Root cause: placement
  ignores geometry — a NE-running TR ridge segment dropped onto a
  NW-flowing flank is incoherent BY CONSTRUCTION; patch sources picked by
  max relief put steep texture everywhere. (Also fixed en route: owner
  stub cells were zero-filled before band decomposition → rectangular
  band-spike outlines copied around; now nearest-valid filled, then
  superseded by the own_view ruling.)
- spl (stream-power + creep, explicit, capped, sinks never incised):
  first variant with real ridge-and-valley organization. User: "only one
  that looks remotely okay but… aggressive seams at the mountain edges and
  really ugly crosshatched noise everywhere". Root causes: (a) flow routing
  on a near-smooth Laplace base is decided by numerical tie-breaking
  (8-dir + epsilon jitter) → channels follow grid artifacts = crosshatch;
  (b) Laplace value-only continuity → crease at seam edges; (c) no
  depression filling; (d) drainage outlets not pinned to ambient rivers;
  (e) seed under erosion was again band-matched noise. 149 s runtime
  (over the 120 s workspace ceiling).
- slab (Hatchling threshold-slab via per-layer EDT): striated "combed"
  texture — literature says it is a soil-creep/aging operator, not a
  structure creator. 22 s.
- droplet2 (deposition-dominant + downcut clamp): pits fixed, still
  uniform fuzz. 25 s.
- hybrid (clone near seam + spl beyond): clone rectangles polluted the
  carry zone (stub zero-fill bug at the time); after the own_view fix it
  was rerun clean but the user had already rejected stamps wholesale.

### Own_view ruling (latest)
`own_view = where(isfinite(oth_w), oth_w, tam_w)` — owner heights where
they exist, tamriel heights under height-less owner stubs. Wired through
context, style reference, seam Dirichlet, border-flow seeds, and renders
(no more gray void patches). Consequence for authoring (Stage E, NOT yet
implemented): a position should only be deleted from tamriel if the
winning owner provides actual VHGT; otherwise keep the tamriel cell —
prevents void holes. NOT yet encoded in any authoring code (none exists).

## 5. The v2 approach (written, unrun, unaudited)

`tools/terrain/solve_region_v2.py` — single focused method:
1. **Owner skirt**: harmonic solve window includes a one-cell owner ring
   with real owner heights → base inherits owner slope direction at the
   seam (fixes the crease).
2. **Macro continuation**: owner low-pass (blur 65) sampled at each
   interior vertex's nearest seam point; added where our base is smoother
   than the continued macro, weight exp(-d/(3.5 cells)). Our spurs are the
   owner's spurs decaying inward — "corresponds to something".
3. **Erosion in the structure-creating regime**: explicit stream-power
   incision (k=1.0, m=0.8, cap 12 GU/step, 250 steps, graph refreshed
   every 10) + hillslope creep (0.05) on the D8 graph; border flow-entry
   verts get +2000 accumulation (river continuation); sinks never incised;
   Dirichlet re-imposed every step. NO added noise HF (micro_sigma
   defaults to 0).
4. Renders include an OWNER REFERENCE crop (TR side at identical scale) +
   a stacked comparison image — the user reviews ours against the actual
   reference.

## 6. Audit questions

1. Is the v2 erosion parameterization in a regime where channel networks
   actually emerge (k=1.0, m=0.8, cap 12 GU/step, 250 steps, graph refresh
   every 10, accum capped 20k, seed boost 2000 at 378 border verts)?
   Sanity-check against Fastscape-style scaling; flag if explicit capped
   steps cannot produce dendritic relief here.
2. Is `coarse_laplace` (ds=8 block solve + cubic zoom) an adequate base,
   or does its residual gridness contaminate erosion routing (the
   round-2 crosshatch suspicion)? Would ds=4, thin-plate/biharmonic, or
   a multigrid solve be materially better?
3. Macro continuation via nearest-seam-point sampling: does it produce
   ridge aliasing along cluster seams (nearest-point discontinuities)?
   Cheaper alternative rejected so far: synthetic spurs. Better options?
4. The style metrics (band σ ratios, global orientation histogram corr,
   σ-vs-distance cliff detector) failed to predict visual rejection in
   round 1. Propose/validate stronger perceptual metrics (windowed
   orientation coherence, drainage density, channel-head density, slope
   histograms on the massif only).
5. Verify the corpus numbers in §2 by re-running
   `python tools/terrain/build_terrain_corpus.py` (≈70 s) and comparing to
   `expected_counts_v1` in the config.
6. Review the rejection evidence yourself: panel renders under
   `solved/panel/` (esp. `tr_vvardenfell_wall_{spl,clone,droplet,hybrid}.png`
   and `..._COMPARISON_spl_clone_hybrid.png`) vs the owner reference crops
   (`solved/v2/tr_vvardenfell_wall_v2_reference.png` requires running v2,
   or crop TR side from `review_crops/`).
7. Check the own_view authoring consequence (§4) is sound for Stage E:
   "delete only if winning owner provides VHGT; else keep tamriel cell".

## 7. How to re-run everything

```
python tools/terrain/build_terrain_corpus.py            # Stage A (~70 s)
python tools/terrain/analyze_border_profiles.py         # Stage B (~60 s)
python tools/terrain/solve_region_blend.py --region tr_vvardenfell_wall
python tools/terrain/panel_region.py --region tr_vvardenfell_wall \
       [--variants noise,clone,spl,slab,droplet,hybrid]
python tools/terrain/solve_region_v2.py --region tr_vvardenfell_wall   # unrun
```
All read `configs/tamriel_reworked_v1.json`; plugins are opened read-only;
outputs land under `output/mapdata/terrain/tamriel_reworked/`.

## 8. Known risks / open defects

- spl runtime 149 s > 120 s workspace ceiling (window crop or fewer
  iterations needed).
- Faint ds=8 coarse-grid texture in flat areas of every variant.
- Seam-line staircase slightly visible at ridge tops in all variants.
- `solve_region_blend.py` and `panel_region.py` duplicate context-building
  code (productionization should merge into one shared module).
- `terrainstyle.band_stack` uses box filters (square kernel artifacts);
  Gaussian stack may be safer for style measurement.
- The v1 solve (`solve_region_blend.py`) is visually superseded; kept only
  as the base-solver library (`coarse_laplace`) reused by panel/v2.
- Erosion explicit-scheme stability is enforced only by per-step caps, not
  by a stability analysis.
