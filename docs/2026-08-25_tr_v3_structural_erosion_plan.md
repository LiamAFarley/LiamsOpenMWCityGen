# Tamriel Reworked v3 — Structural Continuation, Effective Erosion, and Multi-Region Validation Plan

**Date:** 2026-08-25
**Audience:** implementing local agent
**Scope:** Continue from the current working v3 pipeline after the already-accepted terrain assembly / relief / low-frequency seam setup. Build the missing structural-continuation and geomorphic stages, make erosion visibly and meaningfully alter the generated terrain, then validate the system across a diverse set of TR / Skyrim / Cyrodiil seam-heavy regions before any world-wide erosion pass.

## 0. Current accepted state and rules

Stages 1–3 are considered **accepted infrastructure** for this plan. Do not redesign them unless a later stage exposes a hard incompatibility.

The local relief-curve edit currently present in the working tree but not yet pushed is **out of scope for this plan**. Do not revert or redesign it as part of the erosion work.

The current erosion prototype is diagnostic only. It presently uses deterministic priority-flood routing, 8-neighbor MFD accumulation, a six-cell fixed owner halo, 16 cycles, explicit normalized stream-power subtraction with `incision_strength_gu = 3`, simple neighbor-mean hillslope relaxation, and a two-cell post-erosion harmonic seam lock.

The first real run changed the terrain so weakly that cycle 0 and cycle 16 were visually indistinguishable. That is not acceptable. Do not solve this by merely multiplying cycle count, assigning an arbitrary huge GU incision constant, or adding visible procedural noise.

## 1. Why the current erosion pass is visually inert

The current incision is effectively:

```text
delta =
    incision_strength_gu
    * (A / A95)^m
    * slope^n
```

with roughly:

```text
incision_strength_gu = 3
m = 0.5
n = 1
```

For most vertices `A / A95 < 1`, and many slopes are far below 1 when expressed as vertical GU / horizontal GU. Most vertices therefore receive much less than 3 GU per cycle, often below 1 GU. Sixteen cycles can easily produce only a few to a few tens of GU over broad areas, which is effectively invisible at the current map scale. The existing hillslope relaxation also partially erases these weak incisions.

The deeper issue is structural: the current erosion code does not know which TR ridge, valley, plateau, scarp, coastline, or river system should continue across the seam. Erosion can refine a plausible massif or valley; it cannot infer the correct landform from a generic harmonic bridge.

The production sequence therefore remains:

```text
owner feature analysis
        ↓
structural continuation
        ↓
hydrology
        ↓
geomorphic refinement
        ↓
final exact seam lock
```

Erosion is a refinement stage, not a landform-selection stage.

## 2. Target pipeline from this point onward

```text
[ACCEPTED STAGES 1–3]
authoritative terrain / missing-cell base
        ↓
relief-amplified target
        ↓
low-frequency harmonic generated base

[STAGE 4]
owner feature analysis
    - macro terrain
    - terrain orientation
    - ridges
    - valleys / thalwegs
    - plateau tops
    - scarps
    - coastline / river-mouth features
    - owner hydrology

[STAGE 5]
sparse structural continuation
    - continue only important semantic features
    - no raw owner-profile mirroring
    - no nearest-seam-pixel copying
    - no dense independent profiles

[STAGE 6]
structural pre-erosion field
    - already reads as mountain / valley / plateau / lowland
    - still smoother than final TR

[STAGE 7]
efficient routing / owner inflow
    - deterministic depression handling
    - grid-bias-resistant routing
    - real owner-side upstream contribution
    - cached routing topology

[STAGE 8]
effective geomorphic refinement
    - calibrated implicit stream-power incision
    - terrain-type-dependent erosion intensity
    - hillslope transport
    - 16–32 cycles
    - reroute only when needed

[STAGE 9]
narrow post-erosion exact seam lock

[STAGE 10]
regional validation across ~10 diverse seam windows

Later, only after acceptance:
world-wide erosion / map enhancement
```

## 3. Code organization

Add the missing production modules rather than continuing to expand one erosion script:

```text
src/procgen/
    terrain_features.py
    terrain_hydrology.py
    terrain_structure.py
    terrain_erosion.py

tools/terrain/
    erode_region_v3.py
    review_regions_v3.py
```

Keep `terrain_blend.py`, `terrain_inpaint.py`, and `terrain_relief.py` as accepted upstream stages.

## 4. Stage 4 — owner feature analysis

### 4.1 Analysis domain

Analyze only the current solve/review region plus local halos.

Initial defaults:

```text
owner feature-analysis halo: 6–10 cells
generated context halo:      2–4 cells
```

Never analyze the full world for a local solve.

### 4.2 Multiscale owner pyramid

On authoritative owner terrain construct:

```text
H8  = gaussian(owner, sigma=8 verts)
H24 = gaussian(owner, sigma=24 verts)
H64 = gaussian(owner, sigma=64 verts)
```

Interpretation:

```text
H64  massif / plateau / regional trend
H24  major ridge / valley / scarp structure
H8   smaller ridge / channel / flank form
```

Keep all large raster products `float32` and cache them.

### 4.3 Structure tensor

Using `H24`:

```text
gx, gy = gradient(H24)

Jxx = gaussian(gx*gx, sigma_tensor)
Jyy = gaussian(gy*gy, sigma_tensor)
Jxy = gaussian(gx*gy, sigma_tensor)
```

Start with `sigma_tensor ≈ 8–12 verts`.

Derive:

```text
orientation_angle
orientation_coherence
```

High coherence means elongated structure whose direction matters. Low coherence means no orientation should be forced.

These are guidance fields, not texture synthesis.

### 4.4 Ridge detection

Compute Gaussian derivative Hessians at one or two useful scales and derive a ridge score from transverse negative curvature, local prominence, and structure coherence.

Do not skeletonize every small ridge. Threshold relative to owner statistics, for example around owner p85–p90 plus a local-prominence condition.

The output should be a small number of significant seam-crossing ridges.

### 4.5 Valley / thalweg detection

Use owner hydrology plus concave curvature.

For strong valley crossings store:

```text
seam position
incoming direction
owner contributing area
thalweg elevation
local valley depth
estimated width
confidence
```

Major rivers must be distinguishable from small gullies.

### 4.6 Plateau detection

Plateau candidates should combine:

```text
moderate/high regional elevation
low H24 slope over a broad contiguous area
meaningful surrounding local relief
a concentrated nearby high-slope / high-curvature boundary
```

Output:

```text
plateau_top_mask
plateau_confidence
scarp_score
```

### 4.7 Coast / ocean classification

Output:

```text
underwater_mask
shoreline_mask
coastal_band
river_mouth_crossings
```

Do not apply terrestrial erosion underwater and do not let the shoreline move.

## 5. Stage 5 — sparse structural continuation

### 5.1 Forbidden approaches

Do not reintroduce:

```text
raw owner-profile mirroring
nearest-seam-pixel interpolation
one profile per seam pixel
dense edge extrusion
frequency-patch copying
```

These generated stripes, spikes, and enormous numbers of conflicting claims.

Continue only sparse semantic features.

### 5.2 Ridge guide curves

For each strong ridge crossing:

1. take the ridge tangent at the seam;
2. record seam height, width, and prominence;
3. continue a curve 3–6 cells inward;
4. allow exceptionally large massif ridges up to ~8 cells;
5. steer gradually toward compatible high target terrain;
6. constrain turn rate to roughly 10–15 degrees per 8 vertices;
7. smoothly decay authority with distance;
8. stop on compatible target relief, low confidence, leaving generated terrain, or maximum range.

### 5.3 Valley guide curves

Use the owner thalweg tangent, incoming flow direction, and generated target downslope direction.

Continue major valleys 3–6 cells initially; large rivers may continue farther.

The guide must be allowed to bend toward compatible generated low ground. Never force a valley uphill only to preserve its initial tangent.

### 5.4 Plateau continuation

Treat plateau top and scarp separately.

Plateau top:

```text
broad low-frequency height / tilt guide
```

Scarp:

```text
narrow stronger edge guide
```

This allows later erosion to preserve the top while dissecting scarps and outlets.

### 5.5 Guide rasters

Construct:

```text
guide_value
guide_weight
```

Default `guide_weight = 0`.

Use Gaussian ribbon support around semantic curves. Suggested initial widths:

```text
ridge:  4–12 vertices
valley: 4–16 vertices
plateau top: broad mask, low/moderate weight
scarp: narrow stronger weight
```

Decay weights laterally and longitudinally. Never end a strong guide with a hard cutoff.

## 6. Stable structural solve

Let `H0` be the accepted Stage-3 generated field.

Solve only a structural correction:

```text
C = H_structural - H0
```

Use a direct second-order screened-Poisson system:

```text
(L + W) C = W * (Hguide - H0)
```

where `W >= 0` is the sparse guide-weight raster.

Hard constraints remain the accepted seam and outer generated boundary rules.

Do **not** use `A.T @ A`. Do not resurrect the old dense absolute-height target-spring solver.

This system is AMG-friendly and reduces to the accepted harmonic base where `W=0`.

Initial relative guide strengths:

```text
major river / thalweg: 1.0
major ridge:           0.7
plateau scarp:         0.8
plateau top:           0.3
minor features:        0.2–0.4
```

Normalize the operator so these weights remain interpretable and bounded.

## 7. Pre-erosion structural criterion

Before erosion, the field should already correctly read as:

```text
mountain
valley
plateau
rolling hill
flat lowland
coast
```

It does not need final TR fine detail.

Required qualitative checks:

```text
major owner ridges cross the seam coherently
major valleys continue
plateau top/scarp organization survives
missing synthesized regions no longer remain generic smooth rectangles when authoritative surrounding terrain contains structure
```

## 8. Stage 7 — hydrology redesign

### 8.1 Static owner hydrology

Owner terrain is fixed. Compute owner routing and accumulation once per review window / source hash.

Use an owner hydrology halo around 8–12 cells initially.

Extract seam inflows:

```text
generated receiver vertex
incoming accumulated area
incoming flow direction
source owner vertex
```

Do not include millions of owner-halo vertices in every repeated erosion cycle just to recompute the same owner flow.

### 8.2 Generated routing graph

The existing 8-neighbor MFD implementation can remain a fallback, but production should target a compact two-receiver continuous-direction graph.

Preferred: proper D∞.

Acceptable first production implementation:

1. compute local downhill angle on the depression-resolved routing surface;
2. identify the two 8-neighbor directions bracketing the angle;
3. split flow by angular proximity/downhill magnitude;
4. reject non-lower receivers;
5. route all flow to one receiver if only one remains;
6. report a routing defect if no valid receiver exists after depression resolution.

Store:

```text
receiver_1 : int32
receiver_2 : int32
weight_1   : float32
weight_2   : float32
order      : int32
```

This is much smaller than a stored eight-receiver MFD graph and can be reused.

### 8.3 Rerouting cadence

Start with:

```text
reroute_every = 2 cycles
```

If topology remains stable, move to:

```text
reroute_every = 4
```

Between reroutes reuse receiver topology and accumulation; recompute current receiver slopes from the edited height field.

### 8.4 Priority-Flood cadence

Run depression handling only when rebuilding routing, not every cycle when topology is reused.

Use a routing-only adjusted surface. Never globally replace the rendered terrain with the filled routing surface.

### 8.5 Routing perturbation

Retire the regular sine/cosine perturbation as the production symmetry breaker.

Use deterministic seeded correlated random routing noise:

```text
normal RNG
→ Gaussian blur sigma 3–6 verts
→ normalize
→ amplitude only a few GU
```

Cap amplitude by both an absolute GU value and a small fraction of local relief.

Use it only on the routing copy.

## 9. Owner inflow

Owner and generated accumulation must use one consistent rainfall/vertex-area convention.

When static owner flow crosses into generated terrain:

```text
accumulation[generated_crossing] += A_owner_crossing
```

before generated downstream propagation.

This should make:

```text
major owner river → major generated river
small owner gully → small generated gully
ordinary owner slope → no artificial channel source
```

## 10. Stage 8 — effective implicit stream-power incision

### 10.1 Replace the arbitrary GU-strength law

Do not continue tuning `incision_strength_gu`.

Use:

```text
Ahat = A / Aref
```

with an explicit region-independent reference area.

Start:

```text
Aref = 256 vertices
m = 0.5
n = 1.0
```

Do not define the erosion scale from each region's local accumulation p95.

### 10.2 Two-receiver effective downstream state

For two receivers:

```text
Hrec = w1*H[r1] + w2*H[r2]
Lrec = w1*L1     + w2*L2
```

For a single receiver use it directly.

### 10.3 Implicit update

For `n = 1`:

```text
c =
    Kdt
    * Ahat^m
    / Lrec

Hnew =
    (Hold + c*Hrec)
    / (1+c)
```

This is stable and naturally prevents the giant explicit overshoots that previously required arbitrary per-step caps.

## 11. Automatic erosion-strength calibration

Do not guess `Kdt`.

At cycle 0 compute:

```text
q = Ahat^m / Lrec
```

on channel-candidate terrain.

Choose a desired response:

```text
target_c_p90 = 0.15
```

and derive:

```text
Kdt =
    target_c_p90
    / percentile(q[channel_candidates], 90)
```

Clamp within sane global limits.

Useful initial test levels:

```text
light  = target_c_p90 0.08
medium = target_c_p90 0.15
strong = target_c_p90 0.25
```

Do not run a giant parameter grid. On the main development frame, try medium first; if still visually weak, try strong.

Log:

```text
c median
c p75
c p90
c p95
c max
chosen Kdt
```

Healthy initial behavior should be roughly:

```text
median active-channel c: 0.02–0.08
p90:                    0.10–0.25
p95:                    usually <0.5
```

If `c` is ~0.001 everywhere, erosion will again be invisible.

## 12. Terrain-type-dependent erosion

Construct a smooth erosion-factor field from Stage-4 classifications.

Starting factors:

```text
mountain flank:     1.0
major valley/river: 1.1–1.3
rolling hills:      0.6–0.9
flat lowland:       0.2–0.5
plateau interior:   0.1–0.3
plateau scarp:      0.8–1.1
coastal land:       0.3–0.6
underwater:         0.0
```

Smooth this factor field over ~8–16 vertices. No class boundaries should appear as visible edges.

## 13. Channel selectivity

Do not apply full fluvial incision to every vertex.

Build a smooth accumulation-based channel activation such as:

```text
channel_strength =
    smootherstep(
        (logA - logA_start)
        / (logA_full - logA_start)
    )
```

This produces weak/no fluvial incision on tiny catchments and full incision on established drainage.

Hillslope processes handle the diffuse terrain.

## 14. Hillslope transport

The current neighbor-mean relaxation may remain temporarily but must not immediately erase weak channels.

First production behavior:

```text
run every 1–2 cycles
low strength
reduced on plateau interiors
reduced on intentional scarps
```

If necessary replace it with a reusable implicit diffusion solve:

```text
(I - Ddt L) Hnext = Hcurrent
```

The matrix topology is stable, so build one AMG hierarchy and reuse it with changing RHS. Do not rebuild AMG every cycle.

## 15. Geomorphic cycle schedule

Start with:

```text
cycles = 24
reroute_every = 2
snapshots = [0, 4, 8, 16, 24]
```

Once stable, `reroute_every = 4` may be sufficient.

Do not use 100–250 cycles to compensate for bad scaling.

## 16. Erosion-delta diagnostics

For editable generated terrain compute:

```text
delta = Heroded - Hcycle0
```

Report:

```text
RMS |delta|
median |delta|
p75 |delta|
p90 |delta|
p95 |delta|
p99 |delta|
max incision
max positive change from hillslope transport
p95 |delta| / p95 local relief
```

A useful run should not finish with `p95 |delta|` of only a few GU after 16–24 cycles.

Mountain channels should commonly accumulate hundreds of GU of total change if the terrain scale warrants it, while flatlands should change much less.

## 17. TR-style geomorphic matching diagnostics

Compare owner reference and generated terrain using:

```text
slope p50 / p75 / p90 / p95
local relief at 8 / 24 / 64-vertex scales
drainage density
channel-head density
accumulation distribution
ridge-line density
valley-line density
profile-curvature distribution
plan-curvature distribution
orientation coherence
```

Use these as tuning guidance, not as one automatic scalar objective.

Visual landform coherence remains authoritative.

## 18. Final seam lock

After structural continuation and erosion, apply the existing narrow final seam reconciliation.

Do not redesign it unless erosion exposes a new hard failure.

Target width remains roughly 1–2 generated cells.

The final lock must:

```text
preserve the eroded field outside the narrow band
match authoritative seam heights exactly
avoid creating a new one-cell-in secondary cliff
```

If it must repair several cells of terrain, the earlier structural/erosion stages failed.

## 19. Performance requirements

### 19.1 Static owner work

Do once and cache:

```text
owner Gaussian pyramid
structure tensor
ridge/valley/plateau maps
owner routing
owner accumulation
seam inflow
```

### 19.2 Repeated cycles

Do **not** reroute static owner terrain.

Erosion cycles should operate primarily on generated terrain.

### 19.3 AMG reuse

For structural screened-Poisson and optional implicit hillslope solves, reuse matrix topology and AMG hierarchy wherever only RHS changes.

### 19.4 Snapshot rendering

Do not copy the entire global field for every cycle snapshot.

Render from the local working window plus local owner overlay and world-coordinate metadata.

### 19.5 Development outputs

Do not compress/write an entire global NPZ every iteration. Save local field + window coordinates + metrics for regional review runs.

### 19.6 Data types

Prefer:

```text
float32: terrain, guides, weights, feature rasters
int32: receiver indices, topological order
float64: only solver/accumulation operations that require it
```

## 20. Development progression before the final review batch

### Step A — current TR/Vvardenfell development frame

Implement Stages 4–8 here first.

Do not move on until:

```text
cycle 0 and cycle 16/24 are visibly different
without looking like destructive noise
```

Also require:

```text
major ridges/valleys continue plausibly
missing-cell terrain no longer remains unnaturally featureless where surrounding authoritative terrain is structured
no crosshatch
no obvious periodic routing pattern
no massive new pits
no gross plateau destruction
```

### Step B — three-archetype validation

Before the final batch, test:

```text
1. TR high-relief mountain seam
2. river / valley or plateau seam
3. low-relief / coast / missing-cell-heavy seam
```

If one global terrain-adaptive system behaves acceptably on all three, proceed.

Do not create per-region hacks.

## 21. Final ~10-region overnight review set

Once the system is functioning, run roughly ten seam-heavy review windows across TR, Skyrim, and Cyrodiil.

Recommended composition:

### TR — 4

1. current Vvardenfell high-relief mountain wall;
2. second mountain/ridge seam with a different orientation;
3. river/valley or plateau/scarp seam;
4. coast / lowland / missing-cell-heavy seam.

### Skyrim — 3

5. high-relief mountain seam;
6. rolling upland / valley / plateau-like seam;
7. coast or low-relief transition.

### Cyrodiil — 3

8. mountain / foothill seam;
9. lowland / river / rolling terrain seam;
10. coast or mixed-relief seam.

Do not choose ten similar mountain cases.

## 22. Automatic region selection

Use the seam atlas and owner codes.

For each owner source, rank candidate clusters by:

```text
seam mismatch
local relief
terrain class
valley/river presence
plateau confidence
coastal proximity
```

Avoid overlapping review windows.

Prefer windows roughly 10–20 cells across, with enough owner terrain visible to judge style directly.

## 23. Review artifact format

For every final region produce one standardized comparison sheet:

```text
A. owner/reference + original target context
B. accepted Stage-3 base
C. structural continuation before erosion
D. erosion cycle 0
E. erosion cycle 8 or 12
F. final cycle 24
G. post-erosion seam lock
H. optional drainage / feature overlay
```

Also save:

```text
metrics JSON
timings JSON
local final height field
```

## 24. Overnight iteration policy

Do not stop after the first ten renders just because code ran successfully.

Inspect results and continue iterating while failures are systematic and actionable.

Allowed autonomous tuning:

```text
Kdt response target
channel threshold
hillslope strength
terrain-class erosion factors
structural guide weights
guide continuation lengths
routing refresh interval
```

Do not independently tune each region.

Do not replace the architecture with a new unrelated generator unless it fails fundamentally.

## 25. Failure handling

### Erosion still invisible

Inspect `c` distribution and p95 height delta.

Increase `target_c_p90`, not arbitrary cycle count.

### Erosion uniformly lowers terrain

Increase channel selectivity and reduce low-accumulation fluvial effect.

### Grid/crosshatch patterns

Inspect router. Move to the two-receiver D∞/continuous-angle graph if the fallback MFD is still active.

### Plateau destroyed

Reduce plateau-top erosion factor; preserve plateau guide; allow scarp/outlet erosion separately.

### Ridge dies at seam

This is a structural-continuation failure, not an erosion-strength failure.

### River changes character at seam

Inspect owner accumulation crossing, inflow injection, and valley guide.

### Missing-cell region remains bland

Verify Stage-4 analysis uses all authoritative boundaries around it and Stage-5 guides are allowed through it.

Do not restore nearest-edge fill.

### Final lock reintroduces seam

Compare pre-lock vs post-lock. If pre-lock is good, lock width/constraints are wrong. If pre-lock is bad, the lock is not the root cause.

## 26. Suggested config additions

```json
{
  "structure": {
    "owner_analysis_halo_cells": 8,
    "gaussian_scales_verts": [8, 24, 64],
    "tensor_sigma_verts": 10,
    "ridge_percentile": 88,
    "valley_percentile": 88,
    "guide_max_cells": 6,
    "massif_guide_max_cells": 8,
    "ridge_weight": 0.7,
    "valley_weight": 1.0,
    "plateau_top_weight": 0.3,
    "scarp_weight": 0.8
  },
  "hydrology": {
    "routing": "dinf_two_receiver",
    "owner_hydrology_halo_cells": 10,
    "reroute_every": 2,
    "routing_perturbation_gu": 6,
    "routing_perturbation_sigma_verts": 4,
    "carry_owner_inflow": true
  },
  "erosion": {
    "cycles": 24,
    "snapshot_cycles": [0, 4, 8, 16, 24],
    "area_reference_vertices": 256,
    "stream_power_m": 0.5,
    "stream_power_n": 1.0,
    "auto_calibrate_kdt": true,
    "target_c_p90": 0.15,
    "target_c_p90_max": 0.35,
    "channel_area_start_vertices": 32,
    "channel_area_full_vertices": 256,
    "terrain_factor_mountain": 1.0,
    "terrain_factor_valley": 1.2,
    "terrain_factor_hills": 0.75,
    "terrain_factor_flat": 0.35,
    "terrain_factor_plateau": 0.2,
    "terrain_factor_scarp": 0.9,
    "terrain_factor_coast": 0.45,
    "hillslope_enabled": true,
    "hillslope_strength": 0.02,
    "final_lock_cells": 2
  }
}
```

These are starting values only.

## 27. Stop conditions before world-wide erosion

Do **not** begin world-wide geomorphic enhancement until the ~10-region review demonstrates:

```text
mountains: major ridges/massifs continue
valleys: channels/thalwegs continue
plateaus: top + scarp remain coherent
flatland: no gratuitous over-incision
coasts: shoreline remains fixed
missing cells: no flat synthetic rectangles
all: no crosshatch, no tile boundaries, no severe seam artifacts
all: visible but plausible erosion change
```

Final C0 after the seam lock must remain exact in the working field.

## 28. After regional acceptance

Only then begin the separate world-wide enhancement stage.

That later stage should reuse the accepted erosion operator in overlapping large tiles, preserve authoritative landmass terrain as required, use broader hydrologic context, and enhance retained Tamriel away from seams.

Do not combine world-wide enhancement with seam-system development.

## 29. Required end-of-run deliverable

Continue through structural continuation, erosion implementation, local tuning, and the broader review batch until there is a credible candidate production system.

Then deliver:

```text
~10 standardized comparison renders
across TR / Sky / Cyr
```

plus one summary Markdown report containing:

```text
architecture implemented
final config
timings
erosion c-distribution
height-delta statistics
feature counts
owner inflow counts
per-region metrics
known residual defects
paths to all review images
```

Do not claim completion merely because scalar gates pass. The purpose of the overnight batch is to leave the user with a diverse, already-tuned visual review set rather than ten diagnostic first attempts.
