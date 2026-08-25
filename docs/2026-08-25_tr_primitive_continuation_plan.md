# Tamriel Reworked v3 — Terrain-Primitive Continuation Plan
## Plateau/Canyon First Checkpoint, Then Ridges, Hills, Grasslands, and Cliffs

**Date:** 2026-08-25
**Audience:** implementing local agent
**Primary checkpoint:** the southern TR canyon/plateau/mesa region shown in the user-provided reference screenshot
**Repository state assumed:** current `master` after the failed Run A multiscale harmonic-band continuation (`c62c9cb1...` or later compatible state)

## 0. Objective

Replace the failed idea that one generic harmonic/multiscale field continuation can reconstruct every landform with a **terrain-primitive system** that explicitly represents the kinds of geography that must cross an ownership seam.

The system must eventually handle:

- canyon / plateau / mesa terrain;
- broad mountain massifs;
- mountain ridges and spurs;
- valleys and river corridors;
- bumpy / rolling hills;
- smooth grasslands / lowlands;
- cliffs / scarps;
- coastlines;
- missing-height regions that must be synthesized rather than copied flat.

The first checkpoint is deliberately narrower:

> **Extend the southern TR canyon/plateau terrain across the seam as recognizable plateau/mesa terrain with coherent canyon continuation, while improving rather than degrading seam connectivity.**

Do not run the broad ~10-region validation batch until this checkpoint passes visually.

## 1. Why the current Run A method is rejected

The current `terrain_structure.py` computes owner and generated Gaussian bands:

```text
macro = H24 - H64
meso  = H8  - H24
```

and harmonically propagates the owner-band boundary values into a generated corridor.

This method can smoothly propagate *scalar residual amplitude*, but it has no explicit representation of:

- a 2-D plateau footprint;
- a plateau top plane or gently warped top surface;
- a scarp boundary;
- a canyon centerline and width;
- a ridge centerline and ridge-body width;
- a cliff edge;
- a massif support region.

The latest real Run A confirmed the limitation: the code converged numerically, but the generated terrain remained a broad slab with stepped/rectilinear transition bands and did not acquire coherent TR landforms.

**Do not tune `macro_width_cells`, `meso_width_cells`, Gaussian scales, or harmonic-band strengths further as the primary structural solution.**

Retain the existing harmonic machinery only for:

1. low-frequency background / fallback interpolation;
2. final reconciliation between explicitly synthesized terrain primitives;
3. exact seam locking.

## 2. Core design principle

The new structural stage must separate three jobs:

```text
A. Landform inference
   "What object is crossing the seam?"

B. Landform synthesis
   "What geometry should that object have on the generated side?"

C. Reconciliation
   "How do all generated objects blend into the Stage-3 terrain without new seams?"
```

Do not ask erosion to perform A or B.

Erosion is applied only **after** the pre-erosion field already contains the correct macro landforms.

## 3. Shared terrain-primitive model

Create a common representation, for example:

```text
src/procgen/terrain_primitives.py
```

Suggested base data structure:

```python
@dataclass
class TerrainPrimitive:
    primitive_id: int
    kind: str
    confidence: float
    owner_component_id: int
    owner_bbox: tuple[int, int, int, int]
    seam_vertices: np.ndarray
    support_mask: np.ndarray | None
    target_height: np.ndarray | None
    target_weight: np.ndarray | None
    erosion_class: str
    diagnostics: dict
```

Use specialized payloads/classes for:

```text
PlateauPrimitive
CanyonPrimitive
RidgePrimitive
ValleyPrimitive
MassifPrimitive
CliffPrimitive
HillPrimitive
```

The important shared output is always:

```text
candidate height field H_k(x,y)
candidate confidence/support W_k(x,y)
```

over a local bbox.

These complete 2-D candidate surfaces are what the reconciliation solver consumes.

## 4. Shared mathematical composition

Let:

```text
H0 = accepted Stage-3 / inpainted / relief-scaled background terrain
Hk = candidate surface from primitive k
Wk = smooth nonnegative confidence/support of primitive k
```

Construct:

```text
W_total = Σ λ_k W_k
B       = Σ λ_k W_k H_k
```

Then solve the structural field with an **edge-aware screened Poisson system**:

\[
\left(L_g + \mathrm{diag}(W_{\rm total})\right)H
=
B + b_{\rm boundary}
\]

where `L_g` is a weighted graph Laplacian:

\[
(L_g H)_i
=
\sum_{j\in N(i)} g_{ij}(H_i-H_j)
\]

and:

```text
g_ij = 1 normally
g_ij < 1 across an intended scarp/cliff
```

The old method attempted to infer geometry from a PDE. The new method supplies plausible geometry **first**, and the PDE only reconciles overlapping candidate surfaces and connects them to the background.

Properties:

- sparse SPD system;
- compatible with existing PyAMG;
- no `A.T @ A`;
- no biharmonic ringing;
- primitive weights are spatially localized;
- owner terrain remains untouched;
- no hard mask edge except an intentional cliff/scarp.

## 5. Semantic analysis resolution and performance

Do not perform expensive semantic geometry at full 128-GU vertex resolution unless needed.

Use two resolutions:

```text
full terrain grid:
    1 sample per LAND vertex

semantic grid:
    1 sample per 4×4 full-resolution vertices initially
```

This reduces connected-component, contour, distance, path, region-growing, and primitive-fit work by ~16×.

Recommended workflow:

```text
full H8/H24/H64 analysis
    ↓
downsample semantic rasters by factor 4
    ↓
detect and continue primitive geometry on semantic grid
    ↓
upsample support / geometry
    ↓
sample/finalize height profiles on full-resolution grid
```

Do not downsample the final seam itself. All exact seam constraints remain full resolution.

# 6. First checkpoint: plateau / mesa / canyon system

The first implementation milestone must target the attached southern TR terrain.

This region contains broad elevated surfaces separated by steep scarps and canyon-like low corridors. It is an ideal falsification test because a harmonic field cannot reproduce it correctly.

## 6.1 Plateau candidate score

Reuse current owner products:

```text
H8
H24
H64
slope24
plateau_top_mask
scarp_mask
orientation/coherence
```

but replace the current simple plateau mask with component-oriented processing.

Define:

\[
S_{\rm plateau}
=
S_{\rm elev}
S_{\rm flat}
S_{\rm prominence}
\]

Example terms:

\[
S_{\rm elev}
=
\operatorname{smootherstep}
\left(
\frac{H_{64}-z_{\rm low}}
{z_{\rm high}-z_{\rm low}}
\right)
\]

\[
S_{\rm flat}
=
\exp
\left(
-\left(\frac{\|\nabla H_{24}\|}{s_0}\right)^2
\right)
\]

Prominence can use:

\[
R_{64}
=
H_{64} - G_{\sigma=128}(H)
\]

or another broader local baseline.

Do not classify a flat coastal plain as a plateau merely because its slope is low. Require positive prominence or a nearby scarp.

## 6.2 Connected plateau components

Threshold plateau score, then:

```text
binary close
remove small components
connected-component label
```

on the semantic grid.

For each component store:

```text
area
perimeter
mean H64
median H24
slope p50/p90
prominence
seam contact length
scarp contact length
```

Only components that intersect the owner seam need continuation.

## 6.3 Robust plateau top-surface fit

For each seam-crossing plateau component, fit a top surface from owner vertices inside the component.

Start with an affine plane:

\[
z_p(x,y)=a x+b y+c
\]

Fit with robust Huber IRLS:

\[
\min_{a,b,c}
\sum_i
w_i
\rho_\delta
\left(
z_i-(a x_i+b y_i+c)
\right)
\]

Use `H24` or `H64`, not raw fine terrain. Prefer `H24` for plateau tilt, with weights favoring high plateau confidence.

Report:

```text
fit RMS
fit p95 residual
tilt magnitude
```

If a large component has excessive affine residual, permit a regularized quadratic:

\[
z_p=
a x+b y+c+d x^2+e xy+f y^2
\]

with:

\[
E=
\sum_i \rho(z_i-z_p)
+
\lambda_q(d^2+e^2+f^2)
\]

The quadratic is for gentle warping only. It must not reproduce local canyon relief.

# 7. Scarp extraction and modeling

## 7.1 Scarp contour

Use:

```text
plateau component boundary
∩
dilated scarp mask
```

plus slope/curvature evidence.

Represent each contiguous scarp as a polyline.

At semantic resolution:

1. contour the plateau component;
2. attach local scarp confidence;
3. break contour into scarp and non-scarp arcs;
4. simplify with a small Ramer–Douglas–Peucker tolerance;
5. fit cubic B-spline / Catmull-Rom curves.

Store:

```text
position
tangent
outward normal
top height
low-side height
step height
scarp width
confidence
```

## 7.2 Empirical scarp cross-section

For each strong owner scarp sample:

1. sample a line normal to the scarp;
2. sample from plateau interior across the scarp into lower terrain;
3. normalize horizontal position by measured scarp width;
4. normalize vertical height:

\[
u=
\frac{H-z_{\rm low}}
{z_{\rm top}-z_{\rm low}}
\]

Aggregate multiple cross-sections from the same component and use a median binned profile.

Fallback:

\[
f(\xi)
=
\frac{1}{2}
\left[
1-\tanh\left(\frac{\xi}{\sigma}\right)
\right]
\]

or a monotonic smootherstep profile.

Store median width, width p25/p75, and step-height distribution.

# 8. Plateau footprint continuation

This is the most important new operation.

Do **not** continue plateau height as a scalar field. Continue the 2-D support region.

## 8.1 Seam seed

For each plateau component:

```text
S = plateau top vertices touching the seam
```

Convert to semantic-grid seed intervals/polylines.

Measure:

```text
seam contact width
local component thickness on owner side
local scarp positions at seam
component principal axes
```

## 8.2 Preferred continuation direction

At the seam define:

\[
v
=
\operatorname{normalize}
\left(
\alpha n_{\rm seam}
+
(1-\alpha)v_{\rm shape}
\right)
\]

where:

```text
n_seam  = inward owner→generated direction
v_shape = owner plateau principal / medial-axis direction
α       ≈ 0.5–0.8 initially
```

If component orientation is weak, use seam normal.

## 8.3 Generated-side compatibility cost

Build coarse travel cost:

\[
C(x)
=
1
+
\alpha_h C_h
+
\alpha_s C_s
+
\alpha_w C_w
+
\alpha_c C_c
\]

with:

\[
C_h
=
\min
\left(
\frac{|H_{0,64}(x)-z_p(x)|}{R_h},
c_{\max}
\right)
\]

\[
C_s
=
\min
\left(
\frac{\|\nabla H_{0,24}\|}{s_{\rm ref}},
c_{\max}
\right)
\]

`C_w` penalizes excessive lateral expansion relative to incoming plateau width.

`C_c` is a very large penalty for ocean/protected coastline unless the primitive is explicitly coastal.

## 8.4 Anisotropic fast marching / Dijkstra

Continue support from the entire seam seed.

For step direction `d`:

\[
c_{\rm step}
=
\frac{C(y)\|d\|}
{
\epsilon
+
\left[
\eta
+
(1-\eta)
\max(0,d\cdot v)^2
\right]
}
\]

Start with:

```text
η = 0.3–0.5
```

Travel along preferred continuation direction is cheaper than uncontrolled sideways expansion.

Stop when:

```text
geodesic budget exceeded
max continuation exceeded
ocean/protected mask reached
compatibility cost too high
support merges into compatible existing high terrain
```

Initial max distance:

```text
4–6 cells ordinary plateau
6–8 cells large canyon plateau
```

## 8.5 Support probability

Let `D(x)` be anisotropic geodesic distance.

Define longitudinal support:

\[
P(x)
=
1-
\operatorname{smootherstep}
\left(
\frac{D-D_{\rm core}}
{D_{\rm edge}-D_{\rm core}}
\right)
\]

Do not use this to blur the lateral scarp. The lateral scarp is explicit geometry.

# 9. Plateau footprint and signed distance

Threshold a plateau core:

```text
plateau_core = P >= p_core
```

At semantic resolution:

```text
morphological close tiny holes
remove tiny islands
preserve canyon exclusions
```

Upsample to full resolution.

Create signed distance:

\[
d_p(x)
=
\begin{cases}
+\operatorname{dist}(x,\partial P) & x \text{ inside plateau}\\
-\operatorname{dist}(x,\partial P) & x \text{ outside}
\end{cases}
\]

Run EDT only inside the primitive bbox plus margin.

# 10. Plateau candidate surface

Let:

```text
z_top(x) = fitted plateau top
H0(x)    = Stage-3 background
```

Estimate low-side reference from smoothed background outside the footprint.

Use empirical scarp profile with signed distance:

\[
t(x)
=
f
\left(
-\frac{d_p(x)}{w_{\rm scarp}(x)}
\right)
\]

with:

```text
t ≈ 1 inside plateau
t ≈ 0 outside
```

Candidate:

\[
H_{\rm plateau}(x)
=
t(x) z_{\rm top}(x)
+
[1-t(x)] H_{\rm low}(x)
\]

Inside the plateau core force height near `z_top` except where canyon primitives later subtract terrain.

Do not add old Tamriel high-frequency residual to the plateau top.

# 11. Canyon / thalweg extraction

The first checkpoint region is plateau terrain dissected by canyon-like low corridors.

Use:

```text
valley_score
valley_mask
owner accumulation
H8/H24 curvature
```

within or next to the plateau component.

Skeletonize only strong valley components.

For each seam-crossing canyon determine:

```text
thalweg polyline
incoming seam point
tangent
thalweg elevation profile
width profile
depth below plateau top
sidewall profile
owner accumulation
```

Minor tributaries not crossing the seam are left to later erosion.

# 12. Canyon centerline continuation

Continue canyon centerline with a path cost:

\[
C_{\rm valley}(x)
=
\beta_h C_{\rm low}
+
\beta_d C_{\rm direction}
+
\beta_u C_{\rm uphill}
+
\beta_p C_{\rm plateau}
\]

where:

```text
C_low       favors target low corridors
C_direction penalizes abrupt turn
C_uphill    strongly penalizes unrealistic rise
C_plateau   prefers remaining inside continued plateau/massif when appropriate
```

Use A* / Dijkstra on the semantic grid.

Allow the path to bend toward compatible target low ground.

Major river/canyon owner hydrology should have high authority.

# 13. Canyon longitudinal profile

Let seam thalweg height be `z0` and owner incoming gradient be `g0`.

Provisional:

\[
z_{\rm thalweg}(s)=z_0+g_0 s
\]

Then reconcile toward compatible target low terrain.

Enforce non-increasing downstream profile where hydrologically appropriate using isotonic regression / monotonic projection if needed.

# 14. Canyon cross-section

Estimate from owner cross-sections:

```text
half-width
depth
bottom width
wall shape
asymmetry
```

Fallback profile:

\[
q(r)
=
\begin{cases}
1 & |r| \le b\\
\left[
1-
\operatorname{smootherstep}
\left(
\frac{|r|-b}{w-b}
\right)
\right]^\gamma
& b<|r|<w\\
0 & |r|\ge w
\end{cases}
\]

where:

```text
b = half bottom width
w = half canyon influence width
γ = wall-shape exponent
```

Depth:

\[
D(s)=z_{\rm plateau}(s)-z_{\rm thalweg}(s)
\]

Candidate:

\[
H_{\rm canyon}(s,r)
=
H_{\rm plateau}(s,r)-D(s)q(r)
\]

For non-plateau river valleys use the macro background instead of plateau top.

# 15. First-checkpoint composition order

```text
1. Stage-3 background
2. plateau footprint + top + scarp
3. subtract continued canyon primitives
4. edge-aware structural reconciliation
5. narrow final seam lock
6. NO erosion yet
```

This field must already look like canyon/mesa terrain.

If it does not, do not hide the failure with erosion.

# 16. Edge-aware reconciliation

A standard Laplacian will blur intended scarps.

Define scarp confidence `S(x)` and scarp normal `n`.

For edge `i↔j`:

\[
s_{ij}=\frac{S_i+S_j}{2}
\]

\[
a_{ij}=|d_{ij}\cdot n|
\]

Use conductance:

\[
g_{ij}
=
g_{\min}
+
(1-g_{\min})
\exp(-\beta s_{ij}a_{ij})
\]

Starting values:

```text
g_min = 0.05–0.2
β     = 2–5
```

Across a strong scarp normal, smoothing is reduced. Along the scarp or in ordinary terrain, conductance remains high.

The matrix stays symmetric positive definite.

# 17. Plateau checkpoint acceptance

Before mountains/ridges:

## Visual

Generated side must show:

- recognizable broad plateau/mesa top;
- continued scarp geometry;
- major seam-crossing canyon/valley continuation;
- no rectangular harmonic corridor;
- no flat nearest-edge slab;
- no isolated Gaussian bumps;
- no stripe/ribbon artifacts;
- no seam visually worse than Stage-3 baseline.

## Semantic connectivity

For each plateau crossing:

```text
owner plateau seam-contact interval connects to generated plateau support
```

For each scarp:

```text
generated tangent mismatch <= ~25° initially
```

For each major canyon:

```text
thalweg endpoint error <= 2–4 full-res vertices
tangent mismatch <= ~25°
```

## Height continuity

After final lock:

```text
C0 = exact
no catastrophic first-edge jump
final-lock correction localized and small relative to local relief
```

Report final-lock correction:

```text
RMS
p90
p99
max
area >128 GU
area >512 GU
```

If final lock rewrites broad structure, primitive geometry failed.

# 18. Mountain massif / ridge system

After plateau passes, implement mountains as:

```text
massif support
+
ridge network
+
valley network
```

## 18.1 Massif support

Detect broad high-relief components using:

\[
R_{\rm macro}
=
H_{64}-G_{\sigma=128}(H)
\]

plus local relief.

Continue broad support using the same anisotropic geodesic-region method as plateaus, but do not flatten the top.

Candidate macro height:

\[
H_{\rm massif}=H_0+A(x)
\]

where `A(x)` is owner-derived macro prominence continued into generated terrain.

## 18.2 Ridge centerline

Extract high-confidence ridge systems only inside/near massif support.

Represent:

```text
centerline R(s)
crest height z_c(s)
half-width w(s)
prominence A(s)
cross-section f(r/w)
```

Continue seam-intersecting branches using cost favoring:

```text
incoming tangent
compatible target high ground
massif support
low curvature
```

## 18.3 Ridge body, not ribbon

For transverse distance `r`:

\[
H_{\rm ridge}(s,r)
=
H_{\rm base}
+
A(s)f(r/w(s))
\]

Fallback:

\[
f(q)=\exp(-q^p)
\]

with `p≈1.5–3`.

Use owner-derived cross-sections when enough data exists.

# 19. General valley / river primitive

Reuse canyon machinery without requiring plateau support.

Represent:

```text
thalweg
longitudinal profile
width
depth
cross-section
owner accumulation
```

Major rivers strongly constrain pre-erosion structure; erosion later adds tributaries and fine channel form.

# 20. Cliff / escarpment primitive

Detect coherent narrow high-gradient lines.

Represent:

```text
cliff polyline
high-side surface
low-side surface
step height
cross-section
```

Continue edge using incoming tangent + target compatibility + coastline where relevant.

Construct candidate height from signed distance to cliff.

Use low conductance across the cliff normal.

# 21. Rolling / bumpy hills

Do not treat rolling hills as noise.

Detect moderate-scale H24/H64 extrema.

Represent significant hills as broad elliptical mound/depression primitives:

\[
q^2
=
\left(\frac{x'}{a}\right)^2
+
\left(\frac{y'}{b}\right)^2
\]

\[
\Delta H=A\exp(-q^p)
\]

Store:

```text
center
principal axes
orientation
prominence
elliptical radii
profile exponent
```

Continue only hill supports that intersect the seam.

Farther from seam, preserve existing Tamriel hills until world-wide enhancement.

# 22. Smooth grassland / lowland

Smooth grassland is the absence of strong primitives.

For low-relief, low-feature-confidence terrain use:

```text
Stage-3 low-frequency background
minimal seam reconciliation
mild later drainage only
```

Do not inject TR roughness into smooth plains merely to match a global texture metric.

# 23. Primitive selection

Feature vector:

```text
local relief H8/H24/H64
slope
ridge confidence
valley confidence
plateau confidence
scarp confidence
massif confidence
coastal flag
owner accumulation
orientation coherence
```

Rules:

```text
plateau high + scarp high:
    plateau/scarp

massif high + ridge high:
    massif + ridge

valley high + accumulation high:
    river/valley

scarp high without plateau:
    cliff

moderate relief + isolated extrema:
    hills

everything low:
    grassland/background
```

Compatible primitives can overlap:

```text
plateau + canyon
massif + ridge + valley
coastal plateau + cliff
```

# 24. Primitive overlap order

Recommended conceptual order:

```text
background
    ↓
massif / plateau support
    ↓
ridge positive relief
    ↓
valley/canyon subtractive relief
    ↓
scarp/cliff edge enforcement
    ↓
screened-Poisson reconciliation
```

Do not blindly average incompatible candidates.

# 25. Fine-detail strategy

Within high primitive confidence, attenuate old Tamriel fine residual:

```text
Hfine = H0 - Gaussian(H0, sigma=4)
```

Suggested:

```text
primitive core fine_keep = 0.0–0.2
transition → 1.0 with smootherstep
```

Fine detail is recreated by:

```text
owner-derived cross-sections
erosion
hillslope weathering
minor drainage
optional later small-amplitude residual synthesis
```

Do not use patch/stamp copying.

# 26. Semantic seam continuity metrics

Add metrics beyond C0/C1.

## Plateau

```text
owner contact length
generated contact length
contact overlap
scarp tangent mismatch
top-plane mismatch
```

## Ridge

```text
crest position error
tangent mismatch
crest-height mismatch
width ratio
```

## Valley

```text
thalweg position error
tangent mismatch
bed-height mismatch
width ratio
owner accumulation transferred
```

## Cliff

```text
edge position error
tangent mismatch
step-height mismatch
high/low-side orientation
```

These metrics detect "height matches but landform stops."

# 27. Exact seam policy

Owner terrain remains immutable.

After synthesis/reconciliation:

1. narrow final harmonic seam lock;
2. exact owner seam heights;
3. lock limited to ~1–2 generated cells;
4. do not use lock to manufacture missing macro structure.

# 28. Performance architecture

## 28.1 Local bboxes

Each primitive owns a bbox + support margin.

Do not allocate full review-window arrays for each primitive.

## 28.2 Coarse semantic grid

Component analysis, contour extraction, region growth, and pathfinding at 4× reduced resolution.

## 28.3 Local EDT

Run signed-distance EDT inside primitive bboxes only.

## 28.4 One structural AMG solve

Do not solve one AMG system per primitive.

Rasterize all candidates into:

```text
B
W_total
edge conductance g
```

then solve one system for the whole local region.

## 28.5 Cache owner analysis

Cache by source/window/config hash:

```text
H8/H24/H64
plateau components
scarp contours
ridge network
valley network
massif masks
```

## 28.6 No global render copies

Render local windows directly.

# 29. Development order

## P0 — preserve baseline

Keep current Stage-3 and failed Run A artifacts.

Add config switch:

```json
"structure_mode": "primitives"
```

Do not delete old band code yet.

## P1 — plateau analysis

Implement:

```text
plateau score/components
top-surface fit
scarp contour
scarp cross-sections
```

Render classification only.

## P2 — footprint continuation

Implement:

```text
seam seed
anisotropic geodesic continuation
support
signed-distance
```

Render footprint/scarp only.

## P3 — plateau/scarp height synthesis

Render plateau candidate without canyon.

It must look like a plateau, not a ramp.

## P4 — canyon continuation

Implement thalweg/path/profile/cross-section and compose with plateau.

This is the first major visual gate.

## P5 — structural reconciliation

Rasterize candidates and run one edge-aware AMG solve.

Apply narrow final seam lock.

Stop for user review.

# 30. First checkpoint render set

Produce identical framing:

```text
1. owner/reference + Stage-3 context
2. plateau component classification
3. continued footprint + scarp overlay
4. plateau-only candidate
5. plateau + canyon candidate
6. reconciled pre-lock field
7. post-lock field
8. comparison sheet
```

Also one contour/grayscale diagnostic of top surface and scarp.

Do **not** run erosion for this checkpoint.

# 31. First checkpoint stop gate

Do not implement ridge/mountain continuation until plateau/canyon:

- visibly extends mesa/plateau structure;
- improves or preserves seam connectivity;
- creates no new corridor boundary;
- preserves scarp character rather than smoothing it into a ramp;
- suppresses old Tamriel repetitive noise over the new plateau;
- requires only narrow/small final seam correction.

If this fails, fix primitive geometry. Do not use erosion to hide it.

# 32. Mountain checkpoint

After plateau passes:

1. massif support;
2. ridge-body continuation;
3. valley primitive;
4. test on Vvardenfell wall;
5. require clear structural improvement without ribbon scars.

Only then re-enable erosion.

# 33. Erosion role after primitive synthesis

Erosion runs on the primitive structural field.

Responsibilities:

```text
tributary formation
channel hierarchy
ridge sharpening by valley incision
hillslope weathering
small-scale geomorphic organization
```

It must not decide whether a plateau exists.

Semantic erosion factors:

```text
plateau interior: weak
scarp: moderate
massif: normal/high
major valley: high channel authority
grassland: weak
underwater: zero
```

# 34. Broader validation only after structural gates

After plateau and mountain checkpoints pass and erosion visibly improves both, run the ~10-region set across TR/Sky/Cyr.

Select by primitive class, not random seams:

```text
TR:
    plateau/canyon
    Vvardenfell mountain wall
    second ridge/valley
    coast/lowland

Skyrim:
    mountain
    rolling terrain
    coast/valley

Cyrodiil:
    plateau/hill
    valley/lowland
    mixed relief
```

# 35. Suggested config

```json
{
  "terrain_primitives": {
    "enabled": true,
    "semantic_downsample": 4,

    "plateau": {
      "min_confidence": 0.55,
      "min_component_vertices_semantic": 24,
      "max_continuation_cells": 8,
      "ordinary_continuation_cells": 5,
      "direction_seam_weight": 0.65,
      "geodesic_lateral_eta": 0.4,
      "top_fit": "robust_affine",
      "allow_quadratic_fit": true,
      "quadratic_regularization": 0.01,
      "support_core_threshold": 0.65,
      "weight": 1.0
    },

    "scarp": {
      "profile_samples_each_side_verts": 32,
      "profile_bins": 32,
      "fallback_width_verts": 8,
      "conductance_min": 0.1,
      "conductance_beta": 3.0,
      "weight": 1.5
    },

    "canyon": {
      "min_confidence": 0.55,
      "max_continuation_cells": 8,
      "direction_penalty": 1.0,
      "uphill_penalty": 4.0,
      "target_low_penalty": 1.5,
      "weight": 1.2
    },

    "ridge": {
      "max_continuation_cells": 8,
      "weight": 1.0
    },

    "massif": {
      "max_continuation_cells": 10,
      "weight": 0.8
    },

    "hill": {
      "max_continuation_cells": 4,
      "weight": 0.5
    },

    "reconciliation": {
      "linear_solver": "amg_rs_cg",
      "cg_tol": 1e-6,
      "cg_maxiter": 200,
      "amg_max_coarse": 500
    }
  }
}
```

Starting values only.

# 36. Prohibitions

Do not:

- revive raw profile mirroring;
- extend one height profile per seam pixel;
- use nearest-edge copying as structural generation;
- use patch/stamp cloning;
- use harmonic macro/meso continuation as the primary landform generator;
- use erosion to create a missing plateau;
- blur failure until seams disappear;
- clamp terrain to hide overshoot;
- alter owner terrain;
- run the broad validation batch before plateau/canyon passes.

# 37. Required deliverable for this milestone

Implement through **P5**, then stop for user visual review.

Deliver:

```text
southern TR plateau/canyon comparison sheet
all checkpoint renders
primitive metrics JSON
timings JSON
plateau component diagnostics
scarp profile diagnostics
canyon path diagnostics
final seam metrics
final-lock delta metrics
```

Write a handoff Markdown stating:

```text
what plateau components were detected
which crossed the seam
how top surfaces were fit
how footprints were continued
how many scarps/canyons were continued
whether seam connectivity improved
what remains visually wrong
```

Do not report PASS solely because AMG converged.

The checkpoint passes only if the generated side visibly contains coherent canyon/plateau terrain and the seam is no worse than the Stage-3 baseline.
