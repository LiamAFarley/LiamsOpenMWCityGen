# Tamriel Reworked Terrain Synthesis v3 — Implementation Plan

**Date:** 2026-08-24  
**Purpose:** Replace the current seam-generation experiments with a production-oriented terrain pipeline that:

- preserves exact LAND seam compatibility with owner mods;
- raises the broad relief of the retained `tamriel.esm` terrain, with high mountains reaching roughly **3× their original height** while low coastal terrain changes little and underwater terrain is unchanged;
- continues actual owner-mod landforms across borders rather than copying texture or adding statistically matched noise;
- creates coherent drainage, ridges, valleys, plateau edges, and hillslope structure through geomorphic processing;
- avoids grid-aligned crosshatching, tile/stamp artifacts, transition aprons, and one-cell-in secondary seams;
- adapts to mountains, plateaus, valleys, flatlands, hills, coasts, and oceans;
- is efficient enough to run automatically over all seam clusters;
- produces final terrain whose organization and fine-scale structure are much closer to TR than the current smooth/noisy Tamriel field.

This plan supersedes the current `solve_region_v2.py` design as the intended production direction. Parts of v2 remain useful, especially the owner skirt, exact seam constraints, `own_view`, and the general idea of continuing owner macrostructure inward.

---

# 1. Core design rule

The system should no longer be thought of as:

> smooth the mismatch → add TR-like roughness → erode it

Instead use:

> **improve Tamriel relief → infer what owner landforms actually cross the seam → continue those structures into Tamriel → solve a curvature-aware transition surface → establish real drainage topology → erode/weather the result → re-project the exact seam → serialize and verify**

TR should provide **boundary geometry and geomorphic intent**, not merely statistics.

The three problems must be handled separately:

1. **Macro relief shaping**
   - Make retained Tamriel terrain less uniformly low-relief.
   - Raise large mountains dramatically without magnifying underwater terrain or every small coastal bump.

2. **Seam geometry / landform continuation**
   - Guarantee exact owner heights at the seam.
   - Continue slopes, ridges, valleys, plateau surfaces, scarps, and shorelines in a geometrically coherent way.

3. **Geomorphic refinement**
   - Build drainage networks and realistic ridge/valley dissection.
   - Add detail through erosion and hillslope transport, not visible procedural noise.

Do not ask erosion to invent the entire landform from a harmonic surface.

---

# 2. Phase A — broad relief amplification before seam solving or erosion

## 2.1 Goal

The user wants the tallest retained-Tamriel mountains to reach approximately **3× their current elevation above sea level**, while:

- underwater terrain is **bit-identical / numerically unchanged**;
- terrain near sea level changes little;
- ordinary small-scale terrain detail is not multiplied 3×;
- the response ramps smoothly with elevation;
- there is no threshold ring or coastal discontinuity;
- erosion runs **after** this transform so drainage develops on the final large-scale relief rather than being stretched afterward.

A direct transform such as:

```text
H_new = H * gain(H)
```

is too crude because it multiplies every high-frequency bump, erosion groove, and local defect along with the mountain.

The preferred approach is:

> **nonlinearly remap only the low/mid-frequency terrain surface, then restore the original fine residual essentially unchanged.**

This gives the desired increase in mountain height without simply scaling every tiny feature.

---

## 2.2 Recommended mathematical transform

Let:

- `H` = original height field in GU;
- `S` = sea level in GU;
- `B` = broad/macro terrain surface;
- `F = H - B` = fine residual;
- `Gmax = 3.0` initially.

Construct:

```text
B = gaussian_blur(H, sigma_macro)
F = H - B
```

Use a Gaussian filter, never a box filter.

Suggested initial range:

```json
"sigma_macro_verts": 16
```

At 128 GU horizontal spacing this corresponds to a feature scale of roughly 2,000 GU before considering the wider Gaussian support. The correct value should be visually calibrated, but 12–24 vertices is a sensible starting range.

The response curve acts on the broad elevation above sea level:

```text
D = max(B - S, 0)
```

Choose two response elevations:

- `E0`: below this, essentially no amplification;
- `E1`: by this elevation, use the full `Gmax`.

Rather than baking the whole system around fixed GU thresholds, support both fixed and percentile-derived values.

Recommended first implementation:

```text
E0 = max(configured minimum, low positive-land percentile)
E1 = high percentile of B over retained land
```

Practical starting defaults:

```json
"ramp_start_gu": 1024,
"ramp_end_percentile": 95.0,
"max_gain": 3.0
```

Use **smootherstep**, not linear interpolation:

```text
t = clamp((D - E0) / (E1 - E0), 0, 1)
s = t^3 * (t * (t * 6 - 15) + 10)
gain = 1 + (Gmax - 1) * s
```

Then the broad-field elevation delta is:

```text
delta_macro = (gain - 1) * D
```

This means:

- at `D <= E0`, `gain = 1`;
- at `D >= E1`, `gain = 3`;
- the transition has zero first derivative at both ends;
- no hard elevation contour is introduced.

---

## 2.3 Preserve shorelines and underwater terrain exactly

Do **not** apply the macro delta directly everywhere.

Use the original unsmoothed `H` to define the ocean/shore protection gate:

```text
shore_t = clamp((H - S) / shore_protect_height, 0, 1)
shore_gate = smootherstep(shore_t)
```

Then:

```text
H_scaled = H + shore_gate * delta_macro
```

And enforce:

```text
if H <= S:
    H_scaled = H
```

exactly.

Suggested initial config:

```json
"shore_protect_height_gu": 768
```

This creates three desirable behaviors:

1. all original underwater values remain unchanged;
2. terrain immediately above sea level changes only slightly;
3. mountains farther inland receive the full response.

The coastline therefore does not move because of the mountain amplification pass.

---

## 2.4 Why the fine residual should remain near 1×

The transform above adds the macro-height delta to the original field:

```text
H_scaled = H + delta
```

instead of computing:

```text
H_scaled = R(B) + gain * F
```

Therefore the original fine component `F` is retained at approximately 1× amplitude.

This is intentional.

If the mountain is raised by 20,000 GU, a 100-GU small ridge should remain approximately a 100-GU ridge until the subsequent erosion/weathering stage decides otherwise. Multiplying every microfeature by 3 would produce exaggerated corrugation and make the old low-quality Tamriel detail more visible.

A later geomorphic pass should create improved detail.

---

## 2.5 Optional prominence modulation

Start with the elevation-response curve above because it directly matches the requested behavior and is easy to inspect.

However, it may excessively raise broad high plateaus because they are high above sea level even if they are not locally mountainous.

If that occurs, add **optional prominence modulation**, not a replacement algorithm.

Compute a larger-scale regional baseline:

```text
R = gaussian_blur(H, sigma_regional)
P = max(B - R, 0)
```

where:

```text
sigma_regional > sigma_macro
```

For example:

```json
"sigma_regional_verts": 64
```

Build a second smooth gate from `P`, then combine it with the elevation gate.

Example:

```text
mountain_weight =
    lerp(1.0 - prominence_strength,
         1.0,
         prominence_gate)

effective_gain =
    1 + (Gmax - 1) * elevation_gate * mountain_weight
```

With:

```json
"prominence_strength": 0.0
```

the first implementation is pure elevation-based scaling.

Only increase this if visual review shows that high plateaus are being raised too aggressively.

Do not complicate the first test unnecessarily.

---

## 2.6 Relief transform invariants and tests

Before integrating seam solving, write tests for the relief transform.

Required invariants:

1. **Underwater identity**
   ```text
   H <= sea_level  =>  H_scaled == H
   ```

2. **Monotonic response**
   - higher broad elevations never receive a smaller amplification factor than lower broad elevations within the ramp.

3. **No discontinuity at E0/E1**
   - gain and its first derivative are continuous.

4. **Shore protection**
   - points at sea level have zero displacement.
   - displacement grows smoothly above sea level.

5. **Fine-detail retention**
   - subtracting a suitable macro blur before and after the transform should show that high-frequency RMS changes only minimally.

6. **Configurable maximum**
   - `max_gain=1.0` returns the original terrain exactly.
   - `max_gain=3.0` approaches 3× broad elevation above sea at the top of the ramp.

7. **No NaN propagation**
   - process only valid terrain or perform nearest-valid fill solely for filtering, then restore the original validity mask.

---

# 3. Apply the relief pass before erosion

The production order must be:

```text
original tamriel
    ↓
relief response / mountain amplification
    ↓
seam synthesis against owner terrain
    ↓
hydrology + erosion + hillslope refinement
    ↓
exact seam projection
    ↓
VHGT serialization
```

Do **not** perform 2–3× amplification after erosion.

Post-erosion amplification would multiply:

- channel depths;
- cliff faces;
- erosion artifacts;
- local noise;
- any remaining routing/grid defects.

The erosion system must see the exaggerated mountains so it can create drainage appropriate to their final relief.

A very small final relief-restoration factor may later be useful if erosion consistently knocks peaks down too far, but that should be a minor correction, not the main 3× transform.

---

# 4. Phase B — refactor shared seam context before adding another solver

The current scripts duplicate context-building logic.

Before productionizing v3, extract shared functionality from:

- `solve_region_blend.py`
- `panel_region.py`
- `solve_region_v2.py`

into common modules.

Suggested structure:

```text
src/procgen/
    terrainfield.py
    terrain_relief.py
    terrain_blend.py
    terrain_features.py
    terrain_hydrology.py
    terrain_erosion.py
    terrain_metrics.py

tools/terrain/
    solve_region_v3.py
```

`solve_region_v3.py` should be a thin driver, not another monolithic experiment.

Common context should expose:

- `tam_view`
- `own_view`
- owner-valid mask
- retained-Tamriel mask
- exact seam vertex list
- owner halo/skirt
- seam-distance field
- shoreline / sea mask
- region bounds
- local style/terrain descriptors
- raster-to-world coordinate mapping

The `own_view` rule remains:

```text
own_view = where(isfinite(oth_w), oth_w, tam_w)
```

for analysis and height constraints.

---

# 5. Phase C — replace coarse harmonic interpolation with a curvature-aware constrained solve

## 5.1 Why `coarse_laplace(ds=8)` should not remain the production base

The current harmonic base has three fundamental problems:

1. Laplace interpolation guarantees value smoothness but does not explicitly match the **normal slope** of the owner surface at the seam.
2. Harmonic surfaces naturally form broad smooth aprons, which is exactly the visual failure seen around the current TR mountain boundary.
3. `ds=8` means the PDE is actually being solved on a coarse lattice roughly 1,024 GU apart, then reconstructed. Any residual block structure can seed the erosion router and appear as crosshatching or terracing.

Do not replace `ds=8` with `ds=4` and declare the problem solved.

The production solver should preserve the efficiency benefits of a coarse hierarchy while refining the actual solution at full resolution.

---

## 5.2 Preferred surface objective

Solve a screened thin-plate / gradient-constrained surface.

Conceptually minimize:

```text
E(H) =
    Σ wt(x) * (H(x) - Htarget(x))²
  + λg Σ wg(x) * |∇H(x) - Gowner_ext(x)|²
  + λc Σ |ΔH(x)|²
```

subject to:

```text
H(seam_vertex) = owner_height
```

Where:

- `Htarget` is the relief-amplified Tamriel target;
- `Gowner_ext` is a smoothly extended owner gradient field;
- `wt` increases with distance from the seam so the solution returns to Tamriel naturally;
- `wg` is strongest near owner features and decays inward;
- `ΔH` penalizes gratuitous curvature / bending.

This is much closer to the real requirement than a pure Laplace solve.

The seam should be approximately **C1**, not merely C0:

- exact height;
- compatible normal slope;
- no visible crease.

---

## 5.3 Exact boundary constraints

At the owner seam:

```text
H = own_view
```

must be hard Dirichlet constraints.

Also estimate owner-side normal derivatives using valid owner vertices in a halo outside the seam.

Use at least 1 cell of real owner terrain for hard context, preferably 2–4 cells where available for feature estimation.

The boundary slope should enter the objective as a strong gradient constraint.

Do not rely on the mere presence of an owner skirt to make the solver infer the correct derivative automatically.

---

## 5.4 Multigrid / hierarchical solve

Avoid:

```text
block-average → solve once → cubic zoom
```

Use a hierarchy:

1. Gaussian restrict the target and masks;
2. solve the broad system at coarse resolution;
3. prolongate;
4. solve/refine residuals at the next resolution;
5. finish with full-resolution relaxation or sparse solve.

This preserves efficiency without permanently embedding the coarse lattice in the terrain.

A practical first implementation can use:

- SciPy sparse matrices;
- conjugate gradient for symmetric systems;
- LSMR/LSQR if implemented as least-squares rows;
- optional algebraic multigrid if already available in the environment.

Do not add a new heavyweight dependency solely for this unless necessary.

---

# 6. Phase D — multiscale owner-landform continuation

## 6.1 Remove nearest-seam-point macro copying

Do not use:

```text
for each interior vertex:
    find nearest seam point
    copy owner low-pass value
```

as the main continuation mechanism.

The identity of the nearest seam point changes along hidden Voronoi boundaries. That creates derivative discontinuities and ridge aliasing inside irregular border shapes.

Erosion can then amplify those invisible continuation boundaries.

---

## 6.2 Continue frequency bands by PDE, with different penetration distances

Build a Gaussian/Laplacian pyramid of `own_view`.

Suggested conceptual bands:

```text
very broad massif / plateau:
    sigma ~ 32–64 verts

ridge / valley structure:
    sigma ~ 8–32 verts

small geomorphic relief:
    sigma ~ 2–8 verts

microdetail:
    <2 verts
```

Do not treat these as exact fixed bins yet; they are implementation guidance.

Continuation distance should decrease with frequency.

Example initial behavior:

| Structure | Typical continuation |
|---|---:|
| massif / regional slope | 6–10 cells |
| major ridges / valleys | 4–8 cells |
| small ridges / channels | 1–4 cells |
| microtexture | do not copy |

Use a screened Poisson/biharmonic extension for each relevant band, not patch cloning.

The continuation objective should gradually hand control back to the amplified Tamriel target with distance.

This means the actual TR mountain spur that intersects the seam continues inward, but it naturally fades into the existing Tamriel terrain rather than becoming a pasted TR-texture strip.

---

## 6.3 Structure-tensor field

For medium-scale owner terrain, compute a local 2D structure tensor or equivalent orientation field.

Store:

- dominant terrain orientation;
- coherence / anisotropy;
- local slope;
- curvature.

Extend the orientation/coherence field a short distance into Tamriel with a smooth PDE.

Use it as a **constraint / weighting field**, not as a texture synthesizer.

Where coherence is high, the seam solver and feature continuation should prefer preserving the owner ridge/valley direction.

Where coherence is low, allow the target terrain more freedom.

This directly attacks the failure mode where a NE-running owner ridge becomes isotropic roughness or a differently oriented synthetic flank.

---

# 7. Phase E — explicitly identify ridge, valley, plateau, and shoreline crossings

Erosion should not be responsible for guessing every large landform.

For each seam cluster, identify strong semantic features that actually intersect the seam.

---

## 7.1 Valley / thalweg crossings

Use owner-side hydrology plus curvature to identify valleys.

A strong valley crossing should record:

- seam position;
- incoming direction;
- owner contributing area;
- valley depth / local relief;
- approximate width;
- downstream elevation.

These become constraints in the generated region.

The thalweg should continue smoothly into Tamriel before erosion begins.

---

## 7.2 Ridge crossings

Use Hessian/profile-curvature analysis and local maxima transverse to the dominant slope to detect meaningful ridge crests.

Record:

- seam position;
- tangent direction;
- ridge prominence;
- local width;
- relative height above neighboring valleys.

Continue major ridges as soft constraints.

Do not create a dense skeleton of every 1-vertex wiggle. Only continue features that are structurally significant at the owner boundary.

---

## 7.3 Practical first implementation of line continuation

Avoid building a large procedural ridge generator.

For each strong crossing:

1. obtain the tangent direction from the owner field;
2. extrapolate a short cubic/Hermite path into the target region;
3. gently curve the path toward compatible extrema in the amplified Tamriel field;
4. rasterize the path as a soft height/gradient constraint;
5. let the constrained surface solver integrate it into the full terrain.

This is enough for the first production-quality test.

Later, geodesic or optimization-based continuation can replace straight/Hermite extrapolation if necessary.

---

## 7.4 Plateau handling

Plateaus should not be treated as noisy mountains.

Detect plateau-like owner terrain using:

- low local slope across a broad area;
- high regional elevation;
- a concentrated scarp / curvature band at the edge.

Continue separately:

1. plateau top elevation / gentle regional tilt;
2. plateau scarp location and direction.

Then allow erosion primarily on the scarp and drainage outlets rather than dissecting the entire top uniformly.

---

## 7.5 Shoreline handling

Shoreline intersections are semantic constraints.

Preserve:

- sea-level crossing location;
- coast tangent;
- local coastal slope;
- major river-mouth positions.

Do not allow mountain scaling to move the shoreline.

Do not apply ordinary fluvial erosion below sea level.

Bathymetric blending, if needed, should be a separate smooth surface problem.

---

# 8. Phase F — terrain-adaptive blend width

Do not use one uniform seam width.

A broad uniform carry band creates the visible “synthetic halo” around the owner territory.

Compute a local blend width from continuous terrain descriptors:

- local relief;
- owner/Tamriel slope mismatch;
- curvature mismatch;
- ridge/valley crossing strength;
- plateau confidence;
- distance to sea;
- owner feature scale.

Suggested initial bounds:

```json
"blend_width_min_cells": 2,
"blend_width_max_cells": 10
```

Qualitative behavior:

- flatland → narrow;
- rolling hills → moderate;
- major mountain wall → wide;
- valley → elongated along the valley direction rather than a huge radial band;
- plateau → enough room to continue top + scarp;
- coastline → narrow geometric correction except around river mouths / cliffs.

Use smooth spatial weights. Avoid hard terrain-class borders that can themselves become visible.

The existing seam classes may still be useful diagnostics, but they should not be the sole control system.

---

# 9. Phase G — replace D8 + jitter with grid-bias-resistant hydrology

## 9.1 Remove D8 tie-breaking noise

Do not use:

```text
D8 + epsilon jitter
```

on nearly smooth terrain.

That is a direct route to grid-aligned or diagonal crosshatching.

Preferred router:

> **D∞ / continuous-direction flow**, splitting flow between the two neighbors bracketing the local flow vector.

A multi-flow-direction router is an acceptable temporary fallback if robust D∞ implementation becomes a schedule blocker, but production should aim for D∞.

---

## 9.2 Deterministic flat resolution

Plateaus and near-flat lowlands must not use random jitter to choose drainage.

Resolve flats deterministically:

1. identify connected flat regions;
2. locate valid lower outlets;
3. construct a weak auxiliary gradient toward the outlets;
4. construct a weak gradient away from higher surrounding terrain;
5. combine them to produce a drainage ordering.

This auxiliary gradient should affect routing only, not visibly modify the final terrain.

This is critical for plateau seams.

---

## 9.3 Depression handling

Before computing flow accumulation, remove artificial undrained pits.

Use a Priority-Flood-style depression analysis.

Prefer:

- breaching/carving small artificial pits when practical;
- filling only where necessary;
- no giant blanket fill that creates visible flat lakeshelves unless the terrain genuinely contains a basin.

Track which depressions were modified.

If a large natural closed basin exists, do not automatically destroy it simply to make the hydrology graph convenient.

---

# 10. Phase H — carry real owner drainage across the seam

Remove the current concept:

```text
+2000 accumulation at hundreds of border vertices
```

This gives many unrelated seam points equal artificial erosive power and encourages a curtain of similar channels.

Instead compute owner-side hydrology in a halo outside the seam.

For every genuine drainage path crossing into the generated area, carry:

- fractional contributing area;
- flow direction;
- local channel elevation;
- optionally estimated channel width / order.

Only those crossing points inject upstream accumulation.

A major owner river should enter as a major river.

A small gully should enter as a small gully.

An ordinary hillslope vertex should inject no special accumulation.

This is one of the highest-priority changes.

---

# 11. Phase I — replace capped explicit stream power with a normalized implicit formulation

## 11.1 Why the current SPL constants are problematic

Current v2 concept:

```text
k = 1.0
m = 0.8
cap = 12 GU / step
250 steps
accum cap = 20k
border boost = 2000
```

With raw accumulation counts, `A^0.8` becomes enormous very quickly.

The per-step 12-GU cap therefore becomes the real erosion law over much of the active drainage network.

Once most channels are capped, a medium and very large catchment erode nearly identically.

The code is no longer meaningfully behaving like stream-power erosion.

---

## 11.2 Normalize drainage area

Use:

```text
Ahat = A / Aref
```

where `Aref` is a configurable reference drainage area.

This can be represented in vertex-area units, which makes the parameterization resolution-aware and easier to reason about.

Start with:

```json
"stream_power_m": 0.5,
"stream_power_n": 1.0
```

Reasonable exploratory range:

```text
m = 0.4–0.6
n = 1.0 initially
```

Keep `n=1` for the first implementation because it enables a simple stable implicit receiver update.

---

## 11.3 Fastscape-style implicit receiver update

For a node `i` draining to receiver `r`, use a relation of the form:

```text
Hnew_i - Hold_i
    = -Kdt * Ahat_i^m * ((Hnew_i - Hnew_r) / Li)^n
```

For `n = 1`, define:

```text
c = Kdt * Ahat_i^m / Li
```

Then:

```text
Hnew_i = (Hold_i + c * Hnew_r) / (1 + c)
```

assuming nodes are processed in downstream-to-upstream order with receiver heights already available.

This is stable without an arbitrary 12-GU incision cap.

The user-facing parameter should be `Kdt` or a calibrated erosion strength, not an unexplained `k=1`.

---

## 11.4 Calibrate by dimensionless response rather than GU cap

Choose `Kdt` so the distribution of:

```text
c = Kdt * Ahat^m / L
```

lies in a useful range.

Initial target:

```text
median active-channel c: ~0.02–0.10
95th percentile c:      ~0.2–0.5
```

These are starting calibration targets, not sacred constants.

Log the `c` distribution for every solve.

If almost every channel has `c << 0.001`, nothing will happen.

If most large channels have `c >> 1`, the terrain will collapse toward receiver elevations too rapidly.

This is more interpretable than repeatedly tuning a GU-per-step cap.

---

## 11.5 Geomorphic cycles

Do not default to 250 explicit full-field steps.

Start with approximately:

```json
"geomorphic_cycles": 24
```

Per cycle:

1. depression/flat handling;
2. D∞ routing;
3. accumulation, including real owner inflow;
4. one implicit incision update;
5. hillslope transport;
6. restore hard constraints.

Refresh routing every cycle initially.

Once behavior is stable, profiling may show that every second cycle is sufficient in some low-change regions.

Add an optional convergence stop:

```text
stop if RMS terrain change < threshold
and drainage topology change < threshold
for N consecutive cycles
```

---

# 12. Phase J — hillslope transport after incision

Pure stream-power incision tends to create unnaturally sharp slots.

Alternate fluvial incision with hillslope transport.

Use an implicit or otherwise stability-controlled diffusion/creep step.

At minimum:

```text
(I - Ddt * Laplacian) Hnext = Heroded
```

with:

- owner seam fixed;
- underwater mask protected;
- optional reduced diffusion on intentional cliffs/scarps.

A later improvement can use nonlinear slope-limited transport with a critical slope.

The important point is:

> Do not use a fixed explicit `creep=0.05` without knowing the actual discretization stability condition.

Either compute and enforce the stability bound or use an implicit solve.

---

# 13. Phase K — invisible symmetry breaking, not visible noise synthesis

A perfectly smooth numerical surface can still contain routing degeneracies.

It is acceptable to add a **tiny correlated perturbation** before erosion solely to break symmetry.

This is not the old noise system.

Requirements:

- amplitude far below visible terrain relief;
- correlated over a few vertices;
- zero or strongly reduced near exact seam constraints;
- no target band-σ matching;
- never used as the final detail source.

Possible starting scale:

```text
amplitude ~0.1–0.5% of local relief
```

with an absolute cap such as a few to a few tens of GU.

The final surface should not visibly resemble the perturbation field.

If it does, the amplitude is too high or the erosion is too weak.

---

# 14. Phase L — erosion/style adaptation from real owner metrics

Do not attempt to recreate TR by forcing Gaussian-band σ values to match.

Those metrics remain useful descriptors but are insufficient targets.

Measure owner terrain in a halo and derive:

- slope quantiles;
- local-relief quantiles;
- profile/plan curvature distributions;
- drainage density;
- channel-head density;
- basin-area distribution;
- ridge-line density;
- valley-line density;
- structure-tensor coherence;
- orientation field;
- hypsometric distribution.

Use these to adjust:

- blend width;
- erosion strength;
- hillslope transport;
- feature-continuation distance;
- plateau protection;
- channel-head threshold.

The generated region should approach the **geomorphic organization** of the owner terrain, not merely its frequency spectrum.

---

# 15. Stronger seam-quality metrics

The old global metrics failed because a tiled/cloned field can have excellent global orientation or band statistics while being obviously incoherent locally.

Add the following production metrics.

## 15.1 Exact seam height error

```text
max_abs(H_generated_seam - H_owner_seam)
```

Must be zero after final serialization/decoding, not merely before export.

---

## 15.2 Normal-slope mismatch

For each seam vertex compare the one-sided normal derivative on:

- owner side;
- generated side.

Report:

- median;
- p90;
- p99;
- maximum.

This directly measures visible creases.

---

## 15.3 Curvature jump

Measure Laplacian/profile-curvature discontinuity across the seam.

Large localized curvature jumps indicate a hidden apron edge even when heights and slopes appear acceptable.

---

## 15.4 Local orientation coherence

Compute structure-tensor orientation and coherence in moving windows.

Do not compare only one global orientation histogram.

Compare matched windows across the border.

---

## 15.5 Ridge/thalweg continuation score

For significant owner ridge and valley lines that intersect the seam, report:

- fraction continued at least `N` vertices;
- initial angular error;
- maximum bend over the first `N` vertices;
- whether the generated feature terminates abruptly.

This should become one of the main visual-prediction metrics.

---

## 15.6 Hydrologic metrics

Compare owner and generated:

- drainage density;
- channel-head density;
- catchment-area distribution;
- branching-angle distribution if practical;
- fraction of unresolved sinks;
- fraction of seam rivers successfully continued.

---

# 16. Phase M — final seam projection

After all erosion and hillslope transport, enforce the seam again.

Never assume repeated Dirichlet enforcement during the geomorphic loop is sufficient.

Final procedure:

1. set every seam vertex to the exact owner height;
2. run a very narrow local slope-reconciliation solve on the generated side if needed;
3. do not move the hard seam values;
4. preserve owner-side normal slope as strongly as possible;
5. verify no one-cell-in secondary ridge was introduced.

This final correction band should be small. If it needs several cells, the main solver failed and should be fixed rather than hidden.

---

# 17. Phase N — serialize, reread, and verify VHGT

The actual invariant is not equality in the float NumPy working field.

The invariant is:

> **owner and generated LAND edges decode to exactly the same in-game vertex heights after VHGT serialization.**

Therefore Stage E must:

1. quantize/encode generated VHGT;
2. write a temporary output;
3. reread the generated LAND;
4. decode heights;
5. compare every owner-facing seam vertex;
6. fail the build on mismatch.

If the format quantization makes a working-field value impossible to encode exactly, snap the generated seam to the same representable value as the owner before final writing.

---

# 18. CELL ownership versus height ownership

Do not use a single ownership concept for Stage E.

Maintain separately:

```text
cell_owner
height_owner
```

Rules:

- config precedence still determines the winning owner;
- the winning mod owns the CELL semantics;
- the winning owner supplies LAND height only if it has usable VHGT;
- if the winning owner has a height-less LAND stub, use the Tamriel height fallback for geometry.

The current analysis rule:

```text
own_view = where(isfinite(oth_w), oth_w, tam_w)
```

is sound for terrain geometry.

But do not conclude that “no owner VHGT” automatically means “keep the whole Tamriel CELL record.”

CELL metadata and LAND height fallback are different concerns.

Before Stage E, explicitly test OpenMW behavior for:

- owner CELL + owner height-less LAND;
- generated/tamriel LAND fallback at same coordinate;
- absence of a later conflicting Tamriel CELL record.

The deletion/authoring logic should preserve owner-mod cell semantics while preventing terrain holes.

---

# 19. Ocean / bathymetry special case

Mountain scaling:

```text
H <= sea_level => no change
```

must remain absolute.

For seam synthesis:

- do not run normal fluvial incision below sea level;
- preserve coastline crossings;
- preserve river mouths;
- blend bathymetric slope separately if two underwater fields disagree;
- use a smooth/biharmonic underwater transition rather than “TR-style roughness”;
- prevent Priority-Flood from treating the ocean as an inland depression problem.

The coast should be an outlet boundary for terrestrial hydrology.

---

# 20. Runtime strategy

The target is not to perform expensive full-map erosion at full resolution blindly.

## 20.1 Relief transform

The broad relief transform can run globally or in large overlapping tiles because it only requires Gaussian filtering and simple pointwise math.

For tiled execution:

- use halos at least several Gaussian sigmas wide;
- crop halos after filtering;
- ensure identical values where tiles overlap.

---

## 20.2 Seam solving

Operate on seam clusters plus adaptive halos.

Do not allocate the entire 28,834-cell retained map for every seam.

Large adjacent seam clusters should be merged where their solve windows overlap materially.

---

## 20.3 Hydrology

Run hydrology only inside:

- generated seam corridor;
- necessary owner halo;
- sufficient Tamriel context for outlets.

If a basin exits the local window through the generated side, define a valid far-field outlet treatment rather than forcing it to terminate artificially.

---

## 20.4 Expected performance improvement

The intended production design should be faster than the current 149-s SPL experiment because it replaces:

- 250 explicit erosion steps

with roughly:

- 16–32 implicit geomorphic cycles;
- efficient sparse/hierarchical surface solves;
- region-local hydrology.

Profile each stage separately.

Required timing log:

```text
context build
feature extraction
surface solve
hydrology
incision
hillslope
final seam projection
render
```

---

# 21. Suggested config structure

Add a new config section rather than relying on in-code defaults.

Example:

```json
{
  "terrain_relief": {
    "enabled": true,
    "sea_level_gu": 0.0,
    "max_gain": 3.0,
    "sigma_macro_verts": 16.0,
    "ramp_start_gu": 1024.0,
    "ramp_end_percentile": 95.0,
    "shore_protect_height_gu": 768.0,
    "prominence_strength": 0.0,
    "sigma_regional_verts": 64.0
  },

  "solve": {
    "v3": {
      "blend_width_min_cells": 2.0,
      "blend_width_max_cells": 10.0,

      "surface": {
        "target_weight": 1.0,
        "gradient_weight": 1.0,
        "curvature_weight": 1.0,
        "owner_halo_cells": 3
      },

      "continuation": {
        "macro_cells": 8.0,
        "meso_cells": 5.0,
        "small_cells": 2.0
      },

      "hydrology": {
        "routing": "dinf",
        "resolve_flats": true,
        "depression_method": "priority_flood_breach",
        "carry_owner_accumulation": true
      },

      "erosion": {
        "cycles": 24,
        "m": 0.5,
        "n": 1.0,
        "area_reference_vertices": 256.0,
        "kdt": 0.05,
        "reroute_every": 1,
        "symmetry_break_fraction_relief": 0.0025,
        "symmetry_break_cap_gu": 16.0
      },

      "hillslope": {
        "enabled": true,
        "implicit": true,
        "strength": 0.05
      }
    }
  }
}
```

These are **starting values**, not validated final constants.

The implementation must log the effective parameter distributions so they can be tuned from evidence.

---

# 22. Implementation milestones

Do not implement everything in one giant change and then judge only the final image.

Use the following checkpoints.

---

## Milestone 1 — relief response only

Implement:

- Gaussian macro decomposition;
- 3× smootherstep response;
- exact underwater protection;
- shore gate;
- full-map or representative-region before/after renders;
- detail-preservation metrics.

Review whether the basic global terrain relief now looks closer to the desired scale.

Do not add erosion yet.

---

## Milestone 2 — v3 surface solve, no erosion

Implement:

- shared context refactor;
- owner halo;
- exact seam Dirichlet constraints;
- owner normal-slope constraints;
- screened thin-plate / gradient-constrained solve;
- multigrid/hierarchical refinement;
- adaptive blend width.

Render identical seam crops.

Success criterion:

> the seam should already look structurally plausible while still smooth.

If the no-erosion surface still forms an obvious brown/smooth halo, do not proceed by hoping erosion will hide it.

Fix the geometry first.

---

## Milestone 3 — owner feature continuation

Implement:

- multiscale owner-band continuation;
- structure-tensor orientation field;
- strong ridge and thalweg crossings;
- plateau-edge constraints.

Render again **without erosion**.

Success criterion:

> major owner ridges, valleys, scarps, and slopes visibly continue into generated territory in sensible directions.

---

## Milestone 4 — new hydrology

Implement:

- flat resolution;
- Priority-Flood depression handling;
- D∞ routing;
- owner-side flow accumulation;
- real cross-seam upstream-area injection.

Render drainage overlays before erosion.

Success criterion:

> drainage topology looks coherent and not grid-aligned before any terrain is incised.

Do not proceed if the flow map itself contains checkerboard or diagonal routing artifacts.

---

## Milestone 5 — implicit incision

Implement:

- normalized area;
- `m≈0.5`;
- `n=1`;
- implicit receiver update;
- 16–32 geomorphic cycles;
- routing refresh;
- parameter-distribution logging.

Render every ~4 cycles for debugging.

Success criterion:

> channels organize existing landform structure rather than replacing it with uniformly rough texture.

---

## Milestone 6 — hillslope refinement

Add:

- stable implicit diffusion / creep;
- plateau/cliff masks;
- exact seam re-imposition.

Tune against owner slope and curvature distributions.

---

## Milestone 7 — seam-quality metrics and final correction

Add:

- C0 edge error;
- C1 normal-slope jump;
- curvature jump;
- ridge/thalweg continuation;
- local orientation coherence;
- drainage metrics;
- final narrow seam projection.

---

## Milestone 8 — Stage E authoring

Only after visual approval:

- implement CELL deletion / ownership logic;
- implement height-owner fallback;
- write generated LAND;
- reread;
- verify decoded VHGT seam equality;
- fail on mismatch.

---

# 23. Required visual review output

Every solver run should generate a standard comparison sheet with identical framing.

For each representative seam:

1. original Tamriel;
2. owner reference;
3. relief-scaled Tamriel target;
4. pre-erosion v3 surface;
5. drainage overlay;
6. final eroded surface;
7. slope map;
8. curvature map;
9. seam-error overlay.

At minimum retain the existing large seam-centered crop convention.

Representative regression regions should include:

- TR/Vvardenfell mountain wall;
- a flat or rolling TR border;
- a plateau/scarp border;
- a valley/river crossing;
- a coastline;
- an underwater seam;
- Sky_Main seam;
- Cyr_Main seam.

Do not optimize only against `tr_vvardenfell_wall`.

---

# 24. Explicit anti-patterns — do not reintroduce

The following approaches have already failed or are inconsistent with the new design.

Do not reintroduce them as “detail” layers:

- fBm/noise with matched band σ;
- random patch/stamp cloning;
- frequency-split texture transplant;
- box-filter band stacks;
- D8 + epsilon jitter;
- nearest-seam-point copying as the main macro continuation;
- raw `A^0.8` accumulation with a GU-per-step incision cap;
- equal artificial accumulation boosts at hundreds of seam vertices;
- coarse `ds=8` solve followed only by cubic upsampling;
- fading procedural detail to exactly zero at a solve boundary;
- visible random perturbation as the final terrain texture.

Noise is permitted only as an invisible routing symmetry breaker.

---

# 25. Acceptance criteria

A seam solve is acceptable only if all of the following are true.

## Geometry

- exact decoded VHGT seam match;
- no visible height crease;
- no one-cell-in secondary seam;
- no broad uniform transition apron;
- owner ridges/valleys that hit the border continue coherently.

## Terrain type

- mountains remain mountains;
- valleys remain drainage corridors;
- plateaus preserve top/scarp organization;
- flatlands do not become unnecessarily dissected;
- coastlines do not move;
- underwater terrain is not mountain-scaled.

## Detail

- no grid crosshatch;
- no stamp boundaries;
- no isotropic “fuzz” layer;
- local structure is organized around slopes and drainage;
- generated terrain approaches owner-style slope/curvature/drainage statistics.

## Stability

- no thousands of artificial below-sea pits;
- no unresolved routing sinks except intentional basins;
- no NaN propagation;
- no dependence on random seed for large landform placement.

## Performance

- representative seam solve fits the workspace runtime budget or has a clear production batching strategy;
- runtime scales primarily with seam-window area, not total world area.

---

# 26. Recommended first implementation sequence

The agent should proceed in this exact order:

1. **Implement `terrain_relief.py` first.**
   - Produce 1× / 2× / 3× response renders.
   - Verify underwater identity and shoreline preservation.
   - Use the 3× version as the new target for all subsequent seam work.

2. **Refactor common seam context.**

3. **Replace `coarse_laplace` with the constrained multiresolution surface solver.**
   - Hard height constraint.
   - Explicit owner slope constraint.
   - Curvature regularization.
   - No erosion.

4. **Replace nearest-point macro continuation with multiscale PDE continuation.**

5. **Add major ridge/thalweg/plateau crossing constraints.**

6. **Implement deterministic flat handling + D∞ routing.**

7. **Carry real owner contributing area across seam crossings.**

8. **Replace capped SPL with normalized implicit stream-power incision.**

9. **Add stable hillslope transport.**

10. **Add final seam projection and decoded-VHGT verification.**

11. **Only then generalize across all 435 seam clusters and implement Stage E authoring.**

The critical debugging principle is:

> **If the surface does not look plausible before erosion, do not tune erosion harder.**

The previous experiments repeatedly asked noise or erosion to compensate for a geometrically wrong base. v3 should make the large landform correct first, then let erosion add geomorphic organization and detail.

---

# 27. Immediate deliverables for the implementing agent

The first implementation pass should produce:

```text
src/procgen/terrain_relief.py
src/procgen/terrain_blend.py
src/procgen/terrain_features.py
src/procgen/terrain_hydrology.py
src/procgen/terrain_erosion.py
tools/terrain/solve_region_v3.py
```

plus config additions in:

```text
configs/tamriel_reworked_v1.json
```

and a review directory such as:

```text
output/mapdata/terrain/tamriel_reworked/solved/v3/
```

The first review packet should contain:

1. relief-response curve plot;
2. full-map or large-region 1×/2×/3× mountain-scaling comparisons;
3. `tr_vvardenfell_wall` target after relief amplification;
4. v3 no-erosion constrained surface;
5. owner feature overlay;
6. D∞ drainage overlay;
7. erosion-cycle snapshots;
8. final comparison against the owner reference;
9. metrics JSON;
10. stage timing JSON.

Do not begin mass authoring of the replacement ESM until these visual checkpoints are approved.
