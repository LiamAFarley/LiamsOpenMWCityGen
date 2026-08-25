# 2026-08-25 TR Primitive Checkpoint Plan

## Stage P0: Mode and Configuration

Add `structure_mode: "primitives"` and a `terrain_primitives` JSON section with
semantic resolution, bottom-right target fraction, plateau/scarp/canyon
parameters, reconciliation parameters, and output controls. Preserve the old
band implementation for baseline comparison, but the primitive mode must not
call it.

## Stage P1: Plateau Analysis

Add `src/procgen/terrain_primitives.py` with a shared primitive record and
plateau analysis at semantic resolution. Compute plateau score from elevation,
flatness, prominence, and scarp proximity; close/remove components; retain only
owner components contacting the production seam. Fit an affine H24 top plane
with bounded Huber IRLS and report RMS/p95/tilt. Emit classification data and
one classification render.

## Stage P2: Footprint and Scarp

Continue each selected plateau from its full seam seed using semantic Dijkstra
with height, slope, lateral-width, and direction costs. Build support
probability, signed-distance/scarp fields, and owner-derived scarp diagnostics.
Upsample support only after the semantic path is complete. Emit footprint/scarp
render and component/path metrics.

## Stage P3: Plateau Candidate

Construct a complete plateau candidate from fitted top surfaces and empirical
or monotonic fallback scarp profiles. Suppress old generated fine residual in
the primitive core; outside support retain Stage-3 background. Emit plateau-only
candidate render and diagnostics.

## Stage P4: Canyon Candidate

Extract strong owner valley components within or adjacent to selected plateaus.
Continue seam-crossing thalwegs with semantic Dijkstra favoring low compatible
terrain, incoming direction, low uphill cost, and the plateau support. Build
full-resolution centerline, monotonic longitudinal profile, and owner/fallback
cross-section. Subtract canyon relief from the plateau candidate. Emit plateau+
canyon render and path diagnostics.

## Stage P5: Reconciliation and Lock

Rasterize all plateau/canyon candidates into one weighted candidate field and
one spatial support field. Solve one symmetric edge-aware screened-Poisson
system with PyAMG, reducing conductance across explicit scarp normals. Keep
owner and seam/outer boundary values exact, then invoke the existing narrow
final seam lock only. Emit pre-lock/post-lock fields, comparison sheet,
grayscale diagnostics, and final metrics. Do not run erosion.

## Review Gates

Run `review-flash` after each stage. The reviewer is read-only and writes a
dated review file. Fix any blocking issue before the next stage; re-review a
stage after a substantive correction. Do not execute the real crop until P0–P5
source reviews pass.

## Stop Condition

After visual inspection of the real bottom-right TR checkpoint, stop. Do not
implement ridge/massif/hill/general-valley primitives, re-enable erosion, or
start the broad validation batch in this task.
