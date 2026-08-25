# 2026-08-25 Run A Request

## Objective

Implement Sol's Run A structural field replacement for the real
`tr_vvardenfell_wall` crop, then render only the four structural fields needed
for visual review in a fresh output directory.

## Scope

In scope:

- masked one-sided owner derivatives;
- multiscale owner/target bands (`sigma 8`, `24`, `64`);
- harmonic macro continuation and meso continuation;
- generated-corridor fine-detail attenuation;
- one real TR/Vvardenfell Run A render set.

Out of scope:

- hydrology flat resolution;
- routing-domain changes;
- erosion changes or erosion renders;
- final seam lock;
- Skyrim/Cyrodiil or ten-region batch renders;
- world-wide erosion.

## Review Gate

The four outputs must be written below a new run-specific directory and
visually inspected before any downstream stage is run:

- Stage-3 harmonic base;
- cleaned fine-detail field;
- macro continuation;
- macro plus meso continuation.

The generated side must be inspected for removal of the old isotropic slab and
repetitive Tamriel fine noise. The owner side must remain unchanged.
