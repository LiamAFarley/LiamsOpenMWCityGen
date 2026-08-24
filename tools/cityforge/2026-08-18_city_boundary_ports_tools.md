# 2026-08-18 · City-boundary ports and complete arterials

`build_town_ports.py` now constructs the complete city-region boundary, retracts
Voronoi boundary patches that only nick an external road, and emits ports plus
the exact retained patch set. `build_town_arterials.py` consumes that patch set
and carries every red/yellow arterial through to its magenta boundary port.

Core implementation:

- `src/procgen/townlayout/ports.py` — city ring, shallow-incursion and shallow
  texture-contact retraction, true external ports.
- `src/procgen/townlayout/arterial_graph.py` — port-to-fabric connectors.
- `src/procgen/townlayout/arterial_routes.py` — connected main-road tree and
  complete port-to-meeting corridors.
- `src/procgen/townlayout/diagnostics.py` and `road_review.py` — R2/Stage A
  full-resolution review renders.

Walls and gates are deliberately absent from this phase.
