# 2026-08-18 infill and circulation tools

This guide documents the Phase 21 replacement chain:

`R5 wall_front_rows.json -> R10 spatial_roles.json -> R11 alley_infill.json -> R12 circulation_surfaces.json -> R13 city_layout.json`

The R10 tool expands doors from the pinned D-STAMP libraries, computes free
ground after real hull, road, wall-band, and water subtraction, retains the two
deliberate civic spaces, and exposes every substantial inner rear pocket as an
`alley_quarter`. R11 routes obstacle-aware trunks and side branches through real
frontage gaps. It then seats compact Markarth houses along the lanes and within
concave rear pockets, with each chosen primary door facing its physical access.
R12 creates the exact-width alley/open-space textures and explicit door spurs.
R13 rebuilds hull parcels and verifies the rendered access chain from every
primary door through surfaces and roads to a regional port. Each CLI writes a
source-resolution PNG beside its checkpoint and refuses a non-empty output
directory.

The accepted R5 roads, wall, gates, ports, and outer placements are inherited
unchanged. The old `build_town_row_access.py`, `build_town_circulation.py`, and
`build_town_city_layout.py` remain legacy Phase 21 attempts.

The current Falkreath design realizes one public market plaza, one alley-fed
front courtyard, and 21 rear-quarter domains. The dense authoritative run adds
54 inner buildings, 45 lane segments, and 54 door spurs. It does not infer
courtyards from leftover lawn and does not claim service access for a secondary
door unless a future stage assigns that door a real service alley.
