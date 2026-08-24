# 2026-08-18 townlayout circulation realization tools

`realize_townlayout_circulation.py` consumes the accepted R13 city layout,
seated stamp objects, Falkreath region palette, and survey-backed terrain
field. It writes `r13_circulation_realization_v1`, an intermediate product for
future LAND and mesh authoring.

Existing raw-VTEX-78 source-road tiles inside the city domain are emitted as a
separate erase layer using raw VTEX 33 / `T_Sky_TerrGrassRE_01` /
`Tx_Skyrim_grass_03.dds`. Road centerlines and every currently authored civic
surface use raw VTEX 78 / LTEX index 77 / `T_Hr_TerrRoadOH_01` /
`hr\\lnd\\hr_oh_road_01.dds`.
Broad civic polygons are LAND paint requests. Narrow alleys and door/rear
aprons are terrain-following polygon requests with sampled terrain Z vertices.
The source-road erase layer is clipped to the accepted city domain; source
road tiles outside it and areas outside explicit circulation geometry remain
source terrain. No ESP, LAND record, or mesh is authored by this stage.
