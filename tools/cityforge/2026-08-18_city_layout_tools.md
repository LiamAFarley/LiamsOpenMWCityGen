# 2026-08-18 Final city-layout tool

`build_town_city_layout.py` consumes R7 circulation data and writes the R8
`city_layout.json` checkpoint plus a full-town terrain diagnostic. It derives
ownership parcels, door aprons, final frontage coverage, and exact access-graph
reachability from every occupied building to the regional road ports.

