# 2026-08-18 Row-access tool

`build_town_row_access.py` consumes an R5 wall-front-row checkpoint, classifies
residual block interiors, reserves explicit courtyard mouths and pedestrian
paths, and writes `rows_access.json` plus a full-town terrain diagnostic.

It may remove at most two smallest frontage blockers per block when a
qualifying inner courtyard has no clear opening. It does not add rear buildings
when the front-row population is already inside the brief capacity band.

