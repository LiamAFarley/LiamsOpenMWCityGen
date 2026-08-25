# Seam atlas review — Stage B gate (2026-08-24)

**Status: awaiting user review. No heights have been modified.**

## What was produced

`output/mapdata/terrain/tamriel_reworked/review_crops/` — 435 crops, one per
seam cluster, each a large ownership-split hypsometric hillshade centered on
the border (6-cell margin each side, red line = exact seam, title carries
cluster id / class / stats). Start from `_index.md` in that folder — it is a
browse table of every crop with class and mismatch numbers.

## Class census (517 of 531 edges measured; 14 void/void edges skipped)

| class | clusters | meaning |
|---|---|---|
| cliff-wall | 236 | owner terrain differs hugely AND drops/rises steeply at the border — short sharp continuation needed |
| plateau-step | 62 | big offset, near-vertical wall, little slope information — step must be absorbed over the blend width |
| sharp-mixed | 71 | moderate deltas, mixed local character |
| smooth-rise | 26 | gentle multi-cell transitions — wide smooth blend |
| already-matched | 36 | deltas ≤16 GU (all Bloodmoon; Habasi copied Solstheim exactly) |
| void-owner | 4 | owner cell has no heights (stub LAND) — water/no-constraint handling |

## Worst mismatches (all TR, east-coast Vvardenfell approach)

cluster172 dmed 15,660 GU (crop `cluster172_tr_cliff-wall_x-19_y-17.png`),
173, 171, 179, 182, 168, 161, 164, 167, 160 — the render shows TR's snow-cap
ridge terminating in a hard staircase against tamriel lowland. This is the
"big mountain with no backside" case; the blend must continue the descent
steeply on our side.

## What to check in the crops

1. Do the class labels match what you see? (Especially: any cliff-wall that
   should be smooth-rise, or plateau-step that is really a cliff?)
2. Are the margins wide enough to judge the blend zone? (`atlas.crop_margin_cells`)
3. Any seam areas that look already-acceptable (candidate `already-matched`
   reclassification — threshold `matched_max_delta_gu`)?
4. Sky border: rectangular flat-water features on the SHOTN side (e.g.
   cluster063) — confirm they are real mod content, not artifacts to avoid.

## Decision requested

Approve classification (or list cluster ids to reclassify) → Stage C
(amplification preview sweep) and Stage D (blend solve + before/after crops
in this exact framing) proceed. Thresholds live in
`configs/tamriel_reworked_v1.json → atlas.classification` — tuning them and
re-running this tool is cheap (~60 s).
