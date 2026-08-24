"""Working GU-scale constants for V2 townlayout.

Purpose
-------
Single table of lengths used by later townlayout stages.  Field spacing,
site span, and slope gates are measured.  Street/parcel lengths were
first-pass analogs (VTEX tile, TG junction×12) and are retuned against
D-STAMP hulls so lots can actually hold a building.

Stamp scale (Falkreath census): p10 hull ≈ 6.87e5 GU² (~830 GU), p50 ≈
1.99e6 (~1410 GU), p90 ≈ 5.45e6 (~2330 GU).  Eligible OBB widths run
~400–3000 GU.  Source Tamriel roads are site context, not inner
arterials.

Inputs / outputs
----------------
Imported numeric constants only.

Pipeline position
-----------------
V2 townlayout Phase 2 geometry/RNG; no generation.
"""

VERTEX_EPS_GU = 1e-6
JUNCTION_MERGE_GU = 96.0          # Voronoi vertex merge; leave patch topology stable
MIN_PATCH_AREA_GU2 = 800000.0     # ~p10 hull
ARTERIAL_CLEAR_WIDTH_GU = 512.0   # 1 VTEX tile
STREET_CLEAR_WIDTH_GU = 384.0
LANE_CLEAR_WIDTH_GU = 256.0
ALLEY_CLEAR_WIDTH_GU = 256.0      # was 192; alleys need to read at stamp scale
FIELD_SPACING_GU = 128.0          # must match citysite.FIELD_SPACING_GU
SITE_SPAN_GU = 57344.0
PARCEL_YARD_FACTOR = 1.8          # area estimator; locked in validate.py
STAMP_P50_HULL_AREA_GU2 = 1990000.0   # locked p50 hull (arterial growth plan rev2 §3.3)
SERVICED_LOT_AREA_GU2 = STAMP_P50_HULL_AREA_GU2 * PARCEL_YARD_FACTOR  # 3,582,000 GU²
URBAN_SPACE_FACTOR = 1.6
SLOPE_HARD_DEG = 25.0             # citysite.STEEP_BANK_LIMIT_DEG
SLOPE_SOFT_START_DEG = 5.0
SLOPE_BUILDABLE_LIMIT_DEG = 15.0  # citysite.SLOPE_BUILDABLE_LIMIT_DEG
REWRITE_MARGIN_GU = 1536.0
TRANSITION_STUB_LENGTH_GU = 512.0
KEEP_FORECOURT_INSET_GU = 200.0    # Phase 14 keep compound ring
MIN_PARCEL_FRONTAGE_GU = 256.0
MIN_PARCEL_WIDTH_GU = 512.0       # ~small-stamp OBB
ROUTE_REACH_GU = 768.0           # single source of truth for route-endpoint reach used by both mouth discovery and alley_infill._grid_route
ROUTE_CONNECTOR_GU = 512.0       # maximum endpoint-to-eroded-lobe connector in _grid_route
MIN_VERGE_AREA_GU2 = 65536.0      # 256×256; drop sliver leftovers
SEATING_INSET_GU = 64.0           # curb slop; core lots sit on the street
FRONTAGE_TOUCH_GU = 256.0         # must exceed max ward setback
STAMP_FILL_MAX = 0.95             # hull may fill the parcel (dense core)
PACK_SLACK_STONE = 1.08           # Markarth: buildings almost touching
PACK_SLACK_WOOD = 1.20            # Karthgad: still tight, a bit more air
PACK_SLACK_OUTSKIRTS = 1.40
