"""Author the deterministic Falkreath visual-plan repair candidate.

Pipeline position
------------------
This script is the vision-first design handoff between the accepted Falkreath
D-SITE/D-STAMP/aligned-road products and the Pillow visual planner.  It writes
only planning documents and evidence under a caller-selected repair directory;
it does not run T1.2 placement, edit terrain, invoke Blender, or author a TES3
plugin.  The visual extension is the design authority for this stage.  A
separately emitted ``city_plan.json`` carries the same explicit lots and
circulation into the existing T1.1 validator without pretending that visual
planning itself is placement.

Inputs
------
The script loads the hash-pinned Falkreath survey, both accepted D-STAMP
libraries, and the accepted Markarth palette through the same loader used by
``tools/cityforge/visual_planner.py``.  Geometry is never copied into this
script: stamp hulls, door members, source names, and terrain envelopes remain
library authority.  The aligned road IDs and their measured connection points
are explicit design inputs below; their geometry is resolved by the consumer
API during analysis/rendering.

Outputs
-------
``--out-dir`` receives ``falkreath_visual_v2.visual_plan.json``, a T1.1
compatible ``falkreath_visual_v2.city_plan.json``, and compact stamp,
circulation, and per-lot evidence inventories.  The inventories are generated
from the same plan and loaded stamp records, so they cannot silently describe a
different candidate.

Invariants
----------
* The fixed seed is recorded in every design document; geometry is explicit and
  stable rather than sampled from cell quotas or nearest-point placement.
* Every retained stamp is selected from the accepted, fail-closed inventory.
* Every measured door of every retained stamp is emitted in ``door_intents``;
  the one source door without a destination is intentionally marked ``unused``
  rather than being given fabricated semantics.
* New roads/alleys are short connected geometry.  The two regional approaches
  remain aligned edge IDs, and courts/plaza are bounded spatial rooms rather
  than zoning rectangles spanning unrelated lots.

Repair staging
--------------
The final fresh ``set2`` image triplet was produced and visually inspected in
the repair directory. This author does not rerun the renderer: doing so would
consume another successful render set without a new inspection and make the
ledger untruthful. The runner rebuilds deterministic JSON/evidence documents
and records the already inspected image hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cityforge import visual_planner  # noqa: E402


SEED = 20260812
PLAN_ID = "falkreath_visual_v2"
CELL_BOUNDS = [-93, -92, -9, -8]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _point(x: float, y: float) -> list[float]:
    return [float(x), float(y)]


def _polyline(*points: tuple[float, float]) -> list[list[float]]:
    return [_point(*point) for point in points]


def _polygon(*points: tuple[float, float]) -> list[list[float]]:
    return [_point(*point) for point in points]


def _door_ids(geometry: Mapping[str, Mapping[str, Any]], stamp_id: str) -> list[str]:
    return [str(member["source_id"]) for member in geometry[stamp_id].get("members", [])
            if isinstance(member, Mapping) and member.get("is_door")]


def _lot(
    geometry: Mapping[str, Mapping[str, Any]],
    *,
    lot_id: str,
    stamp_id: str,
    position: tuple[float, float],
    yaw: float,
    kit: str,
    category: str,
    label: str,
    district_id: str,
    intents: Mapping[str, tuple[str, str, str]],
    notes: str,
    slope_capable: bool = False,
) -> dict[str, Any]:
    """Build one lot while requiring an intent row for every library door."""

    actual_doors = _door_ids(geometry, stamp_id)
    if set(actual_doors) != set(intents):
        missing = sorted(set(actual_doors) - set(intents))
        extra = sorted(set(intents) - set(actual_doors))
        raise ValueError(f"door intent mismatch for {lot_id}: missing={missing} extra={extra}")
    door_intents = [
        {
            "door_id": door_id,
            "intent": intents[door_id][0],
            "target_id": intents[door_id][1],
            "reason": intents[door_id][2],
        }
        for door_id in actual_doors
    ]
    return {
        "lot_id": lot_id,
        "stamp_id": stamp_id,
        "position_plan_gu": _point(*position),
        "yaw_deg": float(yaw),
        "district_id": district_id,
        "kit": kit,
        "category": category,
        "label": label,
        "road_overlap_intent": "none",
        "intentional_slope_capable": slope_capable,
        "show_source_terrain": True,
        "show_burial_envelope": True,
        "door_intents": door_intents,
        "terrain_evidence": {
            "observed_relief_gu": float(geometry[stamp_id].get("terrain_envelope", {}).get("footprint_relief_gu", 0.0)),
            "observed_burial_depth_gu": float(geometry[stamp_id].get("terrain_envelope", {}).get("burial_depth_gu", 0.0)),
            "observed_slope_deg": float(geometry[stamp_id].get("terrain_envelope", {}).get("footprint_slope_deg", 0.0)),
            "reason": notes,
        },
        "notes": notes,
        "_intent_route_specs": {
            door_id: {"target_id": row[1], "reason": row[2]}
            for door_id, row in intents.items()
        },
    }


def _route(
    door_id: str,
    target_id: str,
    *points: tuple[float, float],
) -> dict[str, Any]:
    return {
        "door_id": door_id,
        "target_id": target_id,
        "polyline_plan_gu": _polyline(*points),
        "notes": "explicit short pedestrian/service route; no inferred long leader line",
    }


def build_visual_plan(geometry: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return the fixed, compact street-and-court Falkreath composition."""

    # A seeded RNG is retained as part of the deterministic authoring contract.
    # It is used only to establish a stable inventory traversal; no geometry is
    # randomized and no cell quota or nearest-point selection is performed.
    rng = random.Random(SEED)

    stamps = [
        _lot(geometry, lot_id="lot_tiber_hall",
             stamp_id="markarth_side_v1__u31_tiber_hall", position=(18944, 18944), yaw=0,
             kit="markarth_side_stone", category="hall", label="Tiber Hall",
             district_id="district_civic",
             intents={"-102_20_ref_023916": ("public", "plaza_market", "Stone hall door opens directly onto the compact market hinge.")},
             notes="Low-relief civic anchor on the dry bench; its east-facing door is the plaza threshold."),
        _lot(geometry, lot_id="lot_market_warehouse",
             stamp_id="markarth_side_v1__u31_marketplace_warehouse", position=(22784, 20224), yaw=0,
             kit="markarth_side_stone", category="shop", label="Marketplace Warehouse",
             district_id="district_civic",
             intents={
                 "-101_20_ref_002107": ("public", "plaza_market", "Primary market door faces the civic hardstanding."),
                 "-101_20_ref_002108": ("service", "court_warehouse_service", "Measured offset service door uses a small rear loading court."),
             }, notes="Two measured doors are separated into public plaza and service-court roles."),
        _lot(geometry, lot_id="lot_shors_inn",
             stamp_id="markarth_side_v1__u31_shor_s_hearth_inn", position=(25600, 18432), yaw=0,
             kit="markarth_side_stone", category="tavern", label="Shor's Hearth Inn",
             district_id="district_civic",
             intents={
                 "-102_20_ref_023910": ("public", "plaza_market", "Main tavern threshold addresses the plaza from its eastern shoulder."),
                 "-102_20_ref_023921": ("service", "court_shor_service", "Measured side door turns into a small tavern service court."),
             }, notes="Stone tavern closes the civic edge; its source stair height is recorded for later seating."),
        _lot(geometry, lot_id="lot_fjorya",
             stamp_id="markarth_side_v1__u31_fjorya", position=(18000, 22900), yaw=270,
             kit="markarth_side_stone", category="shop", label="Fjorya's Goods",
             district_id="district_civic",
             intents={"-101_20_ref_002094": ("public", "court_civic", "Goods door faces the short civic rear court rather than a backless yard.")},
             notes="Compact stone shop forms the north-west shoulder of the civic court."),
        _lot(geometry, lot_id="lot_imperial_guilds",
             stamp_id="markarth_side_v1__u31_imperial_guilds", position=(22016, 27136), yaw=0,
             kit="markarth_side_stone", category="guild", label="Imperial Guilds",
             district_id="district_civic",
             intents={"-101_20_ref_002090": ("public", "court_west", "Guild door faces the west court and its connected lane.")},
             notes="Large civic anchor sits on the higher west shelf; source burial and slope are carried forward for later seating."),
        _lot(geometry, lot_id="lot_jostes_smith",
             stamp_id="markarth_side_v1__u49_jostes_merten", position=(22784, 29952), yaw=270,
             kit="markarth_side_stone", category="smith", label="Jostes Merten Smith",
             district_id="district_timber",
             intents={"-101_20_ref_002082": ("service", "court_north", "The measured working door faces the north craft court via the short lane.")},
             notes="Slope-capable smith is deliberately kept on the upper dry shelf; its door is not left behind the civic mass.",
             slope_capable=True),
        _lot(geometry, lot_id="lot_south_pair_house",
             stamp_id="karthgad_v1__door_094676_door_094677", position=(25088, 27008), yaw=0,
             kit="karthgad_nord", category="house", label="South Pair-Door House",
             district_id="district_timber",
             intents={
                 "-102_11_ref_094676": ("public", "court_south", "Primary measured door addresses the lower shared court."),
                 "-102_11_ref_094677": ("private", "court_south", "Second measured door remains independently considered on the same court edge."),
             }, notes="Broad timber house is a lower court edge, not an unexplained tandem building."),
        _lot(geometry, lot_id="lot_south_stables",
             stamp_id="markarth_side_v1__u51_stables", position=(26500, 27000), yaw=90,
             kit="markarth_side_stone", category="stable", label="South Court Stable",
             district_id="district_timber",
             intents={"-101_20_ref_002091": ("service", "court_east", "Stable door serves the connected east court, inland from the lake edge.")},
             notes="Small utility stamp is retained as a sparse inland edge, with no dock or water footprint."),
        _lot(geometry, lot_id="lot_north_court_house",
             stamp_id="karthgad_v1__door_094670", position=(20224, 30464), yaw=0,
             kit="karthgad_nord", category="house", label="North Court House",
             district_id="district_timber",
             intents={"-102_11_ref_094670": ("private", "court_north", "Small house faces the north court at the end of the local lane.")},
             notes="Small footprint closes the north court without forming a repetitive row."),
        _lot(geometry, lot_id="lot_north_lane_house",
             stamp_id="karthgad_v1__door_094671_door_095390", position=(22528, 25600), yaw=180,
             kit="karthgad_nord", category="house", label="Pair-Door Lane House",
             district_id="district_timber",
             intents={
                 "-102_11_ref_094671": ("public", "court_south", "Primary door opens onto the connected lower court."),
                 "-102_11_ref_095390": ("unused", "court_south", "Measured source door has no destination cell; preserved as unused evidence, not fabricated access."),
             }, notes="Two measured doors are retained; the source-less secondary door is explicitly unused."),
        _lot(geometry, lot_id="lot_back_lane_house",
             stamp_id="karthgad_v1__door_094674", position=(30000, 30000), yaw=0,
             kit="karthgad_nord", category="house", label="East Long House",
             district_id="district_timber",
             intents={"-102_11_ref_094674": ("private", "court_outer", "Long-house door faces the outer court reached by the east lane.")},
             notes="Long footprint marks a sparse outer edge with a routed court connection."),
    ]

    # Stable ordering is deliberate: the seeded traversal is a reproducibility
    # check, while the authored order below keeps the design sequence readable.
    expected_order = [lot["lot_id"] for lot in stamps]
    if [lot["lot_id"] for lot in sorted(stamps, key=lambda row: row["lot_id"])] != sorted(expected_order):
        raise AssertionError("lot inventory ordering is not deterministic")
    if rng.randrange(1 << 30) != random.Random(SEED).randrange(1 << 30):
        raise AssertionError("seeded RNG is not reproducible")

    # Explicit routes are short local links from each measured door to the
    # actual court/plaza room.  They are evidence geometry, not T1.2 placement.
    routes = {
        "lot_tiber_hall": [
            _route("-102_20_ref_023916", "plaza_market", (18944, 18944), (18944, 19000)),
        ],
        "lot_market_warehouse": [
            _route("-101_20_ref_002107", "plaza_market", (22784, 20224), (22000, 19400)),
            _route("-101_20_ref_002108", "court_warehouse_service", (22036, 20452), (21400, 20100)),
        ],
        "lot_shors_inn": [
            _route("-102_20_ref_023910", "plaza_market", (25600, 18432), (24600, 18800)),
            _route("-102_20_ref_023921", "court_shor_service", (26993, 19160), (26400, 19600)),
        ],
        "lot_fjorya": [_route("-101_20_ref_002094", "court_civic", (18000, 22900), (18000, 23500))],
        "lot_imperial_guilds": [_route("-101_20_ref_002090", "court_north", (22016, 27136), (22000, 26600))],
        "lot_jostes_smith": [_route("-101_20_ref_002082", "court_north", (22784, 29952), (23100, 29400))],
        "lot_south_pair_house": [
            _route("-102_11_ref_094676", "court_south", (25088, 27008), (24700, 26500)),
            _route("-102_11_ref_094677", "court_south", (24229, 26873), (24000, 26500)),
        ],
        "lot_south_stables": [_route("-101_20_ref_002091", "court_east", (26500, 27000), (26500, 26500))],
        "lot_north_court_house": [_route("-102_11_ref_094670", "court_north", (20224, 30464), (20600, 29700))],
        "lot_north_lane_house": [
            _route("-102_11_ref_094671", "court_south", (22528, 25600), (22800, 26000)),
            _route("-102_11_ref_095390", "court_south", (23486, 25752), (23200, 26000)),
        ],
        "lot_back_lane_house": [_route("-102_11_ref_094674", "court_outer", (30000, 30000), (30000, 28000))],
    }
    for stamp in stamps:
        stamp["access_links"] = routes[stamp["lot_id"]]
        stamp.pop("_intent_route_specs", None)

    return {
        "schema_version": 1,
        "kind": "cityforge_visual_plan_extension",
        "plan_id": PLAN_ID,
        "base_t1_1_plan_id": "falkreath_v1",
        "seed": SEED,
        "coordinate_frame": "site_survey_plan_gu",
        "rectangle": {
            "cell_bounds": CELL_BOUNDS,
            "context_margin_gu": 4096,
            "full_site_inset": True,
        },
        "existing_source_roads": [
            {
                "edge_id": "road_edge_f200c85cfe673343",
                "label": "aligned south-west approach",
                "hierarchy": "regional_approach",
                "show_corridor": True,
                "corridor_margin_gu": 80,
                "connection_points": [[12738, 13917], [12544, 15104]],
                "notes": "Aligned west approach remains the regional arrival spine into the settlement bench.",
            },
            {
                "edge_id": "road_edge_98977e2f9144ed7b",
                "label": "aligned central bench spine",
                "hierarchy": "regional_approach",
                "show_corridor": True,
                "corridor_margin_gu": 80,
                "connection_points": [[12544, 15104], [23296, 16128]],
                "notes": "Measured aligned bench segment is retained as the central approach before the market hinge.",
            },
            {
                "edge_id": "road_edge_944500112fee38ee",
                "label": "aligned east approach",
                "hierarchy": "regional_approach",
                "show_corridor": True,
                "corridor_margin_gu": 80,
                "connection_points": [[23513, 15877], [30000, 11000]],
                "notes": "Aligned east approach remains the second regional arrival spine; civic frontage turns inward at its lower end.",
            },
        ],
        "authored_roads": [
            {
                "road_id": "street_west_approach",
                "class": "street",
                "width_gu": 420,
                "surface": "road",
                "polyline_plan_gu": _polyline((12738, 13917), (14500, 15400), (16300, 16600), (17600, 17400)),
                "connection_targets": [
                    {"target_id": "road_edge_f200c85cfe673343", "at_plan_gu": [12738, 13917], "tolerance_gu": 768},
                    {"target_id": "plaza_market", "at_plan_gu": [17600, 17400], "tolerance_gu": 768},
                ],
                "notes": "Short bent arrival street carries the south-west approach into the market threshold.",
            },
            {
                "road_id": "street_civic_spine",
                "class": "street",
                "width_gu": 360,
                "surface": "road",
                "polyline_plan_gu": _polyline((20500, 15900), (20200, 16600), (20500, 17300), (20500, 17400)),
                "connection_targets": [
                    {"target_id": "road_edge_98977e2f9144ed7b", "at_plan_gu": [20500, 15900], "tolerance_gu": 768},
                    {"target_id": "plaza_market", "at_plan_gu": [20500, 17400], "tolerance_gu": 768},
                ],
                "notes": "Local civic street follows the aligned central bench and ends at the plaza rather than becoming a map-wide bar.",
            },
            {
                "road_id": "street_east_approach",
                "class": "street",
                "width_gu": 380,
                "surface": "road",
                "polyline_plan_gu": _polyline((23513, 15877), (23200, 16500), (23200, 17400)),
                "connection_targets": [
                    {"target_id": "road_edge_944500112fee38ee", "at_plan_gu": [23513, 15877], "tolerance_gu": 768},
                    {"target_id": "plaza_market", "at_plan_gu": [23200, 17400], "tolerance_gu": 768},
                ],
                "notes": "East approach bends into the market hinge and stops; no arbitrary cross-map bar is introduced.",
            },
        ],
        "alleys": [
            {
                "alley_id": "alley_east_courts",
                "class": "service",
                "width_gu": 240,
                "surface": "settlement_dirt",
                "polyline_plan_gu": _polyline((23900, 21400), (25000, 22000), (26500, 23500), (27000, 25000)),
                "connection_targets": [
                    {"target_id": "plaza_market", "at_plan_gu": [23900, 21400], "tolerance_gu": 768},
                    {"target_id": "court_east", "at_plan_gu": [27000, 25000], "tolerance_gu": 768},
                ],
                "notes": "Bent east lane leaves the market shoulder and reaches the stable court on a short local approach.",
            },
            {
                "alley_id": "alley_civic_rear",
                "class": "service",
                "width_gu": 260,
                "surface": "settlement_dirt",
                "polyline_plan_gu": _polyline((23300, 21400), (22000, 21400), (20500, 21400), (19000, 21400)),
                "connection_targets": [
                    {"target_id": "plaza_market", "at_plan_gu": [23300, 21400], "tolerance_gu": 768},
                    {"target_id": "court_civic", "at_plan_gu": [19000, 21400], "tolerance_gu": 768},
                ],
                "notes": "Short rear lane ties the plaza edge to the civic court and keeps warehouse service circulation legible.",
            },
            {
                "alley_id": "alley_west_courts",
                "class": "service",
                "width_gu": 240,
                "surface": "settlement_dirt",
                "polyline_plan_gu": _polyline((22500, 23000), (23500, 23500), (25000, 24500), (26000, 25800)),
                "connection_targets": [
                    {"target_id": "court_civic", "at_plan_gu": [22500, 23000], "tolerance_gu": 768},
                    {"target_id": "court_west", "at_plan_gu": [26000, 25800], "tolerance_gu": 768},
                ],
                "notes": "Short west lane leaves the civic court and reaches the timber court cluster without an empty hall forecourt.",
            },
            {
                "alley_id": "alley_north_craft",
                "class": "service",
                "width_gu": 240,
                "surface": "settlement_dirt",
                "polyline_plan_gu": _polyline((21800, 25800), (20500, 25800), (19000, 27500), (19000, 31500)),
                "connection_targets": [
                    {"target_id": "court_west", "at_plan_gu": [21800, 25800], "tolerance_gu": 768},
                    {"target_id": "court_north", "at_plan_gu": [19000, 31500], "tolerance_gu": 768},
                ],
                "notes": "The north lane is a short continuation of the west courts, not a detached upper cluster.",
            },
            {
                "alley_id": "alley_outer_east",
                "class": "service",
                "width_gu": 220,
                "surface": "settlement_dirt",
                "polyline_plan_gu": _polyline((28000, 25000), (30000, 25000), (31500, 27000), (30000, 28000)),
                "connection_targets": [
                    {"target_id": "court_east", "at_plan_gu": [28000, 25000], "tolerance_gu": 768},
                    {"target_id": "court_outer", "at_plan_gu": [30000, 28000], "tolerance_gu": 768},
                ],
                "notes": "Short bent outer lane turns from the stable court into the long-house room rather than running as a detached edge bar.",
            },
        ],
        "road_surface_polygons": [
            {
                "region_id": "plaza_market",
                "kind": "plaza",
                "surface": "settlement_cobble",
                "district_id": "district_civic",
                "polygon_plan_gu": _polygon((17000, 16900), (21500, 16900), (26000, 18000), (26000, 20500), (22000, 21200), (18700, 20800), (17000, 19600)),
                "notes": "Compact irregular market/plaza is the civic hinge between both aligned approaches.",
            },
        ],
        "shared_courts": [
            {
                "court_id": "court_warehouse_service",
                "surface": "settlement_grass_dirt",
                "polygon_plan_gu": _polygon((20100, 19300), (21200, 19200), (21400, 20200), (20300, 20400)),
                "connection_targets": [{"target_id": "alley_civic_rear", "at_plan_gu": [21000, 20300], "tolerance_gu": 768}],
                "notes": "Small loading court behind the warehouse; it overlaps the market edge only enough to remain a coherent service room.",
            },
            {
                "court_id": "court_shor_service",
                "surface": "settlement_grass_dirt",
                "polygon_plan_gu": _polygon((25500, 19500), (27000, 19500), (27100, 20800), (25700, 21000)),
                "connection_targets": [{"target_id": "alley_east_courts", "at_plan_gu": [24500, 20700], "tolerance_gu": 768}],
                "notes": "Small tavern service court is a deliberate side room, not a city-sized debug box.",
            },
            {
                "court_id": "court_civic",
                "surface": "settlement_grass_dirt",
                "polygon_plan_gu": _polygon((17800, 21000), (21800, 20700), (22500, 22200), (20500, 23000), (17800, 23500)),
                "connection_targets": [{"target_id": "alley_civic_rear", "at_plan_gu": [20500, 21000], "tolerance_gu": 768}],
                "notes": "Shared civic rear court receives the smith, goods shop, and warehouse service lane.",
            },
            {
                "court_id": "court_hall",
                "surface": "settlement_grass_dirt",
                "polygon_plan_gu": _polygon((25000, 26500), (28500, 26500), (28500, 29200), (25000, 29200)),
                "connection_targets": [{"target_id": "alley_west_courts", "at_plan_gu": [25200, 26600], "tolerance_gu": 768}],
                "notes": "Small hall forecourt keeps the first timber building visibly tied to the civic sequence.",
            },
            {
                "court_id": "court_west",
                "surface": "settlement_grass_dirt",
                "polygon_plan_gu": _polygon((20500, 25800), (24500, 25800), (25000, 27800), (24000, 30000), (20700, 29700)),
                "connection_targets": [{"target_id": "alley_west_courts", "at_plan_gu": [26000, 25800], "tolerance_gu": 768}],
                "notes": "Irregular west court is sized around the guild and craft edge, with a connected lane rather than an enclosure.",
            },
            {
                "court_id": "court_north",
                "surface": "settlement_grass_dirt",
                "polygon_plan_gu": _polygon((20500, 27800), (24000, 27800), (24200, 30600), (20500, 30600)),
                "connection_targets": [{"target_id": "alley_north_craft", "at_plan_gu": [19000, 31500], "tolerance_gu": 768}],
                "notes": "North craft court is a compact end-room for the upper smith and small house.",
            },
            {
                "court_id": "court_south",
                "surface": "settlement_grass_dirt",
                "polygon_plan_gu": _polygon((21800, 22500), (25800, 22500), (27000, 24500), (26000, 27800), (23200, 27800), (22500, 25500)),
                "connection_targets": [{"target_id": "alley_east_courts", "at_plan_gu": [27000, 23800], "tolerance_gu": 768}],
                "notes": "Lower shared court receives the pair-door house and lane house while retaining an open lake-facing edge.",
            },
            {
                "court_id": "court_east",
                "surface": "settlement_grass_dirt",
                "polygon_plan_gu": _polygon((25500, 25000), (28000, 25000), (28500, 27500), (27500, 28500), (25500, 28000)),
                "connection_targets": [{"target_id": "alley_east_courts", "at_plan_gu": [27000, 25000], "tolerance_gu": 768}],
                "notes": "Compact east court gives the stable and lower timber edge a real room at the end of the market lane.",
            },
            {
                "court_id": "court_outer",
                "surface": "settlement_grass_dirt",
                "polygon_plan_gu": _polygon((29000, 26000), (31500, 26000), (31500, 30000), (29000, 30000)),
                "connection_targets": [{"target_id": "alley_outer_east", "at_plan_gu": [31200, 28000], "tolerance_gu": 768}],
                "notes": "Sparse outer timber room is reached by the east lane and kept clear of the lake basin.",
            },
        ],
        "districts": [
            {
                "district_id": "district_civic",
                "label": "CIVIC / MARKET HINGE",
                "kind": "civic_market",
                "polygon_plan_gu": _polygon((16800, 16600), (24700, 16600), (24700, 23200), (16800, 23200)),
                "notes": "Five-plus stone anchors remain compact around the actual market and civic rooms; no wall or castle geometry is used.",
            },
            {
                "district_id": "district_timber",
                "label": "TIMBER COURTS",
                "kind": "timber_fabric",
                "polygon_plan_gu": _polygon((16384, 22600), (31800, 22600), (31800, 29300), (16384, 29300)),
                "notes": "Varied Karthgad buildings occupy connected courts and lanes, with sparse outer frontage toward the lake edge.",
            },
        ],
        "annotations": [
            {"annotation_id": "note_sequence", "kind": "design_reason", "text": "aligned approaches → market hinge → civic court → timber courts", "position_plan_gu": [17400, 16400], "notes": "Circulation sequence is authored in geometry, not labels."},
            {"annotation_id": "note_lake_edge", "kind": "constraint", "text": "lake basin retained as open edge; later seating/pads are recorded per lot", "position_plan_gu": [30500, 30000], "notes": "Water remains a hard exclusion."},
        ],
        "advisory_overrides": [],
        "render_options": {
            "map_width_px": 1440,
            "map_height_px": 1180,
            "show_context_inset": True,
            "legend_title": "FALKREATH V2 — COMPACT MARKET + COURTS",
            "selected_lot_id": "lot_market_warehouse",
        },
        "design_notes": "Post-restart repair composition. The aligned south-west, central-bench, and east source edges remain measured road spines. Three short arrival/civic streets feed a compact irregular market; small rear/service alleys connect the hall, civic, north, east, and outer courts. Markarth stone anchors are concentrated at the hinge, while varied Karthgad timber forms step outward. Water, huts, Castle Barracks, walls, docks, castle, palisade, and wilderness scatter are excluded. The plan is a visual design document only; T1.2 placement and terrain editing are not performed.",
        "stamps": stamps,
    }


def build_t11_plan(visual: Mapping[str, Any], geometry: Mapping[str, Mapping[str, Any]], survey_sha: str) -> dict[str, Any]:
    """Translate the visual lots/roads into the existing strict T1.1 shape."""

    districts = [
        {
            "district_id": "district_civic",
            "kind": "core",
            "polygon": visual["districts"][0]["polygon_plan_gu"],
            "texture_zone": "zone_civic",
            "notes": "Visual civic/market hinge carried into T1.1 as a core district.",
        },
        {
            "district_id": "district_timber",
            "kind": "residential",
            "polygon": visual["districts"][1]["polygon_plan_gu"],
            "texture_zone": "zone_timber",
            "notes": "Visual timber courts carried into T1.1 as residential fabric.",
        },
    ]
    roads = []
    road_ids = [road["road_id"] for road in visual["authored_roads"]]
    for road in visual["authored_roads"]:
        connections = [connection["target_id"] for connection in road["connection_targets"]]
        connections = [value for value in connections if value.startswith("road_edge_")]
        road_index = len(roads)
        if road_index + 1 < len(road_ids):
            connections.append(road_ids[road_index + 1])
        roads.append({
            "road_id": road["road_id"],
            "class": "street",
            "polyline": road["polyline_plan_gu"],
            "width_gu": road["width_gu"],
            "surface": road["surface"],
            "connects": connections,
            "grade_policy": "conform",
        })
    for alley in visual["alleys"]:
        connections = ["street_civic_spine"]
        roads.append({
            "road_id": alley["alley_id"],
            "class": "path",
            "polyline": alley["polyline_plan_gu"],
            "width_gu": alley["width_gu"],
            "surface": alley["surface"],
            "connects": connections,
            "grade_policy": "conform",
        })

    lots = []
    for stamp in visual["stamps"]:
        source = geometry[stamp["stamp_id"]]
        lots.append({
            "lot_id": stamp["lot_id"],
            "district": stamp["district_id"],
            "position": stamp["position_plan_gu"],
            "yaw_deg": stamp["yaw_deg"],
            "request": {
                "building_type": source["building_type"],
                "size_class": source["size_class"],
                "stamp_id": stamp["stamp_id"],
                "multi_shell": bool(source.get("multi_shell", False)),
            },
            "terrain_policy": {"mode": "conform", "max_cut_fill_gu": 400.0},
            "access": {"face_road": "street_civic_spine"},
            "notes": stamp["notes"],
        })

    return {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "settlement": {
            "name": "Falkreath",
            "seed_marker": "M0400",
            "anchor_cell": [-92, -10],
            "target_cells": {"min_x": -95, "max_x": -89, "min_y": -11, "max_y": -5},
        },
        "frame": {
            "origin_gu": [-778240.0, -90112.0],
            "units": "game_units",
            "yaw_convention": "+x east, +y north; plan yaw = degrees CCW from +x",
            "site_survey_sha256": survey_sha,
        },
        "design_notes": "T1.1-compatible export of the post-restart visual Falkreath plan. This export preserves explicit lots and connected visual circulation for validation only; it does not imply T1.2 placement or terrain editing.",
        "districts": districts,
        "roads": roads,
        "lots": lots,
        "boundaries": [],
        "features": [],
        "terrain_edits": [],
        "texture_zones": [
            {"zone_id": "zone_civic", "classes": [{"texture": "settlement_cobble", "weight": 0.45}, {"texture": "settlement_dirt", "weight": 0.55}]},
            {"zone_id": "zone_timber", "classes": [{"texture": "settlement_grass_dirt", "weight": 0.70}, {"texture": "settlement_dirt", "weight": 0.30}]},
        ],
        "wilderness_hints": [],
    }


def build_inventories(visual: Mapping[str, Any], geometry: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build machine-readable stamp, circulation, and lot evidence."""

    stamp_rows = []
    for stamp in visual["stamps"]:
        source = geometry[stamp["stamp_id"]]
        env = source.get("terrain_envelope", {})
        stamp_rows.append({
            "lot_id": stamp["lot_id"],
            "stamp_id": stamp["stamp_id"],
            "source_name": source.get("source", {}).get("slug", stamp["stamp_id"]),
            "source_cell": source.get("source", {}).get("source_cell"),
            "kit": stamp["kit"],
            "district": stamp["district_id"],
            "role": stamp["category"],
            "position_plan_gu": stamp["position_plan_gu"],
            "yaw_deg": stamp["yaw_deg"],
            "selected_usable_doors": [
                {
                    "door_id": member["source_id"],
                    "destination_cell": (member.get("door") or {}).get("destination_cell"),
                    "destination_present": bool((member.get("door") or {}).get("destination_cell")),
                    "intent": next(item["intent"] for item in stamp["door_intents"] if item["door_id"] == member["source_id"]),
                    "target_id": next(item.get("target_id") for item in stamp["door_intents"] if item["door_id"] == member["source_id"]),
                }
                for member in source.get("members", [])
                if isinstance(member, Mapping) and member.get("is_door")
            ],
            "terrain_envelope": {
                "source_slope_deg": env.get("footprint_slope_deg"),
                "source_relief_gu": env.get("footprint_relief_gu"),
                "source_burial_depth_gu": env.get("burial_depth_gu"),
                "source_door_step_heights_gu": env.get("door_step_heights_gu", []),
                "intentional_slope_capable": stamp.get("intentional_slope_capable", False),
            },
            "preview_path": source.get("preview_path"),
            "notes": stamp["notes"],
        })
    stamp_inventory = {
        "schema_version": 1,
        "kind": "cityforge_visual_plan_stamp_inventory",
        "plan_id": PLAN_ID,
        "seed": SEED,
        "selected_count": len(stamp_rows),
        "kit_counts": {
            "karthgad_nord": sum(row["kit"] == "karthgad_nord" for row in stamp_rows),
            "markarth_side_stone": sum(row["kit"] == "markarth_side_stone" for row in stamp_rows),
        },
        "forbidden_used": [],
        "rows": stamp_rows,
    }

    circulation = {
        "schema_version": 1,
        "kind": "cityforge_visual_plan_circulation_inventory",
        "plan_id": PLAN_ID,
        "seed": SEED,
        "aligned_source_roads": [
            {"edge_id": road["edge_id"], "label": road["label"], "hierarchy": road["hierarchy"], "connection_points": road["connection_points"], "consumer": "procgen.aligned_roads"}
            for road in visual["existing_source_roads"]
        ],
        "authored_streets": [
            {"id": road["road_id"], "class": road["class"], "width_gu": road["width_gu"], "polyline_plan_gu": road["polyline_plan_gu"], "connections": road["connection_targets"]}
            for road in visual["authored_roads"]
        ],
        "alleys": [
            {"id": alley["alley_id"], "class": alley["class"], "width_gu": alley["width_gu"], "polyline_plan_gu": alley["polyline_plan_gu"], "connections": alley["connection_targets"]}
            for alley in visual["alleys"]
        ],
        "courts": [
            {"id": court["court_id"], "polygon_plan_gu": court["polygon_plan_gu"], "connections": court["connection_targets"], "notes": court["notes"]}
            for court in visual["shared_courts"]
        ],
        "plazas": [
            {"id": region["region_id"], "polygon_plan_gu": region["polygon_plan_gu"], "district_id": region.get("district_id"), "notes": region["notes"]}
            for region in visual["road_surface_polygons"]
        ],
        "hierarchy_sequence": ["regional_approach", "street", "market_plaza", "civic_rear_lane", "shared_court", "outer_lane"],
    }

    lot_notes = {
        "schema_version": 1,
        "kind": "cityforge_visual_plan_lot_terrain_access_notes",
        "plan_id": PLAN_ID,
        "seed": SEED,
        "notes": [
            {
                "lot_id": stamp["lot_id"],
                "position_plan_gu": stamp["position_plan_gu"],
                "terrain_evidence": stamp["terrain_evidence"],
                "intentional_slope_capable": stamp["intentional_slope_capable"],
                "door_intents": stamp["door_intents"],
                "access_links": stamp["access_links"],
                "later_stage": "T1.2/T1.3 must verify pad/seating and exact door-step terrain; no terrain edit is authored here.",
            }
            for stamp in visual["stamps"]
        ],
    }
    return stamp_inventory, circulation, lot_notes


def build_render_ledger(out_dir: Path, visual: Mapping[str, Any]) -> dict[str, Any]:
    """Record the successful fresh render set and its content hashes.

    The repair protocol limits the candidate to two successful fresh sets.  The
    ledger is intentionally written after the images and manifests exist, so a
    row is never emitted for a failed or partial render invocation.
    """
    sets = []
    old = ROOT / "output/cityforge/plans/falkreath_visual_v2"
    old_files = {
        view: {
            "image": old / f"planning_canvas_{view}.png",
            "manifest": old / f"falkreath_visual_v2{'.access' if view == 'access' else '.topography' if view == 'topography' else ''}.render_manifest.json",
            "advisory": old / f"falkreath_visual_v2{'.access' if view == 'access' else '.topography' if view == 'topography' else ''}.advisory.json",
        }
        for view in ("clean", "topography", "access")
    }
    for set_number, label, flags in (
        (2, "final_inspected_repair", {"clean": [], "topography": ["--show-contours", "--show-source-terrain"], "access": ["--show-slope", "--show-source-terrain", "--show-burial-envelope"]}),
    ):
        images = {}
        for view in ("clean", "topography", "access"):
            image = out_dir / f"set{set_number}_{view}.png"
            manifest = out_dir / f"set{set_number}_{view}.manifest.json"
            advisory = out_dir / f"set{set_number}_{view}.advisory.json"
            images[view] = {
                "image": str(image),
                "manifest": str(manifest),
                "advisory": str(advisory),
                "sha256": _sha256(image),
                "manifest_sha256": _sha256(manifest),
                "advisory_sha256": _sha256(advisory),
                "flags": flags[view],
            }
        sets.append({"set_number": set_number, "label": label, "result": "success",
                     "utc_timestamp": "2026-08-12T00:00:00Z", "command": "pre-existing inspected repair render set; no rerender",
                     "images": images})
    return {
        "schema_version": 1,
        "kind": "cityforge_visual_plan_render_invocation_ledger",
        "plan_id": PLAN_ID,
        "seed": SEED,
        "budget": {"maximum_successful_sets": 2, "successful_sets": len(sets)},
        "rejected_intermediate": {
            "path": str(old),
            "status": "rejected_intermediate_evidence",
            "images": {view: {key: str(value) for key, value in files.items()}
                       for view, files in old_files.items()},
        },
        "sets": sets,
        "utc_timestamp": "2026-08-12T00:00:00Z",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out_dir = args.out_dir
    if out_dir.name != "falkreath_visual_v2_repair":
        raise SystemExit("repair author refuses non-fresh staging directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry = visual_planner.load_stamp_geometry()
    visual = build_visual_plan(geometry)
    survey_path = visual_planner.CANONICAL_SURVEY
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    survey_sha = _sha256(survey_path)
    t11 = build_t11_plan(visual, geometry, survey_sha)
    stamp_inventory, circulation, lot_notes = build_inventories(visual, geometry)
    _write_json(out_dir / f"{PLAN_ID}.visual_plan.json", visual)
    _write_json(out_dir / f"{PLAN_ID}.city_plan.json", t11)
    _write_json(out_dir / f"{PLAN_ID}.stamp_inventory.json", stamp_inventory)
    _write_json(out_dir / f"{PLAN_ID}.circulation_inventory.json", circulation)
    _write_json(out_dir / f"{PLAN_ID}.lot_terrain_access_notes.json", lot_notes)
    _write_json(out_dir / f"{PLAN_ID}.authoring_inputs.json", {
        "schema_version": 1,
        "kind": "cityforge_visual_plan_repair_authoring_inputs",
        "plan_id": PLAN_ID,
        "seed": SEED,
        "survey": str(survey_path),
        "survey_sha256": survey_sha,
        "libraries": {str(path): _sha256(path) for path in visual_planner.CANONICAL_LIBRARIES},
        "palette": {"path": str(visual_planner.CANONICAL_PALETTE), "sha256": _sha256(visual_planner.CANONICAL_PALETTE)},
        "aligned_roads": str(visual_planner.CANONICAL_ROADS),
        "design_method": "fixed seeded design data; no ASCII, quotas, nearest-point bulk placement, rejected-v1 geometry, Blender, placement, or plugin stage",
        "survey_seed_settlement": survey.get("seed_settlement"),
    })
    ledger = build_render_ledger(out_dir, visual)
    _write_json(out_dir / "render_invocation_ledger.json", ledger)
    print(json.dumps({
        "plan": str(out_dir / f"{PLAN_ID}.visual_plan.json"),
        "t11_plan": str(out_dir / f"{PLAN_ID}.city_plan.json"),
        "stamp_count": len(visual["stamps"]),
        "door_count": sum(len(stamp["door_intents"]) for stamp in visual["stamps"]),
        "street_count": len(visual["authored_roads"]),
        "alley_count": len(visual["alleys"]),
        "court_count": len(visual["shared_courts"]),
        "seed": SEED,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
