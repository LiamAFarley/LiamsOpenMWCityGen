"""Build and render the reusable Cityforge visual-planning canvas.

Pipeline position
------------------
This CLI is the deterministic visual-planner worker between the accepted
site/stamp/road products and a vision-capable settlement designer.  It loads
the exact D-SITE terrain field, the aligned road consumer API, and both
accepted D-STAMP libraries; it then validates the versioned visual-plan
extension, renders a Pillow canvas, and emits a separated advisory report.
The only canonical generated data is the labelled synthetic proof under
``output/cityforge/phase1/visual_planner_fixture/``.  This command never runs
Blender, never edits an original plugin/mod file, never consumes source-space
road coordinates/XCF data, and never authors a real Falkreath plan.

Inputs
------
* ``site_survey.json`` + ``survey_fields.npz`` (exact terrain/masks/raw VTEX);
* ``tamriel_aligned_centerlines_v1`` through ``procgen.aligned_roads``;
* the two accepted D-STAMP JSON libraries plus the accepted Markarth palette;
* a visual-plan extension JSON document.

Outputs
-------
The CLI writes a PNG, a canonical advisory JSON report, a background/render
manifest, and hashes.  ``--proof`` writes the four required synthetic image
variants; each variant is a new view of one deliberately labelled synthetic
proof, not a real town plan.  PNGs are rendered with Pillow only and the
iteration count is an explicit command-line value audited in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen import aligned_roads  # noqa: E402
from procgen.visual_planner_advisory import analyze_plan  # noqa: E402
from procgen.visual_planner_eligibility import (  # noqa: E402
    StampEligibilityPolicy,
    build_eligibility_policy,
)
from procgen.visual_planner_format import (  # noqa: E402
    canonical_json_bytes,
    require_valid_extension,
    validate_extension,
)
from procgen.visual_planner_symbols import render_plan_layers  # noqa: E402
from procgen.visual_planner_terrain import TerrainBundle  # noqa: E402


CANONICAL_SURVEY = ROOT / "output/cityforge/sites/falkreath_v1/site_survey.json"
CANONICAL_FIELDS = ROOT / "output/cityforge/sites/falkreath_v1/survey_fields.npz"
CANONICAL_ROADS = ROOT / "output/mapdata/roads/tamriel_aligned_centerlines_v1"
CANONICAL_LIBRARIES = (
    ROOT / "output/cityforge/stamps/karthgad_nord_v2.json",
    ROOT / "output/cityforge/stamps/markarth_side_stone_v2.json",
)
CANONICAL_PALETTE = (
    ROOT / "output/settlement-splits/markarth-side-v2/"
    "final-markarth-extraction-2026-08-10-library/stamp_palette_v1/catalog.json"
)
DEFAULT_PROOF_DIR = ROOT / "output/cityforge/phase1/visual_planner_fixture"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_stamp_geometry(paths: tuple[Path, ...] = CANONICAL_LIBRARIES) -> dict[str, dict[str, Any]]:
    """Load all accepted stamps into one deterministic id-indexed mapping."""

    geometry: dict[str, dict[str, Any]] = {}
    for path in paths:
        library = _load_json(path)
        for stamp in library.get("stamps", []):
            stamp_id = stamp.get("stamp_id")
            if not isinstance(stamp_id, str) or not stamp_id:
                raise ValueError(f"stamp without stable id in {path}")
            if stamp_id in geometry:
                raise ValueError(f"duplicate stamp id {stamp_id}")
            geometry[stamp_id] = stamp
    return dict(sorted(geometry.items()))


def load_inputs(
    plan_path: Path,
    *,
    survey_path: Path = CANONICAL_SURVEY,
    fields_path: Path = CANONICAL_FIELDS,
    roads_path: Path = CANONICAL_ROADS,
    library_paths: tuple[Path, ...] = CANONICAL_LIBRARIES,
) -> tuple[dict[str, Any], TerrainBundle, Any, dict[str, dict[str, Any]], StampEligibilityPolicy]:
    """Load the exact visual-planner input graph, fail-closed."""

    plan = _load_json(plan_path)
    issues = validate_extension(plan)
    if issues:
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in issues[:5])
        raise ValueError(f"visual plan extension is invalid: {detail}")
    terrain = TerrainBundle.from_paths(survey_path, fields_path)
    network = aligned_roads.load_aligned_network(roads_path)
    geometry = load_stamp_geometry(library_paths)
    eligibility = build_eligibility_policy(library_paths, palette_path=CANONICAL_PALETTE)
    eligibility.require_document(plan, geometry)
    return plan, terrain, network, geometry, eligibility


def _rectangle_from_plan(plan: dict[str, Any], terrain: TerrainBundle):
    rectangle = plan["rectangle"]
    return terrain.rectangle(
        cell_bounds=rectangle.get("cell_bounds"),
        world_bounds_gu=rectangle.get("world_bounds_gu"),
        context_margin_gu=float(rectangle.get("context_margin_gu", 0.0)),
        full_site_inset=bool(rectangle.get("full_site_inset", True)),
    )


def render_one(
    plan_path: Path,
    out_path: Path,
    *,
    report_path: Path | None = None,
    manifest_path: Path | None = None,
    title: str = "CITYFORGE VISUAL PLANNING CANVAS",
    size: tuple[int, int] = (1440, 1180),
    show_contours: bool = False,
    show_slope: bool = False,
    show_source_terrain: bool = False,
    show_burial_envelope: bool = False,
    survey_path: Path = CANONICAL_SURVEY,
    fields_path: Path = CANONICAL_FIELDS,
    roads_path: Path = CANONICAL_ROADS,
    library_paths: tuple[Path, ...] = CANONICAL_LIBRARIES,
    iteration: int = 0,
    max_iterations: int = 3,
    show_advisory_markers: bool = False,
    adversarial_proof: bool = False,
) -> dict[str, Any]:
    """Run structural gate, advisory analysis, Pillow render, and audits."""

    if iteration < 0 or max_iterations < 0 or iteration > max_iterations:
        raise ValueError("iteration must be within the declared maximum")
    plan, terrain, network, geometry, eligibility = load_inputs(
        plan_path, survey_path=survey_path, fields_path=fields_path,
        roads_path=roads_path, library_paths=library_paths)
    rectangle = _rectangle_from_plan(plan, terrain)
    advisory = analyze_plan(plan, terrain, rectangle, aligned_network=network,
                            stamp_geometry=geometry)
    if advisory["hard_errors"] and not adversarial_proof:
        raise ValueError(
            f"visual plan has {len(advisory['hard_errors'])} hard errors; refusing to render")
    options = plan.get("render_options", {})
    if isinstance(options, dict):
        show_contours = bool(options.get("show_contours", show_contours))
        show_slope = bool(options.get("show_slope", show_slope))
        show_source_terrain = bool(options.get("show_source_terrain", show_source_terrain))
        show_burial_envelope = bool(options.get("show_burial_envelope", show_burial_envelope))
        title = str(options.get("legend_title", title))
        include_context_inset = bool(options.get("show_context_inset", True))
    else:
        include_context_inset = True
    image, render_audit = render_plan_layers(
        plan, terrain, rectangle, aligned_network=network, stamp_geometry=geometry,
        title=title, size=size, show_contours=show_contours, show_slope=show_slope,
        show_source_terrain=show_source_terrain,
        show_burial_envelope=show_burial_envelope,
        advisory_report=advisory,
        show_advisory_markers=show_advisory_markers,
        include_context_inset=include_context_inset,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG", optimize=False, compress_level=6)
    report_path = report_path or out_path.with_suffix(".advisory.json")
    manifest_path = manifest_path or out_path.with_suffix(".manifest.json")
    _write_json(report_path, advisory)
    manifest = {
        "schema_version": 1,
        "kind": "cityforge_visual_planner_render_audit",
        "plan_id": plan.get("plan_id"),
        "plan_sha256": _sha256(plan_path),
        "output_png_sha256": _sha256(out_path),
        "output_png_size": list(image.size),
        "iteration": iteration,
        "max_iterations": max_iterations,
        "input_hashes": {
            "site_survey": terrain.hashes["site_survey"],
            "survey_fields": terrain.hashes["survey_fields"],
            "aligned_road_product": network.product_sha256,
            "stamp_libraries": {str(path): _sha256(path) for path in library_paths},
        },
        "rectangle": rectangle.to_dict(),
        "background": terrain.manifest(rectangle),
        "render": render_audit,
        "advisory_summary": advisory["summary"],
        "eligibility": {
            "accepted_stamp_count": len(eligibility.accepted_stamp_ids),
            "rejected_stamp_count": len(eligibility.rejected_stamp_ids),
            "rejected_stamp_ids": sorted(eligibility.rejected_stamp_ids),
            "palette_path": eligibility.palette_path,
            "metadata_hashes": dict(eligibility.metadata_hashes),
            "fail_closed": True,
        },
        "tes3_semantics_changed": False,
        "adversarial_proof": bool(adversarial_proof),
    }
    _write_json(manifest_path, manifest)
    return {"image": str(out_path), "report": str(report_path),
            "manifest": str(manifest_path), "sha256": manifest["output_png_sha256"],
            "advisory": advisory["summary"], "render": render_audit,
            "eligibility": manifest["eligibility"]}


def build_proof(out_dir: Path = DEFAULT_PROOF_DIR, *, adversarial_proof: bool = False) -> dict[str, Any]:
    """Write one compact synthetic demonstration and its four proof images."""

    if not adversarial_proof:
        raise ValueError("synthetic proof includes a hard-error case; pass --adversarial-proof explicitly")
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "visual_planner_fixture.synthetic_visual_plan.json"
    plan = synthetic_fixture_plan()
    _write_json(plan_path, plan)
    bad_plan_path = out_dir / "visual_planner_fixture.adversarial_visual_plan.json"
    _write_json(bad_plan_path, synthetic_adversarial_fixture_plan())
    results: dict[str, Any] = {}
    variants = {
        "planning_canvas_clean.png": {
            "show_contours": False, "show_slope": False,
            "show_source_terrain": False, "show_burial_envelope": False,
            "title": "SYNTHETIC VISUAL PLANNER - CLEAN VOCABULARY (NOT A FALKREATH DESIGN)",
        },
        "planning_canvas_topography.png": {
            "show_contours": True, "show_slope": False,
            "show_source_terrain": True, "show_burial_envelope": False,
            "title": "SYNTHETIC VISUAL PLANNER - TOPOGRAPHY (NOT A FALKREATH DESIGN)",
        },
        "planning_canvas_access.png": {
            "show_contours": False, "show_slope": True,
            "show_source_terrain": True, "show_burial_envelope": True,
            "title": "SYNTHETIC VISUAL PLANNER - ALL DOORS + ACCESS (NOT A FALKREATH DESIGN)",
        },
        "planning_canvas_advisory_bad_case.png": {
            "show_contours": False, "show_slope": True,
            "show_source_terrain": False, "show_burial_envelope": False,
            "title": "SYNTHETIC VISUAL PLANNER - LABELLED ADVERSARIAL ADVISORY CASE",
        },
    }
    for name, options in variants.items():
        is_bad_case = name.endswith("advisory_bad_case.png")
        results[name] = render_one(
            bad_plan_path if is_bad_case else plan_path, out_dir / name,
            report_path=out_dir / name.replace(".png", ".advisory.json"),
            manifest_path=out_dir / name.replace(".png", ".manifest.json"),
            title=options["title"], size=(1440, 1180), iteration=0, max_iterations=3,
            show_contours=options["show_contours"], show_slope=options["show_slope"],
            show_source_terrain=options["show_source_terrain"],
            show_burial_envelope=options["show_burial_envelope"],
            show_advisory_markers=is_bad_case,
            adversarial_proof=is_bad_case,
        )
    proof_manifest = {
        "schema_version": 1,
        "kind": "cityforge_visual_planner_synthetic_proof",
        "label": "SYNTHETIC VISUAL PLANNER FIXTURE - NOT A FALKREATH DESIGN",
        "plan": str(plan_path),
        "iteration_count": 1,
        "maximum_cheap_pillow_iterations": 3,
        "images": {name: {"sha256": row["sha256"], "path": row["image"]}
                   for name, row in sorted(results.items())},
        "results": results,
        "eligibility": results[sorted(results)[0]]["eligibility"] if results else None,
        "tes3_semantics_changed": False,
        "adversarial_plan": str(bad_plan_path),
        "adversarial_render_requires_explicit_flag": True,
        "render_invocation_ledger": str(out_dir / "render_invocation_ledger.jsonl"),
        "render_invocations": _read_invocation_ledger(out_dir / "render_invocation_ledger.jsonl"),
    }
    proof_manifest = _relativize_proof_paths(proof_manifest, out_dir)
    _write_json(out_dir / "proof_manifest.json", proof_manifest)
    return proof_manifest


def build_invocation_ledger(out_dir: Path, command: str, *, phase: str,
                            timestamp_utc: str | None = None) -> Path:
    """Append one CLI proof invocation to the repair ledger.

    The repair harness calls this before doing any proof work.  It intentionally
    uses append-only JSONL semantics so a manifest cannot erase earlier command
    attempts.  The caller supplies the timestamp to keep tests and proof files
    deterministic; normal CLI use records the current UTC value.
    """

    import datetime as _datetime
    ledger = out_dir / "render_invocation_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    existing_count = len(_read_invocation_ledger(ledger))
    if existing_count >= 3:
        raise ValueError("proof render invocation budget exhausted (maximum three CLI invocations)")
    row = {
        "sequence": existing_count + 1,
        "timestamp_utc": timestamp_utc or _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "command": command,
        "phase": phase,
    }
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return ledger


def _read_invocation_ledger(path: Path) -> list[dict[str, Any]]:
    """Read the append-only JSONL ledger without changing it."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("render invocation ledger row is not an object")
            rows.append(value)
    return rows


def _relativize_proof_paths(value: Any, out_dir: Path) -> Any:
    """Make proof manifests portable when the repair directory is installed."""

    if isinstance(value, str) and "visual_planner_fixture_repair" in value:
        return value.replace("visual_planner_fixture_repair", out_dir.name)
    if isinstance(value, list):
        return [_relativize_proof_paths(item, out_dir) for item in value]
    if isinstance(value, dict):
        return {key: _relativize_proof_paths(item, out_dir) for key, item in value.items()}
    return value


def synthetic_fixture_plan() -> dict[str, Any]:
    """Return the labelled, deterministic vocabulary demonstration plan."""

    stamps = [
        # The selected example is intentionally an eligible, terrain-backed
        # two-door Karthgad unit.  Castle Barracks is quarantined by accepted
        # palette metadata and must never appear in this normal proof.
        {"lot_id": "lot_keep", "stamp_id": "karthgad_v1__door_094676_door_094677",
         "position_plan_gu": [30000, 18300], "yaw_deg": 0, "kit": "karthgad_nord",
         "category": "civic", "door_intents": [
             {"door_id": "-102_11_ref_094676", "intent": "public", "target_id": "plaza_irregular"},
             {"door_id": "-102_11_ref_094677", "intent": "service", "target_id": "shared_court"}],
         "intentional_slope_capable": True, "show_source_terrain": True,
         "show_burial_envelope": True},
        {"lot_id": "lot_house_pair", "stamp_id": "karthgad_v1__door_094671_door_095390",
         "position_plan_gu": [28500, 25000], "yaw_deg": 15, "kit": "karthgad_nord",
         "category": "house", "door_intents": [
             {"door_id": "-102_11_ref_094671", "intent": "public", "target_id": "road_local"},
             {"door_id": "-102_11_ref_095390", "intent": "private", "target_id": "shared_court"},
          ]},
        {"lot_id": "lot_tavern", "stamp_id": "markarth_side_v1__u31_shor_s_hearth_inn",
               "position_plan_gu": [30000, 27500], "yaw_deg": 90, "kit": "markarth_side_stone",
          "category": "commercial", "door_intents": [
              {"door_id": "-102_20_ref_023910", "intent": "public", "target_id": "road_local"},
              {"door_id": "-102_20_ref_023921", "intent": "service", "target_id": "alley_service"},
          ]},
        {"lot_id": "lot_warehouse", "stamp_id": "markarth_side_v1__u31_marketplace_warehouse",
          "position_plan_gu": [25500, 25000], "yaw_deg": 15, "kit": "markarth_side_stone",
          "category": "commercial", "door_intents": [
              {"door_id": "-101_20_ref_002107", "intent": "public", "target_id": "road_local"},
              {"door_id": "-101_20_ref_002108", "intent": "service", "target_id": "alley_service"},
          ]},
        {"lot_id": "lot_manor", "stamp_id": "markarth_side_v1__u0_snake_tooth_estate_moireh_s_hut",
          "position_plan_gu": [24500, 30500], "yaw_deg": 330, "kit": "markarth_side_stone",
           "category": "house", "door_intents": [{"door_id": "-100_18_ref_027106", "intent": "private", "target_id": "road_edge_83b260210de5d83d"}]},
        {"lot_id": "lot_cottage", "stamp_id": "karthgad_v1__door_094670",
          "position_plan_gu": [31000, 30500], "yaw_deg": 25, "kit": "karthgad_nord",
          "category": "house", "door_intents": [{"door_id": "-102_11_ref_094670", "intent": "private", "target_id": "plaza_irregular"}]},
    ]
    return {
        "schema_version": 1,
        "kind": "cityforge_visual_plan_extension",
        "plan_id": "visual_planner_fixture_v1",
        "base_t1_1_plan_id": "synthetic_validation_fixture_v1",
        "seed": 20260811,
        "coordinate_frame": "site_survey_plan_gu",
        "rectangle": {"cell_bounds": [-93, -92, -9, -8], "context_margin_gu": 1024,
                       "full_site_inset": True},
        "existing_source_roads": [
            {"edge_id": "road_edge_83b260210de5d83d", "label": "old aligned approach",
             "hierarchy": "regional", "show_corridor": True, "corridor_margin_gu": 160},
        ],
        "authored_roads": [
            {"road_id": "road_local", "class": "street", "width_gu": 640,
              "surface": "road", "polyline_plan_gu": [[17600, 24000], [21500, 24000], [25500, 23800], [31000, 23800]],
              "connection_targets": [{"target_id": "road_edge_83b260210de5d83d", "at_plan_gu": [17600, 24000], "tolerance_gu": 900}]},
        ],
        "alleys": [
            {"alley_id": "alley_service", "class": "service", "width_gu": 256, "surface": "settlement_dirt",
              "polyline_plan_gu": [[23000, 32000], [26000, 32000], [30000, 32000], [32500, 32000]],
              "connection_targets": [{"target_id": "shared_court", "at_plan_gu": [26000, 32000]}]},
        ],
        "road_surface_polygons": [
            {"region_id": "plaza_irregular", "kind": "plaza", "surface": "settlement_cobble",
              "polygon_plan_gu": [[27800, 15800], [32200, 15800], [32600, 17700], [30700, 20100], [28000, 19400]]},
        ],
        "shared_courts": [
            {"court_id": "shared_court", "surface": "settlement_grass_dirt",
              "polygon_plan_gu": [[24000, 26500], [27000, 26500], [27500, 28500], [24500, 28800]],
              "connection_targets": [{"target_id": "alley_service", "at_plan_gu": [26000, 32000]}]},
        ],
        "stamps": stamps,
        "districts": [
              {"district_id": "core", "label": "CIVIC / MARKET CORE", "kind": "core",
              "polygon_plan_gu": [[23500, 22000], [32500, 22000], [32500, 30000], [23500, 30000]]},
        ],
        "annotations": [
             {"annotation_id": "note_center", "kind": "design_reason", "text": "synthetic vocabulary · source road · street · court · plaza", "position_plan_gu": [25000, 21500]},
        ],
        "advisory_overrides": [],
         "render_options": {"map_width_px": 1440, "map_height_px": 1180,
                             "show_context_inset": True, "selected_lot_id": "lot_keep"},
        "design_notes": "SYNTHETIC VISUAL PLANNER FIXTURE - NOT A FALKREATH DESIGN. Demonstrates format, symbols, and advisory separation only.",
    }


def synthetic_adversarial_fixture_plan() -> dict[str, Any]:
    """Return a distinct proof-only fixture whose real checks must fire.

    Two identical manor stamps are close and have the same orientation.  The
    rear lot has no usable alley/court route, and a street corridor crosses the
    front lot.  It is rendered only through ``--adversarial-proof``.
    """

    return {
        "schema_version": 1, "kind": "cityforge_visual_plan_extension",
        "plan_id": "visual_planner_adversarial_fixture_v1",
        "base_t1_1_plan_id": "synthetic_validation_fixture_v1", "seed": 20260811,
        "coordinate_frame": "site_survey_plan_gu",
        "rectangle": {"cell_bounds": [-93, -92, -9, -8], "context_margin_gu": 1024,
                       "full_site_inset": True},
        "existing_source_roads": [],
        "authored_roads": [{"road_id": "road_overlap_bad", "class": "street",
            "width_gu": 640, "surface": "road",
            "polyline_plan_gu": [[20500, 20400], [24000, 20400]],
            "connection_targets": []}],
        "alleys": [], "road_surface_polygons": [], "shared_courts": [],
        "stamps": [
            {"lot_id": "bad_front_lot", "stamp_id": "markarth_side_v1__u0_snake_tooth_estate_moireh_s_hut",
             "position_plan_gu": [22000, 20400], "yaw_deg": 0, "kit": "markarth_side_stone",
             "category": "house", "door_intents": [{"door_id": "-100_18_ref_027106",
             "intent": "public", "target_id": "road_overlap_bad"}]},
            {"lot_id": "bad_rear_lot", "stamp_id": "markarth_side_v1__u0_snake_tooth_estate_moireh_s_hut",
             "position_plan_gu": [22000, 22000], "yaw_deg": 0, "kit": "markarth_side_stone",
             "category": "house", "door_intents": [{"door_id": "-100_18_ref_027106",
             "intent": "private", "target_id": "no_usable_access_target"}]},
        ],
        "districts": [], "annotations": [], "advisory_overrides": [],
        "render_options": {"map_width_px": 1440, "map_height_px": 1180,
                           "show_context_inset": True, "selected_lot_id": "bad_rear_lot"},
        "design_notes": "SYNTHETIC ADVERSARIAL PROOF ONLY - expected hard errors and advisories.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--proof", action="store_true")
    parser.add_argument("--proof-dir", type=Path, default=DEFAULT_PROOF_DIR)
    parser.add_argument("--invocation-ledger", type=Path,
                        help="append this CLI invocation to a JSONL render ledger")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--show-contours", action="store_true")
    parser.add_argument("--show-slope", action="store_true")
    parser.add_argument("--show-source-terrain", action="store_true")
    parser.add_argument("--show-burial-envelope", action="store_true")
    parser.add_argument("--adversarial-proof", action="store_true",
                        help="explicitly allow rendering a hard-error proof fixture")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.proof:
            ledger_path = args.invocation_ledger or (args.proof_dir / "render_invocation_ledger.jsonl")
            build_invocation_ledger(
                ledger_path.parent,
                " ".join(str(value) for value in sys.argv),
                phase="proof",
            )
            result = build_proof(args.proof_dir, adversarial_proof=args.adversarial_proof)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.plan is None or args.out is None:
            raise ValueError("--plan and --out are required unless --proof is used")
        if args.adversarial_proof:
            raise ValueError("--adversarial-proof is only valid with --proof")
        result = render_one(
            args.plan, args.out, report_path=args.report, manifest_path=args.manifest,
            iteration=args.iteration, max_iterations=args.max_iterations,
            show_contours=args.show_contours, show_slope=args.show_slope,
            show_source_terrain=args.show_source_terrain,
            show_burial_envelope=args.show_burial_envelope,
            adversarial_proof=False,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - exact failure is part of the CLI contract
        print(f"FAILURE: visual settlement planner {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
