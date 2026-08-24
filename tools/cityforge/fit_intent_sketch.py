"""Fit an authored Cityforge frontage/composition intent into a resolved sketch.

Pipeline position
------------------
This CLI is the JSON-only authoring boundary between a designer's world-GU
intent and the existing ``plan_sketch.py`` renderer.  It loads
manifest-pinned full-precision stamp libraries, the aligned road network, and
terrain masks; builds only named source/authored targets; runs the pure
``procgen.frontage_fit`` geometry, hard-relationship, and optional bounded
improvement stages; and writes three files under a fresh output directory.  It
never renders, runs Blender, authors TES3, changes source libraries/mod files,
selects stamps semantically, invents roads, or claims a global beauty optimum.

Usage and handoff
------------------
``python tools/cityforge/fit_intent_sketch.py --bundle <bundle> --intent
<intent.json> --out <fresh-output-dir>``

The outputs are ``intent.copy.json``, ``resolved.sketch.json``, and
``fit_report.json``.  The resolved sketch is consumed separately by
``plan_sketch.py``; do **not** pass ``--auto-face`` to that command because it
would overwrite the explicit door-target transform solved here.  Unsolved or
inconclusive fits write no partial resolved lot set and exit nonzero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen import frontage_fit, frontage_targets  # noqa: E402
from procgen.visual_planner_format import canonical_json_bytes  # noqa: E402
from procgen.visual_planner_terrain import TerrainBundleError  # noqa: E402
from plan_sketch import load_bundle, load_products  # noqa: E402
from build_planning_bundle import refuse_unless_fresh  # noqa: E402


class _TerrainAdapter:
    """Adapt TerrainBundle's survey-local methods to world-GU intent space.

    Out-of-survey samples are REJECTED (raised), never clamped to edge tile
    values: ``TerrainBundle.tile_buildable`` itself clamps its tile index, so
    a candidate inside the requested site rectangle but outside the survey
    coverage would otherwise be classified by clamped samples.  The fitter
    converts the raise into the plan §6.4 ``terrain_sample_unresolved``
    rejection, keeping the "check scope before sampling" rule fail-closed.
    """

    def __init__(self, terrain: Any, rectangle_gu: Sequence[float]) -> None:
        self.terrain = terrain
        self.rectangle_gu = tuple(float(value) for value in rectangle_gu)
        self.origin_gu = terrain.origin_gu

    def _local(self, x: float, y: float) -> tuple[float, float]:
        return float(x) - self.origin_gu[0], float(y) - self.origin_gu[1]

    def _check_survey(self, local: Sequence[float]) -> None:
        span_x, span_y = self.terrain.site_span_gu
        if not (0.0 <= float(local[0]) <= float(span_x) and
                0.0 <= float(local[1]) <= float(span_y)):
            raise TerrainBundleError(
                f"terrain sample ({float(local[0]):g}, {float(local[1]):g}) "
                f"lies outside survey coverage")

    def water_at(self, x: float, y: float) -> bool:
        local = self._local(x, y)
        self._check_survey(local)
        return bool(self.terrain.water_at(*local))

    def buildable_at(self, x: float, y: float) -> bool:
        local = self._local(x, y)
        self._check_survey(local)
        return bool(self.terrain.tile_buildable(*local))


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label} {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _world_targets(
    site: Mapping[str, Any], intent: Mapping[str, Any], terrain: Any, network: Any
) -> dict[str, dict[str, Any]]:
    """Build source/authored targets in the intent's absolute world-GU frame."""

    origin = terrain.origin_gu
    plan_targets = frontage_targets.build_target_map(site, origin, network)
    targets: dict[str, dict[str, Any]] = {}
    for target_id, target in plan_targets.items():
        row = dict(target)
        if isinstance(row.get("polyline"), list):
            row["polyline"] = [
                [float(point[0]) + origin[0], float(point[1]) + origin[1]]
                for point in row["polyline"]
            ]
        targets[target_id] = row
    for road in intent.get("roads", []):
        targets[road["id"]] = {
            "kind": "authored_road" if road["kind"] == "street" else "alley",
            "polyline": [[float(point[0]), float(point[1])] for point in road["points"]],
            "width_gu": float(road["width_gu"]),
        }
    for space in intent.get("spaces", []):
        targets[space["id"]] = {
            "kind": "road_surface_polygon" if space["kind"] == "plaza" else "shared_court",
            "polygon": [[float(point[0]), float(point[1])] for point in space["polygon"]],
            "width_gu": 0.0,
        }
    return targets


def _stamp_ids_and_geometry(
    stamps: Mapping[str, Any], geometry: Mapping[str, Mapping[str, Any]]
) -> tuple[set[str], dict[str, Mapping[str, Any]]]:
    """Return eligible ids plus library geometry; compact rows only gate identity."""

    ids = {
        str(entry.get("id")) for entry in stamps.get("stamps", [])
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
    }
    return ids, {stamp_id: geometry[stamp_id] for stamp_id in sorted(ids) if stamp_id in geometry}


def run(bundle_dir: Path, intent_path: Path, out_dir: Path) -> tuple[int, dict[str, Any] | None]:
    site, stamps, manifest, _ = load_bundle(bundle_dir)
    raw_bytes = intent_path.read_bytes()
    intent = _load_json(intent_path, "intent")
    terrain, network, geometry = load_products(manifest)
    stamp_ids, geometry = _stamp_ids_and_geometry(stamps, geometry)
    targets = _world_targets(site, intent if isinstance(intent, Mapping) else {}, terrain, network)
    terrain_adapter = _TerrainAdapter(terrain, site["rectangle_gu"])
    intent_copy, resolved, report = frontage_fit.fit_intent(
        intent,
        site_name=str(site["site_name"]),
        site_rect=site["rectangle_gu"],
        stamp_ids=stamp_ids,
        stamp_geometry=geometry,
        targets=targets,
        terrain=terrain_adapter,
        input_sha256=frontage_fit.input_identity(raw_bytes),
    )
    refuse_unless_fresh(out_dir)
    _write_json(out_dir / "intent.copy.json", intent_copy)
    _write_json(out_dir / "resolved.sketch.json", resolved)
    _write_json(out_dir / "fit_report.json", report)
    if report["status"] != "solved":
        print(f"FAILURE: frontage_fit {report['terminal_failure_code']}", file=sys.stderr)
        return 1, report
    print(f"site: {site['site_name']}  lots: {len(resolved['lots'])}")
    print(f"outputs: {out_dir.resolve()}")
    return 0, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        status, _ = run(args.bundle, args.intent, args.out)
        return status
    except frontage_fit.FrontageFitError as exc:
        print(f"FAILURE: frontage_fit {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary contract
        print(f"FAILURE: frontage_fit {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
