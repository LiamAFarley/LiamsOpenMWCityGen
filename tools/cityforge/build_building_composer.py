"""Run the config-driven Phase 6 minimum composer against the real compiled kit.

Inputs are the Phase 5 compiled JSON and complete base/extension requests from
the Phase 6 config. Outputs are canonical JSON evidence only; this driver does
not import Blender, modify source data, or author an ESP.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.building_gen.composer import compose_base, compose_extension
from procgen.censusio import write_deterministic


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else WORKSPACE / candidate


def run(config: dict[str, Any], output_dir: Path | None = None) -> dict[str, Any]:
    kit = _load(_path(config["inputs"]["compiled_kit"]))
    base_results: list[dict[str, Any]] = []
    by_request: dict[str, dict[str, Any]] = {}
    for request in config["base_requests"]:
        result = compose_base(request, kit)
        row = {"request_id": request["request_id"], "status": "accepted", "building": result, "rejections": []}
        base_results.append(row)
        by_request[str(request["request_id"])] = result
    extension_results: list[dict[str, Any]] = []
    for configured in config["extension_requests"]:
        request = copy.deepcopy(dict(configured))
        previous_id = str(request.pop("previous_request_id"))
        request["previous_generated_building"] = copy.deepcopy(by_request[previous_id])
        result = compose_extension(request, kit)
        extension_results.append(result)
    products = {
        "base": {"schema_version": 1, "phase": 6, "results": base_results},
        "extensions": {"schema_version": 1, "phase": 6, "results": extension_results},
    }
    destinations = {
        "base": _path(config["outputs"]["base"]),
        "extensions": _path(config["outputs"]["extensions"]),
    }
    for key, payload in products.items():
        destination = output_dir / destinations[key].name if output_dir is not None else destinations[key]
        write_deterministic(destination, payload)
    summary = {
        "schema_version": 1,
        "phase": 6,
        "base_count": len(base_results),
        "extension_count": len(extension_results),
        "base_statuses": [row["status"] for row in base_results],
        "extension_statuses": [row["status"] for row in extension_results],
    }
    destination = output_dir / _path(config["outputs"]["summary"]).name if output_dir is not None else _path(config["outputs"]["summary"])
    write_deterministic(destination, summary)
    return products | {"summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _load(args.config)
    run(config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
