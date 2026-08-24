#!/usr/bin/env python3
"""Compile the configured Phase 1-4 building evidence into Phase 5 products.

Inputs are selected only by the Phase 5 JSON config.  The pure compiler owns
eligibility and policy decisions; this driver performs read-only JSON loading
and deterministic writes to one configured output directory.  It is the
pipeline seam immediately before the future standalone composer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.building_gen.compiler import compile_rule_kit  # noqa: E402
from procgen.building_gen.normalize import canonicalize  # noqa: E402


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else WORKSPACE / candidate


def _load_documents(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    documents: dict[str, Any] = {}
    for key, value in inputs.items():
        if key == "source_sites":
            continue
        documents[key] = _read(_path(value))
    for site in inputs["source_sites"]:
        site_id = str(site["site_id"])
        documents[f"stamps:{site_id}"] = _read(_path(site["stamp_library"]))
        documents[f"templates:{site_id}"] = _read(_path(site["templates"]))
        documents[f"connections:{site_id}"] = _read(_path(site["connections"]))
    return documents


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(canonicalize(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile the Phase 5 building rule kit")
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE / "configs/kits/xfa_sky_nord_house/phase05_config.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else WORKSPACE / args.config
    try:
        config = _read(config_path)
        documents = _load_documents(config)
        palette_path = _path(config["palette"])
        palette_document = _read(palette_path)
        compiled, eligibility, resolution = compile_rule_kit(config, documents, palette_document)
        output_dir = args.output_dir if args.output_dir is not None else _path(config["outputs"]["root"])
        if not output_dir.is_absolute():
            output_dir = WORKSPACE / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        configured_outputs = (
            ("compiled", compiled),
            ("eligibility", eligibility),
            ("resolution", resolution),
        )
        for output_key, value in configured_outputs:
            configured_path = _path(config["outputs"][output_key])
            destination = output_dir / configured_path.name if args.output_dir is not None else configured_path
            _write(destination, value)
        try:
            display_output_dir = str(output_dir.relative_to(WORKSPACE)).replace("\\", "/")
        except ValueError:
            display_output_dir = str(output_dir)
        print(json.dumps({
            "output_dir": display_output_dir,
            "compiled_rows": compiled["counts"],
            "eligibility": eligibility["counts"],
            "resolution_requests": len(resolution["requests"]),
        }, sort_keys=True))
        return 0
    except Exception as exc:  # explicit pipeline failure protocol for the CLI
        print(f"FAILURE: phase05 {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
