#!/usr/bin/env python3
"""Build and determinism-check the Cityforge T1.5 synthetic render fixture.

This driver is the complete synthetic proof command.  It invokes the public
``render_city.py`` host/Blender pipeline twice, each time in a fresh temporary
directory, compares every generated file byte-for-byte, and only then copies
the first successful run to
``output/cityforge/phase1/t1_5_render_fixture/``.  The installed directory is
strictly diagnostic: it is not a real Falkreath design and contains no
production ESP.

Inputs are the accepted T1.1/T1.2/T1.3 synthetic products selected by
``render_city.py``.  No plan is authored here and no original mod file is ever
opened for writing.  A failed worker, audit, or repeat comparison exits with
``FAILURE: render_city_fixture ...`` and leaves no degraded fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = WORKSPACE / "output" / "cityforge" / "phase1" / "t1_5_render_fixture"
DEFAULT_REPEAT_ROOT = Path(tempfile.gettempdir()) / "cityforge_t15_fixture_repeats"


class FixtureFailure(RuntimeError):
    """Raised when one complete repeat or the byte determinism gate fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureFailure(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run_once(output_dir: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(WORKSPACE / "tools" / "cityforge" / "render_city.py"),
        "--workspace-root",
        str(WORKSPACE),
        "--output-dir",
        str(output_dir),
        "--scratch-root",
        str(args.scratch_root),
        "--tes3conv",
        str(args.tes3conv),
        "--blender",
        str(args.blender),
        "--synthetic",
    ]
    completed = subprocess.run(command, cwd=str(WORKSPACE), check=False)
    require(completed.returncode == 0, f"complete synthetic render repeat failed with exit code {completed.returncode}: {output_dir}")
    for required in ("render_scene.json", "scene_manifest.json", "blender_worker_audit.json", "render_audit.json"):
        require((output_dir / required).is_file(), f"repeat is missing required audit product {required}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run two complete renders, compare them, and install one canonical proof."""

    output = Path(args.output_dir).expanduser().resolve()
    require(output.parent.is_dir(), f"canonical fixture parent does not exist: {output.parent}")
    require(not output.exists(), f"canonical fixture output must be absent: {output}")
    repeat_root = Path(args.repeat_root).expanduser().resolve()
    if repeat_root.exists():
        shutil.rmtree(repeat_root)
    repeat_root.parent.mkdir(parents=True, exist_ok=True)
    repeat_root.mkdir(parents=True)
    first = repeat_root / "repeat_a"
    second = repeat_root / "repeat_b"
    _run_once(first, args)
    _run_once(second, args)
    hashes_a = file_hashes(first)
    hashes_b = file_hashes(second)
    require(hashes_a == hashes_b, "complete synthetic render repeats are not byte-identical")
    shutil.copytree(first, output)
    evidence = {
        "schema_version": 1,
        "stage": "cityforge_t1_5_render_fixture",
        "diagnostic_scope": "synthetic_not_a_falkreath_design",
        "synthetic_banner": "SYNTHETIC ENGINE FIXTURE — NOT A FALKREATH DESIGN",
        "repeat_a_dir": str(first),
        "repeat_b_dir": str(second),
        "repeat_a_file_hashes": hashes_a,
        "repeat_b_file_hashes": hashes_b,
        "byte_identical": True,
        "canonical_source": "repeat_a",
        "canonical_output": str(output),
        "render_pixel_hashes_compared_same_machine": True,
    }
    (output / "determinism.json").write_text(json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"canonical_output": str(output), "file_count": len(hashes_a), "byte_identical": True}, indent=2, sort_keys=True), flush=True)
    return evidence


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--repeat-root", type=Path, default=DEFAULT_REPEAT_ROOT)
    result.add_argument("--scratch-root", type=Path, default=Path(tempfile.gettempdir()) / "cityforge_t15_render_scratch")
    result.add_argument("--tes3conv", type=Path, default=WORKSPACE / "tes3conv-master" / "tes3conv.exe")
    result.add_argument("--blender", type=Path, default=Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parser().parse_args(argv))
    except Exception as exc:
        print(f"FAILURE: render_city_fixture {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
