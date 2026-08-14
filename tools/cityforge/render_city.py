#!/usr/bin/env python3
"""Host-side CLI for the Cityforge T1.5 terrain-backed Blender render.

The CLI is the public pipeline entry point.  It validates accepted T1.1/T1.2/
T1.3 products, makes a labelled *render-only* masterless scratch ESP from a
copy of T1.3 ``land_records.json``, writes the deterministic scene contract,
invokes the Blender worker, and performs the independent final PNG/audit gate.

It never authors a production plugin and never writes into ``C:\\Modding`` or a
configured data root.  The synthetic fixture driver calls this CLI twice from
fresh temporary output directories to prove scene/audit determinism before
installing one successful run into the canonical output fixture.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen import cityrender, tes3json  # noqa: E402


DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe")
DEFAULT_TES3CONV = WORKSPACE / "tes3conv-master" / "tes3conv.exe"
DEFAULT_SCRATCH_ROOT = Path(tempfile.gettempdir()) / "cityforge_t15_render_scratch"
DEFAULT_OUTPUT = WORKSPACE / "output" / "cityforge" / "phase1" / "t1_5_render_fixture"


class RenderCliFailure(RuntimeError):
    """Raised before the CLI can claim a complete render/audit."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RenderCliFailure(message)


def _resolve(value: Path | str) -> Path:
    return Path(value).expanduser().resolve()


def _fresh_output(output_dir: Path, read_only_roots: Sequence[Path]) -> None:
    """Reject unsafe or pre-existing destinations before making any output."""

    output = _resolve(output_dir)
    require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    require(not output.exists(), f"output directory must be fresh and absent: {output}")
    protected_roots = [_resolve(r"C:\Modding"), *[_resolve(root) for root in read_only_roots]]
    output_key = os.path.normcase(os.path.normpath(str(output)))
    for protected in protected_roots:
        root_key = os.path.normcase(os.path.normpath(str(protected)))
        try:
            inside = os.path.commonpath((output_key, root_key)) == root_key
        except ValueError:
            inside = False
        require(not inside, f"output directory is inside read-only root: {protected}")
    output.mkdir(parents=True)


def _copy_and_convert_land_records(
    land_records: Path,
    *,
    tes3conv: Path,
    scratch_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create a labelled masterless render-only ESP from a scratch JSON copy.

    ``land_records`` remains untouched.  Both the forward conversion and the
    JSON round-trip are required; a failed converter or non-masterless header
    is a hard pipeline failure, not a reason to render from JSON directly.
    """

    require(land_records.is_file(), f"T1.3 land_records.json is missing: {land_records}")
    require(tes3conv.is_file(), f"tes3conv executable is missing: {tes3conv}")
    scratch = _resolve(scratch_root)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True)
    source_copy = scratch / "t1_3_land_records_copy.json"
    plugin = scratch / "synthetic_not_a_falkreath_design_render_only_masterless.esp"
    roundtrip = scratch / "roundtrip.json"
    shutil.copy2(land_records, source_copy)
    forward = subprocess.run(
        [str(tes3conv), str(source_copy), str(plugin), "-o"],
        cwd=str(scratch),
        capture_output=True,
        text=True,
        check=False,
    )
    require(forward.returncode == 0, "tes3conv JSON->render-only ESP failed: " + (forward.stderr or forward.stdout).strip())
    require(plugin.is_file() and plugin.stat().st_size > 1024, "tes3conv did not write a non-trivial render-only ESP")
    reverse = subprocess.run(
        [str(tes3conv), str(plugin), str(roundtrip), "-o"],
        cwd=str(scratch),
        capture_output=True,
        text=True,
        check=False,
    )
    require(reverse.returncode == 0, "tes3conv render-only ESP->JSON round-trip failed: " + (reverse.stderr or reverse.stdout).strip())
    roundtrip_doc = json.loads(roundtrip.read_text(encoding="utf-8"))
    require(isinstance(roundtrip_doc, list), "tes3conv round-trip JSON is not a top-level array")
    issues = tes3json.validate(roundtrip_doc)
    require(not issues, "tes3conv round-trip JSON fails tes3json.validate: " + "; ".join(map(str, issues[:5])))
    headers = [row for row in roundtrip_doc if row.get("type") == "Header"]
    require(len(headers) == 1 and headers[0].get("masters") == [], "render-only scratch ESP round-trip is not masterless")
    return plugin, {
        "scratch_dir": str(scratch),
        "source_copy": str(source_copy),
        "plugin": str(plugin),
        "roundtrip_json": str(roundtrip),
        "plugin_sha256": cityrender.sha256_file(plugin),
        "roundtrip_json_sha256": cityrender.sha256_file(roundtrip),
        "roundtrip_record_count": len(roundtrip_doc),
        "forward_stdout": forward.stdout.strip(),
        "forward_stderr": forward.stderr.strip(),
        "reverse_stdout": reverse.stdout.strip(),
        "reverse_stderr": reverse.stderr.strip(),
        "masterless": True,
        "label": "SYNTHETIC ENGINE FIXTURE render-only scratch terrain plugin; not production content",
    }


def _build_paths(args: argparse.Namespace) -> cityrender.RenderInputPaths:
    defaults = cityrender.default_render_input_paths(args.workspace_root)
    def choose(name: str, default: Path) -> Path:
        value = getattr(args, name)
        return _resolve(value) if value is not None else default
    return cityrender.RenderInputPaths(
        workspace_root=_resolve(args.workspace_root),
        plan=choose("plan", defaults.plan),
        validation=choose("validation", defaults.validation),
        placement=choose("placement", defaults.placement),
        placement_manifest=choose("placement_manifest", defaults.placement_manifest) if (getattr(args, "placement_manifest", None) is not None or defaults.placement_manifest is not None) else None,
        t1_3_manifest=choose("t1_3_manifest", defaults.t1_3_manifest) if (getattr(args, "t1_3_manifest", None) is not None or defaults.t1_3_manifest is not None) else None,
        t1_3_validation=choose("t1_3_validation", defaults.t1_3_validation) if (getattr(args, "t1_3_validation", None) is not None or defaults.t1_3_validation is not None) else None,
        land_records=choose("land_records", defaults.land_records),
        final_field=choose("final_field", defaults.final_field),
        final_field_metadata=choose("final_field_metadata", defaults.final_field_metadata),
        procgen_config=choose("procgen_config", defaults.procgen_config),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run host validation, Blender worker, and independent output audit."""

    require(
        bool(args.synthetic),
        "T1.5 is synthetic-only in this dispatch; pass --synthetic and do not provide a real Falkreath plan",
    )
    paths = _build_paths(args)
    # configured_data_roots is imported by cityrender's core; this CLI only
    # needs its resolver roots for the output safety guard and keeps the actual
    # validation in one place.
    from procgen.meshcheck import configured_data_roots

    data_roots = configured_data_roots(paths.procgen_config)
    output_dir = _resolve(args.output_dir)
    _fresh_output(output_dir, data_roots)
    plugin, scratch_audit = _copy_and_convert_land_records(
        paths.land_records,
        tes3conv=_resolve(args.tes3conv),
        scratch_root=_resolve(args.scratch_root),
    )
    scene = cityrender.build_render_scene(paths, scratch_plugin=plugin, synthetic=bool(args.synthetic))
    scene["scratch_conversion"] = scratch_audit
    # The conversion evidence is part of the scene contract, but the identity
    # hash is based on the validated scratch plugin already included by the
    # core.  It is not recomputed from an output directory path.
    cityrender.write_json(output_dir / "render_scene.json", scene)
    cityrender.write_json(output_dir / "scene_manifest.json", scene)

    blender = _resolve(args.blender)
    require(blender.is_file(), f"Blender executable is missing: {blender}")
    worker = WORKSPACE / "tools" / "cityforge" / "blender_render_city.py"
    require(worker.is_file(), f"Blender worker is missing: {worker}")
    command = [
        str(blender),
        "-b",
        "--python",
        str(worker),
        "--",
        "--scene",
        str(output_dir / "render_scene.json"),
        "--output-dir",
        str(output_dir),
    ]
    worker_run = subprocess.run(command, cwd=str(WORKSPACE), check=False)
    require(worker_run.returncode == 0, f"Blender worker failed with exit code {worker_run.returncode}")
    for view in scene["views"]:
        cityrender.normalize_png_bytes(output_dir / str(view["file"]))
    worker_path = output_dir / "blender_worker_audit.json"
    require(worker_path.is_file(), "Blender worker completed without blender_worker_audit.json")
    worker_audit = json.loads(worker_path.read_text(encoding="utf-8"))
    require(isinstance(worker_audit, Mapping), "Blender worker audit is not an object")
    final_audit = cityrender.finalize_render_audit(scene, output_dir, worker_audit)
    final_audit["scratch_conversion"] = scratch_audit
    cityrender.write_json(output_dir / "render_audit.json", final_audit)
    summary = {
        "stage": cityrender.STAGE,
        "output_dir": str(output_dir),
        "build_hash": scene["build_hash"],
        "render_count": len(final_audit["images"]),
        "dimensions": {row["file"]: [row["width"], row["height"]] for row in final_audit["images"]},
        "scene_manifest_sha256": cityrender.sha256_file(output_dir / "scene_manifest.json"),
        "render_audit_sha256": cityrender.sha256_file(output_dir / "render_audit.json"),
        "scratch_plugin_sha256": scratch_audit["plugin_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parser() -> argparse.ArgumentParser:
    defaults = cityrender.default_render_input_paths(WORKSPACE)
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--plan", type=Path)
    result.add_argument("--validation", type=Path)
    result.add_argument("--placement", type=Path)
    result.add_argument("--placement-manifest", type=Path, default=defaults.placement_manifest)
    result.add_argument("--t1-3-manifest", type=Path, default=defaults.t1_3_manifest)
    result.add_argument("--t1-3-validation", type=Path, default=defaults.t1_3_validation)
    result.add_argument("--land-records", type=Path)
    result.add_argument("--final-field", type=Path)
    result.add_argument("--final-field-metadata", type=Path)
    result.add_argument("--procgen-config", type=Path, default=defaults.procgen_config)
    result.add_argument("--tes3conv", type=Path, default=DEFAULT_TES3CONV)
    result.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    result.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    result.add_argument("--synthetic", action="store_true", help="label this downstream render as the synthetic non-Falkreath proof fixture")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"FAILURE: render_city {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
