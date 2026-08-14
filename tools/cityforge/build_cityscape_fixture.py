"""Build and prove the deterministic synthetic Cityforge T1.3 fixture.

Pipeline position
------------------
This bounded harness runs the T1.3 hard landscape engine twice against the
same accepted T1.1/T1.2 products, real Falkreath ``tamriel.esm`` LAND, and the
accepted live remap ESP.  It installs the canonical product only after both
clean runs have identical recursive file hashes.

The fixture is explicitly ``synthetic_not_a_falkreath_design``.  It exists to
prove stitching, edit gates, VNML parity, VTEX/LTEX closure, tes3json assembly,
and T1.2 final re-seat.  It does not invent a city plan, render buildings,
author an ESP, or modify original mod files.

Outputs
-------
``output/cityforge/phase1/t1_3_cityscape_fixture/`` contains the products
listed in the T1.3 guide, plus ``determinism.json`` with the two complete hash
maps and the measured gate summary.  A failure exits with the required
``FAILURE: cityscape ...`` protocol and leaves no claimed degraded result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.censusio import write_deterministic  # noqa: E402
from procgen.cityscape import CityscapeError, build_cityscape, default_paths  # noqa: E402
from procgen.cityscape_field import sha256_file  # noqa: E402


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_fixture(root: Path, output_dir: Path | None = None) -> dict[str, object]:
    paths = default_paths(root, output_dir)
    target = paths.output_dir
    if target.exists():
        shutil.rmtree(target)
    # The exact same canonical output path is used for both clean runs so
    # T1.2's path-bearing contract evidence remains byte-stable as well as the
    # numerical products.
    first = build_cityscape(paths)
    first_hashes = _hash_tree(target)
    shutil.rmtree(target)
    second = build_cityscape(paths)
    second_hashes = _hash_tree(target)
    if first_hashes != second_hashes:
        differing = sorted(set(first_hashes) | set(second_hashes))
        differing = [name for name in differing if first_hashes.get(name) != second_hashes.get(name)]
        raise CityscapeError(f"two clean T1.3 builds are not byte-identical: {differing[:12]}")
    evidence = {
        "schema_version": 1,
        "product": "cityforge_t1_3_determinism",
        "diagnostic_scope": "synthetic_not_a_falkreath_design",
        "build_count": 2,
        "byte_identical": True,
        "first_hashes": first_hashes,
        "second_hashes": second_hashes,
        "hashes_equal": first_hashes == second_hashes,
        "source_hashes": second["source_hashes"],
        "vnml_metrics": second["vnml_gate"]["metrics"],
        "vnml_analytic_oracles": second["validation"]["gates"]["vnml_analytic_oracles"],
        "thu_encoding": second["validation"]["gates"]["terrain_edits"]["final"]["encoding"],
        "t12_final_reseat": second["t12_evidence"],
        "t12_planned_pass": second["t12_planned_evidence"],
        "paint_counts": second["paint"].paint_ledger,
        "synthetic_diagnostic_case_count": len(second["validation"]["gates"]["synthetic_edit_diagnostics"]),
    }
    write_deterministic(target / "determinism.json", evidence)
    # Keep the engine manifest in the compared hash set.  The determinism
    # evidence is an additive sibling rather than a self-referential manifest
    # entry (including each other's hashes would require an impossible cycle).
    print(
        "cityscape fixture PASS "
        f"files={len(first_hashes)} "
        f"source_cells={second['validation']['gates']['source_stitch']['cell_count']} "
        f"max_delta_thu={second['validation']['gates']['terrain_edits']['final']['encoding']['max_abs_encoded_delta_thu']} "
        f"vnml_p95={second['vnml_gate']['metrics']['p95_angle_deg']:.6f} "
        f"paint_support={second['paint'].paint_ledger['support_tile_count']} "
        f"t12={second['t12_evidence']['status']}"
    )
    return {"first": first, "second": second, "hashes": first_hashes, "evidence": evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the synthetic Cityforge T1.3 landscape proof")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    try:
        run_fixture(Path(args.workspace_root).resolve(), Path(args.output_dir).resolve() if args.output_dir else None)
        return 0
    except Exception as exc:
        print(f"FAILURE: cityscape {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
