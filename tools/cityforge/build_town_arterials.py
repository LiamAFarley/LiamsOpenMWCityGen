"""Build the Phase 21 Stage A checkpoint from current R1 and R2 inputs.

Builds one main-road tree around a meeting selected from the current Voronoi
fabric, then writes ``arterials.json`` and its topology/terrain review renders.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.arterial_routes import build_arterials  # noqa: E402
from procgen.townlayout.checkpoint import read_checkpoint, write_checkpoint  # noqa: E402
from procgen.townlayout.road_review import render_arterials  # noqa: E402
from procgen.townlayout.validate import TownLayoutError  # noqa: E402

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 21 R2A arterials")
    parser.add_argument("--macro", required=True)
    parser.add_argument("--ports", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print("FAILURE: A output directory is not empty", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        macro = read_checkpoint(args.macro, expected_stages=("r1",))
        ports = read_checkpoint(args.ports, expected_stages=("r2",))
        product = build_arterials(macro, ports)
        product["preceding_checkpoint"] = str(Path(args.ports).resolve())
        write_checkpoint(product, out / "arterials.json")

        survey_path = (macro.get("identities") or {}).get("survey", {}).get("path")
        if not survey_path:
            raise TownLayoutError("A render: survey identity missing")
        render_arterials(product, survey_path,
                         out / "arterials_topology.png",
                         out / "arterials_terrain.png")
    except (TownLayoutError, OSError, ValueError, KeyError) as exc:
        print(f"FAILURE: A {exc}", file=sys.stderr)
        return 1
    metrics = product["metrics"]
    print(f"r2a_arterials: ports={len(product['ports'])} "
          f"tree V={metrics['tree_node_count']} E={metrics['tree_edge_count']} "
          f"runtime={metrics['runtime_s']:.2f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
