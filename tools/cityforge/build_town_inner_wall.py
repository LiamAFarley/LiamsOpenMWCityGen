"""Build the central inner wall before any minor streets are selected.

Reads an accepted Stage-B road-block checkpoint and writes ``inner_wall.json``
plus matching full-town topology/terrain renders. Only major arterial
centerline crossings become wall gates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.checkpoint import read_checkpoint, write_checkpoint  # noqa: E402
from procgen.townlayout.inner_walls import build_inner_wall  # noqa: E402
from procgen.townlayout.road_review import render_inner_wall  # noqa: E402
from procgen.townlayout.validate import TownLayoutError  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build central inner town wall")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--brief", required=False, default=None, help="town brief JSON (when fortification.mode==none, no wall is produced)")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print("FAILURE: W output directory is not empty", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        source = read_checkpoint(args.input, expected_stages=("r2b_road_blocks",))
        product = build_inner_wall(source, args.fraction)
        product["preceding_checkpoint"] = str(Path(args.input).resolve())
        write_checkpoint(product, out / "inner_wall.json")
        survey_path = (product.get("identities") or {}).get("survey", {}).get("path")
        if not survey_path:
            raise TownLayoutError("W render: survey identity missing")
        render_inner_wall(product, survey_path,
                          out / "inner_wall_topology.png", out / "inner_wall_terrain.png")
    except (TownLayoutError, OSError, ValueError, KeyError) as exc:
        print(f"FAILURE: W {exc}", file=sys.stderr)
        return 1
    m = product["metrics"]
    print(f"r2w_inner_wall: fraction={m['actual_fraction']:.3f} "
          f"patches={m['selected_patch_count']} gates={m['gate_count']} "
          f"runtime={m['runtime_s']:.2f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
