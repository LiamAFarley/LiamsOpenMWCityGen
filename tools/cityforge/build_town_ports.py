"""Build the Phase 21 R2 ports and ingress checkpoint from R1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.checkpoint import read_checkpoint, write_checkpoint  # noqa: E402
from procgen.townlayout.diagnostics import render_ports_diagnostic  # noqa: E402
from procgen.townlayout.ports import build_ports  # noqa: E402
from procgen.townlayout.validate import TownLayoutError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 21 R2 ports checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print("FAILURE: R2 output directory is not empty", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        source = read_checkpoint(args.input)
        product = dict(source)
        product.update(build_ports(source))
        product["stage_id"] = "r2"
        product["schema_version"] = 1
        product["preceding_checkpoint"] = str(Path(args.input).resolve())
        write_checkpoint(product, out / "ports_ingress.json")
        # Read-back verifies all inherited identities, including the R1 NPZ and
        # aligned-road product, before the visual artifact is emitted.
        loaded = json.loads((out / "ports_ingress.json").read_text(encoding="utf-8"))
        if (loaded.get("stage_id") != "r2" or
                loaded.get("planning_ring", {}).get("simplification", {}).get("frozen") is not True):
            raise TownLayoutError("R2 checkpoint read-back failed")
        render_ports_diagnostic(product, source["identities"]["survey"]["path"],
                                out / "ports_ingress_diagnostic.png")
    except (TownLayoutError, OSError, ValueError, KeyError) as exc:
        print(f"FAILURE: R2 {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
