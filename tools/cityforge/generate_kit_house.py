#!/usr/bin/env python3
"""Generate and optionally render a house from kit library/grammar JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.kit_house_runtime import generate_from_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--grammar", type=Path, required=True)
    parser.add_argument("--shell", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--door-slots", default=None)
    parser.add_argument("--windows", default=None)
    parser.add_argument("--stairs", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    slots = None if args.door_slots is None else [v.strip() for v in args.door_slots.split(",") if v.strip()]
    windows = None if args.windows is None else [v.strip() for v in args.windows.split(",") if v.strip()]
    stamp = generate_from_json(
        args.library,
        args.grammar,
        shell_id=args.shell,
        door_slot_ids=slots,
        window_facade_ids=windows,
        access_policy={"stairs": args.stairs},
        generated_id=f"generated__{args.shell}__seed{args.seed}",
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} members={len(stamp['members'])} doors={stamp['door_count']}")
    if not args.render:
        return 0
    sheet = args.out.with_name(args.out.stem + "_sheet_2x3.png")
    command = [
        sys.executable,
        str(WORKSPACE / "tools" / "cityforge" / "render_generated_house.py"),
        "--stamp", str(args.out), "--out", str(sheet),
    ]
    return subprocess.run(command, cwd=WORKSPACE, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
