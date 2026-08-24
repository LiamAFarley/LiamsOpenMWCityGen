#!/usr/bin/env python3
"""Assemble a Falkreath kit house stamp from native NIF bounds.

Texture variants (_a/_b/_c) share family placement. Any window id can be
mixed onto any shell via --window.

Usage::

    python tools/cityforge/generate_fk_house.py --shell sky_FK_house_01_a --render

    python tools/cityforge/generate_fk_house.py --shell sky_FK_house_02_b ^
        --window sky_FK_Window_01a --render

    python tools/cityforge/generate_fk_house.py --shell sky_FK_house_10_a ^
        --doors neg_y --windows pos_y,neg_x,pos_x --render

    python tools/cityforge/generate_fk_house.py --shell sky_FK_house_01_a ^
        --secondary-doors pos_y --render

    python tools/cityforge/generate_fk_house.py --shell sky_FK_house_01_a ^
        --porch sky_FK_Porch_02a --porch-facades neg_x --render

    python tools/cityforge/generate_fk_house.py --shell sky_FK_house_04_a ^
        --porch sky_FK_Porch_01a --porch-facades inner_neg_y ^
        --stair sky_ex_mk_str_02 --stair-facades inner_neg_y --render

    python tools/cityforge/generate_fk_house.py --all-pilots --render
"""

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

from procgen.fk_house import (  # noqa: E402
    SHELL_SPECS,
    DORMER_MODELS,
    WINDOW_MODELS,
    canonical_json_bytes,
    generate_fk_house,
)

OUT_DIR = WORKSPACE / "output" / "cityforge" / "stamps" / "generated" / "falkreath"


def _write_stamp(
    shell_id: str,
    out: Path,
    *,
    door_facades: list[str] | None = None,
    secondary_door_facades: list[str] | None = None,
    window_facades: list[str] | None = None,
    window_model: str | None = None,
    door_model: str | None = None,
    porch_facades: list[str] | None = None,
    porch_model: str | None = None,
    stair_facades: list[str] | None = None,
    stair_model: str | None = None,
    dormer_attachments: list[dict] | None = None,
    legacy_placement: bool = False,
    stamp_id: str | None = None,
) -> Path:
    stamp = generate_fk_house(
        shell_id,
        generated_id=stamp_id or f"generated__{out.stem}",
        door_facades=door_facades,
        secondary_door_facades=secondary_door_facades,
        window_facades=window_facades,
        window_model=window_model,
        door_model=door_model,
        porch_facades=porch_facades,
        porch_model=porch_model,
        stair_facades=stair_facades,
        stair_model=stair_model,
        dormer_attachments=dormer_attachments,
        use_wall_profiles=not legacy_placement,
        allow_review_profiles=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical_json_bytes(stamp))
    print(
        f"wrote {out} members={len(stamp['members'])} doors={stamp['door_count']} "
        f"shell={stamp['source']['shell_id']} window={stamp['source']['window_model']} "
        f"door_faces={stamp['source']['door_facades']} "
        f"window_faces={stamp['source']['window_facades']}"
    )
    return out


def _render(stamp_path: Path, sheet_path: Path) -> int:
    command = [
        sys.executable,
        str(WORKSPACE / "tools" / "cityforge" / "render_generated_house.py"),
        "--stamp",
        str(stamp_path),
        "--out",
        str(sheet_path),
    ]
    print("running:", " ".join(command))
    return subprocess.run(command, cwd=WORKSPACE, check=False).returncode


def _parse_optional_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Falkreath kit house stamp")
    parser.add_argument("--shell", default=None, help="shell id, e.g. sky_FK_house_01_a")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--doors", default=None, help="comma-separated facades: neg_x,pos_y")
    parser.add_argument(
        "--secondary-doors",
        default=None,
        help="additional family-valid side/rear facades: pos_x,neg_y",
    )
    parser.add_argument("--windows", default=None, help="comma-separated facades to receive windows")
    parser.add_argument(
        "--window",
        default=None,
        help=f"window mesh id to mix onto the shell; known: {', '.join(sorted(WINDOW_MODELS))}",
    )
    parser.add_argument("--door", default=None, help="door mesh id, e.g. sky_ex_fk_door_01")
    parser.add_argument(
        "--porch",
        default=None,
        help="porch mesh id; Porch_02 variants include integrated steps",
    )
    parser.add_argument(
        "--porch-facades",
        default=None,
        help="comma-separated generated door facades receiving the porch",
    )
    parser.add_argument(
        "--stair",
        default=None,
        help="standalone stair mesh id, e.g. sky_ex_mk_str_02",
    )
    parser.add_argument(
        "--stair-facades",
        default=None,
        help="comma-separated generated door facades receiving standalone stairs",
    )
    parser.add_argument(
        "--dormers-json",
        type=Path,
        default=None,
        help="JSON list of explicit dormer attachments",
    )
    parser.add_argument(
        "--legacy-placement",
        action="store_true",
        help="review-only placement using family wall bounds without a measured wall profile",
    )
    parser.add_argument("--all-pilots", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.all_pilots:
        jobs = [
            ("sky_FK_house_01_a", None, None, None, None, "sky_FK_house_01_a"),
            ("sky_FK_house_02_a", None, None, None, None, "sky_FK_house_02_a"),
            ("sky_FK_house_03_a", None, None, None, None, "sky_FK_house_03_a"),
            ("sky_FK_house_04_a", None, None, None, None, "sky_FK_house_04_a"),
            ("sky_FK_house_05_a", None, None, None, None, "sky_FK_house_05_a"),
            ("sky_FK_house_06_a", None, None, None, None, "sky_FK_house_06_a"),
            ("sky_FK_house_07_a", None, None, None, None, "sky_FK_house_07_a"),
            ("sky_FK_house_08_a", None, None, None, None, "sky_FK_house_08_a"),
            ("sky_FK_house_09_a", None, None, None, None, "sky_FK_house_09_a"),
            ("sky_FK_house_10_a", None, None, None, None, "sky_FK_house_10_a"),
            ("sky_FK_house_11_a", None, None, None, None, "sky_FK_house_11_a"),
            ("sky_FK_house_12_a", None, None, None, None, "sky_FK_house_12_a"),
        ]
        for shell_id, doors, windows, window_model, door_model, name in jobs:
            if shell_id not in SHELL_SPECS:
                print(f"FAILURE: unknown shell {shell_id}", file=sys.stderr)
                return 1
            stamp_path = OUT_DIR / f"{name}.json"
            _write_stamp(
                shell_id,
                stamp_path,
                door_facades=_parse_optional_list(doors),
                secondary_door_facades=None,
                window_facades=_parse_optional_list(windows),
                window_model=window_model,
                door_model=door_model,
                porch_facades=None,
                porch_model=None,
                stair_facades=None,
                stair_model=None,
                stamp_id=f"generated__{name}",
            )
            if args.render:
                code = _render(stamp_path, OUT_DIR / f"{name}_sheet_2x3.png")
                if code != 0:
                    print(f"FAILURE: render {name} exit {code}", file=sys.stderr)
                    return code
        return 0

    if not args.shell:
        parser.error("provide --shell or --all-pilots")
    dormer_attachments = None
    if args.dormers_json is not None:
        payload = json.loads(args.dormers_json.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            parser.error("--dormers-json must contain a JSON list of attachment objects")
        dormer_attachments = payload
    out = args.out or OUT_DIR / f"{args.shell}.json"
    _write_stamp(
        args.shell,
        out,
        door_facades=_parse_optional_list(args.doors),
        secondary_door_facades=_parse_optional_list(args.secondary_doors),
        window_facades=_parse_optional_list(args.windows),
        window_model=args.window,
        door_model=args.door,
        porch_facades=_parse_optional_list(args.porch_facades),
        porch_model=args.porch,
        stair_facades=_parse_optional_list(args.stair_facades),
        stair_model=args.stair,
        dormer_attachments=dormer_attachments,
        legacy_placement=args.legacy_placement,
    )
    if args.render:
        return _render(out, out.with_name(out.stem + "_sheet_2x3.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
