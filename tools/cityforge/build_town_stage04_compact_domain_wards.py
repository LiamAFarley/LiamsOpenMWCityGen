"""Build the Stage 04 compact-domain/contiguous-wards checkpoint.

Inputs are the normal site survey, field grid, census and TownBrief.  The CLI
executes only patches -> compact domain -> market reservation -> ward BFS and
writes ``domain_wards.json`` plus the zoomed diagnostic overlay.  The product
also carries strict ``brief_provenance``: the exact input brief's town ID,
target triple, and SHA-256.  Downstream checkpoints must not infer brief
identity from an output directory name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PIL import Image, ImageDraw  # noqa: E402
from procgen.townlayout.anchors import place_anchors  # noqa: E402
from procgen.townlayout.approaches import build_rewrite_domain  # noqa: E402
from procgen.townlayout.domain import grow_city_domain  # noqa: E402
from procgen.townlayout.patches import generate_organic_patches  # noqa: E402
from procgen.townlayout.site_context import (  # noqa: E402
    _plan_to_px, build_site_context, diagnostic_view, resolve_topdown_png,
)
from procgen.townlayout.validate import TownLayoutError  # noqa: E402
from procgen.townlayout.wards import assign_wards  # noqa: E402


def brief_provenance(brief: dict, brief_bytes: bytes) -> dict:
    """Return the required, serialized identity of the Stage 04 brief."""
    targets = brief.get("target_buildings")
    if not isinstance(targets, dict) or any(k not in targets for k in ("min", "preferred", "max")):
        raise TownLayoutError("brief_provenance: target_buildings must contain min/preferred/max")
    town_id = brief.get("town_id")
    if not isinstance(town_id, str) or not town_id:
        raise TownLayoutError("brief_provenance: town_id is required")
    return {
        "town_id": town_id,
        "target_buildings": {k: int(targets[k]) for k in ("min", "preferred", "max")},
        "sha256": hashlib.sha256(brief_bytes).hexdigest(),
    }


def build_parser():
    p = argparse.ArgumentParser(description="Build Stage 04 townlayout checkpoint")
    for name in ("survey", "fields", "census", "brief"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--out-dir", dest="out_dir", required=True)
    p.add_argument("--candidate-id", default="c00")
    return p


def _diagnostic(ctx, product, survey_path: Path, out: Path):
    topdown = resolve_topdown_png(survey_path)
    if topdown is None:
        raise TownLayoutError("missing_diagnostic_input: site_topdown.png")
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    # Stage 04 acceptance is a full-resolution land/water inspection, not a
    # cropped thumbnail that can hide ordinary-land holes at the viewport edge.
    image, mapping = diagnostic_view({"_diagnostic_bounds": [product.get("city_domain") or []]},
                                     topdown, survey, full_site=True)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0)); draw = ImageDraw.Draw(overlay, "RGBA")
    colors = {"market": (220, 40, 160, 115), "craft": (220, 120, 30, 100),
              "residential": (40, 170, 80, 100), "outskirts": (180, 140, 40, 100),
              "keep": (80, 80, 80, 120)}
    wards = {pid: ward["ward_type"] for ward in product.get("wards", []) for pid in ward["patch_ids"]}
    to_px = lambda p: _plan_to_px(float(p[0]), float(p[1]), mapping)
    for patch in product.get("patches", []):
        ring = patch.get("polygon") or []
        if len(ring) >= 3 and patch.get("inside_city"):
            kind = wards.get(patch["patch_id"], "residential")
            draw.polygon([to_px(p) for p in ring], fill=colors[kind], outline=(20, 20, 20, 210))
    for anchor in product.get("anchors", []):
        ring = anchor.get("polygon") or []
        if len(ring) >= 3:
            draw.polygon([to_px(p) for p in ring], fill=(255, 0, 220, 150), outline=(255, 255, 255, 255))
    ring = product.get("city_domain") or []
    if len(ring) >= 3: draw.line([to_px(p) for p in ring + [ring[0]]], fill=(0, 230, 255, 255), width=3)
    image = Image.alpha_composite(image, overlay)
    image.save(out / "domain_wards_diagnostic.png")


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print("FAILURE: compact-domain-wards output directory is not empty", file=sys.stderr); return 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        brief_path = Path(args.brief)
        brief_bytes = brief_path.read_bytes()
        brief = json.loads(brief_bytes.decode("utf-8"))
        ctx = build_site_context(survey_json=Path(args.survey), fields_npz=Path(args.fields),
                                 census_json=Path(args.census), town_brief=brief)
        patches = generate_organic_patches(ctx, build_rewrite_domain(ctx), brief,
                                           master_seed=int(brief["master_seed"]), candidate_id=args.candidate_id)
        product = place_anchors(ctx, grow_city_domain(ctx, patches, brief), brief,
                                candidate_id=args.candidate_id)
        product = assign_wards(product, brief, candidate_id=args.candidate_id)
        product["brief_provenance"] = brief_provenance(brief, brief_bytes)
        _diagnostic(ctx, product, Path(args.survey), out)
    except TownLayoutError as exc:
        print(f"FAILURE: compact-domain-wards {exc}", file=sys.stderr); return 1
    (out / "domain_wards.json").write_text(json.dumps(product, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
