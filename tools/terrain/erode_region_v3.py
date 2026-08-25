"""Run structural continuation, effective erosion, and final seam lock.

The command consumes the real Stage-3 local field for one configured region,
analyzes authoritative owner terrain, solves sparse semantic guides, routes
owner inflow plus generated rainfall, and writes a standardized local review
sheet. It never rewrites the upstream harmonic field and never writes a
global snapshot for each cycle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen import terrain_metrics as tmet  # noqa: E402
from procgen.terrain_blend import build_context, solve_surface  # noqa: E402
from procgen.terrain_erosion import erode_field  # noqa: E402
from procgen.terrain_features import analyze_owner_features  # noqa: E402
from procgen.terrain_hydrology import priority_flood_routing_surface  # noqa: E402
from procgen.terrain_structure import (  # noqa: E402
    build_structural_guides,
    solve_screened_structure,
)
from procgen.terrainfield import (  # noqa: E402
    load_config,
    render_split_window,
    save_shade_png,
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _local_cell_owner(ctx: dict) -> np.ndarray:
    r0, r1, c0, c1 = map(int, ctx["win"])
    cy0, cx0 = r0 // 64, c0 // 64
    cy1 = cy0 + int(np.ceil((r1 - r0) / 64))
    cx1 = cx0 + int(np.ceil((c1 - c0) / 64))
    return ctx["cell_owner"][cy0:cy1, cx0:cx1]


def _render_local(field: np.ndarray, ctx: dict, bbox: tuple[int, int, int, int],
                  cfg: dict) -> np.ndarray:
    r0, r1, c0, c1 = bbox
    return render_split_window(
        field, ctx["own_view"], _local_cell_owner(ctx), ctx["base_code"],
        r0, r1, c0, c1, ctx["render"], pad=64,
    )


def _review_bbox(ctx: dict) -> tuple[int, int, int, int]:
    H, W = ctx["own_view"].shape
    region_bbox = ctx["region"].get("review_bbox_cells")
    if region_bbox:
        bx0, by0, bx1, by1 = map(int, region_bbox)
        r0, _, c0, _ = map(int, ctx["win"])
        return (
            max(0, (by0 - ctx["gy0"]) * 64 - r0),
            min(H, (by1 - ctx["gy0"]) * 64 + 65 - r0),
            max(0, (bx0 - ctx["gx0"]) * 64 - c0),
            min(W, (bx1 - ctx["gx0"]) * 64 + 65 - c0),
        )
    return (0, H, 0, W)


def _final_seam_lock(ctx: dict, eroded: np.ndarray, cfg: dict):
    """Apply the existing narrow direct harmonic reconciliation."""
    v3 = cfg["solve"].get("v3", {})
    ecfg = cfg.get("erosion", {})
    band_cells = float(ecfg.get("final_lock_cells", 2.0))
    active = ctx["smask"] & (ctx["dist_seam"] <= band_cells * 64.0)
    active |= ctx["seam_v"]
    labels, _ = ndimage.label(
        active, structure=ndimage.generate_binary_structure(2, 1)
    )
    seam_labels = np.unique(labels[ctx["seam_v"]])
    active = np.isin(labels, seam_labels)
    interior = ndimage.binary_erosion(
        active, structure=ndimage.generate_binary_structure(2, 1), border_value=0
    )
    ring = active & ~ctx["seam_v"] & ~interior
    hard = ctx["seam_v"] | ring
    hard_vals = eroded.copy()
    hard_vals[ctx["seam_v"]] = ctx["own_view"][ctx["seam_v"]]
    final_ctx = dict(ctx)
    final_ctx.update(
        target=eroded,
        tam_h=eroded,
        smask=active,
        ring_v=ring,
        hard=hard,
        hard_vals=hard_vals,
    )
    final_v3 = dict(v3)
    final_v3["surface"] = dict(v3.get("surface", {}))
    field, report = solve_surface(final_ctx, final_v3)
    report["final_lock_cells"] = band_cells
    report["final_lock_active_vertices"] = int(active.sum())
    report["final_lock_boundary_vertices"] = int(ring.sum())
    return field, report


def _comparison_sheet(paths: list[tuple[str, Path]], out_path: Path) -> None:
    images = [(title, Image.open(path).convert("RGB")) for title, path in paths]
    if not images:
        return
    width = max(img.width for _, img in images)
    label_h = 28
    height = sum(img.height + label_h for _, img in images)
    sheet = Image.new("RGB", (width, height), (15, 15, 15))
    draw = ImageDraw.Draw(sheet)
    y = 0
    for title, image in images:
        draw.text((8, y + 6), title, fill=(255, 235, 150))
        y += label_h
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tamriel_reworked_v1.json"))
    ap.add_argument("--region", default="tr_vvardenfell_wall")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(_resolve(args.config))
    ctx = build_context(ROOT, cfg, args.region)
    solve_dir = _resolve(cfg["paths"]["solve_out_dir"]) / "v3"
    field_path = solve_dir / f"{args.region}_v3_field.npz"
    if not field_path.exists():
        print(f"FAILURE: Stage-3 harmonic field missing: {field_path}")
        return 1
    with np.load(field_path) as z:
        full = z["field"].astype(np.float32)
    r0, r1, c0, c1 = map(int, ctx["win"])
    field = full[r0:r1, c0:c1].copy()
    review = _review_bbox(ctx)
    out_dir = solve_dir / "erosion_structural"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_paths: list[tuple[str, Path]] = []

    def save_stage(title: str, stage_field: np.ndarray, filename: str) -> Path:
        image = _render_local(stage_field, ctx, review, cfg)
        path = out_dir / filename
        save_shade_png(image, path, int(ctx["render"]["px_per_vertex"]), title=title)
        render_paths.append((title, path))
        return path

    stage3_path = save_stage(
        "Stage-3 harmonic base", field, f"{args.region}_stage3_base.png"
    )
    owner_path = save_stage(
        "Owner/reference context", ctx["own_view"], f"{args.region}_owner_reference.png"
    )

    structure_cfg = dict(cfg.get("structure", {}))
    owner_analysis = np.where(ctx["owner_mask"], ctx["owner_field"], np.nan)
    features = analyze_owner_features(
        owner_analysis, ctx["owner_mask"], structure_cfg
    )
    guide_value, guide_weight, guide_report = build_structural_guides(
        field, ctx, features, structure_cfg
    )
    structural_fixed = ctx["hard"].copy()
    structural_fixed_values = ctx["hard_vals"].astype(np.float32, copy=True)
    owner_fixed = ctx["owner_mask"] & ctx["smask"]
    structural_fixed |= owner_fixed
    structural_fixed_values[owner_fixed] = ctx["own_view"][owner_fixed]
    structural, structural_report = solve_screened_structure(
        field, ctx["smask"], structural_fixed, structural_fixed_values,
        guide_value, guide_weight, structure_cfg,
    )
    structural_path = save_stage(
        "Structural continuation before erosion", structural,
        f"{args.region}_structural_pre_erosion.png",
    )

    erosion_cfg = dict(cfg.get("erosion", {}))
    erosion_cfg.update(cfg.get("hydrology", {}))
    erosion_cfg["reroute_every"] = int(cfg.get("hydrology", {}).get(
        "reroute_every", erosion_cfg.get("reroute_every", 2)
    ))
    generated_mask = ctx["smask"] & ~ctx["owner_mask"]
    owner_halo_cells = float(erosion_cfg.get("owner_halo_cells", 10.0))
    owner_halo = ctx["owner_mask"] & (
        ndimage.distance_transform_edt(~ctx["smask"]) <= owner_halo_cells * 64.0
    )
    fixed = ctx["seam_v"] | ctx["ring_v"] | owner_halo
    # The owner mask outside the active corridor participates in hydrology but
    # is not a structural unknown. Only active owner vertices are fixed here.
    fixed |= ctx["owner_mask"] & ctx["smask"]
    snapshots: dict[int, Path] = {}

    def snapshot(cycle: int, snapshot_field: np.ndarray):
        path = save_stage(
            f"Erosion cycle {cycle}", snapshot_field,
            f"{args.region}_erosion_cycle_{cycle:02d}.png",
        )
        snapshots[cycle] = path

    eroded, erosion_report = erode_field(
        structural, generated_mask, ctx["owner_mask"], fixed, features,
        erosion_cfg, snapshot_callback=snapshot,
    )
    locked, lock_report = _final_seam_lock(ctx, eroded, cfg)
    final_path = save_stage(
        "Erosion final before seam lock", eroded,
        f"{args.region}_erosion_final_pre_lock.png",
    )
    locked_path = save_stage(
        "Post-erosion exact seam lock", locked,
        f"{args.region}_erosion_final.png",
    )

    seam_c0 = tmet.seam_c0(locked, ctx["own_view"], ctx["seam_v"])
    c1 = tmet.seam_c1_normals(
        locked, ctx["own_view"], ctx["seam_v"], ctx["nx"], ctx["ny"]
    )
    comparison_paths = [
        ("A owner/reference", owner_path),
        ("B Stage-3 base", stage3_path),
        ("C structural continuation", structural_path),
        ("D erosion cycle 0", snapshots.get(0, structural_path)),
        ("E erosion cycle 8", snapshots.get(8, snapshots.get(4, structural_path))),
        ("F erosion final cycle 24", snapshots.get(24, final_path)),
        ("G final seam lock", locked_path),
    ]
    sheet_path = out_dir / f"{args.region}_comparison_sheet.png"
    _comparison_sheet(comparison_paths, sheet_path)
    metrics = {
        "region": args.region,
        "features": features["feature_counts"],
        "structure": {"guides": guide_report, "solve": structural_report},
        "erosion": erosion_report,
        "final_lock": lock_report,
        "seam_c0_max_gu": seam_c0,
        "seam_c1_normals": c1,
        "review_bbox_vertices": list(review),
        "artifacts": {
            "comparison_sheet": str(sheet_path),
            "stage3_base": str(stage3_path),
            "structural_pre_erosion": str(structural_path),
            "erosion_final_pre_lock": str(final_path),
            "erosion_final": str(locked_path),
        },
        "timings": {"total_s": round(time.time() - t0, 1)},
    }
    with open(out_dir / f"{args.region}_structural_erosion_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=lambda value: int(value))
    np.savez_compressed(
        out_dir / f"{args.region}_structural_erosion_field.npz",
        field=locked, structural=structural, win=np.array(ctx["win"], dtype=np.int32),
        gx0=np.int32(ctx["gx0"]), gy0=np.int32(ctx["gy0"]),
    )
    print(json.dumps(metrics, indent=2, default=lambda value: int(value)))
    if seam_c0 > 1e-3:
        print(f"FAILURE: final seam C0 {seam_c0:.3f} GU")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
