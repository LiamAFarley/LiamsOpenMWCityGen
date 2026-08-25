"""Analyze tamriel-reworked seam borders and emit the Stage B review set.

Purpose
    Group corpus seam edges into contiguous clusters, measure owner-vs-base
    height behavior per edge (border delta, cross-border profiles, decay),
    deterministically classify every cluster, and render large hillshade
    crops centered on each cluster for human review BEFORE any heights are
    modified.

Inputs
    --config JSON (default configs/tamriel_reworked_v1.json); corpus npz +
    manifest from tools/terrain/build_terrain_corpus.py; thresholds under
    ``atlas.classification``.

Outputs
    paths.seam_atlas_json, paths.review_crops_dir/<cluster>.png plus
    <dir>/_index.md listing every crop with class and key stats.

Pipeline position
    Stage B of tamriel-reworked-heightmap; consumes Stage A corpus only.
    USER GATE: classification must be reviewed before solve stages run.

Invariants
    Read-only over corpus and plugins; deterministic ordering throughout;
    NaN owner heights (vanilla stub LANDs) surface as the void-owner class
    instead of being silently treated as matched terrain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen.terrainfield import (  # noqa: E402
    load_config,
    load_corpus,
    seam_edges,
    render_split_window,
    save_shade_png,
)


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _edge_axis(a: tuple[int, int], b: tuple[int, int]) -> str:
    return "h" if a[0] != b[0] else "v"


def _shared_line(a, b, gx0: int, gy0: int):
    """Return (fixed_coord, lo, hi, normal) in global vertex indices.
    normal points from a (tam) into b (owner)."""
    ax, ay = a
    bx, by = b
    if _edge_axis(a, b) == "h":          # east/west neighbours
        col = (max(ax, bx) - gx0) * 64
        row0 = (ay - gy0) * 64
        return ("col", col, row0, row0 + 64, (bx - ax, 0))
    row = (max(ay, by) - gy0) * 64       # north/south neighbours
    col0 = (ax - gx0) * 64
    return ("row", row, col0, col0 + 64, (0, by - ay))


def _line_sample(field: np.ndarray, axis: str, fixed: int, lo: int, hi: int):
    seg = field[fixed, lo:hi] if axis == "row" else field[lo:hi, fixed]
    return seg


def edge_features(tam_h, oth_h, a, b, gx0, gy0, probe: int):
    axis, fixed, lo, hi, normal = _shared_line(a, b, gx0, gy0)
    t = _line_sample(tam_h, axis, fixed, lo, hi)
    o = _line_sample(oth_h, axis, fixed, lo, hi)
    both = ~np.isnan(t) & ~np.isnan(o)
    if not both.any():
        return None
    delta = (o - t)[both]
    nan_frac = float(np.isnan(o).mean())

    prof = []
    for k in range(0, probe + 1):
        fk = fixed + normal[0] * 64 * k if axis == "col" else fixed + normal[1] * 64 * k
        ok = _line_sample(oth_h, axis, fk, lo, hi)
        tk = _line_sample(tam_h, axis, fk, lo, hi)
        d = (ok - tk)[~np.isnan(ok) & ~np.isnan(tk)]
        prof.append(float(np.mean(np.abs(d))) if d.size else None)

    s0 = None
    if len(prof) > 1 and prof[0] is not None and prof[1] is not None:
        step = abs(normal[0] or normal[1]) * 64 * 128
        s0 = (prof[1] - prof[0])
    dmed = float(np.median(np.abs(delta)))
    dmax = float(np.max(np.abs(delta)))
    rough = float(np.std(delta))
    return {
        "dmed_gu": round(dmed, 1), "dmax_gu": round(dmax, 1),
        "rough_gu": round(rough, 1), "owner_nan_frac": round(nan_frac, 3),
        "profile_mean_abs": [None if p is None else round(p, 1) for p in prof],
    }


def fit_decay(profile):
    vals = [(k, v) for k, v in enumerate(profile) if v is not None and v > 1.0]
    if len(vals) < 3:
        return None
    ks = np.array([k for k, _ in vals], dtype=float)
    ls = np.log([v for _, v in vals])
    slope, _ = np.polyfit(ks, ls, 1)
    if slope >= -1e-6:
        return None
    return round(min(-1.0 / slope, 64.0), 2)


def classify(feat, lam, cls_cfg):
    if feat["owner_nan_frac"] >= cls_cfg["void_owner_frac"]:
        return "void-owner"
    if feat["owner_nan_frac"] > 0:
        return "coast-partial"
    if feat["dmed_median_gu"] <= cls_cfg["matched_max_delta_gu"]:
        return "already-matched"
    if (abs(feat["slope_gu_per_cell"]) >= cls_cfg["cliff_slope_gu_per_cell"]
            and feat["dmed_median_gu"] >= cls_cfg["cliff_min_delta_gu"]):
        return "cliff-wall"
    if feat["dmed_median_gu"] >= cls_cfg["plateau_min_delta_gu"]:
        return "plateau-step"
    if lam is not None and lam >= cls_cfg["smooth_min_decay_cells"]:
        return "smooth-rise"
    return "sharp-mixed"


def cluster_edges(edges):
    """Connected components over tam-side cell adjacency."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    seen, out = set(), []
    for start in sorted(adj):
        if start in seen:
            continue
        comp, q = [], deque([start])
        seen.add(start)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj.get(cur, ()):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        out.append(comp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tamriel_reworked_v1.json"))
    args = ap.parse_args()
    cfg = load_config(_resolve(ROOT, args.config))

    t0 = time.time()
    arrays, meta = load_corpus(_resolve(ROOT, cfg["paths"]["corpus_npz"]))
    with open(_resolve(ROOT, cfg["paths"]["corpus_manifest"]), encoding="utf-8") as fh:
        manifest = json.load(fh)
    names = manifest["source_names"]
    base_code = names.index(manifest["base_source"]) + 1
    tam_h, oth_h = arrays["tam_h"], arrays["oth_h"]
    cell_owner = arrays["cell_owner"]

    edges = seam_edges(arrays["cell_owner"], base_code, meta["gy0"], meta["gx0"])
    print(f"seam edges: {len(edges)}")

    probe = int(cfg["atlas"]["probe_depth_cells"])
    cls_cfg = cfg["atlas"]["classification"]

    per_edge = []
    for a, b in sorted(edges):
        f = edge_features(tam_h, oth_h, a, b, meta["gx0"], meta["gy0"], probe)
        if f is None:
            continue
        f["a"] = list(a)
        f["b"] = list(b)
        per_edge.append(f)

    comps = cluster_edges([(tuple(e["a"]), tuple(e["b"])) for e in per_edge])
    comp_of = {}
    for ci, comp in enumerate(comps):
        for cell in comp:
            comp_of[cell] = ci

    clusters = []
    for e in per_edge:
        clusters.append(comp_of[tuple(e["a"])])

    render_cfg = cfg["render"]
    margin = int(cfg["atlas"]["crop_margin_cells"])
    ppv = int(render_cfg["px_per_vertex"])

    crops_dir = _resolve(ROOT, cfg["paths"]["review_crops_dir"])
    atlas_path = _resolve(ROOT, cfg["paths"]["seam_atlas_json"])

    cluster_records = []
    for ci in sorted(set(clusters)):
        comp = comps[ci]
        ce = [e for e in per_edge if comp_of[tuple(e["a"])] == ci]
        owners = sorted({
            names[int(arrays["cell_owner"][e["b"][1] - meta["gy0"],
                                           e["b"][0] - meta["gx0"]]) - 1]
            for e in ce
        })
        xs = [c[0] for c in comp]
        ys = [c[1] for c in comp]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        dmeds = [e["dmed_gu"] for e in ce]
        dmaxs = [e["dmax_gu"] for e in ce]
        lams = [fit_decay(e["profile_mean_abs"]) for e in ce]
        lam = max([l for l in lams if l is not None], default=None)
        slopes = []
        for e in ce:
            p = e["profile_mean_abs"]
            if len(p) > 1 and p[0] is not None and p[1] is not None:
                slopes.append(p[1] - p[0])
        slope_med = float(np.median(slopes)) if slopes else 0.0
        feat = {
            "edges": len(ce),
            "dmed_median_gu": round(float(np.median(dmeds)), 1),
            "dmax_max_gu": round(float(np.max(dmaxs)), 1),
            "slope_gu_per_cell": round(slope_med, 1),
            "decay_cells_max": lam,
        }
        cls = classify({**feat, "owner_nan_frac": max(e["owner_nan_frac"] for e in ce)},
                       lam, cls_cfg)

        x0, y0, x1, y1 = bbox
        r_lo = max(0, (y0 - meta["gy0"]) * 64 - margin * 64)
        r_hi = min(tam_h.shape[0], (y1 - meta["gy0"]) * 64 + 65 + margin * 64)
        c_lo = max(0, (x0 - meta["gx0"]) * 64 - margin * 64)
        c_hi = min(tam_h.shape[1], (x1 - meta["gx0"]) * 64 + 65 + margin * 64)

        label = f"{'+'.join(owners)}_{cls}"
        slug = f"cluster{ci:03d}_{label}_x{x0}_y{y0}"

        def make_overlay(draw, scale, comp=comp, ce=ce, r_lo=r_lo, c_lo=c_lo):
            for e in ce:
                a, b = tuple(e["a"]), tuple(e["b"])
                axis, fixed, lo, hi, _ = _shared_line(a, b, meta["gx0"], meta["gy0"])
                if axis == "row":
                    p1 = ((lo - c_lo) * scale, (fixed - r_lo) * scale)
                    p2 = ((hi - c_lo) * scale, (fixed - r_lo) * scale)
                else:
                    p1 = ((fixed - c_lo) * scale, (lo - r_lo) * scale)
                    p2 = ((fixed - c_lo) * scale, (hi - r_lo) * scale)
                draw.line([p1, p2], fill=(255, 60, 60), width=2)

        title = (f"{slug} edges={feat['edges']} dmed={feat['dmed_median_gu']}GU "
                 f"dmax={feat['dmax_max_gu']}GU")
        shade_win = render_split_window(tam_h, oth_h, cell_owner, base_code,
                                        r_lo, r_hi, c_lo, c_hi, render_cfg)
        out = save_shade_png(shade_win,
                             crops_dir / f"{slug}.png",
                             px_per_vertex=ppv, overlay=make_overlay, title=title)
        cluster_records.append({
            "cluster": ci, "owners": owners, "class": cls,
            "bbox_cells_xyxy": list(bbox), "tam_cells": len(comp),
            "features": feat, "crop_png": out.name,
        })
        print(f"  cluster {ci:03d} {cls:<14} owner={'+'.join(owners):<8} "
              f"cells={len(comp):>3} edges={feat['edges']:>3} "
              f"dmed={feat['dmed_median_gu']:>7.0f} dmax={feat['dmax_max_gu']:>7.0f}")

    atlas = {
        "config_echo": {"probe_depth_cells": probe, "classification": cls_cfg},
        "summary": {
            "edges": len(per_edge),
            "clusters": len(cluster_records),
            "by_class": {
                c: sum(1 for r in cluster_records if r["class"] == c)
                for c in sorted({r["class"] for r in cluster_records})
            },
        },
        "clusters": cluster_records,
    }
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    with open(atlas_path, "w", encoding="utf-8") as fh:
        json.dump(atlas, fh, indent=2, default=lambda o: int(o))

    idx_lines = [
        "# Seam atlas review crops", "",
        f"edges={atlas['summary']['edges']} clusters={atlas['summary']['clusters']}",
        f"classes={json.dumps(atlas['summary']['by_class'])}", "",
        "| crop | class | owners | cells | dmed GU | dmax GU |",
        "|---|---|---|---|---|---|",
    ]
    for r in cluster_records:
        idx_lines.append(
            f"| [{r['crop_png']}]({r['crop_png']}) | {r['class']} | "
            f"{'+'.join(r['owners'])} | {r['tam_cells']} | "
            f"{r['features']['dmed_median_gu']} | {r['features']['dmax_max_gu']} |")
    (crops_dir / "_index.md").write_text("\n".join(idx_lines) + "\n", encoding="utf-8")

    print(f"wrote {atlas_path}")
    print(f"wrote crops under {crops_dir}")
    print(f"elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
