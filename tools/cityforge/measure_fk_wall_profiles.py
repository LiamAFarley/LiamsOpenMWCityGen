#!/usr/bin/env python3
"""Build reviewed Falkreath wall profiles from Blender section evidence.

The wrapper launches ``blender_fk_wall_profile.py`` into a scratch raw JSON,
then uses host Shapely 2.x for endpoint snapping, polygonization, candidate
edge support, deterministic IDs, and height-band merging. It writes the
canonical profile (always ``needs_review`` in this stage) and one PNG
diagnostic per shell. Hulls are deliberately forbidden: unresolved main
contours return ``FAILURE: wall-profile topology ...`` and do not produce a
canonical record for that shell.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import polygonize_full, unary_union

WORKSPACE = Path(__file__).resolve().parents[2]
SNAP_GU = 2.0
GEOMETRY_TOL_GU = 8.0
VERTICAL_DEG = 15.0
MIN_EDGE_GU = 160.0


def _snap(value: float) -> float:
    return round(value / SNAP_GU) * SNAP_GU


def _line(row: dict) -> LineString:
    return LineString([(_snap(row["a"][0]), _snap(row["a"][1])),
                       (_snap(row["b"][0]), _snap(row["b"][1]))])


def _vertical(normal: list[float]) -> bool:
    return abs(float(normal[2])) <= math.sin(math.radians(VERTICAL_DEG))


def _edge_support(edge: LineString, rows: list[dict]) -> list[dict]:
    angle = math.atan2(edge.coords[-1][1] - edge.coords[0][1], edge.coords[-1][0] - edge.coords[0][0])
    supported = []
    for row in rows:
        if not _vertical(row["normal"]):
            continue
        source = LineString([(row["a"][0], row["a"][1]), (row["b"][0], row["b"][1])])
        if source.length < 1e-6 or edge.distance(source) > GEOMETRY_TOL_GU:
            continue
        source_angle = math.atan2(source.coords[-1][1] - source.coords[0][1], source.coords[-1][0] - source.coords[0][0])
        delta = abs((angle - source_angle + math.pi / 2) % math.pi - math.pi / 2)
        if delta <= math.radians(VERTICAL_DEG):
            supported.append(row)
    return supported


def _point_key(p) -> tuple[float, float]:
    return (round(float(p[0]), 3), round(float(p[1]), 3))


def _ring_edges(poly: Polygon, sample_rows: list[dict], prefix: str) -> list[dict]:
    edges = []
    # Polygonization retains every triangulation seam. Simplification is only
    # used for candidate-edge discovery; the unsimplified section rings remain
    # in ``components`` and therefore remain available for review.
    simplified = poly.simplify(GEOMETRY_TOL_GU, preserve_topology=True)
    for ring_kind, ring in [("exterior", simplified.exterior), *[("hole", r) for r in simplified.interiors]]:
        coords = list(ring.coords)[:-1]
        for a, b in zip(coords, coords[1:] + coords[:1]):
            edge = LineString([a, b])
            support = _edge_support(edge, sample_rows)
            if edge.length < MIN_EDGE_GU or not support:
                continue
            # Use the ring's actual winding. Polygonize_full does not promise
            # one winding across every source section; deriving a fixed
            # clockwise/CCW normal here was the cause of profile pilot doors
            # and windows being rotated 180 degrees.
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            right = (dy / length, -dx / length)
            left = (-dy / length, dx / length)
            if ring_kind == "exterior":
                outward = right if ring.is_ccw else left
            else:
                outward = left if ring.is_ccw else right
            edges.append({"a": list(_point_key(a)), "b": list(_point_key(b)),
                          "outward": [round(outward[0], 6), round(outward[1], 6)],
                          "length_gu": round(edge.length, 3), "ring": ring_kind,
                          "support_count": len(support)})
    return sorted(edges, key=lambda e: (e["ring"], e["a"], e["b"], e["outward"]))


def _section(shell: dict, section: dict) -> dict:
    raw = section["segments"]
    lines = [_line(row) for row in raw if math.hypot(row["a"][0] - row["b"][0], row["a"][1] - row["b"][1]) > 1e-3]
    if not lines:
        return {"z_gu": section["z_gu"], "components": [], "holes": [], "diagnostics": {"segments": 0, "dangles": 0, "cuts": 0, "invalid": 0}, "candidate_edges": []}
    merged = unary_union(lines)
    polygons, cuts, dangles, invalid = polygonize_full(merged)
    polys = sorted((geom for geom in polygons.geoms if geom.area > 1.0), key=lambda p: (p.bounds, -p.area))
    if not polys:
        return {"z_gu": section["z_gu"], "components": [], "holes": [], "diagnostics": {"segments": len(lines), "dangles": len(dangles.geoms), "cuts": len(cuts.geoms), "invalid": len(invalid.geoms)}, "candidate_edges": []}
    components = []
    edges = []
    for poly in polys:
        components.append({"exterior": [list(_point_key(p)) for p in list(poly.exterior.coords)[:-1]],
                           "holes": [[[list(_point_key(p)) for p in list(r.coords)[:-1]][i] for i in range(len(list(r.coords)[:-1]))] for r in poly.interiors]})
        edges.extend(_ring_edges(poly, raw, ""))
    return {"z_gu": section["z_gu"], "components": components,
            "diagnostics": {"segments": len(lines), "dangles": len(dangles.geoms), "cuts": len(cuts.geoms), "invalid": len(invalid.geoms)},
            "candidate_edges": edges}


def _section_shape(section: dict) -> MultiPolygon:
    return MultiPolygon([Polygon(c["exterior"], c["holes"]) for c in section["components"]])


def _compatible(a: dict, b: dict) -> bool:
    """Compare adjacent sections using the plan's topology tolerances."""
    if len(a["components"]) != len(b["components"]):
        return False
    shape_a, shape_b = _section_shape(a), _section_shape(b)
    scale = max(shape_a.area, shape_b.area, 1.0)
    if shape_a.symmetric_difference(shape_b).area > max(64.0, 0.002 * scale):
        return False
    if shape_a.boundary.hausdorff_distance(shape_b.boundary) > GEOMETRY_TOL_GU:
        return False
    ea, eb = a["candidate_edges"], b["candidate_edges"]
    if len(ea) != len(eb):
        return False
    unmatched = list(eb)
    for edge in ea:
        line = LineString([edge["a"], edge["b"]])
        matches = [other for other in unmatched
                   if line.distance(LineString([other["a"], other["b"]])) <= GEOMETRY_TOL_GU
                   and sum(x * y for x, y in zip(edge["outward"], other["outward"])) >= 0.96]
        if not matches:
            return False
        unmatched.remove(matches[0])
    return True


def _bands(measured: list[dict], z_min: float, z_max: float) -> list[dict]:
    # A closed roof/gable section is useful raw evidence but is not a wall
    # profile band unless it has supported near-vertical candidate edges.
    active = [m for m in measured if m["components"] and m["candidate_edges"]]
    bands = []
    for row in active:
        if not bands or not _compatible(bands[-1]["row"], row):
            bands.append({"z0": row["z_gu"], "z1": row["z_gu"], "sample_z_gu": [row["z_gu"]], "row": row})
        else:
            bands[-1]["z1"] = row["z_gu"]
            bands[-1]["sample_z_gu"].append(row["z_gu"])
    result = []
    for i, band in enumerate(bands, 1):
        row = band["row"]
        edges = []
        for j, edge in enumerate(row["candidate_edges"]):
            item = dict(edge)
            item["edge_id"] = f"band_{i:02d}_edge_{j:03d}"
            edges.append(item)
        result.append({"band_id": f"band_{i:02d}", "z0": round(float(band["z0"]), 3), "z1": round(float(band["z1"]), 3),
                       "sample_z_gu": band["sample_z_gu"], "components": row["components"], "candidate_edges": edges,
                       "topology_diagnostics": row["diagnostics"]})
    return result


def _diagnostic(shell: dict, measured: list[dict], bands: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95, wspace=0.18, hspace=0.2)
    tri = shell["triangles"]
    ax = axes[0, 0]
    source_top = []
    for t in tri:
        pts = [(p[0], p[1]) for p in t["points"]]
        source_top.extend(zip(pts, pts[1:] + pts[:1]))
    ax.add_collection(LineCollection(source_top, colors="0.78", linewidths=0.25))
    ax.autoscale()
    colors = plt.cm.tab20.colors
    contour_top = []
    for i, band in enumerate(bands):
        for c in band["components"]:
            xy = c["exterior"] + [c["exterior"][0]]
            contour_top.extend(zip(xy, xy[1:]))
            for e in band["candidate_edges"]:
                contour_top.append((e["a"], e["b"]))
    ax.add_collection(LineCollection(contour_top, colors="tab:blue", linewidths=0.7))
    candidate_top = [(e["a"], e["b"]) for band in bands for e in band["candidate_edges"]]
    ax.add_collection(LineCollection(candidate_top, colors="red", linewidths=1.5))
    ax.autoscale(); ax.set_title("Source mesh top view + band contours / candidate edges")
    ax.set_aspect("equal")
    ax = axes[0, 1]
    detail = bands[: min(12, len(bands))]
    detail_lines = []
    for i, band in enumerate(detail):
        for c in band["components"]:
            xy = c["exterior"] + [c["exterior"][0]]
            detail_lines.extend(zip(xy, xy[1:]))
            for hole in c["holes"]:
                hp = hole + [hole[0]]
                detail_lines.extend(zip(hp, hp[1:]))
            for e in band["candidate_edges"]:
                detail_lines.append((e["a"], e["b"]))
    ax.add_collection(LineCollection(detail_lines, colors="tab:blue", linewidths=1.2))
    ax.add_collection(LineCollection([(e["a"], e["b"]) for b in detail for e in b["candidate_edges"]], colors="red", linewidths=2))
    ax.autoscale(); ax.set_title("Enlarged recovered contours (first 12 bands; IDs in JSON)"); ax.set_aspect("equal")
    ax = axes[1, 0]
    source_side = []
    for t in tri:
        pts = [(p[2], p[1]) for p in t["points"]]
        source_side.extend(zip(pts, pts[1:] + pts[:1]))
    ax.add_collection(LineCollection(source_side, colors="0.75", linewidths=0.25)); ax.autoscale()
    for band in bands:
        ax.axvspan(band["z0"], band["z1"] or band["z0"] + 1, alpha=0.15)
    ax.set_title("Source mesh side view with accepted sample bands"); ax.set_xlabel("Z (GU)"); ax.set_ylabel("Y (GU)")
    ax = axes[1, 1]; ax.axis("off")
    lines = [f"{shell['model_key']}", f"triangles: {len(tri)}", f"samples: {len(measured)}", f"bands: {len(bands)}", "", "Band diagnostics:"]
    for band in bands:
        d = band["topology_diagnostics"]
        lines.append(f"{band['band_id']} z={band['z0']:.1f}..{band['z1']:.1f} edges={len(band['candidate_edges'])} dangles={d['dangles']} cuts={d['cuts']} invalid={d['invalid']}")
    lines.append("")
    lines.append("Edge IDs are deterministic in wall_profiles.json; red lines are candidate support.")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)


def _measure_shell(shell: dict) -> tuple[dict, list[dict], list[dict]]:
    measured = [_section(shell, row) for row in shell["sections"]]
    return shell, measured, _bands(measured, shell["z_min_gu"], shell["z_max_gu"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--raw", type=Path, default=WORKSPACE / "output" / "falkreath_wall_sections.raw.json")
    parser.add_argument("--out", type=Path, default=WORKSPACE / "configs" / "kits" / "falkreath" / "wall_profiles.json")
    parser.add_argument("--diagnostics", type=Path, default=WORKSPACE / "output" / "cityforge" / "falkreath_wall_profiles")
    parser.add_argument("--mesh", action="append", dest="meshes")
    parser.add_argument("--reuse-raw", action="store_true", help="rebuild profiles from an existing raw extraction")
    args = parser.parse_args()
    if not args.reuse_raw:
        blender = shutil.which(args.blender)
        if blender is None:
            print(f"FAILURE: wall-profile extraction blender not found ({args.blender!r})", file=sys.stderr); return 1
        script = WORKSPACE / "tools" / "cityforge" / "blender_fk_wall_profile.py"
        command = [blender, "-b", "--python", str(script), "--", str(args.raw), *(args.meshes or [])]
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        if completed.returncode:
            print("FAILURE: wall-profile extraction Blender returned non-zero", file=sys.stderr); return completed.returncode
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    profile = {"schema_version": 1, "unit": "gu", "frame": "engine_local", "extraction_parameters": {"sample_step_gu": 20.0, "plane_epsilon_gu": 0.01, "endpoint_snap_tolerance_gu": SNAP_GU, "geometry_tolerance_gu": GEOMETRY_TOL_GU, "vertical_normal_degrees": VERTICAL_DEG, "minimum_candidate_length_gu": MIN_EDGE_GU}, "shells": {}}
    # Section polygonization is independent per shell. Parallelizing this
    # host-only stage keeps the complete authoritative batch within the
    # preparation-time ceiling; Blender extraction itself remains one process.
    with ProcessPoolExecutor(max_workers=min(4, len(raw["shells"]))) as pool:
        processed = list(pool.map(_measure_shell, raw["shells"]))
    for shell, measured, bands in processed:
        if any(b["topology_diagnostics"]["dangles"] or b["topology_diagnostics"]["cuts"] or b["topology_diagnostics"]["invalid"] for b in bands):
            print(f"[wall-profile] diagnostic topology fragments retained for {shell['model_key']}", file=sys.stderr)
        if not bands:
            print(f"FAILURE: wall-profile topology {shell['model_key']} no closed section contour", file=sys.stderr); return 1
        profile["shells"][Path(shell["model_key"]).stem] = {"model_key": shell["model_key"], "z_reference": "primary_foundation_bottom", "foundation_bottom_z_gu": shell["z_min_gu"], "validation_state": "needs_review", "bands": bands}
        _diagnostic(shell, measured, bands, args.diagnostics / f"{Path(shell['model_key']).stem}.png")
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[wall-profile] wrote {args.out} and diagnostics in {args.diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
