"""Build the Phase 21 R1 macro checkpoint.

Loads the complete Falkreath survey once, retunes the compact Voronoi domain to
the city80 trial density, and serializes the selected domain plus aligned-road
provenance for R2.  The output directory must be empty.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.aligned_roads import load_aligned_network  # noqa: E402
from procgen.cityplan import ring_area  # noqa: E402
from procgen.townlayout.anchors import place_anchors  # noqa: E402
from procgen.townlayout.approaches import build_rewrite_domain, build_site_approaches  # noqa: E402
from procgen.townlayout.checkpoint import sha256_file, write_checkpoint, read_checkpoint  # noqa: E402
from procgen.townlayout.diagnostics import render_macro_diagnostic  # noqa: E402
from procgen.townlayout.domain import grow_city_domain  # noqa: E402
from procgen.townlayout.patches import generate_organic_patches  # noqa: E402
from procgen.townlayout.site_context import build_site_context  # noqa: E402
from procgen.townlayout.validate import TownLayoutError  # noqa: E402
from procgen.townlayout.wards import assign_wards  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 21 R1 macro checkpoint")
    for name in ("survey", "fields", "census", "brief", "centerlines"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--candidate-id", default="c00")
    parser.add_argument("--out-dir", required=True)
    return parser


def _identity(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print("FAILURE: R1 output directory is not empty", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        survey = Path(args.survey)
        fields = Path(args.fields)
        census = Path(args.census)
        brief_path = Path(args.brief)
        brief_bytes = brief_path.read_bytes()
        brief = json.loads(brief_bytes.decode("utf-8"))
        ctx = build_site_context(survey_json=survey, fields_npz=fields,
                                 census_json=census, town_brief=brief)
        network = load_aligned_network(args.centerlines)
        preferred_area = float(ctx.estimated_urban_area_gu2["preferred"])
        requested_seeds = max(48, min(96, int(round(
            int(brief["target_buildings"]["preferred"])))))
        city_radius = math.sqrt(preferred_area / math.pi)
        nominal_patch_span = (preferred_area / requested_seeds) ** 0.5
        search_clearance = max(6144.0, 3.0 * nominal_patch_span)
        domain_meta = {}
        search_ring = build_rewrite_domain(ctx, radius_gu=city_radius,
                                           margin_gu=search_clearance,
                                           metadata=domain_meta)
        approaches_product = build_site_approaches(
            ctx, network, candidate_id=args.candidate_id, domain_ring=search_ring)
        patches = generate_organic_patches(
            ctx, search_ring, brief,
            master_seed=int(brief["master_seed"]), candidate_id=args.candidate_id,
            approaches=approaches_product["approaches"],
        )
        product = place_anchors(ctx, grow_city_domain(
            ctx, patches, brief, approaches=approaches_product["approaches"],
            rewrite_domain_meta=domain_meta),
                                brief, approaches=approaches_product["approaches"],
                                candidate_id=args.candidate_id)
        product = assign_wards(product, brief, candidate_id=args.candidate_id)
        core = sorted(p["patch_id"] for p in product["patches"]
                      if p.get("inside_city") and p.get("morphology_region") != "outskirts")
        # The wall must encircle one contiguous core (master plan).  With fine
        # cells an interspersed outskirts patch can split the non-outskirts
        # set; wall only the largest shared-boundary-connected component and
        # leave satellite core patches urban but unwalled (a suburb outside
        # the wall, like the reference maps' slums districts).
        adjacency: dict[str, set[str]] = {pid: set() for pid in core}
        for edge in product.get("boundary_edges", []):
            left, right = edge.get("patch_left"), edge.get("patch_right")
            if left in adjacency and right in adjacency:
                adjacency[left].add(right)
                adjacency[right].add(left)
        area_of = {p["patch_id"]: abs(ring_area(p["polygon"])) for p in product["patches"]}
        components = []
        unseen = set(core)
        while unseen:
            stack = [min(unseen)]
            members = []
            while stack:
                current = stack.pop()
                if current not in unseen:
                    continue
                unseen.discard(current)
                members.append(current)
                stack.extend(sorted(adjacency[current], reverse=True))
            components.append(sorted(members))
        components.sort(key=lambda members: (-sum(area_of[m] for m in members), members))
        walled = components[0] if brief.get("fortification", {}).get("mode") == "palisade" else []
        satellites = [m for members in components[1:] for m in members] if walled else []
        if satellites:
            product.setdefault("reports", []).append({
                "stage": "r1", "status": "ok",
                "message": f"core_components={len(components)} unwalled_satellites={sorted(satellites)}"})
        walled_set = set(walled)
        for patch in product["patches"]:
            patch["inside_wall"] = patch["patch_id"] in walled_set
        city = product["city_domain"]
        xs, ys = zip(*city)
        origin = ctx.origin_world_gu
        expanded = network.edges_in_rect(origin[0] + min(xs) - 12288,
                                         origin[1] + min(ys) - 12288,
                                         origin[0] + max(xs) + 12288,
                                         origin[1] + max(ys) + 12288)
        aligned_edges = []
        for edge in sorted(expanded, key=lambda item: item.id):
            aligned_edges.append({
                "id": edge.id, "from_node": edge.from_node, "to_node": edge.to_node,
                "component_id": edge.component_id,
                "plan_polyline": [list(point) for point in network.edge_site_chain(edge.id, origin)],
                "width": network.corridor_width(edge.id), "provenance": edge.provenance,
            })
        product.update({
            "schema_version": 1, "stage_id": "r1", "candidate_id": args.candidate_id,
            "master_seed": int(brief["master_seed"]),
            "identities": {"survey": _identity(survey), "fields": _identity(fields),
                           "census": _identity(census), "brief": _identity(brief_path),
            "aligned_product": {"path": str((network.product_dir / "tamriel_aligned_centerlines_v1.json").resolve()),
                                                "sha256": network.product_sha256}},
            "terrain": {"npz_path": str(fields), "schema": "survey_fields_v1",
                        "dimensions": [int(ctx.height_gu.shape[0]), int(ctx.height_gu.shape[1]),
                                       int(ctx.spacing_gu)], "sha256": sha256_file(fields)},
            "water_polygons": [list(map(list, poly.exterior.coords)) for poly in ctx.water_polygons()],
            "approaches": approaches_product["approaches"],
            "rewrite_domain": {"polygon": search_ring, "role": "search_envelope",
                               "search_clearance_gu": search_clearance},
            "core_patch_ids": core, "walled_patch_ids": walled,
            "aligned_roads": {"edges": aligned_edges},
            "stage_metrics": {"requested_developed_seeds": requested_seeds,
                              "patch_count": len(product["patches"]),
                              "selected_area_gu2": product.get("domain_metrics", {}).get("selected_area_gu2"),
                              "capacity": product.get("domain_metrics", {}).get("capacity"),
                              "aligned_edge_count": len(aligned_edges)},
        })
        write_checkpoint(product, out / "macro_layout.json")
        read_checkpoint(out / "macro_layout.json")
        render_macro_diagnostic(product, survey, out / "macro_layout_diagnostic.png")
    except (TownLayoutError, OSError, ValueError, KeyError) as exc:
        print(f"FAILURE: R1 {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
