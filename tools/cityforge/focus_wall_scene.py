"""Create a terrain-backed focused scene for one measured wall arc.

Pipeline position
-----------------
Consumes a rendered city scene and its wall stamp document, then writes a
diagnostic scene containing only wall members within a configured arc window.
It changes visibility/framing only; wall geometry, terrain, materials, and
coordinate transforms remain those of the source scene.

Inputs and outputs are explicit CLI paths so the utility can be reused for any
closed composed wall rather than embedding Falkreath paths or dimensions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--wall", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arc", type=float, default=None,
                        help="target wall arc; defaults to the first requested-role arc")
    parser.add_argument("--role", default="slope",
                        help="structural role used when --arc is omitted (default: slope)")
    parser.add_argument("--width", type=float, default=2200.0)
    parser.add_argument(
        "--keep-all-members", action="store_true",
        help="retain every scene mesh while framing only the selected wall window",
    )
    parser.add_argument(
        "--front-facing", action="store_true",
        help="face the camera along the selected member's road-normal axis",
    )
    args = parser.parse_args(argv)

    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    wall = json.loads(args.wall.read_text(encoding="utf-8"))
    members = wall["members"]
    total = float(wall["provenance"]["path_length_gu"])
    role_arcs = [
        float(member["meta"]["arc"])
        for member in members
        if member.get("structural_role") == args.role
        and member.get("meta", {}).get("arc") is not None
    ]
    if args.arc is None and not role_arcs:
        raise ValueError(f"wall contains no members with structural role {args.role!r}")
    target = float(args.arc) if args.arc is not None else min(role_arcs)
    meshes = []
    for mesh in scene.get("meshes", []):
        mesh_id = str(mesh.get("id", ""))
        if not mesh.get("wall") and not mesh_id.startswith("wall_"):
            continue
        index = int(mesh_id.split("_")[1])
        arc = members[index].get("meta", {}).get("arc")
        if arc is None:
            continue
        distance = min(abs(float(arc) - target), total - abs(float(arc) - target))
        if distance <= float(args.width):
            meshes.append(mesh)
    if not meshes:
        raise ValueError("arc window selected no wall members")

    scene["scene_name"] = f"{scene.get('scene_name', 'wall')}_Focused"
    if not args.keep_all_members:
        scene["meshes"] = meshes
    scene.setdefault("camera", {})["subject"] = {
        "ids": [mesh["id"] for mesh in meshes],
        "include_terrain": False,
        "padding": 1.35,
    }
    if args.front_facing:
        target_member = min(
            (
                member for member in members
                if member.get("structural_role") == args.role
                if member.get("meta", {}).get("arc") is not None
            ),
            key=lambda member: min(
                abs(float(member["meta"]["arc"]) - target),
                total - abs(float(member["meta"]["arc"]) - target),
            ),
        )
        yaw = float((target_member.get("rotation") or [0.0, 0.0, 0.0])[2])
        scene.setdefault("camera", {})["view_direction"] = [
            math.sin(yaw), math.cos(yaw), 0.62
        ]
        scene["camera"]["view"] = "oblique"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"target_arc": target, "wall_members": len(meshes), "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
