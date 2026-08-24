"""Blender-only Phase 4 evidence exporter.

Imports every requested NIF in a fresh scene with the established 0.01 scale
correction and writes evaluated, non-degenerate triangles in GU.  It performs
no roof or dormer decisions; unresolved meshes are hard failures.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import bpy  # type: ignore
from mathutils import Vector  # type: ignore

WORKSPACE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(WORKSPACE / "tools"))
import blender_flat_render as render  # noqa: E402
import nif_thumbs  # noqa: E402
GU = 100.0

def args(): return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def main():
    a = args()
    if len(a) != 1: print("usage: blender -b --python blender_roof_dormer_evidence.py -- JOB.json", file=sys.stderr); return 2
    job = json.loads(Path(a[0]).read_text(encoding="utf-8")); roots, resolver = render.load_procgen_meshcheck()
    settings = {"scale_correction": 0.01, "normalize_to_position": False, "use_existing_materials": True, "ignore_collision_nodes": True, "ignore_animations": True, "reuse_meshes": False, "vertex_precision": 0.001}
    rows=[]; failures=[]
    for mesh in sorted(set(job["meshes"]), key=lambda x: str(x).casefold()):
        try:
            bpy.ops.wm.read_factory_settings(use_empty=True); nif_thumbs._configure_engine(nif_thumbs.resolved_config({}, layout="strip", resolution="1536x512"))
            resolved = resolver(mesh, "mesh", roots=roots)
            if resolved is None: raise RuntimeError("unresolved under configured data roots")
            doc={"scene_name":"ProcGen_RoofEvidence", "import":settings, "meshes":[{"id":Path(mesh).stem,"mesh":mesh.replace("/","\\"),"position":[0,0,0]}]}
            objs,_=render.import_meshes(render.resolve_meshes(doc, roots, resolver), render.setup_plugin(roots, settings), settings)
            bpy.context.view_layer.update(); dg=bpy.context.evaluated_depsgraph_get(); tris=[]; bmin=[float("inf")]*3; bmax=[float("-inf")]*3
            for obj in objs:
                if obj.type != "MESH": continue
                ev=obj.evaluated_get(dg); data=ev.to_mesh(); mat=ev.matrix_world
                try:
                    data.calc_loop_triangles(); vs=[]
                    for vert in data.vertices:
                        p=mat @ Vector(vert.co); q=(p.x*GU,p.y*GU,p.z*GU); vs.append(q)
                        for k in range(3): bmin[k]=min(bmin[k],q[k]); bmax[k]=max(bmax[k],q[k])
                    for lt in data.loop_triangles:
                        p=[vs[int(i)] for i in lt.vertices]; ab=(p[1][0]-p[0][0],p[1][1]-p[0][1],p[1][2]-p[0][2]); ac=(p[2][0]-p[0][0],p[2][1]-p[0][1],p[2][2]-p[0][2]); cr=(ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]); l=math.sqrt(sum(x*x for x in cr))
                        if l <= 0: continue
                        tris.append({"verts":[list(x) for x in p],"normal":[x/l for x in cr],"area":l/2,"centroid":[sum(x[k] for x in p)/3 for k in range(3)]})
                finally: ev.to_mesh_clear()
            if not tris: raise RuntimeError("no evaluated triangles")
            rows.append({"model_key":mesh.replace("/","\\"),"resolved_path":str(resolved),"bounds_local_gu":{"min":bmin,"max":bmax},"triangle_count":len(tris),"triangles":tris})
        except Exception as exc: failures.append(f"{mesh}: {exc}")
    out=Path(job["out"]); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"schema_version":1,"unit":"gu","models":rows,"failures":failures},sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return 0 if not failures else 1
if __name__ == "__main__": raise SystemExit(main())
