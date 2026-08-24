"""R10 CLI: actual doors, free ground, and role candidates."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from procgen.townlayout.checkpoint import read_checkpoint, write_checkpoint
from procgen.townlayout.spatial_roles import derive_spatial_roles
from procgen.townlayout.place import write_placement_diagnostic
from procgen.townlayout.site_context import build_site_context, resolve_topdown_png

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--kit-brief',required=True); p.add_argument('--brief',required=True); p.add_argument('--out-dir',required=True); a=p.parse_args(argv)
    out=Path(a.out_dir)
    if out.exists() and any(out.iterdir()): print('FAILURE: R10 output directory is not empty',file=sys.stderr); return 1
    out.mkdir(parents=True,exist_ok=True)
    try:
        src=read_checkpoint(a.input,expected_stages=('r5_wall_front_rows',)); product=derive_spatial_roles(src); product['preceding_checkpoint']=str(Path(a.input).resolve()); write_checkpoint(product,out/'spatial_roles.json')
        survey=Path(src['identities']['survey']['path']); brief=json.loads(Path(a.brief).read_text()); ctx=build_site_context(survey_json=survey,fields_npz=Path(src['identities']['fields']['path']),census_json=Path(src['identities']['census']['path']),town_brief=brief); top=resolve_topdown_png(survey)
        if top is None: raise RuntimeError('R10 terrain render source is missing')
        write_placement_diagnostic(ctx,product,topdown_path=top,survey=json.loads(survey.read_text()),out_png=out/'spatial_roles_terrain.png')
    except Exception as e: print(f'FAILURE: R10 {e}',file=sys.stderr); return 1
    print(json.dumps(product['spatial_role_metrics'],sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
