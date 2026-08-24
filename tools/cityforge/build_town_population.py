"""Stage 07 checkpoint builder: frozen Stage 06 geometry -> populated JSON."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from procgen.townlayout.populate import populate_stamps
from procgen.townlayout.site_context import build_site_context, resolve_topdown_png
from procgen.townlayout.stamp_index import DEFAULT_LIBRARIES, build_stamp_index, load_stamp_libraries
from procgen.townlayout.place import write_placement_diagnostic
from procgen.townlayout.validate import validate_stage07_product

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--kit-brief',required=True); p.add_argument('--out-dir',required=True)
    p.add_argument('--survey'); p.add_argument('--fields'); p.add_argument('--census'); p.add_argument('--brief',required=True); args=p.parse_args(argv)
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        print('FAILURE: stamp-population out-dir not empty', file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        product=json.loads(Path(args.input).read_text()); kit=json.loads(Path(args.kit_brief).read_text())
        # Stage 06 intentionally carries only frozen road/fortification data;
        # ward ownership is read from the accepted Stage 04 checkpoint.
        ward_path = Path(args.input).parents[1] / 'stage04_compact_domain_wards' / 'domain_wards.json'
        if ward_path.is_file():
            product['wards'] = json.loads(ward_path.read_text()).get('wards', [])
        libs=load_stamp_libraries(DEFAULT_LIBRARIES); index=build_stamp_index(kit,libs); ctx=None
        if all((args.survey,args.fields,args.census,args.brief)):
            brief=json.loads(Path(args.brief).read_text()); ctx=build_site_context(survey_json=Path(args.survey),fields_npz=Path(args.fields),census_json=Path(args.census),town_brief=brief)
        result=populate_stamps(product,index,libs,ctx=ctx,master_seed=int(json.loads(Path(args.brief).read_text())['master_seed']),candidate_id=product.get('candidate_id','c00'))
        _, validation_issues = validate_stage07_product(result)
        (out/'population.json').write_text(json.dumps(result,allow_nan=False)+'\n')
        validation = {
            'status': 'failed' if validation_issues else 'ok',
            'error_count': len(validation_issues),
            'warning_count': 0,
            'issues': validation_issues,
        }
        (out/'validation.json').write_text(json.dumps(validation,allow_nan=False)+'\n')
        if ctx and args.survey and resolve_topdown_png(Path(args.survey)):
            write_placement_diagnostic(ctx,result,topdown_path=resolve_topdown_png(Path(args.survey)),survey=json.loads(Path(args.survey).read_text()),out_png=out/'population_diagnostic.png')
        if validation_issues:
            first = validation_issues[0]
            raise RuntimeError('stage07 validation: ' + first.get('path','') + ' ' + first['message'])
        minimum=json.loads(Path(args.brief).read_text())['target_buildings']['min']
        print(json.dumps(result['population_metrics'],sort_keys=True))
        if result['population_metrics']['required_coverage_pct'] < 80.0:
            raise RuntimeError('required arterial coverage %.2f%% < 80%%; counts front=%d paired-rear=%d wall=%d rejections=%s' % (
                result['population_metrics']['required_coverage_pct'], result['population_metrics']['front_count'],
                result['population_metrics']['paired_rear_count'], result['population_metrics']['wall_count'],
                result['population_metrics']['rejections']))
        if result['population_metrics']['collision_count'] or result['population_metrics']['water_overlap_count'] or result['population_metrics']['gate_blockage_count']:
            raise RuntimeError('accepted gate blockage/collision/water is nonzero: %s' % result['population_metrics'])
        if result['population_metrics']['population'] < minimum:
            raise RuntimeError(f"compact-domain capacity: population {result['population_metrics']['population']} < brief minimum {minimum}; front={result['population_metrics']['front_count']} paired-rear={result['population_metrics']['paired_rear_count']} wall={result['population_metrics']['wall_count']} rejections={result['population_metrics']['rejections']}")
        return 0
    except Exception as exc:
        print(f'FAILURE: stamp-population {exc}',file=sys.stderr); return 1
if __name__=='__main__': raise SystemExit(main())
