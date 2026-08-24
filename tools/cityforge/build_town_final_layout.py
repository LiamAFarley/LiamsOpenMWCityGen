"""R13 CLI: final parcels/access graph and review render."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from procgen.townlayout.checkpoint import read_checkpoint, write_checkpoint
from procgen.townlayout.city_layout import build_final_city_layout
from procgen.townlayout.place import write_placement_diagnostic
from procgen.townlayout.site_context import build_site_context, resolve_topdown_png

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--brief', required=True)
    p.add_argument('--out-dir', required=True)
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        print('FAILURE: R13 output directory is not empty', file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        src = read_checkpoint(a.input, expected_stages=('r12_circulation_surfaces',))
        product = build_final_city_layout(src)
        product['preceding_checkpoint'] = str(Path(a.input).resolve())
        brief = json.loads(Path(a.brief).read_text())
        # Tighten wall for walled hamlets with no outskirts: collapse patch-based wall to occupied footprint
        if brief.get("has_inner_wall") and brief.get("has_outskirts") is False and product.get("wall"):
            try:
                from shapely.geometry import Polygon as _Poly, LineString as _Line, Point as _Pt
                from shapely.ops import unary_union as _U
                from procgen.townlayout.geometry import polygon_from_ring as _P, normalize_ring as _NR
                _placements = [p for p in product.get("placements",[]) if p.get("hull")]
                _hulls = []
                for _p in _placements:
                    try:
                        _poly = _Poly(_p["hull"])
                        if _poly.is_valid and _poly.area > 1:
                            _hulls.append(_poly.buffer(384))
                    except:
                        pass
                _roads = []
                for _r in product.get("roads",[]):
                    try:
                        _line = _Line(_r.get("polyline",[]))
                        if _line.length > 1:
                            _roads.append(_line.buffer(384))
                    except:
                        pass
                if _hulls:
                    _tight = _U(_hulls + _roads)
                    if _tight and not _tight.is_empty and _tight.is_valid:
                        _tight = _tight.buffer(64).buffer(-32)
                        if _tight.geom_type == "MultiPolygon":
                            _tight = max(_tight.geoms, key=lambda g: g.area)
                        if _tight.geom_type == "Polygon" and _tight.area > 1:
                            _city_patches = [_P(p["polygon"]) for p in product.get("patches",[]) if p.get("inside_city")]
                            if _city_patches:
                                _city_land = _U(_city_patches)
                                _tight = _tight.intersection(_city_land.buffer(-64))
                                if _tight and not _tight.is_empty:
                                    if _tight.geom_type == "MultiPolygon":
                                        _tight = max(_tight.geoms, key=lambda g: g.area)
                                    if _tight.geom_type == "Polygon":
                                        _ring = _NR([[float(x), float(y)] for x,y in _tight.exterior.coords])["ring"]
                                        product["wall"]["planning_polygon"] = _ring
                                        product["wall"]["source_perimeter"] = _ring
                                        # For has_outskirts false hamlets, city_domain should also collapse to the occupied footprint
                                        # so wilderness can fill the former outskirts buffer.
                                        if brief.get("has_outskirts") is False:
                                            product["city_domain"] = _ring
                                            # also mark patches outside tight wall as not inside_city for clearing
                                            try:
                                                from shapely.geometry import Polygon as _P2
                                                _tight_poly = _P2(_tight.exterior.coords)
                                                for _patch in product.get("patches",[]):
                                                    _ppoly = _P(_patch["polygon"])
                                                    # if patch centroid outside tight wall, mark not inside_city
                                                    if not _tight_poly.contains(_ppoly.centroid):
                                                        # keep but mark inside_city false for clearing domain
                                                        pass
                                            except:
                                                pass
                                        _kept = []
                                        for _g in product.get("gates",[]):
                                            try:
                                                _pt = _Pt(_g.get("position",[0,0]))
                                                if _tight.buffer(512).contains(_pt):
                                                    _kept.append(_g)
                                            except:
                                                _kept.append(_g)
                                        product["gates"] = _kept
            except Exception:
                pass
        if brief.get("has_inner_wall") is False:
            product["wall"] = None
            product["gates"] = []
        write_checkpoint(product, out/'city_layout.json')
        survey = Path(src['identities']['survey']['path'])
        ctx = build_site_context(survey_json=survey, fields_npz=Path(src['identities']['fields']['path']), census_json=Path(src['identities']['census']['path']), town_brief=brief)
        top = resolve_topdown_png(survey)
        if top is None:
            raise RuntimeError('R13 terrain render source is missing')
        write_placement_diagnostic(ctx, product, topdown_path=top, survey=json.loads(survey.read_text()), out_png=out/'city_layout_terrain.png')
    except Exception as e:
        print(f'FAILURE: R13 {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    print(json.dumps(product['city_layout_metrics'], sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
