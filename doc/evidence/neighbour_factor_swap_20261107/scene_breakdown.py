"""Summarize the fixed report and decompose its saved, small per-case bin sums."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches') / HERE.name
MODES = ['predicted_usability_generated_selection', 'true_usability_generated_selection', 'true_usability_true_selection']
TERMS = ['photo_at_old_shape', 'shape_at_old_photo', 'interaction', 'total']
METRICS = ['model_flow_response', 'model_emulator_response', 'delta_response']
MODENAMES = ['predicted / generated', 'actual / generated', 'actual / measured']


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cfg = json.loads((HERE / 'manifest.json').read_text())
    report = json.loads((HERE / 'report.json').read_text())
    assert digest(HERE / 'report.json') == digest(ROOT / 'report.json')
    cases = cfg['cases']
    n = len(cases)
    assert cases == report['cases'] and n == 4
    cuts = cfg['target_cuts']
    idx = [[c[0] for c in cfg['cuts']].index(name) for name in cuts]
    center = cuts.index('mag25_radius_gt0.60')
    weights = np.vstack((np.ones(n), 1 - np.eye(n), np.eye(n)))
    def se(v):
        loo = v[1:n+1]
        return np.sqrt((n-1)/n * np.square(loo-loo.mean(axis=0)).sum(axis=0))
    def fmt(v):
        return f'{v[0]:+.6f} ± {se(v):.6f}'
    contrasts, proxies, fractions = {}, {}, {}
    pooling = 0.
    matching = {}
    checks = 0
    source = digest(HERE / 'evaluate.py')
    assert len(report['audits']) == 8 and all(a['same_photo_selection_exact'] and a['source_sha256']==source for a in report['audits'])
    for axis in [1, 2]:
        arms = {}
        for name in cfg['arms']:
            ms, assoc = [], []
            for case in cases:
                path = ROOT / 'constant' / name / f'axis{axis}' / f'case{case}.npz'
                receipt = json.loads(path.with_suffix('.json').read_text())
                assert receipt['source_sha256'] == source and digest(path) == receipt['data_sha256']
                assert receipt['checkpoint_sha256'] == report['assembly']['arms'][name]['sha256']
                assert receipt['case'] == case and receipt['axis'] == axis and receipt['arm'] == name
                assert all(receipt[k] for k in ['true_selection_mass_exact','simulation_replay_passed','reference_score_replay_passed','pooled_emulator_roundoff_check_passed'])
                matching.setdefault((case,axis),receipt['measured'])
                assert matching[case,axis] == receipt['measured']
                pooling = max(pooling, receipt['pooled_emulator_first_block_max_abs'])
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['cuts'][idx],cuts)
                    ms.append(z['model'][:,:,idx]); assoc.append(z['association'])
            # For every axis/arm/mode/cut/shear leg, divide each size bin's
            # numerator by the total selected mass in that leg, across all bins.
            x = np.einsum('wc,cmlkbq->wmlkbq', weights, np.asarray(ms))
            mass = x[...,0].sum(axis=-1,keepdims=True)
            fractions[axis,name] = (x[...,0]/mass)[:,1,:,center,:].mean(axis=1)
            means = np.take(x,[axis,axis+2],axis=-1) / mass[...,None]
            arms[name] = (means[:,:,0]-means[:,:,1])/.04
            if axis == 1:
                v = weights @ np.asarray(assoc)
                proxies[name] = v[:,2]/v[:,1] - v[:,4]/v[:,3]
        for comp in cfg['factorial_comparisons']:
            compname = comp['name']
            old,new,photo,shape = [arms[comp[k]] for k in ['old','new','photo_only','shape_only']]
            for term, diff in zip(TERMS,[photo-old,shape-old,new-photo-shape+old,new-old]):
                bins = np.concatenate([diff,-diff.sum(axis=-1,keepdims=True)],axis=-1)
                contrasts[axis,compname,term] = bins
                total = bins.sum(axis=3)
                for mi,mode in enumerate(MODES):
                    for ci,cut in enumerate(cuts):
                        ref = report['axes'][str(axis)]['factorial'][compname][mode][cut]
                        for ki,metric in enumerate(METRICS):
                            d = ref[metric][term]
                            np.testing.assert_allclose([total[0,mi,ci,ki],se(total[:,mi,ci,ki])],
                                                       [d['value'][axis-1],d['jackknife_se'][axis-1]],rtol=0,atol=1e-11)
                            checks += 1
    def overall(axis,comp,term,mi=1):
        return contrasts[axis,comp,term][:,mi,center].sum(axis=1)
    comps = [c['name'] for c in cfg['factorial_comparisons']]
    for comp in comps:
        for term in TERMS:
            contrasts['22-11',comp,term] = contrasts[2,comp,term]-contrasts[1,comp,term]
    with (HERE/'cut_contrasts.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(['axis','contrast','term','usability / selection','cut','flow change','flow SE','emulator change','emulator SE','residual change','residual SE',*[f'case{c} residual change' for c in cases]])
        for (axis,comp,term), bins in contrasts.items():
            v = bins.sum(axis=3)
            for mi,mode in enumerate(MODENAMES):
                for ci,cut in enumerate(cuts):
                    a = v[:,mi,ci]
                    writer.writerow([axis,comp,term,mode,cut,*[f'{u:.9f}' for pair in zip(a[0],se(a)) for u in pair],*[f'{u:.9f}' for u in a[n+1:,-1]]])
    lines = ['# Scene and intrinsic-size comparison','',
        'Four reused development scenes; central cut is FLUX_RADIUS > 3 pixels and MAG_AUTO < 25. Errors below are delete-one-scene jackknife SEs, conditional on fixed models and common draws. Differences between axes use the same scene deletion on both axes. The pooled estimate uses leg sums rather than an unweighted mean of the four single-scene ratios.',
        '', 'The [nine-cut table](cut_contrasts.tsv) gives every axis, matched axis difference, flag mode, factor contrast, flow/emulator change, residual change and individual-scene residual change. A positive residual shift means the model response fell: ΔR = simulation − model. Components are changes in predictions.',
        '', '## Central residual and component changes','',
        '| Comparison | Axis | Photo-only residual | Shape-only residual | Interaction | Total residual | Total model flow change | Total model emulator change |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for comp in comps:
        for axis in [1,2,'22-11']:
            terms = [overall(axis,comp,t) for t in TERMS]
            cells = [fmt(t[:,-1]) for t in terms] + [fmt(terms[-1][:,i]) for i in [0,1]]
            lines.append('| '+' | '.join([comp,str(axis),*cells])+' |')
    lines.extend(['','## Flag choices','',
        '| Comparison | Usability / selection | Total ΔR11 shift | Total ΔR22 shift | Shape-only ΔR11 shift | Shape-only ΔR22 shift |',
        '|---|---|---:|---:|---:|---:|'])
    for comp in comps:
        for mi,mode in enumerate(MODENAMES):
            cells=[fmt(overall(a,comp,t,mi)[:,-1]) for t in ['total','shape_at_old_photo'] for a in [1,2]]
            lines.append('| '+' | '.join([comp,mode,*cells])+' |')
    lines.extend(['','Measured selection uses actual usability and measured per-leg cuts, but retains generated inputs to the shape and emulator predictions.','',
        '## Scene-by-scene shape contrast','',
        'Actual usability, generated selection, central cut. No independent case SE is assigned to a single scene.','',
        '| Scene | Regularization: shape-only ΔR11 | ΔR22 | Added moment: shape-only ΔR11 | ΔR22 | Added moment: total ΔR22 − ΔR11 shift |',
        '|---|---:|---:|---:|---:|---:|'])
    for j,case in enumerate(cases,n+1):
        vals=[overall(a,c,'shape_at_old_photo')[j,-1] for c in comps for a in [1,2]]+[overall('22-11',comps[1],'total')[j,-1]]
        lines.append('| '+str(case)+' | '+' | '.join(f'{v:+.6f}' for v in vals)+' |')
    lines.extend(['','## Intrinsic size contribution','',
        'Actual usability, generated selection, central cut. Each bin contributes to the selected population response using the full selected mass on its own shear leg. Rows add to the pooled change; they are neither within-bin responses nor evidence that size causes the shift. Bins use the existing circularized intrinsic radius (arcsec), distinct from the measured FLUX_RADIUS cut.','',
        '| Intrinsic radius | Regularization shape-only ΔR11 | ΔR22 | Added moment shape-only ΔR11 | ΔR22 | Added moment shape-only ΔR22 − ΔR11 |',
        '|---|---:|---:|---:|---:|---:|'])
    edges = cfg['intrinsic_size_edges_arcsec']
    names = [f'< {edges[0]}']+[f'[{l}, {h})' for l,h in zip(edges,edges[1:])]+[f'≥ {edges[-1]}']
    for b,label in enumerate(names):
        cells=[fmt(contrasts[a,c,'shape_at_old_photo'][:,1,center,b,-1]) for c in comps for a in [1,2]]
        cells.append(fmt(contrasts['22-11',comps[1],'shape_at_old_photo'][:,1,center,b,-1]))
        lines.append('| '+' | '.join([label,*cells])+' |')
    shapes = contrasts['22-11',comps[1],'shape_at_old_photo'][:,1,center,:,-1]
    share = 100*shapes[:,1:3].sum(axis=1)/shapes.sum(axis=1)
    selected = [100*fractions[a,'control'][0,1:3].sum() for a in [1,2]]
    lines.extend(['', f'For the added moment, the 0.15–0.40 arcsec bins contribute {fmt(shapes[:,1:3].sum(axis=1))} of the shape-only difference between axes ({share[0]:.1f}% ± {se(share):.1f} percentage points). Their shares of the selected population at fixed control photometry, averaging normalized plus/minus legs, are {selected[0]:.1f}% for axis1 and {selected[1]:.1f}% for axis2.'])
    unmatch = {key:sum(leg[key] for legs in matching.values() for leg in legs) for key in ['cross_without_shape','shape_without_cross','parent_absent_crossmatch']}
    lines.extend(['','## Checks and selected-neighbour proxy','',
        f'All {checks} bin-summed pooled contrasts and their jackknife SEs match the Slurm report to 1e-11. Maximum separate-versus-pooled emulator roundoff: {pooling:.3g}. All 56 files match the recorded hashes. Matching totals across the 16 scene/axis/legs (not unique galaxies): {unmatch["cross_without_shape"]} crossmatch rows without shape, {unmatch["shape_without_cross"]} shape rows without crossmatch, and {unmatch["parent_absent_crossmatch"]} parent rows absent from the measurement crossmatch. Unmatched measurements do not enter measured selection; full truth parents remain available for predicted usability.',
        '',f'On the fixed common-four-usable cohort (axis1 split/coherent catalogues), measured-minus-generated selected reference-neighbour coefficient changes from {fmt(proxies["control"])} to {fmt(proxies["weighted"])}; paired change {fmt(proxies["weighted"]-proxies["control"])}. This is a selected-population proxy; it contains no shear finite difference. Shared photometry gives exactly the same proxy regardless of conditional-shape factor.'])
    for arm,(photo,_) in cfg['recipes'].items():
        np.testing.assert_array_equal(proxies[arm],proxies[photo])
    (HERE/'scene_size_breakdown.md').write_text('\n'.join(lines)+'\n')
    print(lines[-1])
    print(f'{checks} contrasts checked; 56 file checksums; both axes, all cuts and cases.')


if __name__ == '__main__':
    main()
