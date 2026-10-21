#!/usr/bin/env python
"""Read-only, fixed-protocol localization of observation-0 phase-corrected g1.

No fitting, new likelihood evaluations, heldout inputs, cuts, or resampling.
All covariance, tail, genealogy and leave-one-population outputs are descriptive.
Only the complete independent-population score summary carries a ratio MCSE.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.particle_diagnostics import combine_populations
from sbsi.phase_control import FrozenPhaseControl, digest, load_phase_diagnostic


GROUPS = ('v5', 'v6v100', 'v7')
OBSERVATION = 0
QUANTILES = (.01, .05, .25, .5, .75, .95, .99)
TAIL_MASSES = (.01, .05)
# Fixed physical boundaries, declared before inspecting these arrays. None
# denotes an infinite endpoint in JSON; phase is always wrapped to [0, 2pi).
BINS = {
    'm_r': [-np.inf, 18, 20, 22, 24, 26, 28, 30, np.inf],
    'log_radius_arcsec': [-np.inf, *np.log([.01, .03, .1, .3, 1, 3, 10]), np.inf],
    'sersic_n': [-np.inf, 1, 2, 3, 4, 6, 8, np.inf],
    'distortion_a': [0, .2, .4, .6, .8, 1],
    'phase': np.linspace(0, 2*np.pi, 13),
    'crowding_log1p_050': [0, .1, .3, 1, 3, 10, 30, 100, np.inf],
    'crowding_log1p_100': [0, .1, .3, 1, 3, 10, 30, 100, np.inf],
    'crowding_log1p_200': [0, .1, .3, 1, 3, 10, 30, 100, np.inf],
    'response_b': [-np.inf, -4, -2, -1, -.5, 0, .5, 1, 2, 4, np.inf],
    'neighbour_count': [-.5, .5, 4.5, 9.5, 19.5, 49.5, 99.5, np.inf],
}


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('nonfinite report value')
    return value


def weights_checked(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-12,rtol=0)):
        raise ValueError('finite values and aligned normalized weights required')
    return values, weights/weights.sum()


def distribution(values, weights):
    values, weights = weights_checked(values, weights)
    order = np.argsort(values, kind='stable')
    cdf = np.cumsum(weights[order]); cdf[-1] = 1.
    mean = weights@values
    return dict(mean=float(mean), std=float(np.sqrt(weights@np.square(values-mean))),
        minimum=float(values.min()), maximum=float(values.max()),
        quantiles={str(q):float(values[order[np.searchsorted(cdf, q)]]) for q in QUANTILES})


def feature_diagnostic(values, score, weights):
    values, weights = weights_checked(values, weights)
    score, _ = weights_checked(score, weights)
    covariance = weights@((values-weights@values)*(score-weights@score))
    denominator = np.sqrt((weights@np.square(values-weights@values))
                          *(weights@np.square(score-weights@score)))
    return dict(distribution(values, weights), score_covariance=float(covariance),
        score_correlation=None if denominator == 0 else float(covariance/denominator))


def phase_diagnostic(phase, weights):
    phase, weights = weights_checked(np.mod(phase, 2*np.pi), weights)
    vector = weights@np.exp(1j*phase)
    return dict(circular_mean=float(np.mod(np.angle(vector), 2*np.pi)),
                mean_resultant_length=float(abs(vector)),
                convention='intrinsic spin-2 phase psi in [0,2pi); use sin/cos correlations')


def tail_diagnostic(score, weights, fraction, owners=None):
    """Exact upper |score| tail mass; fractional boundary particle, none dropped."""
    score, weights = weights_checked(score, weights)
    if not 0 < fraction < 1:
        raise ValueError('tail mass must be strictly between zero and one')
    order = np.argsort(-np.abs(score), kind='stable')
    before = np.r_[0., np.cumsum(weights[order])[:-1]]
    allocated = np.minimum(weights[order], np.maximum(0., fraction-before))
    tail = np.empty(len(score)); tail[order] = allocated
    absolute = weights@np.abs(score); second = weights@np.square(score)
    result = dict(requested_mass=fraction, actual_mass=float(tail.sum()),
        signed_mean_contribution=float(tail@score),
        positive_mean_contribution=float(tail@np.maximum(score, 0)),
        negative_mean_contribution=float(tail@np.minimum(score, 0)),
        complement_mean_contribution=float((weights-tail)@score),
        absolute_score_share=None if absolute == 0 else float(tail@np.abs(score)/absolute),
        second_moment_share=None if second == 0 else float(tail@np.square(score)/second),
        particles_touched=int(np.count_nonzero(tail)),
        cutoff_absolute_score=float(np.abs(score[order[np.flatnonzero(allocated)[-1]]])))
    if owners is not None:
        owners = np.asarray(owners)
        result['by_population'] = [dict(population=int(owner),
            mass=float(tail[owners == owner].sum()),
            signed_mean_contribution=float(tail[owners == owner]@score[owners == owner]))
            for owner in np.unique(owners)]
    return result


def ancestry_diagnostic(labels, score, weights):
    labels = np.asarray(labels)
    score, weights = weights_checked(score, weights)
    if labels.shape != score.shape:
        raise ValueError('ancestry must align with particles')
    unique, inverse = np.unique(labels, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    signed = np.bincount(inverse, weights=weights*score)
    absolute = np.bincount(inverse, weights=weights*np.abs(score))
    second = np.bincount(inverse, weights=weights*np.square(score))
    records = [dict(label=int(unique[i]), mass=float(mass[i]),
        signed_mean_contribution=float(signed[i]),
        absolute_score_share=None if absolute.sum() == 0 else float(absolute[i]/absolute.sum()),
        second_moment_share=None if second.sum() == 0 else float(second[i]/second.sum()))
        for i in np.argsort(-absolute, kind='stable')[:5]]
    return dict(distinct_initial_ancestors=len(unique), mass_ess=float(1/(mass@mass)),
        largest_particle_mass=float(mass.max()),
        largest_absolute_score_share=None if absolute.sum() == 0 else float(absolute.max()/absolute.sum()),
        top_five_by_absolute_score=records,
        limitation='Genealogy only; post-mutation descendants are not identical or independent draws.')


def binned_contributions(values, score, weights, edges):
    values, weights = weights_checked(values, weights)
    score, _ = weights_checked(score, weights)
    edges = np.asarray(edges, float)
    index = np.searchsorted(edges, values, side='right')-1
    index[values == edges[-1]] = len(edges)-2
    if np.any((index < 0) | (index >= len(edges)-1)):
        raise ValueError('fixed bins must cover every particle')
    mass = np.bincount(index, weights=weights, minlength=len(edges)-1)
    contribution = np.bincount(index, weights=weights*score, minlength=len(edges)-1)
    return dict(edges=[None if np.isinf(x) else float(x) for x in edges],
        convention='left closed/right open; final endpoint included; None endpoint denotes infinity',
        mass=mass.tolist(), score_mean_contribution=contribution.tolist(),
        conditional_score_mean=[float(s/p) if p else None for p,s in zip(mass, contribution)])


def binned_difference(left, right):
    """Descriptive midpoint decomposition, with unmatched bins explicit."""
    if left['edges'] != right['edges']:
        raise ValueError('same predeclared bins required')
    p, q = np.asarray(left['mass']), np.asarray(right['mass'])
    a, b = np.asarray(left['score_mean_contribution']), np.asarray(right['score_mean_contribution'])
    common = (p > 0) & (q > 0)
    composition = ((p[common]-q[common])*(a[common]/p[common]+b[common]/q[common])/2).sum()
    within = ((p[common]+q[common])*(a[common]/p[common]-b[common]/q[common])/2).sum()
    return dict(edges=left['edges'], mass_difference=(p-q).tolist(),
        score_contribution_difference=(a-b).tolist(),
        composition_component=float(composition), within_bin_component=float(within),
        unmatched_bin_component=float((a[~common]-b[~common]).sum()),
        total_difference=float(a.sum()-b.sum()),
        limitation='Fixed-bin descriptive accounting, not causal explanation or a fitted model.')


def population_sensitivity(logz, score_means):
    logz, score_means = np.asarray(logz), np.asarray(score_means)
    complete = combine_populations(logz, score_means[:, None])
    leave_one = []
    for i in range(len(logz)):
        keep = np.arange(len(logz)) != i
        weight = np.exp(logz[keep]-np.max(logz[keep])); weight /= weight.sum()
        mean = weight@score_means[keep]
        leave_one.append(dict(omitted_population=i, mean=float(mean),
            shift_from_complete=float(mean-complete['mean'][0])))
    return dict(complete=complete, leave_one_population_out=leave_one,
        leave_one_note='Descriptive sensitivity; these overlapping estimates are not independent uncertainty.')


def extract_features(coordinates, summaries, scene, flat_start, particles):
    """Map stored native/CSR data; no priors, models, or truth values consulted."""
    x, summary = np.asarray(coordinates), np.asarray(summaries)
    if x.shape != (particles, 10) or summary.shape != (particles, 9):
        raise ValueError('aligned saved coordinates and summaries required')
    rows = slice(flat_start, flat_start+particles)
    phase = np.mod(x[:, 9], 2*np.pi)
    native_phase = np.mod(scene['target_psi'][rows], 2*np.pi)
    if not np.allclose(np.exp(1j*phase), np.exp(1j*native_phase), atol=1e-12, rtol=0):
        raise ValueError('native phase does not match saved coordinates')
    for a,b in [(x[:, 1], scene['target_m_r'][rows]), (x[:, 1],summary[:, 3]),
                (x[:, 6], np.log(summary[:, 4])), (scene['target_n'][rows],summary[:, 2])]:
        if not np.allclose(a,b,atol=1e-10,rtol=0):
            raise ValueError('native scene, coordinates and summaries disagree')
    indptr = np.asarray(scene['indptr'])
    if (indptr.ndim != 1 or indptr[0] != 0 or np.any(np.diff(indptr) < 0)
            or flat_start+particles >= len(indptr)):
        raise ValueError('valid scene CSR required')
    lo,hi = indptr[flat_start],indptr[flat_start+particles]
    counts = np.diff(indptr[flat_start:flat_start+particles+1])
    owner = np.repeat(np.arange(particles),counts)
    offset = scene['offset_arcsec'][lo:hi]
    distance = np.linalg.norm(offset,axis=1)
    ratio = np.exp(-.4*np.log(10)*(scene['neighbour_m_r'][lo:hi]-x[owner,1]))
    ratio_sum = np.bincount(owner,weights=ratio,minlength=particles)
    ratio_max = np.zeros(particles); np.maximum.at(ratio_max,owner,ratio)
    features = dict(m_r=x[:,1],log_radius_arcsec=x[:,6],radius_arcsec=np.exp(x[:,6]),
        log_z=x[:,0],native_radius_kpc=scene['target_R_kpc'][rows],
        sersic_n=scene['target_n'][rows],distortion_a=scene['target_a'][rows],
        phase_sin=np.sin(phase),phase_cos=np.cos(phase),
        intrinsic_e1=summary[:,0],intrinsic_e2=summary[:,1],
        crowding_log1p_050=summary[:,5],crowding_log1p_100=summary[:,6],
        crowding_log1p_200=summary[:,7],response_b=summary[:,8],neighbour_count=counts,
        log1p_neighbour_target_flux_sum=np.log1p(ratio_sum),
        log1p_neighbour_target_flux_max=np.log1p(ratio_max))
    for radius in (.5,1.,2.):
        features[f'neighbours_within_{radius:g}_arcsec'] = np.bincount(owner,
            weights=distance<radius,minlength=particles)
    if not all(np.isfinite(v).all() for v in features.values()):
        raise ValueError('nonfinite physical/scene features')
    return features,phase


def load_population(root, phase_root, group, index, frozen):
    run = root/f'dev_{group}_{index}'; path = phase_root/f'dev_{group}_{index}.json'
    diagnostic, arrays = load_phase_diagnostic(path)
    if (diagnostic.get('derivative_protocol') != 'phase-fine-difference-v2'
            or diagnostic.get('stride') != 1 or diagnostic['run'] != str(run.resolve())
            or diagnostic.get('coefficients',{}).get('sha256')
                != digest(phase_root/'frozen_v1_coefficients.json')):
        raise ValueError('complete v2 diagnostic for this exact population required')
    protocol = json.loads((run/'protocol.json').read_text())
    result = json.loads((run/'result.json').read_text())
    if (result['status'] != 'completed' or any(result.get('extinct',[]))
            or result['scenes_sha256'] != digest(run/'scenes.npz')
            or result['protocol_sha256'] != digest(run/'protocol.json')
            or result['particles_sha256'] != digest(run/'particles.npz')):
        raise ValueError('unchanged complete original population and scene required')
    frozen.validate_production(protocol['seed'], protocol['observations'], diagnostic['target_identity'])
    obj = protocol['observations'].index(OBSERVATION); n = protocol['particles']
    with np.load(run/'particles.npz',allow_pickle=False) as data:
        if (not np.array_equal(data['scores'],arrays['raw_scores'])
                or data['coordinates'].shape != (len(protocol['observations']),n,10)):
            raise ValueError('unaltered complete aligned score and coordinate arrays required')
        coordinates,summaries,labels = [data[k][obj].copy() for k in ('coordinates','summaries','ancestor_labels')]
    with np.load(run/'scenes.npz',allow_pickle=False) as data:
        features,phase = extract_features(coordinates,summaries,data,obj*n,n)
    score = arrays['raw_scores'][obj,:,0]-frozen.coefficients[obj,0]*arrays['phase_control'][obj]
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:
        if not np.array_equal(score,data['corrected_scores'][obj,:,0]):
            raise ValueError('fixed frozen coefficient correction must replay exactly')
    if diagnostic['log_normalizer'][obj] != result['log_normalizer'][obj]:
        raise ValueError('phase and sampling normalizers differ')
    return dict(group=group,index=index,seed=protocol['seed'],particles=n,
        log_normalizer=result['log_normalizer'][obj],score=score,
        raw_score=arrays['raw_scores'][obj,:,0],control=arrays['phase_control'][obj],
        features=features,phase=phase,labels=labels,target_identity=diagnostic['target_identity'],
        source_sha256={str(p.resolve()):digest(p) for p in
            [run/'protocol.json',run/'result.json',run/'particles.npz',run/'scenes.npz',path,path.with_suffix('.npz')]})


def summarize_group(populations):
    logz = np.array([p['log_normalizer'] for p in populations])
    means = np.array([p['score'].mean() for p in populations])
    sensitivity = population_sensitivity(logz,means)
    run_weights = sensitivity['complete']['weights']
    rows=[]
    for p,w in zip(populations,run_weights):
        weights=np.full(p['particles'],1/p['particles'])
        rows.append(dict(index=p['index'],seed=p['seed'],particles=p['particles'],
            log_normalizer=p['log_normalizer'],normalizer_weight=float(w),
            corrected_g1_mean=float(p['score'].mean()),raw_g1_mean=float(p['raw_score'].mean()),
            control_mean=float(p['control'].mean()),
            weighted_corrected_g1_contribution=float(w*p['score'].mean()),
            score_distribution=distribution(p['score'],weights),
            feature_diagnostics={k:feature_diagnostic(v,p['score'],weights) for k,v in p['features'].items()},
            phase=phase_diagnostic(p['phase'],weights),
            absolute_score_tails={str(q):tail_diagnostic(p['score'],weights,q) for q in TAIL_MASSES},
            ancestry=ancestry_diagnostic(p['labels'],p['score'],weights)))
    score=np.concatenate([p['score'] for p in populations])
    weights=np.concatenate([np.full(p['particles'],w/p['particles']) for p,w in zip(populations,run_weights)])
    owners=np.concatenate([np.full(p['particles'],p['index']) for p in populations])
    features={k:np.concatenate([p['features'][k] for p in populations]) for k in populations[0]['features']}
    phase=np.concatenate([p['phase'] for p in populations])
    return dict(**sensitivity,populations=rows,
        pooled_feature_diagnostics={k:feature_diagnostic(v,score,weights) for k,v in features.items()},
        phase=phase_diagnostic(phase,weights),
        score_distribution=distribution(score,weights),
        absolute_score_tails={str(q):tail_diagnostic(score,weights,q,owners) for q in TAIL_MASSES},
        fixed_bins={k:binned_contributions(phase if k=='phase' else features[k],score,weights,e)
                    for k,e in BINS.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('fresh localization output required')
    coefficient_path=args.phase_root/'frozen_v1_coefficients.json'
    frozen=FrozenPhaseControl.load(coefficient_path)
    all_populations={g:[load_population(args.root,args.phase_root,g,i,frozen) for i in range(8)] for g in GROUPS}
    flat=[p for v in all_populations.values() for p in v]
    if (len(set(p['seed'] for p in flat))!=len(flat)
            or any(p['target_identity']!=flat[0]['target_identity'] for p in flat[1:])):
        raise ValueError('distinct production seeds and identical diagnostic targets required')
    groups={g:summarize_group(p) for g,p in all_populations.items()}
    comparisons={}
    for left,right in [('v7','v6v100'),('v7','v5'),('v5','v6v100')]:
        a,b=groups[left],groups[right]
        difference=a['complete']['mean'][0]-b['complete']['mean'][0]
        se=np.hypot(a['complete']['mcse'][0],b['complete']['mcse'][0])
        comparisons[left+'-minus-'+right]=dict(corrected_g1_mean_difference=float(difference),
            combined_independent_population_mcse=float(se),standardized_difference=float(difference/se),
            fixed_bin_accounting={k:binned_difference(a['fixed_bins'][k],b['fixed_bins'][k]) for k in BINS},
            absolute_score_tail_accounting={str(q):dict(
                tail_signed_contribution_difference=a['absolute_score_tails'][str(q)]['signed_mean_contribution']-b['absolute_score_tails'][str(q)]['signed_mean_contribution'],
                complement_signed_contribution_difference=a['absolute_score_tails'][str(q)]['complement_mean_contribution']-b['absolute_score_tails'][str(q)]['complement_mean_contribution']) for q in TAIL_MASSES})
    output=dict(schema='sbsi-phase-score-localization-v1',observation=OBSERVATION,
        groups=groups,comparisons=comparisons,coefficient_sha256=digest(coefficient_path),
        source_sha256={k:v for p in flat for k,v in p['source_sha256'].items()},
        script_sha256=digest(__file__),target_identity=flat[0]['target_identity'],
        protocol=dict(quantiles=QUANTILES,absolute_score_tail_masses=TAIL_MASSES,
            tail_boundary='fractional boundary mass after stable descending absolute-score sort; no particle removed',
            weighting='uniform within final populations; original normalized Zhat across complete runs',
            covariance='exact empirical weighted central product, ddof=0; no independence interpretation',
            phase='circular mean and sin/cos correlations, not linear angle Pearson correlation',
            leave_one='renormalize remaining original Zhat weights; descriptive sensitivity only'),
        limitations=['No fit, new sampling, target changes, new GPU use, or heldout inputs.',
            'Feature/bin/tail accounting is descriptive and cannot establish a causal mechanism.',
            'Initial ancestors describe genealogy, not effective independent samples.',
            'Original F32-sampled numerical measure; phase identity remains explicitly qualified.',
            'Eight-population ratio MCSE can miss rare modes; no existing gate is relaxed.'])
    with args.output.open('x') as handle:
        json.dump(serial(output),handle,indent=2,allow_nan=False);handle.write('\n')
    print('PHASE_SCORE_LOCALIZATION',args.output,flush=True)


if __name__=='__main__':
    main()
