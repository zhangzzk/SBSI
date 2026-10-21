"""Compare independent complete particle populations; never treat particles as chains."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_galsbi_continuous_prior import digest, save
from sbsi.particle_diagnostics import combine_populations, empirical_second_moment


TARGET_SOURCES = ('galsbi_particle_target.py', 'galsbi_target_coordinates.py',
                  'galsbi_summary.py', 'summary_gradients.py', 'models_v3_9_like.json')
REPORT_VERSION = 'independent-populations-v2-empirical-second-moment'


def implementation_fingerprint(protocol):
    """Compare recorded implementation hashes independent of snapshot roots.

    Basenames in this declared source list must be unique. These recorded
    files do not cover every transitive import or external runtime dependency.
    """
    sources = protocol.get('implementation_sha256')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('recorded implementation hashes required')
    result = {}
    for path, value in sources.items():
        name = Path(path).name
        if (name in result or not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('unique source basenames and valid SHA256 hashes required')
        result[name] = value
    if any(name not in result for name in TARGET_SOURCES):
        raise ValueError('recorded target source hashes are incomplete')
    return result


def geometry_fingerprint(protocol):
    geometry = protocol.get('physical_geometry')
    if geometry is None:
        return None
    if not isinstance(geometry, dict) or 'sha256' not in geometry or 'metadata' not in geometry:
        raise ValueError('frozen physical geometry requires hash and metadata')
    # A copied identical artifact is compatible even at another absolute path.
    return dict(sha256=geometry['sha256'], metadata=geometry['metadata'])


def validate_completed_result(protocol, result):
    """Reject incompatible receipts; do not convert unknown failures to zero Z.

    The current driver writes completed results only for surviving populations.
    Per-object extinct receipts need a separate explicit future schema.
    """
    objects, particles = protocol['observations'], protocol['particles']
    n = len(objects)
    if (not n or len(set(objects)) != n or type(particles) is not int or particles < 2
            or result.get('observations') != objects or result.get('particles') != particles):
        raise ValueError('completed result observations and particle count differ from protocol')
    logz = np.asarray(result['log_normalizer'], float)
    mean, variance = np.asarray(result['score_mean'], float), np.asarray(result['score_variance'], float)
    if (logz.shape != (n,) or not np.isfinite(logz).all() or mean.shape != (n, 2)
            or variance.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(variance).all() or np.any(variance < 0)):
        raise ValueError('completed results require finite normalizers and aligned finite score moments')
    final = result['final_stage']
    if final.get('stage') != protocol['stages'] or final.get('beta') != 1.:
        raise ValueError('completed result must reach the requested final stage at beta=1')
    if not np.array_equal(np.asarray(final.get('log_normalizer'), float), logz):
        raise ValueError('final stage and result normalizers disagree')
    regions = np.asarray([final[k] for k in ['target_within_two_magnitudes',
                                           'target_over_three_magnitudes_fainter']], float)
    if (regions.shape != (2, n) or not np.isfinite(regions).all()
            or np.any((regions < 0) | (regions > 1)) or np.any(regions.sum(0) > 1+1e-12)):
        raise ValueError('aligned probabilities for disjoint final scene regions required')
    times = np.asarray([result['timing_seconds'][k] for k in ['initialization', 'sampling', 'total']], float)
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError('finite nonnegative run timings required')


def serial(result):
    return {k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--comparison', type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(args.runs) < 2 or len(set(p.resolve() for p in args.runs)) != len(args.runs):
        raise ValueError('fresh output and distinct complete independent runs required')
    protocols, results, sources, implementations = [], [], {}, []
    for run in args.runs:
        p = json.loads((run/'protocol.json').read_text())
        r = json.loads((run/'result.json').read_text())
        if (r['status'] != 'completed' or r['protocol_sha256'] != digest(run/'protocol.json')
                or r['particles_sha256'] != digest(run/'particles.npz')):
            raise ValueError(f'incomplete or changed independent run: {run}')
        validate_completed_result(p, r)
        fingerprint = implementation_fingerprint(p)
        if protocols:
            keys = ['observations', 'fixed_shear', 'reference_protocol_sha256', 'observation_sha256',
                    'density_batch_size', 'particles', 'stages', 'moves', 'betas', 'scale', 'proposals',
                    'blocks', 'bridge', 'algorithm', 'selection']
            if any(p[k] != protocols[0][k] for k in keys):
                raise ValueError('each group must contain identically configured independent runs')
            if geometry_fingerprint(p) != geometry_fingerprint(protocols[0]):
                raise ValueError('each group must use the same frozen physical geometry')
            if fingerprint != implementations[0]:
                raise ValueError('each group must use identical recorded implementation versions')
        protocols.append(p); results.append(r); implementations.append(fingerprint)
        for name in ['protocol.json', 'result.json', 'particles.npz']:
            sources[str(run/name)] = digest(run/name)
    seeds = [p['seed'] for p in protocols]
    if len(set(seeds)) != len(seeds):
        raise ValueError('independent populations require distinct seeds')
    logz = np.asarray([r['log_normalizer'] for r in results])
    means = np.asarray([r['score_mean'] for r in results])
    variances = np.asarray([r['score_variance'] for r in results])
    regions = np.asarray([[r['final_stage'][k] for k in ['target_within_two_magnitudes',
        'target_over_three_magnitudes_fainter']] for r in results]).transpose(0, 2, 1)
    rows = []
    for i, oid in enumerate(protocols[0]['observations']):
        score = combine_populations(logz[:, i], means[:, i])
        region = combine_populations(logz[:, i], regions[:, i])
        second = score['weights']@empirical_second_moment(
            means[:, i], variances[:, i], protocols[0]['particles'])
        variance = np.maximum(0., second-score['mean']**2)
        mcvar = np.diag(score['covariance'])
        ess = np.divide(variance, mcvar, out=np.full(2, np.inf), where=mcvar > 0)
        precision_gate = bool(len(results) >= 8 and np.all(ess >= 400)
            and score['normalizer_relative_mcse'] < .1 and score['run_weight_ess'] >= .8*len(results))
        rows.append(dict(observation=oid, score=serial(score), region=serial(region),
            posterior_score_variance_estimate=variance.tolist(), score_effective_count_from_replicates=ess.tolist(),
            precision_gate=precision_gate))
    output = dict(runs=[str(p.resolve()) for p in args.runs], seeds=seeds, observations=rows,
        centre=protocols[0]['fixed_shear'], particles=protocols[0]['particles'],
        stages=protocols[0]['stages'], moves=protocols[0]['moves'], replicas=len(results),
        reference_protocol_sha256=protocols[0]['reference_protocol_sha256'],
        observation_sha256=protocols[0]['observation_sha256'], density_batch_size=protocols[0]['density_batch_size'],
        mean_seconds={key:float(np.mean([r['timing_seconds'][key] for r in results]))
                      for key in ['initialization', 'sampling', 'total']},
        total_compute_seconds=sum(r['timing_seconds']['total'] for r in results),
        all_precision_gates=all(r['precision_gate'] for r in rows),
        comparison_gate=False, heldout_gate=False, accepted_sampler=False,
        diagnostics_version=REPORT_VERSION,
        target_source_sha256={name:implementations[0][name] for name in TARGET_SOURCES},
        bridge=protocols[0]['bridge'], selection=protocols[0]['selection'],
        implementation_sha256=implementations[0], physical_geometry=geometry_fingerprint(protocols[0]),
        sources=sources, script_sha256=digest(__file__),
        limitations=['Delta-method errors from complete independent runs; finite-run uncertainty can miss rare events.',
                    'No nominal particle ESS/Rhat is used as an independent posterior sample count.',
                    'Recorded source hashes cover the declared files, not every transitive dependency.',
                    'This report requires completed surviving runs; extinction aggregation needs a future explicit receipt schema.',
                    'Passing precision alone does not establish accuracy or held-out efficiency.'])
    if args.comparison:
        other = json.loads(args.comparison.read_text())
        for key in ['centre', 'reference_protocol_sha256', 'observation_sha256', 'density_batch_size',
                    'target_source_sha256', 'bridge', 'selection', 'diagnostics_version']:
            if other.get(key) != output[key]:
                raise ValueError('comparison targets or numerical likelihood batches disagree')
        if set(seeds) & set(other['seeds']):
            raise ValueError('unpaired comparison requires disjoint population seeds')
        if [o['observation'] for o in other['observations']] != [o['observation'] for o in rows]:
            raise ValueError('comparison observations differ')
        comparisons = []
        for a, b in zip(rows, other['observations']):
            score_difference = np.asarray(a['score']['mean'])-b['score']['mean']
            score_se = np.hypot(a['score']['mcse'], b['score']['mcse'])
            sd = np.sqrt((np.asarray(a['posterior_score_variance_estimate'])+b['posterior_score_variance_estimate'])/2)
            region_difference = np.asarray(a['region']['mean'])-b['region']['mean']
            region_se = np.hypot(a['region']['mcse'], b['region']['mcse'])
            agreement = bool(np.all(np.abs(score_difference) <= 3.5*score_se)
                and np.all(np.abs(score_difference) <= .1*sd)
                and np.all(np.abs(region_difference) <= 3.5*region_se)
                and np.all(np.abs(region_difference) <= .05))
            comparisons.append(dict(observation=a['observation'], score_difference=score_difference.tolist(),
                score_combined_mcse=score_se.tolist(), region_difference=region_difference.tolist(),
                region_combined_mcse=region_se.tolist(), agreement=agreement))
        output['comparison'] = dict(path=str(args.comparison.resolve()), sha256=digest(args.comparison),
                                    observations=comparisons)
        output['comparison_gate'] = (output['all_precision_gates'] and other['all_precision_gates']
                                     and all(c['agreement'] for c in comparisons))
    save(args.output, output)
    print('PARTICLE_REPORT', args.output, 'precision', output['all_precision_gates'],
          'comparison', output['comparison_gate'], flush=True)


if __name__ == '__main__':
    main()
