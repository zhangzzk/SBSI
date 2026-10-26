"""Compare independent complete particle populations; never treat particles as chains."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_galsbi_continuous_prior import digest, save
from sbsi.particle_diagnostics import combine_populations, empirical_second_moment
from sbsi.particle_populations import EXTINCTION_SCHEMA


TARGET_SOURCES = ('galsbi_particle_target.py', 'galsbi_target_coordinates.py',
                  'galsbi_summary.py', 'summary_gradients.py', 'models_v3_9_like.json')
REPORT_VERSION = 'independent-populations-v4-normalizer-comparison'


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
    if protocol.get('extinction_schema') == EXTINCTION_SCHEMA and 'particle_populations.py' not in result:
        raise ValueError('recorded extinction bookkeeping source hash required')
    return result


def geometry_fingerprint(protocol):
    geometry = protocol.get('physical_geometry')
    if geometry is None:
        return None
    if not isinstance(geometry, dict) or 'sha256' not in geometry or 'metadata' not in geometry:
        raise ValueError('frozen physical geometry requires hash and metadata')
    # A copied identical artifact is compatible even at another absolute path.
    return dict(sha256=geometry['sha256'], metadata=geometry['metadata'])


def proposal_fingerprint(protocol):
    proposal = protocol.get('initial_proposal')
    if proposal is None:
        return None
    if not isinstance(proposal, dict) or 'sha256' not in proposal or 'metadata' not in proposal:
        raise ValueError('frozen initial proposal requires hash and metadata')
    return dict(sha256=proposal['sha256'], metadata=proposal['metadata'])


def pilot_seeds(record):
    seeds = set()
    for key in ('physical_geometry', 'initial_proposal'):
        if record.get(key) is not None:
            values = record[key]['metadata'].get('pilot_seeds', [])
            if not isinstance(values, list) or any(type(v) is not int for v in values):
                raise ValueError('integer frozen-artifact pilot seeds required')
            seeds.update(values)
    return seeds


def _masked_array(value, shape, extinct, name):
    raw = np.asarray(value, object)
    if raw.shape != shape:
        raise ValueError(f'aligned {name} values required')
    null = np.fromiter((v is None for v in raw.flat), bool).reshape(shape)
    expected = np.broadcast_to(np.asarray(extinct).reshape((shape[0],)+(1,)*(len(shape)-1)), shape)
    if not np.array_equal(null, expected):
        raise ValueError(f'{name} must be null exactly for explicitly extinct observations')
    values = np.asarray(value, float)
    if not np.isfinite(values[~np.asarray(extinct)]).all():
        raise ValueError(f'finite surviving {name} required')
    return values


def validate_completed_result(protocol, result):
    """Decode declared zero mass only; arbitrary failures are never extinctions."""
    objects, particles = protocol['observations'], protocol['particles']
    n = len(objects)
    if (not n or len(set(objects)) != n or type(particles) is not int or particles < 2
            or result.get('observations') != objects or result.get('particles') != particles):
        raise ValueError('completed result observations and particle count differ from protocol')
    explicit = ('extinction_schema' in protocol or 'extinction_schema' in result or 'extinct' in result)
    extinct = np.zeros(n, bool)
    final = result['final_stage']
    if explicit:
        if (protocol.get('extinction_schema') != EXTINCTION_SCHEMA
                or result.get('extinction_schema') != EXTINCTION_SCHEMA
                or final.get('extinction_schema') != EXTINCTION_SCHEMA):
            raise ValueError('explicit supported extinction schema required in protocol and receipt')
        mask = result.get('extinct')
        if not isinstance(mask, list) or len(mask) != n or any(type(v) is not bool for v in mask):
            raise ValueError('explicit boolean observation extinction mask required')
        extinct = np.asarray(mask)
        stages = result.get('extinction_stage')
        if not isinstance(stages, list) or len(stages) != n:
            raise ValueError('aligned extinction stages required')
        for dead, stage in zip(extinct, stages):
            if ((dead and (type(stage) is not int or not 1 <= stage <= protocol['stages']))
                    or (not dead and stage is not None)):
                raise ValueError('extinction stages must identify first zero incremental mass')
        if final.get('extinct') != mask or final.get('extinction_stage') != stages:
            raise ValueError('final stage and result extinction masks/stages disagree')
    status = 'completed_with_extinction' if extinct.any() else 'completed'
    if result.get('status') != status:
        raise ValueError('unknown failure or inconsistent explicit extinction status')
    logz = _masked_array(result['log_normalizer'], (n,), extinct, 'normalizers')
    mean = _masked_array(result['score_mean'], (n, 2), extinct, 'score means')
    variance = _masked_array(result['score_variance'], (n, 2), extinct, 'score variances')
    if np.any(variance[~extinct] < 0):
        raise ValueError('nonnegative surviving score variances required')
    logz[extinct] = -np.inf
    if final.get('stage') != protocol['stages'] or final.get('beta') != 1.:
        raise ValueError('completed result must reach the requested final stage at beta=1')
    final_logz = _masked_array(final.get('log_normalizer'), (n,), extinct, 'final normalizers')
    final_logz[extinct] = -np.inf
    if not np.array_equal(final_logz, logz):
        raise ValueError('final stage and result normalizers disagree')
    regions = np.column_stack([_masked_array(final[k], (n,), extinct, k) for k in
        ['target_within_two_magnitudes', 'target_over_three_magnitudes_fainter']])
    if (np.any((regions[~extinct] < 0) | (regions[~extinct] > 1))
            or np.any(regions[~extinct].sum(1) > 1+1e-12)):
        raise ValueError('aligned probabilities for disjoint final scene regions required')
    times = np.asarray([result['timing_seconds'][k] for k in ['initialization', 'sampling', 'total']], float)
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError('finite nonnegative run timings required')
    return logz, mean, variance, regions, extinct


def undefined_population_summary(count, dimensions):
    return dict(mean=[None]*dimensions, covariance=[[None]*dimensions for _ in range(dimensions)],
        mcse=[None]*dimensions, weights=[0.]*count, run_weight_ess=0., log_mean_normalizer=None,
        normalizer_relative_mcse=None, independent_populations=count, extinct_populations=count,
        mean_defined=False)


def serial(result):
    return {k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in result.items()}


def compare_normalizers(first, second):
    """Compare independent mean masses without exponentiating absolute log Z.

    Each relative MCSE is the sample standard error of the mean Z divided by
    that mean, not an error of log Z. Scaling both means and standard errors
    by their larger mean preserves their difference test exactly. Finite-run
    standard errors still cannot exclude unseen rare high-weight populations.
    """
    logs = np.asarray([first['log_mean_normalizer'], second['log_mean_normalizer']], float)
    relative = np.asarray([first['normalizer_relative_mcse'],
                           second['normalizer_relative_mcse']], float)
    if not np.isfinite(logs).all() or not np.isfinite(relative).all() or np.any(relative < 0):
        raise ValueError('finite surviving normalizer estimates and nonnegative errors required')
    scaled = np.exp(logs-np.max(logs))
    difference = float(scaled[0]-scaled[1])
    error = float(np.hypot(*(scaled*relative)))
    return dict(normalizer_scaled_difference=difference,
                normalizer_scaled_combined_mcse=error,
                normalizer_agreement=bool(abs(difference) <= 3.5*error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--comparison', type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(args.runs) < 2 or len(set(p.resolve() for p in args.runs)) != len(args.runs):
        raise ValueError('fresh output and distinct complete independent runs required')
    protocols, results, sources, implementations, decoded = [], [], {}, [], []
    for run in args.runs:
        p = json.loads((run/'protocol.json').read_text())
        r = json.loads((run/'result.json').read_text())
        if (r['status'] not in ('completed', 'completed_with_extinction') or r['protocol_sha256'] != digest(run/'protocol.json')
                or r['particles_sha256'] != digest(run/'particles.npz')):
            raise ValueError(f'incomplete or changed independent run: {run}')
        decoded_result = validate_completed_result(p, r)
        if p.get('extinction_schema') == EXTINCTION_SCHEMA:
            if r.get('scenes_sha256') != digest(run/'scenes.npz'):
                raise ValueError('missing or changed native scene artifact')
            with np.load(run/'particles.npz', allow_pickle=False) as stored:
                if (stored['extinct'].dtype != np.dtype(bool)
                        or not np.array_equal(stored['extinct'], decoded_result[4])
                        or not np.array_equal(stored['log_normalizer'], decoded_result[0])):
                    raise ValueError('particle artifact and explicit extinction receipt disagree')
        decoded.append(decoded_result)
        fingerprint = implementation_fingerprint(p)
        if protocols:
            keys = ['observations', 'fixed_shear', 'reference_protocol_sha256', 'observation_sha256',
                    'density_batch_size', 'particles', 'stages', 'moves', 'betas', 'scale', 'proposals',
                    'blocks', 'bridge', 'algorithm', 'selection']
            if any(p[k] != protocols[0][k] for k in keys):
                raise ValueError('each group must contain identically configured independent runs')
            if p.get('extinction_schema') != protocols[0].get('extinction_schema'):
                raise ValueError('each group must use the same extinction receipt schema')
            if geometry_fingerprint(p) != geometry_fingerprint(protocols[0]):
                raise ValueError('each group must use the same frozen physical geometry')
            if proposal_fingerprint(p) != proposal_fingerprint(protocols[0]):
                raise ValueError('each group must use the same frozen initial proposal')
            if fingerprint != implementations[0]:
                raise ValueError('each group must use identical recorded implementation versions')
        protocols.append(p); results.append(r); implementations.append(fingerprint)
        names = ['protocol.json', 'result.json', 'particles.npz']
        if p.get('extinction_schema') == EXTINCTION_SCHEMA:
            names.append('scenes.npz')
        for name in names:
            sources[str(run/name)] = digest(run/name)
    seeds = [p['seed'] for p in protocols]
    if len(set(seeds)) != len(seeds):
        raise ValueError('independent populations require distinct seeds')
    if set(seeds) & pilot_seeds(protocols[0]):
        raise ValueError('production seeds overlap frozen-artifact pilots')
    logz = np.asarray([v[0] for v in decoded])
    means = np.asarray([v[1] for v in decoded])
    variances = np.asarray([v[2] for v in decoded])
    regions = np.asarray([v[3] for v in decoded])
    rows = []
    for i, oid in enumerate(protocols[0]['observations']):
        surviving = np.isfinite(logz[:, i])
        if not surviving.any():
            rows.append(dict(observation=oid, all_populations_extinct=True,
                score=undefined_population_summary(len(results), 2),
                region=undefined_population_summary(len(results), 2),
                posterior_score_variance_estimate=[None, None],
                score_effective_count_from_replicates=[None, None], precision_gate=False))
            continue
        score = combine_populations(logz[:, i], means[:, i])
        region = combine_populations(logz[:, i], regions[:, i])
        second = score['weights']@empirical_second_moment(
            np.where(surviving[:, None], means[:, i], 0.),
            np.where(surviving[:, None], variances[:, i], 0.), protocols[0]['particles'])
        variance = np.maximum(0., second-score['mean']**2)
        mcvar = np.diag(score['covariance'])
        ess = np.divide(variance, mcvar, out=np.full(2, np.inf), where=mcvar > 0)
        precision_gate = bool(len(results) >= 8 and np.isfinite(ess).all() and np.all(ess >= 400)
            and score['normalizer_relative_mcse'] < .1 and score['run_weight_ess'] >= .8*len(results))
        rows.append(dict(observation=oid, all_populations_extinct=False, score=serial(score), region=serial(region),
            posterior_score_variance_estimate=variance.tolist(),
            score_effective_count_from_replicates=np.where(np.isfinite(ess), ess.astype(object), None).tolist(),
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
        initial_proposal=proposal_fingerprint(protocols[0]),
        sources=sources, script_sha256=digest(__file__),
        limitations=['Delta-method errors from complete independent runs; finite-run uncertainty can miss rare events.',
                    'No nominal particle ESS/Rhat is used as an independent posterior sample count.',
                    'Recorded source hashes cover the declared files, not every transitive dependency.',
                    'Explicitly extinct observations retain zero mass and their seed; unknown failures remain errors.',
                    'Passing precision alone does not establish accuracy or held-out efficiency.'])
    if args.comparison:
        other = json.loads(args.comparison.read_text())
        for key in ['centre', 'reference_protocol_sha256', 'observation_sha256', 'density_batch_size',
                    'target_source_sha256', 'bridge', 'selection', 'diagnostics_version']:
            if other.get(key) != output[key]:
                raise ValueError('comparison targets or numerical likelihood batches disagree')
        if set(seeds) & set(other['seeds']):
            raise ValueError('unpaired comparison requires disjoint population seeds')
        if set(seeds) & pilot_seeds(other) or set(other['seeds']) & pilot_seeds(output):
            raise ValueError('comparison production seeds overlap frozen-artifact pilots')
        if [o['observation'] for o in other['observations']] != [o['observation'] for o in rows]:
            raise ValueError('comparison observations differ')
        comparisons = []
        for a, b in zip(rows, other['observations']):
            if a.get('all_populations_extinct') or b.get('all_populations_extinct'):
                comparisons.append(dict(observation=a['observation'], agreement=False,
                    reason='posterior mean undefined because at least one group is entirely extinct',
                    score_difference=[None, None], score_combined_mcse=[None, None],
                    region_difference=[None, None], region_combined_mcse=[None, None],
                    normalizer_scaled_difference=None, normalizer_scaled_combined_mcse=None,
                    normalizer_agreement=False))
                continue
            score_difference = np.asarray(a['score']['mean'])-b['score']['mean']
            score_se = np.hypot(a['score']['mcse'], b['score']['mcse'])
            sd = np.sqrt((np.asarray(a['posterior_score_variance_estimate'])+b['posterior_score_variance_estimate'])/2)
            region_difference = np.asarray(a['region']['mean'])-b['region']['mean']
            region_se = np.hypot(a['region']['mcse'], b['region']['mcse'])
            normalizer = compare_normalizers(a['score'], b['score'])
            agreement = bool(normalizer['normalizer_agreement']
                and np.all(np.abs(score_difference) <= 3.5*score_se)
                and np.all(np.abs(score_difference) <= .1*sd)
                and np.all(np.abs(region_difference) <= 3.5*region_se)
                and np.all(np.abs(region_difference) <= .05))
            comparisons.append(dict(observation=a['observation'], score_difference=score_difference.tolist(),
                score_combined_mcse=score_se.tolist(), region_difference=region_difference.tolist(),
                region_combined_mcse=region_se.tolist(), **normalizer, agreement=agreement))
        output['comparison'] = dict(path=str(args.comparison.resolve()), sha256=digest(args.comparison),
                                    observations=comparisons)
        output['comparison_gate'] = (output['all_precision_gates'] and other['all_precision_gates']
                                     and all(c['agreement'] for c in comparisons))
    save(args.output, output)
    print('PARTICLE_REPORT', args.output, 'precision', output['all_precision_gates'],
          'comparison', output['comparison_gate'], flush=True)


if __name__ == '__main__':
    main()
