#!/usr/bin/env python3
"""Read-only n18 comparison; consumes audited all-population aggregates, never fits."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

GROUPS = {
    'v6v100': ('dev_v6v100_report_v4.json', 'phase_n15_20261022_v2/v6v100_controls.json'),
    'v7': ('dev_v7_report.json', 'phase_n15_20261022_v2/v7_controls.json'),
    'v9': ('dev_v9_vs_v6_report.json', 'phase_v9_20261022/controls.json'),
}
OBSERVATIONS = [58165, 0, 1, 8]
COLUMNS = ['T', 'raw_g1', 'raw_g2', 'corrected_g1', 'corrected_g2']
MEASURES = ['raw_float32_sampled', 'precision_reweighted_float64']
COEFFICIENT_HASH = 'a31062ffe9c970d8c3ae9214f5f9b7a67781829cd14d37f8ee2461fb155412c7'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _hash_stream(stream)


def _hash_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def signed_comparison(a, b):
    """Independent population groups: A minus B, with no absolute-value sign loss."""
    if not (len(a['mean']) == len(b['mean']) == len(a['mcse']) == len(b['mcse'])):
        raise ValueError('inconsistent moment dimensions')
    if any(not math.isfinite(x) for obj in (a, b) for key in ('mean', 'mcse') for x in obj[key]):
        raise ValueError('nonfinite moments')
    if any(x < 0 for obj in (a, b) for x in obj['mcse']):
        raise ValueError('negative MCSE')
    difference = [x-y for x, y in zip(a['mean'], b['mean'])]
    uncertainty = [math.hypot(x, y) for x, y in zip(a['mcse'], b['mcse'])]
    return {'difference': difference, 'combined_mcse': uncertainty,
            'signed_standardized_difference': [d/s if s else (0. if d == 0 else None) for d, s in zip(difference, uncertainty)],
            'nonzero_with_zero_estimated_mcse': [s == 0 and d != 0 for d, s in zip(difference, uncertainty)]}


def normalizer_comparison(a, b):
    shift = max(a['log_mean_normalizer'], b['log_mean_normalizer'])
    za, zb = (math.exp(x['log_mean_normalizer']-shift) for x in (a, b))
    out = signed_comparison({'mean': [za], 'mcse': [za*a['normalizer_relative_mcse']]},
                            {'mean': [zb], 'mcse': [zb*b['normalizer_relative_mcse']]})
    return {'log_ratio': a['log_mean_normalizer']-b['log_mean_normalizer'],
            'ratio': math.exp(a['log_mean_normalizer']-b['log_mean_normalizer']),
            'common_log_scale': shift, **out}


def qualified_moments(source, measure):
    result = copy.deepcopy(source)
    result['historical_control_known_expectation'] = result.pop('control_known_expectation', None)
    # Preserve old files, but never repeat their false F32 known-zero metadata.
    result['control_known_expectation'] = None if measure == MEASURES[0] else 0.
    result['control_expectation_qualification'] = ('unknown for rounded F32 sampling program' if measure == MEASURES[0]
        else 'zero for intended continuous F64 posterior; numerically evaluated, not an exact rounded-program identity')
    result['control_standardized_residual_reference'] = 'zero, a diagnostic reference only under the stated qualification'
    return result


def build_summary(root):
    root = Path(root).resolve()
    sources = {}

    def record(path, expected=None):
        path = Path(path).resolve()
        # This task has a fixed allowlist of development inputs; never traverse heldout.
        if root not in path.parents or 'heldout' in str(path.relative_to(root)).lower():
            raise ValueError(f'input outside development root: {path}')
        digest = sources.get(str(path)) or sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'input hash mismatch: {path}')
        sources[str(path)] = digest
        return path

    def read(path, expected=None):
        return json.loads(record(path, expected).read_text())

    declaration = read(root/'experiment_v9_higher_effort_fitted.json')
    coefficients = read(root/'phase_n15_20261022_v2/frozen_v1_coefficients.json', COEFFICIENT_HASH)
    extra_report = read(root/'dev_v9_vs_v7_report.json')
    groups, all_seeds, identities = {}, set(), []
    for name, (report_name, phase_name) in GROUPS.items():
        report, phase = read(root/report_name), read(root/phase_name)
        seeds = phase['seeds']
        if len(seeds) != 8 or len(set(seeds)) != 8 or seeds != report['seeds'] or all_seeds.intersection(seeds):
            raise ValueError('expected eight fresh independent populations per group')
        if set(seeds).intersection(coefficients['metadata']['pilot_seeds']):
            raise ValueError('pilot seed leakage')
        all_seeds.update(seeds)
        if phase['coefficients']['sha256'] != COEFFICIENT_HASH or phase['moment_columns'] != COLUMNS:
            raise ValueError('different frozen control law or moment order')
        if [o['observation'] for o in phase['observations']] != OBSERVATIONS:
            raise ValueError('wrong observation panel')
        identities.append(phase['target_identity'])
        for path, digest in report['sources'].items():
            record(path, digest)
        timings, runtimes, configurations, checks = [], [], [], []
        if len(phase['sources']) != 8:
            raise ValueError('missing phase populations')
        for run_index, (path, digest) in enumerate(phase['sources'].items()):
            diagnostic = read(path, digest)
            if Path(diagnostic['run']).name != f'dev_{name}_{run_index}' or diagnostic['seed'] != seeds[run_index]:
                raise ValueError('population ordering mismatch')
            if diagnostic['stride'] != 1 or not diagnostic['numerical_checks_pass']:
                raise ValueError('thinned or failed diagnostic')
            for source_path, source_hash in diagnostic['source_hashes'].items():
                record(source_path, source_hash)
            record(Path(path).with_suffix('.npz'), diagnostic['arrays_sha256'])
            protocol = read(Path(diagnostic['run'])/'protocol.json')
            result = read(Path(diagnostic['run'])/'result.json')
            if result['status'] != 'completed' or any(result.get('extinct', [])):
                raise ValueError('summary requires these declared complete nonextinct populations')
            if diagnostic['retained_particles'] != protocol['particles']:
                raise ValueError('not all original particles retained')
            if protocol['seed'] != seeds[run_index] or protocol['observations'] != OBSERVATIONS:
                raise ValueError('protocol/aggregate mismatch')
            timings.append(result['timing_seconds'])
            runtimes.append(protocol.get('runtime'))
            configurations.append({k: protocol[k] for k in ['particles', 'stages', 'moves', 'blocks', 'scale', 'fixed_shear']})
            checks.append(diagnostic['numerical_checks'])
        if any(c != configurations[0] for c in configurations):
            raise ValueError('within-group configuration mismatch')
        if report['replicas'] != 8 or len(report['runs']) != 8:
            raise ValueError('raw report population mismatch')
        observations = []
        for raw, row in zip(report['observations'], phase['observations']):
            if raw['observation'] != row['observation']:
                raise ValueError('report observation mismatch')
            a = row[MEASURES[0]]
            for key in ['mean', 'mcse']:
                if any(not math.isclose(x, y, abs_tol=1e-11, rel_tol=1e-11) for x, y in zip(raw['score'][key], a[key][1:3])):
                    raise ValueError('raw score replay mismatch')
            for measure in MEASURES:
                if row[measure]['independent_populations'] != 8 or row[measure]['extinct_populations'] != 0:
                    raise ValueError('incomplete aggregate')
            if raw['score']['weights'] != a['weights'] or raw['score']['log_mean_normalizer'] != a['log_mean_normalizer']:
                raise ValueError('raw normalizer/weight replay mismatch')
            observations.append({'observation': row['observation'],
                **{m: qualified_moments(row[m], m) for m in MEASURES},
                'precision_mean_shift': [y-x for x, y in zip(a['mean'], row[MEASURES[1]]['mean'])],
                'precision_shift_in_raw_mcse': [(y-x)/s if s else None for x, y, s in zip(a['mean'], row[MEASURES[1]]['mean'], a['mcse'])],
                'original_raw_precision_gate': raw['precision_gate']})
        groups[name] = {'configuration': configurations[0], 'seeds': seeds, 'runtime': runtimes,
            'initial_proposal_sha256': (report.get('initial_proposal') or {}).get('sha256'),
            'geometry_sha256': (report.get('physical_geometry') or {}).get('sha256'),
            'target_source_sha256': report['target_source_sha256'], 'observations': observations,
            'timing_seconds_by_run': timings,
            'total_compute_seconds': sum(t['total'] for t in timings),
            'median_seconds_per_run': statistics.median(t['total'] for t in timings),
            'mean_seconds_per_run': statistics.mean(t['total'] for t in timings),
            'original_all_precision_gates': report['all_precision_gates'],
            'numerical_checks': {'maximum_logK_F64_minus_F32_absolute': max(c['sampled_float32_float64_max'] for c in checks),
                'maximum_phase_scaled_error': max(d['max_scaled_error'] for c in checks for d in c['phase_differences'])}}
    if any(identity != identities[0] for identity in identities):
        raise ValueError('phase target identities differ')
    if groups['v9']['seeds'] != declaration['seeds'] or groups['v9']['initial_proposal_sha256'] != declaration['initial_proposal_sha256']:
        raise ValueError('n18 declaration mismatch')
    if groups['v7']['initial_proposal_sha256'] != groups['v9']['initial_proposal_sha256']:
        raise ValueError('fitted proposal changed')
    primary_report = read(root/GROUPS['v9'][0])
    if extra_report['observations'] != primary_report['observations']:
        raise ValueError('the two v9 raw reports differ')
    comparisons = {}
    for left, right in [('v9', 'v6v100'), ('v9', 'v7'), ('v7', 'v6v100')]:
        a, b = groups[left], groups[right]
        cost_ratio = a['total_compute_seconds']/b['total_compute_seconds']
        comparisons[f'{left}_minus_{right}'] = {'total_compute_cost_ratio': cost_ratio, 'observations': [
            {'observation': x['observation'], **{m: {**signed_comparison(x[m], y[m]),
                'normalizer': normalizer_comparison(x[m], y[m]),
                'estimated_variance_times_cost_ratio': [(u/v)**2*cost_ratio if v else None for u, v in zip(x[m]['mcse'], y[m]['mcse'])]}
                for m in MEASURES}} for x, y in zip(a['observations'], b['observations'])]}
    return {'schema': 'sbsi-n18-completed-comparison-v1', 'moment_columns': COLUMNS,
        'target_identity': identities[0], 'coefficient_sha256': COEFFICIENT_HASH,
        'groups': groups, 'comparisons': comparisons,
        'original_raw_comparison_gates': {'v9_vs_v6': primary_report['comparison_gate'], 'v9_vs_v7': extra_report['comparison_gate']},
        'sources': sources, 'script_sha256': sha256(__file__), 'accepted_sampler': False,
        'limitations': [
            'All groups share old native numerical proposal-support defect: lower uniform clamp omitted in q/p; environment may admit zero real-prior-density marks. Old trajectory occurrence unknown. Fix dca103e is not present in these trajectories.',
            'No exactness certification or exclusive attribution of discrepancies to equilibration.',
            'Eight independent populations provide estimated ratio MCSE, not a guarantee against shared rare-mode bias.',
            'F32 known control expectation is unknown; F64 reweighting is an explicit numerical measure change, with continuous-model identity qualification.',
            'Existing raw score/Z gates unchanged; corrected-score comparisons are diagnostics, not a new acceptance rule.',
            'Cost is recorded inference runtime for all four objects, excludes pilot fitting, phase evaluation, scheduling and startup outside driver timing; variance times cost is descriptive and ignores bias.',
            'No heldout inputs, new model evaluation, refitting, or population/particle dropping.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build_summary(args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha256(args.output), 'inputs': len(result['sources'])}))


if __name__ == '__main__':
    main()
