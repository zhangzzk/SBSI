#!/usr/bin/env python3
"""Completed n16 development comparison; no fits or changed scientific gates."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_v9_particle_comparison import (COEFFICIENT_HASH, COLUMNS, MEASURES,
    OBSERVATIONS, normalizer_comparison, qualified_moments, sha256, signed_comparison)


def comparison_status(report):
    """Absence of a comparator is distinct from a failed comparison."""
    return {'historical_boolean': report['comparison_gate'],
            'status': ('not_evaluated_no_comparator' if 'comparison' not in report
                       else 'pass' if report['comparison_gate'] else 'fail')}


def validate_panel(report, phase, expected_seeds):
    if report['seeds'] != expected_seeds or phase['seeds'] != expected_seeds:
        raise ValueError('expected all eight declared populations in order')
    if len(report['runs']) != 8 or report['replicas'] != 8 or len(phase['sources']) != 8:
        raise ValueError('missing population')
    if any([row['observation'] for row in obj['observations']] != OBSERVATIONS for obj in (report, phase)):
        raise ValueError('wrong observation panel')
    if phase['moment_columns'] != COLUMNS or phase['coefficients']['sha256'] != COEFFICIENT_HASH:
        raise ValueError('different frozen coefficients or moments')


def build_summary(root, endpoint_receipt):
    root = Path(root).resolve()
    sources = {}

    def record(path, expected=None):
        path = Path(path).resolve()
        if root not in path.parents or 'heldout' in str(path.relative_to(root)).lower():
            raise ValueError('outside allowed development root')
        digest = sources.get(str(path)) or sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'hash mismatch: {path}')
        sources[str(path)] = digest
        return path

    def read(path, expected=None):
        return json.loads(record(path, expected).read_text())

    declaration = read(root/'experiment_v8_informed_environment.json')
    coefficients = read(root/'phase_n15_20261022_v2/frozen_v1_coefficients.json', COEFFICIENT_HASH)
    audit = read(endpoint_receipt)
    groups, reports, identities = {}, {}, []
    for arm in ('baseline', 'informed'):
        report = read(root/f'dev_v8_{arm}_report.json')
        phase = read(root/f'phase_v8_20261022/{arm}_controls.json')
        seeds = list(range(declaration[f'{arm}_seed_base'], declaration[f'{arm}_seed_base']+8))
        validate_panel(report, phase, seeds)
        if set(seeds).intersection(coefficients['metadata']['pilot_seeds']):
            raise ValueError('pilot seed leakage')
        for path, digest in report['sources'].items():
            record(path, digest)
        checks, timings, runtimes, configurations = [], [], [], []
        for index, (path, digest) in enumerate(phase['sources'].items()):
            diagnostic = read(path, digest)
            if Path(diagnostic['run']).name != f'dev_v8_{arm}_{index}' or diagnostic['seed'] != seeds[index]:
                raise ValueError('diagnostic ordering mismatch')
            for source_path, source_hash in diagnostic['source_hashes'].items():
                record(source_path, source_hash)
            record(Path(path).with_suffix('.npz'), diagnostic['arrays_sha256'])
            protocol = read(Path(diagnostic['run'])/'protocol.json')
            result = read(Path(diagnostic['run'])/'result.json')
            if diagnostic['stride'] != 1 or diagnostic['retained_particles'] != protocol['particles']:
                raise ValueError('all original particles required')
            if not diagnostic['numerical_checks_pass'] or result['status'] != 'completed' or any(result.get('extinct', [])):
                raise ValueError('failed/incomplete/extinct declared population')
            if protocol['seed'] != seeds[index] or protocol['observations'] != OBSERVATIONS:
                raise ValueError('protocol population mismatch')
            config = {k: protocol[k] for k in ('particles','stages','moves','blocks','scale','fixed_shear')}
            if any(config[k] != declaration[k] for k in ('particles','stages','moves')):
                raise ValueError('declaration effort mismatch')
            configurations.append(config)
            timings.append(result['timing_seconds'])
            runtimes.append(protocol['runtime'])
            checks.append(diagnostic['numerical_checks'])
        if any(c != configurations[0] for c in configurations):
            raise ValueError('within-arm settings differ')
        observations = []
        for raw, row in zip(report['observations'], phase['observations']):
            a = row[MEASURES[0]]
            for key in ('mean','mcse'):
                if any(not math.isclose(x,y,abs_tol=1e-11,rel_tol=1e-11) for x,y in zip(raw['score'][key],a[key][1:3])):
                    raise ValueError('raw score/MCSE replay mismatch')
            if raw['score']['weights'] != a['weights'] or raw['score']['log_mean_normalizer'] != a['log_mean_normalizer']:
                raise ValueError('raw normalizer/weight replay mismatch')
            if raw['region']['weights'] != a['weights']:
                raise ValueError('region population weights differ')
            for measure in MEASURES:
                if row[measure]['independent_populations'] != 8 or row[measure]['extinct_populations'] != 0:
                    raise ValueError('incomplete aggregate')
            observations.append({'observation':row['observation'],
                **{m:qualified_moments(row[m],m) for m in MEASURES},
                'region_raw_float32_sampled':raw['region'],
                'original_precision_gate':raw['precision_gate'],
                'precision_mean_shift':[y-x for x,y in zip(a['mean'],row[MEASURES[1]]['mean'])],
                'precision_shift_in_raw_mcse':[(y-x)/s if s else None for x,y,s in zip(a['mean'],row[MEASURES[1]]['mean'],a['mcse'])]})
        groups[arm] = {'seeds':seeds,'configuration':configurations[0],'observations':observations,
            'runtime':runtimes,'timing_seconds_by_run':timings,
            'total_compute_seconds':sum(t['total'] for t in timings),
            'median_seconds_per_run':statistics.median(t['total'] for t in timings),
            'original_all_precision_gates':report['all_precision_gates'],
            'original_comparison':comparison_status(report),
            'initial_proposal_sha256':report['initial_proposal']['sha256'],
            'geometry_sha256':report['physical_geometry']['sha256'],
            'informed_environment':report['informed_environment'],
            'target_source_sha256':report['target_source_sha256'],
            'bridge':report['bridge'],'endpoint_target':report['endpoint_target'],
            'numerical_checks':{'maximum_logK_F64_minus_F32_absolute':max(c['sampled_float32_float64_max'] for c in checks),
                'maximum_phase_scaled_error':max(d['max_scaled_error'] for c in checks for d in c['phase_differences'])}}
        identities.append(phase['target_identity'])
        reports[arm] = report
    a,b = groups['informed'],groups['baseline']
    if set(a['seeds']).intersection(b['seeds']):
        raise ValueError('arms share seeds')
    if identities[0] != identities[1]:
        raise ValueError('phase target identity mismatch')
    for key in ('configuration','initial_proposal_sha256','geometry_sha256','target_source_sha256','endpoint_target'):
        if a[key] != b[key]:
            raise ValueError(f'arm mismatch: {key}')
    for key,report_key in [('initial_proposal','initial_proposal_sha256'),('physical_geometry','geometry_sha256')]:
        if a[report_key] != declaration[key]['sha256']:
            raise ValueError('declared proposal/geometry mismatch')
    comparison = reports['informed']['comparison']
    record(comparison['path'],comparison['sha256'])
    ratio = a['total_compute_seconds']/b['total_compute_seconds']
    contrasts = []
    for x,y,original in zip(a['observations'],b['observations'],comparison['observations']):
        region = signed_comparison(x['region_raw_float32_sampled'],y['region_raw_float32_sampled'])
        if any(not math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-15) for key,original_key in [('difference','region_difference'),('combined_mcse','region_combined_mcse')] for x,y in zip(region[key],original[original_key])):
            raise ValueError('region comparison replay mismatch')
        contrasts.append({'observation':x['observation'],
            **{m:{**signed_comparison(x[m],y[m]),'normalizer':normalizer_comparison(x[m],y[m]),
                'estimated_variance_times_cost_ratio':[(u/v)**2*ratio if v else None for u,v in zip(x[m]['mcse'],y[m]['mcse'])]}
                for m in MEASURES},'region_raw_float32_sampled':region,
            'original_raw_comparison_agreement':original['agreement']})
    # Historical score contrasts are descriptive under the separately reviewed receipt.
    # They do not invoke or bypass the strict raw-report compatibility gate.
    historical = read(root/'v9_comparison_20261022.json', '92efbd83fa70bbfa98e62d012e14412816f6aa13b2018cf0ef3c2acf00815df4')
    if historical['target_identity'] != identities[0]:
        raise ValueError('historical phase target identity differs')
    history = {}
    baseline = b['observations'][1]
    for name,g in historical['groups'].items():
        if set(g['seeds']).intersection(a['seeds']+b['seeds']):
            raise ValueError('historical contrast is not independent')
        other = g['observations'][1]
        history[name] = {m:signed_comparison(baseline[m],other[m]) for m in MEASURES}
    return {'schema':'sbsi-n16-completed-comparison-v1','moment_columns':COLUMNS,
        'region_columns':['target_within_two_magnitudes','target_over_three_magnitudes_fainter'],
        'region_measure':'raw F32 sampled only; no F64 region reweighting calculated',
        'target_identity':identities[0],'coefficient_sha256':COEFFICIENT_HASH,
        'groups':groups,'informed_minus_baseline':{'total_compute_cost_ratio':ratio,'observations':contrasts},
        'baseline_minus_historical_obs0_descriptive':history,
        'endpoint_receipt':{'path':str(endpoint_receipt),'sha256':sha256(endpoint_receipt),
            'conclusion':audit['scientific_conclusion'],'strict_comparison_gate_bypassed':False},
        'sources':sources,'script_sha256':sha256(__file__),
        'calculation_helper_sha256':sha256(Path(__file__).with_name('summarize_v9_particle_comparison.py')),
        'accepted_sampler':False,'raw_score_normalizer_region_gates_unchanged':True,
        'limitations':[
            'Old baseline can admit marks with zero continuous numerical environment density despite finite analytic target intensity; informed arm has a support guard. Occurrence in these trajectories is unknown. Differences cannot be attributed solely to informed mass.',
            'Matching declared endpoint factors do not certify the old transition kernels or finite-RNG exactness; corrected source is a separate experiment.',
            'Eight independent populations provide estimated ratio MCSE, not proof against shared rare-mode bias.',
            'F32 control expectation unknown; F64 reweighting changes numerical measure and zero expectation is continuous-model qualified.',
            'No fitting, heldout, new model forward, dropped states or acceptance-rule changes.',
            'Recorded inference cost excludes pilot fitting, phase diagnostics, queueing and startup outside driver timing. Variance times cost is descriptive and ignores bias.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--endpoint-receipt',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=build_summary(args.root,args.endpoint_receipt)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'sha256':sha256(args.output),'inputs':len(result['sources'])}))


if __name__=='__main__':main()
