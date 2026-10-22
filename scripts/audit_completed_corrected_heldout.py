#!/usr/bin/env python
"""Read-only, completed-panel audit of the one frozen corrected heldout v3.

Requires the final result before any population/report reads. Imports provenance
validators from the pinned original source, but independently evaluates gates,
cost allocation and the final conjunction. Never runs a reporter or sampler.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021')
PANEL = ROOT/'heldout_corrected_v10_v3_20261022'
SOURCE = ROOT/'source_corrected_heldout_v3_20261022'
DECLARATION_SHA256 = 'ad718136d4ac00a37ffcaddf09dcfad92b15e68e4ebbf82ead597ed1b7c1f758'
MANIFEST_SHA256 = 'fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1'
MEASURES = ('raw_float32_sampled', 'precision_reweighted_float64')
COST_KEYS = ('process_wall_seconds', 'cpu_allocated_seconds', 'gpu_allocated_seconds')
ENDPOINT = 'q_environment(E) * p_analytic(x) * K(y|x,E)'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    def bad(value):
        raise ValueError('nonfinite JSON constant: '+value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: '+key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=bad, object_pairs_hook=unique)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completed_result(panel):
    """This is the only scientific file allowed before the completion barrier."""
    result = read(Path(panel)/'result.json')
    require(result.get('schema') == 'sbsi-corrected-heldout-panel-result-v1',
            'a final panel result, including a failed scientific gate, is required')
    require([g['group'] for g in result['groups']] == list(range(5))
            and all(len(g['observations']) == 4 for g in result['groups']),
            'all five groups and twenty observations required before inspecting reports')
    for key in ('heldout_accuracy_gate', 'heldout_cost_gate', 'heldout_gate', 'accepted_sampler'):
        require(type(result.get(key)) is bool, 'final boolean decision required: '+key)
    require(set(result.get('costs', {}).get('recipes', {})) == {'candidate', 'reference'},
            'final costs for both recipes required')
    return result


def load_authority():
    """Verify every source byte before executing the original validators."""
    require(digest(PANEL/'declaration.json') == DECLARATION_SHA256, 'wrong frozen declaration')
    d = read(PANEL/'declaration.json')
    require(d['source'] == str(SOURCE) and d['output'] == str(PANEL)
            and d['source_manifest_sha256'] == MANIFEST_SHA256, 'wrong source/panel authority')
    require(digest(SOURCE/'SHA256SUMS') == MANIFEST_SHA256, 'changed frozen manifest')
    entries = {}
    for line in (SOURCE/'SHA256SUMS').read_text().splitlines():
        checksum, name = line.split(None, 1)
        name = name.lstrip('*')
        path = (SOURCE/name).resolve()
        require(SOURCE in path.parents and name not in entries, 'unsafe/duplicate manifest path')
        require(digest(path) == checksum, 'changed frozen source: '+name)
        entries[name] = checksum
    require(len(entries) == 320, 'complete original 320-file manifest required')
    for name, module in tuple(sys.modules.items()):
        if name == 'sbsi' or name.startswith('sbsi.'):
            path = Path(getattr(module, '__file__', '')).resolve()
            require(SOURCE in path.parents, 'fresh interpreter required; foreign sbsi already imported')
    sys.path[:0] = [str(SOURCE), str(SOURCE/'scripts')]
    for name in ('fit_declared_particle_phase_control', 'report_galsbi_particles'):
        require(name not in sys.modules, 'fresh interpreter required; validator dependency already imported')
    spec = importlib.util.spec_from_file_location('_frozen_heldout_authority',
                                                SOURCE/'scripts/run_corrected_particle_heldout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked, checked_entries = module.validate_declaration(PANEL/'declaration.json')
    require(checked == d and checked_entries == entries, 'frozen validator authority mismatch')
    return module, d, entries


def finite(value, label, *, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0), 'invalid '+label)
    return float(value)


def vector(value, length, label, *, nonnegative=False):
    require(isinstance(value, list) and len(value) == length, 'wrong shape: '+label)
    return [finite(x, label, nonnegative=nonnegative) for x in value]


def same_flag(saved, computed, label):
    require(type(saved) is bool and saved == computed, 'recorded decision disagrees: '+label)


def summary(value, length):
    require(value['independent_populations'] == 8 and value['extinct_populations'] == 0,
            'exactly eight original nonextinct populations required')
    vector(value['weights'], 8, 'whole-population weights', nonnegative=True)
    return (vector(value['mean'], length, 'moment mean'),
            vector(value['mcse'], length, 'moment MCSE', nonnegative=True))


def precision(row):
    """Recompute the original raw-score precision gate, not its saved boolean."""
    require(row.get('all_populations_extinct') is False, 'raw nonextinction must be explicit')
    score = row['score']
    summary(score, 2)
    summary(row['region'], 2)
    variance = vector(row['posterior_score_variance_estimate'], 2, 'posterior variance', nonnegative=True)
    covariance = score['covariance']
    require(isinstance(covariance, list) and len(covariance) == 2, 'two-dimensional covariance required')
    covariance = [vector(r, 2, 'covariance') for r in covariance]
    effective = [v/covariance[i][i] if covariance[i][i] > 0 else None for i, v in enumerate(variance)]
    saved = row['score_effective_count_from_replicates']
    require(len(saved) == 2 and all((a is None and b is None) or
            (a is not None and b is not None and a == b) for a, b in zip(effective, saved)),
            'effective counts disagree with saved variance/covariance')
    relative = finite(score['normalizer_relative_mcse'], 'normalizer relative MCSE', nonnegative=True)
    run_ess = finite(score['run_weight_ess'], 'run-weight ESS', nonnegative=True)
    requirements = {
        **{f'raw_score_{i+1}_effective_count_ge_400': x is not None and math.isfinite(x) and x >= 400
           for i, x in enumerate(effective)},
        'normalizer_relative_mcse_lt_0.1': relative < .1,
        'run_weight_ess_ge_6.4': run_ess >= .8*8,
    }
    passed = all(requirements.values())
    same_flag(row['precision_gate'], passed, 'raw precision')
    return dict(requirements=requirements, pass_all=passed, effective_counts=effective,
                normalizer_relative_mcse=relative, run_weight_ess=run_ess)


def difference(a, b, length, *, indices=None):
    import numpy as np
    ma, sa = summary(a, length)
    mb, sb = summary(b, length)
    indices = list(range(length)) if indices is None else indices
    delta = np.asarray(ma)[indices]-np.asarray(mb)[indices]
    se = np.hypot(np.asarray(sa)[indices], np.asarray(sb)[indices])
    passed = (np.abs(delta) <= 3.5*se).tolist()
    return dict(difference=delta.tolist(), combined_mcse=se.tolist(), component_pass=passed,
                pass_all=all(passed))


def audit_object(a, b, pa, pb, saved):
    import numpy as np
    require(len({r['observation'] for r in (a, b, pa, pb, saved)}) == 1, 'misaligned observation')
    ca, cb = precision(a), precision(b)
    score, region = difference(a['score'], b['score'], 2), difference(a['region'], b['region'], 2)
    sd = np.sqrt((np.asarray(a['posterior_score_variance_estimate'])
                  + np.asarray(b['posterior_score_variance_estimate']))/2)
    score_absolute = (np.abs(score['difference']) <= .1*sd).tolist()
    region_absolute = (np.abs(region['difference']) <= .05).tolist()
    logs = np.asarray([finite(r['score']['log_mean_normalizer'], 'log mean normalizer') for r in (a, b)])
    mass = np.exp(logs-logs.max())
    normalizer_delta = float(mass[0]-mass[1])
    normalizer_se = float(np.hypot(mass[0]*ca['normalizer_relative_mcse'],
                                   mass[1]*cb['normalizer_relative_mcse']))
    normalizer_pass = abs(normalizer_delta) <= 3.5*normalizer_se
    requirements = {}
    for name, value in (('candidate', ca), ('reference', cb)):
        requirements.update({name+'.'+k: v for k, v in value['requirements'].items()})
    for name, value, absolute in (('raw_score', score, score_absolute), ('region', region, region_absolute)):
        for i in range(2):
            requirements[f'{name}_{i+1}.combined_mcse'] = value['component_pass'][i]
            requirements[f'{name}_{i+1}.absolute_bound'] = absolute[i]
    requirements['mean_normalizer.combined_mcse'] = normalizer_pass
    phase = {}
    for measure in MEASURES:
        phase[measure] = difference(pa[measure], pb[measure], 5, indices=[3, 4])
        for i, passed in enumerate(phase[measure]['component_pass']):
            requirements[f'{measure}.corrected_g{i+1}'] = passed
        require(saved['phase'][measure]['corrected']['component_pass'] == phase[measure]['component_pass'],
                'recorded corrected component gates disagree')
        same_flag(saved['phase'][measure]['corrected']['pass_all'], phase[measure]['pass_all'], measure)
    raw_pass = all(score['component_pass']+region['component_pass']+score_absolute+region_absolute+[normalizer_pass])
    for name, value in (('score', score), ('region', region)):
        require(saved['raw'][name]['component_pass'] == value['component_pass'], 'recorded raw component gates disagree')
        same_flag(saved['raw'][name]['pass_all'], value['pass_all'], name)
    require(saved['raw']['score_absolute_pass'] == score_absolute
            and saved['raw']['region_absolute_pass'] == region_absolute, 'recorded absolute gates disagree')
    same_flag(saved['raw']['normalizer']['pass_all'], normalizer_pass, 'mean normalizer')
    same_flag(saved['raw']['pass_all'], raw_pass, 'raw pair')
    passed = all(requirements.values())
    same_flag(saved['accuracy_pass'], passed, 'object accuracy')
    return dict(observation=a['observation'], accuracy_pass=passed, requirements=requirements,
                failed_requirements=[k for k, v in requirements.items() if not v],
                candidate_precision=ca, reference_precision=cb, raw_score=score, region=region,
                score_absolute_pass=score_absolute, region_absolute_pass=region_absolute,
                normalizer=dict(scaled_difference=normalizer_delta, combined_mcse=normalizer_se,
                                common_log_scale=float(logs.max()), pass_all=normalizer_pass), phase=phase)


def independent_costs(warmups, productions, report, preparation):
    """Independent accounting with the original allocation rule and counts."""
    require(len(warmups) == 5 and len(productions) == 40, 'all 45 pre-report stages required')
    recipes = {k: [] for k in ('candidate', 'reference')}
    shared, actual = [preparation], [preparation]
    count = dict(pilots=0, production_populations=0, phase_checks=0, fits=0, reports=0, step_receipts=0)
    for stage in warmups:
        expected = {f'pilot_{i}{suffix}' for i in range(4) for suffix in ('', '_phase')}
        expected |= {'geometry_fit', 'proposal_fit', 'coefficient_fit'}
        require(len(stage['steps']) == 11 and {s['tag'] for s in stage['steps']} == expected,
                'all eleven unique warmup costs required')
        for step in stage['steps']:
            recipes['candidate'].append(step['cost'])
            if step['tag'] != 'proposal_fit':
                recipes['reference'].append(step['cost'])
        count['pilots'] += 4
        count['phase_checks'] += 4
        count['fits'] += 3
    for i, stage in enumerate(productions):
        replica = i % 8
        order = ('candidate', 'reference') if replica % 2 == 0 else ('reference', 'candidate')
        expected = [f'{arm}_{replica}{suffix}' for arm in order for suffix in ('', '_phase')]
        require([s['tag'] for s in stage['steps']] == expected, 'complete counterbalanced production costs required')
        for step in stage['steps']:
            recipes[step['tag'].split('_')[0]].append(step['cost'])
        count['production_populations'] += 2
        count['phase_checks'] += 2
    expected = {f'group_{i}_{arm}_{label}' for i in range(5) for arm in recipes
                for label in ('raw_report', 'phase_report')}
    require(len(report['steps']) == 20 and {s['tag'] for s in report['steps']} == expected,
            'all twenty unique report costs required')
    for step in report['steps']:
        recipes[step['tag'].split('_')[2]].append(step['cost'])
    count['reports'] = 20
    for stage in [*warmups, *productions, report]:
        shared.append(stage['verification_overhead'])
        actual.append(stage['cost'])
        count['step_receipts'] += len(stage['steps'])
    def total(parts):
        return {key: math.fsum(finite(p[key], key, nonnegative=True) for p in parts) for key in COST_KEYS}
    overhead = total(shared)
    return dict(recipes={arm: {k: v+.5*overhead[k] for k, v in total(parts).items()}
                         for arm, parts in recipes.items()}, actual_work=total(actual),
                shared_verification_and_preflight=overhead), count


def compare_costs(saved, computed):
    for category in ('actual_work', 'shared_verification_and_preflight'):
        for key in COST_KEYS:
            require(math.isclose(finite(saved[category][key], key, nonnegative=True), computed[category][key],
                                 rel_tol=1e-12, abs_tol=1e-9), 'recorded cost disagrees: '+category+'/'+key)
    for arm in ('candidate', 'reference'):
        for key in COST_KEYS:
            require(math.isclose(finite(saved['recipes'][arm][key], key, nonnegative=True),
                                 computed['recipes'][arm][key], rel_tol=1e-12, abs_tol=1e-9),
                    'recorded recipe cost disagrees: '+arm+'/'+key)


def final_decision(groups, costs, saved):
    require([g['group'] for g in groups] == list(range(5))
            and all(len(g['observations']) == 4 for g in groups), 'complete ordered panel required')
    accuracy = all(row['accuracy_pass'] for group in groups for row in group['observations'])
    ca, cb = (costs['recipes'][arm]['process_wall_seconds'] for arm in ('candidate', 'reference'))
    require(ca > 0 and cb > 0, 'positive inclusive recipe costs required')
    efficiency = ca < cb
    answer = dict(heldout_accuracy_gate=accuracy, heldout_cost_gate=efficiency,
                  heldout_gate=accuracy and efficiency, accepted_sampler=accuracy and efficiency)
    for key, value in answer.items():
        same_flag(saved[key], value, key)
    require(math.isclose(saved['reference_to_candidate_cost_ratio'], cb/ca, rel_tol=1e-12, abs_tol=1e-12),
            'recorded inclusive cost ratio disagrees')
    return dict(answer, reference_to_candidate_cost_ratio=cb/ca)


def check_stage(runner, d, path, name, sources):
    root = Path(d['output'])
    stage_path = root/(name+'_stage.json')
    row = read(stage_path)
    runner.validate_stage_receipt(row, d, path, name)
    reservation = root/(name+'_started.json')
    started = read(reservation)
    require(started['declaration_sha256'] == digest(path)
            and started['slurm_job_id'] == row['slurm_job_id'], 'stage reservation differs')
    sources.update({str(p): digest(p) for p in (stage_path, reservation)})
    for step in row['steps']:
        tag = step['tag']
        group = int(tag.split('_')[1]) if name == 'report' else int(name.split('_')[1])//(8 if name.startswith('production') else 1)
        directory = root/f'group_{group}'
        receipt = directory/(tag+'_timing.json')
        log = directory/(tag+'.out')
        require(step['path'] == str(receipt) and step['output_sha256'] == digest(log),
                'step receipt/log identity differs')
        sources.update({str(p): digest(p) for p in (receipt, log)})
    return row


def run_audit(original):
    require(completed_result(PANEL) == original, 'the unchanged complete final result is required first')
    import numpy as np
    runner, d, entries = load_authority()
    require(np.__version__ == d['runtime']['numpy'] and sys.version.split()[0] == d['runtime']['python'],
            'original Python/NumPy versions required for independent gate arithmetic')
    declaration = PANEL/'declaration.json'
    require(original['declaration_sha256'] == DECLARATION_SHA256
            and original['script_sha256'] == entries['scripts/run_corrected_particle_heldout.py'],
            'final result has wrong declaration/runner authority')
    sources = {str(p): digest(p) for p in (PANEL/'result.json', declaration, SOURCE/'SHA256SUMS')}
    warmups, productions = [], []
    for group in range(5):
        runner.validate_warmup(d, declaration, entries, group)
        warmups.append(check_stage(runner, d, declaration, f'warmup_{group}', sources))
        complete = PANEL/f'group_{group}/warmup_complete.json'
        sources[str(complete)] = digest(complete)
        sources.update(read(complete)['artifacts'])
    for index in range(40):
        productions.append(check_stage(runner, d, declaration, f'production_{index}', sources))
    report_stage = check_stage(runner, d, declaration, 'report', sources)
    require(original['stage_sha256'] == digest(PANEL/'report_stage.json'), 'final report-stage hash differs')
    groups = []
    for group in range(5):
        directory = PANEL/f'group_{group}'
        records = {}
        expected_reports = {str(directory/name): digest(directory/name) for name in (
            'candidate_report.json', 'reference_report.json', 'candidate_phase_report.json', 'reference_phase_report.json')}
        require(original['groups'][group]['report_sha256'] == expected_reports, 'final report hashes differ')
        sources.update(expected_reports)
        for arm in ('candidate', 'reference'):
            raw_sources, diagnostics = {}, []
            for replica in range(8):
                diagnostic, _ = runner.validate_phase(d, entries, group, replica, arm)
                diagnostics.append(diagnostic)
                run, phase_path = runner.run_paths(d, group, replica, arm)
                raw_sources.update({str(run/name): digest(run/name) for name in
                                    ('protocol.json', 'result.json', 'particles.npz', 'scenes.npz')})
                sources.update({str(p): digest(p) for p in (phase_path, phase_path.with_suffix('.npz'))})
            sources.update(raw_sources)
            raw, phase = read(directory/f'{arm}_report.json'), read(directory/f'{arm}_phase_report.json')
            runner.validate_report(d, group, arm, raw, phase, directory)
            require(raw['sources'] == raw_sources and raw['runs'] == [str(directory/f'{arm}_{i}') for i in range(8)],
                    'raw report must bind every original population artifact')
            require(raw['script_sha256'] == entries['scripts/report_galsbi_particles.py']
                    and phase['script_sha256'] == entries['scripts/report_particle_phase_control.py'],
                    'reporter source identity differs')
            require(phase['target_identity'] == diagnostics[0]['target_identity']
                    and phase['coefficients'] == diagnostics[0]['coefficients'], 'phase report target/fit differs')
            records[arm] = raw, phase
        a, pa = records['candidate']
        b, pb = records['reference']
        runner.comparison_target_identity(b, a)
        require(pa['target_identity'] == pb['target_identity'] and pa['coefficients'] == pb['coefficients'],
                'candidate/reference phase target or frozen control differs')
        saved = original['groups'][group]
        rows = [audit_object(*parts) for parts in zip(a['observations'], b['observations'],
                pa['observations'], pb['observations'], saved['observations'])]
        same_flag(saved['accuracy_pass'], all(r['accuracy_pass'] for r in rows), 'group accuracy')
        for arm, raw in (('candidate', a), ('reference', b)):
            same_flag(raw['all_precision_gates'], all(r[arm+'_precision']['pass_all'] for r in rows), arm+' precision aggregate')
        groups.append(dict(group=group, observations=rows, accuracy_pass=all(r['accuracy_pass'] for r in rows)))
    costs, counts = independent_costs(warmups, productions, report_stage,
                                     runner.bound_json(d['preparation_cost_receipt']))
    require(counts == dict(pilots=20, production_populations=80, phase_checks=100,
                          fits=15, reports=20, step_receipts=235), 'mandatory accounting counts differ')
    compare_costs(original['costs'], costs)
    decisions = final_decision(groups, costs, original)
    require(original['search_and_development_cost_included'] is False, 'unexpected cost-scope claim')
    # Detect any changed original input/output while this read-only audit ran.
    after, after_entries = runner.validate_declaration(declaration)
    require(after == d and after_entries == entries, 'frozen authority changed during audit')
    require(completed_result(PANEL) == original, 'final result changed during audit')
    runner.check_hash_map(sources)
    return dict(status='audited', evidence_valid=True, **decisions, groups=groups, costs=costs, counts=counts,
        panel_failed_requirements=[name for name in ('heldout_accuracy_gate', 'heldout_cost_gate')
                                   if not decisions[name]],
        failed_objects=[dict(group=g['group'], observation=r['observation'], reasons=r['failed_requirements'])
                        for g in groups for r in g['observations'] if not r['accuracy_pass']],
        authority=dict(declaration_sha256=DECLARATION_SHA256, source_manifest_sha256=MANIFEST_SHA256,
                       frozen_source=str(SOURCE), original_result_sha256=sources[str(PANEL/'result.json')]),
        numpy=np.__version__,
        thresholds=dict(score_effective_count_minimum=400, normalizer_relative_mcse_strict_maximum=.1,
            run_weight_ess_minimum=6.4, independent_populations_per_arm=8, combined_mcse_multiplier=3.5,
            raw_score_absolute_posterior_sd_fraction=.1, region_absolute_mass_difference=.05,
            cost_comparison='candidate summed inclusive process wall strictly less than reference'),
        region_definitions=['target within two magnitudes of measured flux', 'target over three magnitudes fainter'],
        sources=sources, requirements=dict(complete_original_evidence=True, frozen_target_identity=True,
            independent_raw_and_controlled_gates=True, inclusive_costs=True,
            all_twenty_objects_pass=decisions['heldout_accuracy_gate'],
            strictly_lower_inclusive_process_wall=decisions['heldout_cost_gate']),
        scope=dict(endpoint_target=ENDPOINT, f64_measure='likelihood-only exp(logK64-logK32) reweight; same p_A and q_E',
            numerical_mock_target_qN_validated=False, numerical_prior_reweight_performed=False,
            population_shear_calibration_validated=False, universal_mode_coverage_proved=False,
            new_cpu_optimization_or_500_cpu_throughput_validated=False,
            supplementary_not_gated=['controlled-score precision/absolute-SD bounds', 'T=0',
                'all-five-moment comparisons', 'F64 normalizer precision/agreement', 'F32/F64 equivalence'],
            cost_exclusions=['search/development', 'queue time', 'Slurm time outside wrappers', 'final receipt serialization']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON outside original panel/source; never overwritten')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output not in (PANEL, SOURCE) and PANEL not in output.parents and SOURCE not in output.parents,
            'audit output must be outside the original panel and source')
    require(not output.exists(), 'audit output must not already exist')
    # Do not create an audit receipt or inspect any intermediate population if final is absent.
    original = completed_result(PANEL)
    require(os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURM_JOB_GPUS'),
            'complete-panel verification requires a CPU-only Slurm allocation')
    started = time.monotonic()
    receipt = dict(schema='sbsi-completed-corrected-heldout-audit-v1', status='invalid_evidence',
                   evidence_valid=False, accepted_sampler=False, script_sha256=digest(__file__),
                   slurm_job_id=os.environ['SLURM_JOB_ID'], python=sys.version.split()[0])
    try:
        receipt.update(run_audit(original))
        require(digest(__file__) == receipt['script_sha256'], 'independent audit source changed during execution')
    except Exception as error:
        receipt.update(status='invalid_evidence', evidence_valid=False, accepted_sampler=False,
                       error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['audit_process_seconds'] = time.monotonic()-started
        with output.open('x') as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write('\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'evidence_valid', 'heldout_accuracy_gate',
                                            'heldout_cost_gate', 'accepted_sampler', 'failed_objects')}), flush=True)


if __name__ == '__main__':
    main()
