"""Frozen downstream diagnostics for a training-selected per-leg checkpoint.

No checkpoint selection or model modification occurs here. Step zero is an
explicit skip, since its numerical model has already been evaluated.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

DATA = Path('/project/ls-gruen/users/zekang.zhang')
CACHE = DATA/'sbsi_caches'
ROOT = CACHE/'flow_shape_augmentation_20260918_v1'
HARMONIZED = CACHE/'harmonized_secondary_20260917_v1'
CANDIDATE = ROOT/'per_leg_refinement_full'
FRESH = CACHE/'fresh_fullscene_joint_20260918_v1'
STAGED = CACHE/'joint_multiaxis_evaluator_20260918_v1'
INITIAL = HARMONIZED/'refinement/selected.pt'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate_selection(record, protocol):
    """Reject malformed or ineligible training results before diagnostic execution."""
    if record['smoke'] is not False or protocol['smoke'] is not False:
        raise ValueError('full refinement required')
    step = record['best_step']
    if type(step) is not int or not 0 <= step <= protocol['steps']:
        raise ValueError('invalid selected step')
    if record['eligible_for_joint_validation'] is not (step > 0):
        raise ValueError('inconsistent eligibility')
    nll, baseline = record['selected_full_validation_nll'], record['baseline_full_validation_nll']
    loss, initial_loss = record['selected_guard_loss'], record['baseline_guard_loss']
    tolerance = protocol['density_tolerance_nats_per_row']
    if (tolerance != .001 or not all(map(math.isfinite, (nll, baseline, loss, initial_loss)))
            or nll > baseline+tolerance or loss > initial_loss
            or (step > 0 and loss >= initial_loss)):
        raise ValueError('density or response selection constraint failed')
    selected, initial = record['selected_selection_mass'], record['baseline_selection_mass']
    for result in (selected, initial):
        if result['coordinates'] != ['soft_primary', 'hard_primary'] or len(result['residual']) != 2:
            raise ValueError('both primary probability gates required')
    for result, origin in zip(selected['residual'], initial['residual']):
        if not math.isfinite(result) or not math.isfinite(origin) or abs(result) > abs(origin)+1e-12:
            raise ValueError('primary selection probability degraded')
    return step > 0


def run(module, *args):
    command = [sys.executable, '-u', '-m', module, *map(str, args)]
    print('VALIDATION_COMMAND', json.dumps(command), flush=True)
    subprocess.run(command, check=True)


def common():
    args = ['--flow', CANDIDATE/'selected.pt', '--response-candidate',
            DATA/'blendemu_runs/disk_response_moment_20260918_v1/model',
            '--response-composition', 'mobius', '--response-device', 'cuda',
            '--response-threads', 4, '--cohorts', 'truth_parent',
            '--seed', 7301, '--draws', 64, '--batch-size', 1024]
    for seed in (20260913, 20260914, 20260915):
        args += ['--classifier', CACHE/f'fixed_g0_m258_r060_v1/classifier/seed{seed}/selected.pt']
    return args


def compare(baseline, candidate, prefix, *, density=True):
    run('scripts.compare_truth_parent_response', '--baseline', baseline,
        '--candidate', candidate, '--output', CANDIDATE/f'{prefix}_response.json')
    if density:
        run('scripts.compare_joint_predictive_results', '--baseline', baseline,
            '--candidate', candidate, '--output', CANDIDATE/f'{prefix}_density.json')


def existing(endpoint):
    if endpoint == 'main':
        cases, h = list(range(40, 120)), .02
        parent = CACHE/'constgold_truth_parent_20260917_v1/parent_v2'
        catalog = DATA/'lsst_sims_fs2_25876_constant'
        pairs = CACHE/'output_conditioned_response_20260917_v1/pairs'
        lookups = [CACHE/f'constgold_truth_parent_20260917_v1/rblend/cases{i}_{i+9}.feather'
                   for i in range(40, 120, 10)]
        baseline = ROOT/'initial_main_truth_parent.json'
    else:
        cases, h = list(range(200, 240)), .05
        parent = HARMONIZED/'independent_parent'
        catalog = DATA/'lsst_sims_fs2_25876_constant_g005_c140-239'
        pairs = HARMONIZED/'independent_pairs'
        lookups = [HARMONIZED/f'independent_pair_lookups/cases{i}_{i+9}.feather'
                   for i in range(200, 240, 10)]
        baseline = HARMONIZED/'independent_candidate_full.json'
    args = common()+['--parent-root', parent, '--constgold-root', catalog,
                     '--pair-root', pairs, '--h', h, '--predictive-diagnostics']
    for path in lookups:
        args += ['--rblend', path]
    for scope, selected in [('smoke', cases[:2]), ('full', cases)]:
        output = CANDIDATE/f'joint_{endpoint}_{scope}.json'
        run('scripts.evaluate_constgold_truth_parent', *args, '--cases', *selected, '--output', output)
        result = read(output)
        if (result['cases'] != selected or result['h'] != h
                or result['inputs']['flow_sha256'] != digest(CANDIDATE/'selected.pt')
                or result.get('predictive_diagnostics') is None):
            raise ValueError('joint evaluator output mismatch')
    compare(baseline, output, f'{endpoint}_vs_initial', density=endpoint == 'independent')
    if endpoint == 'main':
        for scope, selected in [('smoke', cases[:2]), ('full', cases)]:
            run('scripts.audit_disk_candidate_density', '--flow', CANDIDATE/'selected.pt',
                '--reference-flow', INITIAL, '--device', 'cuda', '--shears', .02, -.02,
                '--cases', *selected, '--output', CANDIDATE/f'transported_density_{scope}.json')


def fresh(axis):
    prepared = FRESH/'full'
    receipt = read(prepared/'preparation.json')
    cases = [20010, 20011, 20012, 20013]
    if (receipt['status'] != 'completed' or receipt['operational_small_scene'] is not False
            or receipt['parent_manifest_sha256'] != digest(prepared/'parent/manifest.json')
            or receipt['group_cases'][f'g{axis}_h0.02'] != cases
            or receipt['group_cases'][f'g{axis}_h0.04'] != [20010]):
        raise ValueError('full prepared fresh population required')
    args = common()+['--parent-root', prepared/'parent', '--pair-root', prepared/'pairs',
                     '--rblend', prepared/'pair_lookup.feather', '--axis', axis, '--device', 'cuda']
    for scope, h, selected, options in [('smoke', .02, cases[:2], ['--predictive-diagnostics']),
                                      ('full', .02, cases, ['--predictive-diagnostics']),
                                      ('amplitude', .04, [20010], ['--operational-single-case'])]:
        output = CANDIDATE/f'fresh_g{axis}_{scope}.json'
        command = [sys.executable, '-u', '-c',
                   'import runpy,sys; runpy.run_path(sys.argv.pop(1),run_name="__main__")',
                   STAGED/'evaluate_constgold_truth_parent.py', *args,
                   '--constgold-root', prepared/f'catalogues/g{axis}_h{h}',
                   '--h', h, '--cases', *selected, '--output', output, *options]
        subprocess.run(list(map(str, command)), check=True)
        result = read(output)
        if (result['cases'] != selected or result['shear_axis'] != axis or result['h'] != h
                or result['inputs']['flow_sha256'] != digest(CANDIDATE/'selected.pt')):
            raise ValueError('fresh evaluator output mismatch')
    compare(prepared/f'results/initial_g{axis}_h0.02.json',
            CANDIDATE/f'fresh_g{axis}_full.json', f'fresh_g{axis}_vs_initial')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('endpoint', choices=('main', 'independent', 'fresh1', 'fresh2'))
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('diagnostics require Slurm')
    frozen = read(args.protocol)
    for path, expected in frozen['file_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'frozen validation input changed: {path}')
    record, protocol = read(CANDIDATE/'completed.json'), read(CANDIDATE/'protocol.json')
    if (record['protocol_sha256'] != digest(CANDIDATE/'protocol.json')
            or record['selected_sha256'] != digest(CANDIDATE/'selected.pt')
            or protocol['initial_flow_sha256'] != digest(INITIAL)
            or protocol['script_sha256'] != digest('scripts/refine_per_leg_guard_flow.py')):
        raise ValueError('refinement identity mismatch')
    eligible = validate_selection(record, protocol)
    status = CANDIDATE/f'validation_{args.endpoint}_status.json'
    if status.exists():
        raise FileExistsError(status)
    report = {'endpoint': args.endpoint, 'selected_sha256': record['selected_sha256'],
              'refinement_completed_sha256': digest(CANDIDATE/'completed.json'),
              'validation_protocol_sha256': digest(args.protocol), 'calibration_claim': False}
    if eligible:
        print('PER_LEG_JOINT_START', args.endpoint, record['best_step'], flush=True)
        if args.endpoint.startswith('fresh'):
            fresh(int(args.endpoint[-1]))
        else:
            existing(args.endpoint)
        report['status'] = 'completed'
    else:
        report.update(status='skipped', reason='step zero retained; unchanged model already evaluated')
    status.write_text(json.dumps(report, indent=2)+'\n')
    print('PER_LEG_JOINT_END', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
