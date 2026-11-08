"""Frozen v7 held-out workflow; launch every stage through Slurm.

The five groups retain four observations each. Warmup is part of the fixed
algorithm, with no outcome-driven adjustment. A failed group rejects the
whole panel; neither seeds nor objects are replaced.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


BOOTSTRAP = ('import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];'
             'import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent'
             '==pathlib.Path.cwd();runpy.run_path(sys.argv.pop(1),run_name="__main__")')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def timed(root, tag, script, arguments):
    """Preserve failures and include process/model loading in measured cost."""
    receipt = root/(tag+'_timing.json')
    if receipt.exists():
        raise FileExistsError(receipt)
    command = [sys.executable, '-u', '-c', BOOTSTRAP, script, *map(str, arguments)]
    started = time.monotonic()
    with (root/(tag+'.out')).open('x') as output:
        process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
    result = dict(command=command, returncode=process.returncode,
                  wall_seconds=time.monotonic()-started,
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    save(receipt, result)
    print('HELDOUT_STEP', tag, result, flush=True)
    if process.returncode:
        raise RuntimeError(f'{tag} failed; see preserved output and timing receipt')
    return result['wall_seconds']


def run_population(root, declaration, group, replica, kind):
    settings = declaration['pilot' if kind == 'pilot' else kind+'_settings']
    seed = settings['seed_base']+10*group+replica
    output = root/f'{kind}_{replica}'
    args = ['--reference', declaration['reference'], '--output', output,
            '--observation-indices', *declaration['groups'][group],
            '--particles', settings['particles'], '--stages', settings['stages'],
            '--moves', settings['moves'], '--seed', seed,
            '--density-batch-size', declaration['density_batch_size'],
            '--shear', *declaration['shear']]
    if kind != 'pilot':
        args += ['--physical-geometry', root/'geometry.json']
    if kind == 'candidate':
        args += ['--initial-proposal', root/'initial_proposal.json']
    if kind == 'reference':
        args += ['--proposal-family', 'student_t', '--proposal-df', 5]
    return timed(root, f'{kind}_{replica}', 'scripts/check_galsbi_particles.py', args)


def validate_report(report, declaration, group, kind):
    settings = declaration[kind+'_settings']
    expected_seeds = [settings['seed_base']+10*group+i for i in range(settings['replicas'])]
    if (report['seeds'] != expected_seeds
            or [o['observation'] for o in report['observations']] != declaration['groups'][group]
            or report['diagnostics_version'] != declaration['comparison_report_version']
            or report['reference_protocol_sha256'] != declaration['reference_protocol_sha256']
            or report['observation_sha256'] != declaration['observation_sha256']
            or report['density_batch_size'] != declaration['density_batch_size']
            or report['centre'] != declaration['shear']
            or any(report[k] != settings[k] for k in ('particles', 'stages', 'moves', 'replicas'))):
        raise ValueError('held-out report does not match the frozen declaration')


def aggregate_gate(groups):
    """Panel gate is conjunctive; speed never compensates for failed accuracy."""
    if len(groups) != 5 or [g['group'] for g in groups] != list(range(5)):
        raise ValueError('all five ordered groups required')
    candidate = sum(g['candidate_all_inclusive_seconds'] for g in groups)
    reference = sum(g['reference_all_inclusive_seconds'] for g in groups)
    if any(not math.isfinite(v) or v <= 0 for v in (candidate, reference)):
        raise ValueError('finite positive complete cost receipts required')
    accuracy = all(g['candidate_precision'] and g['reference_precision'] and g['comparison'] for g in groups)
    cost = candidate < reference
    return dict(heldout_accuracy_gate=accuracy, heldout_cost_gate=cost,
                heldout_gate=accuracy and cost, accepted_sampler=accuracy and cost,
                candidate_all_inclusive_seconds=candidate,
                reference_all_inclusive_seconds=reference,
                reference_to_candidate_cost_ratio=reference/candidate)


def check_runtime(run):
    protocol = json.loads((run/'protocol.json').read_text())
    runtime = protocol['runtime']
    if (runtime['cuda_device_name'] != 'Tesla V100-SXM2-32GB'
            or runtime['slurm_node_list'] != 'th-cl-nv02'
            or runtime['torch_num_threads'] != 2):
        raise ValueError('hardware or thread count differs from declared comparison')


def report_panel(root, declaration):
    groups = []
    for group in range(5):
        directory = root/f'group_{group}'
        elapsed_reports = 0.
        for kind in ('reference', 'candidate'):
            args = ['--runs', *[directory/f'{kind}_{i}' for i in range(8)],
                    '--output', directory/f'{kind}_report.json']
            if kind == 'candidate':
                args += ['--comparison', directory/'reference_report.json']
            elapsed_reports += timed(directory, f'{kind}_report', 'scripts/report_galsbi_particles.py', args)
        reports = {k:json.loads((directory/f'{k}_report.json').read_text()) for k in ('candidate', 'reference')}
        for kind, value in reports.items():
            validate_report(value, declaration, group, kind)
        costs = {}
        for kind, count in [('pilot', 4), ('candidate', 8), ('reference', 8)]:
            costs[kind] = 0.
            for i in range(count):
                check_runtime(directory/f'{kind}_{i}')
                receipt = json.loads((directory/f'{kind}_{i}_timing.json').read_text())
                if receipt['returncode'] != 0:
                    raise ValueError('failed population cannot be omitted from cost')
                costs[kind] += receipt['wall_seconds']
        for fit in ('geometry_fit', 'proposal_fit'):
            receipt = json.loads((directory/f'{fit}_timing.json').read_text())
            if receipt['returncode'] != 0:
                raise ValueError('failed pilot fit')
            costs[fit] = receipt['wall_seconds']
        shared = costs['pilot']+costs['geometry_fit']+elapsed_reports
        groups.append(dict(group=group, observations=declaration['groups'][group],
            candidate_precision=reports['candidate']['all_precision_gates'],
            reference_precision=reports['reference']['all_precision_gates'],
            comparison=reports['candidate']['comparison_gate'], costs=costs,
            report_seconds_charged_to_each=elapsed_reports,
            candidate_all_inclusive_seconds=shared+costs['proposal_fit']+costs['candidate'],
            reference_all_inclusive_seconds=shared+costs['reference'],
            report_sha256={k:digest(directory/f'{k}_report.json') for k in reports}))
    result = dict(schema='sbsi-heldout-v7-panel-result-v1', groups=groups,
                  **aggregate_gate(groups), declaration_sha256=digest(root/'declaration.json'),
                  script_sha256=digest(__file__),
                  scope=declaration['scope'], search_and_development_cost_included=False)
    save(root/'result.json', result)
    print('HELDOUT_PANEL', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['warmup', 'production', 'report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    declaration = json.loads((args.root/'declaration.json').read_text())
    if digest(declaration['panel']) != declaration['panel_sha256']:
        raise ValueError('frozen panel changed')
    if args.stage == 'report':
        report_panel(args.root, declaration)
        return
    import torch
    if torch.cuda.get_device_name(0) != 'Tesla V100-SXM2-32GB':
        raise ValueError('declared V100 hardware required')
    if args.index is None or not 0 <= args.index < (5 if args.stage == 'warmup' else 40):
        raise ValueError('valid array index required')
    group = args.index if args.stage == 'warmup' else args.index//8
    directory = args.root/f'group_{group}'
    if args.stage == 'warmup':
        directory.mkdir()
        for replica in range(4):
            run_population(directory, declaration, group, replica, 'pilot')
        runs = [directory/f'pilot_{i}' for i in range(4)]
        timed(directory, 'geometry_fit', 'scripts/fit_particle_gradient_geometry.py',
              ['--runs', *runs, '--output', directory/'geometry.json', '--device', 'cuda'])
        timed(directory, 'proposal_fit', 'scripts/fit_particle_initial_proposal.py',
              ['--runs', *runs, '--output', directory/'initial_proposal.json'])
    else:
        for kind in ('candidate', 'reference'):
            run_population(directory, declaration, group, args.index % 8, kind)


if __name__ == '__main__':
    main()
