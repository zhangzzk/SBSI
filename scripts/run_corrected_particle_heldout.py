#!/usr/bin/env python
"""Execute the declared corrected heldout procedure, only inside Slurm.

No submission, retry, population replacement, adaptive fit or source change is
performed here. A successful frozen development receipt is a prerequisite.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import sbsi
from sbsi.phase_control import FrozenPhaseControl, digest, load_phase_diagnostic
from sbsi.particle_initial_proposal import FrozenParticleProposal
from sbsi.particle_comparison_gates import corrected_phase_gate, difference_gate, raw_pair_gate
from fit_declared_particle_phase_control import (
    DIAGNOSTIC_SOURCES, NATIVE_SOURCES, PROTOCOL_SOURCES, verified_manifest)
from report_galsbi_particles import REPORT_VERSION, comparison_target_identity, validate_completed_result


BOOTSTRAP = ('import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];'
             'import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent'
             '==pathlib.Path.cwd();runpy.run_path(sys.argv.pop(1),run_name="__main__")')
BLOCKS = ['physical', 'physical', 'nuisance', 'physical', 'environment', 'physical',
          'sed', 'physical', 'positions', 'physical', 'exchange', 'physical',
          'nuisance', 'physical', 'independence', 'physical']
MEASURES = ('raw_float32_sampled', 'precision_reweighted_float64')
BRIDGE = 'q_environment(E) * q0(x)**(1-beta) * (p_analytic(x)*K(y|x,E))**beta'
ENDPOINT = 'q_environment(E) * p_analytic(x) * K(y|x,E)'
SELECTION = 'none in fixed-y posterior; no population shear fit'
PHASE_PROTOCOL = 'phase-fine-difference-v2'
SCRIPTS = {'run': 'scripts/check_galsbi_particles.py',
           'phase': 'scripts/check_particle_phase_control.py',
           'geometry_fit': 'scripts/fit_particle_gradient_geometry.py',
           'proposal_fit': 'scripts/fit_particle_initial_proposal.py',
           'coefficient_fit': 'scripts/fit_declared_particle_phase_control.py',
           'raw_report': 'scripts/report_galsbi_particles.py',
           'phase_report': 'scripts/report_particle_phase_control.py'}
REQUIRED_SOURCE = PROTOCOL_SOURCES | DIAGNOSTIC_SOURCES | set(SCRIPTS.values()) | {
    'scripts/run_corrected_particle_heldout.py', 'scripts/check_galsbi_continuous_prior.py',
    'sbsi/galsbi_prior.py', 'sbsi/galsbi_scene_prior.py', 'sbsi/particle_geometry.py',
    'sbsi/particle_initial_proposal.py', 'sbsi/particle_gradient_geometry.py',
    'sbsi/particle_comparison_gates.py', 'sbsi/particle_diagnostics.py'}
COST_KEYS = ('process_wall_seconds', 'cpu_allocated_seconds', 'gpu_allocated_seconds')


def read(path):
    def bad(value):
        raise ValueError(f'nonfinite JSON: {value}')
    return json.loads(Path(path).read_text(), parse_constant=bad)


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def check_hash_map(mapping, *, required=()):
    if not isinstance(mapping, dict) or not mapping or not set(required) <= set(mapping):
        raise ValueError('complete nonempty source/artifact hash map required')
    for path, checksum in mapping.items():
        if (not Path(path).is_absolute() or str(Path(path).resolve()) != path
                or not isinstance(checksum, str) or len(checksum) != 64
                or digest(path) != checksum):
            raise ValueError(f'changed or noncanonical frozen input: {path}')


def bound_json(record):
    if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
        raise ValueError('exact path/hash receipt required')
    check_hash_map({record['path']: record['sha256']})
    return read(record['path'])


def validate_cost(cost, *, positive=False):
    values = [cost[k] for k in COST_KEYS]
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
            or (positive and values[0] <= 0)):
        raise ValueError('complete finite nonnegative wall/allocation cost required')
    return {k: float(cost[k]) for k in COST_KEYS}


def zero_cost():
    return dict.fromkeys(COST_KEYS, 0.)


def add_cost(*costs):
    return {k: sum(validate_cost(c)[k] for c in costs) for k in COST_KEYS}


def scale_cost(cost, scale):
    return {k: scale*v for k, v in validate_cost(cost).items()}


def measured_cost(seconds, resources):
    return dict(process_wall_seconds=seconds,
                cpu_allocated_seconds=seconds*resources['cpus'],
                gpu_allocated_seconds=seconds*resources['gpus'])


def settings(declaration, kind):
    return declaration['pilot' if kind == 'pilot' else kind+'_settings']


def run_seed(declaration, group, replica, kind):
    return settings(declaration, kind)['seed_base']+10*group+replica


def validate_declaration(path, *, check_execution=True):
    """Validate the frozen inputs before inspecting any population outputs."""
    path = Path(path).resolve()
    d = read(path)
    if d.get('schema') != 'sbsi-heldout-corrected-declaration-v1':
        raise ValueError('corrected heldout declaration required')
    root, source = Path(d['output']).resolve(), Path(d['source']).resolve()
    if path != root/'declaration.json':
        raise ValueError('declaration must be frozen in its declared output directory')
    if check_execution and (Path(__file__).resolve() != source/'scripts/run_corrected_particle_heldout.py'
                            or Path(sbsi.__file__).resolve() != source/'sbsi/__init__.py'):
        raise ValueError('runner and imported package must use declared frozen source')
    entries = verified_manifest(source, d['source_manifest_sha256'])
    if not REQUIRED_SOURCE <= set(entries):
        raise ValueError('complete workflow and target sources required in manifest')
    expected_sources = {str(source/name): checksum for name, checksum in entries.items()}
    if d['source_files'] != expected_sources:
        raise ValueError('source_files must bind the complete frozen manifest')
    # verified_manifest has checked every source byte; external files are separate.
    check_hash_map(d['external_files'])
    development = bound_json(d['development_receipt'])
    comparisons = development.get('comparisons', {})
    if (development.get('schema') != 'sbsi-n23-corrected-comparison-v1'
            or development.get('development_checks_pass') is not True
            or development.get('accepted_sampler') is not False
            or set(comparisons) != {'small_minus_large', 'small_minus_broad', 'large_minus_broad'}
            or any(c.get('all_raw_gates') is not True or c.get('all_corrected_phase_gates') is not True
                   for c in comparisons.values())):
        raise ValueError('completed passing development gate required before heldout work')
    check_hash_map(development['sources'])
    preparation = bound_json(d['preparation_cost_receipt'])
    if preparation.get('kind') != 'scheduler_allocation_wall_upper_bound':
        raise ValueError('scheduler preflight allocation cost receipt required')
    validate_cost(preparation, positive=True)
    check_hash_map({d['panel']: d['panel_sha256']})
    panel = read(d['panel'])
    groups = d['groups']
    if (len(groups) != 5 or any(len(g) != 4 for g in groups)
            or groups != [band['observations'] for band in panel['bands']]):
        raise ValueError('unchanged five ordered panel groups required')
    flat = [oid for g in groups for oid in g]
    if (any(type(x) is not int or x < 0 for x in flat) or len(set(flat)) != 20
            or set(flat) & set(d['development_ids'])):
        raise ValueError('distinct heldout IDs disjoint from development required')
    if d['shear'] != [0., 0.] or d['density_batch_size'] != 4096:
        raise ValueError('declared zero-shear, four-observation batch4096 procedure required')
    seeds = []
    for kind, particles, moves, replicas in [('pilot', 256, 32, 4),
                                            ('candidate', 256, 64, 8), ('reference', 1024, 64, 8)]:
        s = settings(d, kind)
        if (any(type(s[k]) is not int for k in ('particles', 'stages', 'moves', 'replicas', 'seed_base'))
                or (s['particles'], s['stages'], s['moves'], s['replicas']) != (particles, 64, moves, replicas)
                or s['seed_base'] < 0 or s['power'] != 2. or s['scale'] != 1. or s['blocks'] != BLOCKS):
            raise ValueError('fixed corrected heldout pilot/production recipe required')
        seeds.extend(run_seed(d, g, i, kind) for g in range(5) for i in range(replicas))
    if len(set(seeds)) != len(seeds):
        raise ValueError('globally disjoint pilot/candidate/reference seeds required')
    for key, family in [('pilot_proposal_metadata', 'gaussian'), ('reference_proposal_metadata', 'student_t')]:
        laws = d[key]
        if len(laws) != 5 or any(len(g) != 4 for g in laws):
            raise ValueError('complete per-group measured-input proposal laws required')
        for proposal in (p for g in laws for p in g):
            if (proposal['component_family'] != family or proposal['faint_magnitude_shift'] != 5.
                    or proposal['weights'] != [.5, .5] or proposal['phase'] != 'independent uniform periodic'
                    or proposal['uses_generating_truth'] is not False or proposal['retry_or_clip_draws'] is not False
                    or (family == 'student_t' and proposal['student_df'] != 5.)):
                raise ValueError('declared original Gaussian/broad Student-t5 laws required')
    reference = Path(d['reference']).resolve()
    required_reference = {str(reference/'protocol.json'): d['reference_protocol_sha256'],
                          str(reference/'observations.parquet'): d['observation_sha256']}
    check_hash_map(required_reference)
    if (str(reference/'result.json') not in d['external_files']
            or any(d['external_files'].get(p) != h for p, h in required_reference.items())):
        raise ValueError('reference files must be included in frozen external inputs')
    r = d['runtime']
    if (r['cuda_device_name'] != 'Tesla V100-SXM2-32GB' or r['slurm_node_list'] != 'th-cl-nv02'
            or r['torch_num_threads'] != 2 or any(not isinstance(r[k], str) or not r[k]
                                                for k in ('python', 'numpy', 'torch'))):
        raise ValueError('original V100 runtime required')
    return d, entries


def validate_runtime(d, *, gpu):
    import torch
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('all heldout workflow stages require a Slurm allocation')
    actual = dict(python=sys.version.split()[0], numpy=np.__version__, torch=str(torch.__version__))
    if any(d['runtime'][k] != v for k, v in actual.items()):
        raise ValueError('runtime versions differ from declaration')
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    if cpus < (2 if gpu else 1) or (gpu and len(os.sched_getaffinity(0)) < 2):
        raise ValueError('declared CPU allocation required')
    if gpu:
        if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != d['runtime']['cuda_device_name']
                or os.environ.get('SLURM_JOB_NODELIST') != d['runtime']['slurm_node_list']):
            raise ValueError('one declared V100 on the original node required')
        gpus = 1
    else:
        # Report jobs are CPU-only; accidental GPU reservation must not be hidden.
        if os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_GPUS_ON_NODE', '0') != '0':
            raise ValueError('report stage must use a CPU-only allocation')
        gpus = 0
    return dict(cpus=cpus, gpus=gpus)


def check_implementation(record, source, entries, required):
    recorded = {}
    for path, checksum in record['implementation_sha256'].items():
        relative = Path(path).resolve().relative_to(source).as_posix()
        if relative in recorded or entries.get(relative) != checksum:
            raise ValueError('run/diagnostic implementation differs from frozen source')
        recorded[relative] = checksum
    if set(recorded) != set(required):
        raise ValueError('exact declared run/diagnostic implementation set required')


def run_paths(d, group, replica, kind):
    directory = Path(d['output'])/f'group_{group}'
    return directory/f'{kind}_{replica}', directory/f'{kind}_{replica}_phase.json'


def expected_proposals(d, group, kind):
    if kind == 'candidate':
        return [p.metadata for p in FrozenParticleProposal.load(
            Path(d['output'])/f'group_{group}/initial_proposal.json').proposals]
    return d['pilot_proposal_metadata' if kind == 'pilot' else 'reference_proposal_metadata'][group]


def validate_run(d, entries, group, replica, kind):
    run, _ = run_paths(d, group, replica, kind)
    p, result = read(run/'protocol.json'), read(run/'result.json')
    s = settings(d, kind)
    expected = dict(seed=run_seed(d, group, replica, kind), observations=d['groups'][group],
        particles=s['particles'], stages=s['stages'], moves=s['moves'], blocks=s['blocks'], scale=s['scale'],
        betas=(np.linspace(0., 1., s['stages']+1)**s['power']).tolist(), fixed_shear=d['shear'],
        density_batch_size=d['density_batch_size'], reference=str(Path(d['reference']).resolve()),
        reference_protocol_sha256=d['reference_protocol_sha256'], observation_sha256=d['observation_sha256'],
        selection=SELECTION, bridge=BRIDGE, endpoint_target=ENDPOINT,
        proposals=expected_proposals(d, group, kind))
    if any(p.get(k) != v for k, v in expected.items()) or p.get('informed_environment') is not None:
        raise ValueError('population differs from complete declared recipe')
    runtime = p['runtime']
    if runtime.get('device') != 'cuda' or any(runtime.get(k) != v for k, v in d['runtime'].items()):
        raise ValueError('population runtime differs from declared V100 runtime')
    required = set(PROTOCOL_SOURCES)
    directory = run.parent
    for field, filename, needed, module in [
            ('physical_geometry', 'geometry.json', kind != 'pilot', 'sbsi/particle_geometry.py'),
            ('initial_proposal', 'initial_proposal.json', kind == 'candidate', 'sbsi/particle_initial_proposal.py')]:
        artifact = p.get(field)
        if needed:
            payload = read(directory/filename)
            if (not artifact or Path(artifact['path']).resolve() != directory/filename
                    or artifact['sha256'] != digest(directory/filename) or artifact['metadata'] != payload['metadata']):
                raise ValueError('population frozen fit artifact differs')
            required.add(module)
        elif artifact is not None:
            raise ValueError('unexpected pilot/reference fitted artifact')
    check_implementation(p, Path(d['source']), entries, required)
    if result.get('status') != 'completed':
        raise ValueError('extinction or unknown failure rejects the complete heldout panel')
    validate_completed_result(p, result)
    for field, name in [('protocol', 'protocol.json'), ('particles', 'particles.npz'), ('scenes', 'scenes.npz')]:
        if result.get(field+'_sha256') != digest(run/name):
            raise ValueError('changed population artifact')
    return p, result


def validate_phase(d, entries, group, replica, kind):
    p, result = validate_run(d, entries, group, replica, kind)
    run, path = run_paths(d, group, replica, kind)
    report, arrays = load_phase_diagnostic(path)
    source_map = {str(run/name): digest(run/name) for name in ('protocol.json', 'result.json', 'particles.npz')}
    if (report['source_hashes'] != source_map or Path(report['run']).resolve() != run
            or report['seed'] != p['seed'] or report['observations'] != p['observations']
            or report['retained_particles'] != p['particles'] or report['log_normalizer'] != result['log_normalizer']
            or report['device'] != 'cuda' or report['device_name'] != d['runtime']['cuda_device_name']
            or report['derivative_protocol'] != PHASE_PROTOCOL):
        raise ValueError('phase diagnostic does not bind its complete declared population')
    check_implementation(report, Path(d['source']), entries, DIAGNOSTIC_SOURCES)
    identity = report['target_identity']
    for key in ('observations', 'fixed_shear', 'reference_protocol_sha256',
                'observation_sha256', 'density_batch_size', 'selection'):
        if identity.get(key) != p[key]:
            raise ValueError('phase target identity differs from bound protocol')
    native = {Path(name).name: checksum for name, checksum in entries.items()
              if name.startswith(('sbsi/', 'configs/')) and Path(name).name in NATIVE_SOURCES}
    if identity.get('native_source_sha256') != native or identity.get('phase_derivative_protocol') != PHASE_PROTOCOL:
        raise ValueError('phase native/derivative identity differs from declaration')
    if kind == 'pilot':
        if report.get('coefficients') is not None:
            raise ValueError('pilot diagnostics must precede coefficient fitting')
    else:
        coefficient = run.parent/'phase_coefficients.json'
        recorded = report.get('coefficients', {})
        frozen = FrozenPhaseControl.load(coefficient)
        if (Path(recorded.get('path', '')).resolve() != coefficient
                or recorded.get('sha256') != digest(coefficient) or recorded.get('metadata') != frozen.metadata):
            raise ValueError('production phase coefficients differ from frozen group fit')
        frozen.validate_production(p['seed'], p['observations'], identity)
    return report, arrays


def population_arguments(d, group, replica, kind):
    s = settings(d, kind)
    run, _ = run_paths(d, group, replica, kind)
    args = ['--reference', d['reference'], '--output', str(run), '--observation-indices', *map(str, d['groups'][group]),
            '--particles', str(s['particles']), '--stages', str(s['stages']), '--moves', str(s['moves']),
            '--seed', str(run_seed(d, group, replica, kind)), '--density-batch-size', str(d['density_batch_size']),
            '--shear', *map(str, d['shear']), '--device', 'cuda', '--power', str(s['power']), '--scale', str(s['scale'])]
    if kind != 'pilot':
        args += ['--physical-geometry', str(run.parent/'geometry.json')]
    if kind == 'candidate':
        args += ['--initial-proposal', str(run.parent/'initial_proposal.json')]
    else:
        args += ['--proposal-family', 'gaussian' if kind == 'pilot' else 'student_t', '--faint-shift', '5']
        if kind == 'reference':
            args += ['--proposal-df', '5']
    return args


def phase_arguments(d, group, replica, kind):
    run, phase = run_paths(d, group, replica, kind)
    args = ['--run', str(run), '--output', str(phase), '--device', 'cuda', '--stride', '1']
    if kind != 'pilot':
        args += ['--coefficients', str(run.parent/'phase_coefficients.json')]
    return args


class Stage:
    """One immutable stage receipt; failed work and costs are retained."""
    def __init__(self, d, declaration_path, name, resources, started):
        self.d, self.declaration_path, self.name = d, Path(declaration_path), name
        self.resources, self.started, self.steps = resources, started, []
        self.path = Path(d['output'])/(name+'_stage.json')
        # Reserve the stage before any subprocess; even an interrupted stage cannot retry.
        self.reservation = Path(d['output'])/(name+'_started.json')
        save(self.reservation, dict(declaration_sha256=digest(declaration_path),
                                   slurm_job_id=os.environ.get('SLURM_JOB_ID')))

    def timed(self, directory, tag, script, arguments):
        directory = Path(directory)
        receipt, log = directory/(tag+'_timing.json'), directory/(tag+'.out')
        if receipt.exists() or log.exists():
            raise FileExistsError('no replacement or retry of a started step')
        command = [sys.executable, '-u', '-c', BOOTSTRAP, script, *map(str, arguments)]
        started = time.monotonic()
        code, error = None, None
        try:
            with log.open('x') as output:
                code = subprocess.run(command, cwd=self.d['source'], stdout=output,
                                      stderr=subprocess.STDOUT, check=False).returncode
        except BaseException as failure:
            error = type(failure).__name__+': '+str(failure)
            raise
        finally:
            elapsed = time.monotonic()-started
            row = dict(command=command, returncode=code, error=error,
                status='completed' if code == 0 else 'failed', cost=measured_cost(elapsed, self.resources),
                resources=self.resources, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                declaration_sha256=digest(self.declaration_path), output_sha256=digest(log) if log.exists() else None)
            save(receipt, row)
            self.steps.append(dict(path=str(receipt), sha256=digest(receipt), tag=tag, **row))
        if code != 0:
            raise RuntimeError(f'{tag} failed; original output and costs retained')
        print('CORRECTED_HELDOUT_STEP', tag, elapsed, flush=True)

    def finish(self, error=None):
        total = measured_cost(time.monotonic()-self.started, self.resources)
        children = add_cost(*(step['cost'] for step in self.steps))
        overhead = {k: total[k]-children[k] for k in COST_KEYS}
        validate_cost(overhead)
        row = dict(schema='sbsi-corrected-heldout-stage-v1', stage=self.name,
            status='completed' if error is None else 'failed',
            declaration_sha256=digest(self.declaration_path), resources=self.resources,
            cost=total, verification_overhead=overhead, steps=self.steps,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            cost_measure='Full wrapper process wall times allocated resources; not busy compute, queue time, or Slurm lifetime outside this process. Final receipt serialization is outside the timer.')
        if error is not None:
            row.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        save(self.path, row)
        return row


def group_artifacts(d, group):
    directory = Path(d['output'])/f'group_{group}'
    paths = [directory/name for name in ('geometry.json', 'initial_proposal.json', 'phase_coefficients.json')]
    for i in range(4):
        run, phase = run_paths(d, group, i, 'pilot')
        paths.extend([phase, phase.with_suffix('.npz')])
        paths.extend(run/name for name in ('protocol.json', 'result.json', 'particles.npz', 'scenes.npz', 'progress.json'))
    return {str(path): digest(path) for path in paths}


def validate_warmup(d, declaration_path, entries, group):
    directory = Path(d['output'])/f'group_{group}'
    receipt = read(directory/'warmup_complete.json')
    if (receipt.get('status') != 'completed' or receipt['declaration_sha256'] != digest(declaration_path)
            or receipt['group'] != group or receipt['artifacts'] != group_artifacts(d, group)):
        raise ValueError('complete unchanged frozen warmup required before production')
    stage = read(Path(d['output'])/f'warmup_{group}_stage.json')
    validate_stage_receipt(stage, d, declaration_path, f'warmup_{group}')
    if {s['tag'] for s in stage['steps']} != ({f'pilot_{i}' for i in range(4)} |
            {f'pilot_{i}_phase' for i in range(4)} | {'geometry_fit', 'proposal_fit', 'coefficient_fit'}):
        raise ValueError('complete warmup step receipts required')
    for i in range(4):
        validate_phase(d, entries, group, i, 'pilot')
    coefficients = FrozenPhaseControl.load(directory/'phase_coefficients.json')
    metadata = coefficients.metadata
    if (metadata['pilot_seeds'] != [run_seed(d, group, i, 'pilot') for i in range(4)]
            or metadata['declaration'] != dict(path=str(Path(declaration_path).resolve()), sha256=digest(declaration_path))
            or metadata['source_manifest_sha256'] != d['source_manifest_sha256']):
        raise ValueError('coefficient fit differs from declared independent pilots')
    for name in ('geometry.json', 'initial_proposal.json'):
        m = read(directory/name)['metadata']
        if m['pilot_seeds'] != metadata['pilot_seeds'] or m['observations'] != d['groups'][group]:
            raise ValueError('geometry/proposal fits must use the same declared independent pilots')


def warmup(stage, d, entries, group):
    directory = Path(d['output'])/f'group_{group}'
    directory.mkdir()
    for i in range(4):
        stage.timed(directory, f'pilot_{i}', SCRIPTS['run'], population_arguments(d, group, i, 'pilot'))
        validate_run(d, entries, group, i, 'pilot')
    runs = [str(directory/f'pilot_{i}') for i in range(4)]
    stage.timed(directory, 'geometry_fit', SCRIPTS['geometry_fit'],
                ['--runs', *runs, '--output', str(directory/'geometry.json'), '--device', 'cuda', '--batch-size', '4096'])
    stage.timed(directory, 'proposal_fit', SCRIPTS['proposal_fit'],
                ['--runs', *runs, '--output', str(directory/'initial_proposal.json')])
    for i in range(4):
        stage.timed(directory, f'pilot_{i}_phase', SCRIPTS['phase'], phase_arguments(d, group, i, 'pilot'))
        validate_phase(d, entries, group, i, 'pilot')
    stage.timed(directory, 'coefficient_fit', SCRIPTS['coefficient_fit'],
        ['--declaration', str(stage.declaration_path), '--group', str(group),
         '--output', str(directory/'phase_coefficients.json')])
    save(directory/'warmup_complete.json', dict(status='completed', group=group,
        declaration_sha256=digest(stage.declaration_path), artifacts=group_artifacts(d, group)))


def production_order(replica):
    return ('candidate', 'reference') if replica % 2 == 0 else ('reference', 'candidate')


def production(stage, d, entries, index):
    group, replica = divmod(index, 8)
    validate_warmup(d, stage.declaration_path, entries, group)
    directory = Path(d['output'])/f'group_{group}'
    for kind in production_order(replica):
        stage.timed(directory, f'{kind}_{replica}', SCRIPTS['run'], population_arguments(d, group, replica, kind))
        validate_run(d, entries, group, replica, kind)
        stage.timed(directory, f'{kind}_{replica}_phase', SCRIPTS['phase'], phase_arguments(d, group, replica, kind))
        validate_phase(d, entries, group, replica, kind)


def validate_stage_receipt(row, d, declaration_path, name):
    if (row.get('schema') != 'sbsi-corrected-heldout-stage-v1' or row.get('status') != 'completed'
            or row.get('stage') != name or row.get('declaration_sha256') != digest(declaration_path)):
        raise ValueError('all declared stages must have completed without retries')
    validate_cost(row['cost'], positive=True)
    validate_cost(row['verification_overhead'])
    for step in row['steps']:
        stored = bound_json({'path': step['path'], 'sha256': step['sha256']})
        if (stored.get('status') != 'completed' or stored.get('returncode') != 0
                or stored['declaration_sha256'] != row['declaration_sha256']
                or any(stored[k] != step[k] for k in stored)):
            raise ValueError('failed, changed or missing step cost receipt')
        validate_cost(step['cost'], positive=True)
    total = add_cost(row['verification_overhead'], *(s['cost'] for s in row['steps']))
    if any(not math.isclose(total[k], row['cost'][k], rel_tol=1e-12, abs_tol=1e-9) for k in COST_KEYS):
        raise ValueError('stage costs do not account for every step and verification')


def validate_report(d, group, kind, raw, phase, directory):
    s, seeds = settings(d, kind), [run_seed(d, group, i, kind) for i in range(8)]
    observations = d['groups'][group]
    if (raw['seeds'] != seeds or phase['seeds'] != seeds
            or [o['observation'] for o in raw['observations']] != observations
            or [o['observation'] for o in phase['observations']] != observations
            or raw['diagnostics_version'] != REPORT_VERSION
            or phase['schema'] != 'sbsi-independent-phase-diagnostics-v2'
            or any(raw[k] != s[k] for k in ('particles', 'stages', 'moves', 'replicas'))
            or phase['moment_columns'] != ['T', 'raw_g1', 'raw_g2', 'corrected_g1', 'corrected_g2']):
        raise ValueError('complete eight-population raw/phase report required')
    expected = {str(directory/f'{kind}_{i}_phase.json'): digest(directory/f'{kind}_{i}_phase.json') for i in range(8)}
    if phase['sources'] != expected or phase['coefficients']['sha256'] != digest(directory/'phase_coefficients.json'):
        raise ValueError('phase report sources or frozen coefficients changed')
    for row in phase['observations']:
        for measure in MEASURES:
            value = row[measure]
            if (value['independent_populations'] != 8 or value['extinct_populations'] != 0
                    or len(value['weights']) != 8 or len(value['mean']) != 5):
                raise ValueError('complete population denominator required in both phase measures')
    for key in ('reference_protocol_sha256', 'observation_sha256', 'density_batch_size'):
        if raw[key] != d[key]:
            raise ValueError('raw report reference identity differs')
    if raw['centre'] != d['shear'] or raw['endpoint_target'] != ENDPOINT or raw['bridge'] != BRIDGE:
        raise ValueError('raw report target differs')


def compare_group(candidate, reference, candidate_phase, reference_phase):
    comparison_target_identity(reference, candidate)
    if any(len(r['observations']) != 4 for r in (candidate, reference, candidate_phase, reference_phase)):
        raise ValueError('exactly four complete report observations required')
    if (candidate_phase['target_identity'] != reference_phase['target_identity']
            or candidate_phase['coefficients'] != reference_phase['coefficients']
            or set(candidate['seeds']) & set(reference['seeds'])):
        raise ValueError('phase targets, coefficients or independent seeds differ')
    rows = []
    for a, b, pa, pb in zip(candidate['observations'], reference['observations'],
                           candidate_phase['observations'], reference_phase['observations']):
        if len({a['observation'], b['observation'], pa['observation'], pb['observation']}) != 1:
            raise ValueError('ordered report observations differ')
        raw = raw_pair_gate(a, b)
        phase = {m: dict(corrected=corrected_phase_gate(pa[m], pb[m]),
                         all_moments=difference_gate(pa[m], pb[m])) for m in MEASURES}
        passed = a['precision_gate'] and b['precision_gate'] and raw['pass_all'] and all(
            phase[m]['corrected']['pass_all'] for m in MEASURES)
        rows.append(dict(observation=a['observation'], raw=raw, phase=phase, accuracy_pass=bool(passed)))
    return dict(observations=rows, accuracy_pass=all(r['accuracy_pass'] for r in rows))


def recipe_costs(warmups, productions, report_stage, preparation):
    """Count actual work once; charge shared algorithm preparation to both arms."""
    costs = dict(candidate=zero_cost(), reference=zero_cost())
    overhead = zero_cost()
    actual = validate_cost(preparation)
    for group, warm in enumerate(warmups):
        expected = {f'pilot_{i}' for i in range(4)} | {f'pilot_{i}_phase' for i in range(4)} | {
            'geometry_fit', 'proposal_fit', 'coefficient_fit'}
        if len(warm['steps']) != len(expected) or {s['tag'] for s in warm['steps']} != expected:
            raise ValueError('every declared warmup cost is required')
        for step in warm['steps']:
            for kind in ('candidate', 'reference'):
                if step['tag'] != 'proposal_fit' or kind == 'candidate':
                    costs[kind] = add_cost(costs[kind], step['cost'])
        overhead = add_cost(overhead, warm['verification_overhead'])
        actual = add_cost(actual, warm['cost'])
    if len(warmups) != 5 or len(productions) != 40:
        raise ValueError('all five warmups and forty paired production stages required')
    for index, production in enumerate(productions):
        replica = index % 8
        expected = [f'{kind}_{replica}{suffix}' for kind in production_order(replica) for suffix in ('', '_phase')]
        if [s['tag'] for s in production['steps']] != expected:
            raise ValueError('all production and phase costs in counterbalanced order required')
        for step in production['steps']:
            kind = 'candidate' if step['tag'].startswith('candidate_') else 'reference'
            costs[kind] = add_cost(costs[kind], step['cost'])
        overhead = add_cost(overhead, production['verification_overhead'])
        actual = add_cost(actual, production['cost'])
    expected = {f'group_{g}_{kind}_{label}' for g in range(5) for kind in ('candidate', 'reference')
                for label in ('raw_report', 'phase_report')}
    if len(report_stage['steps']) != 20 or {s['tag'] for s in report_stage['steps']} != expected:
        raise ValueError('all raw and phase reporting costs required')
    for step in report_stage['steps']:
        kind = step['tag'].split('_')[2]
        costs[kind] = add_cost(costs[kind], step['cost'])
    overhead = add_cost(overhead, report_stage['verification_overhead'], preparation)
    actual = add_cost(actual, report_stage['cost'])
    for kind in costs:
        costs[kind] = add_cost(costs[kind], scale_cost(overhead, .5))
    return dict(recipes=costs, actual_work=actual, shared_verification_and_preflight=overhead,
        shared_allocation='Shared pilots/phase/geometry/coefficients charged fully to each recipe; proposal fit candidate-only; verification/comparison/preflight split equally; actual work never double counted.',
        measure='Measured process wall and allocated-resource wall, not busy compute or panel elapsed time. Preflight uses a separately recorded scheduler allocation upper bound.')


def report_panel(stage, d, entries):
    root, groups = Path(d['output']), []
    warmups, productions = [], []
    for group in range(5):
        validate_warmup(d, stage.declaration_path, entries, group)
        warmups.append(read(root/f'warmup_{group}_stage.json'))
    for index in range(40):
        row = read(root/f'production_{index}_stage.json')
        validate_stage_receipt(row, d, stage.declaration_path, f'production_{index}')
        productions.append(row)
    for group in range(5):
        directory = root/f'group_{group}'
        records = {}
        for kind in ('candidate', 'reference'):
            for i in range(8):
                validate_phase(d, entries, group, i, kind)
            runs = [str(directory/f'{kind}_{i}') for i in range(8)]
            diagnostics = [str(directory/f'{kind}_{i}_phase.json') for i in range(8)]
            stage.timed(directory, f'group_{group}_{kind}_raw_report', SCRIPTS['raw_report'],
                ['--runs', *runs, '--output', str(directory/f'{kind}_report.json')])
            stage.timed(directory, f'group_{group}_{kind}_phase_report', SCRIPTS['phase_report'],
                ['--diagnostics', *diagnostics, '--coefficients', str(directory/'phase_coefficients.json'),
                 '--output', str(directory/f'{kind}_phase_report.json')])
            raw, phase = read(directory/f'{kind}_report.json'), read(directory/f'{kind}_phase_report.json')
            validate_report(d, group, kind, raw, phase, directory)
            records[kind] = (raw, phase)
        comparison = compare_group(records['candidate'][0], records['reference'][0],
                                   records['candidate'][1], records['reference'][1])
        groups.append(dict(group=group, **comparison, report_sha256={str(directory/name): digest(directory/name)
            for name in ('candidate_report.json', 'reference_report.json', 'candidate_phase_report.json', 'reference_phase_report.json')}))
    return groups, warmups, productions


def panel_result(groups, costs):
    if len(groups) != 5 or [g['group'] for g in groups] != list(range(5)):
        raise ValueError('all ordered groups required')
    accuracy = all(g['accuracy_pass'] for g in groups)
    candidate, reference = (validate_cost(costs['recipes'][kind], positive=True)['process_wall_seconds']
                            for kind in ('candidate', 'reference'))
    lower_cost = candidate < reference
    return dict(heldout_accuracy_gate=accuracy, heldout_cost_gate=lower_cost,
        heldout_gate=accuracy and lower_cost, accepted_sampler=accuracy and lower_cost,
        reference_to_candidate_cost_ratio=reference/candidate)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('warmup', 'production', 'report'))
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if ((args.stage == 'report' and args.index is not None)
            or (args.stage != 'report' and (args.index is None or not 0 <= args.index < (5 if args.stage == 'warmup' else 40)))):
        raise ValueError('valid stage-specific array index required')
    d, entries = validate_declaration(args.declaration)
    resources = validate_runtime(d, gpu=args.stage != 'report')
    name = args.stage if args.index is None else f'{args.stage}_{args.index}'
    stage = Stage(d, args.declaration.resolve(), name, resources, started)
    try:
        if args.stage == 'warmup':
            warmup(stage, d, entries, args.index)
        elif args.stage == 'production':
            production(stage, d, entries, args.index)
        else:
            groups, warmups, productions = report_panel(stage, d, entries)
        # Revalidate every frozen input before any stage can certify success.
        after, _ = validate_declaration(args.declaration)
        if after != d:
            raise ValueError('declaration changed while stage ran')
    except BaseException as error:
        failed = stage.finish(error)
        save(Path(d['output'])/(name+'_panel_rejection.json'), dict(
            status='rejected', heldout_gate=False, accepted_sampler=False,
            reason='A failed or incomplete declared stage rejects the entire panel; no replacement or retry.',
            stage=name, stage_sha256=digest(stage.path), cost_preserved=failed['cost'],
            declaration_sha256=digest(args.declaration)))
        raise
    receipt = stage.finish()
    if args.stage == 'report':
        costs = recipe_costs(warmups, productions, receipt, bound_json(d['preparation_cost_receipt']))
        result = dict(schema='sbsi-corrected-heldout-panel-result-v1', groups=groups, costs=costs,
            **panel_result(groups, costs), declaration_sha256=digest(args.declaration),
            script_sha256=digest(__file__), stage_sha256=digest(stage.path),
            scientific_scope='Conditional fixed-y heldout accuracy/efficiency under frozen fits; no global rare-mode or population shear-bias certification.',
            search_and_development_cost_included=False)
        save(Path(d['output'])/'result.json', result)
        print('CORRECTED_HELDOUT_PANEL', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
