"""CPU-only execution of the real frozen production startup through artifact validation.

The original driver is executed unchanged. An explicit sentinel stops it at
its first torch.set_num_threads call, after geometry/proposal validation and
initial protocol saving, before any model load, CUDA call or particle draw.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ERROR = 'production target identity differs from pilot: native_target_source_sha256'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def harness():
    spec = importlib.util.spec_from_file_location('_startup_benchmark', Path(__file__).with_name('benchmark_prior_reuse_smc.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def configuration(d):
    return dict(schema_version=d['schema_version'], seeds=d['seeds'], observations=d['observations'],
        settings=d['settings'], source_manifests=d['arms'],
        initial_proposals={arm: dict(path=path, sha256=digest(path)) for arm, path in d['initial_proposals'].items()},
        geometry=dict(path=d['physical_geometry'], sha256=digest(d['physical_geometry'])),
        reference=dict(path=d['reference'],
            protocol_sha256=digest(Path(d['reference']) / 'protocol.json'),
            observation_sha256=digest(Path(d['reference']) / 'observations.parquet')),
        native_replay=dict(path=d['proposal_migration']['native_replay'],
                           sha256=d['proposal_migration']['native_replay_sha256']))


def validate_receipt(d, sources):
    item = d['startup_preflight']
    if digest(item['path']) != item['sha256'] or d['external_files'].get(item['path']) != item['sha256']:
        raise ValueError('original successful startup preflight must be pinned')
    r = load(item['path'])
    if r['status'] != 'passed' or r['configuration'] != configuration(d) or set(r['arms']) != {'baseline', 'optimized'}:
        raise ValueError('startup preflight differs from the declared production configuration')
    m = harness()
    for arm, value in r['arms'].items():
        if (value['returncode'] != 0 or value['status'] != 'passed' or value['cuda_initialized'] is not False or
                [row['seed'] for row in value['runs']] != d['seeds'] or
                any(row['status'] != 'passed_before_models_and_sampling' for row in value['runs'])):
            raise ValueError('all eight exact production startup validations required')
        for path, checksum in value['evidence_sha256'].items():
            if d['external_files'].get(path) != checksum or digest(path) != checksum:
                raise ValueError('changed startup protocol, command or worker evidence')
        worker_paths = [path for path in value['evidence_sha256'] if Path(path).name == 'worker.json']
        request_paths = [path for path in value['evidence_sha256'] if Path(path).name == 'request.json']
        if len(worker_paths) != 1 or len(request_paths) != 1:
            raise ValueError('one original startup request and worker result required')
        worker, request = load(worker_paths[0]), load(request_paths[0])
        for key in ('status', 'runs', 'cuda_initialized', 'original_artifact_negative_control'):
            if worker[key] != value[key]:
                raise ValueError('startup summary differs from its original worker result')
        if request['source'] != d['arms'][arm]['source'] or len(request['runs']) != 4:
            raise ValueError('startup request must use the declared source and all four seeds')
        for replica, row in enumerate(value['runs']):
            if value['evidence_sha256'].get(row['protocol']) != row['protocol_sha256']:
                raise ValueError('all original startup protocols must be pinned')
            output = Path(row['protocol']).parent
            expected_command = m.driver_command(d, arm, d['seeds'][replica], output)
            if request['runs'][replica] != dict(seed=d['seeds'][replica], output=str(output), command=expected_command):
                raise ValueError('startup command differs from the final production recipe')
            validate_protocol(m, d, sources, arm, replica, row['protocol'])
        if arm == 'optimized':
            command = m.driver_command(d, arm, d['seeds'][0], Path(request_paths[0]).parent / 'original_artifact_negative_control')
            command[command.index('--initial-proposal') + 1] = d['initial_proposal']
            if request.get('negative_command') != command:
                raise ValueError('negative control must execute the original artifact against optimized source')
    negative = r['arms']['optimized']['original_artifact_negative_control']
    if negative != dict(status='rejected', exception='ValueError', message=ERROR, no_sampled_artifacts=True):
        raise ValueError('original mismatched artifact must still fail the public production validator')
    return r



def validate_protocol(m, d, sources, arm, replica, path):
    protocol = m._provenance.normalize_initial_protocol(load(path), d, arm, sources, m.HASHES)
    protocol = m.normalized_protocol(protocol, d['arms'][arm]['source'], sources[arm])
    expected = m.expected_protocol(d, sources, replica)
    del expected['runtime']
    del expected['proposals']
    m.compare_json(protocol, expected, 'protocol')


def worker(request_path, output):
    request = load(request_path)
    source = Path(request['source']).resolve()
    os.chdir(source)
    sys.path[:0] = [str(source), str(source / 'scripts')]
    import torch
    import sbsi
    if Path(sbsi.__file__).resolve().parent != source / 'sbsi' or torch.cuda.is_initialized():
        raise ValueError('isolated source and uninitialized CUDA required')
    namespace = runpy.run_path(str(source / 'scripts/check_galsbi_particles.py'), run_name='startup_preflight_driver')
    class StartupValidated(Exception): pass
    def stop_before_models(threads):
        raise StartupValidated()
    original_threads, original_argv = torch.set_num_threads, sys.argv
    torch.set_num_threads = stop_before_models
    results = []
    try:
        for item in request['runs']:
            sys.argv = ['check_galsbi_particles.py', *item['command'][5:]]
            try:
                namespace['main']()
            except StartupValidated:
                folder = Path(item['output'])
                if set(p.name for p in folder.iterdir()) != {'protocol.json'}:
                    raise ValueError('startup preflight must stop before sampled artifacts')
                results.append(dict(seed=item['seed'], status='passed_before_models_and_sampling',
                    protocol=str(folder / 'protocol.json'), protocol_sha256=digest(folder / 'protocol.json')))
            else:
                raise ValueError('driver did not reach the exact pre-model checkpoint')
        negative = None
        if request.get('negative_command') is not None:
            command = request['negative_command']
            sys.argv = ['check_galsbi_particles.py', *command[5:]]
            try:
                namespace['main']()
            except ValueError as error:
                if str(error) != ERROR:
                    raise
                folder = Path(command[command.index('--output') + 1])
                if folder.exists() and any(folder.iterdir()):
                    raise ValueError('negative startup control produced output artifacts')
                negative = dict(status='rejected', exception='ValueError', message=ERROR, no_sampled_artifacts=True)
            else:
                raise ValueError('original artifact unexpectedly passed optimized target validation')
        if torch.cuda.is_initialized():
            raise ValueError('CPU startup validation must not initialize CUDA')
        save(output, dict(status='passed', runs=results, cuda_initialized=False,
                          original_artifact_negative_control=negative))
    finally:
        torch.set_num_threads, sys.argv = original_threads, original_argv


def preflight(declaration, output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('production startup preflight requires a CPU scheduler allocation')
    started = time.monotonic()
    m = harness()
    d = load(declaration)
    if d['schema_version'] != 2:
        raise ValueError('explicit v2 provenance migration required')
    sources = m.verify(d, require_preflight=False)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', configuration=configuration(d), arms={})
    try:
        for arm in ('baseline', 'optimized'):
            folder = root / arm
            folder.mkdir()
            request = dict(source=d['arms'][arm]['source'], runs=[])
            for replica, seed in enumerate(d['seeds']):
                dest = folder / f'seed_{seed}'
                command = m.driver_command(d, arm, seed, dest)
                request['runs'].append(dict(seed=seed, output=str(dest), command=command))
            if arm == 'optimized':
                negative = m.driver_command(d, arm, d['seeds'][0], folder / 'original_artifact_negative_control')
                negative[negative.index('--initial-proposal') + 1] = d['initial_proposal']
                request['negative_command'] = negative
            request_path, worker_path = folder / 'request.json', folder / 'worker.json'
            save(request_path, request)
            command = [d['python'], str(Path(__file__).resolve()), 'worker', '--request', str(request_path), '--output', str(worker_path)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
            env.pop('PYTHONPATH', None)
            arm_start = time.monotonic()
            with (folder / 'worker.log').open('x') as log:
                process = subprocess.run(command, cwd=d['arms'][arm]['source'], env=env, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                result['arms'][arm] = dict(returncode=process.returncode, process_seconds=time.monotonic() - arm_start)
                raise ValueError('real production startup validation failed: ' + arm)
            r = load(worker_path)
            r.update(returncode=0, process_seconds=time.monotonic() - arm_start,
                     evidence_sha256={str(request_path): digest(request_path), str(worker_path): digest(worker_path)})
            for replica, row in enumerate(r['runs']):
                validate_protocol(m, d, sources, arm, replica, row['protocol'])
                r['evidence_sha256'][row['protocol']] = row['protocol_sha256']
            result['arms'][arm] = r
        m.verify(d, require_preflight=False)
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['process_seconds'] = time.monotonic() - started
        save(root / 'result.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'worker'])
    p.add_argument('--declaration', type=Path)
    p.add_argument('--request', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'worker': worker(a.request, a.output)
    else: preflight(a.declaration, a.output)
