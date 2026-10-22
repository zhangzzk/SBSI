"""Frozen paired full-SMC replay/cost experiment; no adaptive choices or retries.

Preparation hashes files only. Run and report modes require scheduler allocations.
Four pairs estimate bounded timing variability, never posterior convergence.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import time

BASE_HASH = 'c5102579e3dfd1cd3c3e9ae5a267cf8ac0df9a6a82d60417fd2772e56aebd5ad'
OPT_HASH = '738df16601bbebbabc15ac7f15e4c2f47255919a7cc0e02e8c0e1df0b5d4508e'
SEEDS = [2026103600, 2026103601, 2026103602, 2026103603]
OBSERVATIONS = [58165, 0, 1, 8]
TIMINGS = {
    'result': {('timing_seconds', key) for key in ('initialization', 'sampling', 'total')}
              | {('final_stage', 'elapsed_seconds')},
    'progress': {('stages', '*', 'elapsed_seconds')},
    'protocol': {('runtime', 'slurm_job_id')},
}
ARTIFACT_HASH_KEYS = {'protocol_sha256', 'particles_sha256', 'scenes_sha256'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def manifest(source):
    source = Path(source).resolve()
    result = {}
    for line in (source/'SHA256SUMS').read_text().splitlines():
        checksum, name = line.split('  ', 1)
        path = (source/name).resolve()
        if not path.is_relative_to(source) or name in result or digest(path) != checksum:
            raise ValueError('invalid or changed source manifest: '+name)
        result[name] = checksum
    if not result:
        raise ValueError('empty source manifest')
    return result


def verify(declaration):
    sources = {}
    for arm, item in declaration['arms'].items():
        root = Path(item['source'])
        if digest(root/'SHA256SUMS') != item['manifest_sha256']:
            raise ValueError('changed manifest')
        sources[arm] = manifest(root)
    a, b = sources['baseline'], sources['optimized']
    if set(a) != set(b) or {k for k in a if a[k] != b[k]} != {'sbsi/scene_exchange.py'}:
        raise ValueError('arms must differ in exactly the categorical source')
    if a['sbsi/scene_exchange.py'] != BASE_HASH or b['sbsi/scene_exchange.py'] != OPT_HASH:
        raise ValueError('unknown baseline or candidate')
    for name, checksum in declaration['external_files'].items():
        if digest(name) != checksum:
            raise ValueError('changed external input: '+name)
    for name, checksum in declaration['harness_files'].items():
        if digest(name) != checksum:
            raise ValueError('changed harness: '+name)
    if declaration['seeds'] != SEEDS or declaration['observations'] != OBSERVATIONS:
        raise ValueError('undeclared seed or observation recipe')
    expected_protocol(declaration, sources, 0)
    return sources


def compare_json(a, b, kind, path=()):
    """Exact structure/scalars; only named timing and allocation-ID leaves differ."""
    pattern = tuple('*' if isinstance(p, int) else p for p in path)
    if pattern in TIMINGS[kind]:
        if pattern[-1] != 'slurm_job_id' and not all(
                isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (a,b)):
            raise ValueError('invalid timing: '+str(path))
        return
    if type(a) is not type(b):
        raise ValueError('JSON type mismatch: '+str(path))
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('JSON keys mismatch: '+str(path))
        for key in a:
            compare_json(a[key], b[key], kind, path+(key,))
    elif isinstance(a, list):
        if len(a) != len(b):
            raise ValueError('JSON length mismatch: '+str(path))
        for i,(x,y) in enumerate(zip(a,b)):
            compare_json(x,y,kind,path+(i,))
    elif isinstance(a, float):
        if struct.pack('!d', a) != struct.pack('!d', b):
            raise ValueError('JSON float bits mismatch: '+str(path))
    elif a != b:
        raise ValueError('JSON value mismatch: '+str(path))


def compare_npz(first, second):
    import numpy as np
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('NPZ field mismatch')
        receipt = {}
        for key in sorted(a.files):
            x,y = a[key],b[key]
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                raise ValueError('NPZ bytes/dtype/shape mismatch: '+key)
            receipt[key] = dict(dtype=x.dtype.str, shape=list(x.shape),
                                bytes_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        return receipt


def comparable(run, source, expected_manifest):
    """Validate hashes before normalizing explicitly permitted provenance fields."""
    run, source = Path(run),Path(source).resolve()
    result, protocol = load(run/'result.json'),load(run/'protocol.json')
    if result['status'] not in ('completed','completed_with_extinction'):
        raise ValueError('incomplete population')
    for key in ARTIFACT_HASH_KEYS:
        name = key[:-7]
        suffix = '.json' if name == 'protocol' else '.npz'
        if result[key] != digest(run/(name+suffix)):
            raise ValueError('changed saved artifact: '+name)
        del result[key]
    protocol = normalized_protocol(protocol, source, expected_manifest)
    return result, protocol, load(run/'progress.json')


def normalized_protocol(protocol, source, expected_manifest):
    """Normalize only checked source locations and the known categorical hash."""
    protocol, source = copy.deepcopy(protocol), Path(source).resolve()
    sources = {}
    for path, checksum in protocol['implementation_sha256'].items():
        relative = Path(path).resolve().relative_to(source).as_posix()
        if relative in sources or expected_manifest.get(relative) != checksum:
            raise ValueError('unexpected protocol implementation')
        sources[relative] = checksum
    if 'sbsi/scene_exchange.py' not in sources:
        raise ValueError('exchange implementation missing')
    # Paths become relative; only the validated candidate source hash is replaced.
    sources['sbsi/scene_exchange.py'] = 'validated categorical arm'
    protocol['implementation_sha256'] = sources
    return protocol


def expected_protocol(declaration, sources, replica):
    if type(replica) is not int or replica not in range(4):
        raise ValueError('one of the four declared replicas required')
    item = declaration['protocol_template']
    if digest(item['path']) != item['sha256']:
        raise ValueError('audited development protocol template changed')
    expected = normalized_protocol(load(item['path']), item['source'], sources['baseline'])
    fixed = dict(observations=OBSERVATIONS, particles=256, stages=64, moves=64,
        scale=1., fixed_shear=[0.,0.], density_batch_size=4096,
        betas=[(i/64)**2 for i in range(65)])
    for key,value in fixed.items():
        compare_json(expected[key], value, 'protocol')
    compare_json(declaration['settings'], dict(particles=256,stages=64,moves=64,
        power=2.,scale=1.,shear=[0.,0.],density_batch_size=4096), 'protocol')
    if (expected['reference'] != declaration['reference']
            or expected['initial_proposal']['path'] != declaration['initial_proposal']
            or expected['physical_geometry']['path'] != declaration['physical_geometry']):
        raise ValueError('declared inputs differ from the audited development recipe')
    runtime = {k:v for k,v in expected['runtime'].items() if k != 'slurm_job_id'}
    compare_json(runtime, declaration['expected_runtime'], 'protocol')
    if (runtime['device'] != 'cuda' or runtime['cuda_device_name'] != 'Tesla V100-SXM2-32GB'
            or runtime['slurm_node_list'] != 'th-cl-nv02' or runtime['torch_num_threads'] != 2):
        raise ValueError('declared V100 runtime required')
    expected['seed'] = SEEDS[replica]
    return expected


def validate_saved_run(result, protocol, progress, expected):
    """Both arms must match the audited recipe, not merely each other."""
    compare_json(protocol, expected, 'protocol')
    for key in ('particles', 'observations'):
        compare_json(result[key], expected[key], 'result')
    if (set(progress) != {'requested', 'stages'} or type(progress['requested']) is not int
            or progress['requested'] != 64 or len(progress['stages']) != 64):
        raise ValueError('exactly 64 requested and completed stages required')
    for index,stage in enumerate(progress['stages'], 1):
        compare_json(stage['stage'], index, 'protocol')
        compare_json(stage['beta'], expected['betas'][index], 'protocol')
    # The result's final stage must be the actual last progress record, including
    # its recorded timestamp; no reconstruction or shortened history is allowed.
    compare_json(result['final_stage'], progress['stages'][-1], 'protocol')


def compare_runs(declaration, pair_directory, sources, replica):
    expected = expected_protocol(declaration, sources, replica)
    values = {}
    for arm in ('baseline','optimized'):
        values[arm] = comparable(Path(pair_directory)/arm,
            declaration['arms'][arm]['source'], sources[arm])
        validate_saved_run(*values[arm], expected)
    for i,kind in enumerate(('result','protocol','progress')):
        compare_json(values['baseline'][i],values['optimized'][i],kind)
    return {name:compare_npz(Path(pair_directory)/'baseline'/name,
                             Path(pair_directory)/'optimized'/name)
            for name in ('particles.npz','scenes.npz')}


def driver_command(declaration, arm, seed, output):
    source = declaration['arms'][arm]['source']
    bootstrap = ('import runpy,sys;from pathlib import Path;'
        's=Path(sys.argv.pop(1)).resolve();sys.path[:0]=[str(s),str(s/"scripts")];'
        'import sbsi;assert Path(sbsi.__file__).resolve().parent==s/"sbsi";'
        'runpy.run_path(str(s/"scripts/check_galsbi_particles.py"),run_name="__main__")')
    return [sys.executable,'-u','-c',bootstrap,source,'--reference',declaration['reference'],
        '--output',str(output),'--observation-indices',*map(str,OBSERVATIONS),
        '--particles','256','--stages','64','--moves','64','--power','2','--scale','1',
        '--seed',str(seed),'--shear','0','0','--device','cuda','--density-batch-size','4096',
        '--physical-geometry',declaration['physical_geometry'],
        '--initial-proposal',declaration['initial_proposal']]


def run_pair(declaration_path, replica):
    if 'SLURM_JOB_ID' not in os.environ or replica not in range(4):
        raise ValueError('scheduled declared replica required')
    d = load(declaration_path)
    directory = Path(d['output'])/f'pair_{replica}'
    directory.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='started',replica=replica,seed=SEEDS[replica],
        declaration_sha256=digest(declaration_path),order=(['baseline','optimized'] if replica%2==0
        else ['optimized','baseline']),arms={},slurm_job_id=os.environ['SLURM_JOB_ID'])
    receipt_path = directory/'receipt.json'
    save(receipt_path,receipt)
    try:
        sources = verify(d)
        for arm in receipt['order']:
            command = driver_command(d,arm,SEEDS[replica],directory/arm)
            started = time.monotonic()
            with (directory/(arm+'.log')).open('x') as log:
                process = subprocess.run(command,cwd=d['arms'][arm]['source'],stdout=log,stderr=subprocess.STDOUT)
            receipt['arms'][arm] = dict(returncode=process.returncode,
                subprocess_seconds=time.monotonic()-started,command=command)
            if (directory/arm/'result.json').exists():
                receipt['arms'][arm]['driver_timing_seconds'] = load(directory/arm/'result.json').get('timing_seconds')
            save(receipt_path,receipt)
        if any(a['returncode'] != 0 for a in receipt['arms'].values()):
            raise ValueError('an arm failed; original outputs and costs retained')
        receipt['arrays'] = compare_runs(d,directory,sources,replica)
        verify(d)
        receipt.update(status='completed',bitwise_scientific_replay=True)
    except Exception as error:
        receipt.update(status='failed',error=repr(error),bitwise_scientific_replay=False)
        raise
    finally:
        save(receipt_path,receipt)


def report(declaration_path, output):
    if 'SLURM_JOB_ID' not in os.environ or Path(output).exists():
        raise ValueError('scheduled report and fresh output required')
    start = time.monotonic()
    d=load(declaration_path)
    sources=verify(d)
    pairs=[]
    for i in range(4):
        folder=Path(d['output'])/f'pair_{i}'
        r=load(folder/'receipt.json')
        if r['status'] != 'completed' or r['declaration_sha256'] != digest(declaration_path):
            raise ValueError('all four original pairs must complete and replay')
        compare_runs(d,folder,sources,i)
        process=load(Path(d['output'])/f'pair_{i}_process_cost.json')
        arm_total=sum(r['arms'][a]['subprocess_seconds'] for a in ('baseline','optimized'))
        common=max(0.,process['seconds']-arm_total)
        costs={a:r['arms'][a]['subprocess_seconds']+common/2 for a in ('baseline','optimized')}
        pairs.append(dict(replica=i,seed=r['seed'],order=r['order'],arms=r['arms'],
            pair_process_seconds=process['seconds'],common_verification_comparison_overhead=common,
            charged_seconds=costs,baseline_to_optimized_ratio=costs['baseline']/costs['optimized']))
    report_seconds=time.monotonic()-start
    # Report process startup is recorded separately by the launcher, then charged
    # equally in the finalization command; no result is final before that receipt.
    ratios=[p['baseline_to_optimized_ratio'] for p in pairs]
    save(output,dict(status='pending_report_process_cost',all_four_bitwise_replay=True,pairs=pairs,
        ratio_range=[min(ratios),max(ratios)],ratio_mean=statistics.mean(ratios),
        ratio_mean_standard_error=statistics.stdev(ratios)/2,
        internal_report_seconds=report_seconds,source_preparation_seconds=d.get('source_preparation_seconds',0.),primary_cpu_metric_unchanged=1.3609505278392875,
        limitations=['Four fixed development pairs, one V100; no strong distributional speed claim.',
          'Whole populations are replicates; no convergence or posterior-accuracy inference.',
          'No extra warmup or replay run; per-arm replay diagnostics are inside timed driver.',
          'Shared source verification, comparison and process startup costs split equally.',
          'Driver does not save terminal RNG; all saved arrays and scientific histories compare exactly.']))


def finalize(report_path, cost_path):
    r=load(report_path); cost=load(cost_path)
    if r['status'] != 'pending_report_process_cost' or cost['exit_code'] != 0:
        raise ValueError('successful original report process cost required')
    total={arm:sum(p['charged_seconds'][arm] for p in r['pairs'])+cost['seconds']/2+r.get('source_preparation_seconds',0.)/2
           for arm in ('baseline','optimized')}
    r.update(status='completed',report_process_cost=cost,new_work_seconds_by_arm=total,
        cost_scope='Complete new run processes and measured source-copy/hash/report overhead, conditional on existing frozen fits; historical pilot generation, fitting and search costs are excluded.',
        total_baseline_to_optimized_ratio=total['baseline']/total['optimized'],
        bookkeeping_finalization_cost='Not measured; JSON-only finalization, no scientific work.')
    save(report_path,r)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run-pair','report','finalize'])
    p.add_argument('--declaration',type=Path)
    p.add_argument('--replica',type=int)
    p.add_argument('--output',type=Path)
    p.add_argument('--cost',type=Path)
    a=p.parse_args()
    if a.mode=='run-pair': run_pair(a.declaration,a.replica)
    elif a.mode=='report': report(a.declaration,a.output)
    else: finalize(a.output,a.cost)


if __name__=='__main__': main()
