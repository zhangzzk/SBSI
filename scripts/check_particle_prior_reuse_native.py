#!/usr/bin/env python
"""Exact CPU replay of native-prior reuse on the fixed corrected development panel.

Run through Slurm. No likelihood model, GPU, random draws or held-out scientific
outputs are used. Every mismatch is retained; there is no tolerance fallback.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


AUDIT_SHA256 = '09601af0a43c0b5801bc0467b090e588c745713a3c9fd6d86b58f2f9172d1574'
BASELINE_SHA256 = {
    'sbsi/galsbi_prior.py': 'f6612ec614f966993946e3beb94221e6e6ad3adcf1782393f18cb8facc782652',
    'sbsi/galsbi_target_coordinates.py': '519c9cd4930a2a0a22d9989bad55b8dc436fe16760f8ed6d5dff3d8160047209',
    'sbsi/galsbi_particle_target.py': '9de76a6089b31f73d632e6c3cf768df7667e96f1608e20dff43ee8ab9ce7e5ec',
    'scripts/check_galsbi_continuous_prior.py': '3d310798e4aabedf297b94c2c6e16964d41e0a971513b59eb3358e66c7ee877b',
}
ARMS = [('small', 256, 2026102900), ('large', 1024, 2026103000), ('broad', 1024, 2026103100)]
OBSERVATIONS = [58165, 0, 1, 8]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pin(path, expected, sources):
    path = Path(path).resolve()
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f'changed or unbound input: {path}')
    sources[str(path)] = actual
    return actual


def panel_inputs(root, sources):
    root = Path(root).resolve()
    if any('heldout' in part.lower() for part in root.parts):
        raise ValueError('development inputs required; no held-out data are permitted')
    audit_path = root/'v10_corrected_comparison_20261022_v3.json'
    pin(audit_path, AUDIT_SHA256, sources)
    audit = json.loads(audit_path.read_text())
    if audit.get('schema') != 'sbsi-n23-corrected-comparison-v1' or audit.get('development_checks_pass') is not True:
        raise ValueError('completed authoritative development audit required')
    result = []
    for arm, particles, seed_base in ARMS:
        for replica in range(8):
            directory = root/f'dev_v10_{arm}_{replica}'
            paths = {name: directory/name for name in ('particles.npz', 'protocol.json', 'result.json')}
            for path in paths.values():
                expected = audit['sources'].get(str(path))
                if expected is None:
                    raise ValueError(f'population input absent from authoritative audit: {path}')
                pin(path, expected, sources)
            protocol = json.loads(paths['protocol.json'].read_text())
            receipt = json.loads(paths['result.json'].read_text())
            if (protocol.get('observations') != OBSERVATIONS
                    or protocol.get('particles') != particles
                    or protocol.get('seed') != seed_base+replica
                    or protocol.get('fixed_shear') != [0., 0.]
                    or receipt.get('status') != 'completed'
                    or receipt.get('particles_sha256') != sources[str(paths['particles.npz'])]
                    or receipt.get('protocol_sha256') != sources[str(paths['protocol.json'])]):
                raise ValueError(f'changed population identity: {directory}')
            result.append(dict(arm=arm, replica=replica, seed=seed_base+replica,
                               particles=particles, path=str(paths['particles.npz'])))
    return result


def array_record(value):
    import numpy as np
    array = np.asarray(value)
    description = dict(dtype=array.dtype.str, shape=list(array.shape),
                       bytes_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())
    if np.issubdtype(array.dtype, np.number):
        description.update(finite=int(np.isfinite(array).sum()), nan=int(np.isnan(array).sum()),
                           positive_infinite=int(np.isposinf(array).sum()),
                           negative_infinite=int(np.isneginf(array).sum()))
    return description


def compare_array(candidate, baseline):
    import numpy as np
    a, b = np.asarray(candidate), np.asarray(baseline)
    left, right = array_record(a), array_record(b)
    matched = left == right
    result = dict(matches=matched, candidate=left, baseline=right)
    if not matched and a.shape == b.shape and a.dtype == b.dtype:
        # Compare element bytes: signed zero and distinct NaN payloads matter.
        av = np.ascontiguousarray(a).view(np.uint8).reshape(-1, a.dtype.itemsize)
        bv = np.ascontiguousarray(b).view(np.uint8).reshape(-1, b.dtype.itemsize)
        different = np.flatnonzero(np.any(av != bv, axis=1))
        result['mismatched_elements'] = len(different)
        result['examples'] = [dict(index=list(map(int, np.unravel_index(int(i), a.shape))),
            candidate=repr(a.flat[int(i)].item()), baseline=repr(b.flat[int(i)].item()),
            candidate_hex=av[i].tobytes().hex(), baseline_hex=bv[i].tobytes().hex())
            for i in different[:8]]
    return result


def compare_native(candidate, baseline):
    result = {'log_prior': compare_array(candidate[0], baseline[0]),
              'type_probability': compare_array(candidate[2], baseline[2])}
    if candidate[1].keys() != baseline[1].keys():
        result['native_schema'] = dict(matches=False, candidate=list(candidate[1]), baseline=list(baseline[1]))
    for key in sorted(candidate[1].keys() & baseline[1].keys()):
        result['native/'+key] = compare_array(candidate[1][key], baseline[1][key])
    return dict(matches=all(value['matches'] for value in result.values()), fields=result)


def support_probes(first, prior):
    import numpy as np
    rows, labels = [np.asarray(first, float).copy()], ['control']

    def add(label, column, value):
        row = rows[0].copy()
        row[column] = value
        rows.append(row)
        labels.append(label)

    for kind in prior.types:
        edge = np.log(prior.z_max[kind])
        for direction, value in [('below', np.nextafter(edge, -np.inf)),
                                 ('at', edge), ('above', np.nextafter(edge, np.inf))]:
            add(f'log_redshift_{kind}_{direction}', 0, value)
    for label, column, value in [
            ('redshift_underflow_placeholder', 0, -1000.),
            ('redshift_overflow_placeholder', 0, 1000.),
            ('shape_zero_limit', 8, -1000.), ('shape_upper_limit', 8, 1000.),
            ('sed_zero_component', 2, -1000.), ('sed_one_component', 2, 1000.),
            ('intrinsic_bright', 1, prior.p['gals_mag_min']-10.),
            ('intrinsic_faint', 1, prior.p['gals_mag_max']+10.)]:
        add(label, column, value)
    return np.vstack(rows), labels


def make_target(target_type, coordinate_type, galaxy, cosmo, memo_type):
    distance = memo_type(lambda z: cosmo.background.dist_ang_a(a=1/(1+z)))
    target = target_type.__new__(target_type)
    target.prior = SimpleNamespace(galaxy=galaxy)
    target.coordinates = coordinate_type(galaxy, distance)
    return target


def run(args, receipt):
    root, baseline_root = args.runs_root.resolve(), args.baseline_source.resolve()
    candidate_root = Path(__file__).resolve().parents[1]
    sources = receipt['source_sha256']
    populations = panel_inputs(root, receipt['input_sha256'])
    for name, expected in BASELINE_SHA256.items():
        pin(baseline_root/name, expected, sources)
        pin(candidate_root/name, None, sources)
    if digest(candidate_root/'scripts/check_galsbi_continuous_prior.py') != BASELINE_SHA256['scripts/check_galsbi_continuous_prior.py']:
        raise ValueError('candidate prior factory must remain byte-identical to baseline')
    pin(Path(__file__), None, sources)

    import torch  # Must precede the first PyCosmo import; no GPU/model use.
    import numpy as np
    import scipy
    import sbsi
    from sbsi.galsbi_prior import ExactMemo
    from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
    from sbsi.galsbi_particle_target import GalSBIParticleTarget
    import check_galsbi_continuous_prior as candidate_factory
    if Path(sbsi.__file__).resolve().parent.parent != candidate_root:
        raise ValueError('candidate imports escaped the script checkout; install this checkout')
    if Path(candidate_factory.__file__).resolve() != candidate_root/'scripts/check_galsbi_continuous_prior.py':
        raise ValueError('candidate factory import escaped the script checkout')
    torch.set_num_threads(min(2, len(os.sched_getaffinity(0))))
    original_prior = load_module('sbsi._reuse_original_prior', baseline_root/'sbsi/galsbi_prior.py')
    original_coordinates = load_module('sbsi._reuse_original_coordinates', baseline_root/'sbsi/galsbi_target_coordinates.py')
    original_target = load_module('sbsi._reuse_original_target', baseline_root/'sbsi/galsbi_particle_target.py')
    original_factory = load_module('_reuse_original_factory', baseline_root/'scripts/check_galsbi_continuous_prior.py')
    # Its import statement resolves the installed package. Explicitly bind the
    # pinned original classes so the original factory builds the original prior.
    original_factory.GalSBIContinuousPrior = original_prior.GalSBIContinuousPrior
    original_factory.ExactMemo = original_prior.ExactMemo
    receipt['baseline_reference'] = dict(method='unmodified pinned original native_prior/log_mixture_prior/log_intensity',
        source=str(baseline_root), factory_binding='original GalSBIContinuousPrior and ExactMemo',
        source_hashes_expected=BASELINE_SHA256)
    started = time.monotonic()
    old_galaxy, old_cosmo, old_config = original_factory.build_prior()
    galaxy, cosmo, config = candidate_factory.build_prior()
    receipt['prior_construction_seconds_both'] = time.monotonic()-started
    baseline = make_target(original_target.GalSBIParticleTarget,
        original_coordinates.GalSBITargetCoordinates, old_galaxy, old_cosmo, original_prior.ExactMemo)
    candidate = make_target(GalSBIParticleTarget, GalSBITargetCoordinates, galaxy, cosmo, ExactMemo)
    for module in (config, old_config):
        pin(module.__file__, None, sources)
    receipt['runtime'] = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        torch=str(torch.__version__), torch_threads=torch.get_num_threads(),
        affinity_cpus=len(os.sched_getaffinity(0)), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'),
        thread_environment={key: os.environ.get(key) for key in
                            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    receipt['native_redshift_bounds'] = {kind: float(galaxy.z_max[kind]) for kind in galaxy.types}
    receipt['cases'] = []
    total = 0

    def check_case(identity, x, labels=None):
        started = time.monotonic()
        # Alternate evaluation order for cache hygiene, not a timing comparison.
        if len(receipt['cases']) % 2:
            new, old = candidate.native_prior(x), baseline.native_prior(x)
        else:
            old, new = baseline.native_prior(x), candidate.native_prior(x)
        compared = compare_native(new, old)
        record = dict(identity, rows=len(x), coordinates=array_record(x),
                      comparison=compared, elapsed_seconds=time.monotonic()-started)
        if labels is not None:
            record.update(probe_labels=labels, probe_coordinates=x.tolist())
        receipt['cases'].append(record)
        save(args.output/'progress.json', receipt)
        print('NATIVE_REPLAY', identity['name'], len(x), 'matches', compared['matches'], flush=True)

    first = None
    for population in populations:
        with np.load(population['path'], allow_pickle=False) as data:
            coordinates = data['coordinates']
        if (coordinates.shape != (4, population['particles'], 10)
                or coordinates.dtype != np.dtype('float64') or not np.isfinite(coordinates).all()):
            raise ValueError(f'changed coordinate layout: {population["path"]}')
        x = coordinates.reshape(-1, 10)
        if first is None:
            first = x[0].copy()
        check_case(dict(population, name=f'{population["arm"]}_{population["replica"]}'), x)
        total += len(x)
    if total != 73728 or len(receipt['cases']) != 24:
        raise ValueError('the complete fixed 24-population/73728-row panel is required')
    probes, labels = support_probes(first, galaxy)
    check_case(dict(name='fixed_support_probes'), probes, labels)
    receipt['development_rows'] = total
    receipt['probe_rows'] = len(probes)
    for path, expected in (sources | receipt['input_sha256']).items():
        if digest(path) != expected:
            raise ValueError(f'input/source changed during replay: {path}')
    receipt['post_run_hashes_match'] = True
    receipt['all_exactly_equal'] = all(case['comparison']['matches'] for case in receipt['cases'])
    receipt['status'] = 'passed' if receipt['all_exactly_equal'] else 'mismatch'


def main():
    started, cpu_started = time.monotonic(), time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output directory required')
    args.output.mkdir(parents=True)
    receipt = dict(schema='sbsi-particle-prior-reuse-native-v1', status='running',
        input_sha256={}, source_sha256={},
        restrictions=['Exactly the 24 corrected development populations; no held-out data.',
            'Original full per-population row count; no likelihood or prior batch-size tuning.',
            'Exact dtype, shape and bytes required; no tolerance fallback.',
            'Costs include both reference/candidate constructions, evaluations, comparisons and hashing; not a speedup measurement.',
            'Costs exclude interpreter startup, queue time and final receipt serialization.',
            'No likelihood models, GPU, random sampling or sampler-accuracy claim.'])
    try:
        run(args, receipt)
    except Exception:
        receipt.update(status='failed', error=traceback.format_exc())
    receipt['cost'] = dict(process_elapsed_seconds=time.monotonic()-started,
        process_cpu_seconds=time.process_time()-cpu_started, gpu_seconds=0.)
    save(args.output/'result.json', receipt)
    print('NATIVE_REPLAY_RESULT', receipt['status'], args.output, flush=True)
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
