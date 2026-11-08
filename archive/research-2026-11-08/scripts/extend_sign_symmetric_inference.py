"""Extend a frozen ConstGold sample without resampling its completed C1 chains."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PYTHON = '/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def units(kind, filename, n, size):
    if n == 0:
        return []
    count = (n+size-1)//size
    edges = [n*i//count for i in range(count+1)]
    return [f'{kind}_add{i:04d} {filename} {edges[i]} {edges[i+1]-1}' for i in range(count)]


def validate_prefix(observations, truth, old_obs, old_truth, previous_count, previous_raw=100000):
    import pandas as pd
    pd.testing.assert_frame_equal(observations.iloc[:previous_count].reset_index(drop=True), old_obs,
                                  check_exact=True)
    pd.testing.assert_frame_equal(truth.iloc[:previous_count].reset_index(drop=True), old_truth,
                                  check_exact=True)
    for keys in (['source_case', 'source_input_index'], ['source_case', 'source_detection_id']):
        if truth.duplicated(keys).any():
            raise ValueError('duplicate catalogue identity in combined sample')
    if not (truth.iloc[:previous_count].parent_sample_row < previous_raw).all():
        raise ValueError('old prefix contains extension rows')
    if not (truth.iloc[previous_count:].parent_sample_row >= previous_raw).all():
        raise ValueError('extension overlaps previous prefix')


def verify(root, prepared=True):
    rule = json.loads((root/'rule.json').read_text())
    subprocess.run([PYTHON, str(Path(rule['parent'])/'verify.py')], check=True)
    subprocess.run([PYTHON, str(Path(rule['normalization'])/'verify.py')], check=True)
    for name, sha in json.loads((root/'setup_manifest.json').read_text()).items():
        if digest(root/name) != sha:
            raise ValueError('extension setup changed: '+name)
    if prepared:
        for name, sha in json.loads((root/'prepared_manifest.json').read_text()).items():
            if digest(root/name) != sha:
                raise ValueError('prepared extension input changed: '+name)
        source = Path(rule['normalization'])
        for name, sha in rule['normalization_hashes'].items():
            if digest(source/name) != sha:
                raise ValueError('reused normalization changed: '+name)
    print('EXTENSION_INPUTS_VERIFIED', flush=True)


def prepare(root):
    import numpy as np
    import pandas as pd
    rule = json.loads((root/'rule.json').read_text())
    parent = Path(rule['parent'])
    verify(root, prepared=False)
    sys.path[:0] = [str(root), str(root/'source_snapshot'), str(root/'source_snapshot/scripts')]
    from v39_image_input import selected_prefix
    old_protocol = json.loads((parent/'reference/protocol.json').read_text())
    source = Path(old_protocol['source'])
    total = rule['requested_total_rows']
    previous_raw = rule['new_raw_rows'][0]
    if digest(source/'image_mock_manifest.json') != old_protocol['source_manifest_sha256']:
        raise ValueError('parent catalogue manifest changed')
    manifest = json.loads((source/'image_mock_manifest.json').read_text())
    for name, sha in manifest['output_sha256'].items():
        if digest(source/name) != sha:
            raise ValueError('parent catalogue file changed: '+name)
    if rule.get('append_sample'):
        from freeze_constgold_extension import freeze
        expanded = root/'input'
        freeze(source, expanded, total-previous_raw, rule['extension_sample_seed'])
        source = expanded
        manifest = json.loads((source/'image_mock_manifest.json').read_text())
    observations, truth = selected_prefix(pd.read_parquet(source/'measurements.parquet'),
        pd.read_parquet(source/'truth.parquet'), manifest, total)
    radius = observations.measured_flux_radius.to_numpy()
    ell = np.hypot(observations.measured_ngmix_g1, observations.measured_ngmix_g2).to_numpy()
    keep = (radius < 100) & (ell < .99)
    old_obs = pd.read_parquet(parent/'reference/observations.parquet')
    old_truth = pd.read_parquet(parent/'reference/truth.parquet')
    previous_count = len(old_obs)
    combined = observations.loc[keep, old_obs.columns].reset_index(drop=True)
    provenance = truth.loc[keep].reset_index(drop=True)
    validate_prefix(combined, provenance, old_obs, old_truth, previous_count, previous_raw)
    reference = root/'reference'
    reference.mkdir(exist_ok=False)
    combined.to_parquet(reference/'observations.parquet')
    provenance.to_parquet(reference/'truth.parquet')
    protocol = dict(old_protocol)
    protocol.update(source=str(source), source_manifest_sha256=digest(source/'image_mock_manifest.json'),
        selection=f'First {total} rows of the frozen uniform plus-leg sample, then '
        'measured radius <100 pixels and |e|<0.99; no truth-property cuts',
        comparison_reference=str(parent/'reference'), reused_c1_prefix=previous_count)
    save(reference/'protocol.json', protocol)
    drops = {}
    for name, sl in [('previous', slice(0, previous_raw)), ('additional', slice(previous_raw, total)),
                     ('combined', slice(0, total))]:
        drops[name] = dict(radius_ge_100px=int((radius[sl] >= 100).sum()),
            ellipticity_ge_099=int((ell[sl] >= .99).sum()), total=int((~keep[sl]).sum()))
    result = dict(status='prepared_image_reference', preparation_checks_pass=True,
        observation_count=len(combined), requested_rows=total, previous_count=previous_count,
        additional_count=len(combined)-previous_count, dropped_by_model_cuts=drops,
        observations_sha256=digest(reference/'observations.parquet'),
        truth_sha256=digest(reference/'truth.parquet'), protocol_sha256=digest(reference/'protocol.json'),
        cohorts_sha256=None, cases=int(provenance.source_case.nunique()),
        new_unmatched_rows=0, duplicate_catalogue_identities=0, old_new_overlap=0,
        parent_unmatched_rows='Retained in the hash-pinned source image_mock_manifest.json per_case entries',
        limitation='Same 100 ConstGold cases and known shear; not independent acceptance')
    save(reference/'result.json', result)
    mag = 30-2.5*np.log10(combined.measured_flux_from_mag_auto.to_numpy())
    ids = np.arange(len(combined))
    groups = {'bright': ids[mag < 23], 'faint': ids[mag >= 23], 'bright21': ids[mag < 21],
        'additional_bright': ids[(mag < 23) & (ids >= previous_count)],
        'additional_faint': ids[(mag >= 23) & (ids >= previous_count)]}
    for name, rows in groups.items():
        np.savetxt(root/f'{name}_ids.txt', rows, fmt='%d')
    (root/'c1/claims').mkdir()
    reused = {}
    reused_references = {}
    lines = []
    for line in (parent/'c1/units.txt').read_text().splitlines():
        label, file, first, last = line.split()
        if not (parent/'c1/claims'/label/'done').exists():
            raise ValueError('previous C1 incomplete: '+label)
        (root/'c1'/label).symlink_to(parent/'c1'/label)
        (root/'c1/claims'/label).symlink_to(parent/'c1/claims'/label)
        renamed = 'previous_'+file
        if not (root/renamed).exists():
            shutil.copy2(parent/file, root/renamed)
        lines.append(f'{label} {renamed} {first} {last}')
        unit_protocol = json.loads((parent/'c1'/label/'protocol.json').read_text())
        reused_references[label] = unit_protocol['reference']
        reused[label] = {name: digest(parent/'c1'/label/name) for name in
            ('protocol.json', 'result.json', 'local_profiles.npz')}
    old_units = len(lines)
    if old_units != rule['expected_previous_units']:
        raise ValueError('unexpected previous unit count; seed continuation needs review')
    suffix = f'_ext{old_units}' if rule.get('append_sample') else ''
    lines += units('bright'+suffix, 'additional_bright_ids.txt', len(groups['additional_bright']), 640)
    lines += units('faint'+suffix, 'additional_faint_ids.txt', len(groups['additional_faint']), 1024)
    if len({line.split()[0] for line in lines}) != len(lines):
        raise ValueError('duplicate unit labels')
    covered = []
    for line in lines:
        label, file, first, last = line.split()
        rows = np.loadtxt(root/file, dtype=int, ndmin=1)[int(first):int(last)+1]
        if not len(rows):
            raise ValueError('empty unit')
        if label.startswith('bright') and not (mag[rows] < 23).all():
            raise ValueError('bright unit includes a faint row')
        if label.startswith('faint') and not (mag[rows] >= 23).all():
            raise ValueError('faint unit includes a bright row')
        covered.extend(rows.tolist())
    if sorted(covered) != list(range(len(combined))):
        raise ValueError('combined C1 units do not cover each retained row exactly once')
    (root/'c1/units.txt').write_text('\n'.join(lines)+'\n')
    (root/'c2/units.txt').write_text('\n'.join(units('bright', 'bright21_ids.txt',
                                                    len(groups['bright21']), 61))+'\n')
    save(root/'reused_units.json', reused)
    save(root/'reused_references.json', reused_references)
    origin = Path(rule['estimator_origin'])
    estimate = (origin/'estimate.py').read_text()
    assert estimate.count(f"H = Path('{origin}')") == 1
    estimate = estimate.replace(f"H = Path('{origin}')", f"H = Path('{root}')")
    original = "        if protocol['reference'] != str(reference.resolve()):"
    replacement = ("        expected_reference = str(reference.resolve())\n"
        "        if stage == 'c1' and label in json.loads((H/'reused_units.json').read_text()):\n"
        "            expected_reference = json.loads((H/'reused_references.json').read_text())[label]\n"
        "        if (protocol['reference'] != expected_reference\n"
        "                or protocol['reference_protocol_sha256'] != hashlib.sha256(\n"
        "                    (Path(expected_reference)/'protocol.json').read_bytes()).hexdigest()):")
    assert estimate.count(original) == 1
    (root/'estimator_base.py').write_text(estimate.replace(original, replacement))
    report = dict(result, c1_reused_units=old_units, c1_new_units=len(lines)-old_units,
        c2_units=len((root/'c2/units.txt').read_text().splitlines()),
        bright21=len(groups['bright21']), additional_bright=len(groups['additional_bright']),
        additional_faint=len(groups['additional_faint']))
    save(root/'prepared.json', report)
    files = ['prepared.json', 'reused_units.json', 'reused_references.json', 'estimator_base.py', 'c1/units.txt', 'c2/units.txt']
    files += [str(p.relative_to(root)) for p in reference.iterdir() if p.is_file()]
    files += [p.name for p in root.glob('*_ids.txt')]
    if rule.get('append_sample'):
        files += ['input/image_mock_manifest.json', 'input/measurements.parquet', 'input/truth.parquet']
    save(root/'prepared_manifest.json', {name: digest(root/name) for name in files})
    verify(root)
    print('EXTENSION_PREPARED', json.dumps(report), flush=True)


def preflight(root):
    rule = json.loads((root/'rule.json').read_text())
    snapshot = root/'source_snapshot'
    smoke = root/'preflight'
    smoke.mkdir(exist_ok=False)
    (smoke/'source_snapshot').symlink_to(snapshot)
    for kind in ('bright', 'faint'):
        ids = (root/f'additional_{kind}_ids.txt').read_text().split()[:2]
        args = ['--betas', '1', '.7883', '.6167', '.4449', '.2849', '.1473', '.05014', '.005',
                '--narrow-step-probability', '.5'] if kind == 'bright' else []
        subprocess.run(['bash', str(snapshot/'jobs/job_galsbi_tempered.sh'), kind,
            '--reference', str(root/'reference'), '--shear', '0', '0', '--seed', '2028500001',
            '--initial-bank-seed', '2033500001', '--observation-indices', *ids,
            '--steps', '64', '--warmup', '32', '--profile-stride', '16', *args],
            env=dict(os.environ, SBSI_TEMPER_ROOT=str(smoke)), check=True)
        protocol = json.loads((smoke/kind/'protocol.json').read_text())
        result = json.loads((smoke/kind/'result.json').read_text())
        assert protocol['model_identity'] == rule['model_identity']
        assert protocol['observations'] == list(map(int, ids))
        assert len(result['observations']) == 2
    save(root/'preflight_passed.json', dict(status='integration_passed',
        limitation='64-step bright/faint code smokes, not convergence evidence'))


def estimate(root, stage):
    rule = json.loads((root/'rule.json').read_text())
    for label, files in json.loads((root/'reused_units.json').read_text()).items():
        for name, sha in files.items():
            if digest(root/'c1'/label/name) != sha:
                raise ValueError('reused sampler output changed: '+label+'/'+name)
    module = load_module('extension_estimator', root/'estimator_base.py')
    norm = Path(rule['normalization'])
    selection = load_module('preliminary_selection', norm/'study.py')
    module.selection = lambda: selection.load_selection(norm)
    module.__doc__ = (f"Combined {rule['requested_total_rows']} raw observations after cuts: "
        "reuse completed C1 chains and sample only new rows. Same C1/C2 formulas and bootstrap seeds. "
        "Reuse candidate-specific 10,010,624-scene normalization; selection uncertainty recomputed "
        "for the combined population. C2 reruns all mag<21 galaxies at the combined C1 shear.")
    sys.argv = [str(root/'estimator_base.py'), stage]
    module.main()
    path = root/stage/'estimate.json'
    result = json.loads(path.read_text())
    result['extension'] = dict(preparation=json.loads((root/'prepared.json').read_text()),
                               normalization=str(norm), rule_sha256=digest(root/'rule.json'))
    save(path, result)


def status(root):
    report = dict(stages={})
    for stage in ('c1', 'c2'):
        p = root/stage
        if not (p/'units.txt').exists():
            continue
        labels = [line.split()[0] for line in (p/'units.txt').read_text().splitlines()]
        report['stages'][stage] = dict(planned=len(labels),
            done=sum((p/'claims'/label/'done').exists() for label in labels),
            failed=[label for label in labels if (p/'claims'/label/'failed').exists()])
        if (p/'estimate.json').exists():
            report['stages'][stage]['estimate'] = json.loads((p/'estimate.json').read_text())
    save(root/f'logs/status_{os.environ.get("SLURM_JOB_ID", "manual")}.json', report)
    print(json.dumps(report), flush=True)


def setup(parent, root, total=500000, previous_raw=100000, append_sample=False, max_gpus=4):
    root.mkdir(exist_ok=False)
    for name in ('logs', 'c1', 'c2'):
        (root/name).mkdir()
    for name in ('source_snapshot', 'detect_resources.py'):
        (root/name).symlink_to(parent/name)
    for stage in ('c1', 'c2'):
        (root/stage/'source_snapshot').symlink_to(parent/'source_snapshot')
    parent_rule = json.loads((parent/'rule.json').read_text())
    norm = Path(parent_rule['normalization']) if append_sample else parent/'preliminary_10m'
    origin = Path(parent_rule['parent']) if append_sample else parent
    expected_previous_units = len((parent/'c1/units.txt').read_text().splitlines())
    identity = json.loads((parent/'reference/protocol.json').read_text())['model_identity']
    norm_rule = json.loads((norm/'rule.json').read_text())
    if norm_rule['model_identity'] != identity:
        raise ValueError('normalization model mismatch')
    save(root/'rule.json', dict(parent=str(parent), normalization=str(norm), model_identity=identity,
        requested_total_rows=total, new_raw_rows=[previous_raw, total], max_gpus=max_gpus,
        append_sample=append_sample, extension_sample_seed=20261028,
        expected_previous_units=expected_previous_units, estimator_origin=str(origin),
        c1_seed_base=2028500001, c1_bank_seed_base=2033500001,
        c2_seed_base=2028600001, c2_bank_seed_base=2033600001,
        c1_seeds=f'original unit line index retained; additional unit indices start at {expected_previous_units}',
        c2_policy='all combined mag<21 galaxies, fresh chains at combined C1 shear',
        selection='same measured cuts, prior, conditions, disk and model; reuse exactly 10,010,624 scenes',
        normalization_hashes={name: digest(norm/name) for name in ['rule.json', 'study.py',
            'normalization/complete.json', 'normalization/selection0.npz', 'normalization/selection1.npz',
            'normalization/selection0.json', 'normalization/selection1.json']},
        authorization=f'Owner: run another {total-previous_raw} observations; retain prior spare-GPU authorization',
        limitations='Same 100 simulation cases and known injected shear; no independent acceptance'))
    shutil.copy2(__file__, root/'study.py')
    if append_sample:
        for name in ('freeze_constgold_extension.py', 'prepare_constgold_inference_input.py'):
            shutil.copy2(Path(__file__).with_name(name), root/name)
    worker = (parent/'worker.sh').read_text()
    assert worker.count(f'H={parent}\n') == 1
    (root/'worker.sh').write_text(worker.replace(f'H={parent}\n', f'H={root}\n'))
    (root/'verify.py').write_text('import study\nfrom pathlib import Path\nstudy.verify(Path(__file__).resolve().parent)\n')
    repo = Path(__file__).resolve().parents[1]
    (root/'pipeline.sh').write_text((repo/'jobs/job_sign_symmetric_extension.sh').read_text().replace(
        '@STUDY_ROOT@', str(root)))
    names = ['rule.json', 'study.py', 'worker.sh', 'verify.py', 'pipeline.sh']
    if append_sample:
        names += ['freeze_constgold_extension.py', 'prepare_constgold_inference_input.py']
    save(root/'setup_manifest.json', {name: digest(root/name) for name in names})
    print(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'prepare', 'preflight', 'c1', 'c2', 'check'])
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--total-rows', type=int, default=500000)
    parser.add_argument('--previous-raw-rows', type=int, default=100000)
    parser.add_argument('--append-sample', action='store_true')
    parser.add_argument('--max-gpus', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'setup':
        setup(args.parent.resolve(), args.root.resolve(), args.total_rows, args.previous_raw_rows,
              args.append_sample, args.max_gpus)
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler required for catalogue and inference work')
    root = Path(__file__).resolve().parent
    if args.action == 'prepare':
        prepare(root)
    else:
        verify(root)
        if args.action == 'preflight':
            preflight(root)
        elif args.action == 'check':
            status(root)
        else:
            estimate(root, args.action)


if __name__ == '__main__':
    main()
