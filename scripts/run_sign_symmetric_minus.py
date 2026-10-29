"""Independently selected minus leg coupled to the frozen million-row plus draw."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PY = '/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def load(path):
    spec = importlib.util.spec_from_file_location('study_dependency', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coupled_selection(minus_ids, plus_eligible, plus_sample, uniforms, probability):
    """Equal marginal inclusion q, without conditioning minus on plus selection."""
    import numpy as np
    if not 0 < probability <= 1 or len(uniforms) != len(minus_ids):
        raise ValueError('invalid inclusion design')
    if not np.isin(plus_sample, plus_eligible).all():
        raise ValueError('frozen plus sample outside replayed eligible population')
    shared = np.isin(minus_ids, plus_eligible)
    take = np.where(shared, np.isin(minus_ids, plus_sample), np.asarray(uniforms) < probability)
    return take, shared


def setup(plus, root):
    root.mkdir(exist_ok=False)
    for stage in ['c1', 'c2', 'logs']:
        (root/stage).mkdir()
    for stage in ['c1', 'c2']:
        (root/stage/'source_snapshot').symlink_to(plus/'source_snapshot')
        (root/stage/'claims').mkdir()
    (root/'source_snapshot').symlink_to(plus/'source_snapshot')
    rule = json.loads((plus/'rule.json').read_text())
    rule.update(plus_root=str(plus), minus_only_seed=2026102901, order_seed=2026102902,
        bootstrap_seed=2026102903, requested_total_rows='equal-probability minus sample; count determined by independent cuts',
        c1_seed_base=2028700001, c1_bank_seed_base=2033700001,
        c2_seed_base=2028800001, c2_bank_seed_base=2033800001,
        max_gpus=6, workers='four 32-CPU inter plus two 8-CPU cip A40-16GB for C1; four inter for C2',
        selection_design='q=1000000/18157963. Within plus-eligible IDs reuse frozen plus draw; '
            'minus-only eligible IDs use independent Bernoulli(q). Each leg applies its own measured cuts. '
            'Do not require detection or selection on both legs. Minus count is not fixed.',
        pairing='verify unsheared input identities/properties by case/index; shared-case uncertainty',
        plus_estimate_sha256=digest(plus/'c2/estimate.json'),
        paired_errors='2000 shared-case bootstrap draws and joint galaxy influence covariance; '
            'propagate shared selection-gradient covariance, report independent chain-noise floor; '
            'quote largest marginal error of these three, no convergence guarantee',
        authorization='Owner: run both shear legs; prior spare-GPU authorization retained')
    save(root/'rule.json', rule)
    shutil.copy2(__file__, root/'study.py')
    shutil.copy2(Path(__file__).with_name('prepare_constgold_inference_input.py'), root/'input_adapter.py')
    shutil.copy2(Path(__file__).with_name('paired_sign_symmetric_estimate.py'), root/'pair.py')
    shutil.copy2(plus/'detect_resources.py', root/'detect_resources.py')
    for name in ['worker.sh', 'pipeline.sh', 'estimator_base.py']:
        text = (plus/name).read_text()
        before = f"H = Path('{plus}')" if name.endswith('.py') else f'H={plus}\n'
        after = f"H = Path('{root}')" if name.endswith('.py') else f'H={root}\n'
        assert text.count(before) == 1
        text = text.replace(before, after)
        if name == 'pipeline.sh':
            text = text.replace('2028600001 2033600001', '2028800001 2033800001')
            text = text.replace('"$PY" -I -u "$H/study.py" c2\n',
                                '"$PY" -I -u "$H/study.py" c2\n    "$PY" -I -u "$H/study.py" pair\n')
        if name == 'estimator_base.py':
            # The numerical estimator is unchanged; uncertainty is positive for negative injected shear.
            text = text.replace('100*q[0]/t[0]', '100*q[0]/abs(t[0])')
        (root/name).write_text(text)
    original = Path(rule['recovery']['original_root'])
    text = (original/'cip_worker.sh').read_text().replace(f'H={original}\n', f'H={root}\n')
    text = text.replace('additional_faint_ids.txt', 'faint_ids.txt')
    (root/'cip_worker.sh').write_text(text)
    (root/'verify.py').write_text('from pathlib import Path\nimport study\nstudy.verify(Path(__file__).resolve().parent)\n')
    save(root/'reused_units.json', {})
    save(root/'reused_references.json', {})
    monitor = (plus/'monitor.py').read_text().replace("complete = (root/'c2/estimate.json').exists()",
                                                     "complete = (root/'paired_estimate.json').exists()")
    (root/'monitor.py').write_text(monitor)
    (root/'monitor.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec '+PY+' '+str(root/'monitor.py')+' '+str(root)+'\n')
    names = ['rule.json', 'study.py', 'input_adapter.py', 'pair.py', 'worker.sh', 'cip_worker.sh',
             'pipeline.sh', 'estimator_base.py', 'verify.py', 'reused_units.json', 'reused_references.json',
             'detect_resources.py', 'monitor.py', 'monitor.sh']
    save(root/'setup_manifest.json', {name:digest(root/name) for name in names})
    print('MINUS_SETUP_FROZEN', root, flush=True)


def verify(root, prepared=True):
    rule = json.loads((root/'rule.json').read_text())
    plus = Path(rule['plus_root'])
    if digest(plus/'c2/estimate.json') != rule['plus_estimate_sha256']:
        raise ValueError('plus result changed')
    for manifest in ['setup_manifest.json']+(['prepared_manifest.json'] if prepared else []):
        for name, sha in json.loads((root/manifest).read_text()).items():
            if digest(root/name) != sha:
                raise ValueError('minus frozen input changed: '+name)
    for name, sha in rule['normalization_hashes'].items():
        if digest(Path(rule['normalization'])/name) != sha:
            raise ValueError('normalization changed')
    subprocess.run(['sha256sum', '--check', 'SHA256SUMS', '--quiet'], cwd=root/'source_snapshot', check=True)
    print('MINUS_INPUTS_VERIFIED', flush=True)


def prepare(root):
    import numpy as np
    import pandas as pd
    verify(root, prepared=False)
    rule = json.loads((root/'rule.json').read_text())
    plus = Path(rule['plus_root'])
    subprocess.run([PY, str(plus/'verify.py')], check=True)
    source = Path(json.loads((plus/'reference/protocol.json').read_text())['source'])
    manifest = json.loads((source/'image_mock_manifest.json').read_text())
    for name, sha in manifest['output_sha256'].items():
        assert digest(source/name) == sha, name
    assert manifest['n_objects'] == 1000000 and manifest['n_eligible'] == 18157963
    raw = pd.read_parquet(source/'truth.parquet')
    measurements = pd.read_parquet(source/'measurements.parquet')
    adapter = load(root/'input_adapter.py')
    probability = manifest['n_objects']/manifest['n_eligible']
    chunks, reports = [], []
    kwargs = dict(pixel_size=.2, zero_point=30., primary_mag=None, primary_re=None,
                  measured_radius_min=.6, measured_mag_max=25.8, strict_radius=True, physical_disk_usability=True)
    identity_cols = ['index_input', 'RA_input', 'DEC_input', 'redshift_input', 'Re_input',
                     'axis_ratio_input', 'position_angle_input', 'sersic_n_input', 'r_input']
    for old in manifest['per_case']:
        case = old['case']
        catalogue = Path(old['sources']['truth']['path']).parents[4]
        plus_frame, plus_report = adapter.read_case(catalogue, case, **kwargs)
        assert plus_report == old, f'plus source changed: {case}'
        minus, minus_report = adapter.read_case(catalogue, case, shear=-.02, **kwargs)
        tp = pd.read_feather(old['sources']['truth']['path'], columns=identity_cols).sort_values('index_input').reset_index(drop=True)
        tm = pd.read_feather(minus_report['sources']['truth']['path'], columns=identity_cols).sort_values('index_input').reset_index(drop=True)
        pd.testing.assert_frame_equal(tp, tm, check_exact=True)
        case_raw = raw.loc[raw.source_case == case]
        indexed = plus_frame.set_index('source_input_index', drop=False)
        replay = indexed.loc[case_raw.source_input_index]
        assert np.array_equal(replay.source_detection_id.to_numpy(), case_raw.source_detection_id.to_numpy())
        pd.testing.assert_frame_equal(replay[measurements.columns].reset_index(drop=True),
                                      measurements.loc[case_raw.index].reset_index(drop=True), check_exact=True)
        uniform = np.random.default_rng([rule['minus_only_seed'], case]).random(len(minus))
        take, shared = coupled_selection(minus.source_input_index.to_numpy(), plus_frame.source_input_index.to_numpy(),
                                        case_raw.source_input_index.to_numpy(), uniform, probability)
        selected = minus.loc[take].copy()
        selected['shared_base_eligibility'] = shared[take]
        chunks.append(selected)
        reports.append(dict(case=case, minus=minus_report, input_properties_equal=True,
            shared_eligible=int(shared.sum()), minus_only_eligible=int((~shared).sum()),
            sampled_shared=int((take & shared).sum()), sampled_minus_only=int((take & ~shared).sum())))
        print('MINUS_CASE', case, 'sample', int(take.sum()), 'minus_only', int((take & ~shared).sum()), flush=True)
    selected = pd.concat(chunks, ignore_index=True)
    selected = selected.iloc[np.random.default_rng(rule['order_seed']).permutation(len(selected))].reset_index(drop=True)
    keep = (selected.measured_flux_radius.to_numpy() < 100) & (np.hypot(selected.measured_ngmix_g1, selected.measured_ngmix_g2) < .99)
    chosen = selected.loc[keep].reset_index(drop=True)
    assert not chosen.duplicated(['source_case', 'source_input_index']).any()
    reference = root/'reference'
    reference.mkdir()
    obs_columns = pd.read_parquet(plus/'reference/observations.parquet').columns
    chosen[obs_columns].to_parquet(reference/'observations.parquet', index=False)
    truth = chosen[['source_case', 'source_input_index', 'source_detection_id', 'shared_base_eligibility']].copy()
    truth['injected_g1'], truth['injected_g2'] = -.02, 0.
    truth.to_parquet(reference/'truth.parquet', index=False)
    selection_report = dict(inclusion_probability=probability, raw_sample=len(selected), retained=len(chosen),
        removed_final_cuts=int((~keep).sum()), minus_only_retained=int((~truth.shared_base_eligibility).sum()),
        selection_design=rule['selection_design'], frozen_plus_replay_exact=True, per_case=reports)
    plus_truth = pd.read_parquet(plus/'reference/truth.parquet', columns=['source_case', 'source_input_index'])
    selection_report['paired_selected'] = len(plus_truth.merge(truth, on=['source_case','source_input_index'], validate='one_to_one'))
    selection_report['plus_only_selected'] = len(plus_truth)-selection_report['paired_selected']
    selection_report['minus_only_selected'] = len(truth)-selection_report['paired_selected']
    save(root/'selection_report.json', selection_report)
    protocol = json.loads((plus/'reference/protocol.json').read_text())
    for key in ['reused_c1_prefix', 'source_manifest_sha256']:
        protocol.pop(key, None)
    protocol.update(source=str(root), selection=rule['selection_design'], comparison_reference=str(plus/'reference'),
                    source_manifest_sha256=digest(root/'selection_report.json'))
    save(reference/'protocol.json', protocol)
    result = dict(status='prepared_image_reference', preparation_checks_pass=True,
        observation_count=len(chosen), cases=int(truth.source_case.nunique()),
        observations_sha256=digest(reference/'observations.parquet'), truth_sha256=digest(reference/'truth.parquet'),
        protocol_sha256=digest(reference/'protocol.json'), dropped_by_model_cuts=int((~keep).sum()),
        selection_report_sha256=digest(root/'selection_report.json'), limitation='Known-shear development; paired independent selection')
    save(reference/'result.json', result)
    save(root/'prepared.json', result)
    mag = 30-2.5*np.log10(chosen.measured_flux_from_mag_auto.to_numpy())
    ids = np.arange(len(chosen))
    groups = dict(bright=ids[mag < 23], faint=ids[mag >= 23], bright21=ids[mag < 21])
    for name, values in groups.items():
        np.savetxt(root/(name+'_ids.txt'), values, fmt='%d')
    extension = load(Path(rule['recovery']['original_root'])/'study.py')
    lines = extension.units('bright', 'bright_ids.txt', len(groups['bright']), 640)
    lines += extension.units('faint', 'faint_ids.txt', len(groups['faint']), 1024)
    (root/'c1/units.txt').write_text('\n'.join(lines)+'\n')
    (root/'c2/units.txt').write_text('\n'.join(extension.units('bright', 'bright21_ids.txt', len(groups['bright21']), 61))+'\n')
    files = ['selection_report.json', 'prepared.json', 'c1/units.txt', 'c2/units.txt']
    files += [str(p.relative_to(root)) for p in reference.iterdir() if p.is_file()]
    files += [p.name for p in root.glob('*_ids.txt')]
    save(root/'prepared_manifest.json', {name:digest(root/name) for name in files})
    verify(root)
    print('MINUS_PREPARED', len(chosen), 'PAIRED', selection_report['paired_selected'], flush=True)


def preflight(root):
    rule = json.loads((root/'rule.json').read_text())
    out = root/'preflight'
    out.mkdir()
    (out/'source_snapshot').symlink_to(root/'source_snapshot')
    for label, group, shear in [('bright','bright',(0,0)), ('faint','faint',(0,0)), ('nonzero','bright',(-.019,.001))]:
        ids = (root/(group+'_ids.txt')).read_text().split()[:2]
        opts = ['--betas','1','.7883','.6167','.4449','.2849','.1473','.05014','.005',
                '--narrow-step-probability','.5'] if group == 'bright' else []
        subprocess.run(['bash', str(root/'source_snapshot/jobs/job_galsbi_tempered.sh'), label,
            '--reference',str(root/'reference'),'--shear',*map(str,shear),'--seed','2028700001',
            '--initial-bank-seed','2033700001','--observation-indices',*ids,
            '--steps','64','--warmup','32','--profile-stride','16',*opts],
            env=dict(os.environ,SBSI_TEMPER_ROOT=str(out)),check=True)
        p = json.loads((out/label/'protocol.json').read_text())
        a = json.loads((out/label/'invariant_replay.json').read_text())
        assert p['model_identity'] == rule['model_identity'] and p['nuisance_likelihood_cache'] is False
        assert a['likelihood_calls_skipped'] == 0 and a['checked_states'] > 0
    save(root/'preflight_passed.json', dict(status='passed', limitation='integration, not posterior convergence'))
    print('MINUS_PREFLIGHT_PASSED',flush=True)


def estimate(root, stage):
    rule = json.loads((root/'rule.json').read_text())
    module = load(root/'estimator_base.py')
    norm = Path(rule['normalization'])
    selection = load(norm/'study.py')
    module.selection = lambda: selection.load_selection(norm)
    module.__doc__ = (
        'Independently selected negative-shear image measurements. All minus chains are new, '
        'with full likelihood evaluation for every proposal. Unchanged C1/C2 estimator: '
        'C1 at zero; C2 reruns mag<21 at rounded C1 shear. Reuse candidate-specific '
        '10,010,624-scene normalization. Pair with frozen plus result after both legs finish.')
    sys.argv = [str(root/'estimator_base.py'), stage]
    module.main()
    path = root/stage/'estimate.json'
    result = json.loads(path.read_text())
    result['minus_run'] = dict(preparation=json.loads((root/'prepared.json').read_text()),
                              normalization=str(norm), rule_sha256=digest(root/'rule.json'))
    save(path, result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['setup','prepare','preflight','c1','c2','pair','check'])
    parser.add_argument('--plus',type=Path)
    parser.add_argument('--root',type=Path)
    args=parser.parse_args()
    if args.action=='setup':
        setup(args.plus.resolve(),args.root.resolve());return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler required')
    root=Path(__file__).resolve().parent
    if args.action=='prepare':
        prepare(root);return
    verify(root)
    rule=json.loads((root/'rule.json').read_text())
    if args.action=='preflight':
        preflight(root)
    elif args.action=='pair':
        load(root/'pair.py').run(root)
    else:
        original=load(Path(rule['recovery']['original_root'])/'study.py')
        if args.action=='check':
            original.status(root)
        else:
            estimate(root,args.action)


if __name__=='__main__':
    main()
