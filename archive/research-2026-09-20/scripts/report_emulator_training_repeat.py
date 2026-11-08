"""Append the emulator repeat to the complete single-flow reproducibility evidence."""
import argparse
import json
import os
from pathlib import Path

from scripts.train_full_parent_classifier import sha256, write_json

ENDPOINTS = ('main', 'independent', 'fresh_g1', 'fresh_g2')
PRIMARY = 'mag25.8_radius_gt0.60'
KEY = f'truth_parent/{PRIMARY}/modeled_smooth'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output_root.exists():
        raise RuntimeError('Slurm and new output directory required')
    sources = {}

    def read(path):
        path = Path(path)
        digest = sha256(path)
        data = json.loads(path.read_text())
        if sha256(path) != digest:
            raise ValueError('input changed while reading: '+str(path))
        sources[str(path)] = digest
        return data

    def verify_files(files):
        for path, expected in files.items():
            if sha256(Path(path)) != expected:
                raise ValueError('source identity mismatch: '+path)
            if path in sources and sources[path] != expected:
                raise ValueError('conflicting source identity: '+path)
            sources[path] = expected

    previous = read(args.root/'flow_refinement_repeat_report/evidence.json')
    absolute = read(args.root/'absolute_density_report/evidence.json')
    joint_path = args.root/'emulator_repeat_joint_protocol.json'
    joint = read(joint_path)
    density_protocol_path = args.root/'emulator_repeat_density_protocol.json'
    density_protocol = read(density_protocol_path)
    for record, field in ((previous, 'sources'), (absolute, 'sources'),
                          (joint, 'files'), (density_protocol, 'files')):
        verify_files(record[field])
    if (Path(density_protocol['joint_protocol']) != joint_path
            or joint['template_sha256'] != sha256(Path(density_protocol['joint_template']))):
        raise ValueError('density and joint protocols are not linked')
    if (set(previous['endpoints']) != set(ENDPOINTS)
            or {e['name'] for e in joint['endpoints']} != set(ENDPOINTS)
            or len(joint['endpoints']) != len(ENDPOINTS)
            or previous['flow_count'] != 1 or previous['classifiers_required'] != 1):
        raise ValueError('complete single-flow single-classifier evidence required')
    result = {}
    for endpoint in ENDPOINTS:
        e = next(e for e in joint['endpoints'] if e['name'] == endpoint)
        comparison = read(e['comparison'])
        receipt = read(e['receipt'])
        density = read(Path(density_protocol['output_root'])/f'density_{endpoint}.json')
        expected_sources = {e['baseline']: comparison['baseline_sha256'],
                            e['output']: comparison['candidate_sha256']}
        verify_files(expected_sources)
        if (receipt['endpoint'] != endpoint or density['endpoint'] != endpoint
                or receipt['protocol_sha256'] != sources[str(joint_path)]
                or receipt['comparison_sha256'] != sources[e['comparison']]
                or receipt['result_sha256'] != sources[e['output']]
                or receipt['emulator_sha256'] != joint['repeated_emulator_sha256']
                or density['protocol_sha256'] != sources[str(density_protocol_path)]):
            raise ValueError('emulator-repeat identity mismatch')
        for path in (e['baseline'], e['output'], e['comparison'], e['receipt'], str(joint_path)):
            if density['sources'][path] != sources[path]:
                raise ValueError('density uses a different response artifact')
        verify_files(density['sources'])
        old = previous['endpoints'][endpoint]
        if any(density[k] != old[k] for k in ('cases', 'h', 'axis')):
            raise ValueError('repeat changes scene or shear definition')
        branches = old['all_branch_summaries']['original']
        if set(comparison['comparisons']) != set(branches):
            raise ValueError('emulator comparison does not retain every branch')
        for key, branch in comparison['comparisons'].items():
            if branch['baseline'] != branches[key]:
                raise ValueError('baseline differs from completed reproducibility report')
        expected_density_keys = {
            f'{leg}/{cut}/{arm}' for leg in ('plus', 'minus')
            for cut in ('uncut', PRIMARY, 'mag25.8_radius_ge0.75')
            for arm in ('modeled_usable', 'modeled_control', 'modeled_geometry',
                        'modeled_coordinate', 'modeled_transported', 'modeled_smooth', 'actual_usable_flags')}
        if set(density['summary']) != expected_density_keys:
            raise ValueError('density omits or changes a declared arm/cut/leg')
        result[endpoint] = {**old,
            'all_branch_summaries': {**old['all_branch_summaries'],
                'emulator_training_repeat': {key: value['candidate'] for key, value in comparison['comparisons'].items()}},
            'paired_changes': {**old['paired_changes'], 'emulator_training_repeat': comparison['comparisons']},
            'emulator_repeat_density': density['summary']}
        print('EMULATOR_REPEAT_REPORT_ENDPOINT', endpoint, flush=True)
    lines = ['# Complete single-flow calibration and density repeat evidence', '',
        'Original candidate retained: one flow, one smooth detection/usability classifier and one disk-response emulator. '
        'Measured FLUX_RADIUS >0.6 arcsec, MAG_AUTO<25.8; generated secondary true-r<26 parent.', '',
        'Values are diagonal joint m (%) ± case-bootstrap standard error (percentage points).', '',
        '| Endpoint | Original | Latent repeat | Classifier training repeat | Flow-refinement repeat | Emulator training repeat |',
        '|---|---:|---:|---:|---:|---:|']
    for endpoint, e in result.items():
        values = []
        for variant in ('original', 'latent_repeat', 'classifier_repeat', 'flow_refinement_repeat', 'emulator_training_repeat'):
            s = e['all_branch_summaries'][variant][KEY]
            values.append(f"{s['m_percent']:+.6f} ± {s['m_standard_error_percentage_points']:.6f}")
        lines.append(f"| {endpoint}, {len(e['cases'])} scenes, g{e['axis']}, h={e['h']} | "+' | '.join(values)+' |')
    lines += ['', '## Paired changes from the original', '',
        'Values are percentage-point shifts ± paired case-bootstrap SE. They do not estimate the distribution over training seeds.', '',
        '| Endpoint | Latent repeat | Classifier training repeat | Flow-refinement repeat | Emulator training repeat |', '|---|---:|---:|---:|---:|']
    for endpoint, e in result.items():
        values = []
        for variant in ('latent_repeat', 'classifier_repeat', 'flow_refinement_repeat', 'emulator_training_repeat'):
            s = e['paired_changes'][variant][KEY]
            values.append(f"{s['delta_m_percentage_points']:+.6f} ± {s['paired_delta_se_pp']:.6f}")
        lines.append(f'| {endpoint} | '+' | '.join(values)+' |')
    lines += ['', '## Emulator-repeat density change', '',
        'Properly normalized predicted-U selected parent-and-output NLL: repeated emulator minus original emulator '
        '(nat/object ± case-bootstrap SE). Negative values favor the repeat. The original remains retained regardless of this result.', '',
        '| Endpoint | Plus leg | Minus leg |', '|---|---:|---:|']
    for endpoint, e in result.items():
        values = []
        for leg in ('plus', 'minus'):
            s = e['emulator_repeat_density'][f'{leg}/{PRIMARY}/modeled_smooth']['normalized_delta_nll']
            values.append(f"{s['estimate']:+.8f} ± {s['case_bootstrap_se']:.8f}")
        lines.append(f'| {endpoint} | '+' | '.join(values)+' |')
    lines += ['', '## Scope of this evidence', '',
        '- No endpoint pooling, seed selection, flow ensemble or empirical response correction. All arms and cuts remain in evidence.json.',
        '- The two large samples have different scenes and shear amplitudes; the two small axis tests share four scenes.',
        '- The flow repeat changes refinement random streams, conditional on the same initial checkpoint. Independent full flow pretraining is not tested; the emulator repeat changes its fitting seed on the same response labels.',
        '- Each repeat changes one source of randomness at a time. Interactions among jointly changed training seeds are not tested.',
        '- These are development samples, not a newly untouched acceptance set. Case errors exclude training and finite-integration uncertainty.',
        '- Absolute density diagnostics remain nonzero: primary shape variances are underpredicted by roughly 1.3–2.6%; uncut coarse 4D histogram TV is roughly 3.5–3.9%, including sampling noise. Full original diagnostics are included.',
        '- Predicted usability covers combined detection and usable measurement. Cross-component diagnostics remain recorded separately at lower priority.',
        '- This report assembles evidence for assessment; it does not automatically accept a model or change the configured likelihood.']
    verify_files(dict(sources))
    args.output_root.mkdir(parents=True)
    (args.output_root/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(args.output_root/'evidence.json', {'endpoints': result, 'sources': sources,
        'absolute_density': absolute, 'emulator_training_seed': 20260817,
        'repeated_emulator_sha256': joint['repeated_emulator_sha256'],
        'script_sha256': sha256(Path(__file__)), 'original_candidate_retained': True,
        'flow_count': 1, 'classifiers_required': 1, 'calibration_claim': False, 'production_accepted': False})
    print('EMULATOR_REPEAT_REPORT_COMPLETE', args.output_root, flush=True)


if __name__ == '__main__':
    main()
