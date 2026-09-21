"""Collect every frozen repeat without pooling endpoints or selecting a seed."""
import argparse
import json
import os
from pathlib import Path

from scripts.train_full_parent_classifier import sha256, write_json

PRIMARY = 'mag25.8_radius_gt0.60'
ENDPOINTS = ('main', 'independent', 'fresh_g1', 'fresh_g2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output_root.exists():
        raise RuntimeError('Slurm and new output directory required')
    files = {}
    def read(path):
        digest = sha256(path)
        value = json.loads(path.read_text())
        if sha256(path) != digest:
            raise ValueError('input changed while reading')
        files[str(path)] = digest
        return value
    manifest_path = args.root/'smooth_candidate_manifest.json'
    manifest = read(manifest_path)
    for path, expected in manifest['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen candidate artifact changed: '+path)
    if manifest['flow_count'] != 1 or manifest['classifiers_required'] != 1:
        raise ValueError('single flow and single required classifier expected')
    replica_root = args.root/'replica_results'
    density = {'original': read(args.root/'smooth_density_full.json'),
               'classifier_repeat': read(replica_root/'smooth_density_full.json')}
    influences = {'original': read(args.root/'smooth_full_influence.json'),
                  'classifier_repeat': read(replica_root/'smooth_full_influence.json')}
    endpoints = {}
    for endpoint in ENDPOINTS:
        result_paths = {'original': args.root/f'smooth_joint_{endpoint}.json',
                        'latent_repeat': args.root/f'smooth_joint_{endpoint}_seed7302.json',
                        'classifier_repeat': replica_root/f'smooth_joint_{endpoint}.json'}
        results = {name: read(path) for name, path in result_paths.items()}
        comparisons = {'latent_repeat': read(args.root/f'smooth_joint_{endpoint}_latent_comparison.json'),
                       'classifier_repeat': read(replica_root/f'smooth_training_repeat_{endpoint}.json')}
        for name, comparison in comparisons.items():
            if (comparison['baseline_sha256'] != files[str(result_paths['original'])]
                    or comparison['candidate_sha256'] != files[str(result_paths[name])]):
                raise ValueError('repeat comparison receipt mismatch')
        for name, root in (('original', args.root), ('classifier_repeat', replica_root)):
            receipt_path = root/f'smooth_joint_{endpoint}_comparison.json'
            receipt = read(receipt_path)
            if (receipt['result_sha256'] != files[str(result_paths[name])]
                    or receipt['maximum_reference_sufficient_difference'] != 0):
                raise ValueError('exact retained-arm replay required')
            label = {'main': 'main80', 'independent': 'independent40'}.get(endpoint, endpoint)
            if (density[name]['results'][label]['joint_sha256'] != files[str(result_paths[name])]
                    or density[name]['results'][label]['receipt_sha256'] != files[str(receipt_path)]
                    or influences[name]['endpoints'][endpoint]['result_sha256'] != files[str(result_paths[name])]):
                raise ValueError('density or influence refers to a different result')
        base = results['original']
        for result in results.values():
            for key in ('cases', 'h', 'shear_axis', 'population', 'cuts'):
                if base[key] != result[key]:
                    raise ValueError('repeat changes endpoint definition: '+key)
        endpoints[endpoint] = {
            'cases': base['cases'], 'h': base['h'], 'axis': base['shear_axis'],
            'all_branch_summaries': {name: result['summary'] for name, result in results.items()},
            'paired_changes': {name: comparison['comparisons'] for name, comparison in comparisons.items()},
            'density_scores': {name: {key: value['scores'] for key, value in d['results'][label]['comparison'].items()}
                               for name, d in density.items()},
            'influence': {name: value['endpoints'][endpoint]['influence'] for name, value in influences.items()}}
        print('REPRODUCIBILITY_REPORT_ENDPOINT', endpoint, flush=True)
    lines = ['# Frozen single-flow reproducibility evidence', '',
        'Primary selection: measured FLUX_RADIUS >0.6 arcsec and MAG_AUTO<25.8; generated secondary true-r<26 parent.', '',
        'Bias includes predicted detection/usability and flow-plus-emulator response. Values are m (%) ± case-bootstrap SE (percentage points).', '',
        '| Endpoint | Original classifier, latent7301 | Same model, latent7302 | Classifier training repeat, latent7301 |',
        '|---|---:|---:|---:|']
    for endpoint, e in endpoints.items():
        values = []
        for name in ('original', 'latent_repeat', 'classifier_repeat'):
            s = e['all_branch_summaries'][name][f'truth_parent/{PRIMARY}/modeled_smooth']
            values.append(f"{s['m_percent']:+.6f} ± {s['m_standard_error_percentage_points']:.6f}")
        lines.append(f"| {endpoint}, {len(e['cases'])} scenes, g{e['axis']}, h={e['h']} | "+' | '.join(values)+' |')
    lines += ['', 'All endpoints, cuts and classifier arms are retained in the JSON report. No endpoint pooling, empirical correction, model averaging or favorable-seed selection is performed.', '',
        'The training repeat changes the classifier seed only; flow and emulator training variation are not tested here. The four-scene axes share scenes. Main and independent samples use different shear amplitudes. Case errors omit finite-draw and training variation.', '',
        'Density scores are properly normalized selected parent/output comparisons against the historical classifier. Relative improvement alone does not prove absolute density adequacy. Cross-component diagnostics remain recorded separately and have lower priority under the owner’s guidance.', '',
        'This report collects evidence for scientific assessment; it does not automatically promote a model or assert calibration.']
    for path, expected in files.items():
        if sha256(Path(path)) != expected:
            raise ValueError('report input changed')
    args.output_root.mkdir(parents=True)
    write_json(args.output_root/'evidence.json', {'endpoints': endpoints, 'sources': files,
        'candidate_manifest_sha256': files[str(manifest_path)], 'script_sha256': sha256(Path(__file__)),
        'primary_selection': PRIMARY, 'flow_count': 1, 'classifiers_required': 1,
        'production_accepted': False, 'calibration_claim': False, 'original_candidate_retained': True})
    (args.output_root/'report.md').write_text('\n'.join(lines)+'\n')
    print('REPRODUCIBILITY_REPORT_COMPLETE', args.output_root, flush=True)


if __name__ == '__main__':
    main()
