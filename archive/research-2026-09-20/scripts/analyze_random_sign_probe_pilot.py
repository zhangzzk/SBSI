"""Measure same-scene random-sign self/neighbor response in an operational pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.matched_response_probe import baseline_reference, coherent_response, probe_response, summarize_probes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_leg(root, manifest, measured, group, h, ids):
    matches = [(leg, report) for leg, report in zip(manifest['legs'], measured['legs'])
               if leg['group'] == group and leg['h'] == h]
    if len(matches) != 1:
        raise ValueError('one matching shear leg required')
    leg, report = matches[0]
    path = Path(report['output'])
    if (digest(path) != report['output_sha256'] or digest(leg['catalogue']) != leg['catalogue_sha256']
            or (report['group'], report['h']) != (group, h)):
        raise ValueError('measurement or generated scene changed')
    cross_path = path.parent.parent/'CrossMatch/tile180.0_-0.5_rot0_matched.feather'
    if digest(cross_path) != report['cross_sha256']:
        raise ValueError('measurement crossmatch changed')
    cross, shapes = pd.read_feather(cross_path), pd.read_feather(path)
    if cross['id_input'].duplicated().any() or cross['id_detec'].duplicated().any() or shapes['NUMBER'].duplicated().any():
        raise ValueError('nonunique detections')
    matched = cross.set_index('id_input').reindex(ids)
    values = shapes.set_index('NUMBER').reindex(matched['id_detec'])
    e = values[['NGMIX_G1', 'NGMIX_G2']].to_numpy(float)
    radius, mag = values['FLUX_RADIUS'].to_numpy(float), values['MAG_AUTO'].to_numpy(float)
    usable = np.isfinite(e).all(1) & (np.square(e).sum(1) < 1) & np.isfinite(radius) & (radius > 0) & np.isfinite(mag)
    selected = usable & (radius > 3.) & (mag < 25.8)
    return e, selected, {'group': group, 'h': h, 'truth_parents': len(ids),
                        'without_detection': int(matched['id_detec'].isna().sum()),
                        'usable': int(usable.sum()), 'selected': int(selected.sum()),
                        'measurement_sha256': digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('catalogue diagnostics require Slurm')
    output = args.root/'response_diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((args.root/'manifest.json').read_text())
    measured = json.loads((args.root/'measurement_manifest.json').read_text())
    audit = json.loads((args.root/'render_validation.json').read_text())
    source_root = Path(manifest['source_root'])
    source = json.loads((source_root/'manifest.json').read_text())
    source_measured = json.loads((source_root/'measurement_manifest.json').read_text())
    full = manifest['format'] == 'random_sign_response_full_scene_development_v1'
    if full and (source.get('scope') != 'full_scene_development' or source['area_deg2'] != 1.
                 or source.get('reference_seed') != 123
                 or source.get('scene_count_rule') != 'exact original generated count after prior replay'):
        raise ValueError('full-area source with original-prior replay required')
    if (manifest['format'] not in ('random_sign_response_operational_v1', 'random_sign_response_full_scene_development_v1')
            or measured['status'] != 'completed'
            or manifest['source_manifest_sha256'] != digest(source_root/'manifest.json')
            or manifest['source_measurement_sha256'] != digest(source_root/'measurement_manifest.json')
            or measured['render_audit_sha256'] != digest(args.root/'render_validation.json')
            or audit['manifest_sha256'] != digest(args.root/'manifest.json')
            or digest(args.root/'probe_signs.npz') != manifest['signs_sha256']):
        raise ValueError('completed matched-probe receipts required')
    if measured['source_sha256'].keys() != source_measured['source_sha256'].keys():
        raise ValueError('probe and coherent fitting provenance keys differ')
    driver_changes = {}
    for path, expected in source_measured['source_sha256'].items():
        current = measured['source_sha256'][path]
        if current != expected and Path(path).name == 'measure_joint_multiaxis_pilot.py':
            driver_changes[path] = {'source_sha256': expected, 'current_sha256': current}
        elif current != expected:
            raise ValueError('probe and coherent core fitters differ')
    if driver_changes != manifest['measurement_driver_changes']:
        raise ValueError('undeclared outer measurement-driver change')
    generated = pd.read_feather(source['legs'][0]['catalogue'])
    ids = generated.loc[(generated['index'] >= len(generated)//2) & (generated['r'] < 26), 'index'].to_numpy(np.int64)
    # The pilot audit requires identical rendered source identities in every leg.
    # Reject missing truth parents instead of changing selection between probes.
    for leg in audit['legs']:
        if leg['unrendered_parent_ids']:
            raise ValueError('unrendered target parents; explicit revised population required')
    with np.load(args.root/'probe_signs.npz') as stored:
        np.testing.assert_array_equal(stored['ids'], generated['index'])
        signs = stored['signs'][:, ids]
    zero, w0, zero_report = load_leg(source_root, source, source_measured, 'g0', 0., ids)
    replay, replay_weights, replay_report = load_leg(args.root, manifest, measured, 'replay_g0', 0., ids)
    np.testing.assert_array_equal(replay, zero)
    np.testing.assert_array_equal(replay_weights, w0)
    if audit['zero_replay_image_and_psf_exact'] is not True:
        raise ValueError('exact zero-shear image replay required')
    mean, mass = baseline_reference(zero, w0)
    reports, results = [zero_report, replay_report], {}
    for axis in (1, 2):
        plus, wp, rp = load_leg(source_root, source, source_measured, f'g{axis}_h0.02', .02, ids)
        minus, wm, rm = load_leg(source_root, source, source_measured, f'g{axis}_h0.02', -.02, ids)
        coherent = coherent_response(plus, minus, wp, wm, mean_zero=mean, mass_zero=mass, h=.02)
        reports.extend([rp, rm])
        probes = []
        for probe in range(len(signs)):
            plus, wp, rp = load_leg(args.root, manifest, measured, f'probe{probe}_g{axis}', .02, ids)
            minus, wm, rm = load_leg(args.root, manifest, measured, f'probe{probe}_g{axis}', -.02, ids)
            probes.append(probe_response(plus, minus, wp, wm, parent_signs=signs[probe],
                                         mean_zero=mean, mass_zero=mass, h=.02))
            reports.extend([rp, rm])
        results[f'g{axis}'] = {'coherent': coherent, 'probe_estimates': probes,
                              **summarize_probes(probes, coherent['baseline_influence'])}
    result = {'case': source['case'], 'h': .02, 'primary_cut': 'measured radius>3px (0.6arcsec), MAG_AUTO<25.8',
              'baseline_mean': mean, 'baseline_mass': mass, 'truth_parents': len(ids), 'results': results,
              'reports': reports, 'zero_replay_shapes_and_selection_exact': True,
              'measurement_driver_changes': driver_changes,
              'calibration_claim': False, 'script_sha256': digest(__file__),
              'helper_sha256': digest(Path(__file__).with_name('matched_response_probe.py')),
              'probe_measurement_sha256': digest(args.root/'measurement_manifest.json'),
              'scope': 'full_scene_development' if full else 'small_area_operational',
              'limitations': [('single full-area historical-prior scene; development, not calibration acceptance'
                               if full else 'small operational scene with py31-generated prior; not full-area training prior'),
                              'two probes, one scene: conditional probe SE only; no case-based uncertainty',
                              'baseline-influence and exact endpoint ratio are distinct at finite shear',
                              'finite-step and noise-coupling errors not established by this diagnostic',
                              'no model comparison or calibration acceptance']}
    output.write_text(json.dumps(result, indent=2, allow_nan=False,
                                 default=lambda value: value.tolist())+'\n')
    print('MATCHED_PROBE_DIAGNOSTIC_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
