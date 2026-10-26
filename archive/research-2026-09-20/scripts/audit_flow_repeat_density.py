"""Compare transported flow densities with fixed predicted-U selection weights."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from plots.constgold_property_bias import load_lookup, make_case_weights
from sbsi.disk_response_transport import transported_physical_log_prob
from sbsi.measurement_model import load_measurement_model
from sbsi.output_conditioned_response import DiskResponseMoment, predict_scene_draw_responses
from scripts.compare_truth_parent_response import require_comparable
from scripts.evaluate_smooth_classifier_joint import CUTS, load_candidate_pairs, measured_leg, output_cuts, raw_root
from scripts.joint_response_axes import shear_frames, shear_vector
from scripts.normalize_selected_density_comparison import score_change
from scripts.smooth_classifier_joint import ARMS
from scripts.train_full_parent_classifier import sha256, write_json


def require_fixed_likelihood_parts(reference, candidate):
    """Classifier factors cancel only when every input except the flow is fixed."""
    def fixed_inputs(result):
        return {key: value for key, value in result['inputs'].items() if key not in ('flow', 'flow_sha256')}
    if (fixed_inputs(reference) != fixed_inputs(candidate)
            or reference['experimental_response']['metadata'] != candidate['experimental_response']['metadata']):
        raise ValueError('only the flow may change for this density ratio')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    protocol = json.loads(args.protocol.read_text())
    def verify():
        for path, expected in protocol['files'].items():
            if sha256(Path(path)) != expected:
                raise ValueError('frozen density input changed: '+path)
    verify()
    joint_protocol_path = Path(protocol['joint_protocol'])
    joint_protocol = json.loads(joint_protocol_path.read_text())
    if joint_protocol['template_sha256'] != sha256(Path(protocol['joint_template'])):
        raise ValueError('joint protocol came from another frozen template')
    for path, expected in joint_protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('joint input changed: '+path)
    endpoint = joint_protocol['endpoints'][args.index]
    output = Path(protocol['output_root'])/f"density_{endpoint['name']}.json"
    if output.exists():
        raise FileExistsError(output)
    paths = [Path(endpoint[key]) for key in ('baseline', 'output', 'comparison', 'receipt')]
    sources = {str(p): sha256(p) for p in (*paths, joint_protocol_path)}
    reference, candidate, comparison, receipt = [json.loads(p.read_text()) for p in paths]
    if (receipt['protocol_sha256'] != sources[str(joint_protocol_path)]
            or receipt['result_sha256'] != sources[str(paths[1])]
            or receipt['comparison_sha256'] != sources[str(paths[2])]
            or comparison['baseline_sha256'] != sources[str(paths[0])]
            or comparison['candidate_sha256'] != sources[str(paths[1])]):
        raise ValueError('completed matching response comparisons required')
    require_comparable(reference, candidate)
    require_fixed_likelihood_parts(reference, candidate)
    arguments = endpoint['arguments']
    def argument(name):
        return arguments[arguments.index(name)+1]
    parent_root = Path(argument('--parent-root'))
    pair_root = Path(argument('--pair-root'))
    raw = Path(argument('--constgold-root'))
    cases, h, axis = candidate['cases'], candidate['h'], candidate['shear_axis']
    parent_manifest = json.loads((parent_root/'manifest.json').read_text())
    if sha256(parent_root/'manifest.json') != candidate['inputs']['parent_manifest_sha256']:
        raise ValueError('parent manifest changed')
    lookup_paths = tuple(Path(arguments[i+1]) for i, v in enumerate(arguments) if v == '--rblend')
    lookup = load_lookup(lookup_paths, tuple(cases))
    emulator = DiskResponseMoment(Path(argument('--response-candidate')), nthread=4, device='cuda')
    if emulator.metadata != candidate['experimental_response']['metadata']:
        raise ValueError('emulator mismatch')
    bundles = {name: load_measurement_model(Path(result['inputs']['flow']), device=torch.device('cuda'))
               for name, result in (('reference', reference), ('candidate', candidate))}
    for bundle in bundles.values():
        bundle.model.eval()
    records = []
    for ci, case in enumerate(cases):
        parent_path = parent_root/f'case{case:03d}.npz'
        if sha256(parent_path) != parent_manifest['case_reports'][str(case)]['sha256']:
            raise ValueError('parent changed')
        with np.load(parent_path, allow_pickle=False) as stored:
            ids, context = stored['input_index'], stored['context'].astype(float)
        np.testing.assert_array_equal(ids, lookup[case]['input_index'])
        pairs, _ = load_candidate_pairs(pair_root, case, parent_path, ids, lookup[case]['R_blend'])
        frames = shear_frames(context, h, axis)
        for direction, sign, frame in zip(('plus', 'minus'), (1, -1), frames):
            observed, usable, measured_receipt = measured_leg(raw, case, sign*h, ids)
            if measured_receipt != candidate['case_reports'][str(case)]['legs'][direction]:
                raise ValueError('measurement receipt mismatch')
            values = torch.as_tensor(observed[usable], dtype=torch.float64, device='cuda')
            kept_pairs = pairs.loc[pairs['index_input_p'].isin(ids[usable])]
            response, counts = predict_scene_draw_responses(emulator, kept_pairs, ids[usable], observed[usable, None, :])
            velocity = torch.as_tensor(response[:, 0], device='cuda')[:, None]*values.new_tensor(shear_vector(sign*h, axis))
            nll = {}
            for name, bundle in bundles.items():
                means = values.new_tensor(bundle.target_transform.means)
                scales = values.new_tensor(bundle.target_transform.scales)
                conditions = bundle.context_tensor(frame.loc[usable])
                def log_prob(original):
                    return bundle.log_prob_tensor((original-means)/scales, conditions, batch_size=8192)-scales.log().sum()
                lp = transported_physical_log_prob(values, velocity, log_prob)
                if not bool(torch.isfinite(lp).all()):
                    raise ValueError('nonfinite physical density; no row dropping allowed')
                nll[name] = -lp.cpu().numpy()
            cuts = output_cuts(observed[usable])
            entries = {}
            for k, cut in enumerate(CUTS):
                keep = cuts[:, k]
                entry = candidate['per_case_sufficient'][f'truth_parent/{cut}/modeled_smooth']['measured'][direction]
                np.testing.assert_array_equal(entry['denominator'][ci], [keep.sum()])
                np.testing.assert_allclose(entry['numerator'][ci], observed[usable][keep, :2].sum(0)[None], atol=1e-11)
                entries[cut] = {'count': int(keep.sum()), 'raw_delta_nll': float((nll['candidate'][keep]-nll['reference'][keep]).mean()),
                                'reference_physical_nll': float(nll['reference'][keep].mean()),
                                'candidate_physical_nll': float(nll['candidate'][keep].mean())}
            catalogue = raw_root(raw, case, sign*h)
            for suffix in ('CrossMatch/tile180.0_-0.5_rot0_matched.feather', 'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather'):
                path = catalogue/suffix
                sources[str(path)] = sha256(path)
            records.append({'case': case, 'leg': direction, 'cuts': entries, 'prediction_counts': counts})
            print('FLOW_REPEAT_DENSITY_CASE', endpoint['name'], case, direction, flush=True)
    weights = np.concatenate((np.ones((1, len(cases))), make_case_weights(2000, len(cases), 20260919)))
    summaries = {}
    for direction in ('plus', 'minus'):
        rows = [r for r in records if r['leg'] == direction]
        for cut in CUTS:
            count = np.asarray([r['cuts'][cut]['count'] for r in rows])
            delta = np.asarray([r['cuts'][cut]['raw_delta_nll'] for r in rows])
            for arm in (*ARMS, 'actual_usable_flags'):
                key = f'truth_parent/{cut}/{arm}'
                masses = []
                for result in (reference, candidate):
                    stat = result['per_case_sufficient'][key]
                    np.testing.assert_array_equal(count, np.asarray(stat['measured'][direction]['denominator'])[:, 0])
                    masses.append(np.asarray(stat['model'][direction]['denominator'])[:, 0])
                changes = score_change(count, delta, *masses, weights)
                summaries[f'{direction}/{cut}/{arm}'] = {name: {'estimate': value[0],
                    'case_bootstrap_se': value[1:].std(ddof=1), 'case_bootstrap_ci95': np.quantile(value[1:], [.025, .975])}
                    for name, value in changes.items()}
    verify()
    for path, expected in sources.items():
        if sha256(Path(path)) != expected:
            raise ValueError('density source changed during evaluation')
    write_json(output, {'endpoint': endpoint['name'], 'cases': cases, 'axis': axis, 'h': h,
        'records': records, 'summary': summaries, 'sources': sources, 'protocol_sha256': sha256(args.protocol),
        'script_sha256': sha256(Path(__file__)), 'calibration_claim': False,
        'formula': 'mean observed physical delta NLL + log(candidate selection mass/reference selection mass)',
        'limitations': ['fixed classifier probability factors cancel in raw score; their weighting remains in each model selection mass',
                       'pooled parent-and-output selected law; relative score does not establish absolute adequacy',
                       'case bootstrap excludes finite-draw mass and training uncertainty',
                       'all six predicted-U arms, actual-U oracle and three cuts retained; matched-U is not this per-leg observed density estimand']})
    print('FLOW_REPEAT_DENSITY_COMPLETE', endpoint['name'], flush=True)


if __name__ == '__main__':
    main()
