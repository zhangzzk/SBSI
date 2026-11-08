"""Audit consistent-coordinate self and neighbour usability response on signed legs.

One scene and two probes give conditional probe uncertainty, not calibration
acceptance. Intrinsic-shape event moments are not measured-shape response or m.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.audit_classifier_measurement_transfer import predict
from scripts.audit_coherent_classifier_response import BIN_NAMES
from scripts.evaluate_matched_model_probe import prepare
from scripts.transported_classifier_joint import classifier_frames, load_arms, state_geometry_frames, ARMS
from scripts.train_full_parent_classifier import sha256, write_json


def event_moment(plus, minus, basis, h, signs=None):
    plus, minus, basis = (np.asarray(a, float) for a in (plus, minus, basis))
    if (plus.ndim != 1 or not len(plus) or minus.shape != plus.shape
            or basis.ndim != 2 or len(basis) != len(plus)
            or not all(np.isfinite(a).all() for a in (plus, minus, basis))
            or np.any((plus < 0) | (plus > 1) | (minus < 0) | (minus > 1))
            or not np.isfinite(h) or not 0 < h < 1):
        raise ValueError('finite aligned probabilities, basis and positive shear required')
    delta = (plus-minus)/(2*h)
    if signs is not None:
        signs = np.asarray(signs)
        if signs.shape != plus.shape or not np.isin(signs, [-1, 1]).all():
            raise ValueError('one Rademacher sign per parent required')
        delta *= signs
    return np.mean(delta[:, None]*basis, axis=0)


def summarize_events(events, context, signs):
    mag = context[:, 3]
    masks = [np.ones(len(mag), bool), mag < 24, (mag >= 24) & (mag < 25),
             (mag >= 25) & (mag < 25.8), mag >= 25.8]
    basis = np.column_stack((np.ones(len(mag)), context[:, :2]))
    if len(signs) != 2 or any(not m.any() for m in masks):
        raise ValueError('frozen two probes and nonempty magnitude bins required')
    result = {}
    for axis in (1, 2):
        coherent = np.stack([event_moment(events[f'g{axis}+'][m], events[f'g{axis}-'][m], basis[m], .02) for m in masks])
        probes = np.stack([np.stack([event_moment(events[f'p{p}g{axis}+'][m], events[f'p{p}g{axis}-'][m],
                             basis[m], .02, s[m]) for m in masks]) for p, s in enumerate(signs)])
        result[f'g{axis}'] = {'coherent': coherent, 'probes': probes, 'self': probes.mean(0),
                             'neighbour': coherent-probes.mean(0),
                             'conditional_probe_se': probes.std(0, ddof=1)/np.sqrt(len(probes))}
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('image-catalogue audit requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen input changed: '+path)
    torch.set_num_threads(4)
    case, ids, frames, _, usability, signs, _, pairs, pair_metadata, reports = prepare(SimpleNamespace(
        probe_root=Path(protocol['probe_root']), prepared_root=Path(protocol['prepared_root']),
        operational_small_scene=False))
    if case != protocol['case']:
        raise ValueError('prespecified matched case required')
    context = frames['zero'].to_numpy()
    geometry_frames, geometry_provenance = classifier_frames(frames, ids, context, case, protocol['geometry_root'])
    transported_frames, coordinate_report = state_geometry_frames(
        geometry_frames, ids, context, pairs, pair_metadata['conditions'])
    arms, provenance = load_arms(protocol['classifiers'], protocol['control'], protocol['geometry'],
                                 protocol['transported'], 'cpu')
    old = json.loads(Path(protocol['coherent_result']).read_text())
    reference = {f'g{r["specification"]["axis"]}': next(row for row in r['records'] if row['case'] == case)
                 for r in old['results'] if r['specification']['name'] in ('fresh_g1', 'fresh_g2')}
    observed = summarize_events(usability, context, signs)
    models, replay = {}, {}
    for arm, bundle in arms.items():
        source = (transported_frames if arm in ARMS[3:] else
                  geometry_frames if arm == 'modeled_geometry' else frames)
        probabilities = {key: np.exp(predict(bundle, frame.to_numpy())[:, 0]) for key, frame in source.items()}
        summary = summarize_events(probabilities, context, signs)
        models[arm], replay[arm] = {}, {}
        for axis in (1, 2):
            key = f'g{axis}'
            max_swap_error = 0.
            for p, s in enumerate(signs):
                for leg, opposite in (('+', '-'), ('-', '+')):
                    expected = np.where(s == 1, probabilities[key+leg], probabilities[key+opposite])
                    actual = probabilities[f'p{p}{key}{leg}']
                    max_swap_error = max(max_swap_error, float(np.max(np.abs(actual-expected))))
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
            previous = reference[key]['arms'][arm]['sufficient']
            expected = np.asarray(previous['response_sum'])/np.asarray(previous['n'])[:, None, None]
            np.testing.assert_allclose(summary[key]['coherent'], expected[:, 0], atol=2e-7, rtol=0)
            np.testing.assert_allclose(observed[key]['coherent'], expected[:, 1], atol=1e-12, rtol=0)
            delta = summary[key]['probes']-observed[key]['probes']
            coherent_error = summary[key]['coherent']-observed[key]['coherent']
            models[arm][key] = summary[key] | {'coherent_error': coherent_error,
                'self_error': delta.mean(0), 'neighbour_error': coherent_error-delta.mean(0),
                'error_conditional_probe_se': delta.std(0, ddof=1)/np.sqrt(len(signs))}
            replay[arm][key] = {'signed_probability_swap_max_error': max_swap_error,
                'coherent_probability_moment_max_error': float(np.max(np.abs(summary[key]['coherent']-expected[:, 0])))}
        print('SIGNED_CLASSIFIER_ARM_COMPLETE', arm, flush=True)
    previous_signed = json.loads(Path(protocol['signed_reference']).read_text())
    if previous_signed['case'] != case or previous_signed['parents'] != len(ids):
        raise ValueError('signed reference population mismatch')
    maximum_replay = 0.
    for arm in ARMS[:3]:
        for axis in ('g1', 'g2'):
            for field, old_value in previous_signed['models'][arm][axis].items():
                actual = np.asarray(models[arm][axis][field])
                expected = np.asarray(old_value)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
                maximum_replay = max(maximum_replay, float(np.max(np.abs(actual-expected))))
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen input changed during audit: '+path)
    write_json(args.output, {'status': 'completed', 'case': case, 'parents': len(ids), 'h': .02,
        'magnitude_bins': BIN_NAMES, 'basis': ['one', 'intrinsic_e1', 'intrinsic_e2'],
        'observed_U': observed, 'models': models, 'replay': replay, 'measurement_reports': reports,
        'classifier_provenance': provenance, 'geometry_provenance': geometry_provenance,
        'coordinate_report': coordinate_report, 'maximum_reference_difference': maximum_replay,
        'protocol_sha256': sha256(args.protocol), 'optimizer_steps': 0, 'calibration_claim': False,
        'limitations': ['one previously fixed scene, two independent source-sign probes; conditional probe SE only',
            'finite-h random-sign diagonal extraction can contain nonlinear mixed-shear terms',
            'U includes detection and usable fitting; raw D is not separately decomposed here',
            'intrinsic-shape event moments have no measured cut and are not measured response or m',
            'these classifier inputs depend on primary shape and fixed neighbour flux/distance, so neighbour shear response is zero by construction',
            'all five classifier arms retained; no model selection, empirical correction, flow or emulator evaluation']})
    print('SIGNED_CLASSIFIER_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
