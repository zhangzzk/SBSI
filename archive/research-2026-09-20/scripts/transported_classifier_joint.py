"""Coordinate-aware classifier arms for the frozen joint-draw evaluator."""
from pathlib import Path

import numpy as np

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.selection_model import load_selection_model_ensemble
from scripts import scene_classifier_joint as static
from scripts.scene_classifier_features import GEOMETRY_FEATURES, geometry_features
from scripts.scene_geometry_coordinates import transport_major_distance
from scripts.train_full_parent_classifier import sha256

ARMS = ('modeled_usable', 'modeled_control', 'modeled_geometry', 'modeled_coordinate', 'modeled_transported')
CONDITIONS = {'pixel_size': .2, 'zero_point': 30., 'psf_fwhm': .73, 'moffat_beta': 2.224, 'pixel_rms': .312}
classifier_frames = static.classifier_frames


def validate_coordinate_model(bundle, operational=False):
    if (len(bundle.members) != 1
            or bundle.preprocessor.feature_names != [*FLOW_FEATURES, *GEOMETRY_FEATURES]):
        raise ValueError('one coordinate-aware thirteen-input classifier required')
    meta = bundle.members[0].metadata
    if (meta.get('geometry_coordinate_convention') != 'state_major_axis'
            or meta.get('prepared_format_version') != 'scene_classifier_state_major_geometry_v2'
            or meta.get('response_used') is not False):
        raise ValueError('explicit state-major-axis training contract required')
    full = (meta.get('training_cases') == list(range(40, 100))
            and meta.get('tune_cases') == list(range(100, 120)))
    smoke = meta.get('training_cases') == [40] and meta.get('tune_cases') == [100]
    if not full and not (operational and smoke):
        raise ValueError('full fixed training split required outside operational smoke')


def load_arms(reference, control, geometry, transported, device, operational=False):
    arms, provenance = static.load_arms(reference, control, geometry, device)
    candidate = load_selection_model_ensemble([transported], device=device)
    validate_coordinate_model(candidate, operational)
    arms['modeled_coordinate'] = arms['modeled_geometry']
    arms['modeled_transported'] = candidate
    provenance['modeled_coordinate'] = provenance['modeled_geometry']
    provenance['modeled_transported'] = [{'path': str(transported), 'sha256': sha256(Path(transported))}]
    return arms, provenance


def state_geometry_frames(frames, ids, context, pairs, conditions):
    if conditions != CONDITIONS:
        raise ValueError('fixed audited observation conditions required')
    beta, fwhm = conditions['moffat_beta'], conditions['psf_fwhm']
    psf_radius = fwhm/2*np.sqrt((2**(1/(beta-1))-1)/(2**(1/beta)-1))
    baseline, _ = geometry_features(ids, pairs)
    result, changes = {}, {}
    for leg, frame in frames.items():
        if list(frame.columns) != [*FLOW_FEATURES, *GEOMETRY_FEATURES] or len(frame) != len(ids):
            raise ValueError('aligned static thirteen-column classifier frame required')
        np.testing.assert_array_equal(frame[list(GEOMETRY_FEATURES)], baseline)
        np.testing.assert_array_equal(frame[list(FLOW_FEATURES)[2:]], context[:, 2:])
        moved = transport_major_distance(ids, context, frame.iloc[:, :2].to_numpy(), pairs, psf_radius)
        geometry, _ = geometry_features(ids, moved)
        result[leg] = frame.assign(**{name: geometry[:, i] for i, name in enumerate(GEOMETRY_FEATURES)})
        changes[leg] = int(np.any(geometry != baseline, axis=1).sum())
    return result, {'coordinate_convention': 'state_major_axis', 'psf_half_light_radius': psf_radius,
                    'changed_geometry_parents': changes,
                    'counterfactual_arm': 'modeled_coordinate uses the original static-trained checkpoint',
                    'trained_arm': 'modeled_transported uses the new consistent-coordinate checkpoint'}


def predict_arms(arms, frames, ids, context, pairs, conditions):
    probabilities = static.predict_arms({name: arms[name] for name in ARMS[:3]}, frames)
    transported, report = state_geometry_frames(frames, ids, context, pairs, conditions)
    probabilities.update(static.predict_arms({name: arms[name] for name in ARMS[3:]}, transported))
    return probabilities, report
