"""Six shared-draw classifier arms, adding one fixed smooth-crowding model."""
from pathlib import Path

import pandas as pd

from sbsi.selection_model import load_selection_model_ensemble
from scripts import transported_classifier_joint as original
from scripts.smooth_scene_classifier_features import FEATURES, FORMAT, augment_state
from scripts.train_full_parent_classifier import sha256

ARMS = (*original.ARMS, 'modeled_smooth')
classifier_frames = original.classifier_frames


def load_arms(reference, control, geometry, transported, smooth, device, operational=False):
    arms, provenance = original.load_arms(reference, control, geometry, transported, device)
    candidate = load_selection_model_ensemble([smooth], device=device)
    if len(candidate.members) != 1 or candidate.preprocessor.feature_names != list(FEATURES):
        raise ValueError('one seventeen-input smooth classifier required')
    meta = candidate.members[0].metadata
    if (meta.get('geometry_coordinate_convention') != 'state_major_axis'
            or meta.get('prepared_format_version') != FORMAT or meta.get('response_used') is not False):
        raise ValueError('explicit smooth training contract required')
    full = meta.get('training_cases') == list(range(40, 100)) and meta.get('tune_cases') == list(range(100, 120))
    smoke = meta.get('training_cases') == [40] and meta.get('tune_cases') == [100]
    if not full and not (operational and smoke):
        raise ValueError('fixed full training split required')
    arms['modeled_smooth'] = candidate
    provenance['modeled_smooth'] = [{'path': str(smooth), 'sha256': sha256(Path(smooth))}]
    return arms, provenance


def predict_arms(arms, frames, ids, context, pairs, conditions):
    old = {a: arms[a] for a in original.ARMS}
    probabilities, report = original.predict_arms(old, frames, ids, context, pairs, conditions)
    moved, _ = original.state_geometry_frames(frames, ids, context, pairs, conditions)
    smooth = {leg: pd.DataFrame(augment_state(ids, context, frame.to_numpy(), pairs,
                                              report['psf_half_light_radius']), columns=FEATURES)
              for leg, frame in moved.items()}
    probabilities.update(original.static.predict_arms({'modeled_smooth': arms['modeled_smooth']}, smooth))
    return probabilities, report
