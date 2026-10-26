"""Identity-checked geometry and classifier arms for shared-draw joint validation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.selection_model import load_selection_model_ensemble
from scripts.scene_classifier_features import GEOMETRY_FEATURES
from scripts.train_full_parent_classifier import sha256


def load_arms(reference, control, geometry, device):
    paths = {'modeled_usable': list(reference), 'modeled_control': [control], 'modeled_geometry': [geometry]}
    arms = {name: load_selection_model_ensemble(value, device=device) for name, value in paths.items()}
    for name, arm in arms.items():
        expected = list(FLOW_FEATURES) + (list(GEOMETRY_FEATURES) if name == 'modeled_geometry' else [])
        if arm.preprocessor.feature_names != expected:
            raise ValueError('classifier feature contract mismatch: '+name)
    return arms, {name: [{'path': str(p), 'sha256': sha256(Path(p))} for p in value] for name, value in paths.items()}


def classifier_frames(frames, ids, context, case, root):
    root = Path(root)
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['features'] != [*FLOW_FEATURES, *GEOMETRY_FEATURES] or case not in manifest['cases']:
        raise ValueError('geometry manifest feature/case mismatch')
    path = root/f'cases/case{case:03d}.npz'
    digest = sha256(path)
    if digest != manifest['case_records'][str(case)]['prepared_sha256']:
        raise ValueError('prepared geometry changed')
    with np.load(path) as stored:
        geometry_ids, base, geometry = stored['ids'], stored['base_context'], stored['geometry']
    if len(np.unique(geometry_ids)) != len(geometry_ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('unique geometry and parent identities required')
    index = pd.Index(geometry_ids).get_indexer(ids)
    missing = int((index < 0).sum())
    unused = len(geometry_ids)-int((index >= 0).sum())
    if missing or unused:
        raise ValueError(f'geometry parent mismatch: missing={missing}, unused={unused}')
    np.testing.assert_allclose(base[index], context, atol=2e-6, rtol=2e-6)
    if geometry.shape != (len(ids), len(GEOMETRY_FEATURES)) or not np.isfinite(geometry).all():
        raise ValueError('finite aligned geometry required')
    result = {}
    for name, frame in frames.items():
        if list(frame.columns) != list(FLOW_FEATURES) or len(frame) != len(ids):
            raise ValueError('full-parent flow frame required')
        result[name] = frame.assign(**{feature: geometry[index, j] for j, feature in enumerate(GEOMETRY_FEATURES)})
    return result, {'manifest_sha256': sha256(manifest_path), 'case_sha256': digest,
                    'missing_parents': missing, 'unused_geometry_parents': unused}


def predict_arms(arms, frames):
    probabilities = {name: {leg: np.asarray(arm.predict_proba(frame), dtype=float)
                            for leg, frame in frames.items()} for name, arm in arms.items()}
    for directions in probabilities.values():
        for leg, probability in directions.items():
            if (probability.shape != (len(frames[leg]),) or not np.isfinite(probability).all()
                    or np.any((probability < 0) | (probability > 1))):
                raise ValueError('invalid modeled usability')
    return probabilities
