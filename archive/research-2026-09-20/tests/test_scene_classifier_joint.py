import json

import numpy as np
import pandas as pd
import pytest

from sbsi.fixed_g0_domain import FLOW_FEATURES
from scripts.scene_classifier_features import GEOMETRY_FEATURES
from scripts.scene_classifier_joint import classifier_frames, predict_arms
from scripts.train_full_parent_classifier import sha256


def fixture(root):
    (root/'cases').mkdir()
    context = np.arange(16, dtype=float).reshape(2, 8)/10
    geometry = np.arange(10, dtype=float).reshape(2, 5)
    path = root/'cases/case040.npz'
    np.savez(path, ids=[7, 3], base_context=context, geometry=geometry)
    manifest = {'features': [*FLOW_FEATURES, *GEOMETRY_FEATURES], 'cases': [40],
                'case_records': {'40': {'prepared_sha256': sha256(path)}}}
    (root/'manifest.json').write_text(json.dumps(manifest))
    return context, geometry, path


def test_identity_alignment_and_unchanged_flow_inputs(tmp_path):
    context, geometry, _ = fixture(tmp_path)
    original = pd.DataFrame(context[::-1], columns=FLOW_FEATURES)
    before = original.copy(deep=True)
    frames, report = classifier_frames({'plus': original, 'minus': original}, np.array([3, 7]), context[::-1], 40, tmp_path)
    pd.testing.assert_frame_equal(original, before)
    np.testing.assert_array_equal(frames['plus'][list(FLOW_FEATURES)], before)
    np.testing.assert_array_equal(frames['plus'][list(GEOMETRY_FEATURES)], geometry[::-1])
    assert report['missing_parents'] == report['unused_geometry_parents'] == 0


def test_reject_missing_identity_and_changed_cache(tmp_path):
    context, _, path = fixture(tmp_path)
    frame = pd.DataFrame(context, columns=FLOW_FEATURES)
    with pytest.raises(ValueError, match='missing=1, unused=1'):
        classifier_frames({'plus': frame}, np.array([7, 99]), context, 40, tmp_path)
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        classifier_frames({'plus': frame}, np.array([7, 3]), context, 40, tmp_path)


def test_nonfinite_or_out_of_range_classifier_probabilities_rejected():
    class Bad:
        def predict_proba(self, frame):
            return np.array([.5, np.nan])
    with pytest.raises(ValueError, match='invalid modeled'):
        predict_arms({'bad': Bad()}, {'plus': pd.DataFrame({'x': [1, 2]})})
