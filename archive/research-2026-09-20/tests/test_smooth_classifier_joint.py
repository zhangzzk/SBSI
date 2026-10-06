import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import smooth_classifier_joint as smooth
from scripts.scene_classifier_features import geometry_features
from scripts.scene_geometry_coordinates import major_convolved_scale
from scripts.smooth_scene_classifier_features import FEATURES


def test_new_arm_does_not_change_old_predictions_and_transports_features(monkeypatch):
    ids = np.array([9])
    context = np.array([[.6, 0., 1., 25., .4, 0., 0., 0.]])
    psf = .5
    scale = major_convolved_scale(context[:, 4], context[:, :2], psf)
    pairs = pd.DataFrame({'index_input_p': ids, 'index_input_s': [2], 'r_input_p_scaled': [25.],
                         'r_input_s_scaled': [25.], 'distance_scaled': [.49],
                         'Re_input_p_scaled': np.sqrt(1-(psf/scale)**2)})
    geometry, _ = geometry_features(ids, pairs)
    frame = pd.DataFrame(np.column_stack((context, geometry)), columns=FEATURES[:13])
    old = {a: {'plus': np.array([.7])} for a in smooth.original.ARMS}
    monkeypatch.setattr(smooth.original, 'predict_arms', lambda *args: (old.copy(), {'psf_half_light_radius': psf}))
    monkeypatch.setattr(smooth.original, 'state_geometry_frames', lambda *args: ({'plus': frame}, {}))
    seen = {}
    def predict(arms, frames):
        seen.update(frames)
        return {'modeled_smooth': {'plus': np.array([.8])}}
    monkeypatch.setattr(smooth.original.static, 'predict_arms', predict)
    result, _ = smooth.predict_arms(dict.fromkeys(smooth.ARMS), {'plus': frame}, ids, context, pairs, {})
    for a in smooth.original.ARMS:
        np.testing.assert_array_equal(result[a]['plus'], old[a]['plus'])
    assert list(seen['plus'].columns) == list(FEATURES)
    assert result['modeled_smooth']['plus'][0] == pytest.approx(.8)


def test_six_arm_evaluator_changes_only_classifier_wiring():
    # Scientific draw/extraction loop is the frozen evaluator, with only an
    # additional classifier arm. Verify an explicit transformation of its AST.
    old = Path('scripts/evaluate_transported_classifier_joint.py').read_text()
    expected = old.replace('transported_classifier_joint as scene_classifier', 'smooth_classifier_joint as scene_classifier')
    expected = expected.replace('"modeled_coordinate", "modeled_transported")', '"modeled_coordinate", "modeled_transported", "modeled_smooth")')
    expected = expected.replace('parser.add_argument("--classifier-transported", type=Path, required=True)',
        'parser.add_argument("--classifier-transported", type=Path, required=True)\n    parser.add_argument("--classifier-smooth", type=Path, required=True)')
    expected = expected.replace('args.classifier_geometry, args.classifier_transported,',
        'args.classifier_geometry, args.classifier_transported, args.classifier_smooth,')
    expected = expected.replace('five fixed classifier arms', 'six fixed classifier arms')
    actual = Path('scripts/evaluate_smooth_classifier_joint.py').read_text()
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
