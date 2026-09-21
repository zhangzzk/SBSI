from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sbsi.fixed_g0_domain import FLOW_FEATURES
from scripts.scene_classifier_features import GEOMETRY_FEATURES, geometry_features
from scripts.scene_geometry_coordinates import major_convolved_scale
from scripts.transported_classifier_joint import CONDITIONS, state_geometry_frames, validate_coordinate_model


def test_state_coordinates_preserve_flow_frame_and_use_parent_identity():
    ids = np.array([9, 3])
    context = np.zeros((2, 8))
    context[:, 0], context[:, 4] = [.6, .3], [.4, .3]
    beta = CONDITIONS['moffat_beta']
    psf = CONDITIONS['psf_fwhm']/2*np.sqrt((2**(1/(beta-1))-1)/(2**(1/beta)-1))
    scale = major_convolved_scale(context[:, 4], context[:, :2], psf)
    pairs = pd.DataFrame({'index_input_p': [3, 9], 'index_input_s': [8, 7],
        'r_input_p_scaled': [25., 25.], 'r_input_s_scaled': [25., 25.],
        'distance_scaled': [.49, .49], 'Re_input_p_scaled': np.sqrt(1-(psf/scale[::-1])**2)})
    baseline, _ = geometry_features(ids, pairs)
    frame = pd.DataFrame(context, columns=FLOW_FEATURES).assign(
        **{name: baseline[:, i] for i, name in enumerate(GEOMETRY_FEATURES)})
    zero = frame.copy(deep=True)
    moved = frame.copy(deep=True)
    moved.iloc[:, :2] = 0.
    before = moved.copy(deep=True)
    result, report = state_geometry_frames({'zero': zero, 'moved': moved}, ids, context, pairs, CONDITIONS)
    pd.testing.assert_frame_equal(result['zero'], zero)
    pd.testing.assert_frame_equal(moved, before)
    pd.testing.assert_frame_equal(result['moved'][list(FLOW_FEATURES)], before[list(FLOW_FEATURES)])
    assert report['changed_geometry_parents'] == {'zero': 0, 'moved': 2}
    np.testing.assert_allclose(np.expm1(result['moved'][list(GEOMETRY_FEATURES)]), [[0, 1, 0, 0, 0]]*2, rtol=2e-7)


def test_static_or_smoke_checkpoints_cannot_silently_enter_full_validation():
    meta = {'geometry_coordinate_convention': 'state_major_axis',
            'prepared_format_version': 'scene_classifier_state_major_geometry_v2', 'response_used': False,
            'training_cases': [40], 'tune_cases': [100]}
    bundle = SimpleNamespace(members=[SimpleNamespace(metadata=meta)],
                             preprocessor=SimpleNamespace(feature_names=[*FLOW_FEATURES, *GEOMETRY_FEATURES]))
    validate_coordinate_model(bundle, operational=True)
    with pytest.raises(ValueError, match='full fixed'):
        validate_coordinate_model(bundle)
    meta['training_cases'], meta['tune_cases'] = list(range(40, 100)), list(range(100, 120))
    validate_coordinate_model(bundle)
    meta.pop('geometry_coordinate_convention')
    with pytest.raises(ValueError, match='explicit'):
        validate_coordinate_model(bundle)
