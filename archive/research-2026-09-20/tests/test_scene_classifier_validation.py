from copy import deepcopy

import numpy as np
import pytest

from scripts.validate_scene_classifier import paired_changes, replay_reference


def fixture():
    result = {'cases': [1, 2], 'h': .02, 'draws': 64, 'seed': 7301, 'population': 'fixture',
              'cohorts': ['truth_parent'], 'cuts': ['uncut'], 'case_reports': {}, 'shear_axis': 1,
              'inputs': {k: k for k in ['parent_manifest_sha256', 'flow_sha256', 'classifiers', 'rblend']},
              'summary': {}, 'per_case_sufficient': {}}
    def legs(response):
        return {leg: {'denominator': [[10.], [30.]],
                      'numerator': [[[sign*.02*10*response, 0.]], [[sign*.02*30*response, 0.]]]}
                for leg, sign in [('plus', 1), ('minus', -1)]}
    for arm, value in [('modeled_usable', 1.), ('modeled_control', 1.01), ('modeled_geometry', 1.02)]:
        key = 'truth_parent/uncut/'+arm
        result['per_case_sufficient'][key] = {'measured': legs(1.), 'model': legs(value)}
        result['summary'][key] = {'measured_response': [1., 0.], 'model_response': [value, 0.],
            'model_flow_response': [.8*value, 0.], 'model_blend_response': [.2*value, 0.],
            'm_percent': 100*(1/value-1)}
    return result


def test_reference_replay_rejects_changed_response():
    baseline = fixture()
    candidate = deepcopy(baseline)
    assert replay_reference(baseline, candidate) == 0.
    candidate['per_case_sufficient']['truth_parent/uncut/modeled_usable']['model']['plus']['numerator'][0][0][0] += .01
    with pytest.raises(AssertionError):
        replay_reference(baseline, candidate)


def test_paired_arm_changes_constant_response_ratio():
    result = paired_changes(fixture())['uncut']['modeled_geometry_minus_modeled_control']
    np.testing.assert_allclose(result['delta_m_percentage_points'], 100*(1/1.02-1/1.01))
    assert result['paired_case_se_pp'] < 1e-12
