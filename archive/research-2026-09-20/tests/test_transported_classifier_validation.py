from copy import deepcopy

import numpy as np

from scripts.validate_transported_classifier import paired_changes
from scripts.transported_classifier_joint import ARMS


def test_all_five_arms_have_paired_comparisons_with_shared_measured_population():
    result = {'cases': [1, 2], 'h': .02, 'cuts': ['uncut'], 'shear_axis': 2,
              'summary': {}, 'per_case_sufficient': {}}
    def legs(response):
        return {leg: {'denominator': [[10.], [30.]],
                      'numerator': [[[0., sign*.02*10*response]], [[0., sign*.02*30*response]]]}
                for leg, sign in [('plus', 1), ('minus', -1)]}
    for i, arm in enumerate(ARMS):
        response = 1+i*.01
        key = 'truth_parent/uncut/'+arm
        result['per_case_sufficient'][key] = {'measured': deepcopy(legs(1.)), 'model': legs(response)}
        result['summary'][key] = {'m_percent': 100*(1/response-1)}
    changes = paired_changes(result)['uncut']
    assert len(changes) == 10
    delta = changes['modeled_transported_minus_modeled_coordinate']
    np.testing.assert_allclose(delta['delta_m_percentage_points'], 100*(1/1.04-1/1.03))
    assert delta['paired_case_se_pp'] < 1e-12
