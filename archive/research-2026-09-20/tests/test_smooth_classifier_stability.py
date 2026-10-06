import numpy as np
import pytest

from scripts.smooth_classifier_joint import ARMS
from scripts.summarize_smooth_classifier_stability import influence_report


def example():
    result = {'cases': [10, 11], 'cuts': ['cut'], 'h': .02, 'shear_axis': 1,
              'per_case_sufficient': {}, 'summary': {}}
    def sufficient(values):
        v = np.asarray(values)
        return {leg: {'numerator': np.stack((sign*.02*v, np.zeros_like(v)), axis=-1)[:, None, :],
                      'denominator': np.ones((2, 1))} for leg, sign in [('plus', 1), ('minus', -1)]}
    for arm in ARMS:
        key = 'truth_parent/cut/'+arm
        result['per_case_sufficient'][key] = {'measured': sufficient([1, 2]), 'model': sufficient([2, 1])}
        result['summary'][key] = {'m_percent': 0., 'm_standard_error_percentage_points': 1.}
    return result


def test_all_six_arms_preserve_full_population_despite_influence():
    result = example()
    report = influence_report(result)['cut']
    assert set(report) == set(ARMS)
    for arm in ARMS:
        assert report[arm]['m_percent'] == pytest.approx(0)
        np.testing.assert_allclose(report[arm]['leave_one_out_m_percent'], [100, -50])
        assert report[arm]['all_cases_in_point_estimate']
        assert report[arm]['largest_influence_case'] == 10
    assert result['cases'] == [10, 11]


def test_stored_point_disagreement_is_rejected():
    result = example()
    result['summary']['truth_parent/cut/modeled_smooth']['m_percent'] = .1
    with pytest.raises(AssertionError):
        influence_report(result)
