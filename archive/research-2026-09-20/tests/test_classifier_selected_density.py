from copy import deepcopy

import numpy as np
import pytest

from scripts.audit_classifier_selected_density import compare_endpoint, selected_statistics
from scripts.evaluate_constgold_truth_parent import CUTS
from scripts.transported_classifier_joint import ARMS


def test_selected_log_probability_uses_usable_and_strict_measured_radius():
    values = np.array([[0, 0, 3, 100], [0, 0, 3.1, 100],
                       [0, 0, 4, 100], [0, 0, 4, 100]], float)
    out = selected_statistics(np.log([.2, .3, .4, .5]), values, np.array([True, True, False, True]))
    assert [out[c]['count'] for c in CUTS] == [3, 2, 1]
    assert out[CUTS[1]]['negative_log_usable_sum'] == pytest.approx(-np.log(.3*.5))


def fixture():
    records, summaries = [], {}
    probabilities = dict(zip(ARMS, [.2, .3, .4, .5, .6]))
    counts = [4, 7]
    measurement = {'parent_rows': 20}
    for case, count in zip([10, 11], counts):
        arms = {a: {'cuts': {c: {'count': count, 'negative_log_usable_sum': -count*np.log(p)}
                            for c in CUTS}} for a, p in probabilities.items()}
        records.append({'case': case, 'legs': {leg: {'arms': arms, 'measurement': measurement} for leg in ('plus', 'minus')}})
    for a, p in probabilities.items():
        for cut in CUTS:
            summaries['truth_parent/'+cut+'/'+a] = {
                'measured': {leg: {'denominator': np.array(counts)[:, None]} for leg in ('plus', 'minus')},
                'model': {leg: {'denominator': np.array([8, 12])[:, None]*p} for leg in ('plus', 'minus')}}
    observed = {'specification': {'cases': [10, 11], 'axis': 1, 'h': .02},
                'records': records, 'parent_manifest_sha256': 'parent'}
    joint = {'cases': [10, 11], 'h': .02, 'shear_axis': 1, 'cuts': list(CUTS),
             'inputs': {'parent_manifest_sha256': 'parent', 'classifier_arms': {'test': 'same'}},
             'per_case_sufficient': summaries,
             'case_reports': {str(c): {'legs': {leg: measurement for leg in ('plus', 'minus')}} for c in [10, 11]}}
    return observed, joint


def test_uniform_usable_scaling_preserves_selected_density():
    observed, joint = fixture()
    out = compare_endpoint(observed, joint, {'test': 'same'})
    assert len(out) == 2*len(CUTS)*len(ARMS)
    for value in out.values():
        assert value['scores']['normalized_delta_nll']['estimate'] == pytest.approx(0, abs=1e-14)
    assert out['plus/'+CUTS[1]+'/modeled_transported']['scores']['raw_delta_nll']['estimate'] < 0


def test_reject_different_models_or_measured_population():
    observed, joint = fixture()
    with pytest.raises(ValueError, match='contract mismatch'):
        compare_endpoint(observed, joint, {'test': 'different'})
    changed = deepcopy(joint)
    changed['per_case_sufficient']['truth_parent/'+CUTS[1]+'/'+ARMS[-1]]['measured']['plus']['denominator'][0, 0] += 1
    with pytest.raises(AssertionError):
        compare_endpoint(observed, changed, {'test': 'same'})
