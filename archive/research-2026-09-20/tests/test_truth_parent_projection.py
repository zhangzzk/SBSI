import copy

import pytest

from scripts.project_truth_parent_audit import project


def fixture():
    branches = {f'{cohort}/uncut/actual_usable_flags': {'numerator': [[1, 2], [3, 4]]}
                for cohort in ('truth_parent', 'usable_g0')}
    return {'cases': [4, 6], 'cohorts': ['truth_parent', 'usable_g0'], 'inputs': {'source': 'unchanged'},
        'case_reports': {str(case): {'cohort_counts': {'truth_parent': 12, 'usable_g0': 10},
            'legs': {'plus': {'matched': 11}}, 'matched_usable': 9} for case in (4, 6)},
        'per_case_sufficient': branches, 'summary': {key: {'m_percent': .2} for key in branches}}


def test_projection_preserves_all_truth_rows_statistics_and_source():
    source = fixture()
    before = copy.deepcopy(source)
    result = project(source)
    assert source == before
    assert result['cases'] == source['cases'] and result['inputs'] == source['inputs']
    assert result['cohorts'] == ['truth_parent']
    key = 'truth_parent/uncut/actual_usable_flags'
    assert result['per_case_sufficient'] == {key: source['per_case_sufficient'][key]}
    assert result['summary'] == {key: source['summary'][key]}
    for case, report in result['case_reports'].items():
        assert report['cohort_counts'] == {'truth_parent': 12}
        assert report['legs'] == source['case_reports'][case]['legs']
        assert report['matched_usable'] == 9


def test_incomplete_case_coverage_or_missing_truth_counts_rejected():
    result = fixture()
    del result['case_reports']['4']
    with pytest.raises(ValueError, match='every original case'):
        project(result)
    result = fixture()
    del result['case_reports']['4']['cohort_counts']['truth_parent']
    with pytest.raises(ValueError, match='truth-parent count'):
        project(result)
