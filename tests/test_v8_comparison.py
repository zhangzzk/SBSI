"""No-comparator semantics and complete declared populations for the n16 summary."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('v8summary',Path(__file__).resolve().parents[1]/'scripts/summarize_v8_particle_comparison.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_false_boolean_without_comparator_is_not_a_failed_comparison():
    assert m.comparison_status({'comparison_gate':False})['status']=='not_evaluated_no_comparator'
    assert m.comparison_status({'comparison_gate':False,'comparison':{}})['status']=='fail'
    assert m.comparison_status({'comparison_gate':True,'comparison':{}})['status']=='pass'


def panel():
    seeds=list(range(100,108));obs=[{'observation':o} for o in m.OBSERVATIONS]
    report={'seeds':seeds,'runs':list(range(8)),'replicas':8,'observations':obs}
    phase={'seeds':seeds,'sources':{str(i):str(i) for i in range(8)},'observations':obs,
           'moment_columns':m.COLUMNS,'coefficients':{'sha256':m.COEFFICIENT_HASH}}
    return copy.deepcopy(report),copy.deepcopy(phase),seeds


def test_complete_panel_passes():
    m.validate_panel(*panel())


@pytest.mark.parametrize('defect',['missing_seed','missing_source','wrong_observation','wrong_coefficients'])
def test_incomplete_or_changed_panel_rejected(defect):
    report,phase,seeds=panel()
    if defect=='missing_seed':phase['seeds'].pop()
    if defect=='missing_source':phase['sources'].pop('0')
    if defect=='wrong_observation':report['observations'][0]['observation']=-1
    if defect=='wrong_coefficients':phase['coefficients']['sha256']='changed'
    with pytest.raises(ValueError):m.validate_panel(report,phase,seeds)


def test_weight_replay_allows_one_ulp_but_rejects_substantive_or_missing_weights():
    weights=[.12,.13]
    roundoff=[m.math.nextafter(.12,1.),.13]
    assert m.vectors_match_ulps(weights,roundoff)
    assert not m.vectors_match_ulps(weights,[.12+1e-8,.13])
    assert not m.vectors_match_ulps(weights,[.12])
    assert not m.vectors_match_ulps(weights,[float('nan'),.13])
    assert weights==[.12,.13]
