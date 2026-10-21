"""Signed independent-group accounting and numerical-measure qualifications."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('v9summary', Path(__file__).resolve().parents[1]/'scripts/summarize_v9_particle_comparison.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_signed_differences_use_independent_mcse_and_reversal():
    a = {'mean': [1., -2.], 'mcse': [3., 0.]}
    b = {'mean': [3., -1.], 'mcse': [4., 0.]}
    c = m.signed_comparison(a, b)
    assert c['difference'] == [-2., -1.]
    assert c['combined_mcse'] == [5., 0.]
    assert c['signed_standardized_difference'] == [-.4, None]
    assert c['nonzero_with_zero_estimated_mcse'] == [False, True]
    assert m.signed_comparison(b, a)['difference'] == [2., 1.]


def test_stable_normalizer_comparison_preserves_relative_error_and_sign():
    a = {'log_mean_normalizer': 1000., 'normalizer_relative_mcse': .1}
    b = {'log_mean_normalizer': 999., 'normalizer_relative_mcse': .2}
    c = m.normalizer_comparison(a, b)
    assert c['ratio'] == pytest.approx(m.math.e)
    expected = (1.-m.math.exp(-1))/m.math.hypot(.1,.2*m.math.exp(-1))
    assert c['signed_standardized_difference'] == pytest.approx([expected])
    a['log_mean_normalizer'] -= 2000; b['log_mean_normalizer'] -= 2000
    assert m.normalizer_comparison(a,b)['signed_standardized_difference'] == pytest.approx([expected])


def test_metadata_erratum_preserves_source_without_claiming_f32_zero():
    source = {'control_known_expectation': 0., 'mean': [4.]}
    f32 = m.qualified_moments(source, m.MEASURES[0])
    f64 = m.qualified_moments(source, m.MEASURES[1])
    assert f32['control_known_expectation'] is None
    assert f32['historical_control_known_expectation'] == 0.
    assert f64['control_known_expectation'] == 0.
    assert 'not an exact' in f64['control_expectation_qualification']
    f32['mean'][0] = 9.
    assert source == {'control_known_expectation': 0., 'mean': [4.]}


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_invalid_uncertainty_is_rejected(value):
    with pytest.raises(ValueError):
        m.signed_comparison({'mean': [0.], 'mcse': [value]}, {'mean': [0.], 'mcse': [1.]})
