import numpy as np
import pytest

from scripts.decompose_v36_population_error import allocation, mixture


def sums(masses, response, intercept=0., h=.02):
    mass = np.asarray(masses, float)
    return np.column_stack((mass, mass*(intercept+np.array([h, -h])*response), np.zeros(2)))


def test_population_composition_error_without_conditional_response_error():
    db, df = sums([90, 90], .8), sums([10, 10], .4)
    mb, mf = sums([80, 80], .8), sums([20, 20], .4)
    d, m, terms, _ = allocation(db, df, mb, mf, .02)
    np.testing.assert_allclose(terms['bright_response'], 0., atol=1e-14)
    np.testing.assert_allclose(terms['faint_response'], 0., atol=1e-14)
    np.testing.assert_allclose(terms['selected_composition'], [.04, 0.], atol=1e-14)
    np.testing.assert_allclose(d['response']-m['response'], [.04, 0.], atol=1e-14)


def test_selection_migration_and_symmetric_error_allocation():
    args = (sums([90, 91], .8, .05), sums([10, 9], .4, -.04),
            sums([81, 80], .78, .03), sums([19, 20], .45, -.02))
    d, m, terms, _ = allocation(*args, .02)
    assert abs(terms['selection_migration'][0]) > 0
    np.testing.assert_allclose(sum(terms.values()), d['response']-m['response'], atol=1e-14)
    _, _, reversed_terms, _ = allocation(*args[2:], *args[:2], .02)
    for key in terms:
        np.testing.assert_allclose(terms[key], -reversed_terms[key], atol=1e-14)


def test_rejects_empty_stratum_and_invalid_step():
    with pytest.raises(ValueError):
        mixture(sums([0, 0], .8), sums([10, 10], .4), .02)
    with pytest.raises(ValueError):
        mixture(sums([90, 90], .8), sums([10, 10], .4), 0.)
