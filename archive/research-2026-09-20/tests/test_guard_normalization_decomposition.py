import numpy as np
import pytest

from scripts.decompose_guard_normalization import decompose


def statistics():
    rng = np.random.default_rng(77)
    design = rng.normal(size=(5, 2, 2))
    return {'n': np.full(5, 100.), 'normal': design@design.transpose(0, 2, 1)+np.eye(2),
            'mass_zero': rng.uniform(30, 70, (5, 3)),
            'shape_zero': rng.normal(size=(5, 3, 2)),
            'delta_mass_gamma': rng.normal(size=(5, 3, 2)),
            'delta_shape_gamma': rng.normal(size=(5, 3, 2, 2))}


@pytest.mark.parametrize('mass_change,mean_change', [(False, False), (True, False), (False, True), (True, True)])
def test_exact_normalization_identity_and_isolated_terms(mass_change, mean_change):
    observed = statistics()
    predicted = {key: value.copy() for key, value in observed.items()}
    if mass_change:
        predicted['mass_zero'] *= 1.1
        predicted['shape_zero'] *= 1.1
    if mean_change:
        predicted['shape_zero'] += predicted['mass_zero'][..., None]*np.array([.03, -.02])
    terms, error = decompose(predicted, observed)
    assert error < 1e-13
    if not mass_change:
        np.testing.assert_array_equal(terms['selected_mass_term'], 0)
    if not mean_change:
        np.testing.assert_allclose(terms['centering_term'], 0, atol=1e-15)
    np.testing.assert_allclose(terms['reference_minus_own'],
                               terms['selected_mass_term']+terms['centering_term'], atol=1e-13)


def test_rejects_different_population_and_nonfinite_statistics():
    observed = statistics()
    predicted = {key: value.copy() for key, value in observed.items()}
    predicted['n'][0] += 1
    with pytest.raises(ValueError, match='identical parent'):
        decompose(predicted, observed)
    predicted['n'] = observed['n']
    predicted['shape_zero'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite sufficient'):
        decompose(predicted, observed)


def test_accepts_reduction_roundoff_but_rejects_changed_shear_design():
    observed = statistics()
    predicted = {key: value.copy() for key, value in observed.items()}
    predicted['normal'] *= 1+1e-14
    decompose(predicted, observed)
    predicted['normal'] *= 1+1e-7
    with pytest.raises(ValueError, match='identical parent'):
        decompose(predicted, observed)
