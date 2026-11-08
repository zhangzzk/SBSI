import numpy as np
import pytest

from scripts.plot_disk_faint_corners import contour_levels, coordinates, selected_weights


def test_selected_mixture_weights_use_prior_usability_and_pass_fraction():
    probability = np.array([.2, .8])
    atoms = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    mask = np.array([True, False, False, False, True, True, True, False])
    weights, mass = selected_weights(probability, atoms, mask, 4)
    np.testing.assert_allclose(weights, np.array([.05, .2, .2, .2])/.65)
    assert mass == pytest.approx(.325)


def test_contour_mass_thresholds_and_window_coverage():
    hist = np.array([[.4, .3], [.2, .1]])
    levels = contour_levels(hist)
    np.testing.assert_allclose(levels, [.1, .3])
    with pytest.raises(ValueError):
        contour_levels(hist*.9)


def test_output_units_and_magnitude_transform():
    values = coordinates(np.array([.1, .2, 5., 100.]))
    np.testing.assert_allclose(values, [.1, .2, 1., 25.])
