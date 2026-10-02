"""Initialization units, inverse shape convention and preserved latent support."""
import numpy as np
import pytest
from scipy.special import expit

from sbsi.galsbi_initialization import measurement_seed_coordinates
from sbsi.shear_map import apply_shear_to_ellipticity


class ZeroJitter:
    def normal(self, loc, scale, size):
        return np.zeros(size)


PARAMETERS = dict(sersic_n_min=.2, sersic_n_max=5., ell_beta_emax=.9)
CONDITIONS = dict(pixel_size=.2, zero_point=30., psf_fwhm=0., moffat_beta=2.224)


def test_initial_centres_convert_flux_pixels_and_epsilon_without_changing_nuisance():
    x = np.arange(40, dtype=float).reshape(4, 10)/100
    original = x.copy()
    shear = (.02, -.01)
    e1, e2 = apply_shear_to_ellipticity(.2, -.1, *shear)
    moved, report = measurement_seed_coordinates(x, [e1, e2, 4., 1000.], np.arange(4),
        CONDITIONS, PARAMETERS, shear, ZeroJitter())
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(moved[:, [0, 2, 3, 4, 5]], x[:, [0, 2, 3, 4, 5]])
    np.testing.assert_allclose(moved[:, 1], 22.5)
    np.testing.assert_allclose(np.exp(moved[:, 6]), .8)
    n = .2+4.8*expit(moved[:, 7])
    np.testing.assert_allclose(n, [.75, 1.5, 2.5, 4.])
    distortion = .9*expit(moved[:, 8])
    epsilon = distortion/(1+np.sqrt(1-distortion**2))
    np.testing.assert_allclose(epsilon*np.cos(moved[:, 9]), .2)
    np.testing.assert_allclose(epsilon*np.sin(moved[:, 9]), -.1)
    assert report['shape_initial_guesses_saturated'] == 0


def test_broad_guesses_remain_finite_even_for_unresolved_or_boundary_measurements():
    x = np.zeros((400, 10))
    conditions = dict(CONDITIONS, psf_fwhm=.73)
    moved, report = measurement_seed_coordinates(x, [.999, .01, .1, 100.], np.arange(400) % 4,
        conditions, PARAMETERS, (.02, 0.), np.random.default_rng(512))
    assert np.isfinite(moved).all()
    assert np.all((expit(moved[:, 8]) > 0) & (expit(moved[:, 8]) < 1))
    assert np.all((moved[:, 9] >= 0) & (moved[:, 9] < 2*np.pi))
    assert np.all(np.exp(moved[:, 6]) > 0)
    assert report['shape_initial_guesses_saturated'] > 0
    assert np.unique(moved[:, 1]).size == 400
    with pytest.raises(ValueError, match='measurements'):
        measurement_seed_coordinates(x, [0., 0., 4., -1.], np.arange(400) % 4,
            conditions, PARAMETERS, (.02, 0.), np.random.default_rng(512))
