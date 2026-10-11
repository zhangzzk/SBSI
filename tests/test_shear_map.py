"""Tests for the reduced-shear map of an intrinsic ellipticity."""

import numpy as np

from sbsi.shear_map import apply_shear_to_ellipticity


def test_opposite_shear_undoes_the_map():
    rng = np.random.default_rng(0)
    e1 = rng.uniform(-0.6, 0.6, 1000)
    e2 = rng.uniform(-0.6, 0.6, 1000)
    g1, g2 = 0.07, -0.03
    s1, s2 = apply_shear_to_ellipticity(e1, e2, g1, g2)
    b1, b2 = apply_shear_to_ellipticity(s1, s2, -g1, -g2)
    assert np.allclose(b1, e1, atol=1e-10)
    assert np.allclose(b2, e2, atol=1e-10)


def test_zero_shear_identity():
    e1 = np.array([0.3, -0.1, 0.0])
    e2 = np.array([0.0, 0.2, -0.4])
    s1, s2 = apply_shear_to_ellipticity(e1, e2, 0.0, 0.0)
    assert np.allclose(s1, e1)
    assert np.allclose(s2, e2)


def test_orientation_average_responsivity_is_unity():
    # Fixed |e|, uniform orientation: the mean d e'/d g at g = 0 is the identity.
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, np.pi, 200000)
    e1, e2 = 0.3 * np.cos(2 * theta), 0.3 * np.sin(2 * theta)
    h = 1e-6
    plus = np.stack(apply_shear_to_ellipticity(e1, e2, h, 0.0))
    minus = np.stack(apply_shear_to_ellipticity(e1, e2, -h, 0.0))
    assert np.allclose(((plus - minus) / (2 * h)).mean(axis=1), [1.0, 0.0], atol=2e-3)
