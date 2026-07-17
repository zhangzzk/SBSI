"""Tests for the analytic shear-distortion map S_gamma."""

import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.shear_map import (
    apply_shear_to_ellipticity,
    inverse_shear_to_ellipticity,
    magnification,
    shear_jacobian_at_zero,
)


def test_inverse_round_trip():
    rng = np.random.default_rng(0)
    e1 = rng.uniform(-0.6, 0.6, 1000)
    e2 = rng.uniform(-0.6, 0.6, 1000)
    g1, g2 = 0.07, -0.03
    s1, s2 = apply_shear_to_ellipticity(e1, e2, g1, g2)
    b1, b2 = inverse_shear_to_ellipticity(s1, s2, g1, g2)
    assert np.allclose(b1, e1, atol=1e-10)
    assert np.allclose(b2, e2, atol=1e-10)


def test_zero_shear_identity():
    e1 = np.array([0.3, -0.1, 0.0])
    e2 = np.array([0.0, 0.2, -0.4])
    s1, s2 = apply_shear_to_ellipticity(e1, e2, 0.0, 0.0)
    assert np.allclose(s1, e1)
    assert np.allclose(s2, e2)


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(1)
    e1 = rng.uniform(-0.5, 0.5, 500)
    e2 = rng.uniform(-0.5, 0.5, 500)
    J = shear_jacobian_at_zero(e1, e2)
    h = 1e-6
    # d/dg1
    sp1 = np.stack(apply_shear_to_ellipticity(e1, e2, h, 0.0), axis=-1)
    sm1 = np.stack(apply_shear_to_ellipticity(e1, e2, -h, 0.0), axis=-1)
    dg1 = (sp1 - sm1) / (2 * h)
    # d/dg2
    sp2 = np.stack(apply_shear_to_ellipticity(e1, e2, 0.0, h), axis=-1)
    sm2 = np.stack(apply_shear_to_ellipticity(e1, e2, 0.0, -h), axis=-1)
    dg2 = (sp2 - sm2) / (2 * h)
    assert np.allclose(J[:, :, 0], dg1, atol=1e-5)
    assert np.allclose(J[:, :, 1], dg2, atol=1e-5)


def test_orientation_average_responsivity_is_unity():
    # Fixed |e|, uniform orientation -> <J> = identity (unit responsivity for eps).
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, np.pi, 200000)
    e_abs = 0.3
    e1 = e_abs * np.cos(2 * theta)
    e2 = e_abs * np.sin(2 * theta)
    J = shear_jacobian_at_zero(e1, e2).mean(axis=0)
    assert np.allclose(J, np.eye(2), atol=2e-3)


def test_magnification_positive_weak():
    assert np.isclose(magnification(0.0, 0.0), 1.0)
    assert magnification(0.05, 0.0) > 1.0


if __name__ == "__main__":
    test_inverse_round_trip()
    test_zero_shear_identity()
    test_jacobian_matches_finite_difference()
    test_orientation_average_responsivity_is_unity()
    test_magnification_positive_weak()
    print("all shear_map tests passed")
