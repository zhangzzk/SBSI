import numpy as np

from sbsi.coordinates import (
    SKY_SHEAR_CONVENTION,
    blendemu_shear_to_sky,
    spin2_components_from_angle,
    spin2_project_to_angle,
)


def test_current_blendemu_shear_is_already_canonical_sky_basis():
    theta = np.deg2rad(np.array([0.0, 30.0, 80.0]))
    gamma = np.array([0.1, 0.05, 0.02])
    raw_gamma1 = gamma * np.cos(2.0 * theta)
    raw_gamma2 = gamma * np.sin(2.0 * theta)

    gamma1_sky, gamma2_sky = blendemu_shear_to_sky(
        raw_gamma1,
        raw_gamma2,
        convention=SKY_SHEAR_CONVENTION,
    )

    np.testing.assert_allclose(gamma1_sky, gamma * np.cos(2.0 * theta))
    np.testing.assert_allclose(gamma2_sky, gamma * np.sin(2.0 * theta))


def test_spin2_projection_is_parallel_to_its_own_angle():
    phi = np.deg2rad(np.array([0.0, 35.0, 120.0]))
    component1, component2 = spin2_components_from_angle(phi)

    parallel, cross = spin2_project_to_angle(component1, component2, phi)

    np.testing.assert_allclose(parallel, 1.0)
    np.testing.assert_allclose(cross, 0.0, atol=1.0e-15)
