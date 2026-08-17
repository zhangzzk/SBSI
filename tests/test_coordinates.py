import numpy as np
import pandas as pd

from sbs_shear.coordinates import (
    SKY_SHEAR_CONVENTION,
    add_shear_aligned_gradients,
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


def test_raw_gradient_chain_rule_adds_canonical_sky_components():
    theta = np.deg2rad(np.array([20.0, 70.0]))
    frame = pd.DataFrame({
        "gamma1_input_p": np.cos(2.0 * theta),
        "gamma2_input_p": np.sin(2.0 * theta),
        "gamma1_input_s": np.cos(2.0 * theta),
        "gamma2_input_s": np.sin(2.0 * theta),
        "shear_component_convention": SKY_SHEAR_CONVENTION,
    })
    grad = pd.DataFrame({
        "dPsel_dgamma1_input_p": [3.0, 4.0],
        "dPsel_dgamma2_input_p": [5.0, 6.0],
        "dPsel_dgamma1_input_s": [7.0, 8.0],
        "dPsel_dgamma2_input_s": [9.0, 10.0],
    })

    out = add_shear_aligned_gradients(frame, grad)

    np.testing.assert_allclose(out["dPsel_dgamma1_sky_p"], grad["dPsel_dgamma1_input_p"])
    np.testing.assert_allclose(out["dPsel_dgamma2_sky_p"], grad["dPsel_dgamma2_input_p"])
    expected_parallel = (
        grad["dPsel_dgamma1_input_p"].to_numpy() * np.cos(2.0 * theta)
        + grad["dPsel_dgamma2_input_p"].to_numpy() * np.sin(2.0 * theta)
    )
    np.testing.assert_allclose(out["dPsel_dgamma_parallel_p"], expected_parallel)
