import numpy as np

from scripts.compute_shell_features_v27 import accumulate_shell_flux


def test_accumulate_shell_flux_boundaries_and_exclusion():
    distance = np.array([0.0, 0.49, 0.5, 0.99, 1.0, 1.99, 2.0, 2.99,
                         3.0, 4.99, 5.0, 9.99, 10.0, 11.0])
    flux = np.arange(1.0, len(distance) + 1.0)
    got = accumulate_shell_flux(distance, flux)
    expected = np.array([
        flux[0] + flux[1], flux[2] + flux[3], flux[4] + flux[5],
        flux[6] + flux[7], flux[8] + flux[9], flux[10] + flux[11],
    ])
    np.testing.assert_array_equal(got, expected)


def test_accumulate_shell_flux_empty():
    got = accumulate_shell_flux(np.array([]), np.array([]))
    np.testing.assert_array_equal(got, np.zeros(6))
