import numpy as np

from scripts.compact_neighbor_scene import top2_log_ratio


def test_top2_log_ratio_recovers_linear_flux_remainder():
    all_ratio = np.array([0.0, 1.0, 2.0, 10.0, 100.0])
    third_ratio = np.array([0.0, 0.0, 1.0, 4.0, 80.0])

    got = top2_log_ratio(np.log10(1.0 + all_ratio),
                         np.log10(1.0 + third_ratio))

    np.testing.assert_allclose(got, np.log10(1.0 + all_ratio - third_ratio),
                               rtol=1e-12, atol=1e-12)


def test_top2_log_ratio_clips_roundoff_below_zero():
    log_all = np.log10(1.0 + np.array([1.0, 10.0]))
    log_third = np.nextafter(log_all, np.inf)

    np.testing.assert_array_equal(top2_log_ratio(log_all, log_third), np.zeros(2))
