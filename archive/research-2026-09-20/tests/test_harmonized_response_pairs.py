import pytest

from scripts.audit_harmonized_response_pairs import primary_response_difference


def test_response_difference_groups_shared_primary_labels():
    result = primary_response_difference([10, 10, 20, 20, 20], [1., 1., -2., -2., -2.], [0.]*5)
    assert result['pairs'] == 5
    assert result['represented_primaries'] == 2
    assert result['paired_small_minus_large'] == pytest.approx(-2.)
    assert result['paired_primary_se'] == pytest.approx(4.)
    assert result['response_large'] == 0.


def test_shared_measurement_noise_cancels_before_uncertainty_estimation():
    result = primary_response_difference([10, 10, 20], [101., 101., -49.], [100., 100., -51.])
    assert result['paired_small_minus_large'] == 2.
    assert result['paired_primary_se'] == 0.
