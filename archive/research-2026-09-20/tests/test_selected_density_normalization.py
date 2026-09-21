import numpy as np
import pytest

from scripts.normalize_selected_density_comparison import score_change


def test_equal_selected_law_despite_changed_pass_probability():
    # Densities within S and their pass masses both grow by a factor two.
    # Their normalized selected laws are identical, so the proper score is equal.
    out = score_change([7, 11], [-np.log(2)]*2, [2, 3], [4, 6], [[1, 1], [2, 0], [0, 2]])
    np.testing.assert_allclose(out['normalized_delta_nll'], 0, atol=1e-15)
    assert np.all(out['raw_delta_nll'] < 0)


def test_pooled_parent_measure_and_paired_case_resampling():
    out = score_change([1, 3], [.2, -.1], [1, 4], [2, 3], [[1, 1], [2, 0]])
    np.testing.assert_allclose(out['normalized_delta_nll'], [-.025, .2+np.log(2)])
    # A mean of case-wise log normalizations defines a different conditioning law.
    assert not np.isclose(out['log_selection_mass_ratio'][0], np.mean(np.log([2, .75])))


def test_reject_nonpositive_mass_and_empty_bootstrap():
    for mass, weights in [([0, 1], [[1, 1]]), ([1, 1], [[0, 0]])]:
        with pytest.raises(ValueError, match='positive counts/masses'):
            score_change([1, 1], [0, 0], mass, [1, 1], weights)
