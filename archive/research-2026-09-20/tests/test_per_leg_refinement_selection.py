import numpy as np

from scripts.refine_per_leg_guard_flow import eligible_selection

MASS = {'selection_residual': [.001, .002], 'baseline_selection_residual': [.002, .003]}


def test_subset_pass_does_not_admit_full_density_failure():
    assert not eligible_selection(.1, .2, -4.8, -4.8, -4.79, -4.8, **MASS)
    assert eligible_selection(.1, .2, -4.8, -4.8, -4.81, -4.8, **MASS)


def test_guard_regression_and_nonfinite_scores_preserve_initial_candidate():
    assert not eligible_selection(.3, .2, -4.9, -4.8, -4.9, -4.8, **MASS)
    assert not eligible_selection(.1, .2, -4.8, -4.8, np.nan, -4.8, **MASS)
    assert not eligible_selection(np.nan, .2, -4.8, -4.8, -4.8, -4.8, **MASS)


def test_likelihood_and_guard_improvements_cannot_hide_selection_degradation():
    for residual in ([.003, .002], [.001, -.004], [np.nan, .002]):
        assert not eligible_selection(.1, .2, -4.9, -4.8, -4.9, -4.8,
            selection_residual=residual, baseline_selection_residual=[.002, .003])
