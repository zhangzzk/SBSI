import numpy as np

from scripts.build_v22_constgold_response_dominance import summarize_pair_responses


def test_pair_response_summary_uses_absolute_top_two_without_sign_cancellation():
    result = summarize_pair_responses(
        np.array([4, 4, 4, 9, 9]),
        np.array([0.8, -0.1, 0.1, -0.4, 0.4]),
    ).set_index("input_index")
    np.testing.assert_allclose(result.loc[4, "R_blend"], 0.8)
    np.testing.assert_allclose(result.loc[4, "dominant_abs_response"], 0.8)
    np.testing.assert_allclose(result.loc[4, "runner_up_abs_response"], 0.1)
    np.testing.assert_allclose(
        result.loc[4, "dominant_to_runner_up_abs_response"], 8.0
    )
    # Equal absolute maxima must produce a ratio of one, even with opposite signs.
    np.testing.assert_allclose(
        result.loc[9, "dominant_to_runner_up_abs_response"], 1.0
    )


def test_one_pair_has_zero_runner_and_floor_defined_ratio():
    result = summarize_pair_responses(np.array([7]), np.array([-0.25])).iloc[0]
    assert result.n_pairs == 1
    assert result.runner_up_abs_response == 0.0
    np.testing.assert_allclose(
        result.dominant_to_runner_up_abs_response, 0.25 / 1.0e-12
    )
