import numpy as np
import pandas as pd
import pytest

from scripts.build_fixed_g0_response_profile_predictions import (
    BLEND_RESPONSE_COLUMNS,
    projected_response,
    valid_blend_response_pairs,
)


def test_projected_response_handles_arbitrary_shear_direction():
    gamma = np.array([[0.05, 0.0], [0.03, 0.04]])
    parallel = np.array([1.2, -0.4])
    perpendicular = np.column_stack((-gamma[:, 1], gamma[:, 0]))
    delta = parallel[:, None] * gamma + 0.7 * perpendicular

    np.testing.assert_allclose(projected_response(delta, gamma), parallel)


def test_projected_response_rejects_zero_shear():
    with pytest.raises(ValueError, match="nonzero shears"):
        projected_response(np.zeros((1, 2)), np.zeros((1, 2)))


def test_valid_blend_response_pairs_requires_finite_interior_shapes():
    frame = pd.DataFrame(
        np.ones((3, len(BLEND_RESPONSE_COLUMNS))),
        columns=BLEND_RESPONSE_COLUMNS,
    )
    frame.loc[:, ["measured_e1_m", "measured_e2_m", "measured_e1_p", "measured_e2_p"]] = 0.0
    frame.loc[1, "measured_e1_m"] = 1.0
    frame.loc[2, "distance"] = np.nan

    np.testing.assert_array_equal(
        valid_blend_response_pairs(frame), [True, False, False]
    )
