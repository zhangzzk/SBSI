import numpy as np
import pytest

from scripts.compute_response_target_blend import parse_crowd_quantiles


def test_custom_crowd_quantiles_override_bin_count():
    got = parse_crowd_quantiles("0,.05,.5,.95,1", n_crowd=8)
    np.testing.assert_allclose(got, [0.0, 0.05, 0.5, 0.95, 1.0])


@pytest.mark.parametrize("value", [".1,.5,1", "0,.5,.9", "0,.5,.5,1", "0,nan,1"])
def test_custom_crowd_quantiles_are_strict(value):
    with pytest.raises(ValueError):
        parse_crowd_quantiles(value, n_crowd=8)
