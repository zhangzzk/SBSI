import numpy as np

from scripts.analyze_anchorblend_random_radii import stat


def test_random_radii_stat_uses_case_sem():
    got = stat(np.array([1.0, 3.0]))
    assert got["mean"] == 2.0
    assert np.isclose(got["case_sem"], 1.0)
