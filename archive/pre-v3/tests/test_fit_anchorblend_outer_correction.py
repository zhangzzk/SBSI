import numpy as np

from scripts.fit_anchorblend_outer_correction import make_model, stat


def test_outer_correction_model_is_deterministic_and_nontrivial():
    x = np.arange(15000, dtype=float).reshape(5000, 3) / 5000.0
    y = 0.1 * x[:, 0] - 0.2 * x[:, 1]
    a = make_model(7).fit(x, y).predict(x)
    b = make_model(7).fit(x, y).predict(x)
    np.testing.assert_array_equal(a, b)
    assert np.std(a) > 0


def test_stat_uses_case_sem():
    got = stat(np.array([1.0, 3.0]))
    assert got["mean"] == 2.0
    assert np.isclose(got["case_sem"], 1.0)
