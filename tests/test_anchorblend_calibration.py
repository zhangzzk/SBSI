import numpy as np
import pandas as pd

from scripts.analyze_anchorblend_calibration import case_table, scale_and_residual


def test_case_table_filters_nonfinite_and_averages_by_case():
    frame = pd.DataFrame({
        "case": [0, 0, 1],
        "truth": [2.0, np.nan, 4.0],
        "pred": [1.0, 99.0, 2.0],
    })
    got = case_table(frame, "truth", "pred")
    np.testing.assert_allclose(got.to_numpy(), [[2.0, 1.0], [4.0, 2.0]])


def test_scale_is_fit_only_on_dev_and_scored_on_test():
    dev = np.array([[2.0, 1.0], [6.0, 3.0]])
    test = np.array([[3.0, 2.0], [9.0, 6.0]])
    scale, residual, test_scale = scale_and_residual(dev, test)
    assert scale == 2.0
    assert residual == -2.0
    assert test_scale == 1.5
