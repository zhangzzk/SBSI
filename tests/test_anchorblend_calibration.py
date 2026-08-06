import numpy as np
import pandas as pd
import pytest

from scripts.analyze_anchorblend_calibration import (
    bootstrap_scale,
    case_table,
    scale_and_residual,
)
from scripts.fit_anchorblend_isotonic import read_disjoint_inputs


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


def test_response_inputs_must_have_disjoint_cases(tmp_path):
    first = tmp_path / "first.feather"
    second = tmp_path / "second.feather"
    pd.DataFrame({"case": [0, 1], "value": [2.0, 3.0]}).to_feather(first)
    pd.DataFrame({"case": [2], "value": [4.0]}).to_feather(second)
    got = read_disjoint_inputs([first, second])
    assert got["case"].tolist() == [0, 1, 2]

    pd.DataFrame({"case": [1, 2], "value": [4.0, 5.0]}).to_feather(second)
    with pytest.raises(RuntimeError, match="overlap in case IDs"):
        read_disjoint_inputs([first, second])


def test_all_case_scale_bootstrap_preserves_exact_ratio():
    data = np.array([[2.0, 1.0], [4.0, 2.0], [6.0, 3.0]])
    np.testing.assert_allclose(bootstrap_scale(data, 20, 7), 2.0)
