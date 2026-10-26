import numpy as np

from scripts.diagnose_constgold_fixed_g0_truth_support import PARTITION, truth_groups


def test_truth_groups_use_strict_historical_bounds_and_partition():
    r_input = np.array([20.0, 25.8, 20.0, 25.8, 18.0, 20.0])
    re_input = np.array([0.8, 0.8, 0.37, 0.37, 2.0, 1.5])

    groups = truth_groups(r_input, re_input)

    assert groups["inside_old_truth_support"].tolist() == [True, False, False, False, False, False]
    assert groups["outside_magnitude_only"].tolist() == [False, True, False, False, False, False]
    assert groups["outside_size_only"].tolist() == [False, False, True, False, False, True]
    assert groups["outside_both"].tolist() == [False, False, False, True, True, False]
    assert groups["truth_faint"].tolist() == [False, True, False, True, False, False]
    assert groups["truth_bright"].tolist() == [False, False, False, False, True, False]
    assert groups["truth_small"].tolist() == [False, False, True, True, False, False]
    assert groups["truth_large"].tolist() == [False, False, False, False, True, True]
    np.testing.assert_array_equal(
        sum(groups[name].astype(np.int8) for name in PARTITION),
        np.ones(len(r_input), dtype=np.int8),
    )
