import numpy as np

from scripts.diag_v22_thirdplus_conditional import (
    joint_cells,
    quantile_index,
    within_cell_quartiles,
)


def test_within_cell_quartiles_balance_each_cell():
    exposure = np.array([8, 1, 4, 2, 9, 3, 7, 6, 18, 11, 14, 12, 19, 13, 17, 16.])
    cells = np.repeat([0, 1], 8)
    mask = np.ones(16, dtype=bool)
    ids, info = within_cell_quartiles(exposure, cells, mask, min_cell=8)
    assert info == {"total_cells": 2, "kept_cells": 2, "kept_rows": 16, "excluded_rows": 0}
    for cell in (0, 1):
        np.testing.assert_array_equal(np.bincount(ids[cells == cell], minlength=4), [2, 2, 2, 2])
        means = [exposure[(cells == cell) & (ids == q)].mean() for q in range(4)]
        assert np.all(np.diff(means) > 0)


def test_control_quartiles_and_joint_cells():
    x = np.arange(16, dtype=float)
    mask = np.ones(16, dtype=bool)
    q, edges = quantile_index(x, mask, 4)
    np.testing.assert_array_equal(np.bincount(q, minlength=4), [4, 4, 4, 4])
    cells = joint_cells([q, q[::-1]], [4, 4], mask)
    assert cells.min() >= 0
    assert cells.max() < 16
    assert len(edges) == 5
