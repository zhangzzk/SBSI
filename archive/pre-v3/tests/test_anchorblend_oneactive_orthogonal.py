import numpy as np
import pandas as pd

from scripts.prepare_anchorblend_oneactive_orthogonal import (
    assign_selected_shear,
    select_one_pair,
)


def test_select_one_pair_is_stable_and_one_per_anchor():
    pairs = pd.DataFrame({
        "anchor_index": [10, 10, 10, 20, 20],
        "secondary_index": [3, 2, 1, 5, 4],
        "response": np.arange(5, dtype=float),
    })
    left = select_one_pair(pairs, case=400, seed=7)
    right = select_one_pair(pairs.sample(frac=1.0, random_state=9), case=400, seed=7)
    left_ids = set(map(tuple, left.loc[left.selected, ["anchor_index", "secondary_index"]].to_numpy()))
    right_ids = set(map(tuple, right.loc[right.selected, ["anchor_index", "secondary_index"]].to_numpy()))
    assert left_ids == right_ids
    assert left.loc[left.selected].groupby("anchor_index").size().eq(1).all()
    assert set(left.loc[left.anchor_index == 10, "n_pairs"]) == {3}
    assert set(left.loc[left.anchor_index == 20, "n_pairs"]) == {2}


def test_assign_selected_shear_changes_only_selected_source():
    frame = pd.DataFrame({
        "index": [1, 2, 3], "g1": [0.05, 0.05, 0.05], "g2": [0.0, 0.0, 0.0],
    })
    selected = pd.DataFrame({
        "secondary_index": [2], "u1": [0.6], "u2": [0.8],
        "v1": [-0.8], "v2": [0.6],
    })
    plus_u = assign_selected_shear(frame, selected, "u", +1.0, 0.05)
    minus_v = assign_selected_shear(frame, selected, "v", -1.0, 0.05)
    np.testing.assert_allclose(plus_u[["g1", "g2"]], [[0, 0], [0.03, 0.04], [0, 0]])
    np.testing.assert_allclose(minus_v[["g1", "g2"]], [[0, 0], [0.04, -0.03], [0, 0]])
