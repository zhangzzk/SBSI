import numpy as np
import pandas as pd

from scripts.analyze_anchorblend_eachpair_g1 import add_decomposition
from scripts.prepare_anchorblend_eachpair_g1 import assign_rank_g1, assign_ranks


def test_rank_assignment_is_complete_and_keeps_selected_first():
    pairs = pd.DataFrame({
        "anchor_index": [1, 1, 1, 2, 2],
        "secondary_index": [10, 11, 12, 20, 21],
        "selection_hash": [9, 3, 7, 8, 2],
        "selected": [False, True, False, True, False],
        "response": [0.1] * 5,
        "distance": [1.0] * 5,
        "n_pairs": [3, 3, 3, 2, 2],
    })
    ranked = assign_ranks(pairs)
    assert ranked.loc[ranked.selected, "rank"].eq(0).all()
    assert ranked.groupby("anchor_index")["rank"].apply(list).map(sorted).tolist() == [
        [0, 1, 2], [0, 1],
    ]


def test_rank_g1_shears_only_selected_sources():
    frame = pd.DataFrame({"index": [1, 2, 3, 4], "g1": 1.0, "g2": 2.0})
    selected = pd.DataFrame({"secondary_index": [2, 4]})
    out = assign_rank_g1(frame, selected, sign=-1.0, g=0.05)
    np.testing.assert_allclose(out.g1, [0.0, -0.05, 0.0, -0.05])
    np.testing.assert_allclose(out.g2, 0.0)


def test_decomposition_identity():
    frame = pd.DataFrame({
        "R_model_sum": [0.10, 0.12],
        "R_pair_sum": [0.11, 0.09],
        "R_coherent_truth": [0.13, 0.08],
    })
    out = add_decomposition(frame)
    np.testing.assert_allclose(out.coherent_gap, out.emulator_gap - out.additivity_gap)
    np.testing.assert_allclose(out.decomposition_replay, 0.0)
