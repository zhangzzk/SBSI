import numpy as np
import pandas as pd

from scripts.analyze_anchorblend_eachpair_orthogonal import add_trace_decomposition
from scripts.prepare_anchorblend_eachpair_g2 import assign_coherent_g2, assign_rank_g2


def test_assign_rank_g2_shears_only_frozen_sources():
    frame = pd.DataFrame({"index": [1, 2, 3, 4], "g1": 1.0, "g2": 2.0})
    selected = pd.DataFrame({"secondary_index": [2, 4]})
    out = assign_rank_g2(frame, selected, sign=-1.0, g=0.05)
    np.testing.assert_allclose(out.g1, 0.0)
    np.testing.assert_allclose(out.g2, [0.0, -0.05, 0.0, -0.05])


def test_assign_coherent_g2_leaves_only_anchors_unsheared():
    frame = pd.DataFrame({"index": [1, 2, 3, 4], "g1": 1.0, "g2": 2.0})
    out = assign_coherent_g2(frame, np.array([1, 3]), sign=1.0, g=0.05)
    np.testing.assert_allclose(out.g1, 0.0)
    np.testing.assert_allclose(out.g2, [0.0, 0.05, 0.0, 0.05])


def test_trace_decomposition_and_cross_terms():
    frame = pd.DataFrame({
        "R_model_sum": [0.10],
        "R11_pair": [0.08], "R22_pair": [0.12],
        "R11_coherent": [0.09], "R22_coherent": [0.13],
        "R12_pair": [0.02], "R21_pair": [-0.01],
        "R12_coherent": [0.03], "R21_coherent": [-0.02],
    })
    out = add_trace_decomposition(frame)
    np.testing.assert_allclose(out.R_pair_trace, 0.10)
    np.testing.assert_allclose(out.R_coherent_trace, 0.11)
    np.testing.assert_allclose(
        out.coherent_gap_trace, out.emulator_gap_trace - out.additivity_gap_trace,
    )
    np.testing.assert_allclose(out.trace_replay, 0.0)
    np.testing.assert_allclose(out.pair_cross_symmetric, 0.005)
    np.testing.assert_allclose(out.pair_cross_rotation, -0.015)
