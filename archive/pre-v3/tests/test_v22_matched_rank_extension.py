import numpy as np
import pandas as pd
import pytest

from scripts.extract_v22_matched_decomposition import numbered_modes
from scripts.prepare_v22_matched_rank_extension import (
    assign_rank_shear,
    numbered_rank_bases,
)


def test_rank_modes_are_sorted_numerically_and_must_be_contiguous():
    bases = numbered_rank_bases([
        "rank2=/tmp/r2", "rank0=/tmp/r0", "rank1=/tmp/r1",
    ])
    assert list(bases) == ["rank0", "rank1", "rank2"]
    assert numbered_modes({**bases, "total": "/tmp/t"}, "rank") == [
        "rank0", "rank1", "rank2",
    ]
    with pytest.raises(ValueError, match="contiguous"):
        numbered_rank_bases(["rank0=/tmp/r0", "rank2=/tmp/r2"])


def test_rank_assignment_shears_only_selected_secondary_antithetically():
    frame = pd.DataFrame({
        "index": [10, 20, 30, 40],
        "g1": [9.0, 9.0, 9.0, 9.0],
        "g2": [8.0, 8.0, 8.0, 8.0],
    })
    selected = pd.DataFrame({
        "secondary_index": [20, 40],
        "u1": [1.0, 0.0],
        "u2": [0.0, -1.0],
    })
    plus = assign_rank_shear(frame, +0.05, 0.05, selected)
    minus = assign_rank_shear(frame, -0.05, 0.05, selected)

    np.testing.assert_allclose(plus["g1"], [0.0, 0.05, 0.0, 0.0])
    np.testing.assert_allclose(plus["g2"], [0.0, 0.0, 0.0, -0.05])
    np.testing.assert_allclose(minus[["g1", "g2"]], -plus[["g1", "g2"]])


def test_rank_assignment_rejects_missing_secondary():
    frame = pd.DataFrame({"index": [1], "g1": [0.0], "g2": [0.0]})
    selected = pd.DataFrame({"secondary_index": [2], "u1": [1.0], "u2": [0.0]})
    with pytest.raises(RuntimeError, match="coverage"):
        assign_rank_shear(frame, +0.05, 0.05, selected)
