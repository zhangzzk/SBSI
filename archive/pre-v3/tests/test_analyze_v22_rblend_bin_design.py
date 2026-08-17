import numpy as np
import pytest

from scripts.analyze_v22_rblend_bin_design import DESIGNS, edge_indices, merge_last_axis


def test_designs_are_contiguous_decile_partitions():
    for segments in DESIGNS.values():
        idx = edge_indices(segments)
        assert idx[0] == 0
        assert idx[-1] == 10
        assert len(idx) == len(segments) + 1


def test_merge_last_axis_preserves_counts():
    counts = np.arange(2 * 3 * 10).reshape(2, 3, 10)
    for segments in DESIGNS.values():
        merged = merge_last_axis(counts, segments)
        assert merged.shape == (2, 3, len(segments))
        np.testing.assert_array_equal(merged.sum(axis=-1), counts.sum(axis=-1))


def test_edge_indices_refuses_gap():
    with pytest.raises(ValueError, match="contiguous"):
        edge_indices([(0, 2), (3, 10)])
