import pandas as pd
import pytest

from scripts.compact_key_features import compact_features


def test_compact_features_collapses_consistent_duplicates():
    frame = pd.DataFrame({
        "case": [1, 1, 1], "input_index": [3, 3, 4],
        "r_input_p": [24.0, 24.0, 25.0], "Re_input_p": [0.7, 0.7, 0.9],
    })
    got = compact_features(frame, ["r_input_p", "Re_input_p"])
    assert got[["case", "input_index"]].values.tolist() == [[1, 3], [1, 4]]
    assert got["Re_input_p"].tolist() == [0.7, 0.9]


def test_compact_features_refuses_ambiguous_duplicate():
    frame = pd.DataFrame({
        "case": [1, 1], "input_index": [3, 3],
        "r_input_p": [24.0, 24.1], "Re_input_p": [0.7, 0.7],
    })
    with pytest.raises(RuntimeError, match="r_input_p"):
        compact_features(frame, ["r_input_p", "Re_input_p"])
