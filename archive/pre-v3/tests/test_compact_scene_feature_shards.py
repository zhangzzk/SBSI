import numpy as np
import pandas as pd

from scripts.compact_scene_feature_shards import exact_key_match


def test_exact_key_match_ignores_integer_width():
    left = pd.DataFrame({
        "case": np.array([0, 1], dtype=np.int64),
        "input_index": np.array([10, 20], dtype=np.int64),
    })
    right = pd.DataFrame({
        "case": np.array([0, 1], dtype=np.int16),
        "input_index": np.array([10, 20], dtype=np.int32),
    })
    assert exact_key_match(left, right, ["case", "input_index"])


def test_exact_key_match_rejects_value_or_length_change():
    reference = pd.DataFrame({"case": [0, 1], "input_index": [10, 20]})
    changed = pd.DataFrame({"case": [0, 1], "input_index": [10, 21]})
    shortened = reference.iloc[:1]
    assert not exact_key_match(reference, changed, ["case", "input_index"])
    assert not exact_key_match(reference, shortened, ["case", "input_index"])
