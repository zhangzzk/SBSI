import numpy as np
import pytest

from scripts.prepare_constgold_truth_parent import parent_case


def sample():
    context = np.zeros((4, 8), dtype=np.float32)
    context[:, 3] = [24, 25, 25.9, 26]
    prepared = {"ids": np.array([3, 1, 2, 4]), "x0": context}
    zero = {"input_index": np.array([1, 3]), "context": context[[1, 0]],
            "target": np.array([[0, 0, 3.1, 100], [0, 0, 2.0, 100]])}
    return prepared, zero


def test_truth_parent_keeps_undetected_and_records_g0_masks():
    arrays, counts = parent_case(*sample())
    np.testing.assert_array_equal(arrays["input_index"], [1, 2, 3])
    np.testing.assert_array_equal(arrays["g0_usable"], [True, False, True])
    np.testing.assert_array_equal(arrays["g0_selected"], [True, False, False])
    assert counts["g0_unusable_retained"] == 1
    assert counts["truth_cut_dropped"] == 1
    assert "target" not in arrays


def test_truth_parent_reports_dropped_keys_and_rejects_context_mismatch():
    prepared, zero = sample()
    zero["input_index"][0] = 8
    _, counts = parent_case(prepared, zero)
    assert counts["g0_keys_outside_truth_parent_dropped"] == 1
    prepared, zero = sample()
    zero["context"][0, 0] += 0.1
    with pytest.raises(ValueError, match="contexts differ"):
        parent_case(prepared, zero)


def test_strict_truth_cut_uses_original_precision_not_rounded_context():
    prepared, zero = sample()
    arrays, _ = parent_case(prepared, zero, truth_magnitude=[24, 25, 25.9, 25.9999999])
    assert 4 in arrays["input_index"]
