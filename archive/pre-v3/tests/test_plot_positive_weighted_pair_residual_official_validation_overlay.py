import numpy as np
import pytest

from scripts.plot_positive_weighted_pair_residual_official_validation_overlay import (
    official_split_mask,
    v21_primary_mask,
)
from sbs_shear.domain import in_domain


def test_v21_primary_mask_uses_named_primary_columns_only():
    names = ["r_input_s", "r_input_p", "distance", "Re_input_p"]
    raw = np.array([
        [18.0, 24.0, 1.0, 0.60],
        [18.0, 24.0, 1.0, 0.40],
        [18.0, 27.0, 1.0, 0.60],
    ])
    expected = in_domain(raw[:, 1], raw[:, 3])
    assert np.array_equal(v21_primary_mask(raw, names), expected)


def test_v21_primary_mask_rejects_an_unstamped_layout():
    with pytest.raises(KeyError, match="raw features lack"):
        v21_primary_mask(np.ones((3, 2)), ["r_input_p", "distance"])
    with pytest.raises(ValueError, match="does not match"):
        v21_primary_mask(
            np.ones((3, 3)),
            ["r_input_p", "Re_input_p"],
        )


def test_official_split_masks_are_exact_complements():
    stored = np.array([True, False, True, True, False])
    training = official_split_mask(stored, "training")
    validation = official_split_mask(stored, "validation")
    assert np.array_equal(training, stored)
    assert np.array_equal(validation, ~stored)
    assert not np.any(training & validation)
    assert np.all(training | validation)


def test_official_split_mask_rejects_bad_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        official_split_mask(np.ones((2, 2), dtype=bool), "training")
    with pytest.raises(ValueError, match="unknown official split"):
        official_split_mask(np.ones(3, dtype=bool), "test")
