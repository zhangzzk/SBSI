import copy

import pytest

from scripts.plot_official_train_validation_v21_comparison import (
    infer_split,
    validate_inputs,
)
from scripts.plot_positive_weighted_pair_residual_validation_overlay import MODEL_SPECS


def payload(split, n_pairs):
    is_training = split == "training"
    return {
        "design": {
            "cache": "/cache",
            "case_window": [40, 199],
            "random_state": 321,
            "test_size": 0.2,
            "official_split": split,
            "official_train_value": is_training,
            "v21_primary_domain": True,
            "v21_domain": {
                "primary_re_min": 0.5,
                "primary_sn_min": 10.0,
                "sn_sky_var": 0.52699,
                "sn_gain": 10.7036,
                "psf_re": 0.52678,
                "truth_precision": "original Feather float64",
                "mask_metadata_sha256": "same-mask",
            },
        },
        "n_cases": 160,
        "n_pairs": n_pairs,
        "models": [
            {"tag": spec["tag"], "n_pairs": n_pairs}
            for spec in MODEL_SPECS
        ],
    }


def test_comparison_requires_exact_old_population_partition():
    training = payload("training", 80)
    validation = payload("validation", 20)
    reference = {
        "case_window": [40, 199],
        "v21_primary_domain": True,
        "n_pairs": 100,
    }
    checks = validate_inputs(training, validation, reference)
    assert checks["training_plus_validation_equals_old_all_rows"]
    bad = copy.deepcopy(reference)
    bad["n_pairs"] = 99
    with pytest.raises(RuntimeError, match="do not partition"):
        validate_inputs(training, validation, bad)


def test_split_name_and_boolean_must_agree():
    value = payload("training", 80)
    assert infer_split(value) == "training"
    value["design"]["official_train_value"] = False
    with pytest.raises(RuntimeError, match="disagree"):
        infer_split(value)
