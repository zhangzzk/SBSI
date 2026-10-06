import numpy as np
import pytest

from scripts.truth_parent_cohorts import COHORTS, parent_cohort_masks, resolve_cohorts


def test_legacy_defaults_preserve_all_three_cohorts():
    stored = {"input_index": np.array([4, 8, 12]),
              "g0_usable": np.array([True, False, True]),
              "g0_selected": np.array([False, False, True])}
    masks = parent_cohort_masks(stored, resolve_cohorts({}))
    assert tuple(masks) == COHORTS
    np.testing.assert_array_equal(masks["truth_parent"], [True, True, True])
    np.testing.assert_array_equal(masks["usable_g0"], stored["g0_usable"])
    np.testing.assert_array_equal(masks["fixed_cut_g0"], stored["g0_selected"])


def test_no_g0_parent_requires_explicit_truth_only_request(tmp_path):
    manifest = {"cohorts": ["truth_parent"], "g0_diagnostics_available": False}
    with pytest.raises(ValueError, match="unavailable"):
        resolve_cohorts(manifest)
    requested = resolve_cohorts(manifest, ["truth_parent"])
    path = tmp_path / "parent.npz"
    np.savez(path, input_index=np.array([4, 8]), context=np.zeros((2, 8)))
    with np.load(path, allow_pickle=False) as stored:
        masks = parent_cohort_masks(stored, requested)
    assert tuple(masks) == ("truth_parent",)
    np.testing.assert_array_equal(masks["truth_parent"], [True, True])


def test_false_availability_flag_rejects_inconsistent_manifest():
    with pytest.raises(ValueError, match="unavailable"):
        resolve_cohorts({"cohorts": COHORTS, "g0_diagnostics_available": False})


@pytest.mark.parametrize("requested", [[], ["truth_parent", "truth_parent"], ["invented"]])
def test_invalid_cohort_requests_rejected(requested):
    with pytest.raises(ValueError):
        resolve_cohorts({}, requested)


@pytest.mark.parametrize("mask", [[1, 0], [[True, False]], [True]])
def test_observed_masks_must_be_boolean_and_aligned(mask):
    with pytest.raises(ValueError, match="boolean mask aligned"):
        parent_cohort_masks({"input_index": [4, 8], "g0_usable": mask}, ["usable_g0"])


def test_missing_observed_mask_is_not_replaced_by_truth_parent():
    with pytest.raises(ValueError, match="requires parent field"):
        parent_cohort_masks({"input_index": [4, 8]}, ["usable_g0"])


def test_fixed_cut_must_be_subset_of_usable_g0():
    with pytest.raises(ValueError, match="subset"):
        parent_cohort_masks({"input_index": [4, 8], "g0_usable": [True, False],
                            "g0_selected": [True, True]}, COHORTS)
