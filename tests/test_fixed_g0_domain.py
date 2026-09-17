import numpy as np
import pandas as pd
import pytest

from sbsi.fixed_g0_domain import (
    FLOW_FEATURES,
    FLUX_RADIUS_MIN_PIXELS,
    anchor_membership,
    fixed_g0_anchor_mask,
    make_flow_rows,
    matched_key_indices,
    packed_keys,
    response_primary_target_mask,
    split_group_values,
)
from sbsi.forward_catalogue import EmulatorPairingConfig, select_response_pairs
from sbsi.shear_map import apply_shear_to_ellipticity


def test_fixed_anchor_is_strict_and_has_no_truth_cut():
    assert FLUX_RADIUS_MIN_PIXELS == 3.0
    frame = pd.DataFrame(
        {
            "detected": [True, True, True, True, False],
            "measured_mag_auto": [25.79, 25.8, 20.0, 20.0, 20.0],
            "measured_flux_radius": [3.01, 4.0, FLUX_RADIUS_MIN_PIXELS, 4.0, 4.0],
            # Deliberately outside the former truth domain; these columns must
            # not affect fixed-g0 population membership.
            "r_input_p": [28.9, 20.0, 20.0, 28.9, 20.0],
            "Re_input_p": [0.02, 1.0, 1.0, 0.02, 1.0],
        }
    )
    assert fixed_g0_anchor_mask(frame).tolist() == [True, False, False, True, False]


def test_fixed_anchor_can_omit_flux_radius_science_cut():
    frame = pd.DataFrame(
        {
            "detected": [True, True, True, False],
            "measured_mag_auto": [25.79, 25.8, 20.0, 20.0],
            "measured_flux_radius": [2.0, 2.0, np.nan, 2.0],
        }
    )
    assert fixed_g0_anchor_mask(frame, radius_min_pixels=None).tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_response_primary_target_mask_is_structural_half_boundary():
    index = np.array([0, 4, 5, 9])
    assert response_primary_target_mask(index, 10).tolist() == [
        True,
        True,
        False,
        False,
    ]
    with pytest.raises(ValueError, match="outside"):
        response_primary_target_mask([10], 10)
    with pytest.raises(ValueError, match="finite integers"):
        response_primary_target_mask([0.5], 10)


def test_anchor_is_keyed_and_not_recut_on_sheared_measurements():
    anchor = np.sort(packed_keys([0, 1], [8, 3]))
    sheared = pd.DataFrame(
        {
            "case": [1, 0, 1],
            "input_index": [3, 8, 9],
            # Both anchored rows fail the measured cut on this leg. Membership
            # nevertheless follows the g=0 keys only.
            "measured_mag_auto": [30.0, 30.0, 20.0],
            "measured_flux_radius": [1.0, 1.0, 10.0],
        }
    )
    assert anchor_membership(sheared, anchor).tolist() == [True, True, False]


def test_exact_pair_matching_reports_unmatched_and_rejects_duplicates():
    li, ri, counts = matched_key_indices([0, 0, 1], [2, 4, 1], [1, 0, 2], [1, 2, 9])
    assert li.tolist() == [0, 2]
    assert ri.tolist() == [1, 0]
    assert counts == {
        "left_rows": 3,
        "right_rows": 3,
        "matched_rows": 2,
        "left_unmatched": 1,
        "right_unmatched": 1,
    }
    with pytest.raises(ValueError, match="duplicate key"):
        matched_key_indices([0, 0], [2, 2], [0], [2])


def test_flow_rows_fold_shear_and_only_drop_invalid_measurement():
    frame = pd.DataFrame(
        {
            "case": [0, 0],
            "input_index": [1, 2],
            "detected": [True, True],
            "e1_input_rot0_p": [0.2, 0.1],
            "e2_input_rot0_p": [0.1, 0.0],
            "gamma1_input_p": [0.05, 0.05],
            "gamma2_input_p": [0.0, 0.0],
            "sersic_n_input_p": [1.0, 2.0],
            "r_input_p": [28.9, 20.0],
            "Re_input_p": [0.02, 1.0],
            "axis_ratio_input_p": [0.5, 0.8],
            "nbr_flux_near": [0.0, 1.0],
            "nbr_flux_far": [1.0, 2.0],
            "nbr_flux_max": [0.5, 1.5],
            "measured_ngmix_g1": [0.3, -1.0],
            "measured_ngmix_g2": [0.2, -1.0],
            "measured_mag_auto": [26.5, 24.0],
            "measured_flux_radius": [1.0, 4.0],
        }
    )
    rows = make_flow_rows(frame)
    assert rows.input_index.tolist() == [1]
    assert rows.context.shape == (1, len(FLOW_FEATURES))
    expected = apply_shear_to_ellipticity(0.2, 0.1, 0.05, 0.0)
    np.testing.assert_allclose(rows.context[0, :2], expected)
    # The retained row is outside the former truth and measured cuts. This
    # helper applies neither; only its second row's invalid shape is dropped.
    assert rows.target[0, 2] == 1.0
    assert rows.usable_mask.tolist() == [True, False]


def test_group_split_is_reproducible_and_disjoint():
    train, validation = split_group_values(np.arange(10), 501, 0.2)
    train2, validation2 = split_group_values(np.arange(10), 501, 0.2)
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(validation, validation2)
    assert len(validation) == 2
    assert not (set(train) & set(validation))


def test_no_truth_cut_emulator_metadata_preserves_all_parent_pairs():
    pairs = pd.DataFrame(
        {
            "r_input_s": [28.9, 15.0],
            "r_input_p": [28.9, 15.0],
            "Re_input_s": [0.01, 9.0],
            "Re_input_p": [0.01, 9.0],
            "distance": [0.01, 9.9],
        }
    )
    assert select_response_pairs(pairs, None).equals(pairs)

    class Emulator:
        conditions = {
            "pixel_size": 0.2,
            "zero_point": 30.0,
            "psf_fwhm": 0.73,
            "moffat_beta": 2.224,
            "pixel_rms": 0.312,
        }
        select = {
            "regression": {
                "cuts": None,
                "truth_analysis_cuts_applied": False,
                "r_max": 10.0,
                "k": 20,
            }
        }

    config = EmulatorPairingConfig.from_emulator(Emulator())
    assert config.cuts is None
