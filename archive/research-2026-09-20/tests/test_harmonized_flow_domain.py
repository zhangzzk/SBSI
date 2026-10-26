import numpy as np
import pandas as pd
import pytest

from scripts.prepare_harmonized_flow_domain import generated_secondary_parent, independent_measurement_rows


def test_each_measurement_leg_keeps_its_own_success_population():
    ids = np.array([11, 12, 13])
    context, gamma = np.zeros((3, 8)), np.zeros((3, 2))
    zero = np.array([[.1, .2, 4, 50], [np.nan, np.nan, 4, 50], [.1, .2, 4, 50]])
    shear = np.array([[np.nan, np.nan, 4, 50], [.2, .3, 4, 50], [.2, .3, 4, 50]])
    a = independent_measurement_rows(0, ids, context, gamma, zero)
    b = independent_measurement_rows(0, ids, context, gamma, shear)
    np.testing.assert_array_equal(a.input_index, [11, 13])
    np.testing.assert_array_equal(b.input_index, [12, 13])
    np.testing.assert_array_equal(a.usable_mask, [True, False, True])
    np.testing.assert_array_equal(b.usable_mask, [False, True, True])


def test_physical_validity_does_not_apply_a_science_cut():
    values = np.array([[.1, .2, .1, .01], [1, 0, 4, 50], [.1, .2, 0, 50], [.1, .2, 4, np.inf]])
    rows = independent_measurement_rows(0, np.arange(4), np.zeros((4, 8)), np.zeros((4, 2)), values)
    np.testing.assert_array_equal(rows.input_index, [0])


def test_generated_role_boundary_survives_rendered_row_loss():
    generated = pd.DataFrame({'index': np.arange(8), 'r': [25., 25., 25., 25., 25., 25., 26., 25.]})
    ids, rows, report = generated_secondary_parent(generated, [1, 2, 3, 4, 6, 7], [4, 5, 7])
    np.testing.assert_array_equal(ids, [4, 7])
    np.testing.assert_array_equal(rows, [3, 5])
    assert report['generated_secondary_boundary'] == 4
    assert report['unrendered_truth_parent_ids'] == [5]
    assert report['unrendered_truth_parent_dropped'] == 1
    # Rendered len//2 would wrongly include primary ID3.
    with pytest.raises(AssertionError):
        generated_secondary_parent(generated, [1, 2, 3, 4, 6, 7], [3, 4, 5, 7])
