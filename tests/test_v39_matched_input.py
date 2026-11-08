import numpy as np
import pandas as pd
import pytest

from _script_loader import load_script_module

module = load_script_module('v39_matched_input.py')
CONDITIONS = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73, moffat_beta=2.224, pixel_rms=.312)


def scene():
    return pd.DataFrame(dict(index_input=[31, 8, 17], RA_input=np.array([0., 1., 20.])/3600.,
        DEC_input=[0., 0., 0.], redshift_input=[.5]*3, Re_input=[.8, 1.2, .6],
        axis_ratio_input=[.5, 1., .8], position_angle_input=[0., 30., 45.],
        sersic_n_input=[1., 2., 3.], r_input=[27., 19., 26.5]))


def test_preserves_requested_identity_order_and_complete_neighbour_context():
    zero, pairs, primary = module.build_case_inputs(scene(), [17, 31], CONDITIONS, workers=1)
    assert primary['index'].tolist() == [17, 31]
    assert zero.r_input_p.tolist() == [26.5, 27.]  # no hidden bright/truth cut
    np.testing.assert_allclose(zero.circularized_Re_input_p, [.6*np.sqrt(.8), .8*np.sqrt(.5)])
    assert zero.nbr_close_100.iloc[0] == 0
    assert zero.nbr_close_100.iloc[1] == pytest.approx(np.log1p(10**3.2*np.exp(-.5)), rel=1.e-7)
    assert pairs.index_input_p.tolist() == [31]
    assert pairs.index_input_s.tolist() == [8]  # neighbour outside requested targets retained
    np.testing.assert_allclose(zero[['e1_input_p', 'e2_input_p']], [[0., 1/9], [1/3, 0.]], atol=1.e-15)


def test_missing_and_duplicate_inputs_are_rejected():
    with pytest.raises(ValueError, match='absent'):
        module.build_case_inputs(scene(), [999], CONDITIONS, workers=1)
    with pytest.raises(ValueError, match='duplicate requested'):
        module.build_case_inputs(scene(), [31, 31], CONDITIONS, workers=1)
    raw = scene()
    raw.loc[1, 'index_input'] = 31
    with pytest.raises(ValueError, match='duplicate input'):
        module.build_case_inputs(raw, [31], CONDITIONS, workers=1)


def test_padding_keeps_original_groups_but_marks_only_mock_rows_for_reduction():
    measured = pd.DataFrame({'value': [10., 11., 12., 13.]})
    truth = pd.DataFrame(dict(parent_sample_row=[0, 1, 2, 3], source_case=[40]*4,
        source_input_index=[100, 101, 102, 103], source_detection_id=[50, 51, 52, 53]))
    mock_truth = truth.iloc[[1, 3]].reset_index(drop=True)
    mock = pd.DataFrame({'value': [21., 23.]})
    out, ids = module.evaluation_catalogue(measured, truth, mock, mock_truth)
    assert out.value.tolist() == [10., 21., 12., 23.]
    assert ids.numerical_padding.tolist() == [True, False, True, False]
    assert ids.loc[~ids.numerical_padding, 'parent_sample_row'].tolist() == [1, 3]
    assert measured.value.tolist() == [10., 11., 12., 13.]
    mock_truth.loc[0, 'source_input_index'] = 999
    with pytest.raises(AssertionError):
        module.evaluation_catalogue(measured, truth, mock, mock_truth)
