import numpy as np
import pandas as pd

from scripts.analyze_anchorblend_local10 import split_additive_aperture_test


def test_split_additive_aperture_test_fixes_only_from_development_cases():
    case = pd.DataFrame(
        {
            "R_blend_truth_all": [0.11, 0.13, 0.12, 0.14],
            "R_blend_truth_local": [0.10, 0.12, 0.11, 0.13],
            "R_blend_lsst_r_extnbr_v22_all": [0.10, 0.12, 0.11, 0.13],
        },
        index=pd.Index([200, 201, 202, 203], name="case"),
    )
    got = split_additive_aperture_test(case, development_max=201)
    assert np.isclose(got["fixed_additive_response"], 0.01)
    assert np.isclose(
        got["validation_corrected_v22_minus_all_truth"]["mean"], 0.0,
    )
    assert got["development_cases"] == [200, 201]
    assert got["validation_cases"] == [202, 203]
    assert all(got["gates"].values())


def test_split_additive_aperture_test_requires_both_sides():
    case = pd.DataFrame(
        {
            "R_blend_truth_all": [0.11, 0.12, 0.13],
            "R_blend_truth_local": [0.10, 0.11, 0.12],
            "R_blend_lsst_r_extnbr_v22_all": [0.10, 0.11, 0.12],
        },
        index=[200, 201, 202],
    )
    try:
        split_additive_aperture_test(case, development_max=201)
    except RuntimeError as error:
        assert "leaves 2/1 cases" in str(error)
    else:
        raise AssertionError("an undersized validation split must fail")
