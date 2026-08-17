import numpy as np
import pandas as pd

from scripts.fit_anchorblend_scene_correction import response_summary


def test_response_summary_uses_case_means_and_fixed_edges():
    frame = pd.DataFrame({
        "case": [1, 1, 2, 2],
        "R_blend_truth": [0.2, 0.4, 0.3, 0.5],
        "R_blend_lsst_r_extnbr_v22": [0.1, 0.2, 0.2, 0.4],
    })
    frame.attrs["constant_correction"] = 0.05
    result = response_summary(
        frame, np.asarray([0.1, 0.1, 0.1, 0.1]),
        np.asarray([-np.inf, 0.25, np.inf]),
    )
    assert abs(result["raw"]["mean"] + 0.125) < 1e-12
    assert abs(result["corrected"]["mean"] - (-0.025)) < 1e-12
    assert result["n_cases"] == 2
