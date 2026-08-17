import numpy as np
import pandas as pd

from scripts.diag_v22_halfshear_distance import summarize


def test_summary_forms_response_ratio_inside_each_case():
    frame = pd.DataFrame({
        "case": [1, 1, 2, 2],
        "R_self": [1.0, 1.0, 2.0, 2.0],
        "R_flow": [0.5, 0.5, 1.0, 1.0],
        "gap": [0.5, 0.5, 1.0, 1.0],
    })
    result = summarize(frame, np.ones(len(frame), dtype=bool))
    np.testing.assert_allclose(result["m_self_percent"]["mean"], 100.0)
    np.testing.assert_allclose(result["gap_R_self_minus_R_flow"]["mean"], 0.75)
