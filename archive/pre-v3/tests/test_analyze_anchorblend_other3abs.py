import numpy as np
import pandas as pd

from scripts.analyze_anchorblend_other3abs import BASE, NEW, summarize


def test_summarize_uses_case_means_and_reports_candidate_improvement():
    rows = []
    for case in range(4):
        for index in range(10):
            truth = 2.0 + 0.01 * index
            rows.append({
                "case": case, "input_index": index,
                "R_blend_truth": truth,
                BASE: truth - (0.4 + 0.01 * index),
                NEW: truth - (0.2 + 0.005 * index),
            })
    result = summarize(pd.DataFrame(rows), development_max=1)
    assert result["development"]["n_cases"] == 2
    assert result["validation"]["n_cases"] == 2
    assert abs(result["validation"]["other3abs_minus_truth"]["mean"]) < abs(
        result["validation"]["v22_minus_truth"]["mean"]
    )
    np.testing.assert_allclose(
        result["all"]["other3abs_minus_v22"]["mean"], 0.2225,
    )
