import numpy as np
import pandas as pd

from scripts.analyze_v22_nonnegative_pairs_coherent import summarize_population


def test_nonnegative_pair_summary_closes_by_case():
    rows = []
    for case, truth in zip(range(4), [0.2, 0.3, 0.4, 0.5], strict=True):
        negative = -0.4 - 0.01 * case
        positive = 0.6 + 0.02 * case
        raw = negative + positive
        rows.append({
            "case": case,
            "R_blend_truth": truth,
            "prediction_raw": raw,
            "prediction_nonnegative": positive,
            "negative_pair_sum": negative,
            "positive_pair_sum": positive,
            "clipping_increment": -negative,
            "gap_truth_minus_raw": truth - raw,
            "gap_truth_minus_nonnegative": truth - positive,
            "n_pairs": 3.0,
            "n_negative_pairs": 1.0,
            "has_negative_pair": 1.0,
        })
    result = summarize_population(pd.DataFrame(rows))
    assert result["n_cases"] == 4
    assert result["n_anchors"] == 4
    np.testing.assert_allclose(
        result["raw_prediction"]["mean"],
        result["negative_pair_sum"]["mean"]
        + result["positive_pair_sum"]["mean"],
    )
    np.testing.assert_allclose(
        result["nonnegative_pair_prediction"]["mean"]
        - result["raw_prediction"]["mean"],
        result["clipping_increment"]["mean"],
    )
    np.testing.assert_allclose(
        result["nonnegative_gap_truth_minus_prediction"]["mean"],
        result["raw_gap_truth_minus_prediction"]["mean"]
        - result["clipping_increment"]["mean"],
    )
