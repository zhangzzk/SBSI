import numpy as np
import pandas as pd

from scripts.plot_v22_coherent_gap_vs_measured_response import (
    case_balanced_histogram,
    measured_response_deciles,
    response_distribution_summary,
)


def synthetic_anchors():
    rows = []
    for case in range(4):
        for response in (-2.0, -1.0, 1.0, 2.0):
            gap = response - 0.1 * case
            rows.append({
                "case": case,
                "R_blend_truth": response,
                "gap_truth_minus_raw": gap,
                "gap_truth_minus_demo": gap - 0.2,
            })
    return pd.DataFrame(rows)


def test_measured_response_deciles_use_cases_for_errors():
    frame = synthetic_anchors()
    rows, edges = measured_response_deciles(frame, {"demo": {}}, 4)
    assert len(rows) == 4 and len(edges) == 5
    assert sum(item["n_anchors"] for item in rows) == len(frame)
    assert all(item["n_cases_with_anchors"] == 4 for item in rows)
    np.testing.assert_allclose(
        [item["models"]["demo"]["mean"] for item in rows],
        [item["raw_gap_truth_minus_prediction"]["mean"] - 0.2 for item in rows],
    )


def test_response_histogram_and_summary_close():
    frame = synthetic_anchors()
    mean, sem = case_balanced_histogram(
        frame, np.asarray([-3.0, -1.5, 0.0, 1.5, 3.0]),
    )
    np.testing.assert_allclose(mean.sum(), 1.0, atol=1e-15)
    np.testing.assert_allclose(sem, 0.0, atol=1e-15)
    summary = response_distribution_summary(frame)
    assert summary["n_anchors"] == len(frame)
    assert summary["case_balanced_mean"]["n_cases"] == 4
    assert summary["display_limits"][0] == -summary["display_limits"][1]
