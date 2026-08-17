import numpy as np
import pandas as pd

from scripts.analyze_v22_q3_purity import cross_fitted_predictions, metric_summary


def test_case_cross_fit_has_complete_predictions_and_finds_informative_feature():
    rows = []
    rng = np.random.default_rng(731)
    for case in range(10):
        for index in range(200):
            x = rng.uniform(-1, 1)
            rows.append({"case": case, "x": x, "junk": rng.normal(), "y": 2 * x})
    frame = pd.DataFrame(rows)
    predictions = cross_fitted_predictions(
        frame, "y", {"signal": ["x"], "junk": ["junk"]},
    )
    assert not predictions.isna().any().any()
    assert predictions.groupby(frame["case"])["fold"].nunique().max() == 1
    summary = metric_summary(frame, "y", predictions)
    assert summary["metrics"]["signal"]["mse"] < summary["metrics"]["constant"]["mse"]
    assert summary["metrics"]["signal"]["mse"] < summary["metrics"]["junk"]["mse"]
