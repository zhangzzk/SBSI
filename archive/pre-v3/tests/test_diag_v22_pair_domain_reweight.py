import numpy as np
import pandas as pd

from scripts.diag_v22_pair_domain_reweight import (
    fit_density_odds, predict_odds, transformed_features, weight_diagnostics,
    weighted_case_estimate,
)


def test_density_reweight_recovers_shifted_feature_mean():
    rng = np.random.default_rng(4)
    n = 6000
    source = rng.normal(0.0, 1.0, (n, 7)).astype(np.float32)
    target = rng.normal(0.7, 1.0, (n, 7)).astype(np.float32)
    model, summary = fit_density_odds(
        source, target, seed=8, max_train_per_domain=5000,
    )
    weights = np.minimum(predict_odds(model, source), 10.0)
    assert summary["heldout_auc"] > 0.8
    assert np.average(source[:, 0], weights=weights) > 0.45


def test_weighted_case_estimate_uses_cases_as_independent_units():
    frame = pd.DataFrame({
        "case": [0, 0, 1, 1], "residual": [1.0, 3.0, 2.0, 4.0],
    })
    result = weighted_case_estimate(frame, np.ones(4), 5.0)
    assert result["mean"] == 12.5


def test_transformed_features_are_finite():
    frame = pd.DataFrame({
        "Re_input_p": [0.5], "Re_input_s": [0.2],
        "r_input_p": [24.0], "r_input_s": [25.0],
        "sersic_n_input_p": [1.0], "sersic_n_input_s": [2.0],
        "distance": [0.0],
    })
    assert np.isfinite(transformed_features(frame)).all()


def test_weight_diagnostics_accumulates_float32_in_float64():
    rng = np.random.default_rng(19)
    values = (25.0 + rng.normal(0, 0.2, (300_000, 7))).astype(np.float32)
    result = weight_diagnostics(values, values.copy(), np.ones(len(values)))
    assert result["max_abs_smd"] < 1.0e-10
