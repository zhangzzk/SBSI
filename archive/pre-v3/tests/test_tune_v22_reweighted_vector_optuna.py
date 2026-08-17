import numpy as np

from scripts.tune_v22_reweighted_vector_optuna import (
    VectorEvaluation,
    positive_response_weights,
)


def test_positive_response_weights_ignore_negative_predictions_and_normalize():
    train = np.asarray([-3.0, 0.0, 1.0, 2.0], dtype=float)
    validation = np.asarray([-9.0, 1.0], dtype=float)
    train_weight, validation_weight, metadata = positive_response_weights(
        train, validation, alpha=0.1, cap=50.0
    )
    assert np.isclose(train_weight.mean(), 1.0)
    assert np.isclose(train_weight[0], train_weight[1])
    assert np.isclose(validation_weight[0], train_weight[0])
    assert metadata["train_mean_prediction_power"] == 1.25


def test_vector_evaluation_recovers_known_projection_slope():
    case = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    primary = np.asarray([10, 10, 11, 11, 20, 20, 21, 21])
    angle = np.asarray([0.0, 45.0, 0.0, 45.0] * 2)
    cosine = np.cos(np.deg2rad(2.0 * angle))
    sine = np.sin(np.deg2rad(2.0 * angle))
    measured1 = np.asarray([2.0, 2.0, 4.0, 4.0, 3.0, 3.0, 6.0, 6.0])
    measured2 = np.asarray([1.0, 1.0, -2.0, -2.0, 0.5, 0.5, -1.0, -1.0])
    label = measured1 * cosine + measured2 * sine
    null = -measured1 * sine + measured2 * cosine
    evaluator = VectorEvaluation(case, primary, angle, label, null)
    # One prediction per direction.  The resulting scene vector is exactly
    # half the measured scene vector, hence measured-on-predicted slope = 2.
    prediction = np.asarray([1.0, 0.5, 2.0, -1.0, 1.5, 0.25, 3.0, -0.5])
    result = evaluator.score(prediction, (0, 1))
    assert np.isclose(
        result["slope_measured_on_predicted"]["mean"], 2.0
    )
    assert np.isclose(result["orthogonal_slope_null"]["mean"], 0.0)
