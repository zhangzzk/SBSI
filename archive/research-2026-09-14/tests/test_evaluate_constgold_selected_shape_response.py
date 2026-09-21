from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_constgold_selected_shape_response import _prediction_sufficient


def _bundle(samples):
    return SimpleNamespace(
        model=type("ConditionalDiskAffineFlow", (), {})(),
        target_transform=SimpleNamespace(
            target_names=("measured_ngmix_g1", "measured_ngmix_g2")
        ),
        sample=lambda *args, **kwargs: samples.copy(),
    )


def test_selected_shape_adapter_does_not_reapply_photometry_cuts():
    samples = np.array(
        [
            [[0.2, 0.1], [0.4, -0.1]],
            [[-0.3, 0.2], [-0.1, 0.4]],
        ]
    )
    probability = np.array([0.75, 0.25])
    shift = np.array([[0.1, -0.05], [0.2, 0.3]])
    result, _ = _prediction_sufficient(
        _bundle(samples),
        pd.DataFrame({"id": [1, 2]}),
        detection_probability=probability,
        blend_shift=shift,
        draws=2,
        seed=1,
        batch_size=2,
        abs_shape_max=None,
        mag_max=25.8,
        radius_min_arcsec=0.75,
        truth_shape=np.array([[0.3, 0.0], [-0.1, 0.2]]),
        probability_factorization="direct_selected",
    )
    expected_means = samples.mean(axis=1) + shift
    np.testing.assert_allclose(
        result["complete"]["numerator"],
        (probability[:, None] * expected_means).sum(axis=0),
    )
    assert result["complete"]["denominator"] == pytest.approx(1.0)
    assert result["mean_pass_probability"] == pytest.approx(1.0)
    support = result["direct_selected_conditional_support"]
    assert support["zero_pass_row_count"] == 0
    np.testing.assert_array_equal(support["leg_mean_abs_bound"], [0.0, 0.0])


def test_selected_shape_adapter_rejects_an_extra_shape_cut():
    with pytest.raises(ValueError, match="no-\\|e\\|"):
        _prediction_sufficient(
            _bundle(np.zeros((1, 2, 2))),
            pd.DataFrame({"id": [1]}),
            detection_probability=np.array([0.5]),
            blend_shift=np.zeros((1, 2)),
            draws=2,
            seed=1,
            batch_size=1,
            abs_shape_max=0.6,
            probability_factorization="direct_selected",
        )
