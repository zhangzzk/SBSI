import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.selection_model import (
    SelectionMLP,
    SelectionModelBundle,
    SelectionModelEnsembleBundle,
    TabularPreprocessor,
)


def _constant_bundle(logit, *, feature="x"):
    frame = pd.DataFrame({feature: [0.0, 1.0, 2.0]})
    preprocessor = TabularPreprocessor.fit(
        frame, [feature], add_missing_indicators=False
    )
    model = SelectionMLP(input_dim=1, hidden_dim=2, n_layers=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.fill_(float(logit))
    return SelectionModelBundle(model, preprocessor)


def test_selection_model_ensemble_is_equal_probability_mean():
    first = _constant_bundle(0.0)
    second = _constant_bundle(np.log(3.0))
    ensemble = SelectionModelEnsembleBundle([first, second])

    frame = pd.DataFrame({"x": [-2.0, 4.0]})
    np.testing.assert_allclose(ensemble.predict_proba(frame), [0.625, 0.625])
    assert ensemble.preprocessor.feature_names == ["x"]
    assert ensemble.metadata["aggregation"] == "arithmetic_mean_probability"


def test_selection_model_ensemble_rejects_different_preprocessors():
    with pytest.raises(ValueError, match="different preprocessors"):
        SelectionModelEnsembleBundle(
            [_constant_bundle(0.0, feature="x"), _constant_bundle(0.0, feature="y")]
        )
