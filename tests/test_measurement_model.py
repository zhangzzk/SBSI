import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.measurement_model import (
    ConditionalAffineFlow,
    MeasurementModelBundle,
    TargetStandardizer,
    add_measurement_target_features,
    raw_columns_for_measurement_targets,
)


def test_bundle_compiles_the_named_log_prob_method(monkeypatch):
    model = ConditionalAffineFlow(target_dim=2, context_dim=2, hidden_dim=4, n_layers=1, n_flows=1)
    bundle = MeasurementModelBundle(model, None, None)
    calls = []

    def fake_compile(function, **kwargs):
        calls.append(kwargs)

        def wrapped(*args, **inner_kwargs):
            return function(*args, **inner_kwargs)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)
    bundle.compile_log_prob(mode=None, dynamic=True)
    value = bundle.model.log_prob(torch.zeros(3, 2), torch.zeros(3, 2))
    assert value.shape == (3,)
    assert calls == [{"mode": None, "dynamic": True}]
    assert bundle.compile_log_prob(mode=None, dynamic=True) is bundle
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="different configuration"):
        bundle.compile_log_prob(mode="reduce-overhead", dynamic=True)


def test_measurement_target_features_are_finite_spin2_shape():
    frame = pd.DataFrame(
        {
            "measured_flux_auto": [10.0],
            "measured_flux_radius": [2.0],
            "measured_a_image": [3.0],
            "measured_b_image": [1.0],
            "measured_theta_image": [0.0],
        }
    )

    out = add_measurement_target_features(frame)

    assert np.isfinite(out["measured_log_flux_auto"]).all()
    assert np.isfinite(out["measured_log_flux_radius"]).all()
    np.testing.assert_allclose(out["measured_e1_image"], [0.5])
    np.testing.assert_allclose(out["measured_e2_image"], [0.0], atol=1.0e-15)


def test_raw_columns_for_engineered_measurement_targets():
    columns = raw_columns_for_measurement_targets(
        [
            "measured_log_flux_auto",
            "measured_e1_image",
        ]
    )

    assert columns == {
        "measured_flux_auto",
        "measured_a_image",
        "measured_b_image",
        "measured_theta_image",
    }


def test_target_standardizer_round_trips():
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [2.0, 4.0, 6.0],
        }
    )

    transform = TargetStandardizer.fit(frame, ["a", "b"])
    scaled = transform.transform_frame(frame)
    recovered = transform.inverse_transform_array(scaled)

    np.testing.assert_allclose(recovered, frame[["a", "b"]].to_numpy(dtype=np.float32))


def test_conditional_affine_flow_inverse_and_log_prob_are_finite():
    torch.manual_seed(5)
    model = ConditionalAffineFlow(
        target_dim=4,
        context_dim=3,
        hidden_dim=8,
        n_layers=1,
        n_flows=2,
    )
    x = torch.randn(6, 4)
    context = torch.randn(6, 3)

    z, inverse_logdet = model.inverse(x, context)
    recovered, forward_logdet = model.forward(z, context)
    log_prob = model.log_prob(x, context)
    samples = model.sample(context, n_samples=3)

    torch.testing.assert_close(recovered, x)
    torch.testing.assert_close(inverse_logdet + forward_logdet, torch.zeros_like(inverse_logdet))
    assert torch.isfinite(log_prob).all()
    assert samples.shape == (6, 3, 4)
