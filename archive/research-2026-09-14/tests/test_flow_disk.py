import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.integrate import quad

from sbsi.flow_disk import radial_atanh, radial_tanh
from sbsi.flow_training import complete_flow_context_means
from sbsi.measurement_model import (
    MeasurementModelBundle, TargetStandardizer, build_flow,
    load_measurement_model, save_measurement_model,
)
from sbsi.selection_model import TabularPreprocessor


def _config():
    return dict(flow_type="disk_affine", target_dim=4, context_dim=2,
                hidden_dim=8, n_layers=1, n_flows=2, scale_limit=1.,
                disk_map="radial_tanh", disk_shape_means=[.01, -.02],
                disk_shape_scales=[.3, .4], disk_coordinate_means=[.03, -.04],
                disk_coordinate_scales=[.5, .6])


def _target(dtype="float64"):
    return TargetStandardizer(
        ["measured_ngmix_g1", "measured_ngmix_g2", "mag", "logr"],
        np.array([.01, -.02, 23., -.3]), np.array([.3, .4, 2., .7]), dtype=dtype,
    )


def test_two_target_disk_flow_roundtrip_and_checkpoint(tmp_path):
    config = dict(
        _config(),
        target_dim=2,
        disk_shape_means=[0.01, -0.02],
        disk_shape_scales=[0.3, 0.4],
    )
    target = TargetStandardizer(
        ["measured_ngmix_g1", "measured_ngmix_g2"],
        np.array([0.01, -0.02]),
        np.array([0.3, 0.4]),
        dtype="float64",
    )
    frame = pd.DataFrame({"c0": [0.0, 1.0], "c1": [2.0, 3.0]})
    preprocessor = TabularPreprocessor.fit(
        frame, ["c0", "c1"], add_missing_indicators=False
    )
    model = build_flow(config)
    context = torch.tensor([[0.2, -0.1], [-0.3, 0.4]])
    z = torch.tensor([[0.1, -0.2], [0.4, 0.3]])
    y, logdet = model.forward(z, context)
    restored, inverse_logdet = model.inverse(y, context)
    torch.testing.assert_close(restored, z, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        logdet + inverse_logdet, torch.zeros_like(logdet), rtol=2e-6, atol=2e-6
    )
    path = tmp_path / "selected_shape.pt"
    save_measurement_model(path, model, preprocessor, target, config)
    bundle = load_measurement_model(path)
    draw = bundle.sample(frame, n_samples=4, qmc=True)
    assert draw.shape == (2, 4, 2)
    assert np.all(np.sum(draw**2, axis=-1) < 1.0)


@pytest.mark.parametrize("point", [[0., 0.], [1.e-8, -2.e-8], [.2, -.8], [3., 4.]])
def test_radial_bijection_jacobian_and_gradients(point):
    x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    y, ld = radial_tanh(x)
    back, inverse_ld = radial_atanh(y)
    torch.testing.assert_close(back, x, rtol=2.e-10, atol=1.e-13)
    torch.testing.assert_close(ld + inverse_ld, torch.zeros_like(ld), atol=2.e-10, rtol=0)
    jac = torch.autograd.functional.jacobian(lambda a: radial_tanh(a)[0], x)
    torch.testing.assert_close(torch.linalg.slogdet(jac).logabsdet, ld, atol=2.e-12, rtol=2.e-12)
    assert torch.autograd.gradcheck(radial_tanh, x)
    assert torch.autograd.gradcheck(radial_atanh, y.detach().requires_grad_(), eps=1.e-8, atol=1.e-4, rtol=1.e-4)


def test_radial_rotation_and_representable_boundary_precision():
    x = torch.tensor([[.4, -.7], [6., 7.], [1.e-9, 2.e-9]], dtype=torch.float64)
    angle = .87
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]], dtype=torch.float64)
    y, ld = radial_tanh(x)
    ry, rld = radial_tanh(x @ rotation.T)
    torch.testing.assert_close(ry, y @ rotation.T, atol=3.e-16, rtol=1.e-15)
    torch.testing.assert_close(ld, rld, atol=1.e-14, rtol=1.e-14)
    original = torch.tensor([[.999999986571655, 0.]], dtype=torch.float64)
    u, _ = radial_atanh(original)
    restored, _ = radial_tanh(u)
    torch.testing.assert_close(restored, original, rtol=0, atol=2.e-16)
    assert original.float()[0, 0] == 1  # Explicit reason the density needs raw float64.
    with pytest.raises(ValueError, match="strictly interior"):
        radial_atanh(original.float())
    # No artificial contraction of draws where tanh rounds to one.
    saturated, logdet = radial_tanh(torch.tensor([[30., 0.]], dtype=torch.float64))
    assert saturated[0, 0] == 1 and torch.isfinite(logdet).all()
    with pytest.raises(ValueError):
        radial_atanh(saturated)


def test_independent_disk_density_normalization_and_radial_cdf():
    # A standard 2D Gaussian mapped to the disk has radius tanh(R), where R
    # is Rayleigh. Integrate its density in polar *physical* coordinates.
    def integrand(s):
        u, inverse_ldj = radial_atanh(torch.tensor([s, 0.], dtype=torch.float64))
        logp = -.5 * u.square().sum() - math.log(2 * math.pi) + inverse_ldj
        return 2 * math.pi * s * float(torch.exp(logp))
    for limit in (.3, .9, .99999, 1.):
        actual, error = quad(integrand, 0., limit, epsabs=1.e-9, epsrel=1.e-9)
        expected = 1. if limit == 1. else -math.expm1(-.5 * math.atanh(limit) ** 2)
        assert abs(actual - expected) < 5.e-9
        assert error < 5.e-9


def test_wrapped_density_matches_full_jacobian_and_preserves_photometry():
    torch.manual_seed(88)
    model = build_flow(_config()).double()
    with torch.no_grad():
        for layer in model.base.layers:
            layer.net[-1].weight.normal_(std=.03)
    context = torch.tensor([[.1, -.2]], dtype=torch.float64)
    z = torch.tensor([[.3, -.4, .7, 1.]], dtype=torch.float64, requires_grad=True)
    y, ld = model.forward(z, context)
    back, inverse_ld = model.inverse(y, context)
    torch.testing.assert_close(back, z, rtol=2.e-12, atol=2.e-12)
    torch.testing.assert_close(inverse_ld, -ld, rtol=2.e-12, atol=2.e-12)
    jac = torch.autograd.functional.jacobian(lambda a: model.forward(a[None], context)[0][0], z[0])
    torch.testing.assert_close(torch.linalg.slogdet(jac).logabsdet, ld[0], rtol=2.e-12, atol=2.e-12)
    expected = -.5 * (z.square() + math.log(2 * math.pi)).sum(-1) - ld
    torch.testing.assert_close(model.log_prob(y, context), expected, rtol=2.e-12, atol=2.e-12)
    base_y, _ = model.base.forward(z, context)
    torch.testing.assert_close(y[:, 2:], base_y[:, 2:], rtol=0, atol=0)


def test_invalid_density_is_zero_without_poisoning_valid_gradients():
    model = build_flow(_config())
    target = _target()
    raw = np.array([[.999999986571655, 0., 24., .2], [1., 0., 24., .2],
                    [1.1, .2, 24., .2], [np.nan, 0., 24., .2]])
    y = torch.tensor(target.transform_array(raw), requires_grad=True)
    lp = model.log_prob(y, torch.zeros(4, 2))
    assert torch.isfinite(lp[0]) and torch.isneginf(lp[1:]).all()
    lp[0].backward()
    assert torch.isfinite(y.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="float64"):
        model.log_prob(y.float(), torch.zeros(4, 2))
    with pytest.raises(ValueError, match="buffers"):
        model.float().forward(torch.zeros(1, 4), torch.zeros(1, 2))


def test_response_sampler_maps_each_draw_before_averaging_and_backpropagates():
    torch.manual_seed(89)
    model = build_flow(_config())
    with torch.no_grad():
        for layer in model.base.layers:
            layer.net[-1].weight.normal_(std=.03)
    contexts = [torch.tensor([[.3, -.2], [.5, .1]]), torch.tensor([[.35, -.2], [.55, .1]])]
    means, draws = complete_flow_context_means(model, contexts, 16, seed=13, return_samples=True)
    for mean, sample in zip(means, draws):
        physical = sample[..., :2] * model.disk_shape_scales + model.disk_shape_means
        assert torch.all(physical.square().sum(-1) < 1)
        torch.testing.assert_close(mean, sample.mean(1), rtol=0, atol=0)
        u, _ = radial_atanh(physical)
        wrong_mean, _ = radial_tanh(u.mean(1))
        assert torch.max(torch.abs(wrong_mean - physical.mean(1))) > 1.e-4
    loss = (means[1] - means[0]).square().sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_precision_metadata_roundtrip_and_legacy_standardization_unchanged(tmp_path):
    target = _target()
    config = _config()
    frame = pd.DataFrame({"c0": [0., 1., 2.], "c1": [2., 4., 3.]})
    preprocessor = TabularPreprocessor.fit(frame, ["c0", "c1"], add_missing_indicators=False)
    raw = np.array([[.999999986571655, 0., 24., .2]] * 3)
    for name, column in zip(target.target_names, raw.T):
        frame[name] = column
    model = build_flow(config)
    bundle = MeasurementModelBundle(model, preprocessor, target)
    assert bundle.target_tensor(frame).dtype == torch.float64
    path = tmp_path / "disk.pt"
    save_measurement_model(path, model, preprocessor, target, config)
    restored = load_measurement_model(path)
    assert restored.target_transform.dtype == "float64"
    np.testing.assert_array_equal(bundle.log_prob(frame), restored.log_prob(frame))
    torch.manual_seed(90)
    first = bundle.sample(frame, 4)
    torch.manual_seed(90)
    np.testing.assert_array_equal(first, restored.sample(frame, 4))
    assert first.dtype == np.float64 and restored.sample(frame.iloc[:0]).dtype == np.float64
    legacy = TargetStandardizer.fit(frame, target.target_names)
    legacy_values = frame[target.target_names].to_numpy(np.float32)
    expected = ((legacy_values - legacy.means) / legacy.scales).astype(np.float32)
    np.testing.assert_array_equal(legacy.transform_frame(frame), expected)
    assert legacy.dtype == "float32" and "dtype" not in legacy.to_state()
    assert TargetStandardizer.from_state(legacy.to_state()).dtype == "float32"
    with pytest.raises(ValueError, match="float64"):
        MeasurementModelBundle(model, preprocessor, _target("float32"))
    wrong = dict(config, disk_shape_scales=[.7, .8])
    with pytest.raises(ValueError, match="config disagree"):
        save_measurement_model(path, model, preprocessor, target, wrong)


def test_checkpoint_cannot_silently_mislabel_disk_coordinates(tmp_path):
    model = build_flow(_config())
    target = _target()
    preprocessor = TabularPreprocessor.fit(pd.DataFrame({"a": [1., 2.]}), ["a"])
    path = tmp_path / "disk.pt"
    save_measurement_model(path, model, preprocessor, target, _config())
    state = torch.load(path, weights_only=False)
    state["model_config"]["disk_coordinate_scales"] = [4., 5.]
    torch.save(state, path)
    with pytest.raises(ValueError, match="config disagree"):
        load_measurement_model(path)
    with pytest.raises(ValueError, match="disk coordinate metadata requires"):
        build_flow(dict(_config(), flow_type="typo_disk_affine"))
    with pytest.raises(ValueError, match="disk coordinate metadata requires"):
        build_flow(dict(_config(), flow_type="affine"))
    with pytest.raises(ValueError, match="no mean head"):
        build_flow(dict(_config(), mean_hidden=8))
