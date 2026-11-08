import itertools
import math

import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.disk.precision import physical_log_prob_with_gap, transport_with_log_gap
from sbsi.measurement_model import (TargetStandardizer, build_flow,
    load_measurement_model, save_measurement_model)
from sbsi.selection_model import TabularPreprocessor


FEATURES = ["e1_input_p", "e2_input_p", "sersic_n_input_p", "r_input_p",
            "circularized_Re_input_p", "nbr_close_050", "nbr_close_100", "nbr_close_200"]
TARGETS = ["measured_ngmix_g1", "measured_ngmix_g2", "measured_flux_radius", "measured_flux_from_mag_auto"]


def fixture_model(invariant_features="bounded"):
    torch.manual_seed(611)
    cfg = dict(flow_type="sign_symmetric_physical_disk_affine",
        target_dim=4, context_dim=16, hidden_dim=12, n_layers=2, n_flows=4,
        disk_map="radial_tanh", disk_shape_means=[0., 0.], disk_shape_scales=[.3, .4],
        disk_coordinate_means=[0., 0.], disk_coordinate_scales=[.5, .7],
        physical_map="softplus", physical_means=[3., 100.], physical_scales=[1., 50.],
        physical_units=[1., 100.], physical_coordinate_means=[.2, -.1],
        physical_coordinate_scales=[.8, 1.2], invariant_features=invariant_features)
    model = build_flow(cfg).double()
    # Exercise a nonidentity flow, including nonzero odd/even shifts.
    with torch.no_grad():
        for layer in model.base.layers:
            layer.net[-1].weight.normal_(std=.08)
            layer.net[-1].bias.normal_(std=.04)
    return model, cfg


@pytest.mark.parametrize("signs", list(itertools.product((-1., 1.), repeat=2)))
@pytest.mark.parametrize("invariant_features", ["bounded", "asinh"])
def test_physical_generation_density_gradient_and_transport_equivariance(signs, invariant_features):
    model, _ = fixture_model(invariant_features)
    context = torch.randn(12, 16, dtype=torch.float64)
    context[:4, :2] = torch.tensor([[0, 0], [0, .3], [.2, 0], [1e-12, -1e-12]])
    context[:, 8:] = 0
    z = torch.randn(12, 4, dtype=torch.float64) * .4
    action = torch.tensor([*signs, 1., 1.], dtype=torch.float64)
    context_action = torch.tensor([*signs, *([1.] * 14)], dtype=torch.float64)
    reflected_context = context * context_action
    y, ld, gap = model.forward_physical(z, context, return_shape_log_gap=True)
    reflected_y, reflected_ld, reflected_gap = model.forward_physical(
        z * action, reflected_context, return_shape_log_gap=True)
    torch.testing.assert_close(reflected_y, y * action, rtol=0, atol=1e-13)
    torch.testing.assert_close(reflected_ld, ld, rtol=0, atol=1e-13)
    torch.testing.assert_close(reflected_gap, gap, rtol=0, atol=1e-13)
    means = torch.cat((model.disk_shape_means, model.physical_means))
    scales = torch.cat((model.disk_shape_scales, model.physical_scales))
    standardized = (y - means) / scales
    recovered, inverse_ld = model.inverse(standardized, context)
    torch.testing.assert_close(recovered, z, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(inverse_ld + ld - scales.log().sum(), torch.zeros_like(ld), atol=1e-11, rtol=0)
    logp = physical_log_prob_with_gap(model, y, gap, context)
    torch.testing.assert_close(logp, -.5*(z.square()+math.log(2*math.pi)).sum(-1)-ld, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(model.log_prob(standardized, context)-scales.log().sum(), logp, atol=1e-11, rtol=1e-11)
    reflected_logp = physical_log_prob_with_gap(model, reflected_y, reflected_gap, reflected_context)
    torch.testing.assert_close(reflected_logp, logp, rtol=0, atol=1e-12)
    # Scores at fixed observed measurements, not gradients through generation.
    c = context.detach().requires_grad_(True)
    cm = reflected_context.detach().requires_grad_(True)
    score = torch.autograd.grad(physical_log_prob_with_gap(model, y.detach(), gap.detach(), c).sum(), c)[0]
    reflected_score = torch.autograd.grad(physical_log_prob_with_gap(model, reflected_y.detach(), reflected_gap.detach(), cm).sum(), cm)[0]
    torch.testing.assert_close(reflected_score, score * context_action, rtol=1e-10, atol=1e-11)
    assert torch.isfinite(score).all()
    velocity = torch.tensor([[.02, -.01]], dtype=torch.float64).expand(12, -1)
    moved, moved_gap, moved_ld = transport_with_log_gap(y[:, :2], gap, velocity)
    moved_m, gap_m, ld_m = transport_with_log_gap(reflected_y[:, :2], reflected_gap, velocity * action[:2])
    torch.testing.assert_close(moved_m, moved * action[:2], atol=1e-13, rtol=0)
    torch.testing.assert_close(gap_m, moved_gap, atol=1e-13, rtol=0)
    torch.testing.assert_close(ld_m, moved_ld, atol=1e-13, rtol=0)


@pytest.mark.parametrize("invariant_features", ["bounded", "asinh"])
def test_full_physical_jacobian_and_training_step(invariant_features):
    model, _ = fixture_model(invariant_features)
    context = torch.randn(1, 16, dtype=torch.float64) * .2
    z = torch.tensor([.2, -.1, .4, -.5], dtype=torch.float64, requires_grad=True)
    jac = torch.autograd.functional.jacobian(lambda value: model.forward_physical(value[None], context)[0][0], z)
    y, ld = model.forward_physical(z[None], context)
    torch.testing.assert_close(torch.linalg.slogdet(jac).logabsdet, ld[0], rtol=1e-11, atol=1e-11)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    means = torch.cat((model.disk_shape_means, model.physical_means))
    scales = torch.cat((model.disk_shape_scales, model.physical_scales))
    loss = -model.log_prob((y.detach()-means)/scales, context).mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    reflected_context = context.clone()
    reflected_context[:, :2] *= -1
    action = torch.tensor([-1., -1., 1., 1.])
    actual = model.forward_physical(z[None]*action, reflected_context)[0]
    expected = model.forward_physical(z[None], context)[0]*action
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("invariant_features", ["bounded", "asinh"])
def test_checkpoint_preprocessing_and_contract(tmp_path, invariant_features):
    model, cfg = fixture_model(invariant_features)
    model.base.float()  # Physical coordinate buffers must remain float64.
    pre = TabularPreprocessor(FEATURES, np.zeros(8, np.float32), np.zeros(8, np.float32), np.ones(8, np.float32))
    target = TargetStandardizer(TARGETS, np.array([0., 0., 3., 100.]), np.array([.3, .4, 1., 50.]), "float64")
    path = tmp_path / "candidate.pt"
    save_measurement_model(path, model, pre, target, cfg)
    wrong_cfg = dict(cfg, invariant_features="asinh" if invariant_features == "bounded" else "bounded")
    with pytest.raises(ValueError, match="differs from config"):
        save_measurement_model(tmp_path / "wrong.pt", model, pre, target, wrong_cfg)
    bundle = load_measurement_model(path)
    assert bundle.model.invariant_features == invariant_features
    raw = np.random.default_rng(1).normal(size=(8, 8)).astype(np.float32)
    raw[0, :2] = np.nan
    mirrored = raw.copy()
    mirrored[:, :2] *= -1
    c = bundle.context_tensor(pd.DataFrame(raw, columns=FEATURES))
    cm = bundle.context_tensor(pd.DataFrame(mirrored, columns=FEATURES))
    torch.testing.assert_close(cm[:, :2], -c[:, :2], atol=0, rtol=0)
    torch.testing.assert_close(cm[:, 2:], c[:, 2:], atol=0, rtol=0)
    z = torch.randn(8, 4)
    action = torch.tensor([-1., -1., 1., 1.])
    y, ld, gap = bundle.model.forward_physical(z, c, return_shape_log_gap=True)
    ym, ldm, gapm = bundle.model.forward_physical(z*action, cm, return_shape_log_gap=True)
    torch.testing.assert_close(ym, y*action, atol=0, rtol=0)
    torch.testing.assert_close(ldm, ld, atol=0, rtol=0)
    torch.testing.assert_close(physical_log_prob_with_gap(bundle.model, ym, gapm, cm),
        physical_log_prob_with_gap(bundle.model, y, gap, c), atol=0, rtol=0)
    pre.means[0] = .01
    with pytest.raises(ValueError, match="zero mean"):
        save_measurement_model(path, model, pre, target, cfg)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["condition_preprocessor"]["fill_values"][1] = .1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="zero mean"):
        load_measurement_model(path)


def test_nonzero_physical_centres_rejected():
    _, cfg = fixture_model()
    for key in ("disk_coordinate_means", "disk_shape_means"):
        with pytest.raises(ValueError, match="requires zero"):
            build_flow(dict(cfg, **{key: [.01, 0.]}))


@pytest.mark.parametrize("invariant_features", ["bounded", "asinh"])
def test_extreme_internal_shape_coordinates_have_finite_training_gradients(invariant_features):
    model, _ = fixture_model(invariant_features)
    model.base.float()
    context = torch.randn(4, 16) * .2
    # Stress the base inverse directly: a small physical disk gap can encode
    # large internal coordinates, which must not cause quadratic explosions.
    x = torch.tensor([[1e4, -1e4, .1, .2], [-1e4, 1e4, -.2, .1],
                      [0., 1e4, .2, -.1], [1e4, 0., -.1, .2]])
    loss = -model.base.log_prob(x, context).mean()
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    assert torch.isfinite(norm)


def test_original_feature_map_is_the_checkpoint_default():
    _, cfg = fixture_model()
    cfg.pop("invariant_features")
    assert build_flow(cfg).invariant_features == "bounded"
    with pytest.raises(ValueError, match="invariant feature map"):
        build_flow(dict(cfg, invariant_features="unknown"))
