from types import SimpleNamespace

import numpy as np
import torch

from scripts.probe_disk_likelihood_failure import exact_partition_probe
from sbsi.disk_inference_store import stencil
from scripts.summarize_disk_likelihood_failure import mixture_decomposition
from scripts.probe_disk_derivative_reference import derivatives, shear_context64


def test_partition_sum_effective_atoms_and_normalization():
    weights = np.array([.02, .04, .08, .16, .2, .1, .15, .25])
    h = .001

    def terms(observed, g1, g2, *, atom_indices, **kwargs):
        return torch.as_tensor((np.log(weights[atom_indices])+2*g1-3*g2-.5*(5*g1*g1+7*g2*g2))[None])

    likelihood = SimpleNamespace(disk_response=SimpleNamespace(n_atoms=len(weights)),
        log_importance_weights_tensor=terms,
        population_normalization=SimpleNamespace(log_mass=lambda x, y: .2*x))
    result = exact_partition_probe(likelihood, None, stencil([0., 0.], h)[1:], h, lambda *a: None, block=3)
    np.testing.assert_allclose(result['posterior_effective_atoms'], 1/(weights@weights))
    np.testing.assert_allclose(result['numerator']['score'], [2, -3], atol=1e-10)
    np.testing.assert_allclose(result['selected']['score'], [1.8, -3], atol=1e-10)
    np.testing.assert_allclose(result['numerator']['information'], np.diag([5, 7]), atol=2e-9)
    for bank in result['banks'].values():
        np.testing.assert_allclose(bank['relative_log_shape_to_full'], 0, atol=2e-15)
    np.testing.assert_allclose(result['banks']['(0, 2)']['log_values'][0], np.log(2*weights[::2].sum()))


def test_mixture_can_have_negative_information_without_invalid_components():
    h = .001
    def terms(observed, g1, g2, *, atom_indices, **kwargs):
        centers = np.array([-2., 2., -2., 2.])[atom_indices]
        return torch.as_tensor((-np.log(4)-.5*(g1-centers)**2-.5*g2*g2)[None])
    likelihood = SimpleNamespace(disk_response=SimpleNamespace(n_atoms=4),
        log_importance_weights_tensor=terms,
        population_normalization=SimpleNamespace(log_mass=lambda *p: 0.))
    result = exact_partition_probe(likelihood, None, stencil([0., 0.], h)[1:], h, lambda *a: None, block=3)
    np.testing.assert_allclose(result['numerator']['information'], np.diag([-3., 1.]), atol=3e-6)


def test_mixture_identity_separates_component_curvature_and_switching():
    h = .0001
    points = stencil([0., 0.], h)[1:]
    values = np.array([[-np.log(2)-.5*(x-mu)**2-.5*y*y for mu in (-2., 2.)] for x, y in points])
    result = mixture_decomposition(values, h)
    np.testing.assert_allclose(result['mean_component_information'], np.eye(2), atol=2e-7)
    np.testing.assert_allclose(result['component_score_covariance'], np.diag([4., 0.]), atol=2e-10)
    np.testing.assert_allclose(result['finite_step_information_residual'], 0., atol=2e-7)


def test_reference_shear_matches_numpy_and_retains_autograd():
    from sbsi.shear_map import apply_shear_to_ellipticity
    preprocessor = SimpleNamespace(feature_names=['e1_input_p', 'e2_input_p', 'other'], transform_tensor=lambda x: x)
    raw = torch.tensor([[.2, -.1, 7.]], dtype=torch.float64)
    g = torch.tensor([.0139, .00028], dtype=torch.float64, requires_grad=True)
    context = shear_context64(raw, g, preprocessor)
    expected = apply_shear_to_ellipticity(.2, -.1, .0139, .00028)
    np.testing.assert_allclose(context.detach().numpy()[0], [*expected, 7.], atol=1e-15)
    gradient, = torch.autograd.grad(context.sum(), g)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_autodiff_reference_recovers_quadratic():
    center = torch.tensor([.01, -.02], dtype=torch.float64)
    result = derivatives(lambda g: 2*g[0]-3*g[1]-.5*(5*g[0]**2+7*g[1]**2), center)
    np.testing.assert_allclose(result['score'], [1.95, -2.86])
    np.testing.assert_allclose(result['information'], np.diag([5., 7.]))
