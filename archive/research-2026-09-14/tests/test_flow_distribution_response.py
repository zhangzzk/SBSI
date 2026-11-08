import itertools

import pytest
import torch

from sbsi.flow_distribution_response import (
    GaussianFourierFeatures, draw_feature_secant, feature_secant,
    independent_response_score,
)


def test_features_preserve_rng_have_unit_norm_and_translation_invariant_kernel():
    torch.manual_seed(72)
    state = torch.random.get_rng_state().clone()
    phi = GaussianFourierFeatures(2, seed=19)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    x = torch.tensor([[0.1, -0.7], [0.8, 0.2]], dtype=torch.double)
    torch.testing.assert_close(phi(x).square().sum(-1), torch.ones(2, dtype=torch.double))
    shifted = x + torch.tensor([2.3, -1.2], dtype=torch.double)
    torch.testing.assert_close(phi(x) @ phi(x).T, phi(shifted) @ phi(shifted).T)
    torch.testing.assert_close(phi.frequencies, GaussianFourierFeatures(2, seed=19).frequencies)


def test_distribution_secant_detects_spread_change_with_identical_radius_means():
    phi = GaussianFourierFeatures(1, seed=19)
    zero = torch.tensor([[[-1.], [1.]]], dtype=torch.double)
    scale = torch.tensor(1.2, dtype=torch.double, requires_grad=True)
    one = zero * scale
    torch.testing.assert_close(zero.mean(1), one.mean(1))
    actual = draw_feature_secant(phi, zero, one, .05)
    # Transforming the mean would lose this entire response.
    wrong = feature_secant(phi(zero.mean(1)), phi(one.mean(1)), .05)
    assert float(actual.square().sum().detach()) > 0.1
    torch.testing.assert_close(wrong, torch.zeros_like(wrong))
    actual.square().sum().backward()
    assert float(scale.grad) > 0


def test_joint_secant_detects_association_change_with_identical_marginals():
    zero = torch.tensor([[[-1., -1.], [1., 1.]]], dtype=torch.double)
    one = torch.tensor([[[-1., 1.], [1., -1.]]], dtype=torch.double)
    marginal = GaussianFourierFeatures(1, seed=1501)
    for j in range(2):
        secant = draw_feature_secant(marginal, zero[:, :, j:j+1], one[:, :, j:j+1], .05)
        torch.testing.assert_close(secant, torch.zeros_like(secant))
    joint = draw_feature_secant(GaussianFourierFeatures(2, seed=1501), zero, one, .05)
    assert float(joint.square().sum()) > 1.


def test_independent_score_has_no_model_sampling_variance_or_label_noise_penalty():
    mu = torch.tensor(.6, dtype=torch.double, requires_grad=True)
    sigma = torch.tensor(.9, dtype=torch.double, requires_grad=True)
    truth = .3
    # Enumerate independent signs exactly: no stochastic pass/fail tolerance.
    signs = torch.tensor(list(itertools.product([-1., 1.], repeat=3)), dtype=torch.double)
    a = (mu + sigma * signs[:, 0])[:, None]
    b = (mu + sigma * signs[:, 1])[:, None]
    t = (truth + .7 * signs[:, 2])[:, None]
    score = independent_response_score(a, b, t).mean()
    torch.testing.assert_close(score, (mu-truth).square()-truth**2)
    grad_mu, grad_sigma = torch.autograd.grad(score, (mu, sigma))
    torch.testing.assert_close(grad_mu, 2*(mu-truth))
    torch.testing.assert_close(grad_sigma, torch.zeros_like(sigma))
    # Reusing the same group WOULD incorrectly penalize its sampling variance.
    wrong = independent_response_score(a, a, t).mean()
    torch.testing.assert_close(wrong, score + sigma.square())


def test_secants_use_actual_row_shear_and_reverse_consistently():
    a = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.double)
    b = a + torch.tensor([[.1, -.2], [.4, .6]], dtype=torch.double)
    h = torch.tensor([.05, .1], dtype=torch.double)
    expected = torch.tensor([[2., -4.], [4., 6.]], dtype=torch.double)
    torch.testing.assert_close(feature_secant(a, b, h), expected)
    torch.testing.assert_close(feature_secant(b, a, h), -expected)


def test_endpoint_coupling_does_not_change_mean_feature_secant():
    phi = GaussianFourierFeatures(1)
    a = torch.tensor([[-1.], [1.]], dtype=torch.double)
    b = torch.tensor([[-.4], [1.8]], dtype=torch.double)
    first = feature_secant(phi(a), phi(b), .05).mean(0)
    other = feature_secant(phi(a), phi(b.flip(0)), .05).mean(0)
    torch.testing.assert_close(first, other)


@pytest.mark.parametrize('shear', [0., -1., float('nan'), float('inf'), [0.05, 0.]])
def test_invalid_shear_is_rejected(shear):
    a = torch.zeros(2, 3)
    with pytest.raises(ValueError, match='positive'):
        feature_secant(a, a, shear)


def test_invalid_shapes_and_kernel_scales_are_rejected():
    with pytest.raises(ValueError, match='bandwidths'):
        GaussianFourierFeatures(1, bandwidths=[0.])
    with pytest.raises(ValueError, match='aligned'):
        independent_response_score(torch.zeros(2, 1), torch.zeros(1, 1), torch.zeros(2, 1))
    with pytest.raises(ValueError, match='nonempty'):
        draw_feature_secant(GaussianFourierFeatures(1), torch.zeros(2, 0, 1), torch.zeros(2, 0, 1), .05)


class RadiusScaleFlow(torch.nn.Module):
    target_dim = 4
    context_dim = 2

    def __init__(self):
        super().__init__()
        self.motion = torch.nn.Parameter(torch.tensor(.4))

    def forward(self, z, context):
        shape = z[:, :3]
        radius = z[:, 3:4] * torch.exp(self.motion * context[:, :1])
        return torch.cat((shape, radius), dim=1), torch.zeros(len(z))

    def log_prob(self, values, context):
        return torch.zeros(len(values)) + self.motion * 0


def test_epoch_averages_distribution_score_by_pair_count_and_updates_flow():
    from sbsi.flow_training import epoch_response, paired_distribution_response_loss
    model = RadiusScaleFlow()
    phi = GaussianFourierFeatures(1, frequencies_per_band=4).float()
    zero = torch.zeros(3, 2)
    p1, p2 = zero.clone(), zero.clone()
    p1[:, 0] = .02
    p2[:, 1] = .02
    values = torch.tensor([[0., 0., 0., -.5], [0., 0., 0., .5], [0., 0., 0., 1.]])
    dest = zero.clone(); dest[:, 0] = .05
    # No measured distribution change: a useful target for the spurious scale motion.
    pairs = [(values[:2], zero[:2], values[:2], dest[:2]),
             (values[2:], zero[2:], values[2:], dest[2:])]
    def batch(s):
        return (values[s], zero[s], torch.ones(len(values[s])), zero[s], p1[s], p2[s],
                -p1[s], -p2[s], torch.zeros(len(values[s]), dtype=torch.long))
    batches = [batch(slice(0,2)), batch(slice(2,3))]
    args = (model, batches, torch.device('cpu'), [1.,1.], .02, torch.zeros(1), 0.)
    kw = dict(response_difference='central', response_estimator='complete_flow', nll_weight=0.,
              response_draws=8, response_sample_seed=23, paired_loader=pairs,
              distribution_weight=1., distribution_features=phi, distribution_draws=8)
    expected = (2*paired_distribution_response_loss(model,pairs[0],.05,phi,8,100023)
                + paired_distribution_response_loss(model,pairs[1],.05,phi,8,100025))/3
    result = epoch_response(*args, **kw)
    assert result[-1]['distribution_response'] == pytest.approx(float(expected.detach()))
    assert 'paired_transport' not in result[-1]
    initial = float(expected.detach())
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    epoch_response(*args, optimizer=optimizer, **kw)
    final = epoch_response(*args, **kw)[-1]['distribution_response']
    assert final < initial


def test_distribution_epoch_rejects_missing_paired_data():
    from sbsi.flow_training import epoch_response
    with pytest.raises(ValueError, match='paired data'):
        epoch_response(RadiusScaleFlow(), [], torch.device('cpu'), [1.,1.], .02,
                       torch.zeros(1), 0., distribution_weight=1., response_estimator='complete_flow')
