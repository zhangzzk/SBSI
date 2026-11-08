import pytest
import torch

from sbsi.flow_training import (
    complete_flow_context_means, complete_flow_response_dispersion, epoch_response,
)


@pytest.mark.parametrize("difference", ["forward", "central"])
def test_dispersion_penalizes_scale_motion_and_leaves_translation_free(difference):
    z = torch.tensor([-2., -1., 1., 2.], dtype=torch.double)[None, :, None].expand(2, 4, 4)
    shift = torch.tensor(.7, dtype=torch.double, requires_grad=True)
    scale = torch.tensor(.4, dtype=torch.double, requires_grad=True)
    delta = .02
    plus, minus = z+delta*(shift+scale*z), z-delta*(shift+scale*z)
    samples = [plus, z+delta*shift, minus, z-delta*shift] if difference == "central" else [z, plus, z+delta*shift]
    loss = complete_flow_response_dispersion(samples, delta, difference).mean()
    # Var(z)=2.5, scale changes only in one of two shear directions.
    torch.testing.assert_close(loss, scale.square()*2.5/2)
    loss.backward()
    assert abs(float(shift.grad)) < 1e-12
    assert float(scale.grad) == pytest.approx(1.0)
    translations = [z+delta*shift, z+2*delta*shift, z-delta*shift, z-2*delta*shift]
    if difference == "forward":
        translations = [z, *translations[:2]]
    assert float(complete_flow_response_dispersion(translations, delta, difference).max().detach()) < 1e-24


class ConditionalLocationScale(torch.nn.Module):
    context_dim = 2
    target_dim = 4

    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor(.7, dtype=torch.double))
        self.scale = torch.nn.Parameter(torch.tensor(.4, dtype=torch.double))

    def forward(self, z, context):
        c = context[:, :1]
        return z*(1+self.scale*c)+self.shift*c, torch.zeros(len(z), dtype=z.dtype)

    def log_prob(self, x, context):
        c = context[:, :1]
        scale = 1+self.scale*c
        return (-.5*((x-self.shift*c)/scale).square()-scale.log()).sum(dim=1)


def test_retaining_samples_preserves_means_and_common_draws():
    model = ConditionalLocationScale()
    contexts = [torch.zeros(3, 2, dtype=torch.double), torch.ones(3, 2, dtype=torch.double)*.02]
    plain = complete_flow_context_means(model, contexts, 8, seed=23)
    means, samples = complete_flow_context_means(model, contexts, 8, seed=23, return_samples=True)
    for actual, expected, drawn in zip(means, plain, samples):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, drawn.mean(dim=1), rtol=0, atol=0)
    torch.testing.assert_close(samples[0][:, :4], -samples[0][:, 4:])
    torch.testing.assert_close(samples[1], samples[0]*(1+.02*model.scale)+.02*model.shift)


def test_epoch_uses_weighted_dispersion_and_trains_scale_without_changing_shift():
    model = ConditionalLocationScale().float()
    c = torch.zeros(4, 2)
    p1, p2 = c.clone(), c.clone()
    p1[:, 0] = .02
    p2[:, 1] = .02
    weights = torch.tensor([1., 2., 3., 4.])
    batch = (torch.zeros(4, 4), c, weights, c, p1, p2, -p1, -p2, torch.zeros(4, dtype=torch.long))
    _, samples = complete_flow_context_means(model, [p1, p2, -p1, -p2], 8, seed=23, return_samples=True)
    expected = (weights*complete_flow_response_dispersion(samples, .02, "central")).sum()/weights.sum()
    args = (model, [batch], torch.device('cpu'), [1., 1.], .02, torch.zeros(1), 0.)
    kw = dict(response_difference='central', response_estimator='complete_flow',
              response_draws=8, response_sample_seed=23, nll_weight=0.)
    baseline = epoch_response(*args, **kw)
    assert 'response_dispersion' not in baseline[-1]
    active = epoch_response(*args, dispersion_weight=1., **kw)
    assert active[-1]['response_dispersion'] == pytest.approx(float(expected.detach()))
    shift = model.shift.detach().clone()
    scale = model.scale.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    epoch_response(*args, dispersion_weight=1., optimizer=optimizer, **kw)
    assert model.scale < scale
    torch.testing.assert_close(model.shift, shift, atol=1e-6, rtol=0)


def test_dispersion_rejects_incomplete_stencil():
    with pytest.raises(ValueError, match='aligned'):
        complete_flow_response_dispersion([torch.zeros(2, 4, 4)]*3, .02, 'central')
    with pytest.raises(ValueError, match='two draws'):
        complete_flow_response_dispersion([torch.zeros(2, 1, 4)]*4, .02, 'central')
