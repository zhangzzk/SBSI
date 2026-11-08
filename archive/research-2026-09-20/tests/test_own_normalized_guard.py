from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import own_normalized_guard as helper


def test_response_cancels_common_probability_scaling_in_value_and_gradient():
    raw = torch.tensor([[.7, .07, -.14, .1, -.2, .31, .02, -.04, .38]], dtype=torch.float64)
    factor = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    response = helper.response_from_moments(raw*factor)
    target = torch.zeros_like(response)
    loss = (response-target).square().mean()
    derivative, = torch.autograd.grad(loss, factor)
    torch.testing.assert_close(response, helper.response_from_moments(raw))
    assert abs(derivative.item()) < 1e-14
    # The observed-normalization proxy does not possess this invariance.
    observed_mean, observed_mass = torch.tensor([.1, -.2]), .7
    scaled = raw*factor
    proxy = (scaled[:, 5:].reshape(-1, 2, 2)-observed_mean[None, :, None]*scaled[:, None, 3:5])/observed_mass
    wrong_derivative, = torch.autograd.grad(proxy.square().mean(), factor)
    assert abs(wrong_derivative.item()) > .1


def test_moment_replay_matches_full_autograd_with_selection_and_unmatched_usability(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(.015)
    context = torch.tensor([[1.], [2.], [3.], [4.], [1.1], [1.8], [3.2], [3.8]], dtype=torch.float64)
    population = SimpleNamespace(context=context, pairs=torch.column_stack((torch.arange(4), torch.arange(4)+4)),
        usable=torch.tensor([[True, True], [False, True], [True, False], [False, False]]),
        cuts=[{'name': 'primary', 'radius_min': 3., 'magnitude_max': 25.8}],
        radius_softness=.05, magnitude_softness=.05, zero_point=30.,
        design=torch.tensor([[3., 1.], [-1., 2.], [2., -3.], [-2., -1.]], dtype=torch.float64),
        target=torch.tensor([[.4, 0., 0., .5]], dtype=torch.float64), response_scale=torch.tensor([[.1]], dtype=torch.float64))

    def generate(model, contexts, draws, seed):
        outputs = []
        for frame in contexts:
            value = model(frame)[:, 0]
            result = torch.stack((.2+2*value, -.1+value, 3.+value,
                                  torch.exp(value)*10**((30.-25.8)/2.5)), dim=-1)
            outputs.append(result[:, None, :].expand(-1, draws, -1))
        return outputs

    monkeypatch.setattr(helper, 'physical_context_draws', generate)
    raw = helper.moment_contributions(model, population, torch.arange(4), draws=4, seed=1)
    torch.testing.assert_close(raw[3], torch.zeros_like(raw[3]))
    assert raw[1, 0, 0] == 0 and raw[1, 0, 5:].abs().sum() > 0  # entering U contributes
    integrated = raw.mean(0)
    loss = ((helper.response_from_moments(integrated)-population.target)/population.response_scale).square().mean()
    direct, = torch.autograd.grad(loss, model.weight)
    for batch in (1, 3, 4):
        model.zero_grad(set_to_none=True)
        result = helper.sampled_moment_backward(model, population, np.arange(4), integrated.detach(),
                                                draws=4, batch_size=batch, seed=1)
        torch.testing.assert_close(model.weight.grad, direct, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result['sampled_moments'], integrated.detach(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result['loss_at_integrated_moments'], loss.detach(), rtol=1e-12)


def test_centred_ratio_uses_model_mean_and_rejects_zero_mass():
    raw = torch.tensor([[.5, .1, -.05, .2, -.3, .2, .1, -.1, .25]], dtype=torch.float64)
    expected = (np.array([[.2, .1], [-.1, .25]])-np.outer([.2, -.1], [.2, -.3]))/.5
    np.testing.assert_allclose(helper.response_from_moments(raw).numpy().reshape(2, 2), expected)
    raw[0, 0] = 0
    with pytest.raises(ValueError, match='positive baseline'):
        helper.response_from_moments(raw)
