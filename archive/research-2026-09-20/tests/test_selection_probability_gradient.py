from types import SimpleNamespace

import numpy as np
import torch

from scripts import selection_probability_gradient as helper


def test_mass_gradient_uses_all_parent_denominator_and_own_usability(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(.01)
    context = torch.arange(1., 6., dtype=torch.float64)[:, None]
    usable = torch.tensor([[True], [False], [True], [True], [False]])
    population = SimpleNamespace(context=context, pairs=torch.column_stack((torch.arange(5), torch.arange(5))),
        usable=usable, cuts=[{'name': 'primary', 'radius_min': 3., 'magnitude_max': 25.8}],
        radius_softness=.05, magnitude_softness=.05, zero_point=30.)

    def draws(model, contexts, count, seed):
        radius = 3.+model(contexts[0])[:, 0]
        zero = radius*0
        flux = zero+10**((30.-25.8)/2.5)
        return [torch.stack((zero, zero, radius, flux), -1)[:, None, :].expand(-1, count, -1)]

    monkeypatch.setattr(helper, 'physical_context_draws', draws)
    value = helper.sampled_soft_mass_backward(model, population, np.arange(5), draws=4, batch_size=2, seed=1)
    probability = torch.sigmoid(.01*context[:, 0]/.05)
    expected_value = (probability*.5*usable[:, 0]).sum()/5
    expected_gradient = (probability*(1-probability)*context[:, 0]/.05*.5*usable[:, 0]).sum()/5
    np.testing.assert_allclose(value, expected_value, rtol=1e-12)
    np.testing.assert_allclose(model.weight.grad.item(), expected_gradient, rtol=1e-12)


def test_common_descent_reduces_conflicting_objectives_without_arbitrary_loss_weight():
    response, mass = np.array([2., 0]), np.array([-1., 2.])
    direction, report = helper.common_descent(response, mass)
    assert report['response_only_probability_slope'] > 0
    assert report['strict_common_descent']
    assert response@direction < 0 and mass@direction < 0
    np.testing.assert_allclose(np.linalg.norm(direction), np.linalg.norm(response))
    scaled, _ = helper.common_descent(response, 100*mass)
    np.testing.assert_allclose(scaled, direction)


def test_opposed_objectives_do_not_claim_strict_common_descent():
    direction, report = helper.common_descent([1., 0], [-2., 0])
    assert direction is None and not report['strict_common_descent']
