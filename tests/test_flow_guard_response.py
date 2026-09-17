import numpy as np
import torch

from sbsi.flow_guard_response import (
    GuardResponsePopulation,
    fit_guard_response,
    make_guard_cuts,
    numpy_guard_weights,
    sampled_guard_response_backward,
    torch_guard_weights,
)


class DeterministicPhysicalFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target_dim = 4
        self.context_dim = 4
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward_physical(self, z, context):
        value = self.scale.double() * context.double()
        return value, torch.zeros(len(value), dtype=torch.float64, device=value.device)


def test_guard_bank_and_numpy_torch_weights_agree():
    cuts = make_guard_cuts([3.0], [25.8])
    assert [cut["name"] for cut in cuts] == [
        "global",
        "radius_gt_3",
        "magnitude_lt_25.8",
        "radius_gt_3_magnitude_lt_25.8",
    ]
    values = np.array(
        [[0.0, 0.0, 2.9, 40.0], [0.0, 0.0, 3.1, 80.0]], dtype=np.float64
    )
    expected = numpy_guard_weights(
        values, cuts, radius_softness=0.1, magnitude_softness=0.1
    )
    actual = torch_guard_weights(
        torch.as_tensor(values),
        cuts,
        radius_softness=0.1,
        magnitude_softness=0.1,
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(expected[:, 0], 1.0)


def test_magnitude_guard_has_finite_gradient_at_softplus_underflow():
    cuts = make_guard_cuts([3.0], [25.8])
    values = torch.tensor(
        [[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 3.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = torch_guard_weights(
        values,
        cuts,
        radius_softness=0.05,
        magnitude_softness=0.05,
    )
    weights.sum().backward()
    assert torch.isfinite(weights).all()
    assert torch.isfinite(values.grad).all()
    assert values.grad[0, 3] == 0.0


def test_fit_guard_response_recovers_known_global_matrix():
    rng = np.random.default_rng(14)
    n = 20_000
    gamma = rng.normal(size=(n, 2))
    gamma *= 0.05 / np.linalg.norm(gamma, axis=1)[:, None]
    matrix = np.array([[0.8, -0.2], [0.1, 1.1]])
    zero = np.column_stack(
        (np.zeros((n, 2)), np.full(n, 3.2), np.full(n, 60.0))
    )
    shear = zero.copy()
    shear[:, :2] = gamma @ matrix.T
    result = fit_guard_response(
        zero,
        shear,
        gamma,
        [{"name": "global", "radius_min": None, "magnitude_max": None}],
        radius_softness=0.1,
        magnitude_softness=0.1,
    )
    np.testing.assert_allclose(result["target"][0].reshape(2, 2), matrix, atol=1.0e-12)


def test_sampled_guard_response_backward_is_finite_and_differentiable():
    rng = np.random.default_rng(9)
    n = 200
    gamma = rng.normal(size=(n, 2))
    gamma *= 0.05 / np.linalg.norm(gamma, axis=1)[:, None]
    zero = np.column_stack(
        (np.zeros((n, 2)), np.full(n, 3.0), np.full(n, 60.0))
    )
    shear = zero.copy()
    shear[:, :2] = gamma
    context = torch.as_tensor(np.concatenate((zero, shear)), dtype=torch.float32)
    pairs = np.column_stack((np.arange(n), n + np.arange(n)))
    model = DeterministicPhysicalFlow()
    population = GuardResponsePopulation(
        context,
        pairs,
        zero,
        shear,
        gamma,
        make_guard_cuts([3.0], [25.8]),
        radius_softness=0.1,
        magnitude_softness=0.1,
        response_scale=0.02,
    )
    model.scale.square().backward()
    before = model.scale.grad.detach().clone()
    result = sampled_guard_response_backward(
        model,
        population,
        pairs_per_step=128,
        draws=4,
        chunk_size=31,
        seed=27,
        weight=1.0,
    )
    assert np.isfinite(result["loss"])
    assert result["group_means"].shape == (2, 4, 4)
    assert torch.isfinite(model.scale.grad)
    assert not torch.equal(model.scale.grad, before)
