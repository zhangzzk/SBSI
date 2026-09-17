import numpy as np
import torch

from sbsi.flow_paired_shape import (
    PairedResponsePopulation,
    evaluate_paired_response,
    paired_cross_score,
    paired_response_backward,
    physical_context_means,
    positive_shear_pair_mask,
)


class LinearPhysicalFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target_dim = 4
        self.context_dim = 2
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward_physical(self, z, context):
        shift = torch.stack(
            (
                context[:, 0],
                context[:, 1],
                context[:, 0] + context[:, 1] + 4.0,
                torch.zeros_like(context[:, 0]) + 10.0,
            ),
            dim=1,
        )
        value = z.double() + self.weight.double() * shift.double()
        return value, torch.zeros(len(value), dtype=torch.float64, device=value.device)


def test_positive_shear_pair_mask_drops_zero_without_flooring():
    gamma = np.array([[0.05, 0.0], [0.0, 0.0], [-0.03, 0.04]])
    np.testing.assert_array_equal(
        positive_shear_pair_mask(gamma), np.array([True, False, True])
    )


def test_physical_means_use_antithetic_crn():
    model = LinearPhysicalFlow()
    contexts = [
        torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        torch.tensor([[1.0, 1.0], [3.0, 5.0]]),
    ]
    means = physical_context_means(model, contexts, 8, 12)
    expected = model.weight.double() * torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [1.0, 2.0, 3.0, 0.0]], dtype=torch.float64
    )
    torch.testing.assert_close(means[1] - means[0], expected)


def test_paired_response_backward_preserves_nll_gradient_and_is_finite():
    model = LinearPhysicalFlow()
    context = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
    gamma = np.full((2, 2), [0.05, 0.0])
    delta = np.zeros((2, 3))
    population = PairedResponsePopulation(
        context, pairs, delta, gamma, radius_scale=1.0
    )
    (model.weight.square()).backward()
    before = model.weight.grad.detach().clone()
    result = paired_response_backward(
        model,
        population,
        pairs_per_step=8,
        draws=4,
        chunk_size=3,
        seed=17,
        weight=10.0,
    )
    assert np.isfinite(result["loss"])
    assert np.isfinite(result["component_losses"]).all()
    assert torch.isfinite(model.weight.grad)
    assert not torch.equal(model.weight.grad, before)


def test_cross_score_keeps_fixed_divisor_when_radius_is_added():
    a = torch.tensor([[2.0, 4.0, 6.0]])
    c = torch.tensor([[1.0, 3.0, 5.0]])
    torch.testing.assert_close(paired_cross_score(a, c), torch.tensor([11.0]))


def test_fixed_subset_evaluation_reports_three_components():
    model = LinearPhysicalFlow()
    context = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
    population = PairedResponsePopulation(
        context,
        np.array([[0, 1], [2, 3]], dtype=np.int64),
        np.zeros((2, 3)),
        np.full((2, 2), [0.05, 0.0]),
        radius_scale=1.0,
    )
    result = evaluate_paired_response(
        model, population, np.arange(2), draws=4, groups=4, batch_size=1, seed=20
    )
    assert result["component_losses"].shape == (3,)
    assert result["group_residuals"].shape == (4, 2, 3)
    assert np.isfinite(result["response_loss"])
