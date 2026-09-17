import numpy as np
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_grid_physical import (
    PhysicalGridPopulation,
    assign_grid_cells,
    baseline_frame_gamma,
    fit_physical_grid_response,
    quantile_grid_edges,
    sampled_grid_response_backward,
    summarize_grid_means,
)
from scripts.train_fixed_g0_grid_flows import grid_features


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
                context[:, 0] + context[:, 1],
                torch.zeros_like(context[:, 0]),
            ),
            dim=1,
        )
        value = z.double() + self.weight.double() * shift.double()
        logdet = torch.zeros(len(value), dtype=torch.float64, device=value.device)
        return value, logdet


def test_quantile_grid_assignment_uses_right_bins_and_retains_tails():
    coordinates = np.column_stack(
        (np.arange(12.0), np.arange(12.0) ** 2, np.arange(12.0) ** 3)
    )
    edges = quantile_grid_edges(coordinates, bins=3)
    probe = np.array(
        [
            [edges[0, 1], edges[1, 1], edges[2, 1]],
            [-100.0, -100.0, -100.0],
            [1.0e9, 1.0e9, 1.0e9],
        ]
    )
    np.testing.assert_array_equal(assign_grid_cells(probe, edges), [13, 0, 26])


def test_baseline_frame_gamma_has_parallel_and_cross_components():
    gamma = np.array([[0.03, 0.04], [0.03, 0.04], [0.03, 0.04]])
    ellipticity = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(
        baseline_frame_gamma(gamma, ellipticity),
        [[0.03, 0.04], [0.04, -0.03], [0.03, 0.04]],
    )


def test_joint_grid_fit_recovers_shape_and_radius_responses():
    rng = np.random.default_rng(71)
    n_cases = 14
    n_cells = 2
    directions = 0.05 * np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
        dtype=float,
    )
    base = np.array(
        [
            [1.1, 0.2, -0.1, 0.9, 0.7, -0.2],
            [0.8, -0.3, 0.4, 1.2, -0.5, 0.6],
        ]
    )
    offsets = rng.normal(scale=0.04, size=(n_cases, n_cells, 6))
    delta = []
    gamma = []
    cells = []
    cases = []
    for case in range(n_cases):
        for cell in range(n_cells):
            response = base[cell] + offsets[case, cell]
            shape_matrix = response[:4].reshape(2, 2)
            for shear in directions:
                delta.append(
                    [
                        *(shape_matrix @ shear),
                        response[4:] @ shear,
                    ]
                )
                gamma.append(shear)
                cells.append(cell)
                cases.append(case)
    fitted = fit_physical_grid_response(
        np.asarray(delta),
        np.asarray(gamma),
        np.asarray(gamma),
        np.asarray(cells),
        np.asarray(cases),
        n_cells,
    )
    np.testing.assert_allclose(fitted["target"], base + offsets.mean(axis=0))
    assert fitted["target"].shape == (2, 6)
    assert fitted["precision"].shape == (2, 6, 6)
    assert np.isfinite(fitted["precision"]).all()
    np.testing.assert_array_equal(fitted["counts"], [n_cases * 8] * n_cells)


def test_sampled_grid_backward_preserves_existing_gradient():
    model = LinearPhysicalFlow()
    gamma = np.tile(
        0.05 * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float),
        (2, 1),
    )
    cells = np.repeat(np.arange(2), 4)
    pairs = np.column_stack((2 * np.arange(8), 2 * np.arange(8) + 1))
    context = torch.zeros((16, 2), dtype=torch.float32)
    context[pairs[:, 1]] = torch.as_tensor(gamma, dtype=torch.float32)
    normal = np.stack(
        [gamma[cells == cell].T @ gamma[cells == cell] for cell in range(2)]
    )
    statistics = {
        "target": np.zeros((2, 6)),
        "precision": np.broadcast_to(np.eye(6), (2, 6, 6)).copy(),
        "counts": np.full(2, 4),
        "shape_normal": normal,
        "scalar_normal": normal,
    }
    population = PhysicalGridPopulation(
        context, pairs, cells, gamma, gamma, statistics
    )
    model.weight.square().backward()
    before = model.weight.grad.detach().clone()
    result = sampled_grid_response_backward(
        model,
        population,
        cells_per_step=2,
        pairs_per_cell=6,
        draws=4,
        chunk_size=3,
        seed=17,
        weight=1.0,
    )
    assert np.isfinite(result["loss"])
    assert torch.isfinite(model.weight.grad)
    assert not torch.equal(model.weight.grad, before)


def test_grid_summary_removes_finite_integration_variance():
    rng = np.random.default_rng(91)
    means = rng.normal(size=(5, 3, 6))
    target = means.mean(axis=0)
    precision = np.broadcast_to(np.eye(6), (3, 6, 6)).copy()
    summary = summarize_grid_means(means, target, precision)
    assert summary["mean_quadratic"] < 1.0e-28
    assert summary["loss"] < 0.0
    assert np.isfinite(summary["mc_jackknife_se"])


def test_grid_axes_use_only_aligned_baseline_values():
    n = 2
    context = np.zeros((2 * n, len(FLOW_FEATURES)), dtype=np.float32)
    pairs = np.array([[0, 2], [1, 3]])
    truth_mag = np.array([20.0, 24.0])
    truth_radius = np.array([0.2, 0.8])
    measured_mag = np.array([21.0, 23.0])
    measured_radius = np.array([3.2, 5.1])
    r_blend = np.array([0.1, 0.9])
    context[:n, FLOW_FEATURES.index("r_input_p")] = truth_mag
    context[:n, FLOW_FEATURES.index("circularized_Re_input_p")] = truth_radius
    target = np.zeros((2 * n, 4), dtype=np.float64)
    target[:n, 2] = measured_radius
    target[:n, 3] = 10.0 ** (-0.4 * (measured_mag - 30.0))
    data = {"context_raw": context, "target_raw": target, "pairs": pairs}
    baseline = np.column_stack(
        (truth_mag, measured_mag, measured_radius, r_blend)
    )
    np.testing.assert_allclose(
        grid_features(data, baseline, "truth"),
        np.column_stack((truth_mag, truth_radius, r_blend)),
    )
    np.testing.assert_allclose(
        grid_features(data, baseline, "measured"),
        np.column_stack((measured_mag, measured_radius, r_blend)),
    )
