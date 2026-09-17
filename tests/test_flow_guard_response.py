import numpy as np
import torch

import pytest

from sbsi.flow_guard_response import (
    GuardResponsePopulation,
    fit_guard_response,
    make_guard_band_cuts,
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


def test_hard_guard_weights_are_exact_catalogue_indicators():
    cuts = make_guard_cuts([3.0], [25.8])
    # magnitude = 30 - 2.5*log10(flux); flux 40 -> 25.99 (fails), 80 -> 25.24 (passes)
    values = np.array(
        [[0.0, 0.0, 2.9, 40.0], [0.0, 0.0, 3.1, 80.0]], dtype=np.float64
    )
    weights = numpy_guard_weights(
        values, cuts, radius_softness=0.1, magnitude_softness=0.1, hard=True
    )
    assert set(np.unique(weights)) <= {0.0, 1.0}
    # columns are global, radius_gt_3, magnitude_lt_25.8, joint
    np.testing.assert_array_equal(weights[0], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(weights[1], [1.0, 1.0, 1.0, 1.0])
    torch_weights = torch_guard_weights(
        torch.as_tensor(values),
        cuts,
        radius_softness=0.1,
        magnitude_softness=0.1,
        hard=True,
    )
    np.testing.assert_allclose(torch_weights.numpy(), weights)


def test_sharpening_the_soft_guard_converges_to_the_hard_guard():
    cuts = make_guard_cuts([3.0], [25.8])
    rng = np.random.default_rng(20260917)
    values = np.column_stack(
        (
            rng.normal(size=256),
            rng.normal(size=256),
            rng.uniform(2.0, 4.0, size=256),
            rng.uniform(30.0, 200.0, size=256),
        )
    )
    hard = numpy_guard_weights(
        values, cuts, radius_softness=1.0e-6, magnitude_softness=1.0e-6, hard=True
    )
    sharp = numpy_guard_weights(
        values, cuts, radius_softness=1.0e-6, magnitude_softness=1.0e-6
    )
    np.testing.assert_allclose(sharp, hard, atol=1.0e-9)


def test_hard_fit_weights_each_leg_by_its_own_measurement():
    """The per-leg convention must not apply the g=0 selection to the sheared leg."""

    cuts = make_guard_cuts([3.0], [25.8])
    rows = 512
    rng = np.random.default_rng(4242)
    # Every zero-leg row passes the radius cut; every sheared-leg row fails it.
    zero = np.column_stack(
        (
            rng.normal(scale=0.2, size=rows),
            rng.normal(scale=0.2, size=rows),
            np.full(rows, 3.5),
            np.full(rows, 200.0),
        )
    )
    shear = np.column_stack(
        (
            rng.normal(scale=0.2, size=rows),
            rng.normal(scale=0.2, size=rows),
            np.full(rows, 2.5),
            np.full(rows, 200.0),
        )
    )
    gamma = rng.normal(scale=0.02, size=(rows, 2))
    fitted = fit_guard_response(
        zero, shear, gamma, cuts, radius_softness=0.05, magnitude_softness=0.05, hard=True
    )
    assert fitted["hard"] is True
    radius_index = [cut["name"] for cut in cuts].index("radius_gt_3")
    # The sheared leg contributes zero guarded influence, so the whole radius
    # response comes from minus the zero-leg term.  If the zero-leg weight were
    # reused for the sheared leg the two terms would instead nearly cancel.
    baseline_mass = fitted["baseline_mass"][radius_index]
    assert baseline_mass == 1.0
    response = np.asarray(fitted["target"])[radius_index].reshape(2, 2)
    assert np.isfinite(response).all()

    reference = fit_guard_response(
        zero, zero, gamma, cuts, radius_softness=0.05, magnitude_softness=0.05, hard=True
    )
    identical = np.asarray(reference["target"])[radius_index].reshape(2, 2)
    # Replacing the sheared leg by a copy of the zero leg must change the answer;
    # a leg-invariant (zero-anchored) implementation would be insensitive here.
    assert not np.allclose(response, identical, atol=1.0e-6)


def test_population_records_and_uses_the_hard_convention():
    cuts = make_guard_cuts([3.0], [25.8])
    rows = 64
    rng = np.random.default_rng(99)
    context = torch.as_tensor(
        np.column_stack(
            (
                rng.normal(scale=0.1, size=rows),
                rng.normal(scale=0.1, size=rows),
                rng.uniform(3.2, 4.0, size=rows),
                rng.uniform(80.0, 200.0, size=rows),
            )
        ),
        dtype=torch.float64,
    )
    pairs = np.column_stack(
        (np.arange(0, rows, 2), np.arange(1, rows, 2))
    ).astype(np.int64)
    gamma = rng.normal(scale=0.02, size=(len(pairs), 2))
    measured_zero = context[pairs[:, 0]].numpy()
    measured_shear = context[pairs[:, 1]].numpy()
    population = GuardResponsePopulation(
        context,
        pairs,
        measured_zero,
        measured_shear,
        gamma,
        cuts,
        radius_softness=0.05,
        magnitude_softness=0.05,
        response_scale=np.ones(len(cuts)),
        hard=True,
    )
    assert population.hard is True
    assert population.statistics["hard"] is True
    model = DeterministicPhysicalFlow()
    contribution = population.contributions(
        model, torch.arange(len(pairs)), draws=2, seed=7
    )
    assert torch.isfinite(contribution).all()


def _physical(radius, magnitude, zero_point=30.0):
    radius = np.asarray(radius, dtype=np.float64)
    magnitude = np.broadcast_to(
        np.asarray(magnitude, dtype=np.float64), radius.shape
    )
    flux = 10.0 ** ((zero_point - magnitude) / 2.5)
    return np.column_stack(
        (np.zeros_like(radius), np.zeros_like(radius), radius, flux)
    )


def test_band_guards_partition_the_radius_axis_exactly():
    edges = (2.6, 2.8, 3.0, 3.2)
    cuts = make_guard_band_cuts(edges)
    assert [cut["name"] for cut in cuts] == [
        "radius_band_2.6_2.8",
        "radius_band_2.8_3",
        "radius_band_3_3.2",
        "radius_gt_3.2",
    ]
    radius = np.array([2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 5.0])
    weights = numpy_guard_weights(
        _physical(radius, 24.0),
        cuts,
        radius_softness=0.01,
        magnitude_softness=0.01,
        hard=True,
    )
    # Each object above the first edge lands in exactly one band, and nothing
    # at or below the first edge is claimed by any of them.
    total = weights.sum(axis=1)
    np.testing.assert_allclose(total, (radius > edges[0]).astype(np.float64))
    assert set(np.unique(weights)) <= {0.0, 1.0}


def test_band_guard_soft_weights_converge_to_the_band_indicator():
    cuts = make_guard_band_cuts((2.8, 3.0))
    radius = np.array([2.7, 2.9, 3.1])
    values = _physical(radius, 24.0)
    hard = numpy_guard_weights(
        values, cuts, radius_softness=1.0e-4, magnitude_softness=1.0e-4, hard=True
    )
    previous = None
    for softness in (0.1, 0.01, 0.001):
        soft = numpy_guard_weights(
            values, cuts, radius_softness=softness, magnitude_softness=0.01
        )
        gap = float(np.abs(soft - hard).max())
        if previous is not None:
            assert gap < previous
        previous = gap
    assert previous < 1.0e-3


def test_band_guard_agrees_between_numpy_and_torch():
    cuts = make_guard_band_cuts((2.8, 3.0, 3.4), magnitude_max=25.8)
    rng = np.random.default_rng(5)
    values = _physical(rng.uniform(2.0, 6.0, size=128), rng.uniform(22.0, 27.0, size=128))
    for hard in (False, True):
        expected = numpy_guard_weights(
            values, cuts, radius_softness=0.02, magnitude_softness=0.03, hard=hard
        )
        actual = torch_guard_weights(
            torch.as_tensor(values, dtype=torch.float64),
            cuts,
            radius_softness=0.02,
            magnitude_softness=0.03,
            hard=hard,
        )
        np.testing.assert_allclose(actual.numpy(), expected, atol=1.0e-12)


def test_legacy_three_key_cuts_keep_their_meaning():
    legacy = ({"name": "radius_gt_3", "radius_min": 3.0, "magnitude_max": None},)
    radius = np.array([2.9, 3.1])
    values = _physical(radius, 24.0)
    weights = numpy_guard_weights(
        values, legacy, radius_softness=0.01, magnitude_softness=0.01, hard=True
    )
    np.testing.assert_allclose(weights[:, 0], [0.0, 1.0])
    # The normalised cut carries the new edges as explicit "no limit" entries.
    assert make_guard_cuts([3.0], [25.8])[0] == {
        "name": "global",
        "radius_min": None,
        "radius_max": None,
        "magnitude_min": None,
        "magnitude_max": None,
    }


def test_inverted_bands_and_unknown_keys_are_rejected():
    with pytest.raises(ValueError):
        numpy_guard_weights(
            _physical([3.0], 24.0),
            ({"name": "bad", "radius_min": 3.0, "radius_max": 2.5},),
            radius_softness=0.01,
            magnitude_softness=0.01,
        )
    with pytest.raises(ValueError):
        numpy_guard_weights(
            _physical([3.0], 24.0),
            ({"name": "bad", "radius_floor": 3.0},),
            radius_softness=0.01,
            magnitude_softness=0.01,
        )
    with pytest.raises(ValueError):
        make_guard_band_cuts((3.0, 2.8))


def test_per_component_response_scale_reweights_the_guard_loss():
    cuts = make_guard_cuts([3.0], [25.8])
    rows = 64
    rng = np.random.default_rng(3)
    context = torch.as_tensor(
        np.column_stack(
            (
                rng.normal(scale=0.1, size=rows),
                rng.normal(scale=0.1, size=rows),
                rng.uniform(3.2, 4.0, size=rows),
                rng.uniform(80.0, 200.0, size=rows),
            )
        ),
        dtype=torch.float64,
    )
    pairs = np.column_stack((np.arange(0, rows, 2), np.arange(1, rows, 2))).astype(
        np.int64
    )
    gamma = rng.normal(scale=0.02, size=(len(pairs), 2))

    def build(scale):
        return GuardResponsePopulation(
            context,
            pairs,
            context[pairs[:, 0]].numpy(),
            context[pairs[:, 1]].numpy(),
            gamma,
            cuts,
            radius_softness=0.05,
            magnitude_softness=0.05,
            response_scale=scale,
        )

    flat = build(np.ones(len(cuts)))
    assert flat.response_scale.shape == (len(cuts), 1)
    # Sending the off-diagonal scales to a large value drives their share of
    # the averaged quadratic to zero without touching the diagonal terms.
    per_component = np.ones((len(cuts), 4))
    per_component[:, 1:3] = 1.0e6
    weighted = build(per_component)
    assert weighted.response_scale.shape == (len(cuts), 4)
    model = DeterministicPhysicalFlow()
    flat_loss = sampled_guard_response_backward(
        model, flat, 16, 2, chunk_size=8, seed=11
    )["loss"]
    weighted_loss = sampled_guard_response_backward(
        model, weighted, 16, 2, chunk_size=8, seed=11
    )["loss"]
    assert abs(weighted_loss) < abs(flat_loss)

    with pytest.raises(ValueError):
        build(np.ones((len(cuts), 3)))
